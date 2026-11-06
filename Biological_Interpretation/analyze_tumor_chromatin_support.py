#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from biological_common import (
    atomic_write_tsv,
    now,
    write_json,
    write_tsv,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import numpy as np
import pandas as pd
import json
import math
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import (
    norm,
    chi2,
)
from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

# --------------------------------------------------------------------------
# step 04: Analyze context specificity
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = (PROJECT_ROOT / "Biological_Interpretation").resolve()
Modeling_OUTPUT = (PROJECT_ROOT / "Modeling/output").resolve()

M26_DIR = BIOLOGY_ROOT / "26_host_regulatory_mechanism_interpretation"
M27_DIR = BIOLOGY_ROOT / "27_reference_to_tumor_context_interpretation"

M26_SOURCE = M26_DIR / "26_regulatory_source_matched_comparison.tsv"
M26_FEATURE = M26_DIR / "26_regulatory_feature_matched_comparison.tsv"
M27_CELL_TEST = M27_DIR / "27_cell_line_matched_regulatory_comparison.tsv"
M27_MATCHING = M27_DIR / "27_cell_line_matching_balance.tsv"
M27_SIMILARITY = M27_DIR / "27_cell_line_pairwise_similarity.tsv"
M27_BREADTH = M27_DIR / "27_cell_line_support_breadth.tsv"
M27_PATTERNS = M27_DIR / "27_cell_line_support_patterns.tsv"

OUT_DIR = BIOLOGY_ROOT / "output/04_analyze_context_specificity"

OUT_LEVEL_SUMMARY = OUT_DIR / "04_analysis_level_summary.tsv"


OUT_BREADTH = OUT_DIR / "04_support_breadth_summary.tsv"
OUT_PATTERNS = OUT_DIR / "04_support_pattern_diversity.tsv"
OUT_KEY_RESULTS = OUT_DIR / "04_key_results.tsv"


FDR_THRESHOLD = 0.05
CELL_LINES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

PAIRWISE_N = len(CELL_LINES) * (len(CELL_LINES) - 1) // 2
PATTERN_UNIVERSE_N = 2 ** len(CELL_LINES)

MATCHING_COLUMNS = [
    "eid",
    "cell_line",
    "supported_top1_n",
    "unsupported_top1_n",
    "matched_pairs_n",
    "exact_family_match_rate",
    "global_fallback_n",
]








def summarize_dimensions(
    source: pd.DataFrame,
    feature: pd.DataFrame,
    tests: pd.DataFrame,
    matching: pd.DataFrame,
    similarity: pd.DataFrame,
    breadth: pd.DataFrame,
    patterns: pd.DataFrame,
) -> dict[str, int]:
    per_cell = tests.groupby("eid").size()
    return {
        "m26_source_test_n": len(source),
        "m26_source_feature_n": int(source["feature"].nunique()),
        "m26_matched_pairs_n": int(pd.to_numeric(source["pairs"]).max()),
        "m26_feature_test_n": len(feature),
        "m26_scope_test_n": {k: int(v) for k, v in feature.groupby("scope").size().items()},
        "cell_line_n": int(tests["eid"].nunique()),
        "feature_n": int(tests["feature"].nunique()),
        "cell_line_test_n": len(tests),
        "tests_per_cell_line_min": int(per_cell.min()),
        "tests_per_cell_line_max": int(per_cell.max()),
        "matched_pairs_n_min": int(matching["matched_pairs_n"].min()),
        "matched_pairs_n_max": int(matching["matched_pairs_n"].max()),
        "pairwise_n": len(similarity),
        "pairwise_n_expected": PAIRWISE_N,
        "top1_n": int(breadth["top1_n"].sum()),
        "breadth_level_n": int(breadth["tumor_active_support_breadth"].nunique()),
        "pattern_n": len(patterns),
        "pattern_universe_n": PATTERN_UNIVERSE_N,
    }


def add_cell_test_annotations(tests: pd.DataFrame) -> pd.DataFrame:
    annotated = tests.copy()
    numeric = annotated["feature_type"].eq("numeric")
    numeric_effect = pd.to_numeric(annotated["effect_rank_biserial"], errors="coerce")
    binary_left = pd.to_numeric(
        annotated["discordant_supported1_unsupported0"], errors="coerce"
    )
    binary_right = pd.to_numeric(
        annotated["discordant_supported0_unsupported1"], errors="coerce"
    )
    annotated["signed_effect"] = np.where(
        numeric,
        numeric_effect,
        (binary_left - binary_right) / annotated["matched_pairs_n"],
    )
    annotated["direction"] = np.select(
        [annotated["signed_effect"] > 0, annotated["signed_effect"] < 0],
        ["SUPPORTED_HIGHER", "SUPPORTED_LOWER"],
        default="EQUAL",
    )
    annotated["is_significant_global_fdr0_05"] = (
        annotated["fdr_global"] < FDR_THRESHOLD
    )
    return annotated


def build_level_summary(
    source: pd.DataFrame,
    feature: pd.DataFrame,
    cell_tests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    panels = [
        (
            "POOLED_PREDEFINED_SOURCE",
            source,
            "fdr_bh",
            "regulatory source",
        ),
        (
            "POOLED_COMPACT24",
            feature[feature["scope"] == "compact24"],
            "fdr_bh_within_scope",
            "quantitative regulatory feature",
        ),
        (
            "POOLED_RAW38",
            feature[feature["scope"] == "raw38"],
            "fdr_bh_within_scope",
            "raw regulatory feature",
        ),
        (
            "CELL_LINE_RESOLVED",
            cell_tests,
            "fdr_global",
            "cell-line-feature test",
        ),
    ]
    for level, frame, fdr_column, unit in panels:
        significant_n = int((frame[fdr_column] < FDR_THRESHOLD).sum())
        rows.append(
            {
                "analysis_level": level,
                "test_unit": unit,
                "tests_n": len(frame),
                "significant_fdr0_05_n": significant_n,
                "significant_fraction": significant_n / len(frame),
                "minimum_fdr": float(frame[fdr_column].min()),
            }
        )
    return pd.DataFrame(rows)


def build_cell_summary(
    tests: pd.DataFrame, matching: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for eid, group in tests.groupby("eid", sort=False):
        significant = group[group["is_significant_global_fdr0_05"]]
        strongest = group.sort_values("fdr_global").iloc[0]
        rows.append(
            {
                "eid": eid,
                "cell_line": CELL_LINES[eid],
                "tests_n": len(group),
                "significant_global_fdr0_05_n": len(significant),
                "supported_higher_significant_n": int(
                    (significant["direction"] == "SUPPORTED_HIGHER").sum()
                ),
                "supported_lower_significant_n": int(
                    (significant["direction"] == "SUPPORTED_LOWER").sum()
                ),
                "strongest_feature": strongest["feature"],
                "minimum_global_fdr": float(strongest["fdr_global"]),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.merge(
        matching[MATCHING_COLUMNS],
        on=["eid", "cell_line"],
        how="left",
        validate="one_to_one",
    )


def classify_consistency(group: pd.DataFrame) -> str:
    significant = group[group["is_significant_global_fdr0_05"]]
    if significant.empty:
        return "NO_GLOBAL_SIGNIFICANCE"
    directions = set(significant["direction"]) - {"EQUAL"}
    if len(directions) > 1:
        return "DIRECTION_HETEROGENEOUS"
    if len(significant) == len(CELL_LINES):
        return "SHARED_ALL_FIVE_CONSISTENT"
    return "PARTIAL_CELL_LINE_CONSISTENT"


def build_feature_consistency(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, group in tests.groupby("feature", sort=False):
        significant = group[group["is_significant_global_fdr0_05"]]
        rows.append(
            {
                "feature": feature_name,
                "feature_type": group["feature_type"].iloc[0],
                "cell_lines_tested_n": len(group),
                "significant_cell_lines_n": len(significant),
                "supported_higher_significant_n": int(
                    (significant["direction"] == "SUPPORTED_HIGHER").sum()
                ),
                "supported_lower_significant_n": int(
                    (significant["direction"] == "SUPPORTED_LOWER").sum()
                ),
                "median_signed_effect": float(group["signed_effect"].median()),
                "minimum_signed_effect": float(group["signed_effect"].min()),
                "maximum_signed_effect": float(group["signed_effect"].max()),
                "minimum_global_fdr": float(group["fdr_global"].min()),
                "consistency_class": classify_consistency(group),
                "significant_eids": ";".join(significant["eid"].astype(str)),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["significant_cell_lines_n", "minimum_global_fdr"],
        ascending=[False, True],
    ).reset_index(drop=True)


def step_04() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(M26_SOURCE, sep="\t", low_memory=False)
    feature = pd.read_csv(M26_FEATURE, sep="\t", low_memory=False)
    cell_tests = pd.read_csv(M27_CELL_TEST, sep="\t", low_memory=False)
    matching = pd.read_csv(M27_MATCHING, sep="\t", low_memory=False)
    similarity = pd.read_csv(M27_SIMILARITY, sep="\t", low_memory=False)
    breadth = pd.read_csv(M27_BREADTH, sep="\t", low_memory=False)
    patterns = pd.read_csv(M27_PATTERNS, sep="\t", dtype={"support_pattern": str})

    dims = summarize_dimensions(
        source, feature, cell_tests, matching, similarity, breadth, patterns
    )
    annotated_tests = add_cell_test_annotations(cell_tests)
    level_summary = build_level_summary(source, feature, annotated_tests)
    significant = annotated_tests[
        annotated_tests["is_significant_global_fdr0_05"]
    ].sort_values(["eid", "fdr_global"]).reset_index(drop=True)
    consistency = build_feature_consistency(annotated_tests)
    patterns = patterns.sort_values(
        ["tumor_active_support_breadth", "top1_n"], ascending=[True, False]
    ).reset_index(drop=True)

    by_breadth = breadth.set_index("tumor_active_support_breadth")["top1_n"]
    unsupported_n = int(by_breadth.get(0, 0))
    specific_n = int(by_breadth.get(1, 0))
    multi_n = int(by_breadth.loc[by_breadth.index >= 2].sum())
    all_cell_n = int(by_breadth.get(dims["cell_line_n"], 0))
    class_counts = consistency["consistency_class"].value_counts()
    shared_feature_n = int(class_counts.get("SHARED_ALL_FIVE_CONSISTENT", 0))
    partial_feature_n = int(class_counts.get("PARTIAL_CELL_LINE_CONSISTENT", 0))
    heterogeneous_feature_n = int(class_counts.get("DIRECTION_HETEROGENEOUS", 0))
    n_feature = dims["feature_n"]
    top1_n = dims["top1_n"]

    key_results = pd.DataFrame(
        [
            {
                "metric": "pooled_predefined_source_significant",
                "value": int((source["fdr_bh"] < FDR_THRESHOLD).sum()),
                "denominator": len(source),
            },
            {
                "metric": "pooled_compact24_significant",
                "value": int(
                    (
                        feature.loc[
                            feature["scope"] == "compact24",
                            "fdr_bh_within_scope",
                        ]
                        < FDR_THRESHOLD
                    ).sum()
                ),
                "denominator": dims["m26_scope_test_n"].get("compact24", 0),
            },
            {
                "metric": "cell_line_resolved_significant",
                "value": len(significant),
                "denominator": len(annotated_tests),
            },
            {
                "metric": "shared_all_five_consistent_features",
                "value": shared_feature_n,
                "denominator": n_feature,
            },
            {
                "metric": "partial_cell_line_consistent_features",
                "value": partial_feature_n,
                "denominator": n_feature,
            },
            {
                "metric": "direction_heterogeneous_features",
                "value": heterogeneous_feature_n,
                "denominator": n_feature,
            },
            {
                "metric": "exactly_one_cell_line_supported_top1",
                "value": specific_n,
                "denominator": top1_n,
            },
            {
                "metric": "multiple_cell_line_supported_top1",
                "value": multi_n,
                "denominator": top1_n,
            },
            {
                "metric": "all_cell_line_supported_top1",
                "value": all_cell_n,
                "denominator": top1_n,
            },
            {
                "metric": "no_cell_line_support_top1",
                "value": unsupported_n,
                "denominator": top1_n,
            },
        ]
    )


    atomic_write_tsv(level_summary, OUT_LEVEL_SUMMARY)
    atomic_write_tsv(breadth, OUT_BREADTH)
    atomic_write_tsv(patterns, OUT_PATTERNS)
    atomic_write_tsv(key_results, OUT_KEY_RESULTS)


# --------------------------------------------------------------------------
# step 05: Analyze potential context realization
# --------------------------------------------------------------------------



PROJECT_ROOT__05 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__05 = (PROJECT_ROOT__05 / "Biological_Interpretation").resolve()
Modeling_OUTPUT__05 = (PROJECT_ROOT__05 / "Modeling/output").resolve()

STEP03_DIR = BIOLOGY_ROOT__05 / "output/03_analyze_sequence_grammar"
STEP04_DIR = BIOLOGY_ROOT__05 / "output/04_analyze_context_specificity"
M26_DIR__05 = BIOLOGY_ROOT__05 / "26_host_regulatory_mechanism_interpretation"
M27_DIR__05 = BIOLOGY_ROOT__05 / "27_reference_to_tumor_context_interpretation"

SEQUENCE_SUMMARY = STEP03_DIR / "03_sequence_grammar_comparison_summary.tsv"
SEQUENCE_THRESHOLDS = STEP03_DIR / "03_significant_6mer_effect_thresholds.tsv"
CONTEXT_LEVELS = STEP04_DIR / "04_analysis_level_summary.tsv"
CONTEXT_KEYS = STEP04_DIR / "04_key_results.tsv"
REGULATORY_CORRELATIONS = M26_DIR__05 / "26_regulatory_tumor_support_correlations.tsv"
SUPPORT_BREADTH = M27_DIR__05 / "27_cell_line_support_breadth.tsv"

OUT_DIR__05 = BIOLOGY_ROOT__05 / "output/05_analyze_potential_context_realization"


FDR_THRESHOLD__05 = 0.05








def summarize_dimensions__05(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    levels: pd.DataFrame,
    keys: pd.DataFrame,
    correlations: pd.DataFrame,
    breadth: pd.DataFrame,
) -> dict[str, int]:
    return {
        "kmer_n": int(pd.to_numeric(summary["canonical_6mers_tested"]).max()),
        "sequence_comparison_n": int(summary["comparison"].nunique()),
        "effect_threshold_n": int(thresholds["absolute_rate_delta_threshold"].nunique()),
        "effect_threshold_row_n": len(thresholds),
        "context_level_n": int(levels["analysis_level"].nunique()),
        "context_key_metric_n": int(keys["metric"].nunique()),
        "correlation_n": len(correlations),
        "correlation_valid_n": int(correlations["spearman_rho"].notna().sum()),
        "correlation_universe_n": int(pd.to_numeric(correlations["rows"]).max()),
        "breadth_level_n": int(breadth["tumor_active_support_breadth"].nunique()),
        "top1_n": int(pd.to_numeric(breadth["top1_n"]).sum()),
    }


def build_sequence_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "TOP1_vs_MATCHED_NON_TOP1",
        "TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1",
    ]
    result = summary.set_index("comparison").loc[keep].reset_index().copy()
    result.insert(
        1,
        "biological_layer",
        ["ACTIVITY_POTENTIAL", "CONTEXT_SPECIFIC_REALIZATION"],
    )
    return result


def annotate_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    result = correlations.copy()
    result["absolute_spearman_rho"] = pd.to_numeric(
        result["spearman_rho"], errors="coerce"
    ).abs()
    result["is_significant_fdr0_05"] = result["fdr_bh"] < FDR_THRESHOLD__05
    result["direction"] = "UNAVAILABLE"
    result.loc[result["spearman_rho"] > 0, "direction"] = "POSITIVE"
    result.loc[result["spearman_rho"] < 0, "direction"] = "NEGATIVE"
    return result.sort_values(
        ["is_significant_fdr0_05", "absolute_spearman_rho"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_breadth_gradient(breadth: pd.DataFrame) -> pd.DataFrame:
    result = breadth.sort_values("tumor_active_support_breadth").reset_index(drop=True)
    baseline = result.loc[result["tumor_active_support_breadth"] == 0].iloc[0]
    result["median_rank_improvement_vs_breadth0"] = (
        baseline["median_final_rank"] - result["median_final_rank"]
    )
    result["reference_active_prevalence_delta_vs_breadth0"] = (
        result["median_reference_active_histone_prevalence"]
        - baseline["median_reference_active_histone_prevalence"]
    )
    result["reference_repressive_prevalence_delta_vs_breadth0"] = (
        result["median_reference_repressive_histone_prevalence"]
        - baseline["median_reference_repressive_histone_prevalence"]
    )
    return result


def metric_value(keys: pd.DataFrame, metric: str) -> tuple[int, int]:
    row = keys.loc[keys["metric"] == metric]
    return int(row.iloc[0]["value"]), int(row.iloc[0]["denominator"])


def build_layer_evidence(
    sequence_contrast: pd.DataFrame,
    context_levels: pd.DataFrame,
    context_keys: pd.DataFrame,
    correlations: pd.DataFrame,
    kmer_n: int,
) -> pd.DataFrame:
    sequence_map = sequence_contrast.set_index("comparison")
    resolved = context_levels.set_index("analysis_level").loc["CELL_LINE_RESOLVED"]
    one_n, top1_n = metric_value(
        context_keys, "exactly_one_cell_line_supported_top1"
    )
    multi_n, _ = metric_value(
        context_keys, "multiple_cell_line_supported_top1"
    )
    valid_correlations = correlations[correlations["spearman_rho"].notna()]
    significant_correlations = valid_correlations[
        valid_correlations["is_significant_fdr0_05"]
    ]
    rows = [
        {
            "layer": "ACTIVITY_POTENTIAL",
            "evidence": "Top1 versus matched background sequence composition",
            "observed_n": int(
                sequence_map.loc[
                    "TOP1_vs_MATCHED_NON_TOP1", "significant_fdr0_05"
                ]
            ),
            "denominator_n": kmer_n,
            "observed_fraction": float(
                sequence_map.loc[
                    "TOP1_vs_MATCHED_NON_TOP1", "significant_fdr0_05"
                ]
            )
            / kmer_n,
        },
        {
            "layer": "CONTEXT_SPECIFIC_REALIZATION",
            "evidence": "Tumor-supported versus unsupported Top1 sequence composition",
            "observed_n": int(
                sequence_map.loc[
                    "TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1", "significant_fdr0_05"
                ]
            ),
            "denominator_n": kmer_n,
            "observed_fraction": float(
                sequence_map.loc[
                    "TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1", "significant_fdr0_05"
                ]
            )
            / kmer_n,
        },
        {
            "layer": "CONTEXT_SPECIFIC_REALIZATION",
            "evidence": "Cell-line-resolved host-regulatory associations",
            "observed_n": int(resolved["significant_fdr0_05_n"]),
            "denominator_n": int(resolved["tests_n"]),
            "observed_fraction": float(resolved["significant_fraction"]),
        },
        {
            "layer": "CONTEXT_SPECIFIC_REALIZATION",
            "evidence": "Non-tumor regulatory summaries correlated with tumor support",
            "observed_n": len(significant_correlations),
            "denominator_n": len(valid_correlations),
            "observed_fraction": len(significant_correlations)
            / len(valid_correlations),
        },
        {
            "layer": "REALIZATION_PATTERN",
            "evidence": "Top1 supported in exactly one tumor cell line",
            "observed_n": one_n,
            "denominator_n": top1_n,
            "observed_fraction": one_n / top1_n,
        },
        {
            "layer": "REALIZATION_PATTERN",
            "evidence": "Top1 supported in multiple tumor cell lines",
            "observed_n": multi_n,
            "denominator_n": top1_n,
            "observed_fraction": multi_n / top1_n,
        },
    ]
    return pd.DataFrame(rows)


def step_05() -> None:
    OUT_DIR__05.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# step 06: Analyze tumor regulatory reuse
# --------------------------------------------------------------------------



PROJECT_ROOT__06 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__06 = (PROJECT_ROOT__06 / "Biological_Interpretation").resolve()
Modeling_OUTPUT__06 = (PROJECT_ROOT__06 / "Modeling/output").resolve()

M24_DIR = Modeling_OUTPUT__06 / "24_project_frozen_herv_ranking_to_tumor_epigenomes"
M27_DIR__06 = BIOLOGY_ROOT__06 / "27_reference_to_tumor_context_interpretation"

M24_CANDIDATES = M24_DIR / "24_potential_reactivation_candidates.tsv"
M24_ENRICHMENT = M24_DIR / "24_tumor_support_enrichment.tsv"
M27_TOP1 = M27_DIR__06 / "27_top1_cell_context_annotation.tsv"
M27_SPECIFIC = M27_DIR__06 / "27_cell_line_specific_candidates.tsv"
M27_BREADTH__06 = M27_DIR__06 / "27_cell_line_support_breadth.tsv"
M27_PATTERNS__06 = M27_DIR__06 / "27_cell_line_support_patterns.tsv"
M27_SIMILARITY__06 = M27_DIR__06 / "27_cell_line_pairwise_similarity.tsv"
M27_STATE_TRANSITION = M27_DIR__06 / "27_reference_tumor_state_transition.tsv"
SOURCE_TABLE = BIOLOGY_ROOT__06 / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M28_REPRESENTATIVES = BIOLOGY_ROOT__06 / 'output/28_sequence_grammar_and_representative_loci/28_representative_loci.tsv'

OUT_DIR__06 = BIOLOGY_ROOT__06 / "output/06_analyze_tumor_regulatory_reuse"


ID_COL = "herv_internal_id"

CELL_LINES__06 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

ACTIVE_MARKS = ["H3K27ac", "H3K4me1", "H3K4me3"]

SOURCE_BASE_COLUMNS = [
    ID_COL,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repFamily",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
    "nearest_gene_name",
    "nearest_gene_type",
    "tumor_active_support_breadth",
]

TOP1_CUTOFF = "Top1%"
TOP5_CUTOFF = "Top5%"
TOP10_CUTOFF = "FrozenTop10Candidate"








def source_columns() -> list[str]:
    columns = list(SOURCE_BASE_COLUMNS)
    columns.extend(f"roadmap_{mark}_sample_count" for mark in ACTIVE_MARKS)
    columns.extend(
        f"tumor_{eid}_{mark}_has"
        for eid in CELL_LINES__06
        for mark in ACTIVE_MARKS
    )
    return columns


def build_tumor_exclusive_tables(
    source: pd.DataFrame,
    cutoff_k: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rank = pd.to_numeric(source["herv_functional_priority_rank"], errors="raise")
    universe_n = len(source)
    top1_n = int(cutoff_k[TOP1_CUTOFF])
    top5_n = int(cutoff_k[TOP5_CUTOFF])
    top10_n = int(cutoff_k[TOP10_CUTOFF])
    top1 = rank <= top1_n
    flag_map: dict[str, pd.Series] = {}
    summary_rows = []
    for eid, cell_line in CELL_LINES__06.items():
        for mark in ACTIVE_MARKS:
            non_tumor_absent = pd.to_numeric(
                source[f"roadmap_{mark}_sample_count"], errors="raise"
            ).eq(0)
            tumor_present = pd.to_numeric(
                source[f"tumor_{eid}_{mark}_has"], errors="raise"
            ).eq(1)
            exclusive = non_tumor_absent & tumor_present
            label = f"{eid}:{mark}"
            flag_map[label] = exclusive
            summary_rows.append(
                {
                    "eid": eid,
                    "cell_line": cell_line,
                    "active_mark": mark,
                    "non_tumor_absent_n": int(non_tumor_absent.sum()),
                    "tumor_present_n": int(tumor_present.sum()),
                    "tumor_exclusive_locus_n": int(exclusive.sum()),
                    "top1_tumor_exclusive_locus_n": int((exclusive & top1).sum()),
                }
            )

    flag_frame = pd.DataFrame(flag_map, index=source.index)
    any_exclusive = flag_frame.any(axis=1)
    catalog = source.loc[any_exclusive, SOURCE_BASE_COLUMNS].copy()
    selected_flags = flag_frame.loc[any_exclusive]
    labels = list(selected_flags.columns)
    flag_values = selected_flags.to_numpy(dtype=bool)
    pair_strings = [
        ";".join(label for label, present in zip(labels, row) if present)
        for row in flag_values
    ]
    eid_strings = [
        ";".join(sorted({pair.split(":", 1)[0] for pair in value.split(";")}))
        for value in pair_strings
    ]
    mark_strings = [
        ";".join(sorted({pair.split(":", 1)[1] for pair in value.split(";")}))
        for value in pair_strings
    ]
    catalog["tumor_exclusive_eid_mark_pairs"] = pair_strings
    catalog["tumor_exclusive_eids"] = eid_strings
    catalog["tumor_exclusive_active_marks"] = mark_strings
    catalog["tumor_exclusive_eid_mark_pair_n"] = selected_flags.sum(axis=1).to_numpy()
    catalog["tumor_exclusive_cell_line_n"] = [
        len(value.split(";")) for value in eid_strings
    ]
    catalog["is_activity_top1"] = rank.loc[any_exclusive].le(top1_n).to_numpy()

    non_tumor_active = source[
        [f"roadmap_{mark}_sample_count" for mark in ACTIVE_MARKS]
    ].gt(0).any(axis=1)
    tumor_active = pd.to_numeric(
        source["tumor_active_support_breadth"], errors="raise"
    ).gt(0)
    state_exclusive = (~non_tumor_active) & tumor_active
    group_catalog = source.loc[state_exclusive, SOURCE_BASE_COLUMNS].copy()
    group_eid_flags = []
    group_eid_names = []
    for eid, cell_line in CELL_LINES__06.items():
        eid_active = pd.concat(
            [
                pd.to_numeric(source[f"tumor_{eid}_{mark}_has"], errors="raise").eq(1)
                for mark in ACTIVE_MARKS
            ],
            axis=1,
        ).any(axis=1)
        group_eid_flags.append(eid_active.loc[state_exclusive].to_numpy())
        group_eid_names.append((eid, cell_line))
    group_flag_matrix = np.column_stack(group_eid_flags)
    group_catalog["tumor_group_supported_eids"] = [
        ";".join(
            eid
            for (eid, _), present in zip(group_eid_names, row)
            if present
        )
        for row in group_flag_matrix
    ]
    group_catalog["tumor_group_supporting_cancer_types"] = [
        ";".join(
            cell_line
            for (_, cell_line), present in zip(group_eid_names, row)
            if present
        )
        for row in group_flag_matrix
    ]
    group_catalog["tumor_group_support_breadth"] = group_flag_matrix.sum(axis=1)
    group_catalog["tumor_group_specific"] = True
    group_catalog["is_activity_top1"] = rank.loc[state_exclusive].le(top1_n).to_numpy()
    group_catalog["is_activity_top5"] = rank.loc[state_exclusive].le(top5_n).to_numpy()
    group_catalog["is_activity_top10"] = rank.loc[state_exclusive].le(top10_n).to_numpy()
    group_catalog["activity_rank_percentile"] = (
        rank.loc[state_exclusive].to_numpy() / universe_n
    )
    group_catalog = group_catalog.sort_values(
        "herv_functional_priority_rank"
    ).reset_index(drop=True)
    catalog["is_state_level_tumor_exclusive"] = state_exclusive.loc[
        any_exclusive
    ].to_numpy()
    catalog = catalog.sort_values("herv_functional_priority_rank").reset_index(drop=True)
    counts = {
        "non_tumor_any_active_n": int(non_tumor_active.sum()),
        "tumor_any_active_n": int(tumor_active.sum()),
        "state_level_tumor_exclusive_n": int(state_exclusive.sum()),
        "state_level_tumor_exclusive_top1_n": int(
            (state_exclusive & top1).sum()
        ),
        "same_mark_tumor_exclusive_unique_locus_n": len(catalog),
        "same_mark_tumor_exclusive_top1_n": int(catalog["is_activity_top1"].sum()),
    }
    return pd.DataFrame(summary_rows), catalog, group_catalog, counts


def read_representative_annotations() -> pd.DataFrame:
    columns = [
        ID_COL,
        "reference_active_histone_prevalence_mean",
        "reference_repressive_histone_prevalence_mean",
        "reference_active_minus_repressive_prevalence",
        "tumor_active_support_pattern",
    ]
    frame = pd.read_csv(M28_REPRESENTATIVES, sep="\t", low_memory=False)
    return frame[columns]


def build_candidate_catalog(
    top1: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    annotation_columns = [
        ID_COL,
        "reference_active_histone_prevalence_mean",
        "reference_repressive_histone_prevalence_mean",
        "reference_active_minus_repressive_prevalence",
        "tumor_active_support_pattern",
    ]
    new_columns = [
        column
        for column in annotation_columns
        if column == ID_COL or column not in candidates.columns
    ]
    result = candidates.merge(
        top1[new_columns],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    breadth = result["tumor_active_support_breadth"].astype(int)
    result.insert(
        result.columns.get_loc("tumor_cell_line_support_class") + 1,
        "tumor_regulatory_reuse_scope",
        np.select(
            [
                breadth == 1,
                breadth.between(2, len(CELL_LINES__06) - 1),
                breadth == len(CELL_LINES__06),
            ],
            [
                "CELL_LINE_SPECIFIC",
                "MULTI_CELL_LINE_SHARED",
                "PAN_CELL_LINE_SHARED",
            ],
            default="INVALID",
        ),
    )
    result.insert(
        result.columns.get_loc("tumor_regulatory_reuse_scope") + 1,
        "tumor_regulatory_reuse_candidate",
        True,
    )
    return result.sort_values("herv_functional_priority_rank").reset_index(drop=True)


def build_enrichment_tables(
    enrichment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cutoff_order = enrichment["rank_cutoff"].drop_duplicates().tolist()
    cutoff_k = enrichment.groupby("rank_cutoff")["K"].first().to_dict()
    cell_types = {"any_active", "active_dominant"}
    pan_types = {
        "any_active_any_cell_line",
        "active_dominant_any_cell_line",
        "active_support_multiple_cell_lines",
    }
    keep = (
        (enrichment["scope"] == "cell_line")
        & enrichment["support_type"].isin(cell_types)
    ) | (
        (enrichment["scope"] == "pan_tumor")
        & enrichment["support_type"].isin(pan_types)
    )
    gradient = enrichment.loc[keep].copy()
    cutoff_order = pd.Categorical(
        gradient["rank_cutoff"],
        categories=cutoff_order,
        ordered=True,
    )
    gradient = gradient.assign(_cutoff_order=cutoff_order).sort_values(
        ["scope", "eid", "support_type", "_cutoff_order"]
    ).drop(columns="_cutoff_order").reset_index(drop=True)
    top1 = gradient[gradient["rank_cutoff"] == TOP1_CUTOFF].copy()
    return top1.reset_index(drop=True), gradient, cutoff_k


def build_cell_line_summary(
    top1_enrichment: pd.DataFrame,
    specific: pd.DataFrame,
    top1_n: int,
) -> pd.DataFrame:
    rows = []
    for eid, cell_line in CELL_LINES__06.items():
        subset = top1_enrichment[top1_enrichment["eid"] == eid].set_index(
            "support_type"
        )
        active = subset.loc["any_active"]
        dominant = subset.loc["active_dominant"]
        rows.append(
            {
                "eid": eid,
                "cell_line": cell_line,
                "top1_n": top1_n,
                "active_supported_n": int(active["hits"]),
                "active_supported_rate": float(active["support_rate"]),
                "active_support_fold_enrichment": float(active["fold_enrichment"]),
                "active_dominant_n": int(dominant["hits"]),
                "active_dominant_rate": float(dominant["support_rate"]),
                "active_dominant_fold_enrichment": float(
                    dominant["fold_enrichment"]
                ),
                "cell_line_specific_n": int(
                    (specific["specific_tumor_eid"] == eid).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def step_06() -> None:
    OUT_DIR__06.mkdir(parents=True, exist_ok=True)

    enrichment = pd.read_csv(M24_ENRICHMENT, sep="\t")
    source = pd.read_parquet(SOURCE_TABLE, columns=source_columns())
    source = source.merge(
        read_representative_annotations(),
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    top1_enrichment, enrichment_gradient, cutoff_k = build_enrichment_tables(enrichment)
    top1_n = int(cutoff_k[TOP1_CUTOFF])
    top5_n = int(cutoff_k[TOP5_CUTOFF])
    top10_n = int(cutoff_k[TOP10_CUTOFF])

    tumor_exclusive_marks, tumor_exclusive_loci, tumor_group_catalog, tumor_exclusive_counts = (
        build_tumor_exclusive_tables(source, cutoff_k)
    )

    non_tumor_group_active = source[
        [f"roadmap_{mark}_sample_count" for mark in ACTIVE_MARKS]
    ].gt(0).any(axis=1)
    tumor_group_active = pd.to_numeric(
        source["tumor_active_support_breadth"], errors="raise"
    ).gt(0)
    group_classes = [
        ("TUMOR_GROUP_SPECIFIC", tumor_group_active & ~non_tumor_group_active),
        ("BOTH_GROUPS_SUPPORTED", tumor_group_active & non_tumor_group_active),
        ("NON_TUMOR_ONLY", ~tumor_group_active & non_tumor_group_active),
        ("NEITHER_GROUP_SUPPORTED", ~tumor_group_active & ~non_tumor_group_active),
    ]
    rank = pd.to_numeric(source["herv_functional_priority_rank"], errors="coerce")
    group_summary_rows = []
    for label, mask in group_classes:
        group_summary_rows.append(
            {
                "comparison_class": label,
                "locus_n": int(mask.sum()),
                "fraction_of_genome": float(mask.mean()),
                "top1_n": int((mask & rank.le(top1_n)).sum()),
                "top5_n": int((mask & rank.le(top5_n)).sum()),
                "top10_n": int((mask & rank.le(top10_n)).sum()),
            }
        )


# --------------------------------------------------------------------------
# step 07: Analyze tumor versus non-tumor quantitative support
# --------------------------------------------------------------------------



PROJECT_ROOT__07 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__07 = (PROJECT_ROOT__07 / "Biological_Interpretation").resolve()
M24_ENRICHMENT__07 = PROJECT_ROOT__07 / "Modeling/output/24_project_frozen_herv_ranking_to_tumor_epigenomes" / "24_tumor_support_enrichment.tsv"
M27_MANIFEST = BIOLOGY_ROOT__07 / "output/27_reference_to_tumor_context_interpretation" / "27_reference_to_tumor_context_manifest.json"
SOURCE_TABLE__07 = BIOLOGY_ROOT__07 / 'output/01_build_biological_source_table/01_biological_source_table.parquet'

OUT_DIR__07 = BIOLOGY_ROOT__07 / "output/07_analyze_tumor_vs_nontumor_quantitative"


OUT_STATE_SUMMARY = OUT_DIR__07 / "07_active_state_quantitative_summary.tsv"


ID_COL__07 = "herv_internal_id"

CELL_LINES__07 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}
ACTIVE_MARKS__07 = ["H3K27ac", "H3K4me1", "H3K4me3"]

RANK_SCOPES = {
    "Genomewide": None,
    "Top0.5%": "Top0.5%",
    "Top1%": "Top1%",
    "Top5%": "Top5%",
    "Top10%": "FrozenTop10Candidate",
}

TOP1_SCOPE = "Top1%"
TUMOR_GROUP_UNIT = "TUMOR_GROUP"

BASE_COLUMNS = [
    ID_COL__07,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repFamily",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
    "nearest_gene_name",
    "nearest_gene_type",
]




def comparison_units() -> list[tuple[str, str]]:
    return [
        (TUMOR_GROUP_UNIT, f"Pooled {len(CELL_LINES__07)} tumor cell lines"),
        *CELL_LINES__07.items(),
    ]


def atomic_write(payload: "pd.DataFrame | str", path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(temporary, sep="\t", index=False)
    else:
        temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def source_columns__07() -> list[str]:
    columns = list(BASE_COLUMNS)
    columns.extend(f"roadmap_{mark}_sample_count" for mark in ACTIVE_MARKS__07)
    columns.extend(
        f"tumor_{eid}_{mark}_has" for eid in CELL_LINES__07 for mark in ACTIVE_MARKS__07
    )
    return columns


def load_source() -> pd.DataFrame:
    source = pd.read_parquet(SOURCE_TABLE__07, columns=source_columns__07())
    rank = "herv_functional_priority_rank"
    source[rank] = pd.to_numeric(source[rank], errors="raise").astype(np.int64)
    for mark in ACTIVE_MARKS__07:
        column = f"roadmap_{mark}_sample_count"
        source[column] = pd.to_numeric(source[column], errors="raise").astype(np.int64)
        for eid in CELL_LINES__07:
            flag = f"tumor_{eid}_{mark}_has"
            source[flag] = pd.to_numeric(source[flag], errors="raise").astype(np.int8)
    return source


def load_rank_scopes(universe_n: int) -> dict[str, int]:
    enrichment = pd.read_csv(M24_ENRICHMENT__07, sep="\t", usecols=["rank_cutoff", "K"])
    cutoff_k = enrichment.groupby("rank_cutoff")["K"].first().to_dict()
    return {
        scope: universe_n if label is None else int(cutoff_k[label])
        for scope, label in RANK_SCOPES.items()
    }


def load_non_tumor_denominators() -> dict[str, int]:
    manifest = json.loads(M27_MANIFEST.read_text(encoding="utf-8"))
    denominators = manifest["non_tumor_mark_denominators"]
    return {mark: int(denominators[mark]) for mark in ACTIVE_MARKS__07}


def reference_prevalence(source: pd.DataFrame, denominators: dict[str, int]) -> pd.DataFrame:
    counts = source[[f"roadmap_{m}_sample_count" for m in ACTIVE_MARKS__07]].astype(float)
    counts.columns = ACTIVE_MARKS__07
    return counts.div([float(denominators[m]) for m in ACTIVE_MARKS__07], axis="columns")


def comparison_values(
    source: pd.DataFrame, reference: pd.DataFrame, mark: str, unit: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if unit == TUMOR_GROUP_UNIT:
        observations = source[
            [f"tumor_{eid}_{mark}_has" for eid in CELL_LINES__07]
        ].astype(float)
        return observations.mean(axis=1), observations.gt(0).any(axis=1), reference[mark]
    tumor_prevalence = source[f"tumor_{unit}_{mark}_has"].astype(float)
    return tumor_prevalence, tumor_prevalence.gt(0), reference[mark]


def build_mark_summary(
    source: pd.DataFrame, reference: pd.DataFrame, scopes: dict[str, int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for unit, cell_line in comparison_units():
        for mark in ACTIVE_MARKS__07:
            tumor_prevalence, tumor_supported, reference_values = comparison_values(
                source, reference, mark, unit
            )
            for scope, cutoff in scopes.items():
                mask = source["herv_functional_priority_rank"].le(cutoff)
                tumor_rate = float(tumor_prevalence[mask].mean())
                reference_rate = float(reference_values[mask].mean())
                ratio = (
                    tumor_rate / reference_rate if reference_rate > 0 else np.nan
                )
                rows.append(
                    {
                        "comparison_unit": unit,
                        "cell_line": cell_line,
                        "active_mark": mark,
                        "rank_scope": scope,
                        "rank_cutoff": cutoff,
                        "locus_n": int(mask.sum()),
                        "tumor_support_observations_n": int(round(tumor_prevalence[mask].sum() * (len(CELL_LINES__07) if unit == TUMOR_GROUP_UNIT else 1))),
                        "tumor_support_observation_rate": tumor_rate,
                        "tumor_supported_loci_n": int(tumor_supported[mask].sum()),
                        "tumor_supported_loci_rate": float(tumor_supported[mask].mean()),
                        "non_tumor_positive_loci_n": int(reference_values[mask].gt(0).sum()),
                        "non_tumor_positive_loci_rate": float(reference_values[mask].gt(0).mean()),
                        "non_tumor_mean_sample_prevalence": reference_rate,
                        "tumor_minus_non_tumor_prevalence": tumor_rate - reference_rate,
                        "tumor_to_non_tumor_prevalence_ratio": ratio,
                    }
                )
    return pd.DataFrame(rows)


def build_active_state_summary(
    source: pd.DataFrame, reference: pd.DataFrame, scopes: dict[str, int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference_state = reference.mean(axis=1)
    for unit, cell_line in comparison_units():
        tumor_by_mark = []
        for mark in ACTIVE_MARKS__07:
            tumor_prevalence, _, _ = comparison_values(source, reference, mark, unit)
            tumor_by_mark.append(tumor_prevalence)
        tumor_state = pd.concat(tumor_by_mark, axis=1).mean(axis=1)
        for scope, cutoff in scopes.items():
            mask = source["herv_functional_priority_rank"].le(cutoff)
            tumor_rate = float(tumor_state.loc[mask].mean())
            reference_rate = float(reference_state.loc[mask].mean())
            rows.append(
                {
                    "comparison_unit": unit,
                    "cell_line": cell_line,
                    "rank_scope": scope,
                    "rank_cutoff": cutoff,
                    "locus_n": int(mask.sum()),
                    "tumor_active_mark_prevalence": tumor_rate,
                    "non_tumor_active_mark_prevalence": reference_rate,
                    "tumor_minus_non_tumor_active_mark_prevalence": tumor_rate - reference_rate,
                    "tumor_to_non_tumor_active_mark_prevalence_ratio": (
                        tumor_rate / reference_rate if reference_rate > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_locus_table(
    source: pd.DataFrame, denominators: dict[str, int], universe_n: int,
    top1_n: int, top1_only: bool,
) -> pd.DataFrame:
    cutoff = top1_n if top1_only else universe_n
    mask = source["herv_functional_priority_rank"].le(cutoff)
    base = source.loc[mask, BASE_COLUMNS].copy()
    base["activity_rank_percentile"] = base["herv_functional_priority_rank"] / universe_n
    rows: list[pd.DataFrame] = []
    for mark in ACTIVE_MARKS__07:
        reference_count = source.loc[mask, f"roadmap_{mark}_sample_count"].astype(int)
        reference_prevalence = reference_count.astype(float) / float(denominators[mark])
        tumor_flags = pd.DataFrame(
            {
                eid: source.loc[mask, f"tumor_{eid}_{mark}_has"].astype(int)
                for eid in CELL_LINES__07
            },
            index=base.index,
        )
        support_n = tumor_flags.sum(axis=1)
        tumor_prevalence = support_n.astype(float) / len(CELL_LINES__07)
        mark_rows = base.copy()
        mark_rows["active_mark"] = mark
        mark_rows["non_tumor_sample_count"] = reference_count.to_numpy()
        mark_rows["non_tumor_sample_denominator"] = denominators[mark]
        mark_rows["non_tumor_prevalence"] = reference_prevalence.to_numpy()
        for eid in CELL_LINES__07:
            mark_rows[f"tumor_{eid}_has"] = tumor_flags[eid].to_numpy()
        mark_rows["tumor_group_support_n"] = support_n.to_numpy()
        mark_rows["tumor_group_prevalence"] = tumor_prevalence.to_numpy()
        mark_rows["tumor_group_minus_reference_prevalence"] = (
            tumor_prevalence - reference_prevalence
        ).to_numpy()
        majority_n = len(CELL_LINES__07) // 2 + 1
        mark_rows["tumor_group_supported"] = support_n.gt(0).astype(np.int8).to_numpy()
        mark_rows["tumor_group_majority_supported"] = support_n.ge(majority_n).astype(np.int8).to_numpy()
        mark_rows["tumor_group_supporting_eids"] = [
            ";".join(eid for eid, present in zip(CELL_LINES__07, row) if present)
            for row in tumor_flags.itertuples(index=False, name=None)
        ]
        rows.append(mark_rows.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def top1_tumor_group(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        (frame["comparison_unit"] == TUMOR_GROUP_UNIT)
        & (frame["rank_scope"] == TOP1_SCOPE)
    ]


def build_key_results(
    mark_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    top1_loci: pd.DataFrame,
    differential_loci: pd.DataFrame,
    universe_n: int,
) -> pd.DataFrame:
    top1_group = top1_tumor_group(mark_summary)
    state_top1 = top1_tumor_group(state_summary).iloc[0]
    rows: list[tuple[str, object, int]] = [
        ("top1_tumor_group_active_mark_prevalence",
         float(state_top1["tumor_active_mark_prevalence"]), 1),
        ("top1_non_tumor_active_mark_prevalence",
         float(state_top1["non_tumor_active_mark_prevalence"]), 1),
        ("top1_tumor_minus_non_tumor_active_mark_prevalence",
         float(state_top1["tumor_minus_non_tumor_active_mark_prevalence"]), 1),
        ("top1_positive_tumor_group_mark_differential_pairs",
         int(top1_loci["tumor_group_minus_reference_prevalence"].gt(0).sum()),
         len(top1_loci)),
        ("genomewide_positive_tumor_group_mark_differential_pairs",
         len(differential_loci), universe_n * len(ACTIVE_MARKS__07)),
    ]
    rows.extend(
        (f"top1_{row.active_mark}_tumor_group_minus_reference",
         float(row.tumor_minus_non_tumor_prevalence), 1)
        for row in top1_group.itertuples(index=False)
    )
    return pd.DataFrame(rows, columns=["metric", "value", "denominator"])


def step_07() -> None:
    OUT_DIR__07.mkdir(parents=True, exist_ok=True)

    source = load_source()
    universe_n = len(source)
    scopes = load_rank_scopes(universe_n)
    denominators = load_non_tumor_denominators()
    top1_n = scopes[TOP1_SCOPE]
    reference = reference_prevalence(source, denominators)
    state_summary = build_active_state_summary(source, reference, scopes)
    all_loci = build_locus_table(source, denominators, universe_n, top1_n, False)
    differential_loci = all_loci.loc[
        all_loci["tumor_group_minus_reference_prevalence"].gt(0)
    ].copy()
    differential_loci = differential_loci.sort_values(
        ["herv_functional_priority_rank", "active_mark"]
    ).reset_index(drop=True)

    atomic_write(state_summary, OUT_STATE_SUMMARY)


# --------------------------------------------------------------------------
# step 08: Recover known regulatory evidence
# --------------------------------------------------------------------------



PROJECT_ROOT__08 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__08 = (PROJECT_ROOT__08 / "Biological_Interpretation").resolve()
Modeling_OUTPUT__08 = (PROJECT_ROOT__08 / "Modeling/output").resolve()

M29_VALID = Modeling_OUTPUT__08 / "29/29_validation_scores_and_threshold_labels.parquet"
M19_VALID = Modeling_OUTPUT__08 / '19_build_functional_evidence_candidate_pool/19_valid_functional_evidence_table.parquet'

OUT_DIR__08 = BIOLOGY_ROOT__08 / "output/08_known_regulatory_evidence_recovery"


ID_COL__08 = "herv_internal_id"
FANTOM_LABELS = ["is_fantom5_enhancer", "is_fantom5_enhancer_50bp"]
EVIDENCE_LABELS = [
    "is_ccre_enhancer_like",
    "is_mpra_active_supported",
    "is_serve_gtex_supported",
]

PRIMARY_COMPARISONS = [
    ("final_global", "score_final_global", "FANTOM5", "is_fantom5_enhancer"),
    ("final_global", "score_final_global", "FANTOM5_50bp", "is_fantom5_enhancer_50bp"),
    ("final_global", "score_final_global", "cCRE_enhancer_like", "is_ccre_enhancer_like"),
    ("final_global", "score_final_global", "MPRA_active", "is_mpra_active_supported"),
    ("final_global", "score_final_global", "SERVE_GTEx", "is_serve_gtex_supported"),
    ("cCRE_head", "score_ccre_enhancer", "cCRE_enhancer_like", "is_ccre_enhancer_like"),
    ("MPRA_head", "score_mpra_activity", "MPRA_active", "is_mpra_active_supported"),
    ("SERVE_head", "score_serve_gtex_support", "SERVE_GTEx", "is_serve_gtex_supported"),
    ("sequence_mean", "score_sequence_regulatory", "FANTOM5", "is_fantom5_enhancer"),
]

RANK_CUTOFFS = (
    ("Top1%_validation", 0.01),
    ("Top5%_validation", 0.05),
    ("Top10%_validation", 0.10),
)










def rank_percentile(scores: pd.Series) -> pd.Series:
    values = pd.to_numeric(scores, errors="raise")
    n = len(values)
    ranks = values.rank(method="first", ascending=False).to_numpy(dtype=float)
    return pd.Series((n - ranks + 1.0) / float(n), index=scores.index)


def source_audit(path: Path, frame: pd.DataFrame, source_name: str) -> dict[str, object]:
    return {
        "source": source_name,
        "path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "split_values": ";".join(sorted(frame["split"].astype(str).unique())),
        "unique_herv_ids": int(frame[ID_COL__08].nunique()),
    }


def evaluate_comparison(
    frame: pd.DataFrame,
    score_source: str,
    score_column: str,
    evidence: str,
    label_column: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scores = pd.to_numeric(frame[score_column], errors="raise")
    labels = pd.to_numeric(frame[label_column], errors="raise").astype(int)

    n = len(frame)
    positive_n = int(labels.sum())
    negative_n = int(n - positive_n)
    prevalence = positive_n / float(n)
    rank_pct = rank_percentile(scores)
    ordered = np.argsort(-scores.to_numpy(), kind="mergesort")

    row: dict[str, object] = {
        "score_source": score_source,
        "score_column": score_column,
        "evidence": evidence,
        "label_column": label_column,
        "n": n,
        "positive_n": positive_n,
        "negative_n": negative_n,
        "positive_prevalence": prevalence,
        "mean_score_positive": float(scores[labels.eq(1)].mean()) if positive_n else np.nan,
        "mean_score_negative": float(scores[labels.eq(0)].mean()) if negative_n else np.nan,
        "median_rank_pct_positive": float(rank_pct[labels.eq(1)].median()) if positive_n else np.nan,
        "median_rank_pct_negative": float(rank_pct[labels.eq(0)].median()) if negative_n else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
    }
    if positive_n and negative_n:
        row["auroc"] = float(roc_auc_score(labels, scores))
        row["auprc"] = float(average_precision_score(labels, scores))

    bins: list[dict[str, object]] = []
    for cutoff_name, fraction in RANK_CUTOFFS:
        k = max(1, int(math.floor(n * fraction)))
        top_indices = ordered[:k]
        top_labels = labels.iloc[top_indices]
        hits = int(top_labels.sum())
        top_rate = hits / float(k)
        fold = top_rate / prevalence if prevalence > 0 else np.nan
        bins.append(
            {
                "score_source": score_source,
                "score_column": score_column,
                "evidence": evidence,
                "label_column": label_column,
                "rank_bin": cutoff_name,
                "k": k,
                "n": n,
                "hits": hits,
                "support_rate": top_rate,
                "background_rate": prevalence,
                "fold_enrichment": fold,
            }
        )

    return row, bins


def step_08() -> None:
    OUT_DIR__08.mkdir(parents=True, exist_ok=True)

    score_columns = sorted({c for _, c, _, _ in PRIMARY_COMPARISONS})
    m29 = pd.read_parquet(
        M29_VALID, columns=[ID_COL__08, "split", *score_columns, *FANTOM_LABELS]
    )
    m19 = pd.read_parquet(M19_VALID, columns=[ID_COL__08, "split", *EVIDENCE_LABELS])
    m29[ID_COL__08] = m29[ID_COL__08].astype(str)
    m19[ID_COL__08] = m19[ID_COL__08].astype(str)

    merged = m29.merge(
        m19[[ID_COL__08, *EVIDENCE_LABELS]], on=ID_COL__08, how="inner", validate="one_to_one"
    )

    metrics_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for score_source, score_column, evidence, label_column in PRIMARY_COMPARISONS:
        metric_row, comparison_bins = evaluate_comparison(
            merged,
            score_source,
            score_column,
            evidence,
            label_column,
        )
        metrics_rows.append(metric_row)
        bin_rows.extend(comparison_bins)


# --------------------------------------------------------------------------
# step 09: Analyze unseen tumor context and specificity
# --------------------------------------------------------------------------



PROJECT_ROOT__09 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__09 = (PROJECT_ROOT__09 / "Biological_Interpretation").resolve()
Modeling_OUTPUT__09 = (PROJECT_ROOT__09 / "Modeling/output").resolve()

SOURCE_TABLE__09 = BIOLOGY_ROOT__09 / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M28_REPRESENTATIVES__09 = BIOLOGY_ROOT__09 / 'output/28_sequence_grammar_and_representative_loci/28_representative_loci.tsv'
M23_MANIFEST = Modeling_OUTPUT__09 / '23_generate_genomewide_herv_functional_ranking/23_genomewide_ranking_manifest.json'
M24_MANIFEST = Modeling_OUTPUT__09 / '24_project_frozen_herv_ranking_to_tumor_epigenomes/24_tumor_projection_manifest.json'
M07_SUMMARY = BIOLOGY_ROOT__09 / 'output/07_analyze_tumor_vs_nontumor_quantitative/07_active_state_quantitative_summary.tsv'

OUT_DIR__09 = BIOLOGY_ROOT__09 / "output/09_unseen_tumor_context_and_specificity"


ID_COL__09 = "herv_internal_id"
RANK_COL = "herv_functional_priority_rank"

CELL_LINES__09 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}
ACTIVE_COLS = {
    eid: f"tumor_{eid}_any_active_support" for eid in CELL_LINES__09
}
PATTERN_ORDER = list(CELL_LINES__09)
TOP1_BIN = "Top1%"
RANK_BIN_FRACTIONS = (
    ("Top0.5%", 0.005),
    (TOP1_BIN, 0.01),
    ("Top5%", 0.05),
    ("Top10%", 0.10),
)










def make_rank_bins(n: int) -> dict[str, int]:
    bins = {"Genomewide": n}
    bins.update(
        (name, max(1, int(round(n * fraction)))) for name, fraction in RANK_BIN_FRACTIONS
    )
    return bins


def scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    top1_k = make_rank_bins(len(frame))[TOP1_BIN]
    return {
        "Genomewide": np.ones(len(frame), dtype=bool),
        TOP1_BIN: frame[RANK_COL].to_numpy() <= top1_k,
    }


def summarize_rank_bins(frame: pd.DataFrame, outcomes: dict[str, pd.Series]) -> pd.DataFrame:
    n = len(frame)
    rank = frame[RANK_COL].to_numpy(dtype=np.int64)
    rows: list[dict[str, object]] = []
    for outcome_name, outcome in outcomes.items():
        values = outcome.to_numpy(dtype=bool)
        background = float(values.mean())
        background_n = int(values.sum())
        for rank_bin, k in make_rank_bins(n).items():
            selected = rank <= k
            selected_n = int(selected.sum())
            hits = int(values[selected].sum())
            rate = hits / float(selected_n)
            rows.append(
                {
                    "outcome": outcome_name,
                    "rank_bin": rank_bin,
                    "k": k,
                    "n": n,
                    "hits": hits,
                    "support_rate": rate,
                    "background_n": background_n,
                    "background_rate": background,
                    "fold_enrichment": rate / background if background else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_cell_lines(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    rows: list[dict[str, object]] = []
    rank = frame[RANK_COL].to_numpy(dtype=np.int64)
    for eid, name in CELL_LINES__09.items():
        values = frame[ACTIVE_COLS[eid]].astype(bool).to_numpy()
        background = float(values.mean())
        for rank_bin, k in make_rank_bins(n).items():
            selected = rank <= k
            selected_n = int(selected.sum())
            hits = int(values[selected].sum())
            rate = hits / float(selected_n)
            rows.append(
                {
                    "cell_line_id": eid,
                    "cell_line": name,
                    "rank_bin": rank_bin,
                    "k": k,
                    "n": n,
                    "hits": hits,
                    "support_rate": rate,
                    "background_n": int(values.sum()),
                    "background_rate": background,
                    "fold_enrichment": rate / background if background else np.nan,
                }
            )
    return pd.DataFrame(rows)


def add_pattern(frame: pd.DataFrame) -> pd.Series:
    bits = frame[[ACTIVE_COLS[eid] for eid in PATTERN_ORDER]].astype(int)
    return bits.astype(str).agg("".join, axis=1)


def summarize_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    patterns = add_pattern(frame)
    top1_k = make_rank_bins(len(frame))[TOP1_BIN]
    top1 = frame[RANK_COL] <= top1_k
    top1_patterns = patterns[top1]
    rows: list[dict[str, object]] = []
    for pattern in sorted(patterns.unique()):
        bits = [int(x) for x in pattern]
        supported = [eid for eid, bit in zip(PATTERN_ORDER, bits) if bit]
        full_n = int((patterns == pattern).sum())
        top1_n = int((top1_patterns == pattern).sum())
        rows.append(
            {
                "support_pattern": pattern,
                "supported_cell_lines": ";".join(supported) if supported else "none",
                "breadth": int(sum(bits)),
                "genomewide_n": full_n,
                "genomewide_fraction": full_n / float(len(frame)),
                "top1_n": top1_n,
                "top1_fraction": top1_n / float(top1.sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["top1_n", "genomewide_n", "support_pattern"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    scopes = list(scope_masks(frame).items())
    rows: list[dict[str, object]] = []
    for scope, scope_mask in scopes:
        for eid_a, eid_b in combinations(PATTERN_ORDER, 2):
            a = frame[ACTIVE_COLS[eid_a]].astype(bool).to_numpy() & scope_mask
            b = frame[ACTIVE_COLS[eid_b]].astype(bool).to_numpy() & scope_mask
            both = int((a & b).sum())
            union = int((a | b).sum())
            n_a, n_b = int(a.sum()), int(b.sum())
            rows.append(
                {
                    "scope": scope,
                    "cell_line_a": eid_a,
                    "cell_line_b": eid_b,
                    "a_n": n_a,
                    "b_n": n_b,
                    "both_n": both,
                    "union_n": union,
                    "jaccard": both / float(union) if union else np.nan,
                    "overlap_coefficient": both / float(min(n_a, n_b)) if min(n_a, n_b) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, mask in scope_masks(frame).items():
        sub = frame.loc[mask]
        for breadth in range(len(CELL_LINES__09) + 1):
            part = sub[sub["_support_breadth"] == breadth]
            rows.append(
                {
                    "scope": scope,
                    "breadth": breadth,
                    "n": int(len(part)),
                    "fraction": len(part) / float(len(sub)) if len(sub) else np.nan,
                    "median_rank": float(part[RANK_COL].median()) if len(part) else np.nan,
                    "mean_tumor_support_score": float(part["tumor_epigenomic_support_score"].mean()) if len(part) else np.nan,
                    "mean_reference_active_prevalence": float(part["reference_active_histone_prevalence_mean"].mean()) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def read_tumor_nontumor() -> pd.DataFrame:
    return pd.read_csv(M07_SUMMARY, sep="\t")


def step_09() -> None:
    OUT_DIR__09.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(SOURCE_TABLE__09)
    representatives = pd.read_csv(M28_REPRESENTATIVES__09, sep="\t", low_memory=False)
    frame = frame.merge(
        representatives[[ID_COL__09, "reference_active_histone_prevalence_mean"]],
        on=ID_COL__09,
        how="left",
        validate="one_to_one",
    )
    frame[RANK_COL] = pd.to_numeric(frame[RANK_COL], errors="raise").astype(np.int64)
    frame["_support_breadth"] = (
        frame[list(ACTIVE_COLS.values())].astype(int).sum(axis=1).astype(np.int8)
    )


# --------------------------------------------------------------------------
# step 11: Multivariable tumor generalization
# --------------------------------------------------------------------------



PROJECT_ROOT__11 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__11 = (PROJECT_ROOT__11 / "Biological_Interpretation").resolve()
Modeling_OUTPUT__11 = (PROJECT_ROOT__11 / "Modeling/output").resolve()

SOURCE_TABLE__11 = BIOLOGY_ROOT__11 / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M23_MANIFEST__11 = Modeling_OUTPUT__11 / '23_generate_genomewide_herv_functional_ranking/23_genomewide_ranking_manifest.json'
M24_MANIFEST__11 = Modeling_OUTPUT__11 / '24_project_frozen_herv_ranking_to_tumor_epigenomes/24_tumor_projection_manifest.json'

OUT_DIR__11 = BIOLOGY_ROOT__11 / "output/11_multivariable_tumor_generalization"
OUT_PRIMARY = OUT_DIR__11 / "11_frozen_rank_primary_effects.tsv"


OUT_CONTRACT = OUT_DIR__11 / "11_covariate_contract.tsv"


SCORE_COL = "herv_functional_priority_score"
FDR_ALPHA = 0.05
PRIMARY_TERM = "frozen_priority_per_10pct"

CELL_LINES__11 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}
ACTIVE_COLS__11 = {
    eid: f"tumor_{eid}_any_active_support" for eid in CELL_LINES__11
}
REFERENCE_ACTIVE_COL = "gctx_roadmap_active_mark_presence_count"
REFERENCE_REPRESSIVE_COL = "gctx_roadmap_repressive_mark_presence_count"
CONTINUOUS_TERMS = [
    PRIMARY_TERM,
    "gc_content_z",
    "log1p_length_z",
    "repeat_divergence_z",
    "log1p_abs_tss_distance_z",
    "reference_active_mark_count_z",
    "reference_repressive_mark_count_z",
]










def load_manifests() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(M23_MANIFEST__11.read_text(encoding="utf-8")),
        json.loads(M24_MANIFEST__11.read_text(encoding="utf-8")),
    )


def standardize(values: pd.Series) -> tuple[np.ndarray, float, float]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    return ((numeric.to_numpy() - mean) / sd), mean, sd


def prepare_design(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, np.ndarray]]:
    work = frame.copy()
    priority = pd.to_numeric(work[SCORE_COL], errors="raise")
    priority_mean = float(priority.mean())
    transformed: dict[str, np.ndarray] = {
        PRIMARY_TERM: (priority.to_numpy(dtype=float) - priority_mean) * 10.0,
    }
    contract_rows: list[dict[str, object]] = [
        {
            "term": PRIMARY_TERM,
            "source_column": SCORE_COL,
            "transformation": "mean-center then x10; OR is per 0.10 higher frozen priority percentile",
            "mean_before_standardization": priority_mean,
            "sd_before_standardization": float(priority.std(ddof=0)),
        }
    ]
    transformations = (
        ("gc_content_z", "gc_content", work["gc_content"]),
        ("log1p_length_z", "length", np.log1p(work["length"])),
        (
            "repeat_divergence_z",
            "repeat_divergence",
            work["repeat_divergence"],
        ),
        (
            "log1p_abs_tss_distance_z",
            "dist_to_tss",
            np.log1p(work["dist_to_tss"].abs()),
        ),
        (
            "reference_active_mark_count_z",
            REFERENCE_ACTIVE_COL,
            work[REFERENCE_ACTIVE_COL],
        ),
        (
            "reference_repressive_mark_count_z",
            REFERENCE_REPRESSIVE_COL,
            work[REFERENCE_REPRESSIVE_COL],
        ),
    )
    for term, source, values in transformations:
        z_values, mean, sd = standardize(pd.Series(values))
        transformed[term] = z_values
        transform = "z-score"
        if term.startswith("log1p"):
            transform = "log1p then z-score"
        contract_rows.append(
            {
                "term": term,
                "source_column": source,
                "transformation": transform,
                "mean_before_standardization": mean,
                "sd_before_standardization": sd,
            }
        )

    families = work["repFamily"].fillna("Unknown").astype(str)
    reference_family = families.value_counts().index[0]
    family_levels = sorted(set(families) - {reference_family})
    family_terms = [f"family[{level}]" for level in family_levels]
    for level, term in zip(family_levels, family_terms):
        transformed[term] = families.eq(level).to_numpy(dtype=float)
        contract_rows.append(
            {
                "term": term,
                "source_column": "repFamily",
                "transformation": f"indicator; reference={reference_family}",
                "mean_before_standardization": float(families.eq(level).mean()),
                "sd_before_standardization": np.nan,
            }
        )

    terms = ["Intercept", *CONTINUOUS_TERMS, *family_terms]
    columns = [np.ones(len(work), dtype=float)] + [
        transformed[term] for term in terms[1:]
    ]
    design = np.column_stack(columns)
    contract = pd.DataFrame(contract_rows)
    outcomes = {
        "PAN_TUMOR": pd.to_numeric(
            work["tumor_active_support_breadth"], errors="coerce"
        ).gt(0).to_numpy(dtype=np.int8)
    }
    outcomes.update(
        {
            eid: work[column].astype(int).to_numpy(dtype=np.int8)
            for eid, column in ACTIVE_COLS__11.items()
        }
    )
    return design, terms, contract, outcomes


def fit_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    terms: list[str],
    context_id: str,
    context_name: str,
    specification: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    event_rate = float(outcome.mean())
    start = np.zeros(design.shape[1], dtype=float)
    start[0] = np.log(event_rate / (1.0 - event_rate))

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.logaddexp(0.0, eta).sum() - outcome @ eta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - outcome)

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(result.message)
    beta = np.asarray(result.x, dtype=float)
    probability = expit(design @ beta)
    weights = np.clip(probability * (1.0 - probability), 1e-12, None)
    fisher = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(fisher, hermitian=True)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    log_p_value = np.log(2.0) + norm.logsf(np.abs(z_value))
    p_value = np.exp(log_p_value)
    coefficient_table = pd.DataFrame(
        {
            "context_id": context_id,
            "context_name": context_name,
            "model_specification": specification,
            "term": terms,
            "coefficient_log_odds": beta,
            "standard_error": se,
            "z_value": z_value,
            "p_value": p_value,
            "neg_log10_p_value": -log_p_value / np.log(10.0),
            "odds_ratio": np.exp(beta),
            "ci95_low": np.exp(beta - 1.96 * se),
            "ci95_high": np.exp(beta + 1.96 * se),
        }
    )
    log_likelihood = -objective(beta)
    null_probability = np.full(len(outcome), event_rate)
    null_log_likelihood = float(
        (outcome * np.log(null_probability) + (1 - outcome) * np.log1p(-null_probability)).sum()
    )
    diagnostics: dict[str, object] = {
        "context_id": context_id,
        "context_name": context_name,
        "model_specification": specification,
        "n": int(len(outcome)),
        "events": int(outcome.sum()),
        "event_rate": event_rate,
        "parameter_n": int(len(beta)),
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "log_likelihood": float(log_likelihood),
        "aic": float(-2.0 * log_likelihood + 2.0 * len(beta)),
        "bic": float(-2.0 * log_likelihood + np.log(len(outcome)) * len(beta)),
        "mcfadden_pseudo_r2": float(1.0 - log_likelihood / null_log_likelihood),
        "auc": float(roc_auc_score(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "fisher_condition_number": float(np.linalg.cond(fisher)),
    }
    return coefficient_table, diagnostics


def bh_fdr_from_neg_log10(
    neg_log10_p_values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(neg_log10_p_values, errors="raise").to_numpy(dtype=float)
    log_p = -values * np.log(10.0)
    order = np.argsort(log_p)
    n = len(log_p)
    log_corrected = log_p[order] + np.log(n / np.arange(1, n + 1))
    log_corrected = np.minimum.accumulate(log_corrected[::-1])[::-1]
    log_corrected = np.minimum(log_corrected, 0.0)
    restored = np.empty(n, dtype=float)
    restored[order] = log_corrected
    return np.exp(restored), -restored / np.log(10.0)


def step_11() -> None:
    OUT_DIR__11.mkdir(parents=True, exist_ok=True)
    m23, m24 = load_manifests()
    frame = pd.read_parquet(
        SOURCE_TABLE__11,
        columns=[
            SCORE_COL,
            "repFamily",
            "gc_content",
            "length",
            "repeat_divergence",
            "dist_to_tss",
            REFERENCE_ACTIVE_COL,
            REFERENCE_REPRESSIVE_COL,
            "tumor_active_support_breadth",
            *ACTIVE_COLS__11.values(),
        ],
    )
    for column in ACTIVE_COLS__11.values():
        frame[column] = frame[column].astype(int)

    design, terms, contract, outcomes = prepare_design(frame)
    coefficients: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    contexts = {"PAN_TUMOR": f"Any of {len(CELL_LINES__11)} tumor cell lines", **CELL_LINES__11}
    primary_index = terms.index(PRIMARY_TERM)
    for context_id, context_name in contexts.items():
        outcome = outcomes[context_id]
        unadjusted_design = design[:, [0, primary_index]]
        unadjusted_terms = ["Intercept", PRIMARY_TERM]
        for specification, current_design, current_terms in (
            ("unadjusted", unadjusted_design, unadjusted_terms),
            ("adjusted", design, terms),
        ):
            table, diagnostic = fit_logistic(
                current_design,
                outcome,
                current_terms,
                context_id,
                context_name,
                specification,
            )
            coefficients.append(table)
            diagnostic_rows.append(diagnostic)

    all_coefficients = pd.concat(coefficients, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    primary = all_coefficients[all_coefficients["term"].eq(PRIMARY_TERM)].copy()
    event_lookup = diagnostics.set_index(["context_id", "model_specification"])[
        ["events", "event_rate"]
    ]
    primary["events"] = [
        event_lookup.loc[(row.context_id, row.model_specification), "events"]
        for row in primary.itertuples(index=False)
    ]
    primary["event_rate"] = [
        event_lookup.loc[(row.context_id, row.model_specification), "event_rate"]
        for row in primary.itertuples(index=False)
    ]
    primary["bh_fdr"] = np.nan
    primary["neg_log10_bh_fdr"] = np.nan
    adjusted_mask = primary["model_specification"].eq("adjusted")
    adjusted_fdr, adjusted_neg_log10_fdr = bh_fdr_from_neg_log10(
        primary.loc[adjusted_mask, "neg_log10_p_value"]
    )
    primary.loc[adjusted_mask, "bh_fdr"] = adjusted_fdr
    primary.loc[adjusted_mask, "neg_log10_bh_fdr"] = adjusted_neg_log10_fdr


    atomic_write_tsv(primary, OUT_PRIMARY)
    atomic_write_tsv(contract, OUT_CONTRACT)


# --------------------------------------------------------------------------
# step 17: Compare the nested tumor generalization models
# --------------------------------------------------------------------------



BASE = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE / "Biological_Interpretation/output/01_build_biological_source_table/01_biological_source_table.parquet"
TUMOR_FILE = BASE / "Modeling/output/24_project_frozen_herv_ranking_to_tumor_epigenomes/24_tumor_projection_all_herv.parquet"
CONTRACT_FILE = BASE / "Biological_Interpretation/output/11_multivariable_tumor_generalization/11_covariate_contract.tsv"

OUT = BASE / "Biological_Interpretation/output/17_nested_tumor_generalization"
OUT_CV = OUT / "17_cv_summary.tsv"
OUT_COHORT = OUT / "17_label_cohort_balance.tsv"
OUT_SUMMARY = OUT / "17_summary.json"

ID = "herv_internal_id"
BREADTH = "tumor_active_support_breadth"
OVERALL_LABEL = "tumor_overall_active_support"
LABEL_SUFFIX = "_any_active_support"
LABEL_PREFIX = "tumor_"

RANDOM_STATE = 99999999
N_SPLITS = 5
N_BOOT = 1000
MAX_ITER = 3000










def contract_features(contract: pd.DataFrame) -> dict[str, Any]:
    transform = contract["transformation"].fillna("")
    is_indicator = transform.str.contains("indicator")
    is_standardized = transform.str.contains("z-score")
    ordered = lambda mask: list(dict.fromkeys(contract.loc[mask, "source_column"]))
    priority_row = contract.loc[~is_indicator & ~is_standardized].iloc[0]
    log_rows = contract[transform.str.contains("log1p")]
    log_map = {str(r.source_column): ("log1p_abs" if "abs" in str(r.term) else "log1p")
               for r in log_rows.itertuples()}
    return {"numeric": ordered(is_standardized & ~is_indicator),
            "categorical": ordered(is_indicator),
            "priority_term": str(priority_row["term"]),
            "priority_source": str(priority_row["source_column"]),
            "log_map": log_map,
            "contract": contract}


def _log1p(x: Any) -> Any:
    return np.log1p(np.asarray(x, dtype=float))


def _log1p_abs(x: Any) -> Any:
    return np.log1p(np.abs(np.asarray(x, dtype=float)))


def build_pipeline(numeric: list[str], categorical: list[str], weighted: bool = True,
                   log_map: dict[str, str] | None = None) -> Pipeline:
    log_map = log_map or {}
    log_cols = [c for c in numeric if log_map.get(c) == "log1p"]
    logabs_cols = [c for c in numeric if log_map.get(c) == "log1p_abs"]
    plain_cols = [c for c in numeric if c not in log_map]
    transformers = []
    if plain_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                              ("scale", StandardScaler())]), plain_cols))
    for tag, cols, fn in (("num_log1p", log_cols, _log1p),
                          ("num_log1p_abs", logabs_cols, _log1p_abs)):
        if cols:
            transformers.append((tag, Pipeline([("log", FunctionTransformer(fn, validate=False)),
                                                ("imputer", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler())]), cols))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                             categorical))
    args: dict[str, Any] = {"max_iter": MAX_ITER}
    if weighted:
        args["class_weight"] = "balanced"
    return Pipeline([("pre", ColumnTransformer(transformers)), ("clf", LogisticRegression(**args))])


def lr_pipeline(X: pd.DataFrame, log_map: dict[str, str] | None = None) -> Pipeline:
    log_map = log_map or {}
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    spec = []
    for tag, cols, fn in (("num_log1p", [c for c in numeric_cols if log_map.get(c) == "log1p"], _log1p),
                          ("num_log1p_abs", [c for c in numeric_cols if log_map.get(c) == "log1p_abs"], _log1p_abs)):
        if cols:
            spec.append((tag, Pipeline([("log", FunctionTransformer(fn, validate=False)),
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler())]), cols))
    plain = [c for c in numeric_cols if c not in log_map]
    if plain:
        spec.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), plain))
    spec.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                 X.select_dtypes(exclude=np.number).columns.tolist()))
    return Pipeline([("pre", ColumnTransformer(spec)),
                     ("clf", LogisticRegression(penalty=None, solver="lbfgs", max_iter=MAX_ITER))])


def oof_prediction(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def bootstrap_delta(y: pd.Series, p0: np.ndarray, p1: np.ndarray, metric) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    deltas, values = [], y.to_numpy()
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(values), len(values))
        if len(np.unique(values[idx])) < 2:
            continue
        deltas.append(metric(values[idx], p1[idx]) - metric(values[idx], p0[idx]))
    return {"mean": float(np.mean(deltas)), "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)), "n_bootstrap": len(deltas)}


def likelihood_ratio_test(X0: pd.DataFrame, X1: pd.DataFrame, y: pd.Series,
                         log_map: dict[str, str] | None = None) -> dict[str, Any]:
    m0, m1 = lr_pipeline(X0, log_map), lr_pipeline(X1, log_map)
    m0.fit(X0, y)
    m1.fit(X1, y)
    p0 = np.clip(m0.predict_proba(X0)[:, 1], 1e-12, 1 - 1e-12)
    p1 = np.clip(m1.predict_proba(X1)[:, 1], 1e-12, 1 - 1e-12)
    ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    ll1 = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
    statistic = 2 * (ll1 - ll0)
    return {"lr_chi2": statistic, "lr_df": 1, "lr_p_value": float(chi2.sf(statistic, 1))}


def build_balanced_cohort(df: pd.DataFrame, label: str, seed: int) -> pd.DataFrame:
    y = df[label].astype(int)
    negative = df[y == 0]
    n = min(int(y.sum()), len(negative))
    return (pd.concat([df[y == 1], negative.sample(n=n, random_state=seed)])
            .sample(frac=1, random_state=seed).reset_index(drop=True))


def step_17() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(SOURCE_FILE)
    tumor = pd.read_parquet(TUMOR_FILE)
    contract = pd.read_csv(CONTRACT_FILE, sep="\t")

    tumor_only = [c for c in tumor.columns if c not in set(source.columns)]
    df = (source.merge(tumor[[ID] + tumor_only], on=ID, how="inner") if tumor_only else source).copy()
    df[ID] = df[ID].astype(str)

    features = contract_features(contract)
    numeric, categorical = features["numeric"], features["categorical"]
    df[OVERALL_LABEL] = (df[BREADTH] > 0).astype(int)
    labels = [OVERALL_LABEL] + sorted(c for c in df.columns
                                      if c.startswith(LABEL_PREFIX) and c.endswith(LABEL_SUFFIX))

    center = df[features["priority_source"]].mean()
    results, cohorts = [], []
    for label in labels:
        cohort = build_balanced_cohort(df, label, RANDOM_STATE)
        y = cohort[label].astype(int)
        X0 = cohort[numeric + categorical]
        X1 = X0.copy()
        X1[features["priority_term"]] = (cohort[features["priority_source"]] - center) * 10

        p0 = oof_prediction(build_pipeline(numeric, categorical, log_map=features["log_map"]), X0, y)
        p1 = oof_prediction(build_pipeline(numeric + [features["priority_term"]], categorical,
                                         log_map=features["log_map"]), X1, y)
        auc0, auc1 = roc_auc_score(y, p0), roc_auc_score(y, p1)
        ap0, ap1 = average_precision_score(y, p0), average_precision_score(y, p1)

        row = {"label": label, "n_loci": len(cohort), "n_positive": int(y.sum()),
               "n_negative": int((y == 0).sum()),
               "M0_auc": auc0, "M1_auc": auc1, "delta_auc": auc1 - auc0,
               "M0_auprc": ap0, "M1_auprc": ap1, "delta_auprc": ap1 - ap0}
        for key, ci in (("auc", bootstrap_delta(y, p0, p1, roc_auc_score)),
                        ("auprc", bootstrap_delta(y, p0, p1, average_precision_score))):
            row[f"delta_{key}_ci_low"], row[f"delta_{key}_ci_high"] = ci["ci_low"], ci["ci_high"]
            row[f"delta_{key}_boot_mean"], row["n_bootstrap"] = ci["mean"], ci["n_bootstrap"]
        row.update(likelihood_ratio_test(X0, X1, y, features["log_map"]))
        results.append(row)
        cohorts.append({"label": label, "n_positive_full": int(df[label].sum()),
                        "n_negative_full": int((df[label] == 0).sum()),
                        "n_cohort": len(cohort), "class_ratio": len(cohort) / (2 * y.sum())})

    result, cohort_table = pd.DataFrame(results), pd.DataFrame(cohorts)
    write_tsv(result, OUT_CV)
    write_tsv(cohort_table, OUT_COHORT)

    context = {"generated_at": now(), "n_loci": int(len(df)), "n_labels": len(labels),
               "m0_covariates": {"numeric": numeric, "categorical": categorical},
               "priority_term": features["priority_term"], "priority_source": features["priority_source"],
               "priority_center_used": float(center), "random_state": RANDOM_STATE,
               "n_splits": N_SPLITS, "n_bootstrap": N_BOOT,
               "m24_columns_added_by_merge": tumor_only,
               "duplicated_ids_source": int(source[ID].duplicated().sum()),
               "duplicated_ids_tumor": int(tumor[ID].duplicated().sum()),
               "labels": labels, "results": results}
    write_json(context, OUT_SUMMARY)


def main() -> None:
    step_04()
    step_05()
    step_06()
    step_07()
    step_08()
    step_09()
    step_11()
    step_17()


if __name__ == "__main__":
    main()
