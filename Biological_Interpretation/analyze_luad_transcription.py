#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import json
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    mannwhitneyu,
    chi2,
)

# --------------------------------------------------------------------------
# step 18: Analyze LUAD transcriptional realization
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = PROJECT_ROOT / "Biological_Interpretation"
DATA_ROOT = config.LUAD_TRANSCRIPTOMICS

SOURCE_FILE = BIOLOGY_ROOT / "output/01_build_biological_source_table/01_biological_source_table.tsv.gz"
COUNT_FILE = DATA_ROOT / "LUAD_HERV_locus_counts_merged.tsv"
SAMPLE_FILE = DATA_ROOT / "LUAD_sample_metadata.tsv"

OUT_DIR = BIOLOGY_ROOT / "output/18_luad_transcriptional_realization"

MIN_RECIPROCAL_OVERLAP = 0.50
PRIMARY_RAW_COUNT_THRESHOLD = 1
PRIMARY_PREVALENCE_THRESHOLD = 0.10
SENSITIVITY_RAW_COUNT_THRESHOLDS = (1, 2, 5)
SENSITIVITY_PREVALENCE_THRESHOLDS = (0.05, 0.10, 0.20)
PRIORITY_DECILES = 10

OUTPUTS = {
    "input_consistency": OUT_DIR / "18_input_consistency.tsv",
    "mapping_audit": OUT_DIR / "18_mapping_audit.tsv",
    "mapping_selection": OUT_DIR / "18_mapping_selection_audit.tsv",
    "mapping_family": OUT_DIR / "18_mapping_selection_family.tsv",
    "size_factors": OUT_DIR / "18_sample_size_factors.tsv",
    "loci": OUT_DIR / "18_luad_locus_transcriptional_realization.tsv.gz",
    "group_summary": OUT_DIR / "18_expression_group_summary.tsv",
    "binary_effects": OUT_DIR / "18_binary_outcome_effects.tsv",
    "continuous_effects": OUT_DIR / "18_continuous_outcome_effects.tsv",
    "adjusted_coefficients": OUT_DIR / "18_adjusted_logistic_coefficients.tsv",
    "sensitivity": OUT_DIR / "18_expression_threshold_sensitivity.tsv",
    "deciles": OUT_DIR / "18_priority_decile_expression.tsv",
    "summary": OUT_DIR / "18_summary.json",
}


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "t", "1"])


def exact_binomial_ci(success: int, total: int) -> tuple[float, float]:
    # Exact Clopper-Pearson intervals without an extra statsmodels dependency.
    from scipy.stats import beta

    if total == 0:
        return (float("nan"), float("nan"))
    alpha = 0.05
    low = 0.0 if success == 0 else float(beta.ppf(alpha / 2, success, total - success + 1))
    high = 1.0 if success == total else float(beta.ppf(1 - alpha / 2, success + 1, total - success))
    return low, high


def bh_adjust(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return result
    p = values.loc[valid].to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[valid] = adjusted[np.argsort(order)]
    return result


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_columns = [
        "herv_internal_id", "chrom", "start", "end", "strand", "repName", "repFamily",
        "length", "gc_content", "repeat_divergence", "herv_functional_priority_rank",
        "herv_functional_priority_score", "m23_top1pct", "tumor_E114_any_active_support",
        "tumor_E114_active_mark_count", "tumor_E114_active_total_overlap_bp",
    ]
    source = pd.read_csv(SOURCE_FILE, sep="\t", compression="gzip", usecols=source_columns, low_memory=False)

    telescope = pd.read_csv(COUNT_FILE, sep="\t", low_memory=False)
    samples = pd.read_csv(SAMPLE_FILE, sep="\t", low_memory=False)
    return source, telescope, samples


def validate_inputs(source: pd.DataFrame, telescope: pd.DataFrame, samples: pd.DataFrame) -> dict:
    sample_ids = samples["sample_id"].astype(str).tolist()
    counts = telescope[sample_ids].apply(pd.to_numeric, errors="coerce")
    source["m23_top1pct"] = as_bool(source["m23_top1pct"])
    source["a549_active_support"] = as_bool(source["tumor_E114_any_active_support"])
    for column in ["de_tested", "de_significant", "tumor_upregulated", "tumor_downregulated"]:
        telescope[column] = as_bool(telescope[column])
    for column in ["is_primary_tumor", "is_recurrent_tumor", "is_adjacent_normal",
                   "is_paired_tumor_normal_patient"]:
        samples[column] = as_bool(samples[column])
    return {
        "source_loci": len(source),
        "source_unique_ids": int(source["herv_internal_id"].nunique()),
        "source_top1_loci": int(source["m23_top1pct"].sum()),
        "source_unique_priority_ranks": int(source["herv_functional_priority_rank"].nunique()),
        "source_invalid_strand_n": int((~source["strand"].isin(["+", "-"])).sum()),
        "source_invalid_coordinate_n": int(((source["start"] < 0) | (source["end"] <= source["start"])).sum()),
        "telescope_loci": len(telescope),
        "telescope_unique_ids": int(telescope["transcript"].nunique()),
        "telescope_invalid_strand_n": int((~telescope["strand"].isin(["+", "-"])).sum()),
        "telescope_invalid_coordinate_n": int(((telescope["start"] < 1) | (telescope["end"] < telescope["start"])).sum()),
        "luad_samples": len(samples),
        "luad_unique_sample_ids": int(samples["sample_id"].nunique()),
        "count_columns_missing_from_count_table": sorted(set(sample_ids) - set(telescope.columns)),
        "count_non_numeric_or_negative_cells": int(((counts < 0) | counts.isna()).to_numpy().sum()),
        "count_non_integer_cells": int((~np.equal(counts.to_numpy(dtype=float),
                                                   np.floor(counts.to_numpy(dtype=float)))).sum()),
        "samples_with_zero_library_size": int((counts.sum(axis=0) <= 0).sum()),
    }


def strict_overlap_candidates(source: pd.DataFrame, telescope: pd.DataFrame) -> pd.DataFrame:
    source_groups = {}
    for index, row in source[["chrom", "strand", "start", "end"]].iterrows():
        source_groups.setdefault((row.chrom, row.strand), []).append((int(row.start), int(row.end), int(index)))
    telescope_groups = {}
    for index, row in telescope[["chrom", "strand", "start", "end"]].iterrows():
        telescope_groups.setdefault((row.chrom, row.strand), []).append((int(row.start) - 1, int(row.end), int(index)))

    rows = []
    for key, target_intervals in telescope_groups.items():
        source_intervals = sorted(source_groups.get(key, []), key=lambda value: (value[0], value[1], value[2]))
        target_intervals = sorted(target_intervals, key=lambda value: (value[0], value[1], value[2]))
        active: dict[int, tuple[int, int, int]] = {}
        add_index = 0
        for target_start, target_end, target_index in target_intervals:
            while add_index < len(source_intervals) and source_intervals[add_index][0] < target_end:
                interval = source_intervals[add_index]
                if interval[1] > target_start:
                    active[interval[2]] = interval
                add_index += 1
            active = {idx: interval for idx, interval in active.items() if interval[1] > target_start}
            for source_start, source_end, source_index in active.values():
                overlap_start = max(source_start, target_start)
                overlap_end = min(source_end, target_end)
                overlap_bp = overlap_end - overlap_start
                if overlap_bp <= 0:
                    continue
                source_length = source_end - source_start
                telescope_length = target_end - target_start
                source_fraction = overlap_bp / source_length
                telescope_fraction = overlap_bp / telescope_length
                if source_fraction >= MIN_RECIPROCAL_OVERLAP and telescope_fraction >= MIN_RECIPROCAL_OVERLAP:
                    rows.append(
                        {
                            "mef_index": source_index,
                            "telescope_index": target_index,
                            "mef_id": source.iloc[source_index]["herv_internal_id"],
                            "telescope_id": telescope.iloc[target_index]["transcript"],
                            "overlap_bp": overlap_bp,
                            "mef_length": source_length,
                            "telescope_length": telescope_length,
                            "mef_overlap_fraction": source_fraction,
                            "telescope_overlap_fraction": telescope_fraction,
                            "min_reciprocal_fraction": min(source_fraction, telescope_fraction),
                            "mean_reciprocal_fraction": (source_fraction + telescope_fraction) / 2,
                        }
                    )
    return pd.DataFrame(rows)


def choose_mutual_best(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.sort_values(
        ["mef_id", "min_reciprocal_fraction", "overlap_bp", "mean_reciprocal_fraction", "telescope_id"],
        ascending=[True, False, False, False, True],
    ).copy()
    ranked["mef_choice_rank"] = ranked.groupby("mef_id", sort=False).cumcount() + 1
    ranked = ranked.sort_values(
        ["telescope_id", "min_reciprocal_fraction", "overlap_bp", "mean_reciprocal_fraction", "mef_id"],
        ascending=[True, False, False, False, True],
    ).copy()
    ranked["telescope_choice_rank"] = ranked.groupby("telescope_id", sort=False).cumcount() + 1
    ranked["retained_mutual_best"] = (ranked["mef_choice_rank"] == 1) & (ranked["telescope_choice_rank"] == 1)
    return ranked.loc[ranked["retained_mutual_best"]].copy()


def build_mapping_audit(source: pd.DataFrame, telescope: pd.DataFrame, candidates: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped_source = source.iloc[mapping["mef_index"].to_numpy()]
    rows = {
        "herv_mef_total_loci": len(source),
        "herv_mef_top1_loci": int(source["m23_top1pct"].sum()),
        "telescope_total_loci": len(telescope),
        "same_strand_overlap_pairs": len(candidates),
        "reciprocal_ge_50pct_pairs": len(candidates),
        "reciprocal_ge_50pct_mef_loci": candidates["mef_id"].nunique(),
        "reciprocal_ge_50pct_telescope_loci": candidates["telescope_id"].nunique(),
        "mutual_best_one_to_one_pairs": len(mapping),
        "mapped_herv_mef_top1_loci": int(mapped_source["m23_top1pct"].sum()),
        "mapped_herv_mef_background_loci": int((~mapped_source["m23_top1pct"]).sum()),
        "mapped_a549_supported_loci": int(mapped_source["a549_active_support"].sum()),
        "mapped_top1_a549_supported_loci": int((mapped_source["m23_top1pct"] & mapped_source["a549_active_support"]).sum()),
    }
    return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})


def mapping_selection_audit(
    source: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = source.copy()
    audited["mapped_to_telescope"] = audited["herv_internal_id"].isin(mapping["mef_id"])
    audited["priority_percentile"] = 1 - (
        audited["herv_functional_priority_rank"] - 1
    ) / (len(source) - 1)

    continuous_rows = []
    family_rows = []
    features = ["length", "gc_content", "repeat_divergence", "priority_percentile"]
    scopes = {"all_loci": audited, "top1_only": audited[audited["m23_top1pct"]]}

    for scope, frame in scopes.items():
        mapped = frame[frame["mapped_to_telescope"]]
        unmapped = frame[~frame["mapped_to_telescope"]]
        for feature in features:
            left = pd.to_numeric(mapped[feature], errors="coerce").dropna().to_numpy()
            right = pd.to_numeric(unmapped[feature], errors="coerce").dropna().to_numpy()
            statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")
            rank_biserial = 2 * statistic / (len(left) * len(right)) - 1
            continuous_rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "test": "Mann-Whitney U",
                    "mapped_n": len(left),
                    "mapped_median": np.median(left),
                    "mapped_iqr": np.quantile(left, 0.75) - np.quantile(left, 0.25),
                    "unmapped_n": len(right),
                    "unmapped_median": np.median(right),
                    "unmapped_iqr": np.quantile(right, 0.75) - np.quantile(right, 0.25),
                    "rank_biserial": rank_biserial,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

        family_table = pd.crosstab(frame["repFamily"], frame["mapped_to_telescope"])
        family_table = family_table.reindex(columns=[False, True], fill_value=0)
        chi2, p_value, _, _ = chi2_contingency(family_table.to_numpy())
        continuous_rows.append(
            {
                "scope": scope,
                "feature": "repFamily",
                "test": "chi-square",
                "mapped_n": int(frame["mapped_to_telescope"].sum()),
                "mapped_median": np.nan,
                "mapped_iqr": np.nan,
                "unmapped_n": int((~frame["mapped_to_telescope"]).sum()),
                "unmapped_median": np.nan,
                "unmapped_iqr": np.nan,
                "rank_biserial": math.sqrt(chi2 / len(frame)),
                "statistic": chi2,
                "p_value": p_value,
            }
        )
        family = (
            frame.groupby("repFamily", dropna=False)["mapped_to_telescope"]
            .agg(total_n="size", mapped_n="sum")
            .reset_index()
        )
        family["scope"] = scope
        family["mapping_fraction"] = family["mapped_n"] / family["total_n"]
        family_rows.append(family)

    audit = pd.DataFrame(continuous_rows)
    audit["p_adjust_bh"] = audit.groupby("scope")["p_value"].transform(bh_adjust)
    return audit, pd.concat(family_rows, ignore_index=True)


def estimate_sparse_size_factors(counts: np.ndarray) -> np.ndarray:
    positive = counts > 0
    detected_samples = positive.sum(axis=1)
    usable = detected_samples >= 2
    log_geomean = np.full(counts.shape[0], np.nan, dtype=float)
    log_counts = np.zeros_like(counts, dtype=float)
    np.log(counts, out=log_counts, where=positive)
    log_geomean[usable] = log_counts[usable].sum(axis=1) / detected_samples[usable]
    geomean = np.exp(log_geomean)

    size_factors = np.empty(counts.shape[1], dtype=float)
    for sample_index in range(counts.shape[1]):
        valid = usable & positive[:, sample_index]
        size_factors[sample_index] = np.median(
            counts[valid, sample_index] / geomean[valid]
        )
    size_factors /= np.exp(np.mean(np.log(size_factors)))
    return size_factors


def add_expression_outcomes(
    mapping: pd.DataFrame,
    source: pd.DataFrame,
    telescope: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = samples["sample_id"].astype(str).tolist()
    counts = telescope[sample_ids].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    size_factors = estimate_sparse_size_factors(counts)
    normalized_counts = counts / size_factors[None, :]
    tumor_columns = np.flatnonzero(samples["is_primary_tumor"].to_numpy())
    normal_columns = np.flatnonzero(samples["is_adjacent_normal"].to_numpy())
    telescope_indices = mapping["telescope_index"].to_numpy(dtype=int)
    mef_indices = mapping["mef_index"].to_numpy(dtype=int)
    tumor_counts = counts[telescope_indices][:, tumor_columns]
    normal_counts = counts[telescope_indices][:, normal_columns]
    tumor_normalized = normalized_counts[telescope_indices][:, tumor_columns]
    normal_normalized = normalized_counts[telescope_indices][:, normal_columns]
    mef = source.iloc[mef_indices].reset_index(drop=True)
    tel = telescope.iloc[telescope_indices].reset_index(drop=True)
    result = pd.DataFrame(
        {
            "herv_internal_id": mef["herv_internal_id"].astype(str),
            "mef_chrom": mef["chrom"], "mef_start_0based": mef["start"], "mef_end_halfopen": mef["end"], "mef_strand": mef["strand"],
            "mef_repName": mef["repName"], "mef_repFamily": mef["repFamily"], "mef_length": mef["length"], "mef_gc_content": mef["gc_content"], "mef_repeat_divergence": mef["repeat_divergence"],
            "herv_functional_priority_rank": mef["herv_functional_priority_rank"], "herv_functional_priority_score": mef["herv_functional_priority_score"], "frozen_top1": mef["m23_top1pct"], "a549_active_support": mef["a549_active_support"],
            "a549_active_mark_count": mef["tumor_E114_active_mark_count"], "a549_active_total_overlap_bp": mef["tumor_E114_active_total_overlap_bp"],
            "telescope_id": tel["transcript"].astype(str), "telescope_chrom": tel["chrom"], "telescope_start_1based": tel["start"], "telescope_end_inclusive": tel["end"], "telescope_strand": tel["strand"],
            "telescope_length": tel["end"] - tel["start"] + 1,
            "overlap_bp": mapping["overlap_bp"].to_numpy(), "mef_overlap_fraction": mapping["mef_overlap_fraction"].to_numpy(), "telescope_overlap_fraction": mapping["telescope_overlap_fraction"].to_numpy(), "min_reciprocal_fraction": mapping["min_reciprocal_fraction"].to_numpy(),
            "tumor_detection_prevalence_count_ge_1": (tumor_counts >= PRIMARY_RAW_COUNT_THRESHOLD).mean(axis=1),
            "adjacent_normal_detection_prevalence_count_ge_1": (normal_counts >= PRIMARY_RAW_COUNT_THRESHOLD).mean(axis=1),
            "mean_tumor_log2_normalized_count1": np.log2(tumor_normalized + 1).mean(axis=1),
            "mean_adjacent_normal_log2_normalized_count1": np.log2(normal_normalized + 1).mean(axis=1),
            "de_tested": tel["de_tested"], "de_significant": tel["de_significant"], "tumor_upregulated": tel["tumor_upregulated"], "tumor_downregulated": tel["tumor_downregulated"], "Avg_tumor": tel["Avg_tumor"], "SD_tumor": tel["SD_tumor"], "Avg_normal": tel["Avg_normal"], "SD_normal": tel["SD_normal"], "logFC": tel["logFC"], "P.Value": tel["P.Value"], "adj.P.Val": tel["adj.P.Val"],
        }
    )
    result["tumor_expressed"] = (
        result["tumor_detection_prevalence_count_ge_1"] >= PRIMARY_PREVALENCE_THRESHOLD
    )
    result["positive_logFC"] = np.where(result["de_tested"], result["logFC"] > 0, np.nan)
    result["realization_group"] = np.select([result["frozen_top1"] & result["a549_active_support"], result["frozen_top1"]], ["Top1_A549_positive", "Top1_A549_negative"], default="Background")
    n_deciles = PRIORITY_DECILES
    result["priority_decile"] = np.clip(
        n_deciles + 1 - np.ceil(result["herv_functional_priority_rank"]
                                 / (len(source) / n_deciles)).astype(int), 1, n_deciles)
    size_factor_table = samples[["sample_id", "sample_type_code"]].copy()
    size_factor_table["raw_herv_library_count"] = counts.sum(axis=0)
    size_factor_table["sparse_median_ratio_size_factor"] = size_factors
    return result, size_factor_table


def group_summary(loci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, frame in loci.groupby("realization_group", sort=False):
        event_n = int(frame["tumor_expressed"].sum())
        ci_low, ci_high = exact_binomial_ci(event_n, len(frame))
        tested = frame[frame["de_tested"]]
        up_n = int(tested["tumor_upregulated"].sum())
        up_low, up_high = exact_binomial_ci(up_n, len(tested))
        prevalence = frame["tumor_detection_prevalence_count_ge_1"]
        rows.append({"realization_group": group, "n_loci": len(frame), "tumor_expressed_n": event_n, "tumor_expressed_fraction": event_n / len(frame), "tumor_expressed_ci_low": ci_low, "tumor_expressed_ci_high": ci_high, "median_tumor_detection_prevalence": prevalence.median(), "iqr_tumor_detection_prevalence": prevalence.quantile(.75) - prevalence.quantile(.25), "median_mean_tumor_log2_normalized_count1": frame["mean_tumor_log2_normalized_count1"].median(), "de_tested_n": len(tested), "positive_logFC_n": int((tested["logFC"] > 0).sum()), "median_logFC": tested["logFC"].median(), "tumor_upregulated_n": up_n, "tumor_upregulated_fraction": up_n / len(tested) if len(tested) else np.nan, "tumor_upregulated_ci_low": up_low, "tumor_upregulated_ci_high": up_high})
    return pd.DataFrame(rows)


def comparison_masks(loci: pd.DataFrame, comparison: str) -> tuple[pd.Series, pd.Series]:
    if comparison == "Top1_vs_background":
        return loci["frozen_top1"], ~loci["frozen_top1"]
    if comparison == "A549_positive_vs_negative_within_Top1":
        return loci["frozen_top1"] & loci["a549_active_support"], loci["frozen_top1"] & ~loci["a549_active_support"]
    if comparison == "Top1_A549_positive_vs_background":
        return loci["frozen_top1"] & loci["a549_active_support"], ~loci["frozen_top1"]
    return loci["a549_active_support"], ~loci["a549_active_support"]


def binary_effect(loci: pd.DataFrame, outcome: str, comparison: str, eligible: pd.Series | None = None) -> dict:
    positive, reference = comparison_masks(loci, comparison)
    values = loci[outcome]
    eligible = pd.Series(True, index=loci.index) if eligible is None else eligible
    keep = eligible & values.notna() & (positive | reference)
    positive = positive & keep
    reference = reference & keep
    a = int(values[positive].astype(bool).sum()); b = int((~values[positive].astype(bool)).sum())
    c = int(values[reference].astype(bool).sum()); d = int((~values[reference].astype(bool)).sum())
    table = np.array([[a, b], [c, d]], dtype=int)
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    # Log-odds Wald interval is used as a dependency-light CI; exact P comes from Fisher.
    if a == 0 or b == 0 or c == 0 or d == 0:
        ci_low, ci_high = np.nan, np.nan
    else:
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci_low, ci_high = math.exp(math.log(odds_ratio) - 1.96 * se), math.exp(math.log(odds_ratio) + 1.96 * se)
    positive_rate = a / (a + b) if a + b else np.nan
    reference_rate = c / (c + d) if c + d else np.nan
    return {"outcome": outcome, "comparison": comparison, "positive_n": a + b, "positive_events": a, "positive_rate": positive_rate, "reference_n": c + d, "reference_events": c, "reference_rate": reference_rate, "risk_difference": positive_rate - reference_rate, "risk_ratio": positive_rate / reference_rate if reference_rate else np.nan, "odds_ratio": odds_ratio, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value}


def continuous_effect(loci: pd.DataFrame, outcome: str, comparison: str, eligible: pd.Series | None = None) -> dict:
    positive_mask, reference_mask = comparison_masks(loci, comparison)
    values = pd.to_numeric(loci[outcome], errors="coerce")
    eligible = pd.Series(True, index=loci.index) if eligible is None else eligible
    positive = values[positive_mask & eligible].dropna().to_numpy()
    reference = values[reference_mask & eligible].dropna().to_numpy()
    statistic, p_value = mannwhitneyu(positive, reference, alternative="two-sided")
    pooled = np.concatenate([positive, reference]); ranks = pd.Series(pooled).rank(method="average").to_numpy()
    u = ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2
    rank_biserial = 2 * u / (len(positive) * len(reference)) - 1
    return {"outcome": outcome, "comparison": comparison, "positive_n": len(positive), "positive_median": np.median(positive), "positive_iqr": np.quantile(positive, .75) - np.quantile(positive, .25), "reference_n": len(reference), "reference_median": np.median(reference), "reference_iqr": np.quantile(reference, .75) - np.quantile(reference, .25), "median_difference": np.median(positive) - np.median(reference), "rank_biserial": rank_biserial, "mannwhitney_u": statistic, "p_value": p_value}


def run_effects(loci: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = ["Top1_vs_background", "A549_positive_vs_negative_within_Top1", "Top1_A549_positive_vs_background", "A549_positive_vs_negative_all_mapped"]
    binary_rows, continuous_rows = [], []
    for comparison in comparisons:
        binary_rows.extend([
            binary_effect(loci, "tumor_expressed", comparison),
            binary_effect(loci, "positive_logFC", comparison, loci["de_tested"]),
            binary_effect(loci, "tumor_upregulated", comparison, loci["de_tested"]),
        ])
        continuous_rows.extend([
            continuous_effect(loci, "tumor_detection_prevalence_count_ge_1", comparison),
            continuous_effect(loci, "mean_tumor_log2_normalized_count1", comparison),
            continuous_effect(loci, "logFC", comparison, loci["de_tested"]),
        ])
    binary = pd.DataFrame(binary_rows); continuous = pd.DataFrame(continuous_rows)
    binary["p_adjust_bh"] = bh_adjust(binary["p_value"]); continuous["p_adjust_bh"] = bh_adjust(continuous["p_value"])
    return binary, continuous


def run_sensitivity(loci: pd.DataFrame, mapping: pd.DataFrame, telescope: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    sample_ids = samples["sample_id"].astype(str).tolist()
    counts = telescope[sample_ids].apply(pd.to_numeric, errors="raise").to_numpy(float)
    tumor_columns = np.flatnonzero(samples["is_primary_tumor"].to_numpy())
    mapped_counts = counts[mapping["telescope_index"].to_numpy(dtype=int)][:, tumor_columns]
    rows = []
    for raw_count_threshold in SENSITIVITY_RAW_COUNT_THRESHOLDS:
        prevalence = (mapped_counts >= raw_count_threshold).mean(axis=1)
        for prevalence_threshold in SENSITIVITY_PREVALENCE_THRESHOLDS:
            temporary = loci.copy(); temporary["sensitivity_expressed"] = prevalence >= prevalence_threshold
            for comparison in ["Top1_vs_background", "A549_positive_vs_negative_within_Top1"]:
                row = binary_effect(temporary, "sensitivity_expressed", comparison)
                row.update({"raw_count_threshold": raw_count_threshold, "prevalence_threshold": prevalence_threshold})
                rows.append(row)
    result = pd.DataFrame(rows); result["p_adjust_bh"] = bh_adjust(result["p_value"]); return result


def decile_summary(loci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decile, frame in loci.groupby("priority_decile", sort=True):
        event_n = int(frame["tumor_expressed"].sum()); low, high = exact_binomial_ci(event_n, len(frame))
        rows.append({"priority_decile": decile, "n_loci": len(frame), "tumor_expressed_n": event_n, "tumor_expressed_fraction": event_n / len(frame), "ci_low": low, "ci_high": high, "median_detection_prevalence": frame["tumor_detection_prevalence_count_ge_1"].median(), "a549_supported_fraction": frame["a549_active_support"].mean()})
    return pd.DataFrame(rows).sort_values("priority_decile")


def adjusted_logistic_models(loci: pd.DataFrame, n_source: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = loci.copy()
    model_data["tumor_expressed"] = model_data["tumor_expressed"].astype(int)
    model_data["frozen_top1_numeric"] = model_data["frozen_top1"].astype(int)
    model_data["a549_active_support_numeric"] = model_data["a549_active_support"].astype(int)
    model_data["priority_per_10_percentile"] = 10 * (
        1 - (model_data["herv_functional_priority_rank"] - 1) / (n_source - 1)
    )
    model_data["mef_repFamily"] = model_data["mef_repFamily"].fillna("Unknown").astype(str)

    covariates = {
        "z_log_mef_length": np.log1p(model_data["mef_length"]),
        "z_gc_content": model_data["mef_gc_content"],
        "z_repeat_divergence": model_data["mef_repeat_divergence"],
        "z_log_telescope_length": np.log1p(model_data["telescope_length"]),
        "z_min_reciprocal_fraction": model_data["min_reciprocal_fraction"],
    }
    for name, values in covariates.items():
        numeric = pd.to_numeric(values, errors="coerce")
        model_data[name] = (numeric - numeric.mean()) / numeric.std(ddof=0)

    adjustment = (
        "z_log_mef_length + z_gc_content + z_repeat_divergence + "
        "z_log_telescope_length + z_min_reciprocal_fraction + C(mef_repFamily)"
    )
    formulas = {
        "top1_adjusted": (
            "tumor_expressed ~ frozen_top1_numeric + "
            f"a549_active_support_numeric + {adjustment}"
        ),
        "continuous_priority_adjusted": (
            "tumor_expressed ~ priority_per_10_percentile + "
            f"a549_active_support_numeric + {adjustment}"
        ),
        "priority_by_a549_adjusted": (
            "tumor_expressed ~ priority_per_10_percentile * "
            f"a549_active_support_numeric + {adjustment}"
        ),
    }

    coefficient_rows = []
    diagnostic_rows = []
    for model_name, formula in formulas.items():
        fit = smf.glm(
            formula=formula,
            data=model_data,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC3", maxiter=200)
        confidence = fit.conf_int()
        coefficients = pd.DataFrame(
            {
                "model": model_name,
                "term": fit.params.index,
                "estimate": fit.params.to_numpy(),
                "std_error": fit.bse.to_numpy(),
                "z_value": fit.tvalues.to_numpy(),
                "odds_ratio": np.exp(fit.params.to_numpy()),
                "ci_low": np.exp(confidence[0].to_numpy()),
                "ci_high": np.exp(confidence[1].to_numpy()),
                "p_value": fit.pvalues.to_numpy(),
            }
        )
        coefficient_rows.append(coefficients)
        diagnostic_rows.append(
            {
                "model": model_name,
                "n_loci": int(fit.nobs),
                "n_events": int(model_data.loc[fit.model.data.row_labels, "tumor_expressed"].sum()),
                "converged": bool(fit.converged),
                "covariance": "HC3",
                "log_likelihood": fit.llf,
                "null_log_likelihood": fit.llnull,
                "mcfadden_pseudo_r2": 1 - fit.llf / fit.llnull,
                "aic": fit.aic,
            }
        )

    coefficients = pd.concat(coefficient_rows, ignore_index=True)
    coefficients["p_adjust_bh_within_model"] = coefficients.groupby("model")["p_value"].transform(bh_adjust)
    return coefficients, pd.DataFrame(diagnostic_rows)


def step_18() -> None:
    source, telescope, samples = read_inputs()
    validation = validate_inputs(source, telescope, samples)
    atomic_write(pd.DataFrame({"metric": list(validation), "value": [str(v) for v in validation.values()]}),
                  OUTPUTS["input_consistency"])
    candidates = strict_overlap_candidates(source, telescope); mapping = choose_mutual_best(candidates)
    audit = build_mapping_audit(source, telescope, candidates, mapping)
    atomic_write(audit, OUTPUTS["mapping_audit"])
    selection_audit, selection_family = mapping_selection_audit(source, mapping)
    atomic_write(selection_audit, OUTPUTS["mapping_selection"])
    atomic_write(selection_family, OUTPUTS["mapping_family"])
    loci, size_factors = add_expression_outcomes(mapping, source, telescope, samples)
    atomic_write(loci, OUTPUTS["loci"])
    atomic_write(size_factors, OUTPUTS["size_factors"])
    groups = group_summary(loci)
    binary, continuous = run_effects(loci)
    adjusted = adjusted_logistic_models(loci, len(source))[0]
    sensitivity = run_sensitivity(loci, mapping, telescope, samples)
    deciles = decile_summary(loci)
    atomic_write(groups, OUTPUTS["group_summary"])
    atomic_write(binary, OUTPUTS["binary_effects"])
    atomic_write(continuous, OUTPUTS["continuous_effects"])
    atomic_write(adjusted, OUTPUTS["adjusted_coefficients"])
    atomic_write(sensitivity, OUTPUTS["sensitivity"])
    atomic_write(deciles, OUTPUTS["deciles"])
    summary = {"module": "18_luad_transcriptional_realization", "status": "PASSED", "parameters": {"minimum_reciprocal_overlap": MIN_RECIPROCAL_OVERLAP, "mapping_resolution": "mutual-best one-to-one", "primary_raw_count_threshold": PRIMARY_RAW_COUNT_THRESHOLD, "primary_prevalence_threshold": PRIMARY_PREVALENCE_THRESHOLD, "continuous_normalization": "sparse positive-count median-of-ratios", "adjusted_covariates": ["repFamily", "mef_length", "gc_content", "repeat_divergence", "telescope_length", "min_reciprocal_fraction"], "published_file_s4_recomputed": False}, "input_consistency": validation, "dimensions": {"source_loci": len(source), "telescope_loci": len(telescope), "luad_samples": len(samples), "primary_tumor_samples": int(samples["is_primary_tumor"].sum()), "adjacent_normal_samples": int(samples["is_adjacent_normal"].sum()), "mapped_one_to_one_loci": len(loci), "mapped_top1_loci": int(loci["frozen_top1"].sum()), "mapped_top1_a549_positive_loci": int((loci["frozen_top1"] & loci["a549_active_support"]).sum())}, "outputs": {key: str(path) for key, path in OUTPUTS.items()}}
    atomic_json(summary, OUTPUTS["summary"])


# --------------------------------------------------------------------------
# step 19: Analyze the nested LUAD model increment
# --------------------------------------------------------------------------



PROJECT_ROOT__19 = Path(__file__).resolve().parent.parent
D18 = PROJECT_ROOT__19 / "Biological_Interpretation/output/18_luad_transcriptional_realization"
INPUT_LOCI = D18 / "18_luad_locus_transcriptional_realization.tsv.gz"
INPUT_SUMMARY = D18 / "18_summary.json"
OUT_DIR__19 = PROJECT_ROOT__19 / "Biological_Interpretation/output/19_luad_nested_model_increment"

N_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_SEED = 20261008

BACKGROUND_NUMERIC = [
    "z_log_mef_length", "z_gc_content", "z_repeat_divergence",
    "z_log_telescope_length", "z_min_reciprocal_fraction",
]
MODEL_LABELS = {"M0": "Background", "M1": "+ A549", "M2": "+ HERV priority"}
METRIC_NAMES = ("AUROC", "Average precision")
ADDITIONS = [
    ("M0", "M1", "A549 chromatin support"),
    ("M1", "M2", "HERV priority"),
]


def as_bool__19(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(["true", "t", "1", "1.0"])


def prepare_data(loci: pd.DataFrame, n_source: int):
    data = loci.copy()
    numeric = [
        "mef_length", "mef_gc_content", "mef_repeat_divergence",
        "telescope_length", "min_reciprocal_fraction", "herv_functional_priority_rank",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data["mef_repFamily"] = data["mef_repFamily"].fillna("Unknown").astype(str)
    data = data.dropna(subset=numeric + ["tumor_expressed", "a549_active_support"]).reset_index(drop=True)
    data["tumor_expressed_numeric"] = as_bool__19(data["tumor_expressed"]).astype(int)
    data["a549_active_support_numeric"] = as_bool__19(data["a549_active_support"]).astype(int)
    data["priority_per_10_percentile"] = 10 * (
        1 - (data["herv_functional_priority_rank"] - 1) / (n_source - 1)
    )

    design = pd.DataFrame({
        "const": np.ones(len(data)),
        "z_log_mef_length": np.log1p(data["mef_length"]),
        "z_gc_content": data["mef_gc_content"],
        "z_repeat_divergence": data["mef_repeat_divergence"],
        "z_log_telescope_length": np.log1p(data["telescope_length"]),
        "z_min_reciprocal_fraction": data["min_reciprocal_fraction"],
    })
    # Fixed category columns keep the feature schema aligned across folds.
    family = pd.Categorical(data["mef_repFamily"], categories=sorted(data["mef_repFamily"].unique()))
    family_columns = pd.get_dummies(family, prefix="family", drop_first=True, dtype=float)
    design = pd.concat([design, family_columns], axis=1)
    background_columns = design.columns.tolist()
    design["a549_active_support_numeric"] = data["a549_active_support_numeric"]
    design["priority_per_10_percentile"] = data["priority_per_10_percentile"]
    columns = {
        "M0": background_columns,
        "M1": background_columns + ["a549_active_support_numeric"],
        "M2": background_columns + ["a549_active_support_numeric", "priority_per_10_percentile"],
    }
    return data, design.astype(float), columns


def scale_design(design: pd.DataFrame, training_indices: np.ndarray) -> pd.DataFrame:
    result = design.copy()
    training = design.iloc[training_indices][BACKGROUND_NUMERIC]
    scale = training.std(ddof=0).replace(0, 1)
    result[BACKGROUND_NUMERIC] = (design[BACKGROUND_NUMERIC] - training.mean()) / scale
    return result


def make_folds(outcome: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    fold_ids = np.empty(len(outcome), dtype=int)
    for event in (0, 1):
        indices = rng.permutation(np.flatnonzero(outcome == event))
        for fold, members in enumerate(np.array_split(indices, N_FOLDS), start=1):
            fold_ids[members] = fold
    return fold_ids


def oof_predictions(outcome: np.ndarray, design: pd.DataFrame, columns: dict):
    fold_ids = make_folds(outcome)
    predictions = {model: np.empty(len(outcome), dtype=float) for model in MODEL_LABELS}
    fold_records = []
    for fold in range(1, N_FOLDS + 1):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        scaled = scale_design(design, train)
        for model in MODEL_LABELS:
            fit = sm.GLM(outcome[train], scaled.iloc[train][columns[model]],
                         family=sm.families.Binomial()).fit(maxiter=200)
            predictions[model][test] = np.asarray(fit.predict(scaled.iloc[test][columns[model]]))
            fold_records.append({
                "model": model, "fold": fold, "train_n": len(train), "test_n": len(test),
                "train_events": int(outcome[train].sum()), "test_events": int(outcome[test].sum()),
                "converged": bool(fit.converged),
            })
    return predictions, fold_ids, pd.DataFrame(fold_records)


def score_metrics(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    event_n = int(outcome.sum())
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    auc = (ranks[outcome == 1].sum() - event_n * (event_n + 1) / 2) / (
        event_n * (len(outcome) - event_n)
    )
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_probability)), len(outcome) - 1]
    true_positive = np.cumsum(outcome[order])[ends]
    precision = true_positive / (ends + 1)
    recall = true_positive / event_n
    average_precision = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return np.array([auc, average_precision], dtype=float)


def paired_bootstrap(outcome: np.ndarray, predictions: dict):
    models = list(MODEL_LABELS)
    point = np.array([score_metrics(outcome, predictions[model]) for model in models])
    samples = np.empty((N_BOOTSTRAP, len(models), len(METRIC_NAMES)), dtype=float)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    strata = [np.flatnonzero(outcome == event) for event in (0, 1)]
    for iteration in range(N_BOOTSTRAP):
        indices = np.concatenate([rng.choice(members, size=len(members), replace=True) for members in strata])
        for model_index, model in enumerate(models):
            samples[iteration, model_index] = score_metrics(outcome[indices], predictions[model][indices])
    limits = np.quantile(samples, [0.025, 0.975], axis=0)

    performance = []
    for model_index, model in enumerate(models):
        performance.append({
            "model": model, "model_label": MODEL_LABELS[model], "n_loci": len(outcome),
            "n_events": int(outcome.sum()), "event_prevalence": float(outcome.mean()),
            "cv_folds": N_FOLDS, "bootstrap_n": N_BOOTSTRAP,
            "roc_auc": point[model_index, 0],
            "roc_auc_ci_low": limits[0, model_index, 0], "roc_auc_ci_high": limits[1, model_index, 0],
            "average_precision": point[model_index, 1],
            "average_precision_ci_low": limits[0, model_index, 1],
            "average_precision_ci_high": limits[1, model_index, 1],
        })
    increments = []
    for before, after, added in ADDITIONS:
        before_index, after_index = models.index(before), models.index(after)
        differences = samples[:, after_index] - samples[:, before_index]
        bounds = np.quantile(differences, [0.025, 0.975], axis=0)
        for metric_index, metric in enumerate(METRIC_NAMES):
            increments.append({
                "comparison": f"{after}_vs_{before}", "added_feature": added, "metric": metric,
                "delta": point[after_index, metric_index] - point[before_index, metric_index],
                "ci_low": bounds[0, metric_index], "ci_high": bounds[1, metric_index],
                "n_loci": len(outcome), "bootstrap_n": N_BOOTSTRAP,
            })
    return pd.DataFrame(performance), pd.DataFrame(increments)


def roc_source_data(outcome: np.ndarray, predictions: dict) -> pd.DataFrame:
    records = []
    event_n = outcome.sum()
    nonevent_n = len(outcome) - event_n
    for model, probability in predictions.items():
        order = np.argsort(-probability, kind="mergesort")
        ends = np.r_[np.flatnonzero(np.diff(probability[order])), len(outcome) - 1]
        true_positive = np.cumsum(outcome[order])[ends]
        records.append(pd.DataFrame({
            "model": model, "model_label": MODEL_LABELS[model],
            "false_positive_rate": np.r_[0.0, (ends + 1 - true_positive) / nonevent_n],
            "true_positive_rate": np.r_[0.0, true_positive / event_n],
            "threshold": np.r_[np.inf, probability[order][ends]],
        }))
    return pd.concat(records, ignore_index=True)


def full_model_results(outcome: np.ndarray, design: pd.DataFrame, columns: dict):
    scaled = scale_design(design, np.arange(len(outcome)))
    fits, coefficient_records = {}, []
    for model in MODEL_LABELS:
        fit = sm.GLM(outcome, scaled[columns[model]], family=sm.families.Binomial()).fit(
            cov_type="HC3", maxiter=200
        )
        fits[model] = fit
        confidence = fit.conf_int()
        coefficient_records.append(pd.DataFrame({
            "model": model, "term": fit.params.index, "estimate": fit.params.to_numpy(),
            "std_error": fit.bse.to_numpy(), "odds_ratio": np.exp(fit.params.to_numpy()),
            "ci_low": np.exp(confidence.iloc[:, 0].to_numpy()),
            "ci_high": np.exp(confidence.iloc[:, 1].to_numpy()), "p_value": fit.pvalues.to_numpy(),
        }))
    tests = []
    for before, after, added in ADDITIONS:
        statistic = max(0.0, 2 * (fits[after].llf - fits[before].llf))
        df = int(fits[after].df_model - fits[before].df_model)
        tests.append({
            "comparison": f"{after}_vs_{before}", "added_feature": added,
            "likelihood_ratio": statistic, "df": df, "p_value": float(chi2.sf(statistic, df)),
            "n_loci": int(fits[after].nobs), "converged_before": bool(fits[before].converged),
            "converged_after": bool(fits[after].converged),
        })
    tests = pd.DataFrame(tests)
    order = np.argsort(tests["p_value"].to_numpy())
    ranked = tests["p_value"].to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    tests["p_adjust_bh_two_increment_tests"] = np.clip(adjusted, 0, 1)[np.argsort(order)]
    return pd.concat(coefficient_records, ignore_index=True), tests


def step_19() -> None:
    OUT_DIR__19.mkdir(parents=True, exist_ok=True)
    with INPUT_SUMMARY.open(encoding="utf-8") as handle:
        previous = json.load(handle)
    n_source = int(previous["dimensions"]["source_loci"])
    loci = pd.read_csv(INPUT_LOCI, sep="\t", compression="gzip", low_memory=False)
    data, design, columns = prepare_data(loci, n_source)
    outcome = data["tumor_expressed_numeric"].to_numpy(dtype=int)
    predictions, folds, fold_records = oof_predictions(outcome, design, columns)
    metrics, increments = paired_bootstrap(outcome, predictions)
    coefficients, likelihood_tests = full_model_results(outcome, design, columns)
    prediction_table = data[[
        "herv_internal_id", "telescope_id", "herv_functional_priority_rank",
        "tumor_expressed_numeric", "a549_active_support_numeric", "priority_per_10_percentile",
    ]].copy()
    prediction_table["fold"] = folds
    for model, probability in predictions.items():
        prediction_table[f"{model}_oof_probability"] = probability
    outputs = {
        "19_nested_model_metrics.tsv": metrics,
        "19_nested_model_increments.tsv": increments,
        "19_nested_model_likelihood_tests.tsv": likelihood_tests,
        "19_nested_model_coefficients.tsv": coefficients,
        "19_nested_model_fold_results.tsv": fold_records,
        "19_nested_model_roc_source_data.tsv": roc_source_data(outcome, predictions),
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_DIR__19 / filename, sep="\t", index=False)
    summary = {
        "analysis": "luad_nested_model_increment", "input_loci": str(INPUT_LOCI),
        "input_summary": str(INPUT_SUMMARY), "input_mapped_loci": len(loci),
        "analysis_loci": len(data), "complete_case_removed_n": len(loci) - len(data),
        "detected_loci": int(outcome.sum()), "source_catalogue_n": n_source,
        "outcome": "Module 18 tumor_expressed; detected=1, not detected=0",
        "inherited_detection_thresholds": {
            "raw_count_greater_than_or_equal": previous["parameters"]["primary_raw_count_threshold"],
            "prevalence_greater_than_or_equal": previous["parameters"]["primary_prevalence_threshold"],
        },
        "models": {
            "M0": "family + log MEF length + GC + repeat divergence + log Telescope length + minimum reciprocal overlap",
            "M1": "M0 + A549 active support", "M2": "M1 + frozen priority per 10 percentile points",
        },
        "cross_validation": "five outcome-stratified folds, same folds for every model; scaling fit on training folds",
        "fold_n": N_FOLDS, "random_seed": RANDOM_SEED, "bootstrap_n": N_BOOTSTRAP,
        "bootstrap": "paired outcome-stratified locus resampling of fixed OOF predictions; models not refitted",
        "secondary_metric": "average precision, not trapezoidal PR area",
        "coefficient_covariance": "HC3", "likelihood_test_bh_scope": "two planned incremental tests",
        "herv_mef_retrained": False, "mapping_recomputed": False, "differential_expression_recomputed": False,
    }
    with (OUT_DIR__19 / "19_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    step_18()
    step_19()


if __name__ == "__main__":
    main()
