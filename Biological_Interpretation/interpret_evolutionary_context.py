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
import shutil
import numpy as np
import pandas as pd
from typing import (
    Any,
    Iterable,
)
from scipy.stats import (
    fisher_exact,
    mannwhitneyu,
    binomtest,
    rankdata,
    spearmanr,
    wilcoxon,
    chi2_contingency,
)
from biological_common import (
    now,
    atomic_write_text,
    atomic_write_tsv,
)

# --------------------------------------------------------------------------
# step 25: Evolutionary interpretation of the frozen ranking
# --------------------------------------------------------------------------

# =============================================================================
# 1. Fixed paths and immutable contracts
# =============================================================================

HERE = Path(__file__).resolve().parent
MODELING = (HERE.parent / "Modeling").resolve()

SCRIPT = "25_evolutionary_interpretation"
VERSION = "round2_v2"

M23_DIR = MODELING / "output/23_generate_genomewide_herv_functional_ranking"
M23_ATLAS = M23_DIR / "23_herv_genomewide_functional_ranking.parquet"

M24_DIR = MODELING / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes"
M24_PROJECTION = M24_DIR / "24_tumor_projection_all_herv.parquet"
M24_CANDIDATES = M24_DIR / "24_potential_reactivation_candidates.tsv"

# M05 validation-only sequence comparators. These are used only for the
# preregistered evolutionary comparator panel; no labels are loaded.
M05_DIR = MODELING / "output/05_k6_bpe_complementarity_controls"
M05_PREDICTIONS = M05_DIR / "05_validation_predictions.parquet"

OUT = HERE / "output/25_evolutionary_interpretation"
OUT_FAMILY = OUT / "25_model_layer_family_enrichment.tsv"


OUT_LAYER_SUMMARY = OUT / "25_model_layer_evolution_summary.tsv"


OUT_MATCHED_BACKGROUND = OUT / "25_final_top1_matched_background.tsv"
OUT_MATCHED_COMPARE = OUT / "25_final_top1_matched_divergence_comparison.tsv"
OUT_MATCHING_BALANCE = OUT / "25_final_top1_matching_balance.tsv"


OUT_TOP30 = OUT / "25_top30_reactivation_candidate_evolution.tsv"


ID = "herv_internal_id"
EXPECTED_N = 480_306
TOP_FRAC = 0.01
TOP1_K = int(math.floor(EXPECTED_N * TOP_FRAC))

EXPECTED_GLOBAL = "R4_ALL_COMPACT_bs64_epoch_24"
EXPECTED_PRIMARY = "A3_ALL38_PRIMARY_epoch_01"

MIN_SUBFAMILY_UNIVERSE = 20
MIN_GROUP_FOR_MANNWHITNEY = 10
MATCH_SEED = 99999999
VALID_EXPECTED_N = 72_048
VALID_TOP1_K = int(math.floor(VALID_EXPECTED_N * TOP_FRAC))

VALIDATION_COMPARATOR_SCORES = {
    "K6_ONLY": "score_k6_meta",
    "BPE_ONLY": "score_bpe_meta",
    "FULL_DUAL": "score_full_dual_meta",
    "DUAL_NO_MPRA": "score_dual_no_mpra_meta",
}
VALIDATION_DOWNSTREAM_SCORES = {
    "M10_REPRESENTATION": "score_final_representation",
    "FINAL_GLOBAL": "score_final_global",
    "FINAL_FER": "herv_functional_priority_score",
}

TUMOR_EIDS = ["E114", "E115", "E117", "E118", "E123"]
CELL_LINE_NAMES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

LAYER_ORDER = [
    "PRE_M10_REPRESENTATION",
    "M10_BIOLOGICAL_CALIBRATED",
    "FINAL_GLOBAL",
    "FINAL_FER",
]
LAYER_LABELS = {
    "PRE_M10_REPRESENTATION": (
        "M07-derived pre-M10 representation; no explicit M10 biological calibrator"
    ),
    "M10_BIOLOGICAL_CALIBRATED": "M10 accepted P2_C1 biological calibration",
    "FINAL_GLOBAL": "Frozen environment-aware Final Global ranker",
    "FINAL_FER": "Frozen Final Global + Top10% All38 FER stitched ordering",
}

REQUIRED_M23_COLUMNS = [
    ID,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repClass",
    "repFamily",
    "repeat_divergence",
    "length",
    "gc_content",
    "dist_to_tss",
    "split",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "score_sequence_regulatory",
    "score_final_representation",
    "score_final_global",
    "global_rank",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
]

REQUIRED_M24_COLUMNS = [
    ID,
    "herv_functional_priority_rank",
    "m23_top1pct",
    "potential_reactivation_candidate",
    "tumor_epigenomic_support_score",
    "tumor_repressive_context_score",
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_repressive_support_breadth",
    "tumor_cell_line_support_class",
    "tumor_active_supported_eids",
]
for _eid in TUMOR_EIDS:
    REQUIRED_M24_COLUMNS.extend(
        [
            f"tumor_{_eid}_any_active_support",
            f"tumor_{_eid}_active_mark_count",
            f"tumor_{_eid}_repressive_mark_count",
        ]
    )


# =============================================================================
# 2. Generic utilities
# =============================================================================


def prepare_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    adjusted = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return adjusted
    finite_idx = np.flatnonzero(finite)
    ordered_idx = finite_idx[np.argsort(p[finite_idx], kind="mergesort")]
    running = 1.0
    m = len(ordered_idx)
    for reverse_pos in range(m - 1, -1, -1):
        idx = ordered_idx[reverse_pos]
        rank = reverse_pos + 1
        running = min(running, float(p[idx]) * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def safe_mannwhitney(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < MIN_GROUP_FOR_MANNWHITNEY or len(y) < MIN_GROUP_FOR_MANNWHITNEY:
        return {
            "u": float("nan"),
            "p": float("nan"),
            "rank_biserial_x_greater": float("nan"),
        }
    test = mannwhitneyu(x, y, alternative="two-sided")
    effect = 2.0 * float(test.statistic) / (len(x) * len(y)) - 1.0
    return {
        "u": float(test.statistic),
        "p": float(test.pvalue),
        "rank_biserial_x_greater": float(effect),
    }


def stable_top_ids(frame: pd.DataFrame, score_column: str, k: int) -> set[str]:
    score = pd.to_numeric(frame[score_column], errors="raise")
    ranked = pd.DataFrame(
        {
            ID: frame[ID].astype(str).to_numpy(),
            "score": score.to_numpy(dtype=np.float64),
        }
    ).sort_values(
        ["score", ID],
        ascending=[False, True],
        kind="mergesort",
    )
    return set(ranked.head(k)[ID].astype(str))


# =============================================================================
# 3. Load frozen atlases
# =============================================================================


def audit_and_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = pd.read_parquet(M23_ATLAS)
    projection = pd.read_parquet(M24_PROJECTION)
    candidates = pd.read_csv(M24_CANDIDATES, sep="\t", low_memory=False)

    m05_long = pd.read_parquet(
        M05_PREDICTIONS,
        columns=["model", ID, "score_sequence_mean"],
    )
    m05_long[ID] = m05_long[ID].astype(str)
    m05_validation = (
        m05_long.pivot(index=ID, columns="model", values="score_sequence_mean")
        .rename(columns=lambda col: f"score_{col}")
        .reset_index()
    )

    atlas[ID] = atlas[ID].astype(str)
    projection[ID] = projection[ID].astype(str)

    projection = atlas[[ID]].merge(
        projection,
        on=ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    candidates[ID] = candidates[ID].astype(str)
    return atlas, projection, candidates, m05_validation


def build_analysis_table(atlas: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    table = atlas[
        [
            ID,
            "chrom",
            "start",
            "end",
            "strand",
            "repName",
            "repClass",
            "repFamily",
            "repeat_divergence",
            "length",
            "gc_content",
            "dist_to_tss",
            "split",
            "nearest_gene_id",
            "nearest_gene_name",
            "nearest_gene_type",
            "score_sequence_regulatory",
            "score_final_representation",
            "score_final_global",
            "global_rank",
            "herv_functional_priority_rank",
            "herv_functional_priority_score",
        ]
    ].copy()

    tumor_cols = [
        ID,
        "potential_reactivation_candidate",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "tumor_active_support_breadth",
        "tumor_active_dominant_breadth",
        "tumor_repressive_support_breadth",
        "tumor_cell_line_support_class",
        "tumor_active_supported_eids",
    ]
    for eid in TUMOR_EIDS:
        tumor_cols.extend(
            [
                f"tumor_{eid}_any_active_support",
                f"tumor_{eid}_active_mark_count",
                f"tumor_{eid}_repressive_mark_count",
            ]
        )
    table = table.merge(
        projection[tumor_cols],
        on=ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    table["repFamily"] = table["repFamily"].fillna("NA").astype(str)
    table["repName"] = table["repName"].fillna("NA").astype(str)
    table["repeat_divergence"] = pd.to_numeric(
        table["repeat_divergence"], errors="raise"
    ).astype(np.float64)
    table["length"] = pd.to_numeric(table["length"], errors="raise").astype(np.int64)

    # Rank within family from low divergence to high divergence.
    table["divergence_percentile_within_family"] = table.groupby(
        "repFamily", sort=False, observed=True
    )["repeat_divergence"].rank(method="average", pct=True)

    table["family_universe_count"] = table["repFamily"].map(
        table["repFamily"].value_counts()
    ).astype(np.int64)
    table["subfamily_universe_count"] = table["repName"].map(
        table["repName"].value_counts()
    ).astype(np.int64)

    q = table["divergence_percentile_within_family"].to_numpy(dtype=np.float64)
    table["relative_divergence_class_within_family"] = np.select(
        [q <= 0.25, q <= 0.75],
        [
            "low_divergence_within_family",
            "intermediate_divergence_within_family",
        ],
        default="high_divergence_within_family",
    )

    # Broad family flags are descriptive only.
    upper_family = table["repFamily"].str.upper()
    table["is_ERVK_family"] = upper_family.eq("ERVK")
    table["is_ERV1_family"] = upper_family.eq("ERV1")

    # Frozen genome-wide Top1 membership at each deployable stage.
    top_sets = {
        "PRE_M10_REPRESENTATION": stable_top_ids(
            table, "score_sequence_regulatory", TOP1_K
        ),
        "M10_BIOLOGICAL_CALIBRATED": stable_top_ids(
            table, "score_final_representation", TOP1_K
        ),
        "FINAL_GLOBAL": set(
            table.loc[
                pd.to_numeric(table["global_rank"], errors="raise").le(TOP1_K), ID
            ].astype(str)
        ),
        "FINAL_FER": set(
            table.loc[
                pd.to_numeric(
                    table["herv_functional_priority_rank"], errors="raise"
                ).le(TOP1_K),
                ID,
            ].astype(str)
        ),
    }
    for layer, ids in top_sets.items():
        table[f"top1_{layer}"] = table[ID].isin(ids)

    return table


# =============================================================================
# 4. Model-layer family / subfamily enrichment
# =============================================================================


def enrichment_for_layer(
    table: pd.DataFrame,
    layer: str,
    group_column: str,
    min_universe: int,
) -> pd.DataFrame:
    selected_col = f"top1_{layer}"
    selected = table[selected_col].astype(bool)
    selected_total = int(selected.sum())
    nonselected_total = len(table) - selected_total

    rows: list[dict[str, Any]] = []
    for group, subset in table.groupby(group_column, dropna=False, sort=True):
        universe_n = int(len(subset))
        if universe_n < min_universe:
            continue
        group_selected = int(subset[selected_col].astype(bool).sum())
        group_nonselected = universe_n - group_selected
        other_selected = selected_total - group_selected
        other_nonselected = nonselected_total - group_nonselected
        odds_ratio, p_value = fisher_exact(
            [
                [group_selected, other_selected],
                [group_nonselected, other_nonselected],
            ],
            alternative="two-sided",
        )
        selected_share = group_selected / selected_total
        universe_share = universe_n / len(table)
        rows.append(
            {
                "layer": layer,
                "layer_description": LAYER_LABELS[layer],
                "group_column": group_column,
                "group": "NA" if pd.isna(group) else str(group),
                "top1_count": group_selected,
                "top1_share": selected_share,
                "universe_count": universe_n,
                "universe_share": universe_share,
                "fold_enrichment": (
                    selected_share / universe_share if universe_share > 0 else float("nan")
                ),
                "odds_ratio": float(odds_ratio),
                "fisher_two_sided_p": float(p_value),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_bh_within_layer"] = bh_adjust(result["fisher_two_sided_p"])
    return result.sort_values(
        ["fdr_bh_within_layer", "fold_enrichment", "universe_count", "group"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_enrichment_outputs(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family = pd.concat(
        [
            enrichment_for_layer(table, layer, "repFamily", min_universe=1)
            for layer in LAYER_ORDER
        ],
        ignore_index=True,
    )
    subfamily = pd.concat(
        [
            enrichment_for_layer(
                table, layer, "repName", min_universe=MIN_SUBFAMILY_UNIVERSE
            )
            for layer in LAYER_ORDER
        ],
        ignore_index=True,
    )

    # One row per family: directly expose when enrichment emerges across layers.
    base = family.pivot_table(
        index="group",
        columns="layer",
        values=["top1_count", "fold_enrichment", "fdr_bh_within_layer"],
        aggfunc="first",
    )
    base.columns = [f"{metric}__{layer}" for metric, layer in base.columns]
    transition = base.reset_index().rename(columns={"group": "repFamily"})

    def delta_log2(left: str, right: str) -> np.ndarray:
        lcol = f"fold_enrichment__{left}"
        rcol = f"fold_enrichment__{right}"
        left_v = pd.to_numeric(transition[lcol], errors="coerce").to_numpy(dtype=float)
        right_v = pd.to_numeric(transition[rcol], errors="coerce").to_numpy(dtype=float)
        out = np.full(len(transition), np.nan)
        valid = (left_v > 0) & (right_v > 0) & np.isfinite(left_v) & np.isfinite(right_v)
        out[valid] = np.log2(right_v[valid]) - np.log2(left_v[valid])
        return out

    transition["delta_log2FE_PRE_M10_to_M10"] = delta_log2(
        "PRE_M10_REPRESENTATION", "M10_BIOLOGICAL_CALIBRATED"
    )
    transition["delta_log2FE_M10_to_GLOBAL"] = delta_log2(
        "M10_BIOLOGICAL_CALIBRATED", "FINAL_GLOBAL"
    )
    transition["delta_log2FE_GLOBAL_to_FER"] = delta_log2(
        "FINAL_GLOBAL", "FINAL_FER"
    )
    transition["delta_log2FE_PRE_M10_to_FINAL"] = delta_log2(
        "PRE_M10_REPRESENTATION", "FINAL_FER"
    )

    # A descriptive flag, not a statistical declaration of causality.
    seq_fe = pd.to_numeric(
        transition.get("fold_enrichment__PRE_M10_REPRESENTATION"), errors="coerce"
    )
    m10_fe = pd.to_numeric(
        transition.get("fold_enrichment__M10_BIOLOGICAL_CALIBRATED"), errors="coerce"
    )
    final_fe = pd.to_numeric(
        transition.get("fold_enrichment__FINAL_FER"), errors="coerce"
    )
    transition["enrichment_interpretation_flag"] = np.select(
        [
            (seq_fe <= 1.05) & (m10_fe > 1.20),
            (seq_fe > 1.20) & (final_fe > 1.20),
        ],
        [
            "emerges_after_explicit_M10_prior_descriptive",
            "already_present_pre_M10_and_persists_descriptive",
        ],
        default="mixed_or_weak_pattern",
    )

    return family, subfamily, transition.sort_values("repFamily").reset_index(drop=True)


# =============================================================================
# 5. Layer-level evolutionary summaries and overlaps
# =============================================================================


def build_layer_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in LAYER_ORDER:
        mask = table[f"top1_{layer}"].astype(bool)
        top = table.loc[mask]
        bg = table.loc[~mask]

        div_test = safe_mannwhitney(
            top["repeat_divergence"].to_numpy(), bg["repeat_divergence"].to_numpy()
        )
        pct_test = safe_mannwhitney(
            top["divergence_percentile_within_family"].to_numpy(),
            bg["divergence_percentile_within_family"].to_numpy(),
        )
        len_test = safe_mannwhitney(top["length"].to_numpy(), bg["length"].to_numpy())

        rows.append(
            {
                "layer": layer,
                "layer_description": LAYER_LABELS[layer],
                "top1_n": int(len(top)),
                "ERVK_top1_count": int(top["is_ERVK_family"].sum()),
                "ERVK_top1_share": float(top["is_ERVK_family"].mean()),
                "ERV1_top1_count": int(top["is_ERV1_family"].sum()),
                "ERV1_top1_share": float(top["is_ERV1_family"].mean()),
                "top1_mean_repeat_divergence": float(top["repeat_divergence"].mean()),
                "top1_median_repeat_divergence": float(top["repeat_divergence"].median()),
                "background_median_repeat_divergence": float(bg["repeat_divergence"].median()),
                "repeat_divergence_median_delta_top_minus_background": float(
                    top["repeat_divergence"].median() - bg["repeat_divergence"].median()
                ),
                "repeat_divergence_mannwhitney_p": div_test["p"],
                "repeat_divergence_rank_biserial_top_greater": div_test[
                    "rank_biserial_x_greater"
                ],
                "top1_mean_within_family_divergence_percentile": float(
                    top["divergence_percentile_within_family"].mean()
                ),
                "top1_median_within_family_divergence_percentile": float(
                    top["divergence_percentile_within_family"].median()
                ),
                "background_median_within_family_divergence_percentile": float(
                    bg["divergence_percentile_within_family"].median()
                ),
                "within_family_divergence_median_delta_top_minus_background": float(
                    top["divergence_percentile_within_family"].median()
                    - bg["divergence_percentile_within_family"].median()
                ),
                "within_family_divergence_mannwhitney_p": pct_test["p"],
                "within_family_divergence_rank_biserial_top_greater": pct_test[
                    "rank_biserial_x_greater"
                ],
                "top1_mean_length": float(top["length"].mean()),
                "top1_median_length": float(top["length"].median()),
                "background_median_length": float(bg["length"].median()),
                "length_median_delta_top_minus_background": float(
                    top["length"].median() - bg["length"].median()
                ),
                "length_mannwhitney_p": len_test["p"],
                "length_rank_biserial_top_greater": len_test[
                    "rank_biserial_x_greater"
                ],
            }
        )

    result = pd.DataFrame(rows)
    for source, target in [
        ("repeat_divergence_mannwhitney_p", "repeat_divergence_fdr_bh_across_layers"),
        (
            "within_family_divergence_mannwhitney_p",
            "within_family_divergence_fdr_bh_across_layers",
        ),
        ("length_mannwhitney_p", "length_fdr_bh_across_layers"),
    ]:
        result[target] = bh_adjust(result[source])
    return result


def build_overlap_table(table: pd.DataFrame) -> pd.DataFrame:
    sets = {
        layer: set(table.loc[table[f"top1_{layer}"].astype(bool), ID].astype(str))
        for layer in LAYER_ORDER
    }
    rows = []
    for left in LAYER_ORDER:
        for right in LAYER_ORDER:
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            rows.append(
                {
                    "layer_a": left,
                    "layer_b": right,
                    "top1_n_a": len(sets[left]),
                    "top1_n_b": len(sets[right]),
                    "intersection_n": intersection,
                    "union_n": union,
                    "fraction_of_a_retained_in_b": intersection / len(sets[left]),
                    "fraction_of_b_shared_with_a": intersection / len(sets[right]),
                    "jaccard": intersection / union if union else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 6. Preregistered validation sequence-comparator panel
# =============================================================================


def _generic_enrichment(
    frame: pd.DataFrame,
    selected_col: str,
    group_column: str,
    comparator: str,
    min_universe: int,
) -> pd.DataFrame:
    selected = frame[selected_col].astype(bool)
    selected_total = int(selected.sum())
    nonselected_total = len(frame) - selected_total
    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby(group_column, dropna=False, sort=True):
        universe_n = int(len(subset))
        if universe_n < min_universe:
            continue
        group_selected = int(subset[selected_col].astype(bool).sum())
        group_nonselected = universe_n - group_selected
        other_selected = selected_total - group_selected
        other_nonselected = nonselected_total - group_nonselected
        odds_ratio, p_value = fisher_exact(
            [[group_selected, other_selected], [group_nonselected, other_nonselected]],
            alternative="two-sided",
        )
        selected_share = group_selected / selected_total
        universe_share = universe_n / len(frame)
        rows.append(
            {
                "comparator": comparator,
                "group_column": group_column,
                "group": "NA" if pd.isna(group) else str(group),
                "top1_count": group_selected,
                "top1_share": selected_share,
                "universe_count": universe_n,
                "universe_share": universe_share,
                "fold_enrichment": selected_share / universe_share,
                "odds_ratio": float(odds_ratio),
                "fisher_two_sided_p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_bh_within_comparator"] = bh_adjust(result["fisher_two_sided_p"])
        result = result.sort_values(
            ["fdr_bh_within_comparator", "fold_enrichment", "universe_count", "group"],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return result


def build_validation_sequence_comparator_panel(
    table: pd.DataFrame,
    m05_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_meta = table.loc[
        table["split"].astype(str).eq("valid"),
        [
            ID,
            "repFamily",
            "repName",
            "repeat_divergence",
            "divergence_percentile_within_family",
            "length",
            "is_ERVK_family",
            "is_ERV1_family",
            *VALIDATION_DOWNSTREAM_SCORES.values(),
        ],
    ].copy()
    panel = valid_meta.merge(
        m05_validation,
        on=ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    score_map = {**VALIDATION_COMPARATOR_SCORES, **VALIDATION_DOWNSTREAM_SCORES}

    top_sets: dict[str, set[str]] = {}
    for comparator, score_column in score_map.items():
        ids = stable_top_ids(panel, score_column, VALID_TOP1_K)
        top_sets[comparator] = ids
        panel[f"top1_{comparator}"] = panel[ID].isin(ids)

    rows = []
    for comparator in score_map:
        top = panel.loc[panel[f"top1_{comparator}"]]
        bg = panel.loc[~panel[f"top1_{comparator}"]]
        div_test = safe_mannwhitney(top["repeat_divergence"].to_numpy(), bg["repeat_divergence"].to_numpy())
        within_test = safe_mannwhitney(
            top["divergence_percentile_within_family"].to_numpy(),
            bg["divergence_percentile_within_family"].to_numpy(),
        )
        len_test = safe_mannwhitney(top["length"].to_numpy(), bg["length"].to_numpy())
        rows.append(
            {
                "comparator": comparator,
                "scope": "validation_only",
                "validation_n": len(panel),
                "top1_k": VALID_TOP1_K,
                "ERVK_top1_count": int(top["is_ERVK_family"].sum()),
                "ERVK_top1_share": float(top["is_ERVK_family"].mean()),
                "ERV1_top1_count": int(top["is_ERV1_family"].sum()),
                "ERV1_top1_share": float(top["is_ERV1_family"].mean()),
                "top1_median_repeat_divergence": float(top["repeat_divergence"].median()),
                "background_median_repeat_divergence": float(bg["repeat_divergence"].median()),
                "repeat_divergence_delta": float(top["repeat_divergence"].median() - bg["repeat_divergence"].median()),
                "repeat_divergence_mannwhitney_p": div_test["p"],
                "repeat_divergence_rank_biserial_top_greater": div_test["rank_biserial_x_greater"],
                "top1_median_within_family_divergence_percentile": float(top["divergence_percentile_within_family"].median()),
                "background_median_within_family_divergence_percentile": float(bg["divergence_percentile_within_family"].median()),
                "within_family_divergence_delta": float(
                    top["divergence_percentile_within_family"].median()
                    - bg["divergence_percentile_within_family"].median()
                ),
                "within_family_divergence_mannwhitney_p": within_test["p"],
                "within_family_divergence_rank_biserial_top_greater": within_test["rank_biserial_x_greater"],
                "top1_median_length": float(top["length"].median()),
                "background_median_length": float(bg["length"].median()),
                "length_delta": float(top["length"].median() - bg["length"].median()),
                "length_mannwhitney_p": len_test["p"],
                "length_rank_biserial_top_greater": len_test["rank_biserial_x_greater"],
            }
        )
    summary = pd.DataFrame(rows)
    for source, target in [
        ("repeat_divergence_mannwhitney_p", "repeat_divergence_fdr_bh_across_comparators"),
        ("within_family_divergence_mannwhitney_p", "within_family_divergence_fdr_bh_across_comparators"),
        ("length_mannwhitney_p", "length_fdr_bh_across_comparators"),
    ]:
        summary[target] = bh_adjust(summary[source])

    family = pd.concat(
        [_generic_enrichment(panel, f"top1_{c}", "repFamily", c, 1) for c in score_map],
        ignore_index=True,
    )
    subfamily = pd.concat(
        [_generic_enrichment(panel, f"top1_{c}", "repName", c, MIN_SUBFAMILY_UNIVERSE) for c in score_map],
        ignore_index=True,
    )

    overlap_rows = []
    comparators = list(score_map)
    for left in comparators:
        for right in comparators:
            inter = len(top_sets[left] & top_sets[right])
            union = len(top_sets[left] | top_sets[right])
            overlap_rows.append(
                {
                    "comparator_a": left,
                    "comparator_b": right,
                    "top1_n_a": VALID_TOP1_K,
                    "top1_n_b": VALID_TOP1_K,
                    "intersection_n": inter,
                    "jaccard": inter / union if union else float("nan"),
                    "fraction_of_a_retained_in_b": inter / VALID_TOP1_K,
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    return summary, family, subfamily, overlap


# =============================================================================
# 7. Matched-background control for Final-FER Top1 divergence
# =============================================================================


def _quantile_bin(series: pd.Series, q: int = 10) -> pd.Series:
    return pd.qcut(series, q=q, labels=False, duplicates="drop").astype("Int64")


def build_final_top1_matched_background(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = table.copy()
    work["match_length_bin"] = _quantile_bin(pd.to_numeric(work["length"], errors="raise"))
    work["match_gc_bin"] = _quantile_bin(pd.to_numeric(work["gc_content"], errors="raise"))
    work["match_tss_bin"] = _quantile_bin(
        np.log1p(pd.to_numeric(work["dist_to_tss"], errors="raise").abs())
    )

    top = work.loc[work["top1_FINAL_FER"].astype(bool)].copy()
    bg = work.loc[~work["top1_FINAL_FER"].astype(bool)].copy()

    rng = np.random.default_rng(MATCH_SEED)
    bg = bg.sample(frac=1.0, random_state=MATCH_SEED).reset_index(drop=True)
    unused = np.ones(len(bg), dtype=bool)

    # Fast lookup maps. Each list is already randomized by the deterministic bg shuffle.
    levels = [
        ["repFamily", "match_length_bin", "match_gc_bin", "match_tss_bin"],
        ["repFamily", "match_length_bin", "match_gc_bin"],
        ["repFamily", "match_length_bin"],
        ["repFamily"],
    ]
    maps: list[dict[tuple[Any, ...], list[int]]] = []
    cursors: list[dict[tuple[Any, ...], int]] = []
    for cols in levels:
        mapping: dict[tuple[Any, ...], list[int]] = {}
        for idx, row in bg[cols].iterrows():
            key = tuple(row[c] for c in cols)
            mapping.setdefault(key, []).append(int(idx))
        maps.append(mapping)
        cursors.append({})

    def pop_unused(level_index: int, key: tuple[Any, ...]) -> int | None:
        values = maps[level_index].get(key, [])
        cursor = cursors[level_index].get(key, 0)
        while cursor < len(values) and not unused[values[cursor]]:
            cursor += 1
        cursors[level_index][key] = cursor
        if cursor >= len(values):
            return None
        idx = values[cursor]
        unused[idx] = False
        cursors[level_index][key] = cursor + 1
        return idx

    chosen: list[int] = []
    tiers: list[str] = []
    top_order = top.sample(frac=1.0, random_state=MATCH_SEED)
    for _, row in top_order.iterrows():
        selected_idx = None
        selected_tier = None
        for level_index, cols in enumerate(levels):
            key = tuple(row[c] for c in cols)
            selected_idx = pop_unused(level_index, key)
            if selected_idx is not None:
                selected_tier = "+".join(cols)
                break
        if selected_idx is None:
            available = np.flatnonzero(unused)
            selected_idx = int(rng.choice(available))
            unused[selected_idx] = False
            selected_tier = "global_fallback"
        chosen.append(selected_idx)
        tiers.append(selected_tier)

    matched = bg.iloc[chosen].copy().reset_index(drop=True)
    top_order = top_order.reset_index(drop=True)

    pairs = pd.DataFrame(
        {
            "top1_herv_internal_id": top_order[ID].astype(str),
            "matched_background_herv_internal_id": matched[ID].astype(str),
            "matching_tier": tiers,
            "top1_repFamily": top_order["repFamily"].astype(str),
            "background_repFamily": matched["repFamily"].astype(str),
            "top1_length": top_order["length"].to_numpy(),
            "background_length": matched["length"].to_numpy(),
            "top1_gc_content": top_order["gc_content"].to_numpy(),
            "background_gc_content": matched["gc_content"].to_numpy(),
            "top1_abs_dist_to_tss": top_order["dist_to_tss"].abs().to_numpy(),
            "background_abs_dist_to_tss": matched["dist_to_tss"].abs().to_numpy(),
            "top1_repeat_divergence": top_order["repeat_divergence"].to_numpy(),
            "background_repeat_divergence": matched["repeat_divergence"].to_numpy(),
            "top1_within_family_divergence_percentile": top_order["divergence_percentile_within_family"].to_numpy(),
            "background_within_family_divergence_percentile": matched["divergence_percentile_within_family"].to_numpy(),
        }
    )

    comparison_rows = []
    metrics = [
        ("repeat_divergence", top_order["repeat_divergence"], matched["repeat_divergence"]),
        (
            "within_family_divergence_percentile",
            top_order["divergence_percentile_within_family"],
            matched["divergence_percentile_within_family"],
        ),
        ("length", top_order["length"], matched["length"]),
        ("gc_content", top_order["gc_content"], matched["gc_content"]),
        ("abs_dist_to_tss", top_order["dist_to_tss"].abs(), matched["dist_to_tss"].abs()),
    ]
    for metric, x, y in metrics:
        test = safe_mannwhitney(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        comparison_rows.append(
            {
                "metric": metric,
                "top1_n": len(x),
                "matched_background_n": len(y),
                "top1_median": float(np.median(x)),
                "matched_background_median": float(np.median(y)),
                "median_delta_top_minus_matched": float(np.median(x) - np.median(y)),
                "mannwhitney_p": test["p"],
                "rank_biserial_top_greater": test["rank_biserial_x_greater"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison["fdr_bh_across_matched_metrics"] = bh_adjust(comparison["mannwhitney_p"])

    balance = pd.DataFrame(
        [
            {
                "matching_seed": MATCH_SEED,
                "top1_n": TOP1_K,
                "matched_n": len(matched),
                "exact_family_match_rate": float((top_order["repFamily"].to_numpy() == matched["repFamily"].to_numpy()).mean()),
                "median_abs_length_difference": float(np.median(np.abs(top_order["length"].to_numpy(dtype=float) - matched["length"].to_numpy(dtype=float)))),
                "median_abs_gc_difference": float(np.median(np.abs(top_order["gc_content"].to_numpy(dtype=float) - matched["gc_content"].to_numpy(dtype=float)))),
                "median_abs_log1p_tss_difference": float(np.median(np.abs(
                    np.log1p(top_order["dist_to_tss"].abs().to_numpy(dtype=float))
                    - np.log1p(matched["dist_to_tss"].abs().to_numpy(dtype=float))
                ))),
                "tier_full_strata_n": int(sum(t == "repFamily+match_length_bin+match_gc_bin+match_tss_bin" for t in tiers)),
                "tier_family_length_gc_n": int(sum(t == "repFamily+match_length_bin+match_gc_bin" for t in tiers)),
                "tier_family_length_n": int(sum(t == "repFamily+match_length_bin" for t in tiers)),
                "tier_family_only_n": int(sum(t == "repFamily" for t in tiers)),
                "tier_global_fallback_n": int(sum(t == "global_fallback" for t in tiers)),
                "repeat_divergence_used_for_matching": False,
            }
        ]
    )
    return pairs, comparison, balance


# =============================================================================
# 8. Tumor-supported candidate evolutionary interpretation
# =============================================================================


def final_top1(table: pd.DataFrame) -> pd.DataFrame:
    result = table.loc[table["top1_FINAL_FER"].astype(bool)].copy()
    return result


def build_tumor_comparison(table: pd.DataFrame) -> pd.DataFrame:
    top = final_top1(table)
    candidate_mask = top["potential_reactivation_candidate"].astype(bool)
    supported = top.loc[candidate_mask]
    unsupported = top.loc[~candidate_mask]

    rows = []
    for metric in [
        "repeat_divergence",
        "divergence_percentile_within_family",
        "length",
    ]:
        x = supported[metric].to_numpy(dtype=np.float64)
        y = unsupported[metric].to_numpy(dtype=np.float64)
        test = safe_mannwhitney(x, y)
        rows.append(
            {
                "comparison_scope": "M23_final_Top1",
                "group_a": "potential_reactivation_candidate",
                "group_b": "Top1_without_active_tumor_histone_support",
                "metric": metric,
                "group_a_n": len(x),
                "group_b_n": len(y),
                "group_a_mean": float(np.mean(x)),
                "group_b_mean": float(np.mean(y)),
                "group_a_median": float(np.median(x)),
                "group_b_median": float(np.median(y)),
                "median_delta_a_minus_b": float(np.median(x) - np.median(y)),
                "mannwhitney_u": test["u"],
                "mannwhitney_two_sided_p": test["p"],
                "rank_biserial_a_greater": test["rank_biserial_x_greater"],
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_bh_across_three_metrics"] = bh_adjust(
        result["mannwhitney_two_sided_p"]
    )
    return result


def build_tumor_family_enrichment(table: pd.DataFrame) -> pd.DataFrame:
    top = final_top1(table)
    selected_col = "potential_reactivation_candidate"
    selected_total = int(top[selected_col].astype(bool).sum())
    nonselected_total = len(top) - selected_total
    rows = []
    for level, group_col, min_n in [
        ("family", "repFamily", 1),
        ("subfamily", "repName", MIN_SUBFAMILY_UNIVERSE),
    ]:
        level_rows = []
        for group, subset in top.groupby(group_col, dropna=False, sort=True):
            universe_n = len(subset)
            if universe_n < min_n:
                continue
            a = int(subset[selected_col].astype(bool).sum())
            c = universe_n - a
            b = selected_total - a
            d = nonselected_total - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            candidate_share = a / selected_total if selected_total else float("nan")
            top1_share = universe_n / len(top)
            level_rows.append(
                {
                    "level": level,
                    "group": "NA" if pd.isna(group) else str(group),
                    "candidate_count": a,
                    "candidate_share": candidate_share,
                    "final_top1_count": universe_n,
                    "final_top1_share": top1_share,
                    "fold_enrichment_candidate_vs_final_top1": (
                        candidate_share / top1_share if top1_share > 0 else float("nan")
                    ),
                    "odds_ratio": float(odds),
                    "fisher_two_sided_p": float(p),
                }
            )
        level_frame = pd.DataFrame(level_rows)
        if not level_frame.empty:
            level_frame["fdr_bh_within_level"] = bh_adjust(
                level_frame["fisher_two_sided_p"]
            )
            rows.append(level_frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["level", "fdr_bh_within_level", "fold_enrichment_candidate_vs_final_top1"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def build_breadth_gradient(table: pd.DataFrame) -> pd.DataFrame:
    top = final_top1(table)
    rows = []
    for breadth in range(0, 6):
        subset = top.loc[
            pd.to_numeric(top["tumor_active_support_breadth"], errors="raise").eq(breadth)
        ]
        if subset.empty:
            continue
        rows.append(
            {
                "tumor_active_support_breadth": breadth,
                "rows": len(subset),
                "fraction_of_final_top1": len(subset) / len(top),
                "median_repeat_divergence": float(subset["repeat_divergence"].median()),
                "mean_repeat_divergence": float(subset["repeat_divergence"].mean()),
                "median_within_family_divergence_percentile": float(
                    subset["divergence_percentile_within_family"].median()
                ),
                "mean_within_family_divergence_percentile": float(
                    subset["divergence_percentile_within_family"].mean()
                ),
                "median_length": float(subset["length"].median()),
                "ERVK_share": float(subset["is_ERVK_family"].mean()),
                "ERV1_share": float(subset["is_ERV1_family"].mean()),
                "median_final_rank": float(
                    pd.to_numeric(
                        subset["herv_functional_priority_rank"], errors="raise"
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cell_line_summary(table: pd.DataFrame) -> pd.DataFrame:
    top = final_top1(table)
    rows = []
    for eid in TUMOR_EIDS:
        col = f"tumor_{eid}_any_active_support"
        supported = top.loc[top[col].astype(bool)]
        unsupported = top.loc[~top[col].astype(bool)]
        pct_test = safe_mannwhitney(
            supported["divergence_percentile_within_family"].to_numpy(dtype=float),
            unsupported["divergence_percentile_within_family"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "eid": eid,
                "cell_line": CELL_LINE_NAMES[eid],
                "final_top1_n": len(top),
                "active_supported_n": len(supported),
                "active_supported_rate": len(supported) / len(top),
                "unsupported_n": len(unsupported),
                "supported_median_repeat_divergence": (
                    float(supported["repeat_divergence"].median()) if len(supported) else float("nan")
                ),
                "unsupported_median_repeat_divergence": (
                    float(unsupported["repeat_divergence"].median()) if len(unsupported) else float("nan")
                ),
                "supported_median_within_family_divergence_percentile": (
                    float(supported["divergence_percentile_within_family"].median())
                    if len(supported)
                    else float("nan")
                ),
                "unsupported_median_within_family_divergence_percentile": (
                    float(unsupported["divergence_percentile_within_family"].median())
                    if len(unsupported)
                    else float("nan")
                ),
                "supported_median_length": (
                    float(supported["length"].median()) if len(supported) else float("nan")
                ),
                "unsupported_median_length": (
                    float(unsupported["length"].median()) if len(unsupported) else float("nan")
                ),
                "supported_ERVK_share": (
                    float(supported["is_ERVK_family"].mean()) if len(supported) else float("nan")
                ),
                "supported_ERV1_share": (
                    float(supported["is_ERV1_family"].mean()) if len(supported) else float("nan")
                ),
                "within_family_divergence_mannwhitney_p": pct_test["p"],
                "within_family_divergence_rank_biserial_supported_greater": pct_test[
                    "rank_biserial_x_greater"
                ],
            }
        )
    result = pd.DataFrame(rows)
    result["within_family_divergence_fdr_bh_across_cell_lines"] = bh_adjust(
        result["within_family_divergence_mannwhitney_p"]
    )
    return result


# =============================================================================
# 7. Top30 candidate evolutionary annotation
# =============================================================================


def build_top30(table: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    candidate = candidates.copy()
    candidate["herv_functional_priority_rank"] = pd.to_numeric(
        candidate["herv_functional_priority_rank"], errors="raise"
    ).astype(np.int64)
    candidate = candidate.sort_values(
        ["herv_functional_priority_rank", "tumor_active_support_breadth", ID],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top30 = candidate.head(30).copy()

    evo_cols = [
        ID,
        "family_universe_count",
        "subfamily_universe_count",
        "divergence_percentile_within_family",
        "relative_divergence_class_within_family",
        "is_ERVK_family",
        "is_ERV1_family",
        *[f"top1_{layer}" for layer in LAYER_ORDER],
    ]
    result = top30.merge(
        table[evo_cols],
        on=ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    path_labels = []
    for row in result[[f"top1_{layer}" for layer in LAYER_ORDER]].itertuples(
        index=False, name=None
    ):
        path_labels.append(
            " -> ".join(
                layer for layer, flag in zip(LAYER_ORDER, row) if bool(flag)
            )
            or "not_top1_in_deployable_layers"
        )
    result["top1_membership_path"] = path_labels
    result.insert(0, "M24_candidate_display_order", np.arange(1, len(result) + 1))
    return result


# =============================================================================
# 8. Human-readable report
# =============================================================================


def fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


# =============================================================================
# 9. Main
# =============================================================================


def step_25() -> None:
    prepare_output()


    atlas, projection, candidates, m05_validation = audit_and_load()
    table = build_analysis_table(atlas, projection)

    family, subfamily, transition = build_enrichment_outputs(table)
    layer_summary = build_layer_summary(table)
    (
        validation_evolution,
        validation_family,
        validation_subfamily,
        validation_overlap,
    ) = build_validation_sequence_comparator_panel(table, m05_validation)
    matched_background, matched_compare, matching_balance = (
        build_final_top1_matched_background(table)
    )
    top30 = build_top30(table, candidates)

    atomic_tsv(family, OUT_FAMILY)
    atomic_tsv(layer_summary, OUT_LAYER_SUMMARY)
    atomic_tsv(matched_background, OUT_MATCHED_BACKGROUND)
    atomic_tsv(matched_compare, OUT_MATCHED_COMPARE)
    atomic_tsv(matching_balance, OUT_MATCHING_BALANCE)
    atomic_tsv(top30, OUT_TOP30)


# --------------------------------------------------------------------------
# step 25b: Within-subfamily divergence comparison
# --------------------------------------------------------------------------

HERE__25b = Path(__file__).resolve().parent
MODELING__25b = (HERE__25b.parent / "Modeling").resolve()
RANKING = MODELING__25b / "output/23_generate_genomewide_herv_functional_ranking/23_herv_genomewide_functional_ranking.parquet"
OUT__25b = HERE__25b / "output/25_evolutionary_interpretation"
OUT_TSV = OUT__25b / "25_final_top1_within_subfamily_divergence_comparison.tsv"
TOP1_N = 4803


def step_25b() -> None:
    rank = pd.read_parquet(
        RANKING,
        columns=[
            "herv_internal_id",
            "repName",
            "repFamily",
            "repeat_divergence",
            "herv_functional_priority_rank",
        ],
    )
    rank["is_top1"] = (rank["herv_functional_priority_rank"] <= TOP1_N).astype(int)
    rank["div_pct_within_subfamily"] = rank.groupby("repName")["repeat_divergence"].rank(
        pct=True
    )
    rank["div_pct_within_family"] = rank.groupby("repFamily")[
        "repeat_divergence"
    ].rank(pct=True)

    rows = []
    for metric in [
        "repeat_divergence",
        "div_pct_within_family",
        "div_pct_within_subfamily",
    ]:
        top = rank.loc[rank["is_top1"].eq(1), metric]
        bg = rank.loc[rank["is_top1"].eq(0), metric]
        _, p = mannwhitneyu(top, bg, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "top1_n": int(len(top)),
                "background_n": int(len(bg)),
                "top1_median": float(top.median()),
                "background_median": float(bg.median()),
                "median_delta_top_minus_background": float(top.median() - bg.median()),
                "mannwhitney_p": float(p),
                "direction_top_greater": bool(top.median() > bg.median()),
            }
        )
    out = pd.DataFrame(rows)
    OUT__25b.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TSV, sep="\t", index=False)


# --------------------------------------------------------------------------
# step 26: Host regulatory mechanism interpretation
# --------------------------------------------------------------------------

# =============================================================================
# 1. Frozen paths / identities
# =============================================================================

HERE__26 = Path(__file__).resolve().parent
MODELING__26 = (HERE__26.parent / "Modeling").resolve()

SCRIPT__26 = "26_host_regulatory_mechanism_interpretation"
VERSION__26 = "round2_v1"

M00_DIR = MODELING__26 / "output/00_build_canonical_data"
M00_TABLE = M00_DIR / "00_canonical_herv_mef_table.parquet"

M23_DIR__26 = MODELING__26 / "output/23_generate_genomewide_herv_functional_ranking"
M23_ATLAS__26 = M23_DIR__26 / "23_herv_genomewide_functional_ranking.parquet"

M24_DIR__26 = MODELING__26 / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes"
M24_PROJECTION__26 = M24_DIR__26 / "24_tumor_projection_all_herv.parquet"
M24_CANDIDATES__26 = M24_DIR__26 / "24_potential_reactivation_candidates.tsv"

OUT__26 = HERE__26 / "output/26_host_regulatory_mechanism_interpretation"
OUT_PAIRS = OUT__26 / "26_matched_top1_pairs.tsv"

OUT_FEATURE_COMPARE = OUT__26 / "26_regulatory_feature_matched_comparison.tsv"
OUT_SOURCE_COMPARE = OUT__26 / "26_regulatory_source_matched_comparison.tsv"


OUT_CORRELATIONS = OUT__26 / "26_regulatory_tumor_support_correlations.tsv"
OUT_CANDIDATES = OUT__26 / "26_candidate_regulatory_annotation.tsv"
OUT_TOP30__26 = OUT__26 / "26_top30_regulatory_mechanism.tsv"


ID__26 = "herv_internal_id"
EXPECTED_N__26 = 480_306
TOP_FRAC__26 = 0.01
TOP1_K__26 = int(math.floor(EXPECTED_N__26 * TOP_FRAC__26))
EXPECTED_M24_CANDIDATES = 4_068
EXPECTED_TOP1_UNSUPPORTED = TOP1_K__26 - EXPECTED_M24_CANDIDATES
EXPECTED_GLOBAL__26 = "R4_ALL_COMPACT_bs64_epoch_24"
EXPECTED_PRIMARY__26 = "A3_ALL38_PRIMARY_epoch_01"
TUMOR_EIDS__26 = ["E114", "E115", "E117", "E118", "E123"]
CELL_LINE_NAMES__26 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

ACTIVE_MARKS = ("DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3")
REPRESSIVE_MARKS = ("H3K27me3", "H3K9me3")
ALL_MARKS = (*ACTIVE_MARKS, *REPRESSIVE_MARKS)

ROADMAP18 = [
    f"roadmap_{mark}_{suffix}"
    for mark in ALL_MARKS
    for suffix in ("sample_count", "max_overlap_bp", "total_overlap_bp")
]
MPRA8 = [
    "mpra_overlap_count",
    "mpra_max_overlap_bp",
    "mpra_total_overlap_bp",
    "mpra_active_overlap_count",
    "mpra_active_max_overlap_bp",
    "mpra_active_total_overlap_bp",
    "is_mpra_overlap",
    "is_mpra_active_supported",
]
CCRE6 = [
    "is_ccre_ca_overlap",
    "is_ccre_enhancer_like",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
    "ccre_els_overlap_count",
    "ccre_els_total_overlap_bp",
]
SERVE6 = [
    "serve_gtex_tissue_count",
    "serve_gtex_total_overlap_bp",
    "is_serve_gtex_supported",
    "serve_gtex_annotation_count",
    "serve_noncoding_overlap_count",
    "serve_noncoding_total_overlap_bp",
]
RAW38 = [*ROADMAP18, *MPRA8, *CCRE6, *SERVE6]

RAW_BINARY = {
    "is_mpra_overlap",
    "is_mpra_active_supported",
    "is_ccre_ca_overlap",
    "is_ccre_enhancer_like",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
    "is_serve_gtex_supported",
}

ROADMAP9 = [
    "gctx_roadmap_active_mark_presence_count",
    "gctx_roadmap_repressive_mark_presence_count",
    "gctx_roadmap_active_sample_count_sum",
    "gctx_roadmap_repressive_sample_count_sum",
    "gctx_roadmap_active_max_overlap_bp_max",
    "gctx_roadmap_repressive_max_overlap_bp_max",
    "gctx_roadmap_active_total_overlap_bp_log1p",
    "gctx_roadmap_repressive_total_overlap_bp_log1p",
    "gctx_roadmap_active_minus_repressive_logbp",
]
MPRA6 = [
    "gctx_mpra_is_overlap",
    "gctx_mpra_is_active_supported",
    "gctx_mpra_total_overlap_bp_log1p",
    "gctx_mpra_active_total_overlap_bp_log1p",
    "gctx_mpra_active_count_fraction",
    "gctx_mpra_active_bp_fraction",
]
CCRE_SERVE9 = [
    "gctx_ccre_is_ctcf_overlap",
    "gctx_ccre_is_pls_overlap",
    "gctx_ccre_els_overlap_count_log1p",
    "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_annotation_count_log1p",
    "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_overlap_count_log1p",
    "gctx_serve_noncoding_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]
COMPACT24 = [*ROADMAP9, *MPRA6, *CCRE_SERVE9]

COMPACT_BINARY = {
    "gctx_mpra_is_overlap",
    "gctx_mpra_is_active_supported",
    "gctx_ccre_is_ctcf_overlap",
    "gctx_ccre_is_pls_overlap",
}

M23_COLUMNS = [
    ID__26,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repClass",
    "repFamily",
    "repeat_divergence",
    "length",
    "gc_content",
    "dist_to_tss",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "score_final_global",
    "global_rank",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
]

M24_COLUMNS = [
    ID__26,
    "herv_functional_priority_rank",
    "m23_top1pct",
    "potential_reactivation_candidate",
    "tumor_epigenomic_support_score",
    "tumor_repressive_context_score",
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_repressive_support_breadth",
    "tumor_cell_line_support_class",
    "tumor_active_supported_eids",
]
for _eid__26 in TUMOR_EIDS__26:
    M24_COLUMNS.extend(
        [
            f"tumor_{_eid__26}_any_active_support",
            f"tumor_{_eid__26}_active_mark_count",
            f"tumor_{_eid__26}_repressive_mark_count",
        ]
    )

M00_COLUMNS = [ID__26, *RAW38]

# Predeclared matching variables. Regulatory evidence is intentionally absent.
MATCH_VARIABLES = [
    "herv_functional_priority_rank",
    "length",
    "gc_content",
    "log_abs_dist_to_tss",
]
MATCH_WEIGHTS = {
    "herv_functional_priority_rank": 1.0,
    "length": 0.50,
    "gc_content": 0.50,
    "log_abs_dist_to_tss": 0.50,
}

SOURCE_BINARY_COLUMNS = [
    "non_tumor_roadmap_active_supported",
    "non_tumor_roadmap_repressive_supported",
    "is_mpra_active_supported",
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
]

CORE_SOURCE_AXES = [
    "non_tumor_roadmap_active_supported",
    "is_mpra_active_supported",
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
]

CORRELATION_FEATURES = [
    "non_tumor_regulatory_source_breadth",
    "gctx_roadmap_active_mark_presence_count",
    "gctx_roadmap_repressive_mark_presence_count",
    "gctx_roadmap_active_minus_repressive_logbp",
    "gctx_mpra_active_count_fraction",
    "gctx_mpra_active_bp_fraction",
    "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]

CORRELATION_OUTCOMES = [
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_epigenomic_support_score",
]


# =============================================================================
# 2. Generic utilities
# =============================================================================


def prepare_output__26() -> None:
    if OUT__26.exists():
        shutil.rmtree(OUT__26)
    OUT__26.mkdir(parents=True, exist_ok=True)


def to_jsonable__26(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable__26(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable__26(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def atomic_json__26(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(to_jsonable__26(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_tsv__26(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def bh_adjust__26(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return out
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros_like(numerator, dtype=np.float64)
    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]
    return out


# =============================================================================
# 3. Load analysis table
# =============================================================================


def load_analysis_table() -> pd.DataFrame:
    atlas = pd.read_parquet(M23_ATLAS__26, columns=M23_COLUMNS)
    atlas[ID__26] = atlas[ID__26].astype(str)
    atlas["herv_functional_priority_rank"] = pd.to_numeric(
        atlas["herv_functional_priority_rank"], errors="raise"
    ).astype(np.int64)

    projection = pd.read_parquet(M24_PROJECTION__26, columns=M24_COLUMNS)
    projection[ID__26] = projection[ID__26].astype(str)

    evidence = pd.read_parquet(M00_TABLE, columns=M00_COLUMNS)
    evidence[ID__26] = evidence[ID__26].astype(str)

    for column in RAW38:
        evidence[column] = pd.to_numeric(evidence[column], errors="raise")

    projection = projection.drop(columns=["herv_functional_priority_rank"])
    table = atlas.merge(projection, on=ID__26, how="left", validate="one_to_one", sort=False)
    table = table.merge(evidence, on=ID__26, how="left", validate="one_to_one", sort=False)

    return table


# =============================================================================
# 4. Exact Compact24 and source-profile derivation
# =============================================================================

def derive_compact(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    active_sc = [f"roadmap_{m}_sample_count" for m in ACTIVE_MARKS]
    repressive_sc = [f"roadmap_{m}_sample_count" for m in REPRESSIVE_MARKS]
    active_mx = [f"roadmap_{m}_max_overlap_bp" for m in ACTIVE_MARKS]
    repressive_mx = [f"roadmap_{m}_max_overlap_bp" for m in REPRESSIVE_MARKS]
    active_tb = [f"roadmap_{m}_total_overlap_bp" for m in ACTIVE_MARKS]
    repressive_tb = [f"roadmap_{m}_total_overlap_bp" for m in REPRESSIVE_MARKS]

    AS = raw[active_sc].to_numpy(dtype=np.float64)
    RS = raw[repressive_sc].to_numpy(dtype=np.float64)
    AM = raw[active_mx].to_numpy(dtype=np.float64)
    RM = raw[repressive_mx].to_numpy(dtype=np.float64)
    AT = raw[active_tb].to_numpy(dtype=np.float64)
    RT = raw[repressive_tb].to_numpy(dtype=np.float64)

    active_total_bp = AT.sum(axis=1)
    repressive_total_bp = RT.sum(axis=1)

    out[ROADMAP9[0]] = (AS > 0).sum(axis=1)
    out[ROADMAP9[1]] = (RS > 0).sum(axis=1)
    out[ROADMAP9[2]] = AS.sum(axis=1)
    out[ROADMAP9[3]] = RS.sum(axis=1)
    out[ROADMAP9[4]] = AM.max(axis=1)
    out[ROADMAP9[5]] = RM.max(axis=1)
    out[ROADMAP9[6]] = np.log1p(active_total_bp)
    out[ROADMAP9[7]] = np.log1p(repressive_total_bp)
    out[ROADMAP9[8]] = np.log1p(active_total_bp) - np.log1p(repressive_total_bp)

    out[MPRA6[0]] = raw["is_mpra_overlap"].to_numpy(dtype=np.float64)
    out[MPRA6[1]] = raw["is_mpra_active_supported"].to_numpy(dtype=np.float64)
    out[MPRA6[2]] = np.log1p(raw["mpra_total_overlap_bp"].to_numpy(np.float64))
    out[MPRA6[3]] = np.log1p(
        raw["mpra_active_total_overlap_bp"].to_numpy(np.float64)
    )
    out[MPRA6[4]] = safe_fraction(
        raw["mpra_active_overlap_count"].to_numpy(np.float64),
        raw["mpra_overlap_count"].to_numpy(np.float64),
    )
    out[MPRA6[5]] = safe_fraction(
        raw["mpra_active_total_overlap_bp"].to_numpy(np.float64),
        raw["mpra_total_overlap_bp"].to_numpy(np.float64),
    )

    out[CCRE_SERVE9[0]] = raw["is_ccre_ctcf_overlap"].to_numpy(np.float64)
    out[CCRE_SERVE9[1]] = raw["is_ccre_pls_overlap"].to_numpy(np.float64)
    out[CCRE_SERVE9[2]] = np.log1p(
        raw["ccre_els_overlap_count"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[3]] = np.log1p(
        raw["ccre_els_total_overlap_bp"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[4]] = np.log1p(
        raw["serve_gtex_annotation_count"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[5]] = np.log1p(
        raw["serve_gtex_total_overlap_bp"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[6]] = np.log1p(
        raw["serve_noncoding_overlap_count"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[7]] = np.log1p(
        raw["serve_noncoding_total_overlap_bp"].to_numpy(np.float64)
    )
    out[CCRE_SERVE9[8]] = safe_fraction(
        raw["serve_noncoding_total_overlap_bp"].to_numpy(np.float64),
        raw["serve_gtex_total_overlap_bp"].to_numpy(np.float64),
    )

    return out


def add_regulatory_profiles(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    compact = derive_compact(table[RAW38])
    for column in COMPACT24:
        table[column] = compact[column].to_numpy(dtype=np.float64)

    table["log_abs_dist_to_tss"] = np.log1p(
        np.abs(pd.to_numeric(table["dist_to_tss"], errors="raise").to_numpy(np.float64))
    )

    active_sc = [f"roadmap_{m}_sample_count" for m in ACTIVE_MARKS]
    repressive_sc = [f"roadmap_{m}_sample_count" for m in REPRESSIVE_MARKS]
    table["non_tumor_roadmap_active_supported"] = (
        table[active_sc].to_numpy(dtype=np.float64) > 0
    ).any(axis=1).astype(np.int8)
    table["non_tumor_roadmap_repressive_supported"] = (
        table[repressive_sc].to_numpy(dtype=np.float64) > 0
    ).any(axis=1).astype(np.int8)

    for column in SOURCE_BINARY_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="raise").astype(np.int8)

    table["non_tumor_regulatory_source_breadth"] = table[
        CORE_SOURCE_AXES
    ].sum(axis=1).astype(np.int8)

    table["non_tumor_regulatory_profile"] = (
        "R" + table["non_tumor_roadmap_active_supported"].astype(str)
        + "_M" + table["is_mpra_active_supported"].astype(str)
        + "_C" + table["is_ccre_enhancer_like"].astype(str)
        + "_S" + table["is_serve_gtex_supported"].astype(str)
    )

    active_bp = np.expm1(table["gctx_roadmap_active_total_overlap_bp_log1p"].to_numpy())
    repressive_bp = np.expm1(
        table["gctx_roadmap_repressive_total_overlap_bp_log1p"].to_numpy()
    )
    state = np.full(len(table), "none", dtype=object)
    has_active = table["non_tumor_roadmap_active_supported"].to_numpy(dtype=bool)
    has_rep = table["non_tumor_roadmap_repressive_supported"].to_numpy(dtype=bool)
    state[has_active & ~has_rep] = "active_only"
    state[~has_active & has_rep] = "repressive_only"
    mixed = has_active & has_rep
    state[mixed & (active_bp > repressive_bp)] = "mixed_active_dominant"
    state[mixed & ~(active_bp > repressive_bp)] = "mixed_repressive_or_equal"
    table["non_tumor_roadmap_state_class"] = state

    return table


# =============================================================================
# 5. Deterministic Top1 supported-vs-unsupported matching
# =============================================================================

def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr
    std = float(np.std(values))
    return std if std > 0 else 1.0


def build_matched_pairs(top1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = top1.loc[top1["potential_reactivation_candidate"].astype(bool)].copy()
    unsupported = top1.loc[~top1["potential_reactivation_candidate"].astype(bool)].copy()

    top1 = top1.copy()
    scales = {
        column: robust_scale(top1[column].to_numpy(dtype=np.float64))
        for column in MATCH_VARIABLES
    }

    supported_by_family: dict[str, list[int]] = {}
    for idx, family in zip(supported.index, supported["repFamily"].astype(str)):
        supported_by_family.setdefault(family, []).append(int(idx))

    # Match hardest/rarest unsupported families first, then by frozen rank.
    family_avail = supported["repFamily"].astype(str).value_counts().to_dict()
    unsupported["_family_available"] = unsupported["repFamily"].astype(str).map(
        family_avail
    ).fillna(0).astype(int)
    unsupported = unsupported.sort_values(
        ["_family_available", "herv_functional_priority_rank", ID__26],
        ascending=[True, True, True],
        kind="mergesort",
    )

    unused: set[int] = set(int(i) for i in supported.index)
    rows: list[dict[str, Any]] = []

    for pair_id, (_, anchor) in enumerate(unsupported.iterrows(), start=1):
        family = str(anchor["repFamily"])
        same_family = [i for i in supported_by_family.get(family, []) if i in unused]
        exact_family = bool(same_family)
        candidates = same_family if same_family else sorted(unused)

        cand = supported.loc[candidates]
        cost = np.zeros(len(cand), dtype=np.float64)
        for column in MATCH_VARIABLES:
            diff = np.abs(
                cand[column].to_numpy(dtype=np.float64) - float(anchor[column])
            ) / scales[column]
            cost += MATCH_WEIGHTS[column] * diff

        rank_diff = np.abs(
            cand["herv_functional_priority_rank"].to_numpy(dtype=np.float64)
            - float(anchor["herv_functional_priority_rank"])
        )
        ids = cand[ID__26].astype(str).to_numpy()
        order = np.lexsort((ids, rank_diff, cost))
        chosen_idx = int(cand.index[int(order[0])])
        chosen = supported.loc[chosen_idx]
        unused.remove(chosen_idx)

        row: dict[str, Any] = {
            "pair_id": pair_id,
            "unsupported_herv_internal_id": str(anchor[ID__26]),
            "supported_herv_internal_id": str(chosen[ID__26]),
            "unsupported_repFamily": family,
            "supported_repFamily": str(chosen["repFamily"]),
            "exact_family_match": exact_family,
            "matching_cost": float(cost[int(order[0])]),
        }
        for column in MATCH_VARIABLES:
            row[f"unsupported_{column}"] = float(anchor[column])
            row[f"supported_{column}"] = float(chosen[column])
            row[f"delta_supported_minus_unsupported_{column}"] = (
                float(chosen[column]) - float(anchor[column])
            )
        rows.append(row)

    pairs = pd.DataFrame(rows)

    balance_rows: list[dict[str, Any]] = []
    for column in MATCH_VARIABLES:
        u = pairs[f"unsupported_{column}"].to_numpy(dtype=np.float64)
        s = pairs[f"supported_{column}"].to_numpy(dtype=np.float64)
        diff = s - u
        balance_rows.append(
            {
                "matching_variable": column,
                "unsupported_median": float(np.median(u)),
                "supported_median": float(np.median(s)),
                "median_pair_delta_supported_minus_unsupported": float(np.median(diff)),
                "mean_abs_pair_difference": float(np.mean(np.abs(diff))),
                "scale_used_for_cost": float(scales[column]),
                "matching_weight": float(MATCH_WEIGHTS[column]),
            }
        )

    balance = pd.DataFrame(balance_rows)
    balance["exact_family_match_rate"] = float(pairs["exact_family_match"].mean())
    balance["global_fallback_pairs"] = int((~pairs["exact_family_match"]).sum())
    balance["pairs"] = len(pairs)
    return pairs, balance


# =============================================================================
# 6. Paired regulatory comparisons
# =============================================================================

def paired_rank_biserial(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    total = positive + negative
    return 0.0 if total == 0 else (positive - negative) / total


def numeric_paired_row(
    feature: str,
    scope: str,
    supported_values: np.ndarray,
    unsupported_values: np.ndarray,
) -> dict[str, Any]:
    s = np.asarray(supported_values, dtype=np.float64)
    u = np.asarray(unsupported_values, dtype=np.float64)
    diff = s - u
    nonzero = int(np.sum(diff != 0))
    if nonzero == 0:
        pvalue = 1.0
        statistic = 0.0
    else:
        try:
            result = wilcoxon(s, u, zero_method="wilcox", alternative="two-sided")
            statistic = float(result.statistic)
            pvalue = float(result.pvalue)
        except Exception:
            statistic = float("nan")
            pvalue = float("nan")
    return {
        "feature": feature,
        "scope": scope,
        "feature_type": "numeric",
        "pairs": len(s),
        "supported_median": float(np.median(s)),
        "unsupported_median": float(np.median(u)),
        "median_pair_delta_supported_minus_unsupported": float(np.median(diff)),
        "mean_pair_delta_supported_minus_unsupported": float(np.mean(diff)),
        "nonzero_pair_count": nonzero,
        "paired_rank_biserial": float(paired_rank_biserial(diff)),
        "test": "paired_wilcoxon_two_sided",
        "statistic": statistic,
        "pvalue": pvalue,
    }


def binary_paired_row(
    feature: str,
    scope: str,
    supported_values: np.ndarray,
    unsupported_values: np.ndarray,
) -> dict[str, Any]:
    s = np.asarray(supported_values, dtype=np.int8)
    u = np.asarray(unsupported_values, dtype=np.int8)
    s1_u0 = int(np.sum((s == 1) & (u == 0)))
    s0_u1 = int(np.sum((s == 0) & (u == 1)))
    discordant = s1_u0 + s0_u1
    if discordant == 0:
        pvalue = 1.0
    else:
        pvalue = float(binomtest(s1_u0, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "feature": feature,
        "scope": scope,
        "feature_type": "binary",
        "pairs": len(s),
        "supported_rate": float(np.mean(s)),
        "unsupported_rate": float(np.mean(u)),
        "rate_delta_supported_minus_unsupported": float(np.mean(s) - np.mean(u)),
        "supported1_unsupported0": s1_u0,
        "supported0_unsupported1": s0_u1,
        "discordant_pairs": discordant,
        "matched_odds_ratio_haldane": float((s1_u0 + 0.5) / (s0_u1 + 0.5)),
        "test": "exact_discordant_pair_binomial_two_sided",
        "pvalue": pvalue,
    }


def build_paired_comparisons(
    table: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = table.set_index(ID__26, drop=False)
    supported_ids = pairs["supported_herv_internal_id"].astype(str).tolist()
    unsupported_ids = pairs["unsupported_herv_internal_id"].astype(str).tolist()

    feature_rows: list[dict[str, Any]] = []
    scopes = [
        (
            "raw38",
            [f for f in RAW38 if f not in RAW_BINARY],
            [f for f in RAW38 if f in RAW_BINARY],
        ),
        (
            "compact24",
            [f for f in COMPACT24 if f not in COMPACT_BINARY],
            [f for f in COMPACT24 if f in COMPACT_BINARY],
        ),
    ]

    for scope, numeric_features, binary_features in scopes:
        start = len(feature_rows)
        for feature in numeric_features:
            feature_rows.append(
                numeric_paired_row(
                    feature,
                    scope,
                    lookup.loc[supported_ids, feature].to_numpy(),
                    lookup.loc[unsupported_ids, feature].to_numpy(),
                )
            )
        for feature in binary_features:
            feature_rows.append(
                binary_paired_row(
                    feature,
                    scope,
                    lookup.loc[supported_ids, feature].to_numpy(),
                    lookup.loc[unsupported_ids, feature].to_numpy(),
                )
            )
        stop = len(feature_rows)
        q = bh_adjust__26([row["pvalue"] for row in feature_rows[start:stop]])
        for row, fdr in zip(feature_rows[start:stop], q):
            row["fdr_bh_within_scope"] = float(fdr)

    feature_compare = pd.DataFrame(feature_rows)
    feature_compare = feature_compare.sort_values(
        ["scope", "fdr_bh_within_scope", "pvalue", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)

    source_rows: list[dict[str, Any]] = []
    for feature in SOURCE_BINARY_COLUMNS:
        source_rows.append(
            binary_paired_row(
                feature,
                "predefined_regulatory_source",
                lookup.loc[supported_ids, feature].to_numpy(),
                lookup.loc[unsupported_ids, feature].to_numpy(),
            )
        )
    q = bh_adjust__26([row["pvalue"] for row in source_rows])
    for row, fdr in zip(source_rows, q):
        row["fdr_bh"] = float(fdr)
    source_compare = pd.DataFrame(source_rows).sort_values(
        ["fdr_bh", "pvalue", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    return feature_compare, source_compare


# =============================================================================
# 7. Breadth/profile summaries and external support correlations
# =============================================================================

def build_breadth_summary(top1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for breadth in range(5):
        group = top1.loc[top1["non_tumor_regulatory_source_breadth"].eq(breadth)]
        if group.empty:
            continue
        row: dict[str, Any] = {
            "non_tumor_regulatory_source_breadth": breadth,
            "top1_rows": len(group),
            "top1_fraction": float(len(group) / len(top1)),
            "tumor_supported_rows": int(group["potential_reactivation_candidate"].astype(bool).sum()),
            "tumor_supported_rate": float(group["potential_reactivation_candidate"].astype(bool).mean()),
            "multi_cell_line_supported_rows": int(
                group["tumor_active_support_breadth"].astype(int).ge(2).sum()
            ),
            "multi_cell_line_supported_rate": float(
                group["tumor_active_support_breadth"].astype(int).ge(2).mean()
            ),
            "median_tumor_active_support_breadth": float(
                group["tumor_active_support_breadth"].median()
            ),
            "mean_tumor_active_support_breadth": float(
                group["tumor_active_support_breadth"].mean()
            ),
            "median_tumor_epigenomic_support_score": float(
                group["tumor_epigenomic_support_score"].median()
            ),
            "median_final_rank": float(group["herv_functional_priority_rank"].median()),
        }
        for eid in TUMOR_EIDS__26:
            row[f"{eid}_active_support_rate"] = float(
                group[f"tumor_{eid}_any_active_support"].astype(bool).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_profile_patterns(top1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pattern, group in top1.groupby(
        "non_tumor_regulatory_profile", sort=True, observed=True
    ):
        rows.append(
            {
                "regulatory_profile": str(pattern),
                "roadmap_active": int(group["non_tumor_roadmap_active_supported"].iloc[0]),
                "mpra_active": int(group["is_mpra_active_supported"].iloc[0]),
                "ccre_enhancer_like": int(group["is_ccre_enhancer_like"].iloc[0]),
                "serve_gtex_supported": int(group["is_serve_gtex_supported"].iloc[0]),
                "source_breadth": int(group["non_tumor_regulatory_source_breadth"].iloc[0]),
                "top1_rows": len(group),
                "top1_fraction": float(len(group) / len(top1)),
                "tumor_supported_rows": int(group["potential_reactivation_candidate"].astype(bool).sum()),
                "tumor_supported_rate": float(group["potential_reactivation_candidate"].astype(bool).mean()),
                "multi_cell_line_supported_rate": float(
                    group["tumor_active_support_breadth"].astype(int).ge(2).mean()
                ),
                "median_tumor_active_support_breadth": float(
                    group["tumor_active_support_breadth"].median()
                ),
                "median_tumor_epigenomic_support_score": float(
                    group["tumor_epigenomic_support_score"].median()
                ),
                "median_final_rank": float(group["herv_functional_priority_rank"].median()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["predeclared_descriptive_min_n_20"] = frame["top1_rows"].ge(20)
    return frame.sort_values(
        ["source_breadth", "tumor_supported_rate", "top1_rows", "regulatory_profile"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_correlations(top1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in CORRELATION_FEATURES:
        x = pd.to_numeric(top1[feature], errors="raise").to_numpy(np.float64)
        for outcome in CORRELATION_OUTCOMES:
            y = pd.to_numeric(top1[outcome], errors="raise").to_numpy(np.float64)
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                rho, pvalue = float("nan"), float("nan")
            else:
                result = spearmanr(x, y)
                rho, pvalue = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "feature": feature,
                    "external_tumor_outcome": outcome,
                    "rows": len(top1),
                    "spearman_rho": rho,
                    "pvalue": pvalue,
                }
            )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = bh_adjust__26(frame["pvalue"].tolist())
    return frame.sort_values(
        ["fdr_bh", "pvalue", "feature", "external_tumor_outcome"],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# 8. Candidate annotation
# =============================================================================

def build_candidate_annotations(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        ID__26,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repFamily",
        "nearest_gene_id",
        "nearest_gene_name",
        "nearest_gene_type",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "score_final_global",
        "global_rank",
        "non_tumor_regulatory_source_breadth",
        "non_tumor_regulatory_profile",
        "non_tumor_roadmap_state_class",
        "non_tumor_roadmap_active_supported",
        "non_tumor_roadmap_repressive_supported",
        "is_mpra_active_supported",
        "is_ccre_enhancer_like",
        "is_ccre_ctcf_overlap",
        "is_ccre_pls_overlap",
        "is_serve_gtex_supported",
        "gctx_roadmap_active_mark_presence_count",
        "gctx_roadmap_repressive_mark_presence_count",
        "gctx_roadmap_active_sample_count_sum",
        "gctx_roadmap_repressive_sample_count_sum",
        "gctx_roadmap_active_minus_repressive_logbp",
        "gctx_mpra_active_count_fraction",
        "gctx_mpra_active_bp_fraction",
        "gctx_ccre_els_overlap_count_log1p",
        "gctx_ccre_els_total_overlap_bp_log1p",
        "gctx_serve_annotation_count_log1p",
        "gctx_serve_total_overlap_bp_log1p",
        "gctx_serve_noncoding_bp_fraction",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "tumor_active_support_breadth",
        "tumor_active_dominant_breadth",
        "tumor_repressive_support_breadth",
        "tumor_cell_line_support_class",
        "tumor_active_supported_eids",
    ]
    for eid in TUMOR_EIDS__26:
        columns.extend(
            [
                f"tumor_{eid}_any_active_support",
                f"tumor_{eid}_active_mark_count",
                f"tumor_{eid}_repressive_mark_count",
            ]
        )

    candidate = table.loc[table["potential_reactivation_candidate"].astype(bool), columns].copy()
    candidate = candidate.sort_values(
        ["herv_functional_priority_rank", ID__26], kind="mergesort"
    ).reset_index(drop=True)
    top30 = candidate.head(30).copy()
    top30.insert(0, "top30_order", np.arange(1, len(top30) + 1, dtype=np.int64))
    return candidate, top30


# =============================================================================
# 9. Report
# =============================================================================


# =============================================================================
# 10. Main
# =============================================================================

def step_26() -> None:
    prepare_output__26()


    table = add_regulatory_profiles(load_analysis_table())
    top1 = table.loc[table["herv_functional_priority_rank"].le(TOP1_K__26)].copy()
    top1 = top1.sort_values(["herv_functional_priority_rank", ID__26], kind="mergesort")

    pairs, balance = build_matched_pairs(top1)
    feature_compare, source_compare = build_paired_comparisons(table, pairs)
    correlations = build_correlations(top1)
    candidates, top30 = build_candidate_annotations(table)

    atomic_tsv__26(pairs, OUT_PAIRS)
    atomic_tsv__26(feature_compare, OUT_FEATURE_COMPARE)
    atomic_tsv__26(source_compare, OUT_SOURCE_COMPARE)
    atomic_tsv__26(correlations, OUT_CORRELATIONS)
    atomic_tsv__26(candidates, OUT_CANDIDATES)
    atomic_tsv__26(top30, OUT_TOP30__26)


# --------------------------------------------------------------------------
# step 27: Reference-to-tumor context interpretation
# --------------------------------------------------------------------------

# =============================================================================
# 1. Frozen paths / identities
# =============================================================================

HERE__27 = Path(__file__).resolve().parent
MODELING__27 = (HERE__27.parent / "Modeling").resolve()
DP = (HERE__27.parent / "Data_Preprocessing").resolve()

SCRIPT__27 = "27_reference_to_tumor_context_interpretation"
VERSION__27 = "round2_v1"

M00_DIR__27 = MODELING__27 / "output/00_build_canonical_data"
M00_TABLE__27 = M00_DIR__27 / "00_canonical_herv_mef_table.parquet"

M23_DIR__27 = MODELING__27 / "output/23_generate_genomewide_herv_functional_ranking"
M23_ATLAS__27 = M23_DIR__27 / "23_herv_genomewide_functional_ranking.parquet"

M24_DIR__27 = MODELING__27 / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes"
M24_PROJECTION__27 = M24_DIR__27 / "24_tumor_projection_all_herv.parquet"

DP04_DIR = DP / "output/04_roadmap_manifests"
DP04_SUMMARY = DP04_DIR / "roadmap_manifest_summary_by_set.tsv"

OUT__27 = HERE__27 / "output/27_reference_to_tumor_context_interpretation"

OUT_STATE_TRANS = OUT__27 / "27_reference_tumor_state_transition.tsv"
OUT_BREADTH = OUT__27 / "27_cell_line_support_breadth.tsv"
OUT_PATTERNS = OUT__27 / "27_cell_line_support_patterns.tsv"
OUT_SIMILARITY = OUT__27 / "27_cell_line_pairwise_similarity.tsv"
OUT_SPECIFIC = OUT__27 / "27_cell_line_specific_candidates.tsv"

OUT_BALANCE = OUT__27 / "27_cell_line_matching_balance.tsv"
OUT_MATCHED = OUT__27 / "27_cell_line_matched_regulatory_comparison.tsv"
OUT_TOP1 = OUT__27 / "27_top1_cell_context_annotation.tsv"
OUT_TOP30__27 = OUT__27 / "27_top30_cell_context_annotation.tsv"
OUT_MANIFEST = OUT__27 / "27_reference_to_tumor_context_manifest.json"


ID__27 = "herv_internal_id"
EXPECTED_N__27 = 480_306
TOP_FRAC__27 = 0.01
TOP1_K__27 = int(math.floor(EXPECTED_N__27 * TOP_FRAC__27))
EXPECTED_GLOBAL__27 = "R4_ALL_COMPACT_bs64_epoch_24"
EXPECTED_PRIMARY__27 = "A3_ALL38_PRIMARY_epoch_01"

TUMOR_EIDS__27 = ["E114", "E115", "E117", "E118", "E123"]
CELL_LINE_NAMES__27 = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

ACTIVE_MARKS__27 = ["H3K27ac", "H3K4me1", "H3K4me3"]
REPRESSIVE_MARKS__27 = ["H3K27me3", "H3K9me3"]
COMPARABLE_MARKS = [*ACTIVE_MARKS__27, *REPRESSIVE_MARKS__27]

EXPECTED_NON_TUMOR_MARK_FILES = {
    "H3K27ac": 93,
    "H3K4me1": 122,
    "H3K4me3": 122,
    "H3K27me3": 122,
    "H3K9me3": 122,
}

# Only columns explicitly required for M27. No FANTOM5 field is permitted.
M00_COLUMNS__27 = [
    ID__27,
    *[f"roadmap_{mark}_sample_count" for mark in COMPARABLE_MARKS],
    "is_mpra_active_supported",
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
]

M23_COLUMNS__27 = [
    ID__27,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repClass",
    "repFamily",
    "length",
    "gc_content",
    "dist_to_tss",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
]

M24_COLUMNS__27 = [
    ID__27,
    "herv_functional_priority_rank",
    "m23_top1pct",
    "potential_reactivation_candidate",
    "tumor_epigenomic_support_score",
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_repressive_support_breadth",
    "tumor_cell_line_support_class",
    "tumor_active_supported_eids",
]
for _eid__27 in TUMOR_EIDS__27:
    for _mark in COMPARABLE_MARKS:
        M24_COLUMNS__27.extend(
            [
                f"tumor_{_eid__27}_{_mark}_has",
                f"tumor_{_eid__27}_{_mark}_total_overlap_bp",
            ]
        )
    M24_COLUMNS__27.extend(
        [
            f"tumor_{_eid__27}_active_mark_count",
            f"tumor_{_eid__27}_repressive_mark_count",
            f"tumor_{_eid__27}_active_total_overlap_bp",
            f"tumor_{_eid__27}_repressive_total_overlap_bp",
            f"tumor_{_eid__27}_any_active_support",
            f"tumor_{_eid__27}_any_repressive_support",
            f"tumor_{_eid__27}_active_dominant",
        ]
    )

MATCH_VARIABLES__27 = [
    "herv_functional_priority_rank",
    "length",
    "gc_content",
    "log_abs_dist_to_tss",
]
MATCH_WEIGHTS__27 = {
    "herv_functional_priority_rank": 1.0,
    "length": 0.50,
    "gc_content": 0.50,
    "log_abs_dist_to_tss": 0.50,
}

NUMERIC_CONTEXT_FEATURES = [
    *[f"reference_{mark}_prevalence" for mark in COMPARABLE_MARKS],
    "reference_active_histone_prevalence_mean",
    "reference_repressive_histone_prevalence_mean",
    "reference_active_minus_repressive_prevalence",
]

BINARY_CONTEXT_FEATURES = [
    "reference_active_histone_any",
    "reference_repressive_histone_any",
    "is_mpra_active_supported",
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
]


# =============================================================================
# 2. Generic utilities
# =============================================================================


def prepare_output__27() -> None:
    if OUT__27.exists():
        shutil.rmtree(OUT__27)
    OUT__27.mkdir(parents=True, exist_ok=True)


def to_jsonable__27(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable__27(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable__27(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_tsv__27(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def atomic_json__27(payload: Any, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(to_jsonable__27(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pv = p[finite]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    tmp = np.empty(m, dtype=np.float64)
    tmp[order] = q
    out[np.flatnonzero(finite)] = tmp
    return out


# =============================================================================
# 3. Load mark denominators
# =============================================================================


def load_mark_denominators() -> dict[str, int]:
    dp04 = pd.read_csv(DP04_SUMMARY, sep="\t")
    non_tumor = dp04.loc[dp04["analysis_set"].eq("non_tumor_training")].copy()
    return {
        str(row.mark): int(row.peak_files)
        for row in non_tumor.itertuples(index=False)
        if str(row.mark) in COMPARABLE_MARKS
    }


def load_analysis_table__27(mark_denominators: dict[str, int]) -> pd.DataFrame:
    atlas = pd.read_parquet(M23_ATLAS__27, columns=M23_COLUMNS__27)
    atlas[ID__27] = atlas[ID__27].astype(str)

    projection = pd.read_parquet(M24_PROJECTION__27, columns=M24_COLUMNS__27)
    projection[ID__27] = projection[ID__27].astype(str)
    projection = projection.drop(columns=["herv_functional_priority_rank"])

    evidence = pd.read_parquet(M00_TABLE__27, columns=M00_COLUMNS__27)
    evidence[ID__27] = evidence[ID__27].astype(str)
    for mark in COMPARABLE_MARKS:
        col = f"roadmap_{mark}_sample_count"
        evidence[col] = pd.to_numeric(evidence[col], errors="raise").astype(np.int64)
        denom = int(mark_denominators[mark])
        evidence[f"reference_{mark}_prevalence"] = (
            evidence[col].astype(np.float64) / float(denom)
        )
        evidence[f"reference_{mark}_any"] = evidence[col].gt(0).astype(np.int8)

    for col in [
        "is_mpra_active_supported",
        "is_ccre_enhancer_like",
        "is_serve_gtex_supported",
        "is_ccre_ctcf_overlap",
        "is_ccre_pls_overlap",
    ]:
        evidence[col] = pd.to_numeric(evidence[col], errors="raise").astype(np.int8)

    evidence["reference_active_histone_any"] = (
        evidence[[f"reference_{m}_any" for m in ACTIVE_MARKS__27]]
        .astype(bool)
        .any(axis=1)
        .astype(np.int8)
    )
    evidence["reference_repressive_histone_any"] = (
        evidence[[f"reference_{m}_any" for m in REPRESSIVE_MARKS__27]]
        .astype(bool)
        .any(axis=1)
        .astype(np.int8)
    )
    evidence["reference_active_histone_prevalence_mean"] = evidence[
        [f"reference_{m}_prevalence" for m in ACTIVE_MARKS__27]
    ].mean(axis=1)
    evidence["reference_repressive_histone_prevalence_mean"] = evidence[
        [f"reference_{m}_prevalence" for m in REPRESSIVE_MARKS__27]
    ].mean(axis=1)
    evidence["reference_active_minus_repressive_prevalence"] = (
        evidence["reference_active_histone_prevalence_mean"]
        - evidence["reference_repressive_histone_prevalence_mean"]
    )

    keep_evidence = [
        ID__27,
        *[f"reference_{m}_prevalence" for m in COMPARABLE_MARKS],
        *[f"reference_{m}_any" for m in COMPARABLE_MARKS],
        "reference_active_histone_any",
        "reference_repressive_histone_any",
        "reference_active_histone_prevalence_mean",
        "reference_repressive_histone_prevalence_mean",
        "reference_active_minus_repressive_prevalence",
        "is_mpra_active_supported",
        "is_ccre_enhancer_like",
        "is_serve_gtex_supported",
        "is_ccre_ctcf_overlap",
        "is_ccre_pls_overlap",
    ]

    table = atlas.merge(projection, on=ID__27, how="left", validate="one_to_one", sort=False)
    table = table.merge(
        evidence[keep_evidence], on=ID__27, how="left", validate="one_to_one", sort=False
    )

    table["log_abs_dist_to_tss"] = np.log1p(
        np.abs(pd.to_numeric(table["dist_to_tss"], errors="raise").to_numpy(np.float64))
    )

    return table


# =============================================================================
# 4. Non-tumor-to-tumor support contrasts
# =============================================================================

def build_mark_transition(top1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for eid in TUMOR_EIDS__27:
        for mark in COMPARABLE_MARKS:
            ref = top1[f"reference_{mark}_any"].astype(bool).to_numpy()
            tumor = top1[f"tumor_{eid}_{mark}_has"].astype(bool).to_numpy()

            n00 = int((~ref & ~tumor).sum())
            n01 = int((~ref & tumor).sum())
            n10 = int((ref & ~tumor).sum())
            n11 = int((ref & tumor).sum())
            tumor_n = int(tumor.sum())
            ref_n = int(ref.sum())

            prev = top1[f"reference_{mark}_prevalence"].to_numpy(np.float64)
            rows.append(
                {
                    "eid": eid,
                    "cell_line": CELL_LINE_NAMES__27[eid],
                    "mark": mark,
                    "top1_n": len(top1),
                    "reference_supported_n": ref_n,
                    "reference_supported_rate": ref_n / len(top1),
                    "tumor_supported_n": tumor_n,
                    "tumor_supported_rate": tumor_n / len(top1),
                    "reference_absent_tumor_absent_n": n00,
                    "reference_absent_tumor_present_n": n01,
                    "reference_present_tumor_absent_n": n10,
                    "reference_present_tumor_present_n": n11,
                    "tumor_present_fraction_reference_absent": (
                        n01 / tumor_n if tumor_n else float("nan")
                    ),
                    "tumor_present_fraction_reference_present": (
                        n11 / tumor_n if tumor_n else float("nan")
                    ),
                    "median_reference_prevalence_tumor_present": (
                        float(np.median(prev[tumor])) if tumor_n else float("nan")
                    ),
                    "median_reference_prevalence_tumor_absent": (
                        float(np.median(prev[~tumor])) if int((~tumor).sum()) else float("nan")
                    ),
                    "interpretation_label_for_n01": (
                        "tumor-emergent relative to pooled non-tumor absence"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_state_transition(top1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_specs = [
        (
            "active_histone_state",
            top1["reference_active_histone_any"].astype(bool).to_numpy(),
            "any_active_support",
        ),
        (
            "repressive_histone_state",
            top1["reference_repressive_histone_any"].astype(bool).to_numpy(),
            "any_repressive_support",
        ),
    ]

    for eid in TUMOR_EIDS__27:
        for state_name, ref, tumor_suffix in state_specs:
            tumor = top1[f"tumor_{eid}_{tumor_suffix}"].astype(bool).to_numpy()
            n00 = int((~ref & ~tumor).sum())
            n01 = int((~ref & tumor).sum())
            n10 = int((ref & ~tumor).sum())
            n11 = int((ref & tumor).sum())
            rows.append(
                {
                    "eid": eid,
                    "cell_line": CELL_LINE_NAMES__27[eid],
                    "state": state_name,
                    "top1_n": len(top1),
                    "reference_absent_tumor_absent_n": n00,
                    "reference_absent_tumor_present_n": n01,
                    "reference_present_tumor_absent_n": n10,
                    "reference_present_tumor_present_n": n11,
                    "reference_absent_tumor_present_rate": n01 / len(top1),
                    "reference_present_tumor_present_rate": n11 / len(top1),
                    "tumor_supported_n": int(tumor.sum()),
                    "tumor_supported_fraction_reference_absent": (
                        n01 / int(tumor.sum()) if int(tumor.sum()) else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 5. Cell-line support architecture
# =============================================================================

def support_pattern(top1: pd.DataFrame) -> pd.Series:
    flags = [
        top1[f"tumor_{eid}_any_active_support"].astype(int).astype(str)
        for eid in TUMOR_EIDS__27
    ]
    out = flags[0]
    for series in flags[1:]:
        out = out + series
    return out


def build_breadth(top1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    breadth = top1["tumor_active_support_breadth"].astype(int)
    for value in range(0, 6):
        group = top1.loc[breadth.eq(value)]
        rows.append(
            {
                "tumor_active_support_breadth": value,
                "top1_n": len(group),
                "top1_rate": len(group) / len(top1),
                "median_final_rank": (
                    float(group["herv_functional_priority_rank"].median())
                    if len(group) else float("nan")
                ),
                "median_reference_active_histone_prevalence": (
                    float(group["reference_active_histone_prevalence_mean"].median())
                    if len(group) else float("nan")
                ),
                "median_reference_repressive_histone_prevalence": (
                    float(group["reference_repressive_histone_prevalence_mean"].median())
                    if len(group) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_patterns(top1: pd.DataFrame) -> pd.DataFrame:
    frame = top1[[ID__27, "tumor_active_support_breadth"]].copy()
    frame["support_pattern"] = support_pattern(top1).to_numpy()
    counts = (
        frame.groupby(["support_pattern", "tumor_active_support_breadth"], as_index=False)
        .size()
        .rename(columns={"size": "top1_n"})
    )
    counts["top1_rate"] = counts["top1_n"] / len(top1)
    pattern_names = {}
    for pattern in counts["support_pattern"].astype(str):
        names = [
            CELL_LINE_NAMES__27[eid]
            for eid, flag in zip(TUMOR_EIDS__27, pattern)
            if flag == "1"
        ]
        pattern_names[pattern] = ";".join(names) if names else "None"
    counts["supported_cell_lines"] = counts["support_pattern"].map(pattern_names)
    return counts.sort_values(
        ["top1_n", "support_pattern"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def build_pairwise_similarity(top1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, eid_a in enumerate(TUMOR_EIDS__27):
        a = top1[f"tumor_{eid_a}_any_active_support"].astype(bool).to_numpy()
        for eid_b in TUMOR_EIDS__27[i + 1 :]:
            b = top1[f"tumor_{eid_b}_any_active_support"].astype(bool).to_numpy()
            both = int((a & b).sum())
            a_only = int((a & ~b).sum())
            b_only = int((~a & b).sum())
            neither = int((~a & ~b).sum())
            union = both + a_only + b_only
            min_support = min(int(a.sum()), int(b.sum()))
            rows.append(
                {
                    "eid_a": eid_a,
                    "cell_line_a": CELL_LINE_NAMES__27[eid_a],
                    "eid_b": eid_b,
                    "cell_line_b": CELL_LINE_NAMES__27[eid_b],
                    "both_supported_n": both,
                    "a_only_n": a_only,
                    "b_only_n": b_only,
                    "neither_n": neither,
                    "jaccard": both / union if union else float("nan"),
                    "overlap_coefficient": (
                        both / min_support if min_support else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_specific_candidates(top1: pd.DataFrame) -> pd.DataFrame:
    specific = top1.loc[top1["tumor_active_support_breadth"].astype(int).eq(1)].copy()
    supported_eid = []
    for row in specific[
        [f"tumor_{eid}_any_active_support" for eid in TUMOR_EIDS__27]
    ].itertuples(index=False, name=None):
        eids = [eid for eid, flag in zip(TUMOR_EIDS__27, row) if bool(flag)]
        supported_eid.append(eids[0] if len(eids) == 1 else "")
    specific["specific_tumor_eid"] = supported_eid
    specific["specific_tumor_cell_line"] = specific["specific_tumor_eid"].map(
        CELL_LINE_NAMES__27
    )

    cols = [
        ID__27,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repFamily",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "specific_tumor_eid",
        "specific_tumor_cell_line",
        "reference_active_histone_prevalence_mean",
        "reference_repressive_histone_prevalence_mean",
        "nearest_gene_name",
        "nearest_gene_type",
    ]
    for eid in TUMOR_EIDS__27:
        cols.append(f"tumor_{eid}_active_mark_count")
    return specific[cols].sort_values(
        "herv_functional_priority_rank", kind="mergesort"
    ).reset_index(drop=True)


# =============================================================================
# 6. Deterministic cell-line-specific matched comparisons
# =============================================================================

def robust_scale__27(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr
    std = float(np.std(values))
    return std if std > 0 else 1.0


def _distance(anchor: pd.Series, candidates: pd.DataFrame, scales: dict[str, float]) -> np.ndarray:
    dist = np.zeros(len(candidates), dtype=np.float64)
    for var in MATCH_VARIABLES__27:
        a = float(anchor[var])
        c = candidates[var].to_numpy(np.float64)
        dist += MATCH_WEIGHTS__27[var] * np.abs(c - a) / scales[var]
    return dist


def build_cell_line_pairs(
    top1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_pairs: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []

    top1 = top1.copy()
    for eid in TUMOR_EIDS__27:
        support_col = f"tumor_{eid}_any_active_support"
        supported = top1.loc[top1[support_col].astype(bool)].copy()
        unsupported = top1.loc[~top1[support_col].astype(bool)].copy()

        # Pair the entire smaller group against unique members of the larger group.
        anchor_is_supported = len(supported) <= len(unsupported)
        anchors = supported.copy() if anchor_is_supported else unsupported.copy()
        donors = unsupported.copy() if anchor_is_supported else supported.copy()

        anchors = anchors.sort_values(
            ["repFamily", "herv_functional_priority_rank", ID__27], kind="mergesort"
        ).reset_index(drop=True)
        donors = donors.sort_values(
            ["repFamily", "herv_functional_priority_rank", ID__27], kind="mergesort"
        ).reset_index(drop=True)

        scales = {
            var: robust_scale__27(top1[var].to_numpy(np.float64))
            for var in MATCH_VARIABLES__27
        }
        available = np.ones(len(donors), dtype=bool)
        donor_family = donors["repFamily"].astype(str).to_numpy()

        exact_family = 0
        global_fallback = 0

        for pair_index, (_, anchor) in enumerate(anchors.iterrows(), start=1):
            family = str(anchor["repFamily"])
            candidate_idx = np.flatnonzero(available & (donor_family == family))
            used_exact = True
            if len(candidate_idx) == 0:
                candidate_idx = np.flatnonzero(available)
                used_exact = False
                global_fallback += 1
            else:
                exact_family += 1

            candidate_frame = donors.iloc[candidate_idx]
            distances = _distance(anchor, candidate_frame, scales)
            # deterministic tie-break by distance, rank, ID
            order = np.lexsort(
                (
                    candidate_frame[ID__27].astype(str).to_numpy(),
                    candidate_frame["herv_functional_priority_rank"].to_numpy(np.int64),
                    distances,
                )
            )
            chosen_pos = int(candidate_idx[int(order[0])])
            donor = donors.iloc[chosen_pos]
            available[chosen_pos] = False

            if anchor_is_supported:
                srow, urow = anchor, donor
            else:
                srow, urow = donor, anchor

            record = {
                "eid": eid,
                "cell_line": CELL_LINE_NAMES__27[eid],
                "pair_index": pair_index,
                "supported_id": str(srow[ID__27]),
                "unsupported_id": str(urow[ID__27]),
                "exact_family_match": int(used_exact),
                "supported_repFamily": str(srow["repFamily"]),
                "unsupported_repFamily": str(urow["repFamily"]),
            }
            for var in MATCH_VARIABLES__27:
                record[f"supported_{var}"] = float(srow[var])
                record[f"unsupported_{var}"] = float(urow[var])
            all_pairs.append(record)

        panel_pairs = [x for x in all_pairs if x["eid"] == eid]
        balance_rows.append(
            {
                "eid": eid,
                "cell_line": CELL_LINE_NAMES__27[eid],
                "supported_top1_n": len(supported),
                "unsupported_top1_n": len(unsupported),
                "matched_pairs_n": len(panel_pairs),
                "anchor_group": "supported" if anchor_is_supported else "unsupported",
                "exact_family_match_n": exact_family,
                "exact_family_match_rate": exact_family / len(panel_pairs),
                "global_fallback_n": global_fallback,
            }
        )

    return pd.DataFrame(all_pairs), pd.DataFrame(balance_rows)


def signed_rank_biserial(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta) & (delta != 0)]
    if len(delta) == 0:
        return 0.0
    ranks = rankdata(np.abs(delta), method="average")
    pos = ranks[delta > 0].sum()
    neg = ranks[delta < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom > 0 else 0.0


def compare_cell_line_context(
    top1: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    lookup = top1.set_index(ID__27, drop=False)
    rows: list[dict[str, Any]] = []

    for eid in TUMOR_EIDS__27:
        p = pairs.loc[pairs["eid"].eq(eid)].copy()
        supported = lookup.loc[p["supported_id"].astype(str)].reset_index(drop=True)
        unsupported = lookup.loc[p["unsupported_id"].astype(str)].reset_index(drop=True)

        for feature in NUMERIC_CONTEXT_FEATURES:
            s = supported[feature].to_numpy(np.float64)
            u = unsupported[feature].to_numpy(np.float64)
            delta = s - u
            if np.allclose(delta, 0):
                pvalue = 1.0
                statistic = 0.0
            else:
                test = wilcoxon(
                    delta,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(test.statistic)
                pvalue = float(test.pvalue)
            rows.append(
                {
                    "eid": eid,
                    "cell_line": CELL_LINE_NAMES__27[eid],
                    "feature": feature,
                    "feature_type": "numeric",
                    "matched_pairs_n": len(p),
                    "supported_median": float(np.median(s)),
                    "unsupported_median": float(np.median(u)),
                    "median_delta_supported_minus_unsupported": float(np.median(delta)),
                    "effect_rank_biserial": signed_rank_biserial(delta),
                    "discordant_supported1_unsupported0": np.nan,
                    "discordant_supported0_unsupported1": np.nan,
                    "test": "paired_wilcoxon",
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

        for feature in BINARY_CONTEXT_FEATURES:
            s = supported[feature].astype(int).to_numpy()
            u = unsupported[feature].astype(int).to_numpy()
            s1u0 = int(((s == 1) & (u == 0)).sum())
            s0u1 = int(((s == 0) & (u == 1)).sum())
            discordant = s1u0 + s0u1
            pvalue = (
                float(binomtest(s1u0, discordant, 0.5, alternative="two-sided").pvalue)
                if discordant > 0
                else 1.0
            )
            rows.append(
                {
                    "eid": eid,
                    "cell_line": CELL_LINE_NAMES__27[eid],
                    "feature": feature,
                    "feature_type": "binary",
                    "matched_pairs_n": len(p),
                    "supported_median": float(np.median(s)),
                    "unsupported_median": float(np.median(u)),
                    "median_delta_supported_minus_unsupported": float(np.median(s - u)),
                    "effect_rank_biserial": np.nan,
                    "discordant_supported1_unsupported0": s1u0,
                    "discordant_supported0_unsupported1": s0u1,
                    "test": "exact_discordant_pair_binomial",
                    "statistic": float(s1u0),
                    "pvalue": pvalue,
                }
            )

    result = pd.DataFrame(rows)
    result["fdr_within_cell_line"] = np.nan
    for eid in TUMOR_EIDS__27:
        idx = result["eid"].eq(eid)
        result.loc[idx, "fdr_within_cell_line"] = bh_fdr(
            result.loc[idx, "pvalue"].to_numpy(np.float64)
        )
    result["fdr_global"] = bh_fdr(result["pvalue"].to_numpy(np.float64))
    return result.sort_values(
        ["eid", "fdr_within_cell_line", "pvalue", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# 7. Export annotation tables
# =============================================================================

def add_supported_marks(frame: pd.DataFrame, eid: str) -> pd.Series:
    cols = [f"tumor_{eid}_{mark}_has" for mark in COMPARABLE_MARKS]
    values = []
    for row in frame[cols].itertuples(index=False, name=None):
        values.append(
            ",".join(
                mark for mark, flag in zip(COMPARABLE_MARKS, row) if int(flag) == 1
            )
        )
    return pd.Series(values, index=frame.index, dtype="object")


def build_top1_annotation(top1: pd.DataFrame) -> pd.DataFrame:
    out = top1.copy()
    out["tumor_active_support_pattern"] = support_pattern(out).to_numpy()
    for eid in TUMOR_EIDS__27:
        out[f"tumor_{eid}_supported_marks"] = add_supported_marks(out, eid)

    cols = [
        ID__27,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repClass",
        "repFamily",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "nearest_gene_name",
        "nearest_gene_type",
        "reference_active_histone_prevalence_mean",
        "reference_repressive_histone_prevalence_mean",
        "reference_active_minus_repressive_prevalence",
        "is_mpra_active_supported",
        "is_ccre_enhancer_like",
        "is_serve_gtex_supported",
        "tumor_active_support_breadth",
        "tumor_active_dominant_breadth",
        "tumor_repressive_support_breadth",
        "tumor_epigenomic_support_score",
        "tumor_cell_line_support_class",
        "tumor_active_supported_eids",
        "tumor_active_support_pattern",
    ]
    for mark in COMPARABLE_MARKS:
        cols.append(f"reference_{mark}_prevalence")
    for eid in TUMOR_EIDS__27:
        cols.extend(
            [
                f"tumor_{eid}_active_mark_count",
                f"tumor_{eid}_repressive_mark_count",
                f"tumor_{eid}_supported_marks",
            ]
        )
    return out[cols].sort_values(
        "herv_functional_priority_rank", kind="mergesort"
    ).reset_index(drop=True)


def build_top30__27(top1_annotation: pd.DataFrame) -> pd.DataFrame:
    candidates = top1_annotation.loc[
        top1_annotation["tumor_active_support_breadth"].astype(int).ge(1)
    ].copy()
    return candidates.head(30).reset_index(drop=True)


# =============================================================================
# 8. Report / manifest
# =============================================================================


def build_manifest(
    m24_candidates: int,
    mark_denominators: dict[str, int],
    mark_transition: pd.DataFrame,
    state_transition: pd.DataFrame,
    breadth: pd.DataFrame,
    patterns: pd.DataFrame,
    similarity: pd.DataFrame,
    specific: pd.DataFrame,
    pairs: pd.DataFrame,
    balance: pd.DataFrame,
    matched: pd.DataFrame,
    top1_annotation: pd.DataFrame,
    top30: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "status": "PASSED",
        "script": SCRIPT__27,
        "version": VERSION__27,
        "run_time_utc": now(),
        "final_global": EXPECTED_GLOBAL__27,
        "final_fer": EXPECTED_PRIMARY__27,
        "herv_universe": EXPECTED_N__27,
        "top1_k": TOP1_K__27,
        "m24_candidates": m24_candidates,
        "non_tumor_mark_denominators": mark_denominators,
        "summary": {
            "mark_transition_rows": len(mark_transition),
            "state_transition_rows": len(state_transition),
            "breadth_rows": len(breadth),
            "support_pattern_rows": len(patterns),
            "pairwise_similarity_rows": len(similarity),
            "cell_line_specific_candidates": len(specific),
            "matched_pairs_total": len(pairs),
            "matched_panels": balance.to_dict(orient="records"),
            "matched_tests": len(matched),
            "significant_within_cell_line_fdr_0_05": int(
                (matched["fdr_within_cell_line"] < 0.05).sum()
            ),
            "significant_global_fdr_0_05": int(
                (matched["fdr_global"] < 0.05).sum()
            ),
            "top1_annotation_rows": len(top1_annotation),
            "top30_rows": len(top30),
        },
    }


# =============================================================================
# 9. Main
# =============================================================================

def step_27() -> None:
    prepare_output__27()


    mark_denominators = load_mark_denominators()
    table = load_analysis_table__27(mark_denominators)
    m24_candidates = int(table["potential_reactivation_candidate"].astype(bool).sum())
    top1 = table.loc[
        table["herv_functional_priority_rank"].astype(np.int64).le(TOP1_K__27)
    ].copy()
    top1 = top1.sort_values(
        "herv_functional_priority_rank", kind="mergesort"
    ).reset_index(drop=True)

    mark_transition = build_mark_transition(top1)
    state_transition = build_state_transition(top1)
    breadth = build_breadth(top1)
    patterns = build_patterns(top1)
    similarity = build_pairwise_similarity(top1)
    specific = build_specific_candidates(top1)

    pairs, balance = build_cell_line_pairs(top1)
    matched = compare_cell_line_context(top1, pairs)

    top1_annotation = build_top1_annotation(top1)
    top30 = build_top30__27(top1_annotation)

    atomic_tsv__27(state_transition, OUT_STATE_TRANS)
    atomic_tsv__27(breadth, OUT_BREADTH)
    atomic_tsv__27(patterns, OUT_PATTERNS)
    atomic_tsv__27(similarity, OUT_SIMILARITY)
    atomic_tsv__27(specific, OUT_SPECIFIC)
    atomic_tsv__27(balance, OUT_BALANCE)
    atomic_tsv__27(matched, OUT_MATCHED)
    atomic_tsv__27(top1_annotation, OUT_TOP1)
    atomic_tsv__27(top30, OUT_TOP30__27)


    manifest = build_manifest(
        m24_candidates=m24_candidates,
        mark_denominators=mark_denominators,
        mark_transition=mark_transition,
        state_transition=state_transition,
        breadth=breadth,
        patterns=patterns,
        similarity=similarity,
        specific=specific,
        pairs=pairs,
        balance=balance,
        matched=matched,
        top1_annotation=top1_annotation,
        top30=top30,
    )
    atomic_json__27(manifest, OUT_MANIFEST)


# --------------------------------------------------------------------------
# step 02: Locus-level heterogeneity of the frozen ranking
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = (PROJECT_ROOT / "Biological_Interpretation").resolve()
Modeling = (PROJECT_ROOT / "Modeling").resolve()

SOURCE_TABLE = BIOLOGY_ROOT / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M25_MATCHED_BACKGROUND = HERE / 'output/25_evolutionary_interpretation/25_final_top1_matched_background.tsv'
M25_MATCHING_BALANCE = HERE / 'output/25_evolutionary_interpretation/25_final_top1_matching_balance.tsv'

OUT_DIR = BIOLOGY_ROOT / "output/02_analyze_locus_level_heterogeneity"

OUT_FAMILY__02 = OUT_DIR / "02_family_locus_heterogeneity.tsv"
OUT_SUBFAMILY = OUT_DIR / "02_subfamily_locus_heterogeneity.tsv"
OUT_SIMPLE_FEATURES = OUT_DIR / "02_simple_feature_rank_associations.tsv"
OUT_MATCHED__02 = OUT_DIR / "02_top1_matched_feature_comparison.tsv"
OUT_MATCHING_BALANCE__02 = OUT_DIR / "02_matching_balance.tsv"


OUT_KEY_RESULTS = OUT_DIR / "02_key_results.tsv"
OUT_SUMMARY = OUT_DIR / "02_summary.json"
ID_COL = "herv_internal_id"
TOP1_FRACTION = 0.01
EXPECTED_TOP1_N = 4_803
MIN_SUBFAMILY_N = 20
MIN_FAMILY_CONTEXT_N = 10
MIN_SUBFAMILY_CONTEXT_N = 20

CELL_LINES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

REQUIRED_SOURCE_COLUMNS = [
    ID_COL,
    "repName",
    "repFamily",
    "repeat_divergence",
    "length",
    "gc_content",
    "dist_to_tss",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_repressive_support_breadth",
]

for _eid__02 in CELL_LINES:
    REQUIRED_SOURCE_COLUMNS.extend(
        [
            f"tumor_{_eid__02}_any_active_support",
            f"tumor_{_eid__02}_any_repressive_support",
            f"tumor_{_eid__02}_active_dominant",
        ]
    )



def bh_fdr__02(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return pd.Series(adjusted, index=pvalues.index)

    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n = len(ranked)
    raw = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(raw[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    adjusted[finite_indices[order]] = corrected
    return pd.Series(adjusted, index=pvalues.index)






def prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    ranks = pd.to_numeric(source["herv_functional_priority_rank"], errors="coerce")
    source = source.copy()
    source["repFamily"] = source["repFamily"].fillna("Unknown").astype(str)
    source["repName"] = source["repName"].fillna("Unknown").astype(str)
    source["is_final_top1"] = ranks <= EXPECTED_TOP1_N
    source["abs_dist_to_tss"] = pd.to_numeric(
        source["dist_to_tss"], errors="coerce"
    ).abs()
    source["log1p_abs_dist_to_tss"] = np.log1p(source["abs_dist_to_tss"])

    for eid in CELL_LINES:
        for suffix in (
            "any_active_support",
            "any_repressive_support",
            "active_dominant",
        ):
            column = f"tumor_{eid}_{suffix}"
            values = pd.to_numeric(source[column], errors="coerce")
            source[column] = values.astype(bool)

    active_breadth = pd.to_numeric(
        source["tumor_active_support_breadth"], errors="coerce"
    )
    source["active_shared_ge2"] = active_breadth.ge(2)
    source["active_context_selective_eq1"] = active_breadth.eq(1)
    source["active_pan_five"] = active_breadth.eq(len(CELL_LINES))
    source["active_currently_unrealized"] = active_breadth.eq(0)
    return source


def enrichment_table(
    source: pd.DataFrame, group_column: str, minimum_group_n: int
) -> pd.DataFrame:
    n_total = len(source)
    n_top = int(source["is_final_top1"].sum())
    rows: list[dict[str, object]] = []

    for group, group_frame in source.groupby(group_column, sort=True, dropna=False):
        group_n = len(group_frame)
        if group_n < minimum_group_n:
            continue
        top_n = int(group_frame["is_final_top1"].sum())
        non_top_n = group_n - top_n
        outside_top_n = n_top - top_n
        outside_non_top_n = (n_total - group_n) - outside_top_n
        odds_ratio, pvalue = fisher_exact(
            [[top_n, non_top_n], [outside_top_n, outside_non_top_n]],
            alternative="two-sided",
        )

        scores = pd.to_numeric(
            group_frame["herv_functional_priority_score"], errors="coerce"
        ).dropna()
        quantiles = scores.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        top_rate = top_n / group_n
        background_top_rate = n_top / n_total
        divergence = pd.to_numeric(
            group_frame["repeat_divergence"], errors="coerce"
        )
        lengths = pd.to_numeric(group_frame["length"], errors="coerce")
        gc = pd.to_numeric(group_frame["gc_content"], errors="coerce")
        tss = pd.to_numeric(group_frame["abs_dist_to_tss"], errors="coerce")
        ranks = pd.to_numeric(
            group_frame["herv_functional_priority_rank"], errors="coerce"
        )
        rows.append(
            {
                group_column: group,
                "universe_n": group_n,
                "copy_number": group_n,
                "top1_n": top_n,
                "non_top1_n": non_top_n,
                "top1_rate": top_rate,
                "top1_background_rate": background_top_rate,
                "top1_fold_enrichment": top_rate / background_top_rate,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p": pvalue,
                "priority_rank_median": float(ranks.median()),
                "priority_score_q05": quantiles.loc[0.05],
                "priority_score_q25": quantiles.loc[0.25],
                "priority_score_median": quantiles.loc[0.50],
                "priority_score_q75": quantiles.loc[0.75],
                "priority_score_q95": quantiles.loc[0.95],
                "priority_score_iqr": quantiles.loc[0.75] - quantiles.loc[0.25],
                "priority_score_q95_minus_q05": quantiles.loc[0.95]
                - quantiles.loc[0.05],
                "repeat_divergence_median": float(divergence.median()),
                "repeat_divergence_iqr": float(
                    divergence.quantile(0.75) - divergence.quantile(0.25)
                ),
                "length_median": float(lengths.median()),
                "gc_content_median": float(gc.median()),
                "abs_dist_to_tss_median": float(tss.median()),
                "contains_top1_and_non_top1": bool(top_n > 0 and non_top_n > 0),
            }
        )

    result = pd.DataFrame(rows)
    result["fisher_fdr_bh"] = bh_fdr__02(result["fisher_p"])
    result["top1_enrichment_class"] = np.select(
        [
            result["fisher_fdr_bh"].lt(0.05)
            & result["top1_fold_enrichment"].gt(1),
            result["fisher_fdr_bh"].lt(0.05)
            & result["top1_fold_enrichment"].lt(1),
        ],
        ["enriched", "depleted"],
        default="not_significant",
    )
    return result.sort_values(
        ["top1_fold_enrichment", "universe_n"], ascending=[False, False]
    ).reset_index(drop=True)


def categorical_effect_sizes(
    source: pd.DataFrame, group_column: str, prefix: str
) -> dict[str, float]:
    contingency = pd.crosstab(source[group_column], source["is_final_top1"])
    chi2, pvalue, _, _ = chi2_contingency(contingency)
    denominator = len(source) * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / denominator)) if denominator > 0 else np.nan

    scores = pd.to_numeric(
        source["herv_functional_priority_score"], errors="coerce"
    )
    grand_mean = float(scores.mean())
    total_ss = float(((scores - grand_mean) ** 2).sum())
    between_ss = 0.0
    for _, group in source.groupby(group_column, sort=False):
        group_scores = pd.to_numeric(
            group["herv_functional_priority_score"], errors="coerce"
        )
        between_ss += len(group_scores) * float(group_scores.mean() - grand_mean) ** 2
    eta_squared = between_ss / total_ss if total_ss > 0 else np.nan

    return {
        f"{prefix}_top1_chi2": float(chi2),
        f"{prefix}_top1_chi2_p": float(pvalue),
        f"{prefix}_top1_cramers_v": cramers_v,
        f"{prefix}_priority_score_eta_squared": float(eta_squared),
    }


def tumor_context_outcomes() -> list[tuple[str, str, str, str]]:
    outcomes: list[tuple[str, str, str, str]] = []
    for eid, name in CELL_LINES.items():
        outcomes.extend(
            [
                (
                    f"{eid}_active",
                    f"tumor_{eid}_any_active_support",
                    "active",
                    name,
                ),
                (
                    f"{eid}_active_dominant",
                    f"tumor_{eid}_active_dominant",
                    "active_dominant",
                    name,
                ),
                (
                    f"{eid}_repressive",
                    f"tumor_{eid}_any_repressive_support",
                    "repressive",
                    name,
                ),
            ]
        )
    outcomes.extend(
        [
            (
                "shared_active_ge2",
                "active_shared_ge2",
                "realization_class",
                "Active support in at least two tumor cell lines",
            ),
            (
                "context_selective_active_eq1",
                "active_context_selective_eq1",
                "realization_class",
                "Active support in exactly one tumor cell line",
            ),
            (
                "pan_active_five",
                "active_pan_five",
                "realization_class",
                "Active support in all five tumor cell lines",
            ),
            (
                "currently_unrealized_active",
                "active_currently_unrealized",
                "realization_class",
                "No active support in the five assayed tumor cell lines",
            ),
        ]
    )
    return outcomes


def tumor_context_enrichment_table(
    source: pd.DataFrame,
    group_column: str,
    minimum_group_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = (
        ("Genomewide", source),
        ("FinalTop1", source.loc[source["is_final_top1"]]),
    )
    for scope_name, scope in scopes:
        scope_n = len(scope)
        for outcome_name, outcome_column, evidence_type, context_name in tumor_context_outcomes():
            outcome = scope[outcome_column].astype(bool)
            background_positive_n = int(outcome.sum())
            background_rate = background_positive_n / scope_n
            for group, group_frame in scope.groupby(
                group_column, sort=True, dropna=False
            ):
                group_n = len(group_frame)
                if group_n < minimum_group_n:
                    continue
                positive_n = int(group_frame[outcome_column].astype(bool).sum())
                negative_n = group_n - positive_n
                outside_positive_n = background_positive_n - positive_n
                outside_negative_n = (scope_n - group_n) - outside_positive_n
                odds_ratio, pvalue = fisher_exact(
                    [
                        [positive_n, negative_n],
                        [outside_positive_n, outside_negative_n],
                    ],
                    alternative="two-sided",
                )
                rate = positive_n / group_n
                rows.append(
                    {
                        "analysis_scope": scope_name,
                        "group_level": "family"
                        if group_column == "repFamily"
                        else "subfamily",
                        "group_name": group,
                        "outcome": outcome_name,
                        "evidence_type": evidence_type,
                        "context_name": context_name,
                        "scope_n": scope_n,
                        "group_n": group_n,
                        "positive_n": positive_n,
                        "support_rate": rate,
                        "background_positive_n": background_positive_n,
                        "background_rate": background_rate,
                        "fold_enrichment": rate / background_rate
                        if background_rate > 0
                        else np.nan,
                        "fisher_odds_ratio": float(odds_ratio),
                        "fisher_p": float(pvalue),
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fisher_fdr_bh_within_scope_outcome"] = result.groupby(
        ["analysis_scope", "outcome"], sort=False
    )["fisher_p"].transform(bh_fdr__02)
    result["enrichment_class"] = np.select(
        [
            result["fisher_fdr_bh_within_scope_outcome"].lt(0.05)
            & result["fold_enrichment"].gt(1),
            result["fisher_fdr_bh_within_scope_outcome"].lt(0.05)
            & result["fold_enrichment"].lt(1),
        ],
        ["enriched", "depleted"],
        default="not_significant",
    )
    return result.sort_values(
        [
            "analysis_scope",
            "outcome",
            "fisher_fdr_bh_within_scope_outcome",
            "fold_enrichment",
            "group_name",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def simple_feature_associations(source: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "repeat_divergence",
        "length",
        "gc_content",
        "abs_dist_to_tss",
        "log1p_abs_dist_to_tss",
    ]
    outcome = pd.to_numeric(
        source["herv_functional_priority_score"], errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        values = pd.to_numeric(source[feature], errors="coerce")
        valid = values.notna() & outcome.notna()
        rho, pvalue = spearmanr(values[valid], outcome[valid])
        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "spearman_rho_with_priority_score": float(rho),
                "spearman_p": float(pvalue),
                "absolute_spearman_rho": abs(float(rho)),
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_fdr_bh"] = bh_fdr__02(result["spearman_p"])
    return result.sort_values("absolute_spearman_rho", ascending=False).reset_index(
        drop=True
    )


def paired_feature_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "repeat_divergence": (
            "top1_repeat_divergence",
            "background_repeat_divergence",
        ),
        "within_family_divergence_percentile": (
            "top1_within_family_divergence_percentile",
            "background_within_family_divergence_percentile",
        ),
        "length": ("top1_length", "background_length"),
        "gc_content": ("top1_gc_content", "background_gc_content"),
        "abs_dist_to_tss": (
            "top1_abs_dist_to_tss",
            "background_abs_dist_to_tss",
        ),
    }
    rows: list[dict[str, object]] = []
    for metric, (top_column, background_column) in metrics.items():
        top_values = pd.to_numeric(pairs[top_column], errors="coerce")
        background_values = pd.to_numeric(
            pairs[background_column], errors="coerce"
        )
        valid = top_values.notna() & background_values.notna()
        top_values = top_values[valid].to_numpy(dtype=float)
        background_values = background_values[valid].to_numpy(dtype=float)
        difference = top_values - background_values

        if np.allclose(difference, 0.0):
            statistic, pvalue = 0.0, 1.0
        else:
            test = wilcoxon(
                top_values,
                background_values,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
            statistic, pvalue = float(test.statistic), float(test.pvalue)

        rows.append(
            {
                "metric": metric,
                "paired_n": len(top_values),
                "top1_median": float(np.median(top_values)),
                "matched_background_median": float(np.median(background_values)),
                "median_paired_difference": float(np.median(difference)),
                "median_absolute_paired_difference": float(
                    np.median(np.abs(difference))
                ),
                "fraction_top1_greater": float(np.mean(difference > 0)),
                "fraction_equal": float(np.mean(difference == 0)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["wilcoxon_fdr_bh"] = bh_fdr__02(result["wilcoxon_p"])
    return result


def step_02() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source = prepare_source(
        pd.read_parquet(SOURCE_TABLE, columns=REQUIRED_SOURCE_COLUMNS)
    )
    pairs = pd.read_csv(M25_MATCHED_BACKGROUND, sep="\t", low_memory=False)
    matching_balance = pd.read_csv(M25_MATCHING_BALANCE, sep="\t")

    family = enrichment_table(source, "repFamily", minimum_group_n=1)
    subfamily = enrichment_table(
        source, "repName", minimum_group_n=MIN_SUBFAMILY_N
    )
    family_context = tumor_context_enrichment_table(
        source, "repFamily", minimum_group_n=MIN_FAMILY_CONTEXT_N
    )
    subfamily_context = tumor_context_enrichment_table(
        source, "repName", minimum_group_n=MIN_SUBFAMILY_CONTEXT_N
    )
    associations = simple_feature_associations(source)
    matched = paired_feature_comparison(pairs)
    effect_sizes = {
        **categorical_effect_sizes(source, "repFamily", "family"),
        **categorical_effect_sizes(source, "repName", "subfamily"),
    }

    major_family = family[family["universe_n"] >= 1_000]
    key_results = pd.DataFrame(
        [
            {
                "metric": "universe_n",
                "value": len(source),
                "interpretation": "Frozen HERV analysis universe",
            },
            {
                "metric": "top1_n",
                "value": int(source["is_final_top1"].sum()),
                "interpretation": "Fixed genome-wide Top1 loci",
            },
            {
                "metric": "family_top1_cramers_v",
                "value": effect_sizes["family_top1_cramers_v"],
                "interpretation": "Association strength between family and Top1 membership",
            },
            {
                "metric": "family_priority_score_eta_squared",
                "value": effect_sizes["family_priority_score_eta_squared"],
                "interpretation": "Fraction of priority-score variance described by family means",
            },
            {
                "metric": "subfamily_top1_cramers_v",
                "value": effect_sizes["subfamily_top1_cramers_v"],
                "interpretation": "Association strength between subfamily and Top1 membership",
            },
            {
                "metric": "subfamily_priority_score_eta_squared",
                "value": effect_sizes["subfamily_priority_score_eta_squared"],
                "interpretation": "Descriptive fraction of priority-score variance between subfamily means",
            },
            {
                "metric": "top1_enriched_family_count_fdr0_05",
                "value": int(family["top1_enrichment_class"].eq("enriched").sum()),
                "interpretation": "Families enriched for frozen Top1 membership",
            },
            {
                "metric": "top1_enriched_subfamily_count_fdr0_05",
                "value": int(subfamily["top1_enrichment_class"].eq("enriched").sum()),
                "interpretation": "Tested subfamilies enriched for frozen Top1 membership",
            },
            {
                "metric": "major_families_with_top1_and_non_top1",
                "value": int(major_family["contains_top1_and_non_top1"].sum()),
                "interpretation": "Major families showing within-family locus heterogeneity",
            },
            {
                "metric": "major_family_count",
                "value": len(major_family),
                "interpretation": "Families with at least 1,000 loci",
            },
            {
                "metric": "max_absolute_simple_feature_spearman_rho",
                "value": float(associations["absolute_spearman_rho"].max()),
                "interpretation": "Largest univariate monotonic association with priority score",
            },
            {
                "metric": "family_tumor_context_tests_fdr0_05",
                "value": int(
                    family_context["fisher_fdr_bh_within_scope_outcome"]
                    .lt(0.05)
                    .sum()
                ),
                "interpretation": "Family/context associations across genome-wide and Top1 scopes",
            },
            {
                "metric": "subfamily_tumor_context_tests_fdr0_05",
                "value": int(
                    subfamily_context["fisher_fdr_bh_within_scope_outcome"]
                    .lt(0.05)
                    .sum()
                ),
                "interpretation": "Subfamily/context associations across genome-wide and Top1 scopes",
            },
        ]
    )

    summary = {
        "status": "COMPLETED",
        "script": "02_analyze_locus_level_heterogeneity",
        "created_at_utc": now(),
        "analysis_boundary": {
            "training_performed": False,
            "refit_performed": False,
            "model_selection_performed": False,
            "threshold_search_performed": False,
            "top1_rule": f"rank <= {EXPECTED_TOP1_N}",
            "family_and_divergence_are_model_inputs": True,
            "independent_validation_claimed": False,
        },
        "inputs": {
            "source_table": str(SOURCE_TABLE),
            "m25_matched_background": str(M25_MATCHED_BACKGROUND),
            "m25_matching_balance": str(M25_MATCHING_BALANCE),
        },
        "counts": {
            "universe": len(source),
            "top1": int(source["is_final_top1"].sum()),
            "families": int(source["repFamily"].nunique()),
            "subfamilies_total": int(source["repName"].nunique()),
            "subfamilies_tested": len(subfamily),
            "matched_pairs": len(pairs),
            "family_tumor_context_tests": len(family_context),
            "subfamily_tumor_context_tests": len(subfamily_context),
        },
        "effect_sizes": effect_sizes,
        "outputs": {
            "family": str(OUT_FAMILY__02),
            "subfamily": str(OUT_SUBFAMILY),
            "simple_features": str(OUT_SIMPLE_FEATURES),
            "matched_features": str(OUT_MATCHED__02),
            "matching_balance": str(OUT_MATCHING_BALANCE__02),
            "key_results": str(OUT_KEY_RESULTS),
        },
    }

    atomic_write_tsv(family, OUT_FAMILY__02)
    atomic_write_tsv(subfamily, OUT_SUBFAMILY)
    atomic_write_tsv(associations, OUT_SIMPLE_FEATURES)
    atomic_write_tsv(matched, OUT_MATCHED__02)
    atomic_write_tsv(matching_balance, OUT_MATCHING_BALANCE__02)
    atomic_write_tsv(key_results, OUT_KEY_RESULTS)
    atomic_write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", OUT_SUMMARY
    )


def main() -> None:
    step_25()
    step_25b()
    step_26()
    step_27()
    step_02()


if __name__ == "__main__":
    main()
