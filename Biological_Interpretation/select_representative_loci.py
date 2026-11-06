#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from biological_common import (
    atomic_write_tsv,
    write_tsv,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import json
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# step 10: Build the discovery candidate catalog
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = (PROJECT_ROOT / "Biological_Interpretation").resolve()
Modeling_OUTPUT = (PROJECT_ROOT / "Modeling/output").resolve()

SOURCE_TABLE = BIOLOGY_ROOT / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M23_MANIFEST = Modeling_OUTPUT / '23_generate_genomewide_herv_functional_ranking/23_genomewide_ranking_manifest.json'
M24_MANIFEST = Modeling_OUTPUT / '24_project_frozen_herv_ranking_to_tumor_epigenomes/24_tumor_projection_manifest.json'
M27_MANIFEST = BIOLOGY_ROOT / "27_reference_to_tumor_context_interpretation/27_reference_to_tumor_context_manifest.json"

OUT_DIR = BIOLOGY_ROOT / "output/10_discovery_candidate_catalog"


OUT_CATALOG = OUT_DIR / "10_prioritized_candidate_catalog.tsv"

OUT_UNREALIZED_CATALOG = OUT_DIR / '10_prioritized_currently_unrealized_catalog.tsv'


ID_COL = "herv_internal_id"
RANK_COL = "herv_functional_priority_rank"
TOP10_SCOPE = "FrozenTop10Candidate"

CELL_LINES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

SEQUENCE_POOL_PERCENTILE = 0.95
REPRESENTATION_POOL_PERCENTILE = 0.90
SEQUENCE_HIGH_PERCENTILE = 0.99
REPRESENTATION_HIGH_PERCENTILE = 0.95
MIN_ACTIVE_MARKS_PER_CONTEXT = 2
CASES_PER_CELL_LINE = 3
SHARED_CASES = 10
UNREALIZED_CASES = 6

REFERENCE_MARK_GROUPS = {
    "reference_active_histone_prevalence_mean": (
        "H3K27ac", "H3K4me1", "H3K4me3"
    ),
    "reference_repressive_histone_prevalence_mean": (
        "H3K27me3", "H3K9me3"
    ),
}

DIRECT_REFERENCE_FLAGS = [
    "is_ccre_enhancer_like",
    "is_mpra_active_supported",
    "is_serve_gtex_supported",
]

CATALOG_COLUMNS = [
    "candidate_priority_rank",
    "candidate_tier",
    "candidate_context_class",
    ID_COL,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repFamily",
    "repeat_divergence",
    "length",
    "gc_content",
    "dist_to_tss",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "split",
    "score_sequence_regulatory",
    "rank_pct_score_sequence_regulatory",
    "score_final_representation",
    "rank_pct_score_final_representation",
    "score_final_global",
    RANK_COL,
    "herv_functional_priority_score",
    "family_sequence_percentile",
    "direct_reference_evidence_count",
    "is_ccre_enhancer_like",
    "is_mpra_active_supported",
    "is_serve_gtex_supported",
    "roadmap_DNase_fdr001_sample_count",
    "reference_active_histone_prevalence_mean",
    "reference_repressive_histone_prevalence_mean",
    "tumor_strong_context_count",
    "tumor_strong_supported_eids",
    "tumor_active_support_breadth",
    "tumor_active_supported_eids",
    "tumor_epigenomic_support_score",
    "tumor_repressive_context_score",
    "literature_status",
    "claim_status",
]










def load_manifests() -> tuple[dict, dict, dict]:
    return (
        json.loads(M23_MANIFEST.read_text(encoding="utf-8")),
        json.loads(M24_MANIFEST.read_text(encoding="utf-8")),
        json.loads(M27_MANIFEST.read_text(encoding="utf-8")),
    )


def add_reference_prevalence(
    frame: pd.DataFrame, denominators: dict[str, int]
) -> pd.DataFrame:
    for target, marks in REFERENCE_MARK_GROUPS.items():
        frame[target] = pd.concat(
            [
                pd.to_numeric(
                    frame[f"roadmap_{mark}_sample_count"], errors="raise"
                ) / float(denominators[mark])
                for mark in marks
            ],
            axis=1,
        ).mean(axis=1)
    return frame


def prepare_candidates(
    frame: pd.DataFrame, top10_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_count = frame[DIRECT_REFERENCE_FLAGS].astype(int).sum(axis=1)
    strong = pd.DataFrame(
        {
            eid: pd.to_numeric(
                frame[f"tumor_{eid}_active_mark_count"], errors="coerce"
            ).ge(MIN_ACTIVE_MARKS_PER_CONTEXT)
            for eid in CELL_LINES
        },
        index=frame.index,
    )
    strong_count = strong.sum(axis=1).astype(np.int8)
    strong_eids = strong.apply(
        lambda row: ";".join(
            eid for eid in CELL_LINES if bool(row[eid])
        ),
        axis=1,
    )

    work = frame.copy()
    work["direct_reference_evidence_count"] = direct_count.astype(np.int8)
    work["tumor_strong_context_count"] = strong_count
    work["tumor_strong_supported_eids"] = strong_eids
    work["family_sequence_percentile"] = work.groupby("repFamily")[
        "score_sequence_regulatory"
    ].rank(method="average", pct=True)

    base_mask = (
        work["rank_pct_score_sequence_regulatory"].ge(
            SEQUENCE_POOL_PERCENTILE
        )
        & work["rank_pct_score_final_representation"].ge(
            REPRESENTATION_POOL_PERCENTILE
        )
        & work["direct_reference_evidence_count"].eq(0)
    )
    pool = work.loc[
        base_mask & work["tumor_strong_context_count"].ge(1)
    ].copy()
    unrealized_pool = work.loc[
        base_mask & work["tumor_active_support_breadth"].eq(0)
    ].copy()

    pool["candidate_tier"] = np.where(
        pool["rank_pct_score_sequence_regulatory"].ge(
            SEQUENCE_HIGH_PERCENTILE
        )
        & pool["rank_pct_score_final_representation"].ge(
            REPRESENTATION_HIGH_PERCENTILE
        ),
        "HIGH_SEQUENCE_CONFIDENCE",
        "PRIMARY_DISCOVERY_POOL",
    )
    pool["candidate_context_class"] = np.where(
        pool["tumor_strong_context_count"].eq(1),
        "CELL_LINE_SELECTIVE_STRONG",
        "MULTI_CONTEXT_STRONG",
    )
    pool["literature_status"] = "NOT_ASSESSED_REQUIRES_LOCUS_SEARCH"
    pool["claim_status"] = "PRELIMINARY_MODEL_NOMINATED_CANDIDATE"
    pool = pool.sort_values(
        [
            "candidate_tier",
            "tumor_strong_context_count",
            "tumor_epigenomic_support_score",
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            ID_COL,
        ],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    catalog = pool.loc[pool[RANK_COL].le(top10_k)].copy()
    catalog["_tier_order"] = catalog["candidate_tier"].eq(
        "HIGH_SEQUENCE_CONFIDENCE"
    ).map({True: 0, False: 1})
    catalog = catalog.sort_values(
        [
            "_tier_order",
            "tumor_strong_context_count",
            "tumor_epigenomic_support_score",
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            RANK_COL,
            ID_COL,
        ],
        ascending=[True, False, False, False, False, True, True],
        kind="mergesort",
    ).drop(columns="_tier_order").reset_index(drop=True)
    catalog.insert(
        0, "candidate_priority_rank", np.arange(1, len(catalog) + 1)
    )
    pool.insert(
        0, "candidate_priority_rank", np.arange(1, len(pool) + 1)
    )

    unrealized_pool["candidate_tier"] = np.where(
        unrealized_pool["rank_pct_score_sequence_regulatory"].ge(
            SEQUENCE_HIGH_PERCENTILE
        )
        & unrealized_pool["rank_pct_score_final_representation"].ge(
            REPRESENTATION_HIGH_PERCENTILE
        ),
        "HIGH_SEQUENCE_CONFIDENCE",
        "UNREALIZED_COMPARISON_POOL",
    )
    unrealized_pool["candidate_context_class"] = (
        "CURRENTLY_UNREALIZED_IN_FIVE_ASSAYED_CONTEXTS"
    )
    unrealized_pool["literature_status"] = (
        "NOT_ASSESSED_REQUIRES_LOCUS_SEARCH"
    )
    unrealized_pool["claim_status"] = "PRELIMINARY_UNREALIZED_COMPARISON"
    unrealized_pool = unrealized_pool.sort_values(
        [
            "candidate_tier",
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            RANK_COL,
            ID_COL,
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    unrealized_catalog = unrealized_pool.loc[
        unrealized_pool[RANK_COL].le(top10_k)
    ].copy()
    unrealized_catalog.insert(
        0, "candidate_priority_rank",
        np.arange(1, len(unrealized_catalog) + 1),
    )
    unrealized_pool.insert(
        0, "candidate_priority_rank",
        np.arange(1, len(unrealized_pool) + 1),
    )
    return pool, catalog, unrealized_pool, unrealized_catalog


def match_supported_to_unrealized(
    supported: pd.DataFrame, unrealized: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if supported.empty or unrealized.empty:
        return pd.DataFrame(), pd.DataFrame()

    left = supported.copy()
    right = unrealized.copy()
    left["_used"] = False
    right["_used"] = False
    pairs = []
    supported_order = left.sort_values(
        ["candidate_tier", RANK_COL, ID_COL],
        ascending=[True, True, True],
        kind="mergesort",
    )
    for _, row in supported_order.iterrows():
        available = right.loc[~right["_used"]]
        same_family = available[
            available["repFamily"].astype(str).eq(str(row["repFamily"]))
        ]
        candidates = same_family if not same_family.empty else available
        if candidates.empty:
            break

        distance = (
            (
                candidates["rank_pct_score_sequence_regulatory"]
                - row["rank_pct_score_sequence_regulatory"]
            ).abs()
            + (
                candidates["rank_pct_score_final_representation"]
                - row["rank_pct_score_final_representation"]
            ).abs()
            + (
                candidates["repeat_divergence"] - row["repeat_divergence"]
            ).abs() / 100.0
            + (candidates["length"] - row["length"]).abs() / 10000.0
        )
        selected_id = distance.sort_values(kind="mergesort").index[0]
        target = right.loc[selected_id]
        right.loc[selected_id, "_used"] = True
        pairs.append({
            "supported_id": row[ID_COL],
            "unrealized_id": target[ID_COL],
            "supported_family": row["repFamily"],
            "unrealized_family": target["repFamily"],
            "supported_final_rank": row[RANK_COL],
            "unrealized_final_rank": target[RANK_COL],
            "supported_sequence_percentile": (
                row["rank_pct_score_sequence_regulatory"]
            ),
            "unrealized_sequence_percentile": (
                target["rank_pct_score_sequence_regulatory"]
            ),
            "supported_tumor_score": row["tumor_epigenomic_support_score"],
            "unrealized_tumor_score": (
                target["tumor_epigenomic_support_score"]
            ),
            "supported_strong_context_n": row["tumor_strong_context_count"],
            "unrealized_strong_context_n": (
                target["tumor_strong_context_count"]
            ),
        })

    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty:
        return pair_frame, pair_frame

    comparison = pd.DataFrame([
        {
            "metric": metric,
            "paired_n": len(pair_frame),
            "supported_median": float(pair_frame[s_col].median()),
            "unrealized_median": float(pair_frame[u_col].median()),
            "median_supported_minus_unrealized": float(
                (pair_frame[s_col] - pair_frame[u_col]).median()
            ),
        }
        for metric, s_col, u_col in [
            (
                "final_rank",
                "supported_final_rank",
                "unrealized_final_rank",
            ),
            (
                "sequence_percentile",
                "supported_sequence_percentile",
                "unrealized_sequence_percentile",
            ),
            (
                "tumor_epigenomic_support_score",
                "supported_tumor_score",
                "unrealized_tumor_score",
            ),
            (
                "strong_context_n",
                "supported_strong_context_n",
                "unrealized_strong_context_n",
            ),
        ]
    ])
    return pair_frame, comparison


def select_cases(
    catalog: pd.DataFrame, unrealized_catalog: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for eid, name in CELL_LINES.items():
        selected = catalog[
            catalog["candidate_context_class"].eq(
                "CELL_LINE_SELECTIVE_STRONG"
            )
            & catalog["tumor_strong_supported_eids"].eq(eid)
        ].head(CASES_PER_CELL_LINE).copy()
        selected.insert(0, "case_category", f"{eid}_SELECTIVE")
        selected.insert(1, "case_context", name)
        rows.append(selected)

    shared = catalog[
        catalog["candidate_context_class"].eq("MULTI_CONTEXT_STRONG")
    ].head(SHARED_CASES).copy()
    shared.insert(0, "case_category", "SHARED_MULTI_CONTEXT")
    shared.insert(1, "case_context", "Multiple tumor cell lines")
    rows.append(shared)

    unrealized = unrealized_catalog.head(UNREALIZED_CASES).copy()
    unrealized.insert(0, "case_category", "CURRENTLY_UNREALIZED")
    unrealized.insert(
        1, "case_context", "No active support in five assayed tumor contexts"
    )
    rows.append(unrealized)
    if not rows:
        return pd.DataFrame()
    cases = pd.concat(rows, ignore_index=True)
    return cases[["case_category", "case_context"] + CATALOG_COLUMNS]


def summarize_tiers(
    pool: pd.DataFrame,
    catalog: pd.DataFrame,
    unrealized_pool: pd.DataFrame,
    unrealized_catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scope, data in (
        ("incremental_pool", pool),
        ("prioritized_top10", catalog),
        ("currently_unrealized_pool", unrealized_pool),
        ("currently_unrealized_top10", unrealized_catalog),
    ):
        base_tier = (
            "UNREALIZED_COMPARISON_POOL"
            if scope.startswith("currently_unrealized")
            else "PRIMARY_DISCOVERY_POOL"
        )
        for tier in ("ALL", "HIGH_SEQUENCE_CONFIDENCE", base_tier):
            part = data if tier == "ALL" else data[
                data["candidate_tier"].eq(tier)
            ]
            rows.append({
                "scope": scope,
                "candidate_tier": tier,
                "n": int(len(part)),
                "cell_line_selective_n": int(
                    part["candidate_context_class"].eq(
                        "CELL_LINE_SELECTIVE_STRONG"
                    ).sum()
                ),
                "multi_context_n": int(
                    part["candidate_context_class"].eq(
                        "MULTI_CONTEXT_STRONG"
                    ).sum()
                ),
                "currently_unrealized_n": int(
                    part["candidate_context_class"].eq(
                        "CURRENTLY_UNREALIZED_IN_FIVE_ASSAYED_CONTEXTS"
                    ).sum()
                ),
                "median_final_rank": (
                    float(part[RANK_COL].median()) if len(part) else np.nan
                ),
                "median_sequence_percentile": (
                    float(
                        part["rank_pct_score_sequence_regulatory"].median()
                    ) if len(part) else np.nan
                ),
                "median_tumor_support_score": (
                    float(part["tumor_epigenomic_support_score"].median())
                    if len(part) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def summarize_families(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, part in catalog.groupby("repFamily", dropna=False):
        rows.append({
            "repFamily": family,
            "candidate_n": int(len(part)),
            "high_sequence_confidence_n": int(
                part["candidate_tier"].eq(
                    "HIGH_SEQUENCE_CONFIDENCE"
                ).sum()
            ),
            "cell_line_selective_n": int(
                part["candidate_context_class"].eq(
                    "CELL_LINE_SELECTIVE_STRONG"
                ).sum()
            ),
            "multi_context_n": int(
                part["candidate_context_class"].eq(
                    "MULTI_CONTEXT_STRONG"
                ).sum()
            ),
            "median_final_rank": float(part[RANK_COL].median()),
            "median_family_sequence_percentile": float(
                part["family_sequence_percentile"].median()
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["candidate_n", "repFamily"], ascending=[False, True]
    ).reset_index(drop=True)


def step_10() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m23, m24, m27 = load_manifests()
    top10_k = int(m23["candidate_K"])
    denominators = m27["non_tumor_mark_denominators"]

    frame = pd.read_parquet(SOURCE_TABLE)
    frame = add_reference_prevalence(frame, denominators)
    frame[RANK_COL] = pd.to_numeric(
        frame[RANK_COL], errors="raise"
    ).astype(np.int64)
    for column in DIRECT_REFERENCE_FLAGS:
        frame[column] = frame[column].astype(int)

    pool, catalog, unrealized_pool, unrealized_catalog = prepare_candidates(
        frame, top10_k
    )
    matched_pairs, matched_comparison = match_supported_to_unrealized(
        catalog, unrealized_catalog
    )


    atomic_write_tsv(catalog[CATALOG_COLUMNS], OUT_CATALOG)
    atomic_write_tsv(
        unrealized_catalog[CATALOG_COLUMNS], OUT_UNREALIZED_CATALOG
    )


# --------------------------------------------------------------------------
# step 13: Deep dive the representative loci
# --------------------------------------------------------------------------



PROJECT_ROOT__13 = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT__13 = PROJECT_ROOT__13 / "Biological_Interpretation"
MODEL_OUTPUT__13 = PROJECT_ROOT__13 / "Modeling/output"

M03_DIR = BIOLOGY_ROOT__13 / "output/03_analyze_sequence_grammar"
M05_DIR = MODEL_OUTPUT__13 / "05_k6_bpe_complementarity_controls"
M10_DIR__13 = BIOLOGY_ROOT__13 / "output/10_discovery_candidate_catalog"
M11_DIR = BIOLOGY_ROOT__13 / "output/11_multivariable_tumor_generalization"
M12_DIR = config.DATA_DIR / "representative_loci"
M28_DIR = BIOLOGY_ROOT__13 / "28_sequence_grammar_and_representative_loci"

SOURCE_TABLE__13 = BIOLOGY_ROOT__13 / 'output/01_build_biological_source_table/01_biological_source_table.parquet'

OUT_DIR__13 = BIOLOGY_ROOT__13 / "output/13_representative_locus_deep_dive"

OUT_MAIN = OUT_DIR__13 / "13_representative_locus_deep_dive.tsv"


OUT_CHROM = OUT_DIR__13 / "13_chromatin_evidence.tsv"

OUT_REVIEW = OUT_DIR__13 / "13_literature_review_reference_table.tsv"


ID = "herv_internal_id"
EIDS = ["E114", "E115", "E117", "E118", "E123"]
MARKS = ["H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3"]
ACTIVE_MARKS = {"H3K27ac", "H3K4me1", "H3K4me3"}










def clean_text(value: object) -> str:
    if (
        value is None
        or (isinstance(value, float) and np.isnan(value))
        or pd.isna(value)
    ):
        return ""
    return str(value)


def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def source_columns() -> list[str]:
    cols = [
        ID,
        "gctx_roadmap_active_mark_presence_count",
        "gctx_roadmap_repressive_mark_presence_count",
        "gctx_roadmap_active_sample_count_sum",
        "gctx_roadmap_repressive_sample_count_sum",
    ]
    for eid in EIDS:
        for mark in MARKS:
            cols += [
                f"tumor_{eid}_{mark}_has",
                f"tumor_{eid}_{mark}_total_overlap_bp",
            ]
        cols += [
            f"tumor_{eid}_active_mark_count",
            f"tumor_{eid}_repressive_mark_count",
            f"tumor_{eid}_active_total_overlap_bp",
            f"tumor_{eid}_repressive_total_overlap_bp",
            f"tumor_{eid}_any_active_support",
            f"tumor_{eid}_any_repressive_support",
            f"tumor_{eid}_active_dominant",
        ]
    cols += [
        "tumor_active_support_breadth",
        "tumor_repressive_support_breadth",
        "tumor_total_active_mark_hits",
        "tumor_total_repressive_mark_hits",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
    ]
    return cols


def read_source(ids: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(SOURCE_TABLE__13, columns=source_columns())
    return frame[frame[ID].isin(ids)].copy()


def top_family_module(modules: pd.DataFrame) -> pd.DataFrame:
    if modules.empty:
        return pd.DataFrame(columns=["repFamily"])

    m = modules.copy()
    m["is_significant_fdr0_05"] = (
        m["is_significant_fdr0_05"].fillna(False).astype(bool)
    )
    m = numeric(
        m,
        [
            "fdr_bh_within_family",
            "rate_delta_high_minus_low",
            "odds_ratio",
            "high_present_rate",
            "low_present_rate",
            "fisher_p",
        ],
    )
    m = m.sort_values(
        [
            "repFamily",
            "is_significant_fdr0_05",
            "fdr_bh_within_family",
            "rate_delta_high_minus_low",
            "odds_ratio",
        ],
        ascending=[True, False, True, False, False],
    )
    return m.drop_duplicates("repFamily", keep="first")


def make_selection(
    primary: pd.DataFrame,
    unrealized: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = primary.copy()
    p["literature_class"] = p["literature_class"].fillna("")
    p["candidate_context_class"] = p["candidate_context_class"].fillna("")
    p = numeric(
        p,
        [
            "candidate_priority_rank",
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
    )
    selected: list[dict[str, object]] = []
    used: set[str] = set()

    def pick(
        label: str,
        mask: pd.Series,
        sort_cols: list[str],
        ascending: list[bool],
    ) -> None:
        pool = p.loc[mask & ~p[ID].isin(used)].copy()
        if pool.empty:
            return

        pool = pool.sort_values(
            sort_cols,
            ascending=ascending,
            na_position="last",
        )
        row = pool.iloc[0].to_dict()
        row["selection_category"] = label
        row["selection_source"] = "M10_PRIMARY_DISCOVERY"
        selected.append(row)
        used.add(clean_text(row[ID]))

    candidate = p["classification"].isin(
        ["UNREVIEWED", "NOVEL_CANDIDATE"]
    )
    extension = p["classification"].eq("EXTENSION")
    shared = p["candidate_context_class"].eq("MULTI_CONTEXT_STRONG")
    selective = p["candidate_context_class"].eq(
        "CELL_LINE_SELECTIVE_STRONG"
    )

    pick(
        "HIGHEST_RANK_CANDIDATE",
        candidate,
        ["herv_functional_priority_rank"],
        [True],
    )
    pick(
        "CANDIDATE_SHARED_HIGH_SEQUENCE",
        candidate & shared,
        [
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
        [False, False, True],
    )
    pick(
        "CANDIDATE_CELL_LINE_SELECTIVE",
        candidate & selective,
        [
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
        [False, False, True],
    )
    pick(
        "EXTENSION_HIGH_PRIORITY",
        extension,
        [
            "herv_functional_priority_rank",
            "rank_pct_score_sequence_regulatory",
        ],
        [True, False],
    )

    cluster = candidate & p["nearest_gene_name"].fillna("").str.upper().isin(
        {"BSG", "BSG-AS1"}
    )
    pick(
        "CANDIDATE_LOCAL_BSG_CLUSTER",
        cluster,
        [
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
        [False, True],
    )

    u = unrealized.copy()
    u["classification"] = "COMPARISON_ONLY"
    u["literature_class"] = "Currently unrealized comparison"
    u["selection_category"] = "CURRENTLY_UNREALIZED_HIGH_SEQUENCE"
    u["selection_source"] = "M10_CURRENTLY_UNREALIZED"
    u = numeric(
        u,
        [
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
    )
    u = u.sort_values(
        [
            "rank_pct_score_sequence_regulatory",
            "rank_pct_score_final_representation",
            "herv_functional_priority_rank",
        ],
        ascending=[False, False, True],
    )
    for _, row in u.iterrows():
        if clean_text(row[ID]) not in used:
            selected.append(row.to_dict())
            used.add(clean_text(row[ID]))
            break

    selected_frame = pd.DataFrame(selected)
    selected_frame.insert(
        0,
        "representative_order",
        np.arange(1, len(selected_frame) + 1),
    )
    audit = selected_frame[
        [
            "representative_order",
            ID,
            "selection_category",
            "selection_source",
            "candidate_context_class",
            "literature_class",
            "candidate_priority_rank",
            "herv_functional_priority_rank",
        ]
    ].copy()
    return selected_frame, audit


def step_13() -> None:
    OUT_DIR__13.mkdir(parents=True, exist_ok=True)

    inputs = {
        "m03_sequence": M03_DIR / "03_within_family_sequence_comparison.tsv",
        "m03_modules": M03_DIR / "03_within_family_6mer_modules.tsv",
        "m05_route": M05_DIR / "05_route_decision.json",
        "m10_catalog": M10_DIR__13 / "10_prioritized_candidate_catalog.tsv",
        "m10_unrealized": (
            M10_DIR__13 / "10_prioritized_currently_unrealized_catalog.tsv"
        ),
        "m11_effects": M11_DIR / "11_frozen_rank_primary_effects.tsv",
        "curated_candidates": M12_DIR / "literature_curated_candidates.tsv",
        "evidence_ledger": M12_DIR / "literature_evidence_ledger.tsv",
        "m28_attribution": (
            M28_DIR / "28_selected_fer_feature_attribution.tsv"
        ),
        "m28_membership": (
            M28_DIR / "28_k6_bpe_validation_top1_membership.tsv"
        ),
        "source_table": SOURCE_TABLE__13,
    }

    primary = pd.read_csv(
        inputs["m10_catalog"], sep="\t", low_memory=False
    )
    unrealized = pd.read_csv(
        inputs["m10_unrealized"], sep="\t", low_memory=False
    )
    curated_all = pd.read_csv(
        inputs["curated_candidates"], sep="\t", low_memory=False
    )
    curated = curated_all[
        curated_all["candidate_set"].eq("M10_PRIMARY_DISCOVERY")
    ].copy()

    literature_cols = [
        ID,
        "classification",
        "literature_class",
        "classification_confidence",
        "evidence_id",
        "evidence_scope",
        "classification_reason",
        "manual_review_required",
        "novelty_claim_allowed",
    ]
    literature_cols = [
        c for c in literature_cols if c in curated.columns
    ]
    primary = primary.merge(
        curated[literature_cols].drop_duplicates(ID),
        on=ID,
        how="left",
    )
    selected, selection_audit = make_selection(primary, unrealized)
    selected_ids = [clean_text(x) for x in selected[ID]]

    source = read_source(selected_ids)
    selected = selected.merge(
        source,
        on=ID,
        how="left",
        suffixes=("", "_source"),
    )

    membership = pd.read_csv(
        inputs["m28_membership"],
        sep="\t",
        usecols=[
            ID,
            "score_k6_meta",
            "score_bpe_meta",
            "score_dual_no_mpra_meta",
            "k6_top1",
            "bpe_top1",
            "dual_no_mpra_top1",
        ],
    )
    selected = selected.merge(
        membership[membership[ID].isin(selected_ids)],
        on=ID,
        how="left",
    )

    seq = pd.read_csv(inputs["m03_sequence"], sep="\t")
    modules = top_family_module(
        pd.read_csv(inputs["m03_modules"], sep="\t")
    )
    selected = selected.merge(
        seq,
        on="repFamily",
        how="left",
        suffixes=("", "_within_family"),
    )
    selected = selected.merge(
        modules.add_prefix("module_"),
        left_on="repFamily",
        right_on="module_repFamily",
        how="left",
    )

    attribution = pd.read_csv(inputs["m28_attribution"], sep="\t")
    attribution = numeric(
        attribution,
        [
            "attribution_rank",
            "normalized_attribution",
            "mean_abs_grad_x_input",
        ],
    )
    attribution = attribution.sort_values(
        "attribution_rank"
    ).reset_index(drop=True)

    top_features = attribution.head(5).apply(
        lambda r: (
            f"{clean_text(r.get('feature'))}:"
            f"{float(r.get('normalized_attribution')):.4g}"
        ),
        axis=1,
    ).tolist()
    selected["global_attribution_top5_features"] = ";".join(top_features)
    selected["locus_specific_attribution_available"] = False
    selected["attribution_scope"] = (
        "GLOBAL_M28_SELECTED_FER_FEATURE_ATTRIBUTION"
    )

    m05_route = json.loads(
        inputs["m05_route"].read_text(encoding="utf-8")
    )
    sequence_metrics = m05_route["sequence_score_by_model"]
    aligned_model = "dual_no_mpra_meta"
    shuffled_model = "k6_shuffled_bpe_meta"
    aligned_hits = int(sequence_metrics[aligned_model]["top1_hits"])
    shuffled_hits = int(sequence_metrics[shuffled_model]["top1_hits"])

    selected["sequence_perturbation_status"] = (
        "GLOBAL_SHUFFLED_BPE_CONTROL_ONLY"
    )
    selected["sequence_perturbation_scope"] = (
        "M05 aligned-versus-shuffled BPE global control"
    )
    selected["bpe_aligned_model"] = aligned_model
    selected["bpe_shuffled_model"] = shuffled_model
    selected["bpe_aligned_top1_hits_global"] = aligned_hits
    selected["bpe_shuffled_top1_hits_global"] = shuffled_hits
    selected["bpe_aligned_minus_shuffled_top1_hits_global"] = (
        aligned_hits - shuffled_hits
    )
    selected["bpe_alignment_control_supported"] = (
        aligned_hits > shuffled_hits
    )

    chrom_rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        for eid in EIDS:
            mark_values = []
            active_count = 0
            repressive_count = 0
            for mark in MARKS:
                has = bool(row.get(f"tumor_{eid}_{mark}_has", False))
                bp = pd.to_numeric(
                    pd.Series([
                        row.get(f"tumor_{eid}_{mark}_total_overlap_bp", 0)
                    ]),
                    errors="coerce",
                ).iloc[0]
                mark_values.append(
                    f"{mark}={int(has)}:"
                    f"{0 if pd.isna(bp) else int(bp)}"
                )
                if has and mark in ACTIVE_MARKS:
                    active_count += 1
                if has and mark not in ACTIVE_MARKS:
                    repressive_count += 1

            chrom_rows.append({
                ID: row[ID],
                "representative_order": row["representative_order"],
                "eid": eid,
                "active_mark_count": active_count,
                "repressive_mark_count": repressive_count,
                "mark_overlap_profile": ";".join(mark_values),
                "active_supported": active_count > 0,
                "active_dominant": active_count > repressive_count,
            })

    chrom = pd.DataFrame(chrom_rows)
    write_tsv(chrom, OUT_CHROM)
    profiles = chrom.groupby(ID).apply(
        lambda g: "; ".join(
            f"{r.eid}:active{int(r.active_mark_count)}"
            f"/repressive{int(r.repressive_mark_count)}"
            for r in g.itertuples()
        ),
        include_groups=False,
    ).rename("tumor_chromatin_profile").reset_index()
    selected = selected.merge(profiles, on=ID, how="left")

    selected["nearby_gene_annotated"] = selected["nearest_gene_name"].apply(
        lambda value: bool(clean_text(value))
    )
    selected["literature_status_for_deep_dive"] = (
        selected["literature_class"].fillna("")
    )
    selected["selection_rule_is_predeclared"] = True

    selected_cols = [
        "representative_order",
        "selection_category",
        "selection_source",
        ID,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repFamily",
        "repeat_divergence",
        "length",
        "gc_content",
        "dist_to_tss",
        "nearest_gene_id",
        "nearest_gene_name",
        "nearest_gene_type",
        "candidate_context_class",
        "candidate_tier",
        "candidate_priority_rank",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "score_sequence_regulatory",
        "rank_pct_score_sequence_regulatory",
        "score_final_representation",
        "rank_pct_score_final_representation",
        "score_final_global",
        "tumor_active_support_breadth",
        "tumor_active_supported_eids",
        "tumor_strong_supported_eids",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "classification",
        "literature_class",
        "classification_confidence",
        "evidence_id",
        "evidence_scope",
        "classification_reason",
        "manual_review_required",
        "novelty_claim_allowed",
        "module_canonical_6mer",
        "module_rate_delta_high_minus_low",
        "module_odds_ratio",
        "module_fdr_bh_within_family",
        "module_is_significant_fdr0_05",
        "global_attribution_top5_features",
        "locus_specific_attribution_available",
        "attribution_scope",
        "sequence_perturbation_status",
        "sequence_perturbation_scope",
        "bpe_aligned_top1_hits_global",
        "bpe_shuffled_top1_hits_global",
        "bpe_aligned_minus_shuffled_top1_hits_global",
        "bpe_alignment_control_supported",
        "bpe_aligned_model",
        "bpe_shuffled_model",
        "score_k6_meta",
        "score_bpe_meta",
        "score_dual_no_mpra_meta",
        "k6_top1",
        "bpe_top1",
        "dual_no_mpra_top1",
        "tumor_chromatin_profile",
        "nearby_gene_annotated",
        "selection_rule_is_predeclared",
    ]
    selected_cols = [
        c for c in selected_cols if c in selected.columns
    ]
    deep = selected[selected_cols].copy()
    write_tsv(deep, OUT_MAIN)


    ledger = pd.read_csv(
        inputs["evidence_ledger"], sep="\t", low_memory=False
    )
    model_fields = [
        ID,
        "candidate_tier",
        "candidate_context_class",
        "literature_class",
        "classification",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "score_sequence_regulatory",
        "rank_pct_score_sequence_regulatory",
        "score_final_representation",
        "rank_pct_score_final_representation",
        "tumor_active_support_breadth",
        "tumor_active_supported_eids",
        "tumor_strong_supported_eids",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "nearest_gene_type",
        "direct_reference_evidence_count",
    ]
    model_fields = [
        c for c in model_fields if c in curated.columns
    ]
    review = ledger.merge(
        curated_all[model_fields].drop_duplicates(ID),
        on=ID,
        how="left",
        suffixes=("", "_catalog"),
    )

    if "literature_class" not in review.columns:
        review["literature_class"] = ""
    review["literature_class"] = review["literature_class"].fillna("")
    class_map = {
        "KNOWN": "Known",
        "EXTENSION": "Extension",
        "NOVEL_CANDIDATE": "Novel candidate",
        "UNREVIEWED": "Pending review",
        "COMPARISON_ONLY": "Comparison only",
    }
    missing_class = review["literature_class"].eq("")
    review.loc[missing_class, "literature_class"] = (
        review.loc[missing_class, "classification"]
        .map(class_map)
        .fillna("")
    )

    selection_fields = selection_audit[
        [ID, "representative_order", "selection_category"]
    ].copy()
    review = review.merge(selection_fields, on=ID, how="left")
    review["record_class"] = np.where(
        review["candidate_set"].eq("EXTERNAL_LITERATURE_ANCHOR"),
        "KNOWN_OR_EXTERNAL_ANCHOR",
        np.where(
            review["representative_order"].notna(),
            "REPRESENTATIVE_CANDIDATE",
            "M10_CANDIDATE",
        ),
    )
    review["coordinate"] = review.apply(
        lambda r: (
            f"{clean_text(r.get('chrom'))}:"
            f"{clean_text(r.get('start'))}-{clean_text(r.get('end'))}"
        ),
        axis=1,
    )
    review["manual_search_query"] = review.apply(
        lambda r: " OR ".join(
            q for q in [
                f'"{clean_text(r.get(ID))}"',
                f'"{clean_text(r.get("coordinate"))}"',
                (
                    f'"{clean_text(r.get("repName"))}" '
                    f'"{clean_text(r.get("nearest_gene_name"))}"'
                ),
            ]
            if q.strip('" ')
        ),
        axis=1,
    )
    reference_cols = [
        ID,
        "is_ccre_enhancer_like",
        "is_mpra_active_supported",
        "is_serve_gtex_supported",
        "gctx_roadmap_active_mark_presence_count",
        "gctx_roadmap_repressive_mark_presence_count",
        "dist_to_tss",
        "roadmap_DNase_fdr001_sample_count",
        "roadmap_H3K27ac_sample_count",
        "roadmap_H3K4me1_sample_count",
        "roadmap_H3K4me3_sample_count",
        "roadmap_H3K27me3_sample_count",
        "roadmap_H3K9me3_sample_count",
    ]
    reference = pd.read_parquet(
        SOURCE_TABLE__13, columns=reference_cols
    )
    reference = reference[
        reference[ID].isin(
            set(review[ID].astype(str)) | set(selected_ids)
        )
    ].copy()
    reference = reference.rename(columns={
        "is_ccre_enhancer_like": "reference_ccre_support",
        "is_mpra_active_supported": "reference_mpra_support",
        "is_serve_gtex_supported": "reference_serve_gtex_support",
        "gctx_roadmap_active_mark_presence_count": (
            "reference_roadmap_active_mark_count"
        ),
        "gctx_roadmap_repressive_mark_presence_count": (
            "reference_roadmap_repressive_mark_count"
        ),
    })
    review = review.merge(
        reference, on=ID, how="left", validate="one_to_one"
    )

    for col in [
        "reference_ccre_support",
        "reference_mpra_support",
        "reference_serve_gtex_support",
    ]:
        review[col] = review[col].fillna(False).astype(bool)

    review["reference_roadmap_active_mark_count"] = pd.to_numeric(
        review["reference_roadmap_active_mark_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    review["reference_roadmap_repressive_mark_count"] = pd.to_numeric(
        review["reference_roadmap_repressive_mark_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    review["reference_roadmap_active_support"] = (
        review["reference_roadmap_active_mark_count"].gt(0)
    )
    review["reference_evidence_count"] = review[
        [
            "reference_ccre_support",
            "reference_mpra_support",
            "reference_serve_gtex_support",
            "reference_roadmap_active_support",
        ]
    ].astype(int).sum(axis=1)

    review["gene_distance_bp"] = pd.to_numeric(
        review["dist_to_tss"], errors="coerce"
    ).abs()
    review["gene_linkage_type"] = "nearest only"
    review["gene_linkage_confidence"] = "LOW"
    review["gene_linkage_evidence"] = "nearest TSS annotation only"
    review["putative_target_gene_allowed"] = False

    doi_map = {
        "ERV_AML_2020": "10.1038/s41467-020-17206-4",
        "TE_CANCER_2019": "10.1038/s41588-019-0373-3",
    }
    review["evidence_doi"] = (
        review["evidence_id"].map(doi_map).fillna("")
    )
    review["evidence_pmid"] = ""
    review["literature_evidence_level"] = np.select(
        [
            review["exact_locus_reported"].fillna(False).astype(bool),
            review["literature_class"].astype(str).eq("Extension"),
            review["literature_class"].astype(str).eq("Novel candidate"),
        ],
        [
            "L2_EXACT_LOCUS_REPORTED_SOURCE_REVIEW_REQUIRED",
            "L3_RELATED_FAMILY_OR_SUBFAMILY",
            "L5_NO_RELEVANT_RECORD_IN_CURRENT_LEDGER",
        ],
        default="L5_NOT_CURATED",
    )

    unrealized_selected = selected[
        selected["selection_category"].eq(
            "CURRENTLY_UNREALIZED_HIGH_SEQUENCE"
        )
    ]
    if not unrealized_selected.empty:
        u = unrealized_selected.iloc[0]
        urow = {col: np.nan for col in review.columns}

        for col in review.columns:
            if col in u.index:
                urow[col] = u[col]

        uref = reference[reference[ID].eq(u[ID])]
        if not uref.empty:
            for col in reference.columns:
                if col != ID and col in urow:
                    urow[col] = uref.iloc[0][col]

        urow.update({
            "record_class": "CURRENTLY_UNREALIZED_COMPARISON",
            "candidate_set": "M10_CURRENTLY_UNREALIZED",
            "representative_order": u.get("representative_order"),
            "selection_category": "CURRENTLY_UNREALIZED_HIGH_SEQUENCE",
            "classification": "COMPARISON_ONLY",
            "literature_class": "Currently unrealized comparison",
            "classification_confidence": "NOT_ASSESSED",
            "evidence_id": "",
            "evidence_scope": "no direct literature curation performed",
            "evidence_type": "",
            "evidence_title": "",
            "evidence_year": "",
            "evidence_journal": "",
            "evidence_url": "",
            "evidence_summary": (
                "High model potential but no active support in the five "
                "assayed tumor contexts; retained as a biological comparison, "
                "not a novelty claim."
            ),
            "classification_reason": (
                "M10 currently-unrealized comparison set"
            ),
            "exact_locus_reported": False,
            "manual_review_required": True,
            "novelty_claim_allowed": False,
            "literature_evidence_level": "L5_NOT_CURATED_COMPARISON",
        })

        urow["coordinate"] = (
            f"{clean_text(u.get('chrom'))}:"
            f"{clean_text(u.get('start'))}-{clean_text(u.get('end'))}"
        )
        urow["reference_roadmap_active_support"] = bool(
            urow.get("reference_roadmap_active_mark_count", 0) > 0
        )
        urow["reference_evidence_count"] = (
            int(bool(urow.get("reference_ccre_support", False)))
            + int(bool(urow.get("reference_mpra_support", False)))
            + int(bool(urow.get("reference_serve_gtex_support", False)))
            + int(bool(urow["reference_roadmap_active_support"]))
        )
        urow["gene_distance_bp"] = (
            abs(float(urow["dist_to_tss"]))
            if pd.notna(urow.get("dist_to_tss"))
            else np.nan
        )
        urow["gene_linkage_type"] = "nearest only"
        urow["gene_linkage_confidence"] = "LOW"
        urow["gene_linkage_evidence"] = "nearest TSS annotation only"
        urow["putative_target_gene_allowed"] = False
        urow["manual_search_query"] = " OR ".join(
            q for q in [
                f'"{clean_text(u.get(ID))}"',
                f'"{urow["coordinate"]}"',
                (
                    f'"{clean_text(u.get("repName"))}" '
                    f'"{clean_text(u.get("nearest_gene_name"))}"'
                ),
            ]
            if q.strip('" ')
        )
        review = pd.concat(
            [review, pd.DataFrame([urow])],
            ignore_index=True,
        )

    review["representative_order"] = pd.to_numeric(
        review["representative_order"], errors="coerce"
    )
    review["herv_functional_priority_rank"] = pd.to_numeric(
        review["herv_functional_priority_rank"], errors="coerce"
    )
    review = review.sort_values(
        [
            "record_class",
            "representative_order",
            "herv_functional_priority_rank",
            ID,
        ],
        na_position="last",
    )

    review_cols = [
        "record_class",
        "representative_order",
        "selection_category",
        ID,
        "coordinate",
        "chrom",
        "start",
        "end",
        "repName",
        "repFamily",
        "nearest_gene_name",
        "nearest_gene_type",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "score_sequence_regulatory",
        "rank_pct_score_sequence_regulatory",
        "score_final_representation",
        "rank_pct_score_final_representation",
        "candidate_context_class",
        "tumor_active_support_breadth",
        "tumor_active_supported_eids",
        "tumor_strong_supported_eids",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "classification",
        "literature_class",
        "classification_confidence",
        "evidence_id",
        "evidence_scope",
        "evidence_type",
        "evidence_title",
        "evidence_year",
        "evidence_journal",
        "evidence_url",
        "evidence_summary",
        "classification_reason",
        "evidence_pmid",
        "evidence_doi",
        "literature_evidence_level",
        "exact_locus_reported",
        "manual_review_required",
        "novelty_claim_allowed",
        "reference_ccre_support",
        "reference_mpra_support",
        "reference_serve_gtex_support",
        "reference_roadmap_active_support",
        "reference_roadmap_active_mark_count",
        "reference_roadmap_repressive_mark_count",
        "roadmap_DNase_fdr001_sample_count",
        "roadmap_H3K27ac_sample_count",
        "roadmap_H3K4me1_sample_count",
        "roadmap_H3K4me3_sample_count",
        "roadmap_H3K27me3_sample_count",
        "roadmap_H3K9me3_sample_count",
        "reference_evidence_count",
        "gene_distance_bp",
        "gene_linkage_type",
        "gene_linkage_confidence",
        "gene_linkage_evidence",
        "putative_target_gene_allowed",
        "manual_search_query",
    ]
    review_cols = [
        c for c in review_cols if c in review.columns
    ]
    write_tsv(review[review_cols], OUT_REVIEW)


def main() -> None:
    step_10()
    step_13()


if __name__ == "__main__":
    main()
