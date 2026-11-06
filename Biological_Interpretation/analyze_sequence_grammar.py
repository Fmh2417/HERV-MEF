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
import math
import shutil
import numpy as np
import pandas as pd
import torch
from collections import Counter
from typing import (
    Any,
    Iterable,
)
from scipy.stats import (
    binomtest,
    fisher_exact,
)

# --------------------------------------------------------------------------
# step 28: Sequence grammar and representative loci
# --------------------------------------------------------------------------



# =============================================================================
# 1. Frozen paths and contracts
# =============================================================================

HERE = Path(__file__).resolve().parent
MODELING = (HERE.parent / "Modeling").resolve()

SCRIPT = "28_sequence_grammar_and_representative_loci"
VERSION = "round2_v1"

M00_DIR = MODELING / "output/00_build_canonical_data"
M00_TABLE = M00_DIR / "00_canonical_herv_mef_table.parquet"

M05_DIR = MODELING / "output/05_k6_bpe_complementarity_controls"
M05_PRED = M05_DIR / "05_validation_predictions.parquet"

M20_DIR = MODELING / "output/20_train_functional_evidence_reranker"
M20_ATTR = M20_DIR / "20_all_feature_attribution.tsv"

M23_DIR = MODELING / "output/23_generate_genomewide_herv_functional_ranking"
M23_ATLAS = M23_DIR / "23_herv_genomewide_functional_ranking.parquet"

M24_DIR = MODELING / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes"
M24_PROJECTION = M24_DIR / "24_tumor_projection_all_herv.parquet"

M25_DIR = HERE / "output/25_evolutionary_interpretation"
M25_MATCHED = M25_DIR / "25_final_top1_matched_background.tsv"
M25_TOP30 = M25_DIR / "25_top30_reactivation_candidate_evolution.tsv"

M26_DIR = HERE / "output/26_host_regulatory_mechanism_interpretation"
M26_PAIRS = M26_DIR / "26_matched_top1_pairs.tsv"
M26_TOP30 = M26_DIR / "26_top30_regulatory_mechanism.tsv"

M27_DIR = HERE / "output/27_reference_to_tumor_context_interpretation"
M27_TOP1 = M27_DIR / "27_top1_cell_context_annotation.tsv"
M27_TOP30 = M27_DIR / "27_top30_cell_context_annotation.tsv"

OUT = HERE / "output/28_sequence_grammar_and_representative_loci"
OUT_TOP_MATCHED = OUT / "28_final_top1_vs_matched_6mer.tsv"
OUT_TUMOR_MATCHED = OUT / "28_tumor_supported_vs_unsupported_matched_6mer.tsv"
OUT_BRANCH_MEMBERSHIP = OUT / "28_k6_bpe_validation_top1_membership.tsv"
OUT_BRANCH_KMER = OUT / "28_k6_vs_bpe_specific_6mer.tsv"
OUT_FER_ATTR = OUT / "28_selected_fer_feature_attribution.tsv"
OUT_REPRESENTATIVE = OUT / "28_representative_loci.tsv"


ID = "herv_internal_id"
EXPECTED_N = 480_306
TOP1_K = int(math.floor(EXPECTED_N * 0.01))
VALID_N = 72_048
VALID_TOP1_K = int(math.floor(VALID_N * 0.01))
EXPECTED_GLOBAL = "R4_ALL_COMPACT_bs64_epoch_24"
EXPECTED_FER = "A3_ALL38_PRIMARY_epoch_01"
K = 6
MIN_GROUP_N = 20
REPORT_TOP_KMERS = 30

TUMOR_EIDS = ["E114", "E115", "E117", "E118", "E123"]
CELL_LINES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

M23_COLS = [
    ID, "chrom", "start", "end", "strand",
    "repName", "repClass", "repFamily", "repeat_divergence",
    "length", "gc_content", "dist_to_tss",
    "nearest_gene_id", "nearest_gene_name", "nearest_gene_type",
    "split",
    "score_sequence_regulatory", "score_final_representation",
    "score_final_global", "global_rank",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
]

M24_COLS = [
    ID, "potential_reactivation_candidate",
    "tumor_epigenomic_support_score",
    "tumor_active_support_breadth",
    "tumor_active_dominant_breadth",
    "tumor_repressive_support_breadth",
    "tumor_cell_line_support_class",
    "tumor_active_supported_eids",
]
for _eid in TUMOR_EIDS:
    M24_COLS.extend([
        f"tumor_{_eid}_any_active_support",
        f"tumor_{_eid}_active_mark_count",
        f"tumor_{_eid}_repressive_mark_count",
    ])

M27_CONTEXT_COLS = [
    ID,
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
for _eid in TUMOR_EIDS:
    M27_CONTEXT_COLS.extend([
        f"tumor_{_eid}_active_mark_count",
        f"tumor_{_eid}_repressive_mark_count",
        f"tumor_{_eid}_supported_marks",
    ])


# =============================================================================
# 2. Utilities
# =============================================================================



def prepare_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)


def atomic_tsv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def atomic_json(obj: Any, path: Path) -> None:
    def cv(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            x = float(v)
            return x if math.isfinite(x) else None
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        return v

    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(cv(obj), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    back = np.empty(m, dtype=float)
    back[order] = q
    out[np.flatnonzero(ok)] = back
    return out


def odds_ratio_haldane(a: int, b: int, c: int, d: int) -> float:
    # Stable finite descriptive OR using 0.5 correction.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


# =============================================================================
# 3. Sequence grammar
# =============================================================================

_RC = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def canonical_kmer(kmer: str) -> str:
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def sequence_kmer_presence(seq: str) -> set[str]:
    seq = str(seq).upper()
    out: set[str] = set()
    for i in range(0, len(seq) - K + 1):
        token = seq[i:i + K]
        if all(base in "ACGT" for base in token):
            out.add(canonical_kmer(token))
    return out


def build_presence_map(sequence_df: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for j, row in enumerate(sequence_df[[ID, "sequence"]].itertuples(index=False), 1):
        result[str(row[0])] = sequence_kmer_presence(str(row[1]))
    return result


def paired_kmer_test(
    pairs: pd.DataFrame,
    left_col: str,
    right_col: str,
    presence: dict[str, set[str]],
    comparison: str,
) -> pd.DataFrame:
    left_ids = pairs[left_col].astype(str).tolist()
    right_ids = pairs[right_col].astype(str).tolist()

    left_count = Counter()
    right_count = Counter()
    left_only = Counter()
    right_only = Counter()
    both = Counter()

    for l_id, r_id in zip(left_ids, right_ids):
        l = presence[l_id]
        r = presence[r_id]
        for kmer in l:
            left_count[kmer] += 1
        for kmer in r:
            right_count[kmer] += 1
        for kmer in l & r:
            both[kmer] += 1
        for kmer in l - r:
            left_only[kmer] += 1
        for kmer in r - l:
            right_only[kmer] += 1

    kmers = sorted(set(left_count) | set(right_count))
    rows = []
    n = len(left_ids)
    for kmer in kmers:
        lo = int(left_only[kmer])
        ro = int(right_only[kmer])
        discordant = lo + ro
        p = (
            float(binomtest(lo, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant else 1.0
        )
        lp = int(left_count[kmer])
        rp = int(right_count[kmer])
        rows.append({
            "comparison": comparison,
            "canonical_6mer": kmer,
            "reverse_complement": reverse_complement(kmer),
            "pair_n": n,
            "left_present_n": lp,
            "right_present_n": rp,
            "left_present_rate": lp / n,
            "right_present_rate": rp / n,
            "rate_delta_left_minus_right": (lp - rp) / n,
            "discordant_left_only_n": lo,
            "discordant_right_only_n": ro,
            "discordant_pair_n": discordant,
            "paired_odds_ratio_haldane": (lo + 0.5) / (ro + 0.5),
            "exact_discordant_binomial_p": p,
        })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = bh_fdr(result["exact_discordant_binomial_p"])
    return result.sort_values(
        ["fdr_bh", "paired_odds_ratio_haldane", "canonical_6mer"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def unpaired_kmer_test(
    left_ids: list[str],
    right_ids: list[str],
    presence: dict[str, set[str]],
    comparison: str,
) -> pd.DataFrame:
    left_ids = list(dict.fromkeys(map(str, left_ids)))
    right_ids = list(dict.fromkeys(map(str, right_ids)))

    lc = Counter()
    rc = Counter()
    for identifier in left_ids:
        for kmer in presence[identifier]:
            lc[kmer] += 1
    for identifier in right_ids:
        for kmer in presence[identifier]:
            rc[kmer] += 1

    rows = []
    nl, nr = len(left_ids), len(right_ids)
    for kmer in sorted(set(lc) | set(rc)):
        a = int(lc[kmer])
        c = int(rc[kmer])
        b = nl - a
        d = nr - c
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        rows.append({
            "comparison": comparison,
            "canonical_6mer": kmer,
            "reverse_complement": reverse_complement(kmer),
            "left_n": nl,
            "right_n": nr,
            "left_present_n": a,
            "right_present_n": c,
            "left_present_rate": a / nl,
            "right_present_rate": c / nr,
            "rate_delta_left_minus_right": a / nl - c / nr,
            "odds_ratio_haldane": odds_ratio_haldane(a, b, c, d),
            "fisher_two_sided_p": float(p),
        })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = bh_fdr(result["fisher_two_sided_p"])
    return result.sort_values(
        ["fdr_bh", "odds_ratio_haldane", "canonical_6mer"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# 4. Load upstream data
# =============================================================================


def load_core() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = pd.read_parquet(M23_ATLAS, columns=M23_COLS)
    atlas[ID] = atlas[ID].astype(str)

    proj = pd.read_parquet(M24_PROJECTION, columns=M24_COLS)
    proj[ID] = proj[ID].astype(str)
    table = atlas.merge(proj, on=ID, how="left", validate="one_to_one", sort=False)
    top1 = table.loc[
        table["herv_functional_priority_rank"].astype(int).le(TOP1_K)
    ].copy()
    context = pd.read_csv(M27_TOP1, sep="\t", low_memory=False)
    context[ID] = context[ID].astype(str)
    top1 = top1.merge(
        context[M27_CONTEXT_COLS],
        on=ID, how="left", validate="one_to_one", sort=False,
        suffixes=("", "_m27"),
    )
    return table, top1, context


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    m25 = pd.read_csv(M25_MATCHED, sep="\t", low_memory=False)

    m26 = pd.read_csv(M26_PAIRS, sep="\t", low_memory=False)
    return m25, m26


def load_m05_branch_membership() -> pd.DataFrame:
    m05_long = pd.read_parquet(
        M05_PRED, columns=["model", ID, "score_sequence_mean"]
    )
    m05_long[ID] = m05_long[ID].astype(str)
    pred = (
        m05_long.pivot(index=ID, columns="model", values="score_sequence_mean")
        .rename(columns=lambda col: f"score_{col}")
        .reset_index()
    )

    def top_set(score: str) -> set[str]:
        ranked = pred[[ID, score]].sort_values(
            [score, ID], ascending=[False, True], kind="mergesort"
        )
        return set(ranked.head(VALID_TOP1_K)[ID].astype(str))

    k6 = top_set("score_k6_meta")
    bpe = top_set("score_bpe_meta")
    dual = top_set("score_dual_no_mpra_meta")

    out = pred.copy()
    out["k6_top1"] = out[ID].isin(k6)
    out["bpe_top1"] = out[ID].isin(bpe)
    out["dual_no_mpra_top1"] = out[ID].isin(dual)
    out["k6_bpe_class"] = np.select(
        [
            out["k6_top1"] & out["bpe_top1"],
            out["k6_top1"] & ~out["bpe_top1"],
            ~out["k6_top1"] & out["bpe_top1"],
        ],
        ["K6_BPE_SHARED", "K6_SPECIFIC", "BPE_SPECIFIC"],
        default="NEITHER_TOP1",
    )
    return out


def load_needed_sequences(ids: set[str]) -> pd.DataFrame:
    # Deliberately read only ID+sequence. No FANTOM5 label or training target.
    seq = pd.read_parquet(M00_TABLE, columns=[ID, "sequence"])
    seq[ID] = seq[ID].astype(str)
    seq = seq.loc[seq[ID].isin(ids)].copy()
    return seq


# =============================================================================
# 5. Representative loci and integrated evidence
# =============================================================================

def _pick_distinct(
    group: pd.DataFrame,
    n: int,
    used_ids: set[str],
    used_repnames: set[str],
    distinct_col: str | None = None,
    used_distinct: set[str] | None = None,
) -> list[pd.Series]:
    picked: list[pd.Series] = []
    ordered = group.sort_values(
        ["herv_functional_priority_rank", ID],
        ascending=[True, True],
        kind="mergesort",
    )
    # pass 1: prefer distinct repName and optional group key
    for _, row in ordered.iterrows():
        identifier = str(row[ID])
        repname = str(row["repName"])
        dvalue = str(row[distinct_col]) if distinct_col else ""
        if identifier in used_ids:
            continue
        if repname in used_repnames:
            continue
        if distinct_col and used_distinct is not None and dvalue in used_distinct:
            continue
        picked.append(row)
        used_ids.add(identifier)
        used_repnames.add(repname)
        if distinct_col and used_distinct is not None:
            used_distinct.add(dvalue)
        if len(picked) >= n:
            return picked
    # pass 2: only ID uniqueness required
    for _, row in ordered.iterrows():
        identifier = str(row[ID])
        if identifier in used_ids:
            continue
        picked.append(row)
        used_ids.add(identifier)
        used_repnames.add(str(row["repName"]))
        if distinct_col and used_distinct is not None:
            used_distinct.add(str(row[distinct_col]))
        if len(picked) >= n:
            break
    return picked


def select_representatives(top1: pd.DataFrame) -> pd.DataFrame:
    candidates = top1.loc[top1["potential_reactivation_candidate"].astype(bool)].copy()
    used_ids: set[str] = set()
    used_repnames: set[str] = set()
    rows: list[dict[str, Any]] = []

    pan = candidates.loc[candidates["tumor_active_support_breadth"].astype(int).eq(5)]
    for row in _pick_distinct(pan, 2, used_ids, used_repnames):
        rows.append({"representative_category": "PAN_TUMOR_BREADTH5", **row.to_dict()})

    specific = candidates.loc[
        candidates["tumor_active_support_breadth"].astype(int).eq(1)
    ].copy()
    specific["specific_eid"] = specific["tumor_active_supported_eids"].astype(str)
    used_eids: set[str] = set()
    for row in _pick_distinct(
        specific, 2, used_ids, used_repnames,
        distinct_col="specific_eid", used_distinct=used_eids
    ):
        rows.append({"representative_category": "CELL_LINE_SPECIFIC", **row.to_dict()})

    multi = candidates.loc[
        candidates["tumor_active_support_breadth"].astype(int).between(2, 4)
    ].copy()
    used_patterns: set[str] = set()
    for row in _pick_distinct(
        multi, 2, used_ids, used_repnames,
        distinct_col="tumor_active_support_pattern", used_distinct=used_patterns
    ):
        rows.append({"representative_category": "INTERMEDIATE_MULTI_TUMOR", **row.to_dict()})

    result = pd.DataFrame(rows)
    result.insert(0, "representative_order", np.arange(1, len(result) + 1))
    result["literature_status"] = "not_assessed_in_M28"
    result["selection_used_literature"] = False
    return result


def annotate_representative_kmers(
    reps: pd.DataFrame,
    presence: dict[str, set[str]],
    top1_kmer: pd.DataFrame,
    tumor_kmer: pd.DataFrame,
) -> pd.DataFrame:
    # Candidate grammar dictionary: significant and enriched in left group.
    sig_top = top1_kmer.loc[
        (top1_kmer["fdr_bh"] < 0.05)
        & (top1_kmer["rate_delta_left_minus_right"] > 0)
    ].head(200)
    sig_tumor = tumor_kmer.loc[
        (tumor_kmer["fdr_bh"] < 0.05)
        & (tumor_kmer["rate_delta_left_minus_right"] > 0)
    ].head(200)

    top_order = sig_top["canonical_6mer"].astype(str).tolist()
    tumor_order = sig_tumor["canonical_6mer"].astype(str).tolist()

    top_hits = []
    tumor_hits = []
    for identifier in reps[ID].astype(str):
        kmers = presence[identifier]
        top_hits.append(";".join([k for k in top_order if k in kmers][:10]))
        tumor_hits.append(";".join([k for k in tumor_order if k in kmers][:10]))
    out = reps.copy()
    out["top1_enriched_6mer_patterns_present"] = top_hits
    out["tumor_support_enriched_6mer_patterns_present"] = tumor_hits
    out["6mer_interpretation"] = (
        "candidate cis-regulatory sequence grammar; not TF-binding proof"
    )
    return out


def load_fer_attribution() -> pd.DataFrame:
    arm, epoch = EXPECTED_FER.rsplit("_epoch_", 1)
    attr = pd.read_csv(M20_ATTR, sep="\t", low_memory=False)
    attr = attr.loc[
        attr["arm"].eq(arm) & attr["epoch"].eq(int(epoch))
    ].copy()
    attr["normalized_attribution"] = pd.to_numeric(
        attr["normalized_attribution"], errors="raise"
    )
    attr = attr.sort_values(
        ["normalized_attribution", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    attr.insert(0, "attribution_rank", np.arange(1, len(attr) + 1))
    attr["interpretation"] = (
        "global selected-FER model attribution; not locus-specific and not causal"
    )
    return attr


def build_top30_integrated() -> pd.DataFrame:
    base = pd.read_csv(M27_TOP30, sep="\t", low_memory=False)
    base[ID] = base[ID].astype(str)

    for path, prefix in [(M25_TOP30, "m25_"), (M26_TOP30, "m26_")]:
        extra = pd.read_csv(path, sep="\t", low_memory=False)
        extra[ID] = extra[ID].astype(str)
        selected = [ID]
        rename = {}
        for col in extra.columns:
            if col == ID:
                continue
            if col not in base.columns:
                selected.append(col)
                rename[col] = prefix + col
        if len(selected) > 1:
            base = base.merge(
                extra[selected].rename(columns=rename),
                on=ID, how="left", validate="one_to_one", sort=False
            )
    return base


# =============================================================================
# 6. Human-readable report
# =============================================================================

def md_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    view = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    if view.empty:
        return "_No rows._"
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


# =============================================================================
# 7. Main
# =============================================================================

def step_28() -> None:
    prepare_output()


    top1 = load_core()[1]
    m25_pairs, m26_pairs = load_pairs()
    branch = load_m05_branch_membership()

    # Needed sequence IDs only.
    ids: set[str] = set()
    ids.update(m25_pairs["top1_herv_internal_id"].astype(str))
    ids.update(m25_pairs["matched_background_herv_internal_id"].astype(str))
    ids.update(m26_pairs["supported_herv_internal_id"].astype(str))
    ids.update(m26_pairs["unsupported_herv_internal_id"].astype(str))
    ids.update(
        branch.loc[
            branch["k6_bpe_class"].isin(
                ["K6_SPECIFIC", "BPE_SPECIFIC", "K6_BPE_SHARED"]
            ),
            ID,
        ].astype(str)
    )

    # Representatives are selected before sequence loading so their IDs are included.
    reps = select_representatives(top1)
    ids.update(reps[ID].astype(str))

    seq = load_needed_sequences(ids)
    presence = build_presence_map(seq)

    top1_kmer = paired_kmer_test(
        m25_pairs,
        "top1_herv_internal_id",
        "matched_background_herv_internal_id",
        presence,
        "FINAL_TOP1_vs_MATCHED_NON_TOP1",
    )
    tumor_kmer = paired_kmer_test(
        m26_pairs,
        "supported_herv_internal_id",
        "unsupported_herv_internal_id",
        presence,
        "TUMOR_SUPPORTED_vs_UNSUPPORTED_FINAL_TOP1_MATCHED",
    )

    k6_specific = branch.loc[
        branch["k6_bpe_class"].eq("K6_SPECIFIC"), ID
    ].astype(str).tolist()
    bpe_specific = branch.loc[
        branch["k6_bpe_class"].eq("BPE_SPECIFIC"), ID
    ].astype(str).tolist()

    branch_kmer = unpaired_kmer_test(
        k6_specific,
        bpe_specific,
        presence,
        "K6_SPECIFIC_vs_BPE_SPECIFIC_VALIDATION_TOP1",
    )

    reps = annotate_representative_kmers(
        reps, presence, top1_kmer, tumor_kmer
    )

    fer_attr = load_fer_attribution()

    atomic_tsv(top1_kmer, OUT_TOP_MATCHED)
    atomic_tsv(tumor_kmer, OUT_TUMOR_MATCHED)
    atomic_tsv(branch, OUT_BRANCH_MEMBERSHIP)
    atomic_tsv(branch_kmer, OUT_BRANCH_KMER)
    atomic_tsv(fer_attr, OUT_FER_ATTR)
    atomic_tsv(reps, OUT_REPRESENTATIVE)


# --------------------------------------------------------------------------
# step 03: Analyze the sequence grammar
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = PROJECT_ROOT / "Biological_Interpretation"
Modeling = PROJECT_ROOT / "Modeling"

SOURCE_TABLE = BIOLOGY_ROOT / 'output/01_build_biological_source_table/01_biological_source_table.parquet'
M00_TABLE__03 = Modeling / 'output/00_build_canonical_data/00_canonical_herv_mef_table.parquet'
M28_DIR = BIOLOGY_ROOT / "output/28_sequence_grammar_and_representative_loci"
TOP1_6MER = M28_DIR / "28_final_top1_vs_matched_6mer.tsv"
TUMOR_6MER = M28_DIR / "28_tumor_supported_vs_unsupported_matched_6mer.tsv"
BRANCH_6MER = M28_DIR / "28_k6_vs_bpe_specific_6mer.tsv"
BRANCH_MEMBERSHIP = M28_DIR / "28_k6_bpe_validation_top1_membership.tsv"
M25_MATCHING_BALANCE = BIOLOGY_ROOT / 'output/25_evolutionary_interpretation/25_final_top1_matching_balance.tsv'

OUT_DIR = BIOLOGY_ROOT / "output/03_analyze_sequence_grammar"

OUT_COMPARISON_SUMMARY = OUT_DIR / "03_sequence_grammar_comparison_summary.tsv"
OUT_TOP1_SIGNIFICANT = OUT_DIR / "03_top1_significant_6mers.tsv"


OUT_EFFECT_THRESHOLDS = OUT_DIR / "03_significant_6mer_effect_thresholds.tsv"
OUT_WITHIN_FAMILY = OUT_DIR / "03_within_family_sequence_comparison.tsv"
OUT_WITHIN_FAMILY_MODULES = OUT_DIR / "03_within_family_6mer_modules.tsv"


ID_COL = "herv_internal_id"
TOP1_FRACTION = 0.01
KMER_K = 6
CANONICAL_KMER_N = (4 ** KMER_K + 4 ** (KMER_K // 2)) // 2
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
WITHIN_FAMILY_QUANTILE = 0.20
MIN_WITHIN_FAMILY_GROUP_N = 20

COMPARISONS = [
    ("TOP1_vs_MATCHED_NON_TOP1", "pair_n", "pair_n"),
    ("TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1", "pair_n", "pair_n"),
    ("K6_SPECIFIC_vs_BPE_SPECIFIC", "left_n", "right_n"),
]

WITHIN_FAMILY_SOURCE_COLUMNS = [
    ID_COL,
    "repName",
    "repFamily",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
    "score_sequence_regulatory",
    "rank_pct_score_sequence_regulatory",
    "length",
    "gc_content",
    "repeat_divergence",
    "dist_to_tss",
]








def summarize_universe(frame: pd.DataFrame) -> dict[str, int | float]:
    rank = pd.to_numeric(frame["herv_functional_priority_rank"], errors="coerce")
    universe_n = len(frame)
    top1_n = int(universe_n * TOP1_FRACTION)
    return {
        "universe_n": universe_n,
        "top1_fraction": TOP1_FRACTION,
        "top1_n": top1_n,
        "top1_locus_n": int((rank <= top1_n).sum()),
    }


def load_source_for_within_family(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=WITHIN_FAMILY_SOURCE_COLUMNS)
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame["repFamily"] = frame["repFamily"].fillna("Unknown").astype(str)
    frame["repName"] = frame["repName"].fillna("Unknown").astype(str)

    for column in [
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "score_sequence_regulatory",
        "rank_pct_score_sequence_regulatory",
        "length",
        "gc_content",
        "repeat_divergence",
        "dist_to_tss",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["abs_dist_to_tss"] = frame["dist_to_tss"].abs()
    frame["_family_rank_pct"] = frame.groupby("repFamily")[
        "herv_functional_priority_score"
    ].rank(method="average", pct=True)
    frame["_family_copy_n"] = frame.groupby("repFamily")[ID_COL].transform("size")
    return frame


def load_sequences(path: Path, ids: set[str]) -> pd.DataFrame:
    sequence = pd.read_parquet(path, columns=[ID_COL, "sequence"])
    sequence[ID_COL] = sequence[ID_COL].astype(str)
    sequence["sequence"] = sequence["sequence"].astype(str).str.upper()
    return sequence.loc[sequence[ID_COL].isin(ids)].copy()


def bh_fdr__03(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna()
    if not valid.any():
        return result

    arr = numeric.loc[valid].to_numpy(dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    n = len(ranked)
    corrected = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    indices = numeric.loc[valid].index.to_numpy()[order]
    result.loc[indices] = np.clip(corrected, 0.0, 1.0)
    return result


def load_6mer_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_significant_fdr0_05"] = frame["fdr_bh"] < FDR_THRESHOLD
    delta = pd.to_numeric(
        frame["rate_delta_left_minus_right"], errors="coerce"
    )
    frame["absolute_rate_delta"] = delta.abs()
    frame["direction"] = np.select(
        [delta > 0, delta < 0],
        ["LEFT_ENRICHED", "RIGHT_ENRICHED"],
        default="EQUAL",
    )
    return frame


def observed_sample_n(frame: pd.DataFrame, column: str) -> int:
    return int(pd.to_numeric(frame[column], errors="coerce").max())


def comparison_summary(
    name: str,
    frame: pd.DataFrame,
    left_size_column: str,
    right_size_column: str,
) -> dict[str, object]:
    significant = frame[frame["is_significant_fdr0_05"]]
    return {
        "comparison": name,
        "left_sample_n": observed_sample_n(frame, left_size_column),
        "right_sample_n": observed_sample_n(frame, right_size_column),
        "canonical_6mers_tested": len(frame),
        "significant_fdr0_05": len(significant),
        "left_enriched_significant": int(
            significant["direction"].eq("LEFT_ENRICHED").sum()
        ),
        "right_enriched_significant": int(
            significant["direction"].eq("RIGHT_ENRICHED").sum()
        ),
        "median_absolute_rate_delta_significant": (
            float(significant["absolute_rate_delta"].median())
            if len(significant) else None
        ),
        "maximum_absolute_rate_delta_significant": (
            float(significant["absolute_rate_delta"].max())
            if len(significant) else None
        ),
        "minimum_fdr": float(frame["fdr_bh"].min()),
    }


def effect_threshold_summary(
    comparison_name: str, frame: pd.DataFrame
) -> list[dict[str, object]]:
    significant = frame[frame["is_significant_fdr0_05"]]
    rows = []
    for threshold in EFFECT_THRESHOLDS:
        retained = significant[
            significant["absolute_rate_delta"] >= threshold
        ]
        rows.append({
            "comparison": comparison_name,
            "absolute_rate_delta_threshold": threshold,
            "significant_6mers_meeting_threshold": len(retained),
            "left_enriched": int(
                retained["direction"].eq("LEFT_ENRICHED").sum()
            ),
            "right_enriched": int(
                retained["direction"].eq("RIGHT_ENRICHED").sum()
            ),
        })
    return rows


def canonical_kmers(sequence: str, k: int = KMER_K) -> set[str]:
    sequence = str(sequence).upper()
    kmers = set()
    for index in range(len(sequence) - k + 1):
        kmer = sequence[index:index + k]
        if set(kmer) - set("ACGT"):
            continue
        reverse = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        kmers.add(min(kmer, reverse))
    return kmers


def within_family_sequence_analysis(
    source: pd.DataFrame, sequence: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = source.merge(
        sequence, on=ID_COL, how="inner", validate="one_to_one"
    )
    eligible = merged["_family_copy_n"].ge(MIN_WITHIN_FAMILY_GROUP_N)
    merged["_high"] = (
        eligible
        & merged["_family_rank_pct"].gt(1.0 - WITHIN_FAMILY_QUANTILE)
    )
    merged["_low"] = (
        eligible
        & merged["_family_rank_pct"].le(WITHIN_FAMILY_QUANTILE)
    )

    rows = []
    module_rows = []
    family_groups = merged.loc[eligible].groupby("repFamily", sort=True)

    for family, group in family_groups:
        high = group[group["_high"]]
        low = group[group["_low"]]
        if len(high) < 2 or len(low) < 2:
            continue

        high_kmers = [canonical_kmers(seq) for seq in high["sequence"]]
        low_kmers = [canonical_kmers(seq) for seq in low["sequence"]]
        high_union = set().union(*high_kmers)
        low_union = set().union(*low_kmers)

        for kmer in sorted(high_union | low_union):
            high_present = sum(kmer in values for values in high_kmers)
            low_present = sum(kmer in values for values in low_kmers)
            table = [
                [high_present, len(high) - high_present],
                [low_present, len(low) - low_present],
            ]
            odds, pvalue = fisher_exact(table, alternative="two-sided")
            high_rate = high_present / len(high)
            low_rate = low_present / len(low)

            module_rows.append({
                "repFamily": family,
                "high_n": len(high),
                "low_n": len(low),
                "canonical_6mer": kmer,
                "high_present_n": high_present,
                "low_present_n": low_present,
                "high_present_rate": high_rate,
                "low_present_rate": low_rate,
                "rate_delta_high_minus_low": high_rate - low_rate,
                "odds_ratio": float(odds),
                "fisher_p": float(pvalue),
            })

        rows.append({
            "repFamily": family,
            "family_n": int(group[ID_COL].nunique()),
            "high_n": len(high),
            "low_n": len(low),
            "high_score_median": float(
                high["herv_functional_priority_score"].median()
            ),
            "low_score_median": float(
                low["herv_functional_priority_score"].median()
            ),
            "high_sequence_score_median": float(
                high["score_sequence_regulatory"].median()
            ),
            "low_sequence_score_median": float(
                low["score_sequence_regulatory"].median()
            ),
            "high_divergence_median": float(
                high["repeat_divergence"].median()
            ),
            "low_divergence_median": float(
                low["repeat_divergence"].median()
            ),
            "high_length_median": float(high["length"].median()),
            "low_length_median": float(low["length"].median()),
            "high_gc_median": float(high["gc_content"].median()),
            "low_gc_median": float(low["gc_content"].median()),
            "high_abs_tss_median": float(high["abs_dist_to_tss"].median()),
            "low_abs_tss_median": float(low["abs_dist_to_tss"].median()),
        })

    comparisons = pd.DataFrame(rows)
    modules = pd.DataFrame(module_rows)

    if not modules.empty:
        modules["fdr_bh_within_family"] = modules.groupby(
            "repFamily", sort=False
        )["fisher_p"].transform(bh_fdr__03)
        modules["is_significant_fdr0_05"] = (
            modules["fdr_bh_within_family"].lt(FDR_THRESHOLD)
        )
        significant = modules["is_significant_fdr0_05"]
        delta = modules["rate_delta_high_minus_low"]
        modules["module_class"] = np.select(
            [significant & delta.gt(0), significant & delta.lt(0)],
            ["HIGH_ENRICHED", "LOW_ENRICHED"],
            default="NOT_SIGNIFICANT",
        )
        modules = modules.sort_values(
            ["fdr_bh_within_family", "rate_delta_high_minus_low"],
            ascending=[True, False],
        ).reset_index(drop=True)

    status = pd.DataFrame([{
        "analysis": "sequence_attribution",
        "status": "NOT_AVAILABLE_IN_FROZEN_M28",
        "source": (
            "M28 has global FER feature attribution only; "
            "no locus-level sequence attribution output"
        ),
        "interpretation": (
            "within-family 6-mer frequency comparisons; "
            "no model gradient attribution"
        ),
    }])
    return comparisons, modules, status


def parse_boolean_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.eq("true")


def summarize_membership(frame: pd.DataFrame) -> dict[str, object]:
    k6 = parse_boolean_column(frame["k6_top1"])
    bpe = parse_boolean_column(frame["bpe_top1"])
    k6_n = int(k6.sum())
    bpe_n = int(bpe.sum())
    shared_n = int((k6 & bpe).sum())
    union_n = int((k6 | bpe).sum())

    return {
        "validation_n": len(frame),
        "k6_top1_n": k6_n,
        "bpe_top1_n": bpe_n,
        "shared_top1_n": shared_n,
        "k6_specific_n": int((k6 & ~bpe).sum()),
        "bpe_specific_n": int((~k6 & bpe).sum()),
        "union_top1_n": union_n,
        "jaccard_k6_bpe_top1": shared_n / union_n,
        "shared_fraction_of_k6_top1": shared_n / k6_n,
        "shared_fraction_of_bpe_top1": shared_n / bpe_n,
    }


def set_overlap_table(significant_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for left_name, left in significant_sets.items():
        for right_name, right in significant_sets.items():
            union = left | right
            rows.append({
                "left_set": left_name,
                "right_set": right_name,
                "left_n": len(left),
                "right_n": len(right),
                "intersection_n": len(left & right),
                "union_n": len(union),
                "jaccard": len(left & right) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def step_03() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    within_source = load_source_for_within_family(SOURCE_TABLE)
    sequence = load_sequences(M00_TABLE__03, set(within_source[ID_COL]))

    top1 = load_6mer_table(
        pd.read_csv(TOP1_6MER, sep="\t", low_memory=False)
    )
    tumor = load_6mer_table(
        pd.read_csv(TUMOR_6MER, sep="\t", low_memory=False)
    )
    branch = load_6mer_table(
        pd.read_csv(BRANCH_6MER, sep="\t", low_memory=False)
    )

    comparison_frames = {
        "TOP1_vs_MATCHED_NON_TOP1": top1,
        "TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1": tumor,
        "K6_SPECIFIC_vs_BPE_SPECIFIC": branch,
    }
    comparison_rows = [
        comparison_summary(name, comparison_frames[name], left, right)
        for name, left, right in COMPARISONS
    ]
    comparison = pd.DataFrame(comparison_rows)

    top1_significant = top1[
        top1["is_significant_fdr0_05"]
    ].sort_values(
        ["fdr_bh", "absolute_rate_delta"], ascending=[True, False]
    ).reset_index(drop=True)


    effect_thresholds = pd.DataFrame(
        effect_threshold_summary("TOP1_vs_MATCHED_NON_TOP1", top1)
        + effect_threshold_summary(
            "TUMOR_SUPPORTED_vs_UNSUPPORTED_TOP1", tumor
        )
        + effect_threshold_summary(
            "K6_SPECIFIC_vs_BPE_SPECIFIC", branch
        )
    )

    within_family, within_modules, attribution_status = (
        within_family_sequence_analysis(within_source, sequence)
    )


    atomic_write_tsv(comparison, OUT_COMPARISON_SUMMARY)
    atomic_write_tsv(top1_significant, OUT_TOP1_SIGNIFICANT)
    atomic_write_tsv(effect_thresholds, OUT_EFFECT_THRESHOLDS)
    atomic_write_tsv(within_family, OUT_WITHIN_FAMILY)
    atomic_write_tsv(within_modules, OUT_WITHIN_FAMILY_MODULES)


# --------------------------------------------------------------------------
# step 15: Analyze the K6/BPE sequence grammar
# --------------------------------------------------------------------------



BASE = Path(__file__).resolve().parent.parent
M01 = BASE / "Modeling/output/01_build_dual_tensors"
M25 = BASE / "Biological_Interpretation/output/25_evolutionary_interpretation"
M26 = BASE / "Biological_Interpretation/output/26_host_regulatory_mechanism_interpretation"
M28 = BASE / "Biological_Interpretation/output/28_sequence_grammar_and_representative_loci"

TOKENIZER = M01 / "01_herv_bpe_tokenizer.json"
TENSOR_PATHS = [M01 / "01_train_dual_tensors.pt",
                M01 / "01_valid_dual_tensors.pt",
                M01 / "01_test_dual_inputs.pt"]
M25_PAIRS = M25 / "25_final_top1_matched_background.tsv"
M26_PAIRS__15 = M26 / "26_matched_top1_pairs.tsv"
M28_MEMBERSHIP = M28 / "28_k6_bpe_validation_top1_membership.tsv"

OUT__15 = BASE / "Biological_Interpretation/output/15_k6_bpe_sequence_grammar"
OUT_TOP1 = OUT__15 / "15_bpe_top1_vs_matched_token.tsv"
OUT_TUMOR = OUT__15 / "15_bpe_tumor_supported_vs_unsupported_token.tsv"
OUT_K6BPE = OUT__15 / "15_bpe_k6_specific_vs_bpe_specific_token.tsv"
OUT_K6 = OUT__15 / "15_k6_existing_result_summary.tsv"


ID__15 = "herv_internal_id"
MIN_TOKEN_GLOBAL_PRESENCE_N = 20










def bh_fdr__15(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def special_token_ids() -> set[int]:
    return {int(t["id"]) for t in json.loads(TOKENIZER.read_text(encoding="utf-8"))["added_tokens"]
            if t.get("special")}


def decode_tokens() -> dict[int, str]:
    vocab = json.loads(TOKENIZER.read_text(encoding="utf-8"))["model"]["vocab"]
    return {int(v): k for k, v in vocab.items()}


def load_bpe_presence(special: set[int]) -> dict[str, np.ndarray]:
    presence: dict[str, np.ndarray] = {}
    for path in TENSOR_PATHS:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        ids = obj["input_ids_bpe"].numpy()
        mask = obj["mask_bpe"].numpy().astype(bool)
        tokens = np.where(mask, ids, -1)
        tokens[np.isin(tokens, list(special))] = -1
        for position, herv_id in enumerate(obj["info_df"][ID__15].astype(str)):
            row = tokens[position]
            presence[herv_id] = np.unique(row[row >= 0])
    return presence


def resolve(presence: dict[str, np.ndarray], left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    a, b = left.astype(str).to_numpy(), right.astype(str).to_numpy()
    keep = np.array([x in presence and y in presence for x, y in zip(a, b)])
    counts = {"pairs_input": int(len(a)), "pairs_resolved": int(keep.sum()), "pairs_dropped": int((~keep).sum())}
    return a[keep], b[keep], counts


def paired_token_test(case: np.ndarray, control: np.ndarray, presence: dict[str, np.ndarray],
                      special: set[int]) -> pd.DataFrame:
    case_set = [set(presence[x].tolist()) for x in case]
    control_set = [set(presence[x].tolist()) for x in control]
    tokens = set().union(*case_set, *control_set)
    rows = []
    for token in sorted(tokens - special):
        a = sum(token in s for s in case_set)
        b = sum(token in s for s in control_set)
        if a + b < MIN_TOKEN_GLOBAL_PRESENCE_N:
            continue
        case_only = sum(t in x and t not in y for t, x, y in zip([token] * len(case_set), case_set, control_set))
        control_only = sum(t not in x and t in y for t, x, y in zip([token] * len(case_set), case_set, control_set))
        n_disc = case_only + control_only
        rows.append({
            "token_id": token, "group_a_n": len(case), "group_b_n": len(control),
            "group_a_presence_n": int(a), "group_b_presence_n": int(b),
            "group_a_presence_rate": a / len(case), "group_b_presence_rate": b / len(control),
            "rate_delta_a_minus_b": a / len(case) - b / len(control),
            "discordant_case_only": case_only, "discordant_control_only": control_only,
            "odds_ratio_haldane": (case_only + 0.5) / (control_only + 0.5),
            "p_value": binomtest(case_only, n=n_disc, p=0.5).pvalue if n_disc else 1.0,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = bh_fdr__15(out["p_value"].tolist())
        out = out.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    return out


def unpaired_token_test(group_a: np.ndarray, group_b: np.ndarray, presence: dict[str, np.ndarray],
                       special: set[int]) -> pd.DataFrame:
    set_a = [set(presence[x].tolist()) for x in group_a]
    set_b = [set(presence[x].tolist()) for x in group_b]
    tokens = (set().union(*set_a) if set_a else set()) | (set().union(*set_b) if set_b else set())
    rows = []
    for token in sorted(tokens - special):
        a = sum(token in s for s in set_a)
        b = sum(token in s for s in set_b)
        if a + b < MIN_TOKEN_GLOBAL_PRESENCE_N:
            continue
        _, p = fisher_exact([[a, len(set_a) - a], [b, len(set_b) - b]], alternative="two-sided")
        rows.append({
            "token_id": token, "group_a_n": len(set_a), "group_b_n": len(set_b),
            "group_a_presence_n": int(a), "group_b_presence_n": int(b),
            "group_a_presence_rate": a / len(set_a), "group_b_presence_rate": b / len(set_b),
            "rate_delta_a_minus_b": a / len(set_a) - b / len(set_b),
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh"] = bh_fdr__15(out["p_value"].tolist())
    return out


def k6_membership_summary(membership: pd.DataFrame) -> pd.DataFrame:
    score_columns = [c for c in membership.columns if c.startswith("score_")]
    summary = membership.groupby("k6_bpe_class")[score_columns].mean().reset_index()
    counts = membership["k6_bpe_class"].value_counts().rename("n").reset_index()
    counts.columns = ["k6_bpe_class", "n"]
    return counts.merge(summary, on="k6_bpe_class")


def step_15() -> None:
    OUT__15.mkdir(parents=True, exist_ok=True)
    special = special_token_ids()
    presence = load_bpe_presence(special)
    tokens = decode_tokens()

    m25 = pd.read_csv(M25_PAIRS, sep="\t")
    m26 = pd.read_csv(M26_PAIRS__15, sep="\t")
    membership = pd.read_csv(M28_MEMBERSHIP, sep="\t")

    top1_a, top1_b, top1_counts = resolve(presence, m25["top1_herv_internal_id"],
                                          m25["matched_background_herv_internal_id"])
    tum_a, tum_b, tum_counts = resolve(presence, m26["supported_herv_internal_id"],
                                       m26["unsupported_herv_internal_id"])
    k6_specific = membership.loc[membership["k6_bpe_class"] == "K6_SPECIFIC", ID__15].astype(str).to_numpy()
    bpe_specific = membership.loc[membership["k6_bpe_class"] == "BPE_SPECIFIC", ID__15].astype(str).to_numpy()

    top1 = paired_token_test(top1_a, top1_b, presence, special)
    tumor = paired_token_test(tum_a, tum_b, presence, special)
    k6bpe = unpaired_token_test(k6_specific, bpe_specific, presence, special)
    k6 = k6_membership_summary(membership)

    for frame, path in ((top1, OUT_TOP1), (tumor, OUT_TUMOR), (k6bpe, OUT_K6BPE), (k6, OUT_K6)):
        write_tsv(frame.assign(decoded_token=frame["token_id"].map(tokens)) if "token_id" in frame else frame, path)


def main() -> None:
    step_28()
    step_03()
    step_15()


if __name__ == "__main__":
    main()
