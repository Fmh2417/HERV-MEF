#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import re
import numpy as np
import pandas as pd
import gzip
import subprocess
from typing import Any
from scipy.stats import (
    wilcoxon,
    fisher_exact,
    mannwhitneyu,
)
from biological_common import (
    tensors_for_rows,
    score_rows,
    write_tsv,
    write_json,
    build_frozen_scorer,
    now,
)
from scipy.cluster.hierarchy import (
    fcluster,
    linkage,
)
from scipy.spatial.distance import squareform

# --------------------------------------------------------------------------
# step 14c: Validate the MER11 flanks against the Chen 2025 experiments
# --------------------------------------------------------------------------



HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL_OUTPUT = PROJECT_ROOT / "Modeling/output"
M00_DIR = MODEL_OUTPUT / "00_build_canonical_data"
M23_DIR = MODEL_OUTPUT / "23_generate_genomewide_herv_functional_ranking"

CANONICAL = M00_DIR / "00_canonical_herv_mef_table.parquet"
RANKING = M23_DIR / "23_herv_genomewide_functional_ranking.parquet"

OUT_DIR = HERE / "output/14c_validate_chen2025_mer11_flank_homology"
OUT_MUTATIONS = OUT_DIR / "14c_mer11_chen2025_ism_per_locus.tsv"
OUT_SUMMARY = OUT_DIR / "14c_summary.json"

ID = "herv_internal_id"
SCORE_BATCH_SIZE = 32

RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def rev_comp(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]



def score_frame(frame: pd.DataFrame, sequences: list[str], scorer: dict[str, Any]) -> np.ndarray:
    tensors, _ = tensors_for_rows(
        frame, scorer["tokenizer"], scorer["config"],
        scorer["encoder"], scorer["scaler"], sequences=sequences
    )
    return score_rows(
        scorer["model"], tensors, scorer["device"],
        scorer["score_head_indices"], batch_size=SCORE_BATCH_SIZE
    )

def patch_insert(window: str, coord: int, base: str) -> str:
    return window[:coord] + base + window[coord: len(window) - 1]

def patch_delete(window: str, coord: int) -> str:
    pad_base = window[-1] if len(window) > 0 else "N"
    return window[:coord] + window[coord + 1:] + pad_base

# =============================================================================
# =============================================================================

P144_FWD_ANC = re.compile(r"ATGTTTATCAATGACAATGC", re.IGNORECASE)
P144_FWD_DEL = re.compile(r"ATGTTATCAATGACAATGC", re.IGNORECASE)

def find_locus_144(seq: str) -> tuple[str, int, int]:
    m = P144_FWD_ANC.search(seq)
    if m:
        return "ANCESTRAL", m.start(), m.start() + 5
    m_del = P144_FWD_DEL.search(seq)
    if m_del:
        return "DELETED", m_del.start(), m_del.start() + 5
    return "UNRESOLVED", -1, -1


def step_14c() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_parquet(CANONICAL)
    ranking = pd.read_parquet(RANKING, columns=[ID, "score_sequence_regulatory", "herv_functional_priority_rank"])

    mer11 = canonical[canonical["repName"].isin(["MER11A", "MER11B", "MER11C", "MER11D"])].copy()
    mer11 = mer11.merge(ranking, on=ID, how="inner").reset_index(drop=True)

    scorer = build_frozen_scorer()

    records_144 = []
    for idx, row in mer11.iterrows():
        seq = str(row["sequence"])
        strand = str(row["strand"])
        status, m_start, mut_coord = find_locus_144(seq)
        if status != "UNRESOLVED":
            records_144.append({
                "mer11_idx": idx,
                ID: str(row[ID]),
                "repName": str(row["repName"]),
                "strand": strand,
                "status_144": status,
                "motif_start": m_start,
                "mutate_coord": mut_coord,
                "original_sequence": seq,
                "baseline_score": float(row["score_sequence_regulatory"]),
                "functional_rank": int(row["herv_functional_priority_rank"])
            })

    df_144 = pd.DataFrame(records_144)
    required_meta = list(dict.fromkeys(list(scorer["config"]["cat_features"])
                                       + list(scorer["scaler"].feature_names_in_)))
    metadata = mer11[required_meta].copy()
    metadata.insert(0, "mer11_idx", metadata.index)
    df_144 = df_144.drop(columns=["repName"]).merge(metadata, on="mer11_idx", how="left")

    mutation_results = []

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    deleted_copies = df_144[df_144["status_144"] == "DELETED"].copy().reset_index(drop=True)
    if len(deleted_copies) > 0:
        patched_seqs = []
        for _, r in deleted_copies.iterrows():
            seq = r["original_sequence"]
            coord = r["mutate_coord"]
            base_to_insert = "T"
            patched_seqs.append(patch_insert(seq, coord, base_to_insert))

        orig_scores = score_frame(deleted_copies, deleted_copies["original_sequence"].tolist(), scorer)
        patched_scores = score_frame(deleted_copies, patched_seqs, scorer)

        for i, r in deleted_copies.iterrows():
            mutation_results.append({
                "experiment": "REVERT_144_T_INSERTION",
                "hypothesis": "SCORE_DROP_EXPECTED",
                ID: r[ID], "repName": r["repName"], "strand": r["strand"],
                "original_score": float(orig_scores[i]),
                "patched_score": float(patched_scores[i]),
                "score_delta": float(patched_scores[i] - orig_scores[i]),
                "is_score_dropped": bool(patched_scores[i] < orig_scores[i]),
            })

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    ancestral_copies = df_144[df_144["status_144"] == "ANCESTRAL"].copy().reset_index(drop=True)
    if len(ancestral_copies) > 0:
        patched_seqs_ko = []
        for _, r in ancestral_copies.iterrows():
            seq = r["original_sequence"]
            coord = r["mutate_coord"]
            patched_seqs_ko.append(patch_delete(seq, coord))

        orig_scores_ko = score_frame(ancestral_copies, ancestral_copies["original_sequence"].tolist(), scorer)
        patched_scores_ko = score_frame(ancestral_copies, patched_seqs_ko, scorer)

        for i, r in ancestral_copies.iterrows():
            mutation_results.append({
                "experiment": "KNOCKOUT_144_T_DELETION",
                "hypothesis": "SCORE_GAIN_EXPECTED",
                ID: r[ID], "repName": r["repName"], "strand": r["strand"],
                "original_score": float(orig_scores_ko[i]),
                "patched_score": float(patched_scores_ko[i]),
                "score_delta": float(patched_scores_ko[i] - orig_scores_ko[i]),
                "is_score_gained": bool(patched_scores_ko[i] > orig_scores_ko[i]),
            })

    df_mut = pd.DataFrame(mutation_results)
    write_tsv(df_mut, OUT_MUTATIONS)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    revert_sub = df_mut[df_mut["experiment"] == "REVERT_144_T_INSERTION"]
    ko_sub = df_mut[df_mut["experiment"] == "KNOCKOUT_144_T_DELETION"]

    p_revert = wilcoxon(revert_sub["score_delta"], alternative="two-sided").pvalue if len(revert_sub) >= 5 else np.nan
    p_ko = wilcoxon(ko_sub["score_delta"], alternative="two-sided").pvalue if len(ko_sub) >= 5 else np.nan

    summary_stats = {
        "generated_at": now(),
        "total_mer11_surveyed": len(mer11),
        "locus_144_resolved": len(df_144),
        "subfamily_resolved_breakdown": df_144["repName"].value_counts().to_dict(),
        "revert_experiment": {
            "n_copies": len(revert_sub),
            "mean_delta": float(revert_sub["score_delta"].mean()) if len(revert_sub) else 0.0,
            "fraction_dropped": float(revert_sub["is_score_dropped"].mean()) if len(revert_sub) else 0.0,
            "wilcoxon_p_value": float(p_revert)
        },
        "knockout_experiment": {
            "n_copies": len(ko_sub),
            "mean_delta": float(ko_sub["score_delta"].mean()) if len(ko_sub) else 0.0,
            "fraction_gained": float(ko_sub["is_score_gained"].mean()) if len(ko_sub) else 0.0,
            "wilcoxon_p_value": float(p_ko)
        }
    }
    write_json(summary_stats, OUT_SUMMARY)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    report_lines = [
        "# Chen 2025 (Sci Adv) MER11 Locus 144 In Silico Homology & Mutagenesis Report",
        "",
        f"- Generated: {summary_stats['generated_at']}",
        f"- Total MER11 Elements Surveyed: {summary_stats['total_mer11_surveyed']}",
        f"- Locus 144 Resolved in Genome: {summary_stats['locus_144_resolved']} copies",
        "",
        "## 1. Subfamily Resolution around the Homologous SOX Core",
        "The canonical POU/SOX motif `ATCAATGACAATGC` was matched across both `+` and `-` strands.",
        "",
        pd.DataFrame([summary_stats['subfamily_resolved_breakdown']]).to_markdown(index=False),
        "",
        "## 2. In Silico Functional Mutagenesis Results",
        "",
        "| Experiment | Mutation Description | Target N | Hypothesis | Mean Delta | Directional Consistency Rate | Wilcoxon P-Value |",
        "|---|---|---|---|---|---|---|",
        f"| Revert (Ancestralization) | Insert missing T into deleted copies | {summary_stats['revert_experiment']['n_copies']} | Score Drop | {summary_stats['revert_experiment']['mean_delta']:.6f} | {summary_stats['revert_experiment']['fraction_dropped']*100:.2f}% Dropped | {summary_stats['revert_experiment']['wilcoxon_p_value']:.4e} |",
        f"| Knockout (Derived Divergence) | Delete T from ancestral intact copies | {summary_stats['knockout_experiment']['n_copies']} | Score Gain | {summary_stats['knockout_experiment']['mean_delta']:.6f} | {summary_stats['knockout_experiment']['fraction_gained']*100:.2f}% Gained | {summary_stats['knockout_experiment']['wilcoxon_p_value']:.4e} |",
        "",
        "## 3. Definitive Biological Conclusion",
    ]

    revert_p = summary_stats["revert_experiment"]["wilcoxon_p_value"]
    revert_delta = summary_stats["revert_experiment"]["mean_delta"]
    ko_p = summary_stats["knockout_experiment"]["wilcoxon_p_value"]
    ko_delta = summary_stats["knockout_experiment"]["mean_delta"]

    significant = (revert_p < 0.05) or (ko_p < 0.05)
    matches_chen = (revert_delta < 0) or (ko_delta > 0)

    if significant:
        report_lines.append("**STATUS: SIGNIFICANT SINGLE-BASE EFFECT**")
        report_lines.append(
            f"Locus 144 T deletion significantly changes the frozen score "
            f"(revert two-sided P={revert_p:.3g}, knockout two-sided P={ko_p:.3g}); "
            f"the observed direction is "
            f"{'consistent with' if matches_chen else 'opposite to'} "
            f"Chen 2025's stem-cell MPRA (deletion = gain)."
        )
    else:
        report_lines.append("**STATUS: NO SIGNIFICANT SINGLE-BASE EFFECT**")
        report_lines.append(
            "Neither revert nor knockout produced a significant score change for the "
            "Locus 144 T deletion."
        )

    report_lines.append("")


# --------------------------------------------------------------------------
# step 14d: Compare ESC and adult chromatin direction at the Chen loci
# --------------------------------------------------------------------------



HERE__14d = Path(__file__).resolve().parent
MODELING = HERE__14d.parent / "Modeling"
DP = HERE__14d.parent / "Data_Preprocessing"

M14B = HERE__14d / 'output/14b_analyze_key_subfamily_single_base_determinism/14b_single_base_revert_scores.tsv.gz'
M14C = HERE__14d / 'output/14c_validate_chen2025_mer11_flank_homology/14c_mer11_chen2025_ism_per_locus.tsv'
M00 = MODELING / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
MANIFEST = DP / "output/04_roadmap_manifests/non_tumor_roadmap_manifest.csv"
MAPPED = DP / "output/05_separated_roadmap_features/mapped"

OUT_DIR__14d = HERE__14d / "output/14d_esc_vs_adult_chen_locus_epigenomic_direction"

BEDTOOLS = config.BEDTOOLS

ID__14d = "herv_internal_id"
ACTIVE_MARKS = ["H3K27ac", "H3K4me1", "H3K4me3"]
ESC_GROUPS = ["ESC", "ES-deriv", "iPSC"]




def load_copy_groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    b14 = pd.read_csv(M14B, sep="\t")
    sub158 = b14.loc[b14["reference_position"].eq(157)].copy()
    group158 = []
    for arm, label in [
        ("restored_base", "C_del"),
        ("matched_control_insertion", "C_intact"),
    ]:
        ids = sub158.loc[sub158["arm"].eq(arm), ID__14d].drop_duplicates()
        group158.append(pd.DataFrame({ID__14d: ids, "locus158_group": label}))
    locus158 = pd.concat(group158, ignore_index=True)

    c14 = pd.read_csv(M14C, sep="\t")
    group144 = []
    for exp, label in [
        ("REVERT_144_T_INSERTION", "T_del"),
        ("KNOCKOUT_144_T_DELETION", "T_intact"),
    ]:
        ids = c14.loc[c14["experiment"].eq(exp), ID__14d].drop_duplicates()
        group144.append(pd.DataFrame({ID__14d: ids, "locus144_group": label}))
    locus144 = pd.concat(group144, ignore_index=True)
    return locus158, locus144


def load_canonical(ids: set[str]) -> pd.DataFrame:
    cols = [
        ID__14d,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repFamily",
        "length",
        "gc_content",
        "repeat_divergence",
    ]
    canon = pd.read_parquet(M00, columns=cols)
    canon = canon.loc[canon[ID__14d].isin(ids)].copy()
    canon[ID__14d] = canon[ID__14d].astype(str)
    return canon


def load_eid_groups() -> dict[str, set[str]]:
    manifest = pd.read_csv(MANIFEST)
    manifest["eid"] = manifest["eid"].astype(str)
    esc = set(manifest.loc[manifest["sample_group"].isin(ESC_GROUPS), "eid"])
    adult = set(manifest.loc[~manifest["sample_group"].isin(ESC_GROUPS), "eid"])
    return {"esc": esc, "adult": adult}


def write_copy_bed(copies: pd.DataFrame, path: Path) -> None:
    copies = copies.sort_values([ID__14d]).reset_index(drop=True)
    bed = pd.DataFrame(
        {
            "chrom": copies["chrom"].astype(str),
            "start": copies["start"].astype(int),
            "end": copies["end"].astype(int),
            "id": copies[ID__14d].astype(str),
        }
    )
    bed.to_csv(path, sep="\t", header=False, index=False)


def available_eids_for_mark(mark: str) -> set[str]:
    out = set()
    for bed in MAPPED.glob(f"*_{mark}_non_tumor.hg38.bed.gz"):
        eid = bed.name.split("_")[0]
        out.add(eid)
    return out


def merge_mark_bed(mark: str, eids: set[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        for eid in sorted(eids):
            bed = MAPPED / f"{eid}_{mark}_non_tumor.hg38.bed.gz"
            if not bed.exists():
                continue
            with gzip.open(bed, "rt", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 3:
                        continue
                    out.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{eid}\n")


def intersect_mark(
    copies_bed: Path,
    mark: str,
    eids: set[str],
) -> dict[str, set[str]]:
    merged = OUT_DIR__14d / f"_tmp_{mark}.bed"
    merge_mark_bed(mark, eids, merged)
    cmd = [
        BEDTOOLS,
        "intersect",
        "-a",
        str(copies_bed),
        "-b",
        str(merged),
        "-wo",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    result: dict[str, set[str]] = {}
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        copy_id = fields[3]
        eid = fields[7]
        result.setdefault(copy_id, set()).add(eid)
    merged.unlink(missing_ok=True)
    return result


def build_support_table(
    copies: pd.DataFrame,
    eid_groups: dict[str, set[str]],
) -> pd.DataFrame:
    all_eids = eid_groups["esc"] | eid_groups["adult"]
    copies_bed = OUT_DIR__14d / "_tmp_copies.bed"
    write_copy_bed(copies, copies_bed)

    mark_hits: dict[str, dict[str, set[str]]] = {}
    for mark in ACTIVE_MARKS:
        avail = available_eids_for_mark(mark)
        use = all_eids & avail
        mark_hits[mark] = intersect_mark(copies_bed, mark, use)

    copies_bed.unlink(missing_ok=True)

    eid_marks: dict[str, set[str]] = {}
    for eid in all_eids:
        eid_marks[eid] = {
            mark for mark in ACTIVE_MARKS
            if (MAPPED / f"{eid}_{mark}_non_tumor.hg38.bed.gz").exists()
        }

    rows = []
    for copy_id in copies[ID__14d].astype(str):
        supported = set()
        for mark in ACTIVE_MARKS:
            supported |= mark_hits[mark].get(copy_id, set())
        esc_support = supported & eid_groups["esc"]
        adult_support = supported & eid_groups["adult"]
        esc_den = {e for e in eid_groups["esc"] if eid_marks.get(e)}
        adult_den = {e for e in eid_groups["adult"] if eid_marks.get(e)}
        rows.append(
            {
                ID__14d: copy_id,
                "esc_support_eids": sorted(esc_support),
                "adult_support_eids": sorted(adult_support),
                "esc_support_fraction": (
                    len(esc_support) / len(esc_den) if esc_den else float("nan")
                ),
                "adult_support_fraction": (
                    len(adult_support) / len(adult_den) if adult_den else float("nan")
                ),
            }
        )
    support = pd.DataFrame(rows)
    return support


def compare_groups(
    copies: pd.DataFrame,
    support: pd.DataFrame,
    locus: str,
    group_col: str,
    del_label: str,
    intact_label: str,
) -> pd.DataFrame:
    merged = copies.merge(support, on=ID__14d, how="left", validate="one_to_one")
    rows = []
    for outcome in ["esc_support_fraction", "adult_support_fraction"]:
        x = merged.loc[merged[group_col].eq(del_label), outcome].dropna()
        y = merged.loc[merged[group_col].eq(intact_label), outcome].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        test = mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "locus": locus,
                "group_col": group_col,
                "outcome": outcome,
                "del_n": len(x),
                "intact_n": len(y),
                "del_median": float(np.median(x)),
                "intact_median": float(np.median(y)),
                "median_delta_del_minus_intact": float(np.median(x) - np.median(y)),
                "mannwhitney_p": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def compare_groups_by_repname(
    copies: pd.DataFrame,
    support: pd.DataFrame,
    group_col: str,
    del_label: str,
    intact_label: str,
) -> pd.DataFrame:
    merged = copies.merge(support, on=ID__14d, how="left", validate="one_to_one")
    rows = []
    for repname, sub in merged.groupby("repName", sort=True):
        if group_col not in sub.columns:
            continue
        del_n = int(sub[group_col].eq(del_label).sum())
        intact_n = int(sub[group_col].eq(intact_label).sum())
        if del_n < 3 or intact_n < 3:
            continue
        for outcome in ["esc_support_fraction", "adult_support_fraction"]:
            x = sub.loc[sub[group_col].eq(del_label), outcome].dropna()
            y = sub.loc[sub[group_col].eq(intact_label), outcome].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            test = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "repName": repname,
                    "outcome": outcome,
                    "del_n": len(x),
                    "intact_n": len(y),
                    "del_median": float(np.median(x)),
                    "intact_median": float(np.median(y)),
                    "median_delta_del_minus_intact": float(np.median(x) - np.median(y)),
                    "mannwhitney_p": float(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def fisher_stratified(
    copies: pd.DataFrame,
    support: pd.DataFrame,
    group_col: str,
    del_label: str,
    intact_label: str,
    locus: str,
) -> pd.DataFrame:
    merged = copies.merge(support, on=ID__14d, how="left", validate="one_to_one")
    merged["esc_active"] = merged["esc_support_fraction"].gt(0).astype(int)
    merged["adult_active"] = merged["adult_support_fraction"].gt(0).astype(int)
    rows = []
    for repname, sub in merged.groupby("repName", sort=True):
        del_n = int(sub[group_col].eq(del_label).sum())
        intact_n = int(sub[group_col].eq(intact_label).sum())
        if del_n < 3 or intact_n < 3:
            continue
        for outcome in ["esc_active", "adult_active"]:
            a = int((sub[group_col].eq(del_label) & sub[outcome].eq(1)).sum())
            b = int((sub[group_col].eq(del_label) & sub[outcome].eq(0)).sum())
            c = int((sub[group_col].eq(intact_label) & sub[outcome].eq(1)).sum())
            d = int((sub[group_col].eq(intact_label) & sub[outcome].eq(0)).sum())
            odds_ratio, p = fisher_exact([[a, b], [c, d]])
            rows.append(
                {
                    "locus": locus,
                    "repName": repname,
                    "outcome": outcome,
                    "del_active": a,
                    "del_inactive": b,
                    "intact_active": c,
                    "intact_inactive": d,
                    "del_rate": a / (a + b) if (a + b) else float("nan"),
                    "intact_rate": c / (c + d) if (c + d) else float("nan"),
                    "odds_ratio": float(odds_ratio),
                    "fisher_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def logistic_adjusted(
    copies: pd.DataFrame,
    support: pd.DataFrame,
    group_col: str,
    del_label: str,
    intact_label: str,
    locus: str,
) -> pd.DataFrame:
    import statsmodels.api as sm

    merged = copies.merge(support, on=ID__14d, how="left", validate="one_to_one")
    merged["esc_active"] = merged["esc_support_fraction"].gt(0).astype(int)
    merged["adult_active"] = merged["adult_support_fraction"].gt(0).astype(int)
    merged["is_del"] = merged[group_col].eq(del_label).astype(int)

    del_reps = set(merged.loc[merged["is_del"].eq(1), "repName"])
    intact_reps = set(merged.loc[merged["is_del"].eq(0), "repName"])
    sub = merged.loc[merged["repName"].isin(del_reps & intact_reps)].copy()

    rows = []
    for outcome in ["esc_active", "adult_active"]:
        d = sub[
            [outcome, "is_del", "repName", "length", "gc_content", "repeat_divergence"]
        ].dropna()
        d = d.copy()
        X = pd.get_dummies(
            d[["is_del", "repName"]], columns=["repName"], drop_first=True, dtype=float
        )
        for col in ["length", "gc_content", "repeat_divergence"]:
            X[col] = d[col].astype(float)
        X = sm.add_constant(X)
        y = d[outcome].astype(int)
        try:
            model = sm.Logit(y, X).fit(disp=False)
            coef = float(model.params["is_del"])
            ci = model.conf_int().loc["is_del"]
            p = float(model.pvalues["is_del"])
            rows.append(
                {
                    "locus": locus,
                    "outcome": outcome,
                    "n": int(len(d)),
                    "coef_del": coef,
                    "OR_del": float(np.exp(coef)),
                    "ci_low": float(np.exp(ci[0])),
                    "ci_high": float(np.exp(ci[1])),
                    "p_value": p,
                    "converged": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "locus": locus,
                    "outcome": outcome,
                    "n": int(len(d)),
                    "coef_del": float("nan"),
                    "OR_del": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "converged": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def step_14d() -> None:
    OUT_DIR__14d.mkdir(parents=True, exist_ok=True)


    locus158, locus144 = load_copy_groups()
    all_ids = set(locus158[ID__14d]) | set(locus144[ID__14d])
    canon = load_canonical(all_ids)
    eid_groups = load_eid_groups()


    support = build_support_table(canon, eid_groups)

    out = canon.merge(support, on=ID__14d, how="left", validate="one_to_one")
    out = out.merge(locus158, on=ID__14d, how="left", validate="one_to_one")
    out = out.merge(locus144, on=ID__14d, how="left", validate="one_to_one")
    out["esc_support_eids"] = out["esc_support_eids"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )
    out["adult_support_eids"] = out["adult_support_eids"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )


# --------------------------------------------------------------------------
# step 14e: Audit the homology linkage of the Chen loci
# --------------------------------------------------------------------------



HERE__14e = Path(__file__).resolve().parent
MODELING__14e = HERE__14e.parent / "Modeling"

M00__14e = MODELING__14e / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M14C__14e = HERE__14e / 'output/14c_validate_chen2025_mer11_flank_homology/14c_mer11_chen2025_ism_per_locus.tsv'
M14B__14e = HERE__14e / 'output/14b_analyze_key_subfamily_single_base_determinism/14b_single_base_revert_scores.tsv.gz'

OUT_DIR__14e = HERE__14e / "output/14e_chen_locus_homology_linkage_audit"
OUT_L144_DOWN = OUT_DIR__14e / "14e_locus144_downstream_patterns.tsv"
OUT_L158_DOWN = OUT_DIR__14e / "14e_locus158_downstream_patterns.tsv"

ID__14e = "herv_internal_id"

P144_ANC = re.compile(r"ATGTTTATCAATGACAATGC", re.IGNORECASE)
P144_DEL = re.compile(r"ATGTTATCAATGACAATGC", re.IGNORECASE)


def load_canonical__14e() -> pd.DataFrame:
    return pd.read_parquet(
        M00__14e, columns=[ID__14e, "sequence", "strand", "repName"]
    )


def audit_locus144(canon: pd.DataFrame) -> pd.DataFrame:
    c14 = pd.read_csv(M14C__14e, sep="\t")
    grp = c14[[ID__14e, "experiment"]].drop_duplicates()
    grp["group"] = grp["experiment"].map(
        {"REVERT_144_T_INSERTION": "T_del", "KNOCKOUT_144_T_DELETION": "T_intact"}
    )
    merged = grp.merge(canon, on=ID__14e, how="inner", validate="one_to_one")

    rows = []
    for _, r in merged.iterrows():
        seq = str(r["sequence"])
        m = P144_ANC.search(seq)
        if not m:
            m = P144_DEL.search(seq)
        if not m:
            continue
        start = m.start()
        motif = m.group(0)
        downstream = seq[start + len(motif): start + len(motif) + 30]
        rows.append(
            {
                ID__14e: r[ID__14e],
                "group": r["group"],
                "repName": r["repName"],
                "strand": r["strand"],
                "motif": motif,
                "downstream30": downstream,
            }
        )
    return pd.DataFrame(rows)


def audit_locus158(canon: pd.DataFrame) -> pd.DataFrame:
    b14 = pd.read_csv(M14B__14e, sep="\t")
    sub = b14.loc[b14["reference_position"].eq(157)].copy()
    sub["group"] = sub["arm"].map(
        {"restored_base": "C_del", "matched_control_insertion": "C_intact"}
    )
    merged = sub[
        [ID__14e, "group", "repName", "strand", "window_coordinate"]
    ].merge(canon, on=ID__14e, how="inner", validate="one_to_one", suffixes=("", "_canon"))

    rows = []
    for _, r in merged.iterrows():
        seq = str(r["sequence"])
        coord = int(r["window_coordinate"])
        downstream = seq[coord: coord + 30]
        rows.append(
            {
                ID__14e: r[ID__14e],
                "group": r["group"],
                "repName": r["repName_canon"],
                "strand": r["strand_canon"],
                "window_coordinate": coord,
                "downstream30": downstream,
            }
        )
    return pd.DataFrame(rows)


def pattern_summary(frame: pd.DataFrame, group_col: str, pattern_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby(group_col, sort=True):
        counts = sub[pattern_col].value_counts()
        for pattern, n in counts.head(12).items():
            rows.append(
                {
                    "group": group,
                    "pattern": pattern,
                    "n": int(n),
                    "share_in_group": float(n / len(sub)),
                }
            )
    return pd.DataFrame(rows)


def step_14e() -> None:
    OUT_DIR__14e.mkdir(parents=True, exist_ok=True)
    canon = load_canonical__14e()

    l144 = audit_locus144(canon)
    l158 = audit_locus158(canon)

    l144_pat = pattern_summary(l144, "group", "downstream30")
    l158_pat = pattern_summary(l158, "group", "downstream30")

    l144_pat.to_csv(OUT_L144_DOWN, sep="\t", index=False)
    l158_pat.to_csv(OUT_L158_DOWN, sep="\t", index=False)

    def separated(frame: pd.DataFrame, group_col: str, pattern_col: str) -> bool:
        groups = sorted(frame[group_col].unique())
        sets = [set(frame.loc[frame[group_col].eq(g), pattern_col]) for g in groups]
        return all(len(sets[i] & sets[j]) == 0 for i in range(len(sets)) for j in range(i + 1, len(sets)))


# --------------------------------------------------------------------------
# step 14f: Classify the chromatin state at the Chen 144/158 loci
# --------------------------------------------------------------------------



HERE__14f = Path(__file__).resolve().parent
MODELING__14f = HERE__14f.parent / "Modeling"

M00__14f = MODELING__14f / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"

OUT_DIR__14f = HERE__14f / "output/14f_chen_locus_144_158_state_classification"
OUT_STATE = OUT_DIR__14f / "14f_locus_state_classification.tsv"

OUT_SUPPORT = OUT_DIR__14f / "14f_state_epigenomic_support.tsv"
OUT_FISHER = OUT_DIR__14f / "14f_repname_stratified_fisher.tsv"


ID__14f = "herv_internal_id"
CORE13 = "ATCAATGACAATG"
MER11 = ["MER11A", "MER11B", "MER11C", "MER11D"]


def classify_locus(seq: str) -> tuple[int, str, str] | None:
    for m in re.finditer(CORE13, seq, re.IGNORECASE):
        j = m.start()
        after = seq[j + 13 : j + 17].upper()
        if after.startswith("CGTG"):
            c158 = "C_intact"
        elif after.startswith("GTG"):
            c158 = "C_del"
        else:
            continue
        up6 = seq[j - 6 : j].upper()
        up5 = seq[j - 5 : j].upper()
        if up6 == "ATGTTT":
            t144 = "T_intact"
        elif up5 == "ATGTT":
            t144 = "T_del"
        else:
            continue
        return j, t144, c158
    return None


def load_classification() -> pd.DataFrame:
    canon = pd.read_parquet(
        M00__14f,
        columns=[
            ID__14f, "sequence", "strand", "repName", "chrom", "start", "end",
            "length", "gc_content", "repeat_divergence",
        ],
    )
    canon = canon.loc[canon["repName"].isin(MER11)].copy()
    canon[ID__14f] = canon[ID__14f].astype(str)
    rows = []
    for _, r in canon.iterrows():
        res = classify_locus(str(r["sequence"]))
        if res is None:
            continue
        j, t144, c158 = res
        state = f"{t144}+{c158}"
        rows.append(
            {
                ID__14f: r[ID__14f],
                "repName": r["repName"],
                "strand": r["strand"],
                "chrom": r["chrom"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "length": int(r["length"]),
                "gc_content": float(r["gc_content"]),
                "repeat_divergence": float(r["repeat_divergence"]),
                "core_start": j,
                "t144": t144,
                "c158": c158,
                "state": state,
            }
        )
    return pd.DataFrame(rows)


def load_support(state_table: pd.DataFrame) -> pd.DataFrame:
    copies = state_table[[ID__14f, "chrom", "start", "end"]].copy()
    eid_groups = load_eid_groups()
    support = build_support_table(copies, eid_groups)
    return support


def _active_flags(state: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    merged = state.merge(support, on=ID__14f, how="left", validate="one_to_one")
    merged["esc_active"] = merged["esc_support_fraction"].gt(0).astype(int)
    merged["adult_active"] = merged["adult_support_fraction"].gt(0).astype(int)
    return merged


def fisher_three_state(merged: pd.DataFrame, effect: str) -> pd.DataFrame:
    if effect == "T144":
        sub = merged.loc[merged["c158"].eq("C_intact")].copy()
        sub["is_del"] = sub["t144"].eq("T_del").astype(int)
        locus = "locus144_T_del"
    else:
        sub = merged.loc[merged["t144"].eq("T_del")].copy()
        sub["is_del"] = sub["c158"].eq("C_del").astype(int)
        locus = "locus158_C_del"

    rows = []
    for repname, grp in sub.groupby("repName", sort=True):
        del_n = int(grp["is_del"].eq(1).sum())
        intact_n = int(grp["is_del"].eq(0).sum())
        if del_n < 3 or intact_n < 3:
            continue
        for outcome in ["esc_active", "adult_active"]:
            a = int((grp["is_del"].eq(1) & grp[outcome].eq(1)).sum())
            b = int((grp["is_del"].eq(1) & grp[outcome].eq(0)).sum())
            c = int((grp["is_del"].eq(0) & grp[outcome].eq(1)).sum())
            d = int((grp["is_del"].eq(0) & grp[outcome].eq(0)).sum())
            odds_ratio, p = fisher_exact([[a, b], [c, d]])
            rows.append(
                {
                    "effect": effect,
                    "locus": locus,
                    "repName": repname,
                    "outcome": outcome,
                    "del_n": del_n,
                    "intact_n": intact_n,
                    "del_active": a,
                    "del_inactive": b,
                    "intact_active": c,
                    "intact_inactive": d,
                    "del_rate": a / (a + b) if (a + b) else float("nan"),
                    "intact_rate": c / (c + d) if (c + d) else float("nan"),
                    "odds_ratio": float(odds_ratio),
                    "fisher_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def logistic_three_state(
    merged: pd.DataFrame, effect: str
) -> pd.DataFrame:
    import statsmodels.api as sm

    if effect == "T144":
        sub = merged.loc[merged["c158"].eq("C_intact")].copy()
        sub["is_del"] = sub["t144"].eq("T_del").astype(int)
        locus = "locus144_T_del"
    else:
        sub = merged.loc[merged["t144"].eq("T_del")].copy()
        sub["is_del"] = sub["c158"].eq("C_del").astype(int)
        locus = "locus158_C_del"

    del_reps = set(sub.loc[sub["is_del"].eq(1), "repName"])
    intact_reps = set(sub.loc[sub["is_del"].eq(0), "repName"])
    sub = sub.loc[sub["repName"].isin(del_reps & intact_reps)].copy()

    rows = []
    for outcome in ["esc_active", "adult_active"]:
        d = sub[
            [outcome, "is_del", "repName", "length", "gc_content", "repeat_divergence"]
        ].dropna()
        d = d.copy()
        X = pd.get_dummies(
            d[["is_del", "repName"]], columns=["repName"], drop_first=True, dtype=float
        )
        for col in ["length", "gc_content", "repeat_divergence"]:
            X[col] = d[col].astype(float)
        X = sm.add_constant(X)
        y = d[outcome].astype(int)
        try:
            model = sm.Logit(y, X).fit(disp=False)
            coef = float(model.params["is_del"])
            ci = model.conf_int().loc["is_del"]
            p = float(model.pvalues["is_del"])
            rows.append(
                {
                    "effect": effect,
                    "locus": locus,
                    "outcome": outcome,
                    "n": int(len(d)),
                    "coef_del": coef,
                    "OR_del": float(np.exp(coef)),
                    "ci_low": float(np.exp(ci[0])),
                    "ci_high": float(np.exp(ci[1])),
                    "p_value": p,
                    "converged": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "effect": effect,
                    "locus": locus,
                    "outcome": outcome,
                    "n": int(len(d)),
                    "coef_del": float("nan"),
                    "OR_del": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "converged": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def step_14f() -> None:
    OUT_DIR__14f.mkdir(parents=True, exist_ok=True)
    state = load_classification()
    state.to_csv(OUT_STATE, sep="\t", index=False)


    support = load_support(state)
    out = state.merge(support, on=ID__14f, how="left", validate="one_to_one")
    out["esc_support_eids"] = out["esc_support_eids"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )
    out["adult_support_eids"] = out["adult_support_eids"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )
    out.to_csv(OUT_SUPPORT, sep="\t", index=False)

    merged = _active_flags(state, support)
    fisher = pd.concat(
        [
            fisher_three_state(merged, "T144"),
            fisher_three_state(merged, "C158"),
        ],
        ignore_index=True,
    )
    fisher.to_csv(OUT_FISHER, sep="\t", index=False)


# --------------------------------------------------------------------------
# step 14g: Compute the ISM at the Chen 144/158 loci
# --------------------------------------------------------------------------



HERE__14g = Path(__file__).resolve().parent
MODELING__14g = HERE__14g.parent / "Modeling"

M00__14g = MODELING__14g / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M14F = HERE__14g / 'output/14f_chen_locus_144_158_state_classification/14f_locus_state_classification.tsv'

OUT_DIR__14g = HERE__14g / "output/14g_chen_locus_144_158_ism"
OUT_MUT = OUT_DIR__14g / "14g_locus_144_158_ism.tsv"


ID__14g = "herv_internal_id"


def step_14g() -> None:
    OUT_DIR__14g.mkdir(parents=True, exist_ok=True)

    state = pd.read_csv(M14F, sep="\t")
    state[ID__14g] = state[ID__14g].astype(str)
    canon = pd.read_parquet(M00__14g)
    canon[ID__14g] = canon[ID__14g].astype(str)
    merged = state.merge(canon, on=ID__14g, how="inner", validate="one_to_one")

    scorer = build_frozen_scorer()

    required_meta = list(dict.fromkeys(
        list(scorer["config"]["cat_features"])
        + list(scorer["scaler"].feature_names_in_)
    ))
    for col in required_meta:
        if col not in merged.columns:
            merged[col] = canon.set_index(ID__14g)[col].reindex(merged[ID__14g]).to_numpy()

    experiments = {
        "T_intact+C_intact": [
            ("delete_T_144", lambda s, j: patch_delete(s, j - 1)),
            ("delete_C_158", lambda s, j: patch_delete(s, j + 13)),
        ],
        "T_del+C_intact": [
            ("insert_T_144", lambda s, j: patch_insert(s, j, "T")),
            ("delete_C_158", lambda s, j: patch_delete(s, j + 13)),
        ],
        "T_del+C_del": [
            ("insert_T_144", lambda s, j: patch_insert(s, j, "T")),
            ("insert_C_158", lambda s, j: patch_insert(s, j + 13, "C")),
        ],
    }

    rows = []
    for state_name, exprs in experiments.items():
        sub = merged.loc[merged["state"].eq(state_name)].reset_index(drop=True)
        if sub.empty:
            continue
        orig_scores = score_frame(
            sub, sub["sequence"].astype(str).tolist(), scorer
        )
        for exp_name, patch_fn in exprs:
            patched = []
            for _, r in sub.iterrows():
                seq = str(r["sequence"])
                j = int(r["core_start"])
                patched.append(patch_fn(seq, j))
            patched_scores = score_frame(sub, patched, scorer)
            for i, r in sub.iterrows():
                rows.append(
                    {
                        ID__14g: r[ID__14g],
                        "repName": r["repName"],
                        "state": state_name,
                        "experiment": exp_name,
                        "original_score": float(orig_scores[i]),
                        "patched_score": float(patched_scores[i]),
                        "score_delta": float(patched_scores[i] - orig_scores[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_MUT, sep="\t", index=False)

    from scipy.stats import wilcoxon

    summary = {}
    report_lines = [
        "# 14g Chen Locus 144/158 in silico mutagenesis",
        "",
        "| experiment | state | n | mean_delta | fraction_positive | two-sided P |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for exp_name, sub in df.groupby("experiment"):
        delta = sub["score_delta"]
        p = wilcoxon(delta, alternative="two-sided").pvalue if len(delta) >= 5 else np.nan
        summary[exp_name] = {
            "n": int(len(sub)),
            "mean_delta": float(delta.mean()),
            "fraction_positive": float((delta > 0).mean()),
            "two_sided_p": float(p),
        }
        report_lines.append(
            f"| {exp_name} | {sub['state'].iloc[0]} | {len(sub)} | "
            f"{delta.mean():.6f} | {(delta>0).mean():.4f} | {p:.3g} |"
        )


# --------------------------------------------------------------------------
# step 14h: Summarise the MER11 three-state activity
# --------------------------------------------------------------------------



HERE__14h = Path(__file__).resolve().parent
MODELING__14h = HERE__14h.parent / "Modeling"

M00__14h = MODELING__14h / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
SUPPORT_TSV = HERE__14h / 'output/14f_chen_locus_144_158_state_classification/14f_state_epigenomic_support.tsv'

OUT_DIR__14h = HERE__14h / "output/14h_mer11_three_state_mh_or"
OUT_RATE = OUT_DIR__14h / "14h_three_state_activity_rate.tsv"


RNG_SEED = 20260929
N_BOOT = 5000
N_CLUSTERS = 25
KMER = 6


def kmer_presence(seq: str, k: int = KMER) -> set[str]:
    s = str(seq).upper()
    return {s[i : i + k] for i in range(len(s) - k + 1) if "N" not in s[i : i + k]}


def build_similarity_clusters(ids: list[str], seqs: list[str], n_clusters: int) -> pd.DataFrame:
    vocab: set[str] = set()
    for seq in seqs:
        vocab |= kmer_presence(seq)
    vocab = sorted(vocab)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}

    mat = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for kmer in kmer_presence(seq):
            mat[i, vocab_index[kmer]] = 1.0

    norms = np.linalg.norm(mat, axis=1)
    norm_mat = mat / np.maximum(norms[:, None], 1e-12)
    sim = norm_mat @ norm_mat.T
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    dist = np.maximum(dist, 0.0)

    cond = squareform(dist, checks=False)
    z = linkage(cond, method="average")
    labels = fcluster(z, t=n_clusters, criterion="maxclust")
    return pd.DataFrame({"herv_internal_id": ids, "similarity_cluster": labels})


def mh_or(tables: dict[str, tuple[int, int, int, int]]) -> float:
    num = 0.0
    den = 0.0
    for a, b, c, d in tables.values():
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
    if den == 0.0:
        return float("inf") if num > 0 else float("nan")
    return num / den


def stratify_tables(sub: pd.DataFrame, is_del: str, outcome: str) -> dict[str, tuple[int, int, int, int]]:
    tables: dict[str, tuple[int, int, int, int]] = {}
    for rep, g in sub.groupby("repName", sort=True):
        a = int(((g[is_del] == 1) & (g[outcome] == 1)).sum())
        b = int(((g[is_del] == 1) & (g[outcome] == 0)).sum())
        c = int(((g[is_del] == 0) & (g[outcome] == 1)).sum())
        d = int(((g[is_del] == 0) & (g[outcome] == 0)).sum())
        tables[rep] = (a, b, c, d)
    return tables


def cluster_bootstrap(
    sub: pd.DataFrame,
    is_del: str,
    outcome: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    clusters = [g for _, g in sub.groupby("similarity_cluster", sort=True)]
    k = len(clusters)
    est: list[float] = []
    for _ in range(n_boot):
        pick = [clusters[i] for i in rng.integers(0, k, k)]
        sub2 = pd.concat(pick, ignore_index=True)
        est.append(mh_or(stratify_tables(sub2, is_del, outcome)))
    est_arr = np.asarray(est, dtype=float)
    finite = est_arr[np.isfinite(est_arr)]
    lo, hi = float(np.nanpercentile(est_arr, 2.5)), float(np.nanpercentile(est_arr, 97.5))
    return lo, hi, float(len(finite)) / float(len(est_arr))


def step_14h() -> None:
    OUT_DIR__14h.mkdir(parents=True, exist_ok=True)

    support = pd.read_csv(SUPPORT_TSV, sep="\t", dtype={"herv_internal_id": str})
    support["esc_active"] = support["esc_support_fraction"].gt(0).astype(int)
    support["adult_active"] = support["adult_support_fraction"].gt(0).astype(int)
    support["t144_del"] = support["t144"].eq("T_del").astype(int)
    support["c158_del"] = support["c158"].eq("C_del").astype(int)

    canon = pd.read_parquet(
        M00__14h, columns=["herv_internal_id", "sequence"]
    ).astype({"herv_internal_id": str})
    canon = canon.set_index("herv_internal_id")["sequence"]
    ids = support["herv_internal_id"].tolist()
    seqs = [canon.loc[i] for i in ids]
    clusters = build_similarity_clusters(ids, seqs, N_CLUSTERS)
    support = support.merge(clusters, on="herv_internal_id", how="left", validate="one_to_one")

    rate_rows = []
    for state in ["T_intact+C_intact", "T_del+C_intact", "T_del+C_del"]:
        g = support.loc[support["state"].eq(state)]
        rate_rows.append(
            {
                "state": state,
                "n": int(len(g)),
                "esc_active_n": int(g["esc_active"].sum()),
                "esc_active_rate": float(g["esc_active"].mean()),
                "adult_active_n": int(g["adult_active"].sum()),
                "adult_active_rate": float(g["adult_active"].mean()),
            }
        )
    rate = pd.DataFrame(rate_rows)
    rate.to_csv(OUT_RATE, sep="\t", index=False)

    rng = np.random.default_rng(RNG_SEED)
    mh_rows = []
    for effect, mask, is_del in [
        ("T144", support["c158"].eq("C_intact"), "t144_del"),
        ("C158", support["t144"].eq("T_del"), "c158_del"),
    ]:
        sub = support.loc[mask].copy()
        for outcome in ["esc_active", "adult_active"]:
            pt = mh_or(stratify_tables(sub, is_del, outcome))
            lo, hi, finite_frac = cluster_bootstrap(sub, is_del, outcome, rng)
            mh_rows.append(
                {
                    "effect": effect,
                    "locus": "locus144_T_del" if effect == "T144" else "locus158_C_del",
                    "outcome": outcome,
                    "del_n": int(sub[is_del].sum()),
                    "intact_n": int((sub[is_del] == 0).sum()),
                    "mh_or": float(pt),
                    "boot_ci_low": float(lo),
                    "boot_ci_high": float(hi),
                    "finite_boot_frac": float(finite_frac),
                    "n_clusters": int(sub["similarity_cluster"].nunique()),
                }
            )


def main() -> None:
    step_14c()
    step_14d()
    step_14e()
    step_14f()
    step_14g()
    step_14h()


if __name__ == "__main__":
    main()
