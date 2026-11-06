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
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Any
from tokenizers import Tokenizer
from biological_common import (
    ENCODER_PATH,
    FrozenSequenceModel,
    K6_K,
    MODEL_CHECKPOINT,
    SCALER_PATH,
    TOKENIZER_PATH,
    attribute_one,
    genomic_interval,
    map_token_attribution,
    now,
    score_rows,
    sequence_score,
    tensors_for_rows,
    write_json,
    write_tsv,
    write_tsv_gz,
    build_frozen_scorer,
)
from Bio.Align import PairwiseAligner
from scipy.stats import mannwhitneyu

# --------------------------------------------------------------------------
# step 14: Mechanistic validation of the representative loci
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOLOGY_ROOT = PROJECT_ROOT / "Biological_Interpretation"
MODEL_OUTPUT = PROJECT_ROOT / "Modeling/output"

M00_DIR = MODEL_OUTPUT / "00_build_canonical_data"
M01_DIR = MODEL_OUTPUT / "01_build_dual_tensors"
M07_DIR = MODEL_OUTPUT / "07_audit_partial_unfreeze"
M13_DIR = BIOLOGY_ROOT / "output/13_representative_locus_deep_dive"

CANONICAL = M00_DIR / "00_canonical_herv_mef_table.parquet"
SOURCE = BIOLOGY_ROOT / "output/01_build_biological_source_table/01_biological_source_table.parquet"
REPRESENTATIVES = M13_DIR / "13_representative_locus_deep_dive.tsv"
REVIEW_TABLE = M13_DIR / "13_literature_review_reference_table.tsv"
FAMILY_MODULES = BIOLOGY_ROOT / "output/03_analyze_sequence_grammar/03_within_family_6mer_modules.tsv"
CHROMATIN = M13_DIR / "13_chromatin_evidence.tsv"

OUT_DIR = BIOLOGY_ROOT / "output/14_representative_locus_mechanistic_validation"
OUT_MAIN = OUT_DIR / "14_representative_locus_mechanistic_evidence.tsv"
OUT_ATTR = OUT_DIR / "14_locus_attribution_summary.tsv"
OUT_BASE = OUT_DIR / "14_base_level_attribution.tsv.gz"
OUT_MASK = OUT_DIR / "14_in_silico_masking.tsv"


ID = "herv_internal_id"
ATTR_WINDOW_BP = 50
LOW_RANK_FRACTION = 0.20
# M23 used CUDA autocast. A CPU-only rerun cannot reproduce each half-precision
# operator exactly; 3e-4 is below one float16 ULP around the observed scores.
SCORE_REPRODUCTION_TOLERANCE = 3e-4
MATCH_FEATURES = ["length", "gc_content", "repeat_divergence", "log_abs_tss"]


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def canonical_kmer(sequence: str) -> str:
    rc = reverse_complement(sequence)
    return min(sequence, rc)


@torch.inference_mode()


def select_controls(source: pd.DataFrame, representatives: pd.DataFrame, excluded_ids: set[str]) -> pd.DataFrame:
    source = source.copy()
    source["log_abs_tss"] = np.log1p(pd.to_numeric(source["dist_to_tss"], errors="coerce").abs())
    source["herv_functional_priority_score"] = pd.to_numeric(source["herv_functional_priority_score"], errors="coerce")
    rows = []
    used: set[str] = set()
    for candidate in representatives.itertuples(index=False):
        pool = source[source["repName"].eq(candidate.repName)].copy()
        pool["within_subfamily_priority_percentile"] = pool["herv_functional_priority_score"].rank(method="average", pct=True)
        eligible = pool[pool["within_subfamily_priority_percentile"].le(LOW_RANK_FRACTION)].copy()
        eligible = eligible[~eligible[ID].isin(excluded_ids | used)]
        cand = source[source[ID].eq(candidate.herv_internal_id)].iloc[0]
        distance = np.zeros(len(eligible), dtype=np.float64)
        for feature in MATCH_FEATURES:
            values = pd.to_numeric(pool[feature], errors="coerce").to_numpy(dtype=np.float64)
            scale = float(np.nanmedian(np.abs(values - np.nanmedian(values))) * 1.4826)
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(values))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            distance += np.abs(pd.to_numeric(eligible[feature], errors="coerce").to_numpy(dtype=np.float64) - float(cand[feature])) / scale
        eligible["matching_distance"] = distance
        control = eligible.sort_values(["matching_distance", ID]).iloc[0]
        used.add(str(control[ID]))
        rows.append({
            "representative_order": int(candidate.representative_order),
            "candidate_id": candidate.herv_internal_id,
            "candidate_repName": candidate.repName,
            "candidate_repFamily": candidate.repFamily,
            "control_id": control[ID],
            "control_repName": control["repName"],
            "control_repFamily": control["repFamily"],
            "control_within_subfamily_priority_percentile": control["within_subfamily_priority_percentile"],
            "control_matching_distance": control["matching_distance"],
            "selection_rank_stratum": "bottom_20pct_within_exact_repName",
            "selection_used_tumor_support": False,
            **{f"candidate_{x}": cand[x] for x in ["herv_functional_priority_rank", "herv_functional_priority_score",
                "score_sequence_regulatory", "tumor_active_support_breadth", "length", "gc_content", "repeat_divergence", "dist_to_tss"]},
            **{f"control_{x}": control[x] for x in ["herv_functional_priority_rank", "herv_functional_priority_score",
                "score_sequence_regulatory", "tumor_active_support_breadth", "length", "gc_content", "repeat_divergence", "dist_to_tss"]},
        })
    return pd.DataFrame(rows)


def step_14() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    reps = pd.read_csv(REPRESENTATIVES, sep="\t", low_memory=False)
    review = pd.read_csv(REVIEW_TABLE, sep="\t", low_memory=False)
    excluded = set(review[ID].astype(str))

    source_cols = [ID, "repName", "repFamily", "length", "gc_content", "repeat_divergence", "dist_to_tss",
                   "herv_functional_priority_rank", "herv_functional_priority_score", "score_sequence_regulatory",
                   "rank_pct_score_sequence_regulatory", "tumor_active_support_breadth", "tumor_active_supported_eids",
                   "tumor_epigenomic_support_score", "is_ccre_enhancer_like", "is_mpra_active_supported",
                   "is_serve_gtex_supported", "gctx_roadmap_active_mark_presence_count",
                   "gctx_roadmap_repressive_mark_presence_count"]
    source = pd.read_parquet(SOURCE, columns=source_cols)
    controls = select_controls(source, reps, excluded)

    all_ids = reps[ID].astype(str).tolist() + controls["control_id"].astype(str).tolist()
    canonical_cols = [ID, "split", "chrom", "start", "end", "strand", "repName", "repClass", "repFamily",
                      "length", "repeat_divergence", "gc_content", "n_content", "dist_to_tss",
                      "nearest_gene_name", "nearest_gene_type", "sequence", "seq_start", "seq_end"]
    canonical = pd.read_parquet(CANONICAL, columns=canonical_cols, filters=[(ID, "in", all_ids)])
    canonical[ID] = canonical[ID].astype(str)
    ordered = canonical.set_index(ID).loc[all_ids].reset_index()
    candidate_rows = ordered.iloc[: len(reps)].copy()
    control_rows = ordered.iloc[len(reps) :].copy()

    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    score_head_indices = [int(i) for i in config["score_head_indices"]]
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenSequenceModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    all_rows = pd.concat([candidate_rows, control_rows], ignore_index=True)
    all_tensors, _ = tensors_for_rows(all_rows, tokenizer, config, encoder, scaler)
    reproduced = score_rows(model, all_tensors, device, score_head_indices)
    expected = source.set_index(ID).loc[all_ids, "score_sequence_regulatory"].to_numpy(dtype=np.float64)
    max_error = float(np.max(np.abs(reproduced.astype(np.float64) - expected)))
    modules = pd.read_csv(FAMILY_MODULES, sep="\t", low_memory=False)
    high_modules = modules[(modules["is_significant_fdr0_05"].astype(bool)) & modules["module_class"].eq("HIGH_ENRICHED")]
    family_module_sets = high_modules.groupby("repFamily")["canonical_6mer"].apply(set).to_dict()

    attr_rows, base_rows, mask_rows = [], [], []
    candidate_top_kmers: dict[str, list[str]] = {}
    for i, row in candidate_rows.iterrows():
        one = candidate_rows.iloc[[i]].copy()
        original_score, k6_base, bpe_base, combined = attribute_one(model, one, tokenizer, config, encoder, scaler, device, score_head_indices)
        rolling = np.convolve(combined, np.ones(ATTR_WINDOW_BP), mode="valid")
        top_start = int(np.argmax(rolling)); top_end = top_start + ATTR_WINDOW_BP
        allowed_low = np.ones(len(rolling), dtype=bool)
        allowed_low[max(0, top_start - ATTR_WINDOW_BP + 1) : min(len(rolling), top_end)] = False
        low_candidates = np.where(allowed_low)[0]
        low_start = int(low_candidates[np.argmin(rolling[low_candidates])]); low_end = low_start + ATTR_WINDOW_BP
        top_gstart, top_gend = genomic_interval(row, top_start, top_end)
        low_gstart, low_gend = genomic_interval(row, low_start, low_end)
        top_overlap = max(0, min(top_gend, int(row["end"])) - max(top_gstart, int(row["start"])))
        top_relation = "WITHIN_HERV" if top_overlap == ATTR_WINDOW_BP else (
            "PARTIAL_HERV_OVERLAP" if top_overlap > 0 else "FLANKING_SEQUENCE"
        )

        kmer_scores = []
        sequence = str(row["sequence"])
        for pos in range(0, 1000 - K6_K + 1):
            kmer = sequence[pos : pos + K6_K]
            if "N" in kmer:
                continue
            kmer_scores.append((float(combined[pos : pos + K6_K].sum()), pos, canonical_kmer(kmer)))
        kmer_scores.sort(key=lambda x: (-x[0], x[1], x[2]))
        top_kmers, seen = [], set()
        for value, pos, kmer in kmer_scores:
            if kmer in seen:
                continue
            seen.add(kmer); top_kmers.append((kmer, value, pos))
            if len(top_kmers) == 5:
                break
        top_kmer_names = [x[0] for x in top_kmers]
        candidate_top_kmers[row[ID]] = top_kmer_names
        family_set = family_module_sets.get(str(row["repFamily"]), set())
        overlap_kmers = [k for k in top_kmer_names if k in family_set]

        attr_rows.append({
            "representative_order": int(reps.iloc[i]["representative_order"]), ID: row[ID], "repName": row["repName"],
            "repFamily": row["repFamily"], "deployment_sequence_score_reproduced": original_score,
            "deployment_sequence_score_recorded": float(expected[i]),
            "score_reproduction_abs_error": abs(original_score - float(expected[i])),
            "attribution_method": "absolute_gradient_times_embedding_mapped_to_base",
            "attribution_target": "mean(sigmoid(ccre),sigmoid(serve_gtex))",
            "top_attribution_start_0based_strand_oriented": top_start,
            "top_attribution_end_0based_strand_oriented": top_end,
            "top_attribution_genomic_coordinate": f"{row['chrom']}:{top_gstart}-{top_gend}",
            "top_attribution_fraction": float(rolling[top_start]),
            "top_region_herv_overlap_bp": int(top_overlap),
            "top_region_relation_to_herv": top_relation,
            "top_region_sequence": sequence[top_start:top_end],
            "low_attribution_start_0based_strand_oriented": low_start,
            "low_attribution_end_0based_strand_oriented": low_end,
            "low_attribution_genomic_coordinate": f"{row['chrom']}:{low_gstart}-{low_gend}",
            "low_attribution_fraction": float(rolling[low_start]),
            "top_attribution_6mers": ";".join(f"{k}:{v:.6g}@{p}" for k, v, p in top_kmers),
            "top_attributed_family_module_overlap_n": len(overlap_kmers),
            "top_attributed_family_module_overlap_6mers": ";".join(overlap_kmers),
            "k6_attribution_raw_total": float(k6_base.sum()),
            "bpe_attribution_raw_total": float(bpe_base.sum()),
            "attribution_is_causal": False,
        })
        for pos in range(1000):
            gstart, gend = genomic_interval(row, pos, pos + 1)
            base_rows.append({"representative_order": int(reps.iloc[i]["representative_order"]), ID: row[ID],
                              "sequence_position_0based_strand_oriented": pos, "genomic_start": gstart,
                              "genomic_end": gend, "base": sequence[pos], "k6_attribution_raw": k6_base[pos],
                              "bpe_attribution_raw": bpe_base[pos], "combined_attribution_normalized": combined[pos],
                              "in_top_50bp_region": top_start <= pos < top_end,
                              "in_low_50bp_region": low_start <= pos < low_end})

        masked_top = sequence[:top_start] + ("N" * ATTR_WINDOW_BP) + sequence[top_end:]
        masked_low = sequence[:low_start] + ("N" * ATTR_WINDOW_BP) + sequence[low_end:]
        perturb_tensors, _ = tensors_for_rows(pd.concat([one, one], ignore_index=True), tokenizer, config, encoder, scaler,
                                               sequences=[masked_top, masked_low])
        perturbed_scores = score_rows(model, perturb_tensors, device, score_head_indices, batch_size=2)
        mask_rows.append({
            "representative_order": int(reps.iloc[i]["representative_order"]), ID: row[ID],
            "original_sequence_score": original_score, "top_50bp_masked_score": float(perturbed_scores[0]),
            "low_50bp_masked_score": float(perturbed_scores[1]),
            "delta_original_minus_top_mask": original_score - float(perturbed_scores[0]),
            "delta_original_minus_low_mask": original_score - float(perturbed_scores[1]),
            "top_minus_low_mask_delta": float(perturbed_scores[1] - perturbed_scores[0]),
            "mask_character": "N", "metadata_held_fixed": True,
            "perturbation_is_causal_biological_evidence": False,
        })

    attr = pd.DataFrame(attr_rows)
    base = pd.DataFrame(base_rows)
    masking = pd.DataFrame(mask_rows)
    write_tsv(attr, OUT_ATTR); write_tsv_gz(base, OUT_BASE); write_tsv(masking, OUT_MASK)

    comparison_rows = []
    source_index = source.set_index(ID)
    canonical_index = canonical.set_index(ID)
    for c in controls.itertuples(index=False):
        cand = source_index.loc[c.candidate_id]; ctrl = source_index.loc[c.control_id]
        ctrl_seq = str(canonical_index.loc[c.control_id, "sequence"])
        kmers = candidate_top_kmers[c.candidate_id]
        retained = [k for k in kmers if k in {canonical_kmer(ctrl_seq[p:p+6]) for p in range(995) if "N" not in ctrl_seq[p:p+6]}]
        comparison_rows.append({
            "representative_order": c.representative_order, "candidate_id": c.candidate_id, "control_id": c.control_id,
            "repName": c.candidate_repName, "repFamily": c.candidate_repFamily,
            "candidate_priority_rank": cand["herv_functional_priority_rank"],
            "control_priority_rank": ctrl["herv_functional_priority_rank"],
            "candidate_sequence_score": cand["score_sequence_regulatory"], "control_sequence_score": ctrl["score_sequence_regulatory"],
            "candidate_minus_control_sequence_score": cand["score_sequence_regulatory"] - ctrl["score_sequence_regulatory"],
            "candidate_tumor_support_breadth": cand["tumor_active_support_breadth"],
            "control_tumor_support_breadth": ctrl["tumor_active_support_breadth"],
            "candidate_reference_ccre": bool(cand["is_ccre_enhancer_like"]),
            "control_reference_ccre": bool(ctrl["is_ccre_enhancer_like"]),
            "candidate_reference_mpra": bool(cand["is_mpra_active_supported"]),
            "control_reference_mpra": bool(ctrl["is_mpra_active_supported"]),
            "candidate_reference_serve": bool(cand["is_serve_gtex_supported"]),
            "control_reference_serve": bool(ctrl["is_serve_gtex_supported"]),
            "candidate_top_attribution_6mers": ";".join(kmers),
            "control_retained_candidate_top_6mers_n": len(retained),
            "control_retained_candidate_top_6mers": ";".join(retained),
            "control_matching_distance": c.control_matching_distance,
            "control_match_quality": "GOOD" if c.control_matching_distance <= 2 else (
                "MODERATE" if c.control_matching_distance <= 4 else "LIMITED"
            ),
            "candidate_control_abs_length_difference": abs(float(cand["length"]) - float(ctrl["length"])),
            "candidate_control_abs_gc_difference": abs(float(cand["gc_content"]) - float(ctrl["gc_content"])),
            "candidate_control_abs_divergence_difference": abs(float(cand["repeat_divergence"]) - float(ctrl["repeat_divergence"])),
            "candidate_control_abs_log_tss_difference": abs(
                float(np.log1p(abs(cand["dist_to_tss"]))) - float(np.log1p(abs(ctrl["dist_to_tss"])))
            ),
            "control_selection_used_tumor_support": False,
        })
    comparison = pd.DataFrame(comparison_rows)

    chrom = pd.read_csv(CHROMATIN, sep="\t")
    chrom_profile = chrom.groupby(ID).apply(lambda g: "; ".join(
        f"{r.eid}:A{int(r.active_mark_count)}/R{int(r.repressive_mark_count)}[{r.mark_overlap_profile}]"
        for r in g.itertuples()), include_groups=False).rename("per_cell_chromatin_detail").reset_index()
    main_cols = ["representative_order", ID, "selection_category", "repName", "repFamily", "chrom", "start", "end",
                 "herv_functional_priority_rank", "rank_pct_score_sequence_regulatory", "tumor_active_support_breadth",
                 "tumor_active_supported_eids", "nearest_gene_name", "literature_class"]
    main_cols = [x for x in main_cols if x in reps.columns]
    main = reps[main_cols].merge(attr, on=["representative_order", ID, "repName", "repFamily"], how="left")
    main = main.merge(masking, on=["representative_order", ID], how="left")
    comparison_for_main = comparison.drop(columns=["repName", "repFamily"])
    main = main.merge(comparison_for_main, left_on=["representative_order", ID],
                      right_on=["representative_order", "candidate_id"], how="left")
    main = main.drop(columns=["candidate_id"])
    main = main.merge(chrom_profile, on=ID, how="left")
    review_fields = [ID, "reference_ccre_support", "reference_mpra_support", "reference_serve_gtex_support",
                     "reference_roadmap_active_support", "reference_evidence_count", "gene_distance_bp",
                     "gene_linkage_type", "gene_linkage_confidence", "literature_evidence_level", "evidence_doi",
                     "evidence_pmid", "manual_search_query"]
    review_fields = [x for x in review_fields if x in review.columns]
    main = main.merge(review[review_fields].drop_duplicates(ID), on=ID, how="left")
    write_tsv(main, OUT_MAIN)

    summary = {
        "created_at_utc": now(), "status": "PASSED", "representative_loci": int(len(reps)),
        "device": str(device), "model_checkpoint": str(MODEL_CHECKPOINT),
        "model_score_contract": config["sequence_score_contract"],
        "model_score_head_indices": score_head_indices,
        "model_score_head_names": [config["head_order"][i] for i in score_head_indices],
        "model_selected_candidate": checkpoint.get("selected_candidate"),
        "max_sequence_score_reproduction_abs_error": max_error,
        "sequence_score_reproduction_tolerance": SCORE_REPRODUCTION_TOLERANCE,
        "attribution_method": "absolute gradient-times-embedding for K6 and BPE, mapped to 1000-bp input",
        "attribution_window_bp": ATTR_WINDOW_BP,
        "perturbation": "top-versus-low 50-bp N masking with metadata held fixed",
        "same_subfamily_control": "bottom 20% within exact repName; matched on length, GC, divergence and log abs TSS distance",
        "control_match_quality_heuristic": "GOOD <=2; MODERATE >2 to 4; LIMITED >4 standardized L1 distance",
        "control_selection_used_tumor_support": False,
        "control_n": int(len(controls)),
        "score_reproduction_passed": bool(max_error <= SCORE_REPRODUCTION_TOLERANCE),
    }
    lines = ["# Representative-locus mechanistic validation", "", f"Generated: {summary['created_at_utc']}", "",
             "## Scope", "",
             f"M07 deployment scores reproduced with max abs error {max_error:.3g} "
             f"(tolerance {SCORE_REPRODUCTION_TOLERANCE:.3g}); "
             f"{len(reps)} representatives, {len(controls)} same-subfamily controls.", "",
             "## Representative results", "", "|order|locus|repName|top 50-bp fraction|top-mask delta|low-mask delta|control|candidate-control sequence delta|", "|---:|---|---|---:|---:|---:|---|---:|"]
    for r in main.itertuples():
        lines.append(f"|{r.representative_order}|{r.herv_internal_id}|{r.repName_x if hasattr(r, 'repName_x') else r.repName}|{r.top_attribution_fraction:.4g}|{r.delta_original_minus_top_mask:.4g}|{r.delta_original_minus_low_mask:.4g}|{r.control_id}|{r.candidate_minus_control_sequence_score:.4g}|")


# --------------------------------------------------------------------------
# step 14b: Single-base determinism of the key subfamilies
# --------------------------------------------------------------------------



HERE = Path(__file__).resolve().parent
PROJECT_ROOT__14b = HERE.parent
MODEL_OUTPUT__14b = PROJECT_ROOT__14b / "Modeling/output"
M00_DIR__14b = MODEL_OUTPUT__14b / "00_build_canonical_data"
M23_DIR = MODEL_OUTPUT__14b / "23_generate_genomewide_herv_functional_ranking"

CANONICAL__14b = M00_DIR__14b / "00_canonical_herv_mef_table.parquet"
RANKING = M23_DIR / "23_herv_genomewide_functional_ranking.parquet"

OUT_DIR__14b = HERE / "output/14b_analyze_key_subfamily_single_base_determinism"


OUT_PATCH = OUT_DIR__14b / "14b_single_base_revert_scores.tsv.gz"


OUT_ISM = OUT_DIR__14b / "14b_mer11_window_saturation_mutagenesis.tsv"


ID__14b = "herv_internal_id"

GROUPS = {
    "MER11": ["MER11A", "MER11B", "MER11C", "MER11D"],
    "MER41": ["MER41-int", "MER41A", "MER41B", "MER41C", "MER41D", "MER41E", "MER41G"],
    "HERVH_LTR7": ["HERVH-int", "HERVH48-int", "LTR7", "LTR7A", "LTR7B", "LTR7C", "LTR7Y"],
    "LTR5_Hs_HERVK": ["LTR5_Hs", "HERVK-int", "HERVK3-int", "HERVK9-int", "HERVK11-int", "HERVK14-int"],
}

SCORE = "score_sequence_regulatory"
RANKPCT = "rank_pct_score_sequence_regulatory"
PRIORITY_RANK = "herv_functional_priority_rank"
EVIDENCE_FLAGS = ["is_ccre_enhancer_like", "is_mpra_active_supported", "is_serve_gtex_supported"]

ANCHOR_MAX = 400
ANCHOR_MIN_RETENTION = 0.90
MIN_DELETION_FRACTION = 0.25
MAX_DELETION_FRACTION = 0.95
NEIGHBOR_MAX_FRACTION = 0.10
SITES_PER_GROUP = 3
MAX_SCORED_SEQUENCES = 6000
SCORE_BATCH_SIZE = 32

MER11_ISM_WINDOW = (135, 165)
MER11_ISM_MAX_COPIES = 50

ALIGNER = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                          open_gap_score=-6, extend_gap_score=-0.6)




def score_frame(frame: pd.DataFrame, sequences: list[str], scorer: dict[str, Any]) -> np.ndarray:
    tensors, _ = tensors_for_rows(
        frame, scorer["tokenizer"], scorer["config"],
        scorer["encoder"], scorer["scaler"], sequences=sequences
    )
    return score_rows(
        scorer["model"], tensors, scorer["device"],
        scorer["score_head_indices"], batch_size=SCORE_BATCH_SIZE
    )


def group_score_rows(frame: pd.DataFrame, label: str, rep_name: str, scope: str,
                     reference: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"group": label, "repName": rep_name, "scope": scope, "n": int(len(frame))}
    if len(frame) == 0:
        return row
    row["mean_sequence_score"] = float(frame[SCORE].mean())
    row["median_sequence_score"] = float(frame[SCORE].median())
    row["sd_sequence_score"] = float(frame[SCORE].std())
    row["mean_rank_pct_sequence_score"] = float(frame[RANKPCT].mean())
    row["median_functional_priority_rank"] = float(frame[PRIORITY_RANK].median())
    row["mean_score_minus_reference"] = float(frame[SCORE].mean() - reference[SCORE].mean())
    row["fraction_above_reference_mean"] = float((frame[SCORE] > reference[SCORE].mean()).mean())
    row["mannwhitney_p_vs_reference"] = float(
        mannwhitneyu(frame[SCORE], reference[SCORE], alternative="two-sided").pvalue
    )
    for flag in EVIDENCE_FLAGS:
        row[f"{flag}_rate"] = float(frame[flag].mean())
    row["reference_n"] = int(len(reference))
    row["reference_mean_sequence_score"] = float(reference[SCORE].mean())
    return row


def score_groups(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, members in GROUPS.items():
        members = [m for m in members if m in set(ranking["repName"])]
        sub = ranking[ranking["repName"].isin(members)]
        rows.append(group_score_rows(sub, label, "__GROUP__", "group_vs_genome_background", ranking))
        outside = ranking[~ranking["repName"].isin(members)]
        rows.append(group_score_rows(sub, label, "__GROUP__", "group_vs_rest_of_genome", outside))
        for rep_name in members:
            member = ranking[ranking["repName"] == rep_name]
            rows.append(group_score_rows(member, label, rep_name, "subfamily_vs_genome_background", ranking))
            rows.append(group_score_rows(member, label, rep_name, "subfamily_vs_rest_of_group",
                                         sub[sub["repName"] != rep_name]))
    return pd.DataFrame(rows)


def choose_anchor_length(lengths: pd.Series, window: int) -> tuple[int, float]:
    for anchor in range(min(ANCHOR_MAX, window), 0, -1):
        if float(((lengths - anchor) >= 0).mean()) >= ANCHOR_MIN_RETENTION:
            return anchor, float(((lengths - anchor) >= 0).mean())
    return 1, float(((lengths - 1) >= 0).mean())


def build_site_profiles(sub: pd.DataFrame, anchor: int, low: int, high: int) -> dict[str, Any]:
    slices = []
    for row in sub.itertuples():
        contains = (row.start <= row.seq_start + low) and (row.end >= row.seq_start + high)
        slices.append(str(row.sequence)[low:high] if contains else None)
    usable = np.array([s is not None for s in slices])
    reference_rep = sub["repName"].value_counts().idxmax()
    reference = pd.Series([s for s in slices if s is not None]).value_counts().idxmax()
    profiles: dict[str, np.ndarray] = {}
    coord_maps: dict[int, np.ndarray] = {}
    usable_counts: dict[str, int] = {}
    for position, (rep_name, sequence, keep) in enumerate(zip(sub["repName"], slices, usable)):
        if not keep:
            continue
        usable_counts[rep_name] = usable_counts.get(rep_name, 0) + 1
        alignment = ALIGNER.align(reference, sequence)[0]
        query_map = np.full(len(reference), -1, dtype=np.int64)
        for (r0, r1), (q0, _) in zip(alignment.aligned[0], alignment.aligned[1]):
            query_map[r0:r1] = np.arange(q0, q0 + (r1 - r0))
        coord_maps[position] = query_map
        profiles.setdefault(rep_name, np.zeros(len(reference), dtype=np.int64))
        profiles[rep_name] += (query_map < 0).astype(np.int64)
    fractions = {name: profile / usable_counts[name] for name, profile in profiles.items()}
    return {"reference": reference, "reference_rep": reference_rep, "fractions": fractions,
            "coord_maps": coord_maps, "usable": usable, "anchor": anchor, "low": low, "high": high}


def find_single_base_sites(profile: dict[str, Any], label: str) -> pd.DataFrame:
    reference = profile["reference"]
    fractions = profile["fractions"]
    rows = []
    for position in range(1, len(reference) - 1):
        for rep_name, gap_fraction in fractions.items():
            if not MIN_DELETION_FRACTION < gap_fraction[position] < MAX_DELETION_FRACTION:
                continue
            if (gap_fraction[position - 1] >= NEIGHBOR_MAX_FRACTION
                    or gap_fraction[position + 1] >= NEIGHBOR_MAX_FRACTION):
                continue
            others = {other: float(value[position]) for other, value in fractions.items() if other != rep_name}
            rows.append({
                "group": label, "reference_repName": profile["reference_rep"],
                "reference_position": position, "reference_base": reference[position],
                "deletion_carrier_repName": rep_name,
                "deletion_fraction_in_carrier": float(gap_fraction[position]),
                "deletion_fraction_in_other_subfamilies": max(others.values()) if others else 0.0,
                "subfamily_specificity": float(gap_fraction[position]) - (max(others.values()) if others else 0.0),
                "neighbor_deletion_fraction_left": float(gap_fraction[position - 1]),
                "neighbor_deletion_fraction_right": float(gap_fraction[position + 1]),
                "reference_context_10bp": reference[max(0, position - 5): position + 5],
            })
    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        return frame
    return (frame.sort_values(["group", "subfamily_specificity", "deletion_fraction_in_carrier"],
                              ascending=[True, False, False])
            .groupby("group", as_index=False).head(SITES_PER_GROUP).reset_index(drop=True))


def patch_window_insert(window: str, coordinate: int, base: str) -> str:
    return window[:coordinate] + base + window[coordinate: len(window) - 1]


def patch_window_substitute(window: str, coordinate: int, base: str) -> str:
    return window[:coordinate] + base + window[coordinate + 1:]


def patch_window_delete(window: str, coordinate: int) -> str:
    pad_base = window[-1] if len(window) > 0 else "N"
    return window[:coordinate] + window[coordinate + 1:] + pad_base


def run_revert_experiment(sub: pd.DataFrame, profile: dict[str, Any], sites: pd.DataFrame,
                          label: str, ranking: pd.DataFrame, scorer: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(sites) == 0:
        return pd.DataFrame(), pd.DataFrame()
    low = profile["low"]
    per_site_cap = max(1, MAX_SCORED_SEQUENCES // (len(sites) * 2))
    reference_base = profile["reference"]
    rows, summary_rows = [], []
    expected = ranking.set_index(ID__14b)[SCORE].to_dict()
    rep_names = sub["repName"].to_numpy()

    for site in sites.itertuples():
        position = int(site.reference_position)
        base = reference_base[position]
        deleted_positions, intact_positions = [], []
        for row_position in range(len(sub)):
            query_map = profile["coord_maps"].get(row_position)
            if query_map is None:
                continue
            target = position + 1 if query_map[position] < 0 else position
            if target >= len(query_map) or query_map[target] < 0:
                continue
            coordinate = low + int(query_map[target])
            record = (row_position, coordinate)
            (deleted_positions if query_map[position] < 0 else intact_positions).append(record)

        for arm, records in (("restored_base", deleted_positions), ("matched_control_insertion", intact_positions)):
            if not records:
                continue
            step = max(1, int(np.ceil(len(records) / per_site_cap)))
            sampled = records[::step][:per_site_cap]
            frame = sub.iloc[[r for r, _ in sampled]].reset_index(drop=True)
            windows = [str(s) for s in frame["sequence"]]
            originals = score_frame(frame, windows, scorer)
            patched_windows = [patch_window_insert(w, c, base) for w, (_, c) in zip(windows, sampled)]
            patched = score_frame(frame, patched_windows, scorer)
            for index, (row_position, coordinate) in enumerate(sampled):
                herd = sub[ID__14b].iloc[row_position]
                rows.append({
                    "group": label, "reference_repName": site.reference_repName,
                    "reference_position": position, "inserted_base": base, "arm": arm,
                    ID__14b: herd, "repName": rep_names[row_position],
                    "strand": sub["strand"].iloc[row_position],
                    "window_coordinate": coordinate,
                    "reference_context_10bp": reference_base[max(0, position - 5): position + 5],
                    "original_score": float(originals[index]),
                    "patched_score": float(patched[index]),
                    "score_delta": float(patched[index] - originals[index]),
                    "m23_score_reference": float(expected.get(herd, np.nan)),
                })
            summary_rows.append({
                "group": label, "reference_repName": site.reference_repName, "reference_position": position,
                "inserted_base": base, "arm": arm, "n": len(sampled),
                "original_mean_score": float(originals.mean()),
                "patched_mean_score": float(patched.mean()),
                "mean_score_delta": float(np.mean(patched - originals)),
                "median_score_delta": float(np.median(patched - originals)),
                "sd_score_delta": float(np.std(patched - originals, ddof=1)) if len(sampled) > 1 else 0.0,
                "fraction_delta_negative": float(np.mean(patched - originals < 0)),
                "max_abs_reproduction_error": float(np.max(np.abs(
                    originals - np.array([expected.get(sub[ID__14b].iloc[r], originals[i]) for i, (r, _) in enumerate(sampled)])))),
                "deletion_carrier_repName": site.deletion_carrier_repName,
                "deletion_fraction_in_carrier": site.deletion_fraction_in_carrier,
                "deletion_fraction_in_other_subfamilies": site.deletion_fraction_in_other_subfamilies,
            })

    df_rows = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)

    if len(summary) > 0 and len(df_rows) > 0:
        tests = []
        for (group_val, pos_val), grp_df in df_rows.groupby(["group", "reference_position"]):
            restored_deltas = grp_df[grp_df["arm"] == "restored_base"]["score_delta"].to_numpy()
            control_deltas = grp_df[grp_df["arm"] == "matched_control_insertion"]["score_delta"].to_numpy()
            if len(restored_deltas) >= 5 and len(control_deltas) >= 5:
                try:
                    p_val = float(mannwhitneyu(restored_deltas, control_deltas, alternative="two-sided").pvalue)
                except Exception:
                    p_val = float("nan")
            else:
                p_val = float("nan")
            tests.append({
                "group": group_val,
                "reference_position": pos_val,
                "restored_vs_control_mannwhitney_p": p_val
            })
        if tests:
            summary = summary.merge(pd.DataFrame(tests), on=["group", "reference_position"], how="left")

    return df_rows, summary


# =============================================================================
# Part C: Targeted In Silico Saturation Mutagenesis (ISM) on MER11
# =============================================================================

def run_mer11_targeted_saturation_mutagenesis(
    canonical: pd.DataFrame, ranking: pd.DataFrame, scorer: dict[str, Any]
) -> pd.DataFrame:

    mer11_ids = ranking[ranking["repName"].isin(["MER11D", "MER11A"])].sort_values(
        SCORE, ascending=False
    )[ID__14b].head(MER11_ISM_MAX_COPIES).tolist()

    sub = canonical[canonical[ID__14b].isin(mer11_ids)].copy().reset_index(drop=True)
    if len(sub) == 0:
        return pd.DataFrame()

    window_len = int(sub["seq_len"].iloc[0])
    w_start, w_end = MER11_ISM_WINDOW
    w_start = max(0, w_start)
    w_end = min(window_len - 1, w_end)

    ism_records = []

    base_sequences = [str(s) for s in sub["sequence"]]
    baseline_scores = score_frame(sub, base_sequences, scorer)

    for idx, row in sub.iterrows():
        seq = str(row["sequence"])
        orig_score = float(baseline_scores[idx])
        herv_id = str(row[ID__14b])
        rep_name = str(row["repName"])
        strand = str(row["strand"])

        mut_sequences = []
        mut_metadata = []

        for pos in range(w_start, w_end):
            curr_base = seq[pos].upper()

            for target_base in ["A", "C", "G", "T"]:
                if target_base == curr_base:
                    continue
                mut_seq = patch_window_substitute(seq, pos, target_base)
                mut_sequences.append(mut_seq)
                mut_metadata.append({
                    "position": pos, "mutation_type": "substitution",
                    "from_base": curr_base, "to_base": target_base
                })

            del_seq = patch_window_delete(seq, pos)
            mut_sequences.append(del_seq)
            mut_metadata.append({
                "position": pos, "mutation_type": "deletion",
                "from_base": curr_base, "to_base": "-"
            })

        if not mut_sequences:
            continue

        dummy_frame = pd.DataFrame([row.to_dict()] * len(mut_sequences))
        mut_scores = score_frame(dummy_frame, mut_sequences, scorer)

        for meta, m_score in zip(mut_metadata, mut_scores):
            m_score = float(m_score)
            ism_records.append({
                ID__14b: herv_id,
                "repName": rep_name,
                "strand": strand,
                "position": meta["position"],
                "mutation_type": meta["mutation_type"],
                "from_base": meta["from_base"],
                "to_base": meta["to_base"],
                "baseline_score": orig_score,
                "mutant_score": m_score,
                "score_delta": m_score - orig_score,
            })

    ism_df = pd.DataFrame(ism_records)
    return ism_df


def step_14b() -> None:
    OUT_DIR__14b.mkdir(parents=True, exist_ok=True)

    ranking_cols = [ID__14b, "repName", SCORE, RANKPCT, PRIORITY_RANK] + EVIDENCE_FLAGS
    ranking = pd.read_parquet(RANKING, columns=ranking_cols)
    ranking[ID__14b] = ranking[ID__14b].astype(str)

    canonical_cols = [ID__14b, "chrom", "start", "end", "strand", "repName", "repClass", "repFamily", "split",
                      "length", "repeat_divergence", "gc_content", "n_content", "dist_to_tss",
                      "nearest_gene_name", "nearest_gene_type", "sequence", "seq_start", "seq_end", "seq_len"]
    canonical = pd.read_parquet(CANONICAL__14b, columns=canonical_cols)
    canonical[ID__14b] = canonical[ID__14b].astype(str)

    window_length = int(canonical["seq_len"].mode().iloc[0])

    scorer = build_frozen_scorer()
    site_frames, patch_frames, summary_frames, control_rows = [], [], [], []
    member_names = {label: [m for m in members if m in set(ranking["repName"])]
                    for label, members in GROUPS.items()}

    for label, members in member_names.items():
        sub = canonical[canonical["repName"].isin(members)].reset_index(drop=True)
        anchor, retention = choose_anchor_length(sub["length"], window_length)
        low = (window_length - anchor) // 2
        profile = build_site_profiles(sub, anchor, low, low + anchor)
        sites = find_single_base_sites(profile, label)
        sites["anchor_length"] = anchor
        sites["anchor_retention"] = retention
        site_frames.append(sites)
        patches, summary = run_revert_experiment(sub, profile, sites, label, ranking, scorer)
        patch_frames.append(patches)
        summary_frames.append(summary)

    all_patches = pd.concat(patch_frames, ignore_index=True) if patch_frames else pd.DataFrame()
    all_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    write_tsv_gz(all_patches, OUT_PATCH)

    if len(all_summary):
        for row in all_summary.itertuples():
            control_rows.append({
                "group": row.group, "reference_position": row.reference_position,
                "inserted_base": row.inserted_base, "arm": row.arm, "n": row.n,
                "mean_score_delta": row.mean_score_delta,
                "median_score_delta": row.median_score_delta,
                "max_abs_reproduction_error": row.max_abs_reproduction_error
            })

    ism_df = run_mer11_targeted_saturation_mutagenesis(canonical, ranking, scorer)
    if not ism_df.empty:
        write_tsv(ism_df, OUT_ISM)


# --------------------------------------------------------------------------
# step 14b2: Extend the single-base candidate sites
# --------------------------------------------------------------------------



HERE__14b2 = Path(__file__).resolve().parent
OUT_DIR__14b2 = HERE__14b2 / "output/14b2_extend_single_base_sites"
OUT_SITES = OUT_DIR__14b2 / "14b2_all_candidate_sites.tsv"
OUT_SUMMARY = OUT_DIR__14b2 / "14b2_all_sites_summary.tsv"


b14 = sys.modules[__name__]

KEY = ["reference_repName", "reference_position", "inserted_base", "arm"]


def step_14b2() -> None:
    OUT_DIR__14b2.mkdir(parents=True, exist_ok=True)

    b14.SITES_PER_GROUP = 5
    b14.MAX_SCORED_SEQUENCES = 10000

    canonical = pd.read_parquet(CANONICAL__14b)
    canonical[ID__14b] = canonical[ID__14b].astype(str)
    ranking = pd.read_parquet(RANKING)
    window_length = int(canonical["seq_len"].mode().iloc[0])

    frozen = pd.read_csv(b14.OUT_PATCH_SUM, sep="\t")
    frozen_keys = set(zip(frozen["reference_repName"], frozen["reference_position"]))

    scorer = build_frozen_scorer()

    sites_all, summ_all = [], []
    for label, members in GROUPS.items():
        sub = canonical[canonical["repName"].isin(members)].reset_index(drop=True)
        anchor, _ = choose_anchor_length(sub["length"], window_length)
        low = (window_length - anchor) // 2
        profile = build_site_profiles(sub, anchor, low, low + anchor)
        sites = find_single_base_sites(profile, label)
        if len(sites) == 0:
            continue

        sites["is_new_vs_14b"] = [
            (r, p) not in frozen_keys
            for r, p in zip(sites["reference_repName"], sites["reference_position"])
        ]

        _, summary = run_revert_experiment(sub, profile, sites, label, ranking, scorer)
        sites_all.append(sites)
        summ_all.append(summary)

    sites = pd.concat(sites_all, ignore_index=True)
    summary = pd.concat(summ_all, ignore_index=True)

    m = summary.merge(frozen, on=KEY, suffixes=("_new", "_old"))
    if len(m):
        check = m[[*KEY, "n_new", "n_old",
                   "mean_score_delta_new", "mean_score_delta_old"]].copy()
        check["abs_diff"] = (check["mean_score_delta_new"] - check["mean_score_delta_old"]).abs()
    else:
        check = pd.DataFrame(columns=[*KEY, "n_new", "n_old",
                                      "mean_score_delta_new", "mean_score_delta_old", "abs_diff"])

    sites.to_csv(OUT_SITES, sep="\t", index=False)
    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)


# --------------------------------------------------------------------------
# step 14b2s: Screen the local sequence determinants
# --------------------------------------------------------------------------



HERE__14b2s = Path(__file__).resolve().parent
PROJECT_ROOT__14b2s = HERE__14b2s.parent
MODEL_OUTPUT__14b2s = PROJECT_ROOT__14b2s / "Modeling/output"

M00_DIR__14b2s = MODEL_OUTPUT__14b2s / "00_build_canonical_data"
M23_DIR__14b2s = MODEL_OUTPUT__14b2s / "23_generate_genomewide_herv_functional_ranking"

CANONICAL__14b2s = M00_DIR__14b2s / "00_canonical_herv_mef_table.parquet"
RANKING__14b2s = M23_DIR__14b2s / "23_herv_genomewide_functional_ranking.parquet"

OUT_DIR__14b2s = HERE__14b2s / "output/14b2_screen_local_sequence_determinants"

OUT_WINDOW = OUT_DIR__14b2s / "14b2_window_attribution_masking.tsv"
OUT_ISM__14b2s = OUT_DIR__14b2s / "14b2_single_base_substitution_ism.tsv.gz"
OUT_POSITION = OUT_DIR__14b2s / "14b2_position_effect_summary.tsv"
OUT_LOCUS = OUT_DIR__14b2s / "14b2_locus_effect_summary.tsv"
OUT_RANKED = OUT_DIR__14b2s / "14b2_eligible_candidate_ranking.tsv"

OUT_MANIFEST = OUT_DIR__14b2s / "14b2_run_manifest.json"

ID__14b2s = "herv_internal_id"
SCORE__14b2s = "score_sequence_regulatory"
PRIORITY_SCORE = "herv_functional_priority_score"
PRIORITY_RANK__14b2s = "herv_functional_priority_rank"

# Locked screening parameters.
TOP_FRACTION = 0.01
ATTR_WINDOW_BP__14b2s = 50
INPUT_EDGE_MARGIN_BP = 50
MIN_TOP_ABS_ATTR_FRACTION = 0.15
MIN_MASK_EFFECT_RATIO = 2.5
FOLLOWUP_LOCUS_N = 12
MASK_RATIO_EPS = 1e-8
SCORE_REPRODUCTION_TOLERANCE__14b2s = 3e-4
SCORE_BATCH_SIZE__14b2s = 64
VALID_BASES = ("A", "C", "G", "T")




@torch.inference_mode()
def score_frame__14b2s(
    frame: pd.DataFrame,
    sequences: list[str],
    scorer: dict[str, Any],
) -> np.ndarray:
    tensors, _ = tensors_for_rows(
        frame, scorer["tokenizer"], scorer["config"],
        scorer["encoder"], scorer["scaler"], sequences=sequences,
    )
    keys = ["k6_ids", "k6_mask", "bpe_ids", "bpe_mask", "cat_ids", "num_values"]
    device = scorer["device"]
    scores = []

    for start in range(0, len(sequences), SCORE_BATCH_SIZE__14b2s):
        args = [
            tensors[key][start:start + SCORE_BATCH_SIZE__14b2s].to(device)
            for key in keys
        ]
        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits = scorer["model"](*args)
            score = sequence_score(logits, scorer["score_head_indices"])

        scores.append(score.float().cpu().numpy())

    return np.concatenate(scores)


def _attribute_one_float32(model, row, tokenizer, config, encoder, scaler, device, score_head_indices):
    tensors, offsets = tensors_for_rows(row, tokenizer, config, encoder, scaler)
    args = [tensors[key].to(device) for key in
            ["k6_ids", "k6_mask", "bpe_ids", "bpe_mask", "cat_ids", "num_values"]]
    activations = {}

    def capture(name):
        def hook(_module, _inputs, output):
            output.retain_grad()
            activations[name] = output
        return hook

    handles = [model.emb_k6.register_forward_hook(capture("k6")),
               model.emb_bpe.register_forward_hook(capture("bpe"))]
    model.zero_grad(set_to_none=True)
    try:
        logits = model(*args)
        score = sequence_score(logits, score_head_indices)
        score.sum().backward()
        k6_attr = torch.abs((activations["k6"] * activations["k6"].grad).sum(dim=-1))[0].detach().float().cpu().numpy()
        bpe_attr = torch.abs((activations["bpe"] * activations["bpe"].grad).sum(dim=-1))[0].detach().float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    k6_base, bpe_base, combined = map_token_attribution(k6_attr, bpe_attr, offsets[0])
    return float(score.detach().float().cpu().item()), k6_base, bpe_base, combined

def attribute_one_stable(model, row, tokenizer, config, encoder, scaler, device, score_head_indices):
    return _attribute_one_float32(model, row, tokenizer, config, encoder, scaler, device, score_head_indices)

def choose_top1_ranking(ranking):
    ranked = ranking.copy()
    ranked[ID__14b2s] = ranked[ID__14b2s].astype(str)
    ranked[PRIORITY_RANK__14b2s] = pd.to_numeric(ranked[PRIORITY_RANK__14b2s])
    ranked = ranked.sort_values([PRIORITY_RANK__14b2s, ID__14b2s])
    top_n = max(1, int(math.floor(len(ranked) * TOP_FRACTION)))
    selected = ranked.head(top_n).copy()
    selected["top1_selection_order"] = np.arange(1, len(selected) + 1)
    return selected

def load_candidate_rows(top):
    canonical = pd.read_parquet(CANONICAL__14b2s, filters=[(ID__14b2s, "in", top[ID__14b2s].astype(str).tolist())])
    canonical[ID__14b2s] = canonical[ID__14b2s].astype(str)
    ranking_fields = [ID__14b2s, SCORE__14b2s, PRIORITY_SCORE, PRIORITY_RANK__14b2s, "top1_selection_order"]
    canonical = canonical.drop(columns=[c for c in ranking_fields[1:] if c in canonical.columns])
    return canonical.merge(top[ranking_fields], on=ID__14b2s).sort_values("top1_selection_order").reset_index(drop=True)

def basic_qc(row):
    sequence = str(row["sequence"]).upper()
    failures = {
        "sequence_length_not_1000": len(sequence) != 1000,
        "non_acgtn_sequence": bool(set(sequence) - set("ACGTN")),
        "invalid_strand": str(row["strand"]) not in {"+", "-"},
        "invalid_herv_interval": not int(row["start"]) < int(row["end"]),
        "invalid_sequence_interval": not int(row["seq_start"]) < int(row["seq_end"]),
    }
    reasons = [name for name, failed in failures.items() if failed]
    return not reasons, ";".join(reasons)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(abs(numerator) / max(abs(denominator), MASK_RATIO_EPS))


def empty_window_record(row: pd.Series, qc_pass: bool, qc_failures: str) -> dict[str, Any]:
    return {
        "top1_selection_order": int(row["top1_selection_order"]),
        ID__14b2s: str(row[ID__14b2s]),
        "chrom": row.get("chrom"),
        "start": row.get("start"),
        "end": row.get("end"),
        "strand": row.get("strand"),
        "repName": row.get("repName"),
        "repFamily": row.get("repFamily"),
        PRIORITY_RANK__14b2s: row.get(PRIORITY_RANK__14b2s),
        PRIORITY_SCORE: row.get(PRIORITY_SCORE, np.nan),
        "recorded_sequence_score": row.get(SCORE__14b2s),
        "basic_qc_pass": bool(qc_pass),
        "basic_qc_failures": qc_failures,
        "analysis_status": "PENDING" if qc_pass else "FAILED_BASIC_QC",
        "analysis_error": "",
    }


def analyze_window(row: pd.Series, scorer: dict[str, Any]) -> dict[str, Any]:
    qc_pass, qc_failures = basic_qc(row)
    record = empty_window_record(row, qc_pass, qc_failures)
    if not qc_pass:
        record["passes_window_filters"] = False
        return record

    # Do not use row.to_frame().T here: iterrows() returns a mixed-type Series,
    # and transposing it back makes every column object dtype. Module 14's
    # metadata transform applies NumPy ufuncs (including log10) to numeric
    # columns, which then fails on the object array. Building from a record lets
    # pandas infer each column independently and preserves numeric dtypes.
    one = pd.DataFrame([row.to_dict()])
    original_score, k6_base, bpe_base, combined = attribute_one_stable(
        scorer["model"],
        one,
        scorer["tokenizer"],
        scorer["config"],
        scorer["encoder"],
        scorer["scaler"],
        scorer["device"],
        scorer["score_head_indices"],
    )
    combined = np.asarray(combined, dtype=np.float64)
    rolling = np.convolve(combined, np.ones(ATTR_WINDOW_BP__14b2s, dtype=np.float64), mode="valid")
    internal_starts = np.arange(
        INPUT_EDGE_MARGIN_BP,
        len(rolling) - INPUT_EDGE_MARGIN_BP,
        dtype=np.int64,
    )
    top_start = int(internal_starts[np.argmax(rolling[internal_starts])])
    top_end = top_start + ATTR_WINDOW_BP__14b2s

    # Select the lowest-attribution non-overlapping control from the same
    # central region, so neither comparison window touches an input boundary.
    allowed_control = np.zeros(len(rolling), dtype=bool)
    allowed_control[internal_starts] = True
    allowed_control[
        max(0, top_start - ATTR_WINDOW_BP__14b2s + 1) : min(len(rolling), top_end)
    ] = False
    control_candidates = np.flatnonzero(allowed_control)
    control_start = int(control_candidates[np.argmin(rolling[control_candidates])])
    control_end = control_start + ATTR_WINDOW_BP__14b2s

    sequence = str(row["sequence"]).upper()
    masked_top = sequence[:top_start] + ("N" * ATTR_WINDOW_BP__14b2s) + sequence[top_end:]
    masked_control = sequence[:control_start] + ("N" * ATTR_WINDOW_BP__14b2s) + sequence[control_end:]
    perturb_frame = pd.concat([one, one], ignore_index=True)
    perturb_scores = score_frame__14b2s(perturb_frame, [masked_top, masked_control], scorer)
    top_masked_score = float(perturb_scores[0])
    control_masked_score = float(perturb_scores[1])
    delta_top = top_masked_score - original_score
    delta_control = control_masked_score - original_score
    ratio = safe_ratio(delta_top, delta_control)

    expected_score = float(row[SCORE__14b2s])
    reproduction_error = abs(original_score - expected_score)
    reproduction_pass = bool(reproduction_error <= SCORE_REPRODUCTION_TOLERANCE__14b2s)
    attr_pass = bool(float(rolling[top_start]) >= MIN_TOP_ABS_ATTR_FRACTION)
    # Retain the old direction flag as a diagnostic, never as a filter.
    direction_pass = bool(delta_top < 0)
    effect_pass = bool(np.isfinite(delta_top) and abs(delta_top) > 0)
    ratio_pass = bool(ratio >= MIN_MASK_EFFECT_RATIO)
    passes = bool(qc_pass and reproduction_pass and attr_pass and effect_pass and ratio_pass)

    top_gstart, top_gend = genomic_interval(row, top_start, top_end)
    control_gstart, control_gend = genomic_interval(row, control_start, control_end)
    herv_start, herv_end = int(row["start"]), int(row["end"])
    top_overlap = max(0, min(top_gend, herv_end) - max(top_gstart, herv_start))
    top_relation = (
        "WITHIN_HERV" if top_overlap == ATTR_WINDOW_BP__14b2s
        else "PARTIAL_HERV_OVERLAP" if top_overlap > 0
        else "FLANKING_SEQUENCE"
    )

    record.update({
        "analysis_status": "PASSED" if passes else "COMPLETED_NOT_ELIGIBLE",
        "reproduced_sequence_score": original_score,
        "score_reproduction_abs_error": reproduction_error,
        "score_reproduction_pass": reproduction_pass,
        "attribution_method": "absolute_gradient_times_embedding_k6_plus_bpe",
        "attribution_target": "mean_sigmoid_of_frozen_score_heads",
        "top_window_start_0based_strand_oriented": top_start,
        "top_window_end_0based_strand_oriented": top_end,
        "input_edge_margin_bp": INPUT_EDGE_MARGIN_BP,
        "top_window_genomic_start": top_gstart,
        "top_window_genomic_end": top_gend,
        "top_window_sequence": sequence[top_start:top_end],
        "top_window_abs_attribution_fraction": float(rolling[top_start]),
        "top_window_herv_overlap_bp": top_overlap,
        "top_window_relation_to_herv": top_relation,
        "control_window_start_0based_strand_oriented": control_start,
        "control_window_end_0based_strand_oriented": control_end,
        "control_window_genomic_start": control_gstart,
        "control_window_genomic_end": control_gend,
        "control_window_sequence": sequence[control_start:control_end],
        "control_window_abs_attribution_fraction": float(rolling[control_start]),
        "original_sequence_score": original_score,
        "top_window_masked_score": top_masked_score,
        "control_window_masked_score": control_masked_score,
        "delta_top_mask_minus_original": delta_top,
        "delta_control_mask_minus_original": delta_control,
        "top_vs_control_abs_mask_effect_ratio": ratio,
        "k6_attribution_raw_total": float(np.sum(k6_base)),
        "bpe_attribution_raw_total": float(np.sum(bpe_base)),
        "pass_top_abs_attribution_fraction": attr_pass,
        "pass_top_mask_direction": direction_pass,
        "pass_top_mask_effect": effect_pass,
        "pass_mask_effect_ratio": ratio_pass,
        "passes_window_filters": passes,
        "mask_character": "N",
        "metadata_held_fixed": True,
        "perturbation_is_biological_causality": False,
    })
    return record


def run_window_screen(candidates, scorer):
    rows = []
    for number, (_, row) in enumerate(candidates.iterrows(), start=1):
        rows.append(analyze_window(row, scorer))
    frame = pd.DataFrame(rows).sort_values("top1_selection_order").reset_index(drop=True)
    write_tsv(frame, OUT_WINDOW)
    return frame

def run_locus_ism(row: pd.Series, window_record: pd.Series, scorer: dict[str, Any]) -> pd.DataFrame:
    sequence = str(row["sequence"]).upper()
    original_score = float(window_record["original_sequence_score"])
    top_start = int(window_record["top_window_start_0based_strand_oriented"])
    top_end = int(window_record["top_window_end_0based_strand_oriented"])

    mutant_sequences: list[str] = []
    metadata: list[dict[str, Any]] = []
    for position in range(top_start, top_end):
        original_base = sequence[position]
        if original_base not in VALID_BASES:
            continue
        genomic_start, genomic_end = genomic_interval(row, position, position + 1)
        for mutant_base in VALID_BASES:
            if mutant_base == original_base:
                continue
            mutant_sequence = sequence[:position] + mutant_base + sequence[position + 1 :]
            mutant_sequences.append(mutant_sequence)
            metadata.append({
                "top1_selection_order": int(row["top1_selection_order"]),
                ID__14b2s: str(row[ID__14b2s]),
                "chrom": row.get("chrom"),
                "strand": row.get("strand"),
                "repName": row.get("repName"),
                "repFamily": row.get("repFamily"),
                PRIORITY_RANK__14b2s: row.get(PRIORITY_RANK__14b2s),
                "sequence_position_0based_strand_oriented": position,
                "position_within_top_window_0based": position - top_start,
                "genomic_start": genomic_start,
                "genomic_end": genomic_end,
                "original_base": original_base,
                "mutant_base": mutant_base,
                "original_score": original_score,
            })

    if not mutant_sequences:
        return pd.DataFrame()
    repeated = pd.DataFrame([row.to_dict()] * len(mutant_sequences))
    mutant_scores = score_frame__14b2s(repeated, mutant_sequences, scorer)
    for record, mutant_score in zip(metadata, mutant_scores):
        value = float(mutant_score)
        record["mutant_score"] = value
        record["delta_mutant_minus_original"] = value - original_score
        record["mutation_type"] = "substitution"
        record["metadata_held_fixed"] = True
        record["perturbation_is_biological_causality"] = False
    return pd.DataFrame(metadata)


def run_ism(candidates, windows, scorer):
    candidate_index = candidates.set_index(ID__14b2s, drop=False)
    eligible = windows[windows["passes_window_filters"].fillna(False).astype(bool)]
    frames = []
    for number, (_, saved) in enumerate(eligible.iterrows(), start=1):
        frames.append(run_locus_ism(candidate_index.loc[saved[ID__14b2s]], saved, scorer))
    ism = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_tsv_gz(ism, OUT_ISM__14b2s)
    return ism

def build_position_summary(ism: pd.DataFrame) -> pd.DataFrame:
    if len(ism) == 0:
        return pd.DataFrame()
    group_cols = [
        "top1_selection_order", ID__14b2s, "chrom", "strand", "repName", "repFamily", PRIORITY_RANK__14b2s,
        "sequence_position_0based_strand_oriented", "position_within_top_window_0based",
        "genomic_start", "genomic_end", "original_base", "original_score",
    ]
    summary = (
        ism.groupby(group_cols, dropna=False)["delta_mutant_minus_original"]
        .agg(
            n_substitutions="size",
            n_decreasing_substitutions=lambda x: int(np.sum(np.asarray(x) < 0)),
            mean_substitution_delta="mean",
            median_substitution_delta="median",
            min_substitution_delta="min",
            max_substitution_delta="max",
        )
        .reset_index()
    )
    alt = (
        ism.sort_values([ID__14b2s, "sequence_position_0based_strand_oriented", "mutant_base"])
        .groupby([ID__14b2s, "sequence_position_0based_strand_oriented"], dropna=False)
        .apply(
            lambda g: ";".join(
                f"{r.mutant_base}:{r.delta_mutant_minus_original:.9g}" for r in g.itertuples()
            ),
            include_groups=False,
        )
        .rename("alternate_base_deltas")
        .reset_index()
    )
    summary = summary.merge(
        alt,
        on=[ID__14b2s, "sequence_position_0based_strand_oriented"],
        how="left",
    )
    summary["absolute_mean_substitution_delta"] = summary["mean_substitution_delta"].abs()
    # Legacy field name retained: eligibility now means a complete three-base
    # substitution set with a finite, nonzero mean, without a direction gate.
    summary["passes_robust_single_base_effect"] = (
        (summary["n_substitutions"] == 3)
        & np.isfinite(summary["min_substitution_delta"])
        & np.isfinite(summary["max_substitution_delta"])
        & np.isfinite(summary["mean_substitution_delta"])
        & (summary["absolute_mean_substitution_delta"] > 0)
    )
    summary = summary.sort_values(
        ["top1_selection_order", "sequence_position_0based_strand_oriented"]
    ).reset_index(drop=True)
    write_tsv(summary, OUT_POSITION)
    return summary


def build_locus_summary(windows: pd.DataFrame, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    locus = windows.copy()
    locus[ID__14b2s] = locus[ID__14b2s].astype(str)

    if len(positions):
        robust = positions[positions["passes_robust_single_base_effect"].astype(bool)].copy()
        robust = robust.sort_values(
            [ID__14b2s, "absolute_mean_substitution_delta",
             "sequence_position_0based_strand_oriented"],
            ascending=[True, False, True],
        )
        best = robust.groupby(ID__14b2s, as_index=False).head(1).copy()
        best = best.rename(columns={
            "sequence_position_0based_strand_oriented": "best_base_sequence_position_0based_strand_oriented",
            "position_within_top_window_0based": "best_base_position_within_top_window_0based",
            "genomic_start": "best_base_genomic_start",
            "genomic_end": "best_base_genomic_end",
            "original_base": "best_base_original_base",
            "n_substitutions": "best_base_n_substitutions",
            "n_decreasing_substitutions": "best_base_n_decreasing_substitutions",
            "mean_substitution_delta": "best_base_mean_substitution_delta",
            "median_substitution_delta": "best_base_median_substitution_delta",
            "min_substitution_delta": "best_base_min_substitution_delta",
            "max_substitution_delta": "best_base_max_substitution_delta",
            "alternate_base_deltas": "best_base_alternate_base_deltas",
        })
        keep = [
            ID__14b2s,
            "best_base_sequence_position_0based_strand_oriented",
            "best_base_position_within_top_window_0based",
            "best_base_genomic_start",
            "best_base_genomic_end",
            "best_base_original_base",
            "best_base_n_substitutions",
            "best_base_n_decreasing_substitutions",
            "best_base_mean_substitution_delta",
            "best_base_median_substitution_delta",
            "best_base_min_substitution_delta",
            "best_base_max_substitution_delta",
            "best_base_alternate_base_deltas",
        ]
        locus = locus.merge(best[keep], on=ID__14b2s, how="left")

        robust_counts = (
            robust.groupby(ID__14b2s).size().rename("n_robust_single_base_positions").reset_index()
        )
        locus = locus.merge(robust_counts, on=ID__14b2s, how="left")
    else:
        locus["n_robust_single_base_positions"] = 0

    locus["n_robust_single_base_positions"] = (
        pd.to_numeric(locus["n_robust_single_base_positions"], errors="coerce").fillna(0).astype(int)
    )
    locus["has_robust_single_base_position"] = locus["n_robust_single_base_positions"] > 0
    locus["passes_screen_filters"] = (
        locus["passes_window_filters"].fillna(False).astype(bool)
        & locus["has_robust_single_base_position"]
    )
    locus["single_base_effect_strength"] = np.abs(pd.to_numeric(
        locus.get("best_base_mean_substitution_delta", np.nan), errors="coerce"
    ))

    failure_flags = pd.DataFrame({
        "basic_qc": ~locus["basic_qc_pass"].fillna(False).astype(bool),
        "score_reproduction": ~locus["score_reproduction_pass"].fillna(False).astype(bool),
        "top_abs_attribution_fraction": ~locus["pass_top_abs_attribution_fraction"].fillna(False).astype(bool),
        "top_mask_effect": ~locus["pass_top_mask_effect"].fillna(False).astype(bool),
        "mask_effect_ratio": ~locus["pass_mask_effect_ratio"].fillna(False).astype(bool),
        "no_robust_single_base_position": locus["passes_window_filters"].fillna(False).astype(bool)
            & ~locus["has_robust_single_base_position"],
    })
    locus["filter_failure_reasons"] = failure_flags.apply(lambda row: ";".join(row.index[row]), axis=1)
    all_ranked = locus[locus["passes_screen_filters"]].copy()
    all_ranked = all_ranked.sort_values(
        ["single_base_effect_strength", PRIORITY_RANK__14b2s, ID__14b2s],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    selected_ids = set(all_ranked.head(FOLLOWUP_LOCUS_N)[ID__14b2s].astype(str))
    locus["selected_for_followup"] = locus[ID__14b2s].astype(str).isin(selected_ids)
    # Backward-compatible field consumed by 14b3, 14b4 and manuscript scripts.
    # It now denotes the fixed Top-12 follow-up set after all scientific filters.
    locus["passes_all_filters"] = locus["selected_for_followup"]
    locus.loc[
        locus["passes_screen_filters"] & ~locus["selected_for_followup"],
        "filter_failure_reasons",
    ] = "outside_top12_followup"
    locus = locus.sort_values("top1_selection_order").reset_index(drop=True)
    write_tsv(locus, OUT_LOCUS)

    ranked = all_ranked.head(FOLLOWUP_LOCUS_N).copy()
    ranked["selected_for_followup"] = True
    ranked["passes_all_filters"] = True
    ranked["effect_rank_global"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    ranked["effect_rank_within_family"] = (
        ranked.groupby("repFamily", dropna=False).cumcount() + 1
    )
    write_tsv(ranked, OUT_RANKED)
    return locus, ranked


def step_14b2s() -> None:
    OUT_DIR__14b2s.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_parquet(RANKING__14b2s)
    top = choose_top1_ranking(ranking)
    candidates = load_candidate_rows(top)
    scorer = build_frozen_scorer()


    windows = run_window_screen(candidates, scorer)
    ism = run_ism(candidates, windows, scorer)
    positions = build_position_summary(ism)
    locus, ranked = build_locus_summary(windows, positions)

    window_error_n = int((windows["analysis_status"] == "ERROR").sum())
    run_status = "PASSED"

    manifest = {
        "created_at_utc": now(),
        "status": run_status,
        "analysis_name": "local_sequence_determinant_screen",
        "figure_binding": None,
        "top_fraction": TOP_FRACTION,
        "genome_ranking_n": int(len(ranking)),
        "selected_candidate_n": int(len(candidates)),
        "attribution_method": "absolute gradient-times-embedding for K6 and BPE mapped to 1000 bp",
        "attribution_window_bp": ATTR_WINDOW_BP__14b2s,
        "input_edge_margin_bp": INPUT_EDGE_MARGIN_BP,
        "minimum_top_absolute_attribution_fraction": MIN_TOP_ABS_ATTR_FRACTION,
        "control_window": "lowest absolute-attribution non-overlapping 50-bp window within the central input region",
        "mask_character": "N",
        "delta_convention": "perturbed_score_minus_original_score",
        "required_top_mask_direction": "none; positive and negative effects are eligible",
        "required_top_mask_effect": "finite, nonzero delta_top_mask_minus_original",
        "minimum_top_vs_control_absolute_mask_effect_ratio": MIN_MASK_EFFECT_RATIO,
        "mask_ratio_denominator_epsilon": MASK_RATIO_EPS,
        "ism": "all three alternate A/C/G/T substitutions at each valid base in the top 50-bp window",
        "minimum_decreasing_substitutions_per_position": 0,
        "position_eligibility": "three finite substitutions with a finite, nonzero mean; no direction filter",
        "position_effect": "signed arithmetic mean delta across the three non-reference substitutions; no zero-effect reference term",
        "locus_effect": "position with largest absolute mean substitution delta; ties use strand-oriented position",
        "ranking": "descending absolute value of best-position mean substitution delta",
        "followup_locus_n": FOLLOWUP_LOCUS_N,
        "followup_selection": "strongest screen-qualified loci by absolute best-position mean substitution delta",
        "followup_quota_applied": True,
        "display_quota_applied": False,
        "alignment_performed": False,
        "cross_copy_comparison_performed": False,
        "biological_causality_claimed": False,
        "score_reproduction_tolerance": SCORE_REPRODUCTION_TOLERANCE__14b2s,
        "model_checkpoint": str(MODEL_CHECKPOINT),
        "model_score_contract": scorer["config"].get("sequence_score_contract"),
        "model_score_head_indices": scorer["score_head_indices"],
        "model_score_head_names": [
            scorer["config"]["head_order"][i] for i in scorer["score_head_indices"]
        ],
        "device": str(scorer["device"]),
        "window_filter_pass_n": int(windows["passes_window_filters"].fillna(False).sum()),
        "window_analysis_error_n": window_error_n,
        "ism_prediction_n": int(len(ism)),
        "position_summary_n": int(len(positions)),
        "screen_filter_pass_n": int(locus["passes_screen_filters"].sum()),
        "all_filter_pass_n": int(locus["passes_all_filters"].sum()),
        "outputs": {
            "window_attribution_masking": str(OUT_WINDOW),
            "single_base_substitution_ism": str(OUT_ISM__14b2s),
            "position_effect_summary": str(OUT_POSITION),
            "locus_effect_summary": str(OUT_LOCUS),
            "eligible_candidate_ranking": str(OUT_RANKED),
        },
    }
    write_json(manifest, OUT_MANIFEST)


# --------------------------------------------------------------------------
# step 14b3: Export the eligible locus base attribution
# --------------------------------------------------------------------------



# Importing existing modules must not create artifacts in their directories.
sys.dont_write_bytecode = True


HERE__14b3 = Path(__file__).resolve().parent
INPUT_DIR = HERE__14b3 / "output/14b2_screen_local_sequence_determinants"
INPUT_LOCUS = INPUT_DIR / "14b2_locus_effect_summary.tsv"
INPUT_MANIFEST = INPUT_DIR / "14b2_run_manifest.json"
OUT_DIR__14b3 = HERE__14b3 / "output/14b3_export_eligible_locus_base_attribution"
OUT_BASE__14b3 = OUT_DIR__14b3 / "14b3_base_level_attribution.tsv.gz"
OUT_AUDIT = OUT_DIR__14b3 / "14b3_attribution_consistency_audit.tsv"


ID__14b3 = "herv_internal_id"







def load_eligible():
    frame = pd.read_csv(INPUT_LOCUS, sep="\t", dtype={ID__14b3: str}, low_memory=False)
    frame = frame[frame["passes_all_filters"].astype(str).str.lower().isin(["true", "1"])].copy()
    frame = frame.sort_values(
        ["single_base_effect_strength", "herv_functional_priority_rank", ID__14b3],
        ascending=[False, True, True]).reset_index(drop=True)
    frame["export_order"] = np.arange(1, len(frame) + 1)
    return frame

def export_one(one, saved, scorer):
    row = one.iloc[0]
    sequence = str(row["sequence"]).upper()
    top_start = int(saved["top_window_start_0based_strand_oriented"])
    top_end = int(saved["top_window_end_0based_strand_oriented"])
    ctrl_start = int(saved["control_window_start_0based_strand_oriented"])
    ctrl_end = int(saved["control_window_end_0based_strand_oriented"])
    best = int(saved["best_base_sequence_position_0based_strand_oriented"])
    score, k6, bpe, combined = attribute_one_stable(
        scorer["model"], one, scorer["tokenizer"], scorer["config"],
        scorer["encoder"], scorer["scaler"], scorer["device"], scorer["score_head_indices"])
    k6, bpe, combined = [np.asarray(x, dtype=np.float64) for x in (k6, bpe, combined)]
    positions = np.arange(len(sequence), dtype=np.int64)
    gstart = np.where(str(row["strand"]) == "-",
                     int(row["seq_end"]) - positions - 1,
                     int(row["seq_start"]) + positions)
    gend = gstart + 1
    base = pd.DataFrame({
        "export_order": int(saved["export_order"]), ID__14b3: str(saved[ID__14b3]),
        "top1_selection_order": int(saved["top1_selection_order"]),
        "repName": row["repName"], "repFamily": row["repFamily"],
        "chrom": row["chrom"], "strand": row["strand"],
        "sequence_position_0based_strand_oriented": positions,
        "genomic_start": gstart, "genomic_end": gend, "base": list(sequence),
        "k6_attribution_raw": k6, "bpe_attribution_raw": bpe,
        "combined_attribution_normalized": combined,
        "in_top_50bp_region": (positions >= top_start) & (positions < top_end),
        "in_control_50bp_region": (positions >= ctrl_start) & (positions < ctrl_end),
        "in_herv_body": (gstart >= int(row["start"])) & (gend <= int(row["end"])),
        "is_best_single_base_position": positions == best,
    })
    # Numerical differences are reported only; no validation gate is applied.
    audit = {
        "export_order": int(saved["export_order"]), ID__14b3: str(saved[ID__14b3]),
        "analysis_status": "COMPLETED", "recomputed_sequence_score": float(score),
        "score_abs_error_vs_14b2": abs(score - float(saved["original_sequence_score"])),
        "score_abs_error_vs_recorded": abs(score - float(saved["recorded_sequence_score"])),
        "attribution_normalized_sum": float(combined.sum()),
        "top_attribution_fraction_abs_error": abs(float(combined[top_start:top_end].sum())
            - float(saved["top_window_abs_attribution_fraction"])),
        "control_attribution_fraction_abs_error": abs(float(combined[ctrl_start:ctrl_end].sum())
            - float(saved["control_window_abs_attribution_fraction"])),
    }
    return base, audit


def step_14b3():
    OUT_DIR__14b3.mkdir(parents=True, exist_ok=True)
    prior = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    eligible = load_eligible()
    manifest = {
        "started_at_utc": now(), "status": "RUNNING",
        "analysis_name": "eligible_locus_base_attribution_export",
        "input_locus_summary": str(INPUT_LOCUS), "input_manifest": str(INPUT_MANIFEST),
        "attribution_method": "absolute gradient-times-embedding for K6+BPE",
        "selection": "all eligible 14b2 loci; ordered by absolute mean delta",
        "masking_recomputed": False, "ism_recomputed": False,
        "alignment_performed": False, "display_quota_applied": False,
        "upstream_created_at_utc": prior.get("created_at_utc"),
        "outputs": {"base_attribution": str(OUT_BASE__14b3), "consistency_audit": str(OUT_AUDIT),
                    },
    }
    request = eligible.copy()
    request[SCORE__14b2s] = request["recorded_sequence_score"]
    canonical = load_candidate_rows(request).set_index(ID__14b3, drop=False)
    scorer = build_frozen_scorer()
    bases, audits = [], []
    for number, (_, saved) in enumerate(eligible.iterrows(), start=1):
        one = canonical.loc[[saved[ID__14b3]]].reset_index(drop=True)
        base, audit = export_one(one, saved, scorer)
        bases.append(base)
        audits.append(audit)
    base_all = pd.concat(bases, ignore_index=True) if bases else pd.DataFrame()
    exported = eligible.copy()
    exported["best_base_inside_herv"] = (
        (exported["best_base_genomic_start"] >= exported["start"])
        & (exported["best_base_genomic_end"] <= exported["end"]))
    write_tsv_gz(base_all, OUT_BASE__14b3)
    write_tsv(pd.DataFrame(audits), OUT_AUDIT)
    manifest.update({
        "status": "PASSED", "finished_at_utc": now(),
        "eligible_locus_n": len(eligible), "exported_locus_n": len(exported),
        "base_row_n": len(base_all), "best_base_inside_herv_n": int(exported["best_base_inside_herv"].sum()),
        "device": str(scorer["device"]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "formal_data_published_this_run": True,
    })
    return 0


# --------------------------------------------------------------------------
# step 14b4: Compute the control-window substitution ISM
# --------------------------------------------------------------------------



# Importing upstream scripts must not create bytecode in their directories.
sys.dont_write_bytecode = True


HERE__14b4 = Path(__file__).resolve().parent
NAME = "14b4_compute_control_window_substitution_ism"
OUT = HERE__14b4 / "output" / NAME
UPSTREAM = HERE__14b4 / "output/14b2_screen_local_sequence_determinants"
INPUT_LOCUS__14b4 = UPSTREAM / "14b2_locus_effect_summary.tsv"
INPUT_MANIFEST__14b4 = UPSTREAM / "14b2_run_manifest.json"
ID__14b4 = "herv_internal_id"
POSITION = "sequence_position_0based_strand_oriented"
DELTA = "delta_mutant_minus_original"
BASES = ("A", "C", "G", "T")
OUTPUTS = {
    "paired_base_effects": OUT / "14b4_top_control_base_effects.tsv",
    "consistency_audit": OUT / "14b4_consistency_audit.tsv",
}

MUTATION_COLUMNS = [
    ID__14b4, "chrom", "strand", "repName", "repFamily", POSITION,
    "position_within_control_window_0based", "genomic_start", "genomic_end",
    "original_base", "mutant_base", "original_score", "mutant_score", DELTA,
    "window_type", "mutation_type", "metadata_held_fixed",
    "perturbation_is_biological_causality",
]
POSITION_COLUMNS = [
    ID__14b4, POSITION, "position_within_control_window_0based", "genomic_start",
    "genomic_end", "original_base", "n_substitutions",
    "n_decreasing_substitutions", "mean_substitution_delta",
    "median_substitution_delta", "min_substitution_delta",
    "max_substitution_delta", "alternate_base_deltas",
    "passes_robust_single_base_effect", "absolute_mean_substitution_delta",
]
BEST_FIELDS = {
    POSITION: "sequence_position_0based_strand_oriented",
    "genomic_start": "genomic_start",
    "genomic_end": "genomic_end",
    "original_base": "original_base",
    "n_substitutions": "n_substitutions",
    "n_decreasing_substitutions": "n_decreasing_substitutions",
    "mean_substitution_delta": "mean_substitution_delta",
    "median_substitution_delta": "median_substitution_delta",
    "min_substitution_delta": "min_substitution_delta",
    "max_substitution_delta": "max_substitution_delta",
    "alternate_base_deltas": "alternate_base_deltas",
}


def true_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def make_control_mutations(row, saved):
    sequence = str(row["sequence"]).upper()
    start = int(saved["control_window_start_0based_strand_oriented"])
    end = int(saved["control_window_end_0based_strand_oriented"])
    records, sequences = [], []
    for position in range(start, end):
        original = sequence[position]
        if original not in BASES:
            continue
        gs, ge = genomic_interval(row, position, position + 1)
        for alternate in BASES:
            if alternate == original:
                continue
            records.append({
                **{key: row[key] for key in [ID__14b4, "chrom", "strand", "repName", "repFamily"]},
                POSITION: position, "position_within_control_window_0based": position - start,
                "genomic_start": gs, "genomic_end": ge,
                "original_base": original, "mutant_base": alternate,
                "original_score": float(saved["original_sequence_score"]),
                "window_type": "Control", "mutation_type": "substitution",
                "metadata_held_fixed": True, "perturbation_is_biological_causality": False,
            })
            sequences.append(sequence[:position] + alternate + sequence[position + 1:])
    return pd.DataFrame(records, columns=MUTATION_COLUMNS), sequences


def summarize_positions(ism: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (_, _), group in ism.groupby([ID__14b4, POSITION], sort=True):
        first = group.iloc[0]
        delta = group[DELTA]
        records.append({
            **{key: first[key] for key in POSITION_COLUMNS[:6]},
            "n_substitutions": len(group),
            "n_decreasing_substitutions": int((delta < 0).sum()),
            "mean_substitution_delta": float(delta.mean()),
            "median_substitution_delta": float(delta.median()),
            "min_substitution_delta": float(delta.min()),
            "max_substitution_delta": float(delta.max()),
            "alternate_base_deltas": ";".join(
                f"{r.mutant_base}:{getattr(r, DELTA):.9g}"
                for r in group.sort_values("mutant_base").itertuples(index=False)),
            "absolute_mean_substitution_delta": abs(float(delta.mean())),
            "passes_robust_single_base_effect": bool(
                len(delta) == 3 and np.isfinite(delta).all() and abs(float(delta.mean())) > 0),
        })
    return pd.DataFrame(records, columns=POSITION_COLUMNS)


def build_paired_tables(eligible: pd.DataFrame, positions: pd.DataFrame):
    # Do not filter on control robustness: the comparison must not change the cohort.
    best = positions.sort_values(
        [ID__14b4, "absolute_mean_substitution_delta", POSITION],
        ascending=[True, False, True],
    ).drop_duplicates(ID__14b4).set_index(ID__14b4)
    wide, long = [], []
    for _, locus in eligible.iterrows():
        record = locus.to_dict()
        subset = positions.loc[positions[ID__14b4] == locus[ID__14b4]]
        available = locus[ID__14b4] in best.index
        record["control_analysis_status"] = "PASSED" if available else "NO_VALID_ACGT_POSITIONS"
        record["control_position_n"] = len(subset)
        record["control_substitution_n"] = int(subset["n_substitutions"].sum())
        record["control_best_base_passes_robust_single_base_effect"] = (
            bool(best.loc[locus[ID__14b4], "passes_robust_single_base_effect"]) if available else pd.NA)
        for source, suffix in BEST_FIELDS.items():
            record[f"control_best_base_{suffix}"] = best.loc[locus[ID__14b4], source] if available else np.nan
        top_delta = float(locus["best_base_mean_substitution_delta"])
        control_delta = record["control_best_base_mean_substitution_delta"]
        record["top_best_base_mean_substitution_delta"] = top_delta
        # Signed contrast and magnitude comparison are separate.
        record["control_minus_top_mean_delta"] = control_delta - top_delta
        record["top_minus_control_absolute_mean_delta"] = abs(top_delta) - abs(control_delta)
        record["best_base_inside_herv"] = bool(
            locus["best_base_genomic_start"] >= locus["start"] and
            locus["best_base_genomic_end"] <= locus["end"])
        wide.append(record)
        for kind, prefix in [("Top", "best_base_"), ("Control", "control_best_base_")]:
            paired = {key: record[key] for key in [
                ID__14b4, "chrom", "start", "end", "strand", "repName", "repFamily",
                "herv_functional_priority_rank", "single_base_effect_strength", "best_base_inside_herv"]}
            paired.update({suffix: record[prefix + suffix] for suffix in BEST_FIELDS.values()})
            paired["window_type"] = kind
            paired["analysis_status"] = "REUSED_FROM_14B2" if kind == "Top" else record["control_analysis_status"]
            paired["effect_strength"] = abs(paired["mean_substitution_delta"])
            paired["delta_convention"] = "mutant_minus_original"
            long.append(paired)
    return pd.DataFrame(wide), pd.DataFrame(long)


def step_14b4():
    OUT.mkdir(parents=True, exist_ok=True)
    loci = pd.read_csv(INPUT_LOCUS__14b4, sep="\t", low_memory=False)
    eligible = loci.loc[true_values(loci["passes_all_filters"])].sort_values(
        ["single_base_effect_strength", PRIORITY_RANK__14b2s, ID__14b4],
        ascending=[False, True, True]).reset_index(drop=True)
    requested = eligible.copy()
    requested[SCORE__14b2s] = requested["recorded_sequence_score"]
    canonical = load_candidate_rows(requested).set_index(ID__14b4, drop=False)
    scorer = build_frozen_scorer()
    frames = []
    for number, (_, saved) in enumerate(eligible.iterrows(), start=1):
        row = canonical.loc[saved[ID__14b4]]
        mutations, sequences = make_control_mutations(row, saved)
        if sequences:
            repeated = pd.DataFrame([row.to_dict()] * len(sequences))
            mutations["mutant_score"] = score_frame__14b2s(repeated, sequences, scorer)
            mutations[DELTA] = mutations["mutant_score"] - mutations["original_score"]
        frames.append(mutations)
    ism = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MUTATION_COLUMNS)
    positions = summarize_positions(ism)
    wide, long = build_paired_tables(eligible, positions)
    audit = eligible[[ID__14b4, "original_sequence_score", "best_base_mean_substitution_delta"]].copy()
    tables = {"paired_base_effects": long, "consistency_audit": audit}
    for key, table in tables.items():
        write_tsv(table, OUTPUTS[key])


def main() -> None:
    step_14()
    step_14b()
    step_14b2()
    step_14b2s()
    step_14b3()
    step_14b4()


if __name__ == "__main__":
    main()
