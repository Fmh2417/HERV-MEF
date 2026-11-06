#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import json
import shutil
import pandas as pd
import torch
import gc
import math
import joblib
import numpy as np
import torch.nn as nn
import time
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from datetime import (
    datetime,
    timezone,
)
from typing import Any

# --------------------------------------------------------------------------
# step 22: Finalize the development model registry
# --------------------------------------------------------------------------



M = Path(__file__).resolve().parent
OUT = M / "output/22_finalize_development_model_registry"
FRZ = OUT / "22_frozen"

M03 = M / "output/03_audit_dual/03_frozen"
M04 = M / "output/04_train_sequence_branch_ablations/04_checkpoints"
M10 = M / "output/10_eval_bio_refinement_B/10_frozen"
M17 = M / "output/17_audit_select_final_global_ranker"
M17F = M17 / "17_frozen"
M18 = M / "output/18_export_final_global_strict_oof"
M19 = M / "output/19_build_functional_evidence_candidate_pool"
M20 = M / "output/20_train_functional_evidence_reranker"
M21 = M / "output/21_select_and_freeze_functional_evidence_reranker"
M21F = M21 / "21_frozen"
M21_PRIMARY = M21 / "21_final_primary_model_manifest.json"


ROADMAP18 = [f"roadmap_{m}_{s}" for m in (
    "DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3")
    for s in ("sample_count", "max_overlap_bp", "total_overlap_bp")]
MPRA8 = ["mpra_overlap_count", "mpra_max_overlap_bp", "mpra_total_overlap_bp",
         "mpra_active_overlap_count", "mpra_active_max_overlap_bp",
         "mpra_active_total_overlap_bp", "is_mpra_overlap", "is_mpra_active_supported"]
CCRE6 = ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_ccre_ctcf_overlap",
         "is_ccre_pls_overlap", "ccre_els_overlap_count", "ccre_els_total_overlap_bp"]
SERVE6 = ["serve_gtex_tissue_count", "serve_gtex_total_overlap_bp",
          "is_serve_gtex_supported", "serve_gtex_annotation_count",
          "serve_noncoding_overlap_count", "serve_noncoding_total_overlap_bp"]
EVIDENCE38 = [*ROADMAP18, *MPRA8, *CCRE6, *SERVE6]

COPY_PAIRS = [
    (M21F / "21_frozen_final_global_ranker.pt",
     FRZ / "22_final_global_ranker.pt"),
    (M21F / "21_frozen_final_global_preprocessor.joblib",
     FRZ / "22_final_global_preprocessor.joblib"),
    (M21F / "21_frozen_primary_fer_component.pt",
     FRZ / "22_primary_fer_component.pt"),
    (M21F / "21_frozen_primary_fer_preprocessor.joblib",
     FRZ / "22_primary_fer_preprocessor.joblib"),
    (M21_PRIMARY,
     FRZ / "22_final_primary_manifest.json"),
    (M04 / "04_k6_meta.pt",
     FRZ / "22_k6_comparator.pt"),
    (M04 / "04_bpe_meta.pt",
     FRZ / "22_bpe_comparator.pt"),
]

ROUTES = [
    ("K6_META", "K6 + metadata comparator",
     "M04", M04 / "04_k6_meta.pt"),
    ("BPE_META", "BPE + metadata comparator",
     "M04", M04 / "04_bpe_meta.pt"),
    ("FULL_DUAL", "Full Dual basic reference",
     "M03", M03 / "03_frozen_dual_reference.pt"),
    ("M10_REPRESENTATION", "M10 final sequence repr",
     "M10", M10 / "10_frozen_final_representation_contract.json"),
    ("M17_FINAL_GLOBAL", "Final Global Ranker",
     "M17", M17F / "17_frozen_final_global_ranker.pt"),
    ("M21_FINAL_PRIMARY", "Final Global + All38 FER",
     "M21", M21_PRIMARY),
]

DEPS = [
    ("FULL_DUAL", "checkpoint",
     M03 / "03_frozen_dual_reference.pt"),
    ("K6_META", "checkpoint",
     M04 / "04_k6_meta.pt"),
    ("BPE_META", "checkpoint",
     M04 / "04_bpe_meta.pt"),
    ("M10_REPRESENTATION", "contract_json",
     M10 / "10_frozen_final_representation_contract.json"),
    ("M10_REPRESENTATION", "contract_joblib",
     M10 / "10_frozen_final_representation_contract.joblib"),
    ("M10_REPRESENTATION", "calibrator",
     M10 / "10_frozen_biological_calibrator.joblib"),
    ("M17_FINAL_GLOBAL", "checkpoint",
     M17F / "17_frozen_final_global_ranker.pt"),
    ("M17_FINAL_GLOBAL", "preprocessor",
     M17F / "17_frozen_final_global_preprocessor.joblib"),
    ("M18_OOF_BRIDGE", "train_oof_predictions",
     M18 / "18_train_oof_final_global_predictions.parquet"),
    ("M18_OOF_BRIDGE", "fold_registry",
     M18 / "18_fold_model_registry.tsv"),
    ("M20_FER_TRAINING", "checkpoint_registry",
     M20 / "20_checkpoint_registry.tsv"),
    ("M21_FINAL_PRIMARY", "final_global_checkpoint",
     M21F / "21_frozen_final_global_ranker.pt"),
    ("M21_FINAL_PRIMARY", "final_global_preprocessor",
     M21F / "21_frozen_final_global_preprocessor.joblib"),
    ("M21_FINAL_PRIMARY", "FER_checkpoint",
     M21F / "21_frozen_primary_fer_component.pt"),
    ("M21_FINAL_PRIMARY", "FER_preprocessor",
     M21F / "21_frozen_primary_fer_preprocessor.joblib"),
    ("M21_FINAL_PRIMARY", "final_manifest",
     M21_PRIMARY),
]


def W(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    elif path.suffix == ".tsv":
        obj.to_csv(tmp, sep="\t", index=False)
    else:
        obj.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def copy_identical(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def norm_topk(path):
    df = pd.read_csv(path, sep="\t")
    hits_col = next(c for c in ("hits", "Hits") if c in df.columns)
    fe_col = next(c for c in ("fe", "FE", "fold_enrichment") if c in df.columns)
    out = {}
    for fr, name in [(0.005, "top0_5"), (0.01, "top1"), (0.05, "top5")]:
        row = df.loc[(df["fraction"].astype(float) - fr).abs() <= 1e-12].iloc[0]
        out[f"{name}_hits"] = int(row[hits_col])
        out[f"{name}_fe"] = float(row[fe_col])
    return out


def norm_random(path):
    df = pd.read_csv(path, sep="\t")
    ap = "auprc" if "auprc" in df.columns else "AUPRC"
    au = "auroc" if "auroc" in df.columns else "AUROC"
    return {
        "auprc_mean": float(df[ap].mean()),
        "auprc_std": float(df[ap].std(ddof=1)),
        "auroc_mean": float(df[au].mean()),
        "auroc_std": float(df[au].std(ddof=1)),
    }


def load_primary():
    m = json.loads(M21_PRIMARY.read_text(encoding="utf-8"))
    primary = {
        "candidate_id": str(m["selected_candidate_id"]),
        "arm": str(m["selected_arm"]),
        "epoch": int(m["selected_epoch"]),
        "model_type": str(m.get("selected_model_type", "FER")),
        "global_candidate": m.get("final_global_candidate"),
        "a3_accepted": bool(m.get("A3_FER_accepted", False)),
        "a0_fallback": bool(m.get("A0_safe_fallback_selected", False)),
    }
    return primary


def write_self_contained_frozen_metadata(primary):
    ckpt = torch.load(
        M21F / "21_frozen_final_global_ranker.pt",
        map_location="cpu", weights_only=False)

    W({
        "status": "FROZEN_FINAL_GLOBAL",
        "source_candidate_id": ckpt.get("source_candidate_id", primary["global_candidate"]),
        "source_route": ckpt.get("source_route"),
        "selected_epoch": ckpt.get("selected_epoch"),
        "source_features": ckpt.get("source_features", []),
        "model_state_keys": list(ckpt.get("model_state_dict", {}).keys()),
        "test_used": False,
    }, FRZ / "22_final_global_manifest.json")

    W(pd.DataFrame({
        "feature_index": range(len(EVIDENCE38)),
        "feature": EVIDENCE38,
        "group": ["Roadmap18"] * 18 + ["MPRA8"] * 8 + ["cCRE6"] * 6 + ["SERVE_GTEx6"] * 6,
        "representation": "full_resolution",
    }), FRZ / "22_38d_manifest.tsv")

    W({
        "status": "FROZEN",
        "pool_name": "FinalGlobal_Top10pct",
        "fraction": 0.10,
        "size_rule": "max(1, int(round(N * fraction)))",
        "fraction_selection_performed": False,
        "test_used": False,
    }, FRZ / "22_pool_rule.json")

    W({
        "status": "FROZEN",
        "inside_candidate_order": [
            "FER score desc", "Final Global desc",
            "canonical_split_row_index asc", "herv_internal_id asc",
        ],
        "outside_candidate_order": "preserve Final Global order exactly",
        "numeric_score_mixing": False,
        "test_used": False,
    }, FRZ / "22_stitching_rule.json")


def step_22():
    OUT.mkdir(parents=True, exist_ok=True)
    FRZ.mkdir(parents=True, exist_ok=True)

    primary = load_primary()

    copied = 0
    for src, dst in COPY_PAIRS:
        if copy_identical(src, dst):
            copied += 1
    write_self_contained_frozen_metadata(primary)

    m17_top = norm_topk(M17 / "17_selected_validation_topk_metrics.tsv")
    m17_rand = norm_random(M17 / "17_selected_validation_random1to1_metrics.tsv")
    m21_top = norm_topk(M21 / "21_selected_validation_topk_metrics.tsv")
    m21_rand = norm_random(M21 / "21_selected_validation_random1to1_metrics.tsv")

    gold = pd.DataFrame([
        {
            "stage": "M17",
            "candidate_id": primary["global_candidate"],
            "role": "Final Global",
            **m17_top, **m17_rand,
        },
        {
            "stage": "M21",
            "candidate_id": primary["candidate_id"],
            "role": "Final Development Primary",
            "top0_5_hits": int(m21_top["top0_5_hits"]),
            "top0_5_fe": m21_top["top0_5_fe"],
            "top1_hits": int(m21_top["top1_hits"]),
            "top1_fe": m21_top["top1_fe"],
            "top5_hits": int(m21_top["top5_hits"]),
            "top5_fe": m21_top["top5_fe"],
            "auprc_mean": m21_rand["auprc_mean"],
            "auprc_std": m21_rand["auprc_std"],
            "auroc_mean": m21_rand["auroc_mean"],
            "auroc_std": m21_rand["auroc_std"],
        },
    ])
    W(gold, OUT / "22_development_gold_snapshot.tsv")

    deps = pd.DataFrame([
        {
            "route_or_stage": r,
            "component_role": role,
            "path": str(p),
            "exists": p.is_file(),
            "frozen_before_M22": True,
            "test_used": False,
        }
        for r, role, p in DEPS
    ])
    W(deps, OUT / "22_dependency_registry.tsv")


    W({
        "status": "FROZEN_FINAL_DEVELOPMENT_PACKAGE",
        "development_primary": {
            "route_id": "M21_FINAL_PRIMARY",
            "candidate_id": primary["candidate_id"],
            "arm": primary["arm"],
            "epoch": primary["epoch"],
            "model_type": primary["model_type"],
            "Final_Global": primary["global_candidate"],
            "A0_fallback_selected": primary["a0_fallback"],
            "A3_FER_accepted": primary["a3_accepted"],
            "validation_gold": {
                "top0_5_hits": int(m21_top["top0_5_hits"]),
                "top1_hits": int(m21_top["top1_hits"]),
                "top5_hits": int(m21_top["top5_hits"]),
                "auprc_mean": m21_rand["auprc_mean"],
                "auroc_mean": m21_rand["auroc_mean"],
            },
        },
        "frozen_files": sorted(str(p.name) for p in FRZ.iterdir()),
        "comparators": [r[0] for r in ROUTES],
        "governance": {
            "Global_modification_after_M17": False,
            "FER_modification_after_M21": False,
            "M22_selection": False,
            "historical_test_reuse": False,
            "next_scope": "external_or_orthogonal_only",
        },
        "training_performed": False,
        "inference_performed": False,
        "test_used": False,
    }, OUT / "22_final_development_manifest.json")


# --------------------------------------------------------------------------
# step 23: Generate the genome-wide functional ranking
# --------------------------------------------------------------------------



M__23 = Path(__file__).resolve().parent
M00 = M__23 / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M01T = {s: M__23 / f"output/01_build_dual_tensors/01_{s}_dual_{'tensors' if s != 'test' else 'inputs'}.pt"
        for s in ("train", "valid", "test")}
M07 = M__23 / "output/07_audit_partial_unfreeze/07_frozen/07_frozen_sequence_reference.pt"
M10J = M__23 / "output/10_eval_bio_refinement_B/10_frozen/10_frozen_biological_calibrator.joblib"
M10_CONTRACT = M__23 / "output/10_eval_bio_refinement_B/10_frozen/10_frozen_final_representation_contract.json"
M12V = M__23 / "output/12_export_bio_refinement_predictions/12_valid_full_train_representation_predictions.parquet"
M18V = M__23 / "output/18_export_final_global_strict_oof/18_valid_full_train_final_global_predictions.parquet"
M22F = M__23 / "output/22_finalize_development_model_registry/22_frozen"
OUT__23 = M__23 / "output/23_generate_genomewide_herv_functional_ranking"

ID, SPLIT, ROW = "herv_internal_id", "split", "canonical_global_row_index"
SPLIT_N = {"train": 336_217, "valid": 72_048, "test": 72_041}
N_TOTAL = 480_306


SEQ_BS, GCR_BS, FER_BS = 128, 8192, 8192
K6_LEN, BPE_LEN, META_DIM, META_HID = 996, 256, 16, 64
EPS, FER_RS = 1e-6, 0.15
FAMILIES = ("ERV1", "ERVK", "ERVL", "ERVL-MALR", "GYPSY")
CAND_FRAC, TOP1_FRAC = 0.10, 0.01
REPRO_TOL = 2e-6

ACTIVE_MARKS = ("DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3")
REPRESSIVE_MARKS = ("H3K27me3", "H3K9me3")
ROADMAP18__23 = [f"roadmap_{m}_{s}" for m in (*ACTIVE_MARKS, *REPRESSIVE_MARKS)
             for s in ("sample_count", "max_overlap_bp", "total_overlap_bp")]
MPRA8__23 = ["mpra_overlap_count", "mpra_max_overlap_bp", "mpra_total_overlap_bp",
         "mpra_active_overlap_count", "mpra_active_max_overlap_bp",
         "mpra_active_total_overlap_bp", "is_mpra_overlap", "is_mpra_active_supported"]
CCRE6__23 = ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_ccre_ctcf_overlap",
         "is_ccre_pls_overlap", "ccre_els_overlap_count", "ccre_els_total_overlap_bp"]
SERVE6__23 = ["serve_gtex_tissue_count", "serve_gtex_total_overlap_bp",
          "is_serve_gtex_supported", "serve_gtex_annotation_count",
          "serve_noncoding_overlap_count", "serve_noncoding_total_overlap_bp"]
RAW38 = [*ROADMAP18__23, *MPRA8__23, *CCRE6__23, *SERVE6__23]

ROADMAP9 = ["gctx_roadmap_active_mark_presence_count", "gctx_roadmap_repressive_mark_presence_count",
            "gctx_roadmap_active_sample_count_sum", "gctx_roadmap_repressive_sample_count_sum",
            "gctx_roadmap_active_max_overlap_bp_max", "gctx_roadmap_repressive_max_overlap_bp_max",
            "gctx_roadmap_active_total_overlap_bp_log1p", "gctx_roadmap_repressive_total_overlap_bp_log1p",
            "gctx_roadmap_active_minus_repressive_logbp"]
MPRA6 = ["gctx_mpra_is_overlap", "gctx_mpra_is_active_supported",
         "gctx_mpra_total_overlap_bp_log1p", "gctx_mpra_active_total_overlap_bp_log1p",
         "gctx_mpra_active_count_fraction", "gctx_mpra_active_bp_fraction"]
CCRE_SERVE9 = ["gctx_ccre_is_ctcf_overlap", "gctx_ccre_is_pls_overlap",
               "gctx_ccre_els_overlap_count_log1p", "gctx_ccre_els_total_overlap_bp_log1p",
               "gctx_serve_annotation_count_log1p", "gctx_serve_total_overlap_bp_log1p",
               "gctx_serve_noncoding_overlap_count_log1p", "gctx_serve_noncoding_total_overlap_bp_log1p",
               "gctx_serve_noncoding_bp_fraction"]
COMPACT24 = [*ROADMAP9, *MPRA6, *CCRE_SERVE9]

BASE23 = ["score_ccre_enhancer", "score_serve_gtex_support", "score_sequence_regulatory",
          "score_final_representation", "calibration_delta",
          "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
          "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
          "repeat_divergence", "length", "log_length", "gc_content", "n_content",
          "dist_to_tss", "log_abs_dist_to_tss", "is_ERV1", "is_ERVK", "is_ERVL",
          "is_ERVL_MALR", "is_GYPSY", "is_ccre_ca_overlap", "is_ccre_enhancer_like",
          "is_serve_gtex_supported", "serve_gtex_tissue_count"]
CATEG = ["repClass", "repFamily", "nearest_gene_type"]
GLOBAL47 = [*BASE23, *COMPACT24]; GLOBAL50 = [*GLOBAL47, *CATEG]
M00_WHITELIST = [ID, SPLIT, "chrom", "start", "end", "strand", "repName", "repClass",
                 "repFamily", "repeat_divergence", "length", "gc_content", "n_content",
                 "dist_to_tss", "nearest_gene_id", "nearest_gene_name", "nearest_gene_type",
                 "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY", *RAW38]


def W__23(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix == ".json":
        import json
        tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    elif path.suffix == ".tsv":
        obj.to_csv(tmp, sep="\t", index=False)
    else:
        obj.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def safe_div(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.divide(n, d, out=np.zeros_like(n), where=d > 0)


def rank_pct(v):
    return v.rank(method="average", pct=True).to_numpy(np.float64)


def to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


class PE(nn.Module):
    def __init__(self, d, mlen, drop):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        pe = torch.zeros(mlen, d)
        pos = torch.arange(0, mlen, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * (div if d % 2 == 0 else div[:-1]))
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class SeqModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.cat_features = list(cfg["cat_features"])
        self.emb_k6 = nn.Embedding(int(cfg["vocab_k6"]), d, padding_idx=0)
        self.cnn_k6 = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU())
        self.pos_k6 = PE(d, K6_LEN, float(cfg["dropout"]))
        self.trans_k6 = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d, nhead=int(cfg["nhead"]), dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]), activation="gelu", batch_first=True, norm_first=True),
            num_layers=int(cfg["num_layers"]))
        self.emb_bpe = nn.Embedding(int(cfg["vocab_bpe"]), d, padding_idx=0)
        self.cnn_bpe = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU())
        self.pos_bpe = PE(d, BPE_LEN, float(cfg["dropout"]))
        self.trans_bpe = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d, nhead=int(cfg["nhead"]), dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]), activation="gelu", batch_first=True, norm_first=True),
            num_layers=int(cfg["num_layers"]))
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(int(cfg["cat_cardinalities"][f]), META_DIM, padding_idx=0)
            for f in self.cat_features])
        self.metadata_mlp = nn.Sequential(
            nn.Linear(len(self.cat_features) * META_DIM + int(cfg["num_num_features"]), META_HID),
            nn.LayerNorm(META_HID), nn.GELU())
        self.bpe_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.meta_gate = nn.Sequential(nn.Linear(d + META_HID, d), nn.Sigmoid())
        self.project_meta = nn.Linear(META_HID, d)
        self.fuse_norm = nn.LayerNorm(d); self.fuse_dropout = nn.Dropout(float(cfg["dropout"]))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in range(3)
        ])

    def forward(self, k6, m6, bpe, mb, cat, num):
        f6 = self.emb_k6(k6)
        f6 = self.cnn_k6(f6.transpose(1, 2)).transpose(1, 2)
        f6 = self.trans_k6(self.pos_k6(f6), src_key_padding_mask=(m6 == 0))[:, 0, :]
        fb = self.emb_bpe(bpe)
        fb = self.cnn_bpe(fb.transpose(1, 2)).transpose(1, 2)
        fb = self.trans_bpe(self.pos_bpe(fb), src_key_padding_mask=(mb == 0))[:, 0, :]
        cm = [self.cat_embeddings[i](cat[:, i]) for i in range(len(self.cat_features))]
        fm = self.metadata_mlp(torch.cat(cm + [num], dim=1))
        gb = self.bpe_gate(torch.cat([f6, fb], dim=1))
        gm = self.meta_gate(torch.cat([f6, fm], dim=1))
        f = self.fuse_dropout(self.fuse_norm(f6 + gb * fb + gm * self.project_meta(fm)))
        return [head(f).squeeze(-1) for head in self.heads]


class SeqDS(Dataset):
    def __init__(self, b): self.b = b
    def __len__(self): return int(self.b["rows"])
    def __getitem__(self, i):
        return (self.b["input_ids_k6"][i].long(), self.b["mask_k6"][i].long(),
                self.b["input_ids_bpe"][i].long(), self.b["mask_bpe"][i].long(),
                self.b["cat_ids"][i].long(), self.b["num_values"][i].float())


class GCR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
                                  nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.20),
                                  nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class FER(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                  nn.Linear(64, 16), nn.LayerNorm(16), nn.GELU(),
                                  nn.Linear(16, 1))
    def forward(self, e, b):
        d = FER_RS * torch.tanh(self.net(e).squeeze(-1))
        return b + d, d


@torch.inference_mode()
def infer_sequence(ckpt, split, canonical_split, device):
    bundle = torch.load(M01T[split], map_location="cpu", weights_only=False)
    info = bundle["info_df"][[ID, SPLIT]].copy()
    info[ID] = info[ID].astype(str); info[SPLIT] = info[SPLIT].astype(str)
    bundle["info_df"] = info

    model = SeqModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True); model.eval()
    loader = DataLoader(SeqDS(bundle), batch_size=SEQ_BS, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=False)
    ccre, serve = [], []
    for i, batch in enumerate(loader, 1):
        m = [v.to(device, non_blocking=True) for v in batch]
        with torch.amp.autocast(device_type="cuda"):
            out = model(*m)
        ccre.append(torch.sigmoid(out[0]).float().cpu().numpy())
        serve.append(torch.sigmoid(out[1]).float().cpu().numpy())
    c = np.concatenate(ccre).astype(np.float32).astype(np.float64)
    s = np.concatenate(serve).astype(np.float32).astype(np.float64)
    seq = np.mean(np.column_stack([c, s]), axis=1).astype(np.float32).astype(np.float64)
    del model, loader, bundle; gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame({ID: canonical_split[ID].astype(str).to_numpy(), SPLIT: split,
                         "score_ccre_enhancer": c, "score_serve_gtex_support": s,
                         "score_sequence_regulatory": seq})


def apply_m10(canonical, representation, calibrator, divergence_mean, divergence_std):
    div = canonical["repeat_divergence"].to_numpy(float)
    seq = representation["score_sequence_regulatory"].to_numpy(float)
    mean, std = float(divergence_mean), float(divergence_std)
    fam = canonical["repFamily"].astype(str).str.upper().to_numpy()
    dz = (div - mean) / std
    ind = {n: (fam == n).astype(float) for n in FAMILIES}
    x = np.column_stack([to_logit(seq, EPS)] + [ind[n] for n in FAMILIES]
                        + [ind[n] * dz for n in FAMILIES])
    final = np.asarray(calibrator.predict_proba(x)[:, 1], np.float32).astype(np.float64)
    representation = representation.copy()
    representation["score_final_representation"] = final
    representation["calibration_delta"] = (representation["score_final_representation"]
                                            - representation["score_sequence_regulatory"])
    for col in ["score_ccre_enhancer", "score_serve_gtex_support",
                "score_sequence_regulatory", "score_final_representation"]:
        representation[f"rank_pct_{col}"] = np.nan
        for split in ("train", "valid", "test"):
            m = representation[SPLIT].astype(str).eq(split)
            representation.loc[m, f"rank_pct_{col}"] = rank_pct(
                representation.loc[m, col]).astype(np.float32).astype(np.float64)
    return representation


# ---------- Compact24 + Final Global ----------

def derive_compact(raw):
    out = pd.DataFrame(index=raw.index)
    A = lambda sfx: raw[[f"roadmap_{m}_{sfx}" for m in ACTIVE_MARKS]].to_numpy(float)
    R = lambda sfx: raw[[f"roadmap_{m}_{sfx}" for m in REPRESSIVE_MARKS]].to_numpy(float)
    AS, RS, AM, RM, AT, RT = (A("sample_count"), R("sample_count"),
                               A("max_overlap_bp"), R("max_overlap_bp"),
                               A("total_overlap_bp"), R("total_overlap_bp"))
    at, rt = AT.sum(1), RT.sum(1)
    vals = [(AS > 0).sum(1), (RS > 0).sum(1), AS.sum(1), RS.sum(1), AM.max(1), RM.max(1),
            np.log1p(at), np.log1p(rt), np.log1p(at) - np.log1p(rt),
            raw["is_mpra_overlap"].to_numpy(float), raw["is_mpra_active_supported"].to_numpy(float),
            np.log1p(raw["mpra_total_overlap_bp"].to_numpy(float)),
            np.log1p(raw["mpra_active_total_overlap_bp"].to_numpy(float)),
            safe_div(raw["mpra_active_overlap_count"], raw["mpra_overlap_count"]),
            safe_div(raw["mpra_active_total_overlap_bp"], raw["mpra_total_overlap_bp"]),
            raw["is_ccre_ctcf_overlap"].to_numpy(float), raw["is_ccre_pls_overlap"].to_numpy(float),
            np.log1p(raw["ccre_els_overlap_count"].to_numpy(float)),
            np.log1p(np.maximum(raw["ccre_els_total_overlap_bp"].to_numpy(float), 0)),
            np.log1p(raw["serve_gtex_annotation_count"].to_numpy(float)),
            np.log1p(raw["serve_gtex_total_overlap_bp"].to_numpy(float)),
            np.log1p(raw["serve_noncoding_overlap_count"].to_numpy(float)),
            np.log1p(raw["serve_noncoding_total_overlap_bp"].to_numpy(float)),
            safe_div(raw["serve_noncoding_total_overlap_bp"], raw["serve_gtex_total_overlap_bp"])]
    for n, v in zip(COMPACT24, vals): out[n] = v
    return out


def build_global_source(canonical, representation, compact):
    src = pd.DataFrame(index=canonical.index)
    for col in ["score_ccre_enhancer", "score_serve_gtex_support", "score_sequence_regulatory",
                "score_final_representation", "calibration_delta",
                "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
                "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation"]:
        src[col] = representation[col].to_numpy()
    src["repeat_divergence"] = canonical["repeat_divergence"].to_numpy(float)
    src["length"] = canonical["length"].to_numpy(float)
    src["log_length"] = np.log1p(canonical["length"].to_numpy(float))
    src["gc_content"] = canonical["gc_content"].to_numpy(float)
    src["n_content"] = canonical["n_content"].to_numpy(float)
    src["dist_to_tss"] = canonical["dist_to_tss"].to_numpy(float)
    src["log_abs_dist_to_tss"] = np.log1p(np.abs(canonical["dist_to_tss"].to_numpy(float)))
    for n in FAMILIES:
        src[f"is_{n.replace('-', '_')}"] = canonical[f"is_{n.replace('-', '_')}"].to_numpy(float)
    for c in ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_serve_gtex_supported", "serve_gtex_tissue_count"]:
        src[c] = canonical[c].to_numpy(float)
    for c in COMPACT24: src[c] = compact[c].to_numpy(float)
    for c in CATEG: src[c] = canonical[c].astype(str).to_numpy()
    return src


def stitch(canonical, global_scores, cand_idx, fer_scores, global_order):
    cids = canonical.iloc[cand_idx][ID].astype(str).to_numpy()
    crows = canonical.iloc[cand_idx][ROW].to_numpy(np.int64)
    cglob = global_scores[cand_idx]
    local = np.lexsort((cids, crows, -cglob, -np.asarray(fer_scores, float))).astype(np.int64)
    cblock = cand_idx[local]
    mask = np.zeros(N_TOTAL, dtype=bool); mask[cand_idx] = True
    nblock = global_order[~mask[global_order]]
    order = np.concatenate([cblock, nblock]).astype(np.int64)
    return order


def step_23():
    OUT__23.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    canonical = pd.read_parquet(M00, columns=M00_WHITELIST)
    canonical[ID] = canonical[ID].astype(str); canonical[SPLIT] = canonical[SPLIT].astype(str)
    canonical[ROW] = np.arange(N_TOTAL, dtype=np.int64)
    for c in ["repeat_divergence", "length", "gc_content", "n_content", "dist_to_tss", *RAW38]:
        canonical[c] = pd.to_numeric(canonical[c], errors="raise")

    cache = OUT__23 / "23_m07_sequence_representation_cache.parquet"
    if cache.is_file():
        representation = pd.read_parquet(cache)
    else:
        ckpt = torch.load(M07, map_location="cpu", weights_only=False)
        parts = []
        for split in ("train", "valid", "test"):
            cs = canonical.loc[canonical[SPLIT] == split].reset_index(drop=True)
            parts.append(infer_sequence(ckpt, split, cs, device))
        representation = pd.concat(parts, ignore_index=True)
        representation = canonical[[ID, SPLIT]].merge(representation, on=[ID, SPLIT],
                                                       how="left", validate="one_to_one")
        W__23(representation, cache)

    m10 = joblib.load(M10J)
    contract10 = json.loads(M10_CONTRACT.read_text(encoding="utf-8"))
    representation = apply_m10(
        canonical, representation, m10,
        contract10.get("divergence_mean", 0.0),
        contract10.get("divergence_std", 1.0),
    )

    ref12 = pd.read_parquet(M12V, columns=[ID, "score_final_representation"])
    valid_mask = canonical[SPLIT].eq("valid").to_numpy()
    aligned = pd.DataFrame({ID: canonical.loc[valid_mask, ID].astype(str)}).merge(
        ref12, on=ID, how="left", validate="one_to_one")
    delta12 = float(np.max(np.abs(
        representation.loc[valid_mask, "score_final_representation"].to_numpy()
        - aligned["score_final_representation"].to_numpy())))

    compact = derive_compact(canonical[RAW38])
    src = build_global_source(canonical, representation, compact)

    gckpt = torch.load(M22F / "22_final_global_ranker.pt", map_location="cpu", weights_only=False)
    gpre = joblib.load(M22F / "22_final_global_preprocessor.joblib")
    gx = np.concatenate([
        gpre["scaler"].transform(src[GLOBAL47].to_numpy(float)).astype(np.float32),
        gpre["encoder"].transform(src[CATEG].astype(str)).astype(np.float32)], axis=1)
    gm = GCR(gx.shape[1]).to(device)
    gm.load_state_dict(gckpt["model_state_dict"], strict=True); gm.eval()
    with torch.inference_mode():
        gs = np.concatenate([
            torch.sigmoid(gm(torch.from_numpy(gx[s:s+GCR_BS]).to(device))).float().cpu().numpy()
            for s in range(0, N_TOTAL, GCR_BS)]).astype(np.float64)
    del gm; gc.collect(); torch.cuda.empty_cache()

    ref18 = pd.read_parquet(M18V, columns=[ID, "score_final_global"])
    aligned18 = pd.DataFrame({ID: canonical.loc[valid_mask, ID].astype(str)}).merge(
        ref18, on=ID, how="left", validate="one_to_one")
    delta18 = float(np.max(np.abs(gs[valid_mask] - aligned18["score_final_global"].to_numpy())))

    ids = canonical[ID].to_numpy(); rows = canonical[ROW].to_numpy(np.int64)
    gorder = np.lexsort((ids, rows, -gs)).astype(np.int64)
    grank = np.empty(N_TOTAL, dtype=np.int64); grank[gorder] = np.arange(1, N_TOTAL + 1)
    cand_k = max(1, int(round(N_TOTAL * CAND_FRAC)))
    cand_idx = gorder[:cand_k].copy()

    fckpt = torch.load(M22F / "22_primary_fer_component.pt", map_location="cpu", weights_only=False)
    fpre = joblib.load(M22F / "22_primary_fer_preprocessor.joblib")
    raw = canonical.iloc[cand_idx][RAW38].to_numpy(float)
    fx = fpre["scaler"].transform(np.log1p(np.maximum(raw, 0))).astype(np.float32)
    base = to_logit(gs[cand_idx])
    fm = FER(len(RAW38)).to(device)
    fm.load_state_dict(fckpt["model_state_dict"], strict=True); fm.eval()
    with torch.inference_mode():
        sp, dp = [], []
        for s in range(0, len(fx), FER_BS):
            bx = torch.from_numpy(fx[s:s+FER_BS]).to(device)
            bb = torch.from_numpy(base[s:s+FER_BS]).to(device)
            lg, d = fm(bx, bb)
            sp.append(torch.sigmoid(lg).float().cpu().numpy())
            dp.append(d.float().cpu().numpy())
    fs = np.concatenate(sp).astype(np.float64)
    fd = np.concatenate(dp).astype(np.float64)
    del fm; gc.collect(); torch.cuda.empty_cache()

    final_order = stitch(canonical, gs, cand_idx, fs, gorder)
    frank = np.empty(N_TOTAL, dtype=np.int64); frank[final_order] = np.arange(1, N_TOTAL + 1)

    is_cand = np.zeros(N_TOTAL, dtype=bool); is_cand[cand_idx] = True
    fs_full = np.full(N_TOTAL, np.nan); fd_full = np.full(N_TOTAL, np.nan)
    fs_full[cand_idx] = fs; fd_full[cand_idx] = fd

    atlas = pd.DataFrame({
        ID: canonical[ID].to_numpy(), SPLIT: canonical[SPLIT].to_numpy(),
        "chrom": canonical["chrom"].astype(str), "start": canonical["start"].to_numpy(np.int64),
        "end": canonical["end"].to_numpy(np.int64), "strand": canonical["strand"].astype(str),
        "repName": canonical["repName"].astype(str), "repClass": canonical["repClass"].astype(str),
        "repFamily": canonical["repFamily"].astype(str),
        "repeat_divergence": canonical["repeat_divergence"].to_numpy(float),
        "length": canonical["length"].to_numpy(float),
        "gc_content": canonical["gc_content"].to_numpy(float),
        "n_content": canonical["n_content"].to_numpy(float),
        "dist_to_tss": canonical["dist_to_tss"].to_numpy(float),
        "nearest_gene_id": canonical["nearest_gene_id"].astype(str),
        "nearest_gene_name": canonical["nearest_gene_name"].astype(str),
        "nearest_gene_type": canonical["nearest_gene_type"].astype(str),
        ROW: canonical[ROW].to_numpy(np.int64),
        "representation_prediction_origin": "uniform_frozen_full_train_M07_M10_deployment",
        "score_ccre_enhancer": representation["score_ccre_enhancer"].to_numpy(float),
        "score_serve_gtex_support": representation["score_serve_gtex_support"].to_numpy(float),
        "score_sequence_regulatory": representation["score_sequence_regulatory"].to_numpy(float),
        "score_final_representation": representation["score_final_representation"].to_numpy(float),
        "calibration_delta": representation["calibration_delta"].to_numpy(float),
        "rank_pct_score_ccre_enhancer": representation["rank_pct_score_ccre_enhancer"].to_numpy(float),
        "rank_pct_score_serve_gtex_support": representation["rank_pct_score_serve_gtex_support"].to_numpy(float),
        "rank_pct_score_sequence_regulatory": representation["rank_pct_score_sequence_regulatory"].to_numpy(float),
        "rank_pct_score_final_representation": representation["rank_pct_score_final_representation"].to_numpy(float),
        "score_final_global": gs, "global_rank": grank,
        "global_rank_pct": (N_TOTAL - grank + 1).astype(float) / N_TOTAL,
        "is_genomewide_top10_candidate": is_cand,
        "score_fer_candidate": fs_full, "delta_logit_candidate": fd_full,
        "final_ranking_block": np.where(is_cand, "candidate_top10", "non_candidate"),
        "herv_functional_priority_rank": frank,
        "herv_functional_priority_score": (N_TOTAL - frank + 1).astype(float) / N_TOTAL,
    })
    for c in RAW38: atlas[c] = canonical[c].to_numpy()
    for c in COMPACT24: atlas[c] = compact[c].to_numpy(float)

    W__23(atlas, OUT__23 / "23_herv_genomewide_functional_ranking.parquet")
    top1_k = int(math.floor(N_TOTAL * TOP1_FRAC))

    W__23({"status": "PASSED", "N": N_TOTAL,
       "Global": "R4_ALL_COMPACT_bs64_epoch_24", "FER": "A3_ALL38_PRIMARY_epoch_01",
       "candidate_K": cand_k, "top1_K": top1_k,
       "M12_repro_delta": delta12, "M18_repro_delta": delta18,
       "priority_score": "(N - rank + 1) / N",
       "training": False, "selection": False, "test_used": False},
      OUT__23 / "23_genomewide_ranking_manifest.json")


# --------------------------------------------------------------------------
# step 24: Project the frozen ranking to tumor epigenomes
# --------------------------------------------------------------------------



# =============================================================================
# 1. Fixed paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
MODELING = (ROOT / "Modeling").resolve()
DP = (ROOT / "Data_Preprocessing").resolve()

SCRIPT = "24_project_frozen_herv_ranking_to_tumor_epigenomes"
VERSION = "round2_v1"

M23_DIR = MODELING / "output/23_generate_genomewide_herv_functional_ranking"
M23_ATLAS = M23_DIR / "23_herv_genomewide_functional_ranking.parquet"

DP05_DIR = DP / "output/05_separated_roadmap_features"
TUMOR_LONG = DP05_DIR / "herv_tumor_cell_line_roadmap_evidence_long.parquet"

OUT__24 = MODELING / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes"
OUT_ALL = OUT__24 / "24_tumor_projection_all_herv.parquet"
OUT_CANDIDATES = OUT__24 / "24_potential_reactivation_candidates.tsv"
OUT_ENRICH = OUT__24 / "24_tumor_support_enrichment.tsv"

OUT_MANIFEST = OUT__24 / "24_tumor_projection_manifest.json"


# =============================================================================
# 2. Frozen identities / biological contracts
# =============================================================================

ID__24 = "herv_internal_id"

EXPECTED_HERV_N = 480_306
EXPECTED_GLOBAL = "R4_ALL_COMPACT_bs64_epoch_24"
EXPECTED_PRIMARY = "A3_ALL38_PRIMARY_epoch_01"

TUMOR_EIDS = ["E114", "E115", "E117", "E118", "E123"]
CELL_LINE_NAMES = {
    "E114": "A549 Lung Carcinoma",
    "E115": "Dnd41 T-cell Leukemia",
    "E117": "HeLa-S3 Cervical Carcinoma",
    "E118": "HepG2 Hepatocellular Carcinoma",
    "E123": "K562 Leukemia",
}

ACTIVE_MARKS__24 = ["H3K27ac", "H3K4me1", "H3K4me3"]
REPRESSIVE_MARKS__24 = ["H3K27me3", "H3K9me3"]
TUMOR_MARKS = [*ACTIVE_MARKS__24, *REPRESSIVE_MARKS__24]

# Gold-rank style fractions are descriptive only here.
TOP_FRACTIONS = [0.005, 0.01, 0.05]
TOP1_K = int(math.floor(EXPECTED_HERV_N * 0.01))

M23_READ_COLUMNS = [
    ID__24,
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repClass",
    "repFamily",
    "repeat_divergence",
    "length",
    "dist_to_tss",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "split",
    "score_final_global",
    "global_rank",
    "is_genomewide_top10_candidate",
    "herv_functional_priority_rank",
    "herv_functional_priority_score",
]

TUMOR_LONG_COLUMNS = [
    ID__24,
    "eid",
    "sample_name",
    "mark",
    "overlap_count",
    "max_overlap_bp",
    "total_overlap_bp",
    "has_overlap",
]


# =============================================================================
# 3. Generic utilities
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output() -> None:
    if OUT__24.exists():
        shutil.rmtree(OUT__24)
    OUT__24.mkdir(parents=True, exist_ok=True)


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


# =============================================================================
# 4. Load M23 and tumor evidence
# =============================================================================

def load_m23_atlas() -> pd.DataFrame:
    frame = pd.read_parquet(M23_ATLAS, columns=M23_READ_COLUMNS)
    frame[ID__24] = frame[ID__24].astype(str)
    return frame


def load_tumor_long(
    atlas_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(TUMOR_LONG, columns=TUMOR_LONG_COLUMNS)

    frame = frame.copy()
    frame[ID__24] = frame[ID__24].astype(str)
    frame["eid"] = frame["eid"].astype(str)
    frame["mark"] = frame["mark"].astype(str)

    for col in [
        "overlap_count",
        "max_overlap_bp",
        "total_overlap_bp",
        "has_overlap",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    original_rows = len(frame)
    original_unique_hervs = frame[ID__24].nunique()
    in_model = frame[ID__24].isin(atlas_ids)
    excluded = frame.loc[~in_model]
    frame = frame.loc[in_model].reset_index(drop=True)

    audit = {
        "positive_rows_before_model_universe_filter": original_rows,
        "unique_HERVs_before_model_universe_filter": original_unique_hervs,
        "positive_rows_in_M23_model_universe": len(frame),
        "unique_HERVs_in_M23_model_universe": int(frame[ID__24].nunique()),
        "excluded_positive_rows_outside_M23_model_universe": len(excluded),
        "excluded_unique_HERVs_outside_M23_model_universe": int(
            excluded[ID__24].nunique()
        ),
        "excluded_IDs": sorted(excluded[ID__24].unique().tolist()),
    }
    return frame, audit


# =============================================================================
# 6. Build full tumor projection
# =============================================================================

def flat_has_col(eid: str, mark: str) -> str:
    return f"tumor_{eid}_{mark}_has"


def flat_bp_col(eid: str, mark: str) -> str:
    return f"tumor_{eid}_{mark}_total_overlap_bp"


def build_projection(
    atlas: pd.DataFrame,
    tumor: pd.DataFrame,
) -> pd.DataFrame:
    ids = pd.Index(atlas[ID__24].astype(str), name=ID__24)

    has = tumor.pivot(
        index=ID__24,
        columns=["eid", "mark"],
        values="has_overlap",
    )
    bp = tumor.pivot(
        index=ID__24,
        columns=["eid", "mark"],
        values="total_overlap_bp",
    )

    projection = atlas.copy()

    for eid in TUMOR_EIDS:
        for mark in TUMOR_MARKS:
            key = (eid, mark)
            has_series = (
                has[key] if key in has.columns else pd.Series(dtype=float)
            )
            bp_series = (
                bp[key] if key in bp.columns else pd.Series(dtype=float)
            )

            projection[flat_has_col(eid, mark)] = (
                ids.to_series(index=ids)
                .map(has_series)
                .fillna(0)
                .astype(np.int8)
                .to_numpy()
            )
            projection[flat_bp_col(eid, mark)] = (
                ids.to_series(index=ids)
                .map(bp_series)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )

    for eid in TUMOR_EIDS:
        active_cols = [flat_has_col(eid, mark) for mark in ACTIVE_MARKS__24]
        rep_cols = [flat_has_col(eid, mark) for mark in REPRESSIVE_MARKS__24]
        active_bp_cols = [flat_bp_col(eid, mark) for mark in ACTIVE_MARKS__24]
        rep_bp_cols = [flat_bp_col(eid, mark) for mark in REPRESSIVE_MARKS__24]

        projection[f"tumor_{eid}_active_mark_count"] = (
            projection[active_cols].sum(axis=1).astype(np.int8)
        )
        projection[f"tumor_{eid}_repressive_mark_count"] = (
            projection[rep_cols].sum(axis=1).astype(np.int8)
        )
        projection[f"tumor_{eid}_active_total_overlap_bp"] = (
            projection[active_bp_cols].sum(axis=1).astype(np.int64)
        )
        projection[f"tumor_{eid}_repressive_total_overlap_bp"] = (
            projection[rep_bp_cols].sum(axis=1).astype(np.int64)
        )
        projection[f"tumor_{eid}_any_active_support"] = (
            projection[f"tumor_{eid}_active_mark_count"] > 0
        )
        projection[f"tumor_{eid}_any_repressive_support"] = (
            projection[f"tumor_{eid}_repressive_mark_count"] > 0
        )
        projection[f"tumor_{eid}_active_dominant"] = (
            projection[f"tumor_{eid}_active_mark_count"]
            > projection[f"tumor_{eid}_repressive_mark_count"]
        )

    active_any_cols = [
        f"tumor_{eid}_any_active_support" for eid in TUMOR_EIDS
    ]
    active_dom_cols = [
        f"tumor_{eid}_active_dominant" for eid in TUMOR_EIDS
    ]
    rep_any_cols = [
        f"tumor_{eid}_any_repressive_support" for eid in TUMOR_EIDS
    ]
    active_count_cols = [
        f"tumor_{eid}_active_mark_count" for eid in TUMOR_EIDS
    ]
    rep_count_cols = [
        f"tumor_{eid}_repressive_mark_count" for eid in TUMOR_EIDS
    ]

    projection["tumor_active_support_breadth"] = (
        projection[active_any_cols].sum(axis=1).astype(np.int8)
    )
    projection["tumor_active_dominant_breadth"] = (
        projection[active_dom_cols].sum(axis=1).astype(np.int8)
    )
    projection["tumor_repressive_support_breadth"] = (
        projection[rep_any_cols].sum(axis=1).astype(np.int8)
    )
    projection["tumor_total_active_mark_hits"] = (
        projection[active_count_cols].sum(axis=1).astype(np.int8)
    )
    projection["tumor_total_repressive_mark_hits"] = (
        projection[rep_count_cols].sum(axis=1).astype(np.int8)
    )
    projection["tumor_epigenomic_support_score"] = (
        projection["tumor_total_active_mark_hits"].astype(np.float64) / 15.0
    )
    projection["tumor_repressive_context_score"] = (
        projection["tumor_total_repressive_mark_hits"].astype(np.float64) / 10.0
    )

    projection["m23_top1pct"] = (
        projection["herv_functional_priority_rank"].astype(np.int64) <= TOP1_K
    )

    projection["potential_reactivation_candidate"] = (
        projection["m23_top1pct"]
        & (projection["tumor_active_support_breadth"] >= 1)
    )

    support_class = np.full(
        len(projection),
        "no_active_tumor_histone_support",
        dtype=object,
    )
    breadth = projection["tumor_active_support_breadth"].to_numpy(np.int8)
    support_class[breadth == 1] = "cell_line_specific"
    support_class[breadth >= 2] = "multi_cell_line_shared"
    projection["tumor_cell_line_support_class"] = support_class

    # Readable cell-line list for downstream tables/figures.
    supported_lists = []
    dominant_lists = []
    for row in projection[active_any_cols + active_dom_cols].itertuples(
        index=False,
        name=None,
    ):
        active_flags = row[: len(TUMOR_EIDS)]
        dominant_flags = row[len(TUMOR_EIDS) :]
        supported_lists.append(
            ";".join(
                eid for eid, flag in zip(TUMOR_EIDS, active_flags) if bool(flag)
            )
        )
        dominant_lists.append(
            ";".join(
                eid for eid, flag in zip(TUMOR_EIDS, dominant_flags) if bool(flag)
            )
        )
    projection["tumor_active_supported_eids"] = supported_lists
    projection["tumor_active_dominant_eids"] = dominant_lists

    return projection


# =============================================================================
# 7. Descriptive enrichment
# =============================================================================

def cutoff_definitions(
    projection: pd.DataFrame,
) -> list[tuple[str, np.ndarray]]:
    n = len(projection)
    definitions = []
    for fraction in TOP_FRACTIONS:
        k = int(math.floor(n * fraction))
        definitions.append(
            (
                f"Top{fraction * 100:g}%",
                (
                    projection["herv_functional_priority_rank"]
                    .to_numpy(np.int64)
                    <= k
                ),
            )
        )

    definitions.append(
        (
            "FrozenTop10Candidate",
            projection["is_genomewide_top10_candidate"]
            .astype(bool)
            .to_numpy(),
        )
    )
    return definitions


def support_vectors(
    projection: pd.DataFrame,
    eid: str,
) -> list[tuple[str, np.ndarray]]:
    vectors: list[tuple[str, np.ndarray]] = [
        (
            "any_active",
            projection[f"tumor_{eid}_any_active_support"]
            .astype(bool)
            .to_numpy(),
        ),
        (
            "active_dominant",
            projection[f"tumor_{eid}_active_dominant"]
            .astype(bool)
            .to_numpy(),
        ),
    ]
    for mark in ACTIVE_MARKS__24:
        vectors.append(
            (
                mark,
                projection[flat_has_col(eid, mark)]
                .astype(bool)
                .to_numpy(),
            )
        )
    vectors.append(
        (
            "any_repressive",
            projection[f"tumor_{eid}_any_repressive_support"]
            .astype(bool)
            .to_numpy(),
        )
    )
    for mark in REPRESSIVE_MARKS__24:
        vectors.append(
            (
                mark,
                projection[flat_has_col(eid, mark)]
                .astype(bool)
                .to_numpy(),
            )
        )
    return vectors


def build_enrichment(projection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cutoffs = cutoff_definitions(projection)
    n = len(projection)

    for eid in TUMOR_EIDS:
        for support_name, support in support_vectors(projection, eid):
            background_hits = int(support.sum())
            background_rate = background_hits / n

            for cutoff_name, selected in cutoffs:
                k = int(selected.sum())
                hits = int(support[selected].sum())
                rate = hits / k if k else float("nan")
                fe = (
                    rate / background_rate
                    if background_rate > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "scope": "cell_line",
                        "eid": eid,
                        "cell_line": CELL_LINE_NAMES[eid],
                        "support_type": support_name,
                        "rank_cutoff": cutoff_name,
                        "K": k,
                        "hits": hits,
                        "support_rate": rate,
                        "background_hits": background_hits,
                        "background_rate": background_rate,
                        "fold_enrichment": fe,
                    }
                )

    # Pan-tumor summaries.
    pan_vectors = [
        (
            "any_active_any_cell_line",
            projection["tumor_active_support_breadth"].to_numpy(np.int8) >= 1,
        ),
        (
            "active_dominant_any_cell_line",
            projection["tumor_active_dominant_breadth"].to_numpy(np.int8) >= 1,
        ),
        (
            "any_repressive_any_cell_line",
            projection["tumor_repressive_support_breadth"].to_numpy(np.int8) >= 1,
        ),
        (
            "active_support_multiple_cell_lines",
            projection["tumor_active_support_breadth"].to_numpy(np.int8) >= 2,
        ),
    ]
    for mark in TUMOR_MARKS:
        cols = [flat_has_col(eid, mark) for eid in TUMOR_EIDS]
        pan_vectors.append(
            (
                f"{mark}_any_cell_line",
                projection[cols].astype(bool).any(axis=1).to_numpy(),
            )
        )

    for support_name, support in pan_vectors:
        background_hits = int(support.sum())
        background_rate = background_hits / n
        for cutoff_name, selected in cutoffs:
            k = int(selected.sum())
            hits = int(support[selected].sum())
            rate = hits / k if k else float("nan")
            fe = rate / background_rate if background_rate > 0 else float("nan")
            rows.append(
                {
                    "scope": "pan_tumor",
                    "eid": "PAN_TUMOR",
                    "cell_line": "Any of five tumor cell lines",
                    "support_type": support_name,
                    "rank_cutoff": cutoff_name,
                    "K": k,
                    "hits": hits,
                    "support_rate": rate,
                    "background_hits": background_hits,
                    "background_rate": background_rate,
                    "fold_enrichment": fe,
                }
            )

    result = pd.DataFrame(rows)
    return result


# =============================================================================
# 8. Candidate and family interpretation
# =============================================================================

def mark_string(projection: pd.DataFrame, eid: str) -> pd.Series:
    values = []
    columns = [flat_has_col(eid, mark) for mark in TUMOR_MARKS]
    for row in projection[columns].itertuples(index=False, name=None):
        values.append(
            ",".join(
                mark for mark, flag in zip(TUMOR_MARKS, row) if int(flag) == 1
            )
        )
    return pd.Series(values, index=projection.index, dtype="object")


def build_candidate_table(projection: pd.DataFrame) -> pd.DataFrame:
    candidate = projection.loc[
        projection["potential_reactivation_candidate"].astype(bool)
    ].copy()

    for eid in TUMOR_EIDS:
        candidate[f"tumor_{eid}_supported_marks"] = mark_string(
            candidate,
            eid,
        )

    selected_columns = [
        ID__24,
        "chrom",
        "start",
        "end",
        "strand",
        "repName",
        "repClass",
        "repFamily",
        "repeat_divergence",
        "length",
        "dist_to_tss",
        "nearest_gene_id",
        "nearest_gene_name",
        "nearest_gene_type",
        "score_final_global",
        "herv_functional_priority_rank",
        "herv_functional_priority_score",
        "tumor_epigenomic_support_score",
        "tumor_repressive_context_score",
        "tumor_active_support_breadth",
        "tumor_active_dominant_breadth",
        "tumor_repressive_support_breadth",
        "tumor_total_active_mark_hits",
        "tumor_total_repressive_mark_hits",
        "tumor_cell_line_support_class",
        "tumor_active_supported_eids",
        "tumor_active_dominant_eids",
    ]
    for eid in TUMOR_EIDS:
        selected_columns.extend(
            [
                f"tumor_{eid}_supported_marks",
                f"tumor_{eid}_active_mark_count",
                f"tumor_{eid}_repressive_mark_count",
                f"tumor_{eid}_active_total_overlap_bp",
                f"tumor_{eid}_repressive_total_overlap_bp",
            ]
        )

    candidate = candidate[selected_columns].sort_values(
        [
            "herv_functional_priority_rank",
            "tumor_active_support_breadth",
            ID__24,
        ],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return candidate


def build_family_interpretation(
    projection: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    top1 = projection.loc[projection["m23_top1pct"].astype(bool)].copy()

    top1_counts = (
        top1["repFamily"]
        .fillna("NA")
        .astype(str)
        .value_counts()
        .rename("top1_family_count")
    )
    candidate_counts = (
        candidate["repFamily"]
        .fillna("NA")
        .astype(str)
        .value_counts()
        .rename("candidate_family_count")
    )

    families = sorted(set(top1_counts.index) | set(candidate_counts.index))
    rows = []

    for family in families:
        top1_n = int(top1_counts.get(family, 0))
        cand_n = int(candidate_counts.get(family, 0))
        top1_fraction = top1_n / len(top1) if len(top1) else float("nan")
        cand_fraction = cand_n / len(candidate) if len(candidate) else float("nan")
        fe = (
            cand_fraction / top1_fraction
            if top1_fraction > 0 and len(candidate) > 0
            else float("nan")
        )

        subset = candidate.loc[
            candidate["repFamily"].fillna("NA").astype(str).eq(family)
        ]
        rows.append(
            {
                "repFamily": family,
                "candidate_family_count": cand_n,
                "top1_family_count": top1_n,
                "candidate_family_fraction": cand_fraction,
                "top1_family_fraction": top1_fraction,
                "fold_enrichment_vs_M23_Top1": fe,
                "candidate_mean_active_support_breadth": (
                    float(subset["tumor_active_support_breadth"].mean())
                    if len(subset)
                    else None
                ),
                "candidate_multi_cell_line_count": (
                    int(
                        (
                            subset["tumor_active_support_breadth"]
                            .astype(int)
                            >= 2
                        ).sum()
                    )
                    if len(subset)
                    else 0
                ),
                "candidate_best_rank": (
                    int(subset["herv_functional_priority_rank"].min())
                    if len(subset)
                    else None
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        [
            "candidate_family_count",
            "fold_enrichment_vs_M23_Top1",
            "repFamily",
        ],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return result


# =============================================================================
# 9. Main
# =============================================================================

def step_24() -> None:
    started = time.perf_counter()
    prepare_output()


    atlas = load_m23_atlas()
    tumor, tumor_filter_audit = load_tumor_long(set(atlas[ID__24]))


    projection = build_projection(atlas, tumor)

    enrichment = build_enrichment(projection)
    candidates = build_candidate_table(projection)

    atomic_parquet(projection, OUT_ALL)
    atomic_tsv(candidates, OUT_CANDIDATES)
    atomic_tsv(enrichment, OUT_ENRICH)

    # Core descriptive summaries.
    active_any = int((projection["tumor_active_support_breadth"] >= 1).sum())
    active_multi = int((projection["tumor_active_support_breadth"] >= 2).sum())
    active_dom_any = int((projection["tumor_active_dominant_breadth"] >= 1).sum())
    candidate_n = len(candidates)
    candidate_multi = int(
        (candidates["tumor_active_support_breadth"].astype(int) >= 2).sum()
    ) if candidate_n else 0
    candidate_specific = int(
        (candidates["tumor_active_support_breadth"].astype(int) == 1).sum()
    ) if candidate_n else 0

    cell_line_summary = {}
    for eid in TUMOR_EIDS:
        cell_line_summary[eid] = {
            "name": CELL_LINE_NAMES[eid],
            "any_active_HERVs": int(
                projection[f"tumor_{eid}_any_active_support"].sum()
            ),
            "active_dominant_HERVs": int(
                projection[f"tumor_{eid}_active_dominant"].sum()
            ),
            "any_repressive_HERVs": int(
                projection[f"tumor_{eid}_any_repressive_support"].sum()
            ),
            "Top1_potential_candidates": int(
                (
                    projection["m23_top1pct"]
                    & projection[f"tumor_{eid}_any_active_support"]
                ).sum()
            ),
        }

    # Pull a few headline FE values only for logging/manifest, not selection.
    headline = {}
    for eid in [*TUMOR_EIDS, "PAN_TUMOR"]:
        scope_rows = enrichment.loc[enrichment["eid"].eq(eid)]
        if eid == "PAN_TUMOR":
            support_name = "any_active_any_cell_line"
        else:
            support_name = "any_active"
        row = scope_rows.loc[
            scope_rows["support_type"].eq(support_name)
            & scope_rows["rank_cutoff"].eq("Top1%")
        ]
        if len(row) == 1:
            headline[eid] = float(row.iloc[0]["fold_enrichment"])

    elapsed = time.perf_counter() - started

    manifest = {
        "script": SCRIPT,
        "version": VERSION,
        "status": "PASSED",
        "created_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "final_global": EXPECTED_GLOBAL,
        "final_primary": EXPECTED_PRIMARY,
        "herv_rows": EXPECTED_HERV_N,
        "top1_k": TOP1_K,
        "tumor_eids": {
            eid: CELL_LINE_NAMES[eid] for eid in TUMOR_EIDS
        },
        "active_marks": ACTIVE_MARKS__24,
        "repressive_marks": REPRESSIVE_MARKS__24,
        "summary": {
            "HERVs_with_active_support_any_cell_line": active_any,
            "HERVs_with_active_support_multiple_cell_lines": active_multi,
            "HERVs_with_active_dominant_support_any_cell_line": active_dom_any,
            "potential_reactivation_candidates": candidate_n,
            "cell_line_specific_candidates": candidate_specific,
            "multi_cell_line_candidates": candidate_multi,
            "top1_any_active_FE": headline,
            "by_cell_line": cell_line_summary,
        },
        "filter_to_M23_model_universe": tumor_filter_audit,
    }
    atomic_json(OUT_MANIFEST, manifest)


def main() -> None:
    step_22()
    step_23()
    step_24()


if __name__ == "__main__":
    main()
