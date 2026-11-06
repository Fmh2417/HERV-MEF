#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from modeling_common import write_parquet


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import json
import math
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# step 19: Build the functional evidence candidate pool
# --------------------------------------------------------------------------



MODELING = Path(__file__).resolve().parent
M00_TABLE = MODELING / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M18_VALID = MODELING / "output/18_export_final_global_strict_oof/18_valid_full_train_final_global_predictions.parquet"
OUT = MODELING / "output/19_build_functional_evidence_candidate_pool"

ID, SPLIT, ROW, LABEL = "herv_internal_id", "split", "canonical_split_row_index", "is_fantom5_enhancer"
GLOBAL_SCORE, GLOBAL_RANK, GLOBAL_RANK_PCT = "score_final_global", "rank_final_global", "rank_pct_final_global"
TRAIN_ORIGIN, VALID_ORIGIN = "train_oof_final_global", "valid_full_train_final_global"

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
M00_READ_COLUMNS = [ID, SPLIT, *EVIDENCE38]

FORMAL_FRACTION, FORMAL_POOL_NAME = 0.10, "FinalGlobal_Top10pct"
FORMAL_M20_ARM = "A3_ALL38_PRIMARY"






def load_m18(path, split):
    df = pd.read_parquet(path).reset_index(drop=True)
    return df


def attach(left, evidence, split):
    right = evidence.loc[evidence[SPLIT].astype(str) == split].copy()
    joined = left.merge(right, on=[ID, SPLIT], how="left", sort=False, validate="one_to_one")
    return joined


def rank(df):
    out = df.sort_values([GLOBAL_SCORE, ROW, ID],
                         ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    out["final_global_rank_recomputed"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def pool(ranked):
    k = max(1, int(round(len(ranked) * FORMAL_FRACTION)))
    p = ranked.iloc[:k].copy()
    p["candidate_pool_name"] = FORMAL_POOL_NAME
    p["candidate_fraction"] = FORMAL_FRACTION
    p["candidate_size"] = k
    p["is_formal_top10_candidate"] = True
    return p

def step_19():
    OUT.mkdir(parents=True, exist_ok=True)


    valid = attach(load_m18(M18_VALID, "valid"), pq.read_table(
        M00_TABLE, columns=M00_READ_COLUMNS,
        filters=[(SPLIT, "in", ["train", "valid"])]).to_pandas(), "valid")


    write_parquet(valid, OUT / "19_valid_functional_evidence_table.parquet")

    rv = rank(valid)
    vp = pool(rv)
    write_parquet(vp, OUT / "19_valid_top10_candidate_table.parquet")


# --------------------------------------------------------------------------
# step 20: Train the functional evidence reranker
# --------------------------------------------------------------------------



ARMS = ("A1_ROADMAP18", "A2_NONROADMAP20", "A3_ALL38_PRIMARY")


MODELING__20 = Path(__file__).resolve().parent
M19_DIR = MODELING__20 / "output/19_build_functional_evidence_candidate_pool"
OUT__20 = MODELING__20 / "output/20_train_functional_evidence_reranker"
RUN_ROOT = OUT__20 / "20_runs"

ID__20, SPLIT__20, ROW__20, LABEL__20 = "herv_internal_id", "split", "canonical_split_row_index", "is_fantom5_enhancer"
GLOBAL_SCORE__20, RECOMP_RANK = "score_final_global", "final_global_rank_recomputed"
POOL_NAME, N_TRAIN, N_VALID = "FinalGlobal_Top10pct", 33_622, 7_205

ROADMAP18__20 = [f"roadmap_{m}_{s}" for m in (
    "DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3")
    for s in ("sample_count", "max_overlap_bp", "total_overlap_bp")]
MPRA8__20 = ["mpra_overlap_count", "mpra_max_overlap_bp", "mpra_total_overlap_bp",
         "mpra_active_overlap_count", "mpra_active_max_overlap_bp",
         "mpra_active_total_overlap_bp", "is_mpra_overlap", "is_mpra_active_supported"]
CCRE6__20 = ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_ccre_ctcf_overlap",
         "is_ccre_pls_overlap", "ccre_els_overlap_count", "ccre_els_total_overlap_bp"]
SERVE6__20 = ["serve_gtex_tissue_count", "serve_gtex_total_overlap_bp",
          "is_serve_gtex_supported", "serve_gtex_annotation_count",
          "serve_noncoding_overlap_count", "serve_noncoding_total_overlap_bp"]
EVIDENCE38__20 = [*ROADMAP18__20, *MPRA8__20, *CCRE6__20, *SERVE6__20]
ARM_FEATURES = {"A1_ROADMAP18": ROADMAP18__20,
                "A2_NONROADMAP20": [*MPRA8__20, *CCRE6__20, *SERVE6__20],
                "A3_ALL38_PRIMARY": EVIDENCE38__20}

BS, SEED, EPOCHS = 32, 99999999, 30
RS, PM, PW, RL2 = 0.15, 1.0, 0.50, 0.01
LR, WD, PRED_BS = 1e-3, 1e-4, 8192
ATTR_N, ATTR_SEED = 4096, 99999999


def wp(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False) if path.suffix == ".tsv" else df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def wt(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp); os.replace(tmp, path)


def load_pool(split):
    df = pd.read_parquet(M19_DIR / f"19_{split}_top10_candidate_table.parquet").reset_index(drop=True)
    return df


def seed_all():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Res(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 16), nn.LayerNorm(16), nn.GELU(),
                                 nn.Linear(16, 1))
    def delta(self, e): return RS * torch.tanh(self.net(e).squeeze(-1))
    def forward(self, e, b):
        d = self.delta(e); return b + d, d


def pair_loss(logits, labels):
    p, n = logits[labels > 0.5], logits[labels <= 0.5]
    return logits.new_zeros(()) if p.numel() == 0 or n.numel() == 0 else \
        F.relu(PM - (p[:, None] - n[None, :])).mean()


def batches(n, epoch):
    rng = np.random.default_rng(SEED + epoch); idx = rng.permutation(n).astype(np.int64)
    b = [idx[s:s+BS] for s in range(0, n, BS)]
    if len(b[-1]) == 1:
        prev, last = b[-2], b[-1]
        b[-2], b[-1] = prev[:-1], np.concatenate([prev[-1:], last])
    return b


def train_epoch(model, opt, x, base, y, pos_w, epoch, device):
    model.train()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    total, n = 0.0, 0
    for idx in batches(len(x), epoch):
        bx = torch.from_numpy(x[idx]).to(device); bb = torch.from_numpy(base[idx]).to(device)
        by = torch.from_numpy(y[idx]).to(device)
        opt.zero_grad(set_to_none=True)
        logits, d = model(bx, bb)
        loss = bce_fn(logits, by) + PW * pair_loss(logits, by) + RL2 * torch.mean(d.square())
        loss.backward(); opt.step()
        total += float(loss.detach().cpu()) * len(idx); n += len(idx)
    return total / n


@torch.inference_mode()
def predict(model, x, base, device):
    model.eval(); sp, dp, lp = [], [], []
    for s in range(0, len(x), PRED_BS):
        bx = torch.from_numpy(x[s:s+PRED_BS]).to(device)
        bb = torch.from_numpy(base[s:s+PRED_BS]).to(device)
        lg, d = model(bx, bb)
        sp.append(torch.sigmoid(lg).float().cpu().numpy())
        dp.append(d.float().cpu().numpy()); lp.append(lg.float().cpu().numpy())
    return (np.concatenate(sp).astype(float), np.concatenate(dp).astype(float),
            np.concatenate(lp).astype(float))


def attr_compute(model, sx, features, arm, epoch, device):
    x = torch.tensor(sx, dtype=torch.float32, device=device, requires_grad=True)
    d = model.delta(x)
    g = torch.autograd.grad(d.sum(), x)[0]
    v = torch.mean(torch.abs(g * x), dim=0).detach().cpu().numpy().astype(float)
    return pd.DataFrame({"arm": arm, "epoch": epoch, "feature_index": range(len(features)),
                         "feature": features, "mean_abs_grad_x_input": v,
                         "normalized_attribution": v / v.sum() if v.sum() > 0 else np.zeros_like(v)})


def run_arm(arm, attr_sample):
    device = torch.device("cuda:0")

    train, valid = load_pool("train"), load_pool("valid")
    features = ARM_FEATURES[arm]
    root = RUN_ROOT / arm
    if root.exists():
        import shutil; shutil.rmtree(root)
    (root / "checkpoints").mkdir(parents=True); (root / "valid_predictions").mkdir(parents=True)

    tr_log = np.log1p(np.maximum(train[features].to_numpy(float), 0))
    va_log = np.log1p(np.maximum(valid[features].to_numpy(float), 0))
    sc = StandardScaler().fit(tr_log)
    train_x = sc.transform(tr_log).astype(np.float32); valid_x = sc.transform(va_log).astype(np.float32)
    joblib.dump({"scaler": sc, "features": features}, root / f"20_{arm}_preprocessor.joblib", compress=3)

    lt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).astype(np.float32)
    train_base = lt(train[GLOBAL_SCORE__20].to_numpy(float)); valid_base = lt(valid[GLOBAL_SCORE__20].to_numpy(float))
    train_y = train[LABEL__20].to_numpy(np.float32)
    pos_w = float((len(train_y) - train_y.sum()) / train_y.sum())
    attr_x = train_x[attr_sample["train_candidate_row_index"].to_numpy(np.int64)]

    seed_all()
    model = Res(len(features)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    history, attrs, registry = [], [], []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, opt, train_x, train_base, train_y, pos_w, epoch, device)
        scores, delta, logits = predict(model, valid_x, valid_base, device)

        wp(pd.DataFrame({ID__20: valid[ID__20].astype(str), SPLIT__20: "valid",
                         ROW__20: valid[ROW__20].to_numpy(np.int64), LABEL__20: valid[LABEL__20].to_numpy(np.int8),
                         GLOBAL_SCORE__20: valid[GLOBAL_SCORE__20].to_numpy(float),
                         RECOMP_RANK: valid[RECOMP_RANK].to_numpy(np.int64),
                         "arm": arm, "epoch": epoch, "score_fer": scores, "delta_logit": delta,
                         "final_candidate_logit": logits}),
           root / "valid_predictions" / f"20_{arm}_epoch_{epoch:02d}.parquet")

        attrs.append(attr_compute(model, attr_x, features, arm, epoch, device))
        ckpt = root / "checkpoints" / f"20_{arm}_epoch_{epoch:02d}.pt"
        wt({"arm": arm, "epoch": epoch, "features": features,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, ckpt)

        history.append({"arm": arm, "epoch": epoch, "train_loss": loss,
                        "mean_abs_delta": float(np.mean(np.abs(delta)))})
        registry.append({"candidate_id": f"{arm}_epoch_{epoch:02d}", "arm": arm,
                         "epoch": epoch, "checkpoint_path": str(ckpt)})
        wp(pd.DataFrame(history), root / f"20_{arm}_training_history.tsv")
        wp(pd.concat(attrs, ignore_index=True), root / f"20_{arm}_feature_attribution.tsv")
        wp(pd.DataFrame(registry), root / f"20_{arm}_checkpoint_registry.tsv")


def build_a0(valid):
    wp(pd.DataFrame({ID__20: valid[ID__20].astype(str), SPLIT__20: "valid",
                     ROW__20: valid[ROW__20].to_numpy(np.int64), LABEL__20: valid[LABEL__20].to_numpy(np.int8),
                     GLOBAL_SCORE__20: valid[GLOBAL_SCORE__20].to_numpy(float),
                     RECOMP_RANK: valid[RECOMP_RANK].to_numpy(np.int64),
                     "arm": "A0_GLOBAL_IDENTITY", "epoch": 0,
                     "score_fer": valid[GLOBAL_SCORE__20].to_numpy(float),
                     "delta_logit": np.zeros(len(valid))}),
       OUT__20 / "20_A0_valid_candidate_predictions.parquet")


def build_attr_sample(train):
    rng = np.random.default_rng(ATTR_SEED)
    idx = np.sort(rng.choice(len(train), size=ATTR_N, replace=False))
    return pd.DataFrame({"sample_index": range(ATTR_N), "train_candidate_row_index": idx,
                         ID__20: train.iloc[idx][ID__20].astype(str).to_numpy(),
                         ROW__20: train.iloc[idx][ROW__20].to_numpy(np.int64),
                         RECOMP_RANK: train.iloc[idx][RECOMP_RANK].to_numpy(np.int64),
                         GLOBAL_SCORE__20: train.iloc[idx][GLOBAL_SCORE__20].to_numpy(float)})


def step_20():
    OUT__20.mkdir(parents=True, exist_ok=True); RUN_ROOT.mkdir(parents=True, exist_ok=True)

    train, valid = load_pool("train"), load_pool("valid")
    build_a0(valid)
    attr_sample = build_attr_sample(train)
    wp(attr_sample, OUT__20 / "20_attribution_sample_manifest.parquet")

    for arm in ARMS:
        run_arm(arm, attr_sample)

    for name, files in [("20_all_arm_training_history.tsv", "training_history"),
                        ("20_all_feature_attribution.tsv", "feature_attribution"),
                        ("20_checkpoint_registry.tsv", "checkpoint_registry")]:
        df = pd.concat([pd.read_csv(RUN_ROOT / arm / f"20_{arm}_{files}.tsv", sep="\t")
                        for arm in ARMS], ignore_index=True)
        wp(df, OUT__20 / name)


# --------------------------------------------------------------------------
# step 21: Select and freeze the functional evidence reranker
# --------------------------------------------------------------------------



MODELING__21 = Path(__file__).resolve().parent
M17_FROZEN = MODELING__21 / "output/17_audit_select_final_global_ranker/17_frozen"
M19_DIR__21 = MODELING__21 / "output/19_build_functional_evidence_candidate_pool"
M20_DIR = MODELING__21 / "output/20_train_functional_evidence_reranker"
M20_RUNS = M20_DIR / "20_runs"
OUT__21 = MODELING__21 / "output/21_select_and_freeze_functional_evidence_reranker"
FROZEN = OUT__21 / "21_frozen"

ID__21, SPLIT__21, ROW__21, LABEL__21 = "herv_internal_id", "split", "canonical_split_row_index", "is_fantom5_enhancer"
GLOBAL_SCORE__21 = "score_final_global"
N_VALID__21, N_VALID_POS, N_POOL, N_POOL_POS = 72_048, 1_077, 7_205, 612
POOL_NAME__21 = "FinalGlobal_Top10pct"
TRAINED_ARMS = ("A1_ROADMAP18", "A2_NONROADMAP20", "A3_ALL38_PRIMARY")
FORMAL_ARM, A0_ID = "A3_ALL38_PRIMARY", "A0_GLOBAL_IDENTITY"
REPEATS, SEED__21, PRED_BS__21, EPOCHS__21 = 100, 99999999, 8192, 30
FLOAT_TOL, RESIDUAL_SCALE = 1e-5, 0.15


ROADMAP18__21 = [f"roadmap_{m}_{s}" for m in (
    "DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3")
    for s in ("sample_count", "max_overlap_bp", "total_overlap_bp")]
MPRA8__21 = ["mpra_overlap_count", "mpra_max_overlap_bp", "mpra_total_overlap_bp",
         "mpra_active_overlap_count", "mpra_active_max_overlap_bp",
         "mpra_active_total_overlap_bp", "is_mpra_overlap", "is_mpra_active_supported"]
CCRE6__21 = ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_ccre_ctcf_overlap",
         "is_ccre_pls_overlap", "ccre_els_overlap_count", "ccre_els_total_overlap_bp"]
SERVE6__21 = ["serve_gtex_tissue_count", "serve_gtex_total_overlap_bp",
          "is_serve_gtex_supported", "serve_gtex_annotation_count",
          "serve_noncoding_overlap_count", "serve_noncoding_total_overlap_bp"]
EVIDENCE38__21 = [*ROADMAP18__21, *MPRA8__21, *CCRE6__21, *SERVE6__21]
ARM_FEATURES__21 = {"A1_ROADMAP18": ROADMAP18__21,
                "A2_NONROADMAP20": [*MPRA8__21, *CCRE6__21, *SERVE6__21],
                "A3_ALL38_PRIMARY": EVIDENCE38__21}


def wp__21(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False) if path.suffix == ".tsv" else df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def wj(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class FER(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 16), nn.LayerNorm(16), nn.GELU(),
                                 nn.Linear(16, 1))

    def delta(self, e):
        return RESIDUAL_SCALE * torch.tanh(self.net(e).squeeze(-1))

    def forward(self, e, b):
        d = self.delta(e)
        return b + d, d


def arm_ckpt_path(arm, epoch):
    return M20_RUNS / arm / "checkpoints" / f"20_{arm}_epoch_{epoch:02d}.pt"


def arm_pred_path(arm, epoch):
    return M20_RUNS / arm / "valid_predictions" / f"20_{arm}_epoch_{epoch:02d}.parquet"


def load_data():
    full = pd.read_parquet(M19_DIR__21 / "19_valid_functional_evidence_table.parquet").reset_index(drop=True)
    pool = pd.read_parquet(M19_DIR__21 / "19_valid_top10_candidate_table.parquet").reset_index(drop=True)
    ids = full[ID__21].astype(str).to_numpy()
    rows = full[ROW__21].to_numpy(np.int64)
    sc = full[GLOBAL_SCORE__21].to_numpy(np.float64)
    global_order = np.lexsort((ids, rows, -sc))
    return full, pool, global_order


def transform_pool(pool, arm):
    pre = joblib.load(M20_RUNS / arm / f"20_{arm}_preprocessor.joblib")
    raw = pool[ARM_FEATURES__21[arm]].to_numpy(np.float64)
    return pre["scaler"].transform(np.log1p(np.maximum(raw, 0))).astype(np.float32)


def reinfer_arm(arm, pool, base_logit, device):
    x = transform_pool(pool, arm)
    out = {}
    for epoch in range(1, EPOCHS__21 + 1):
        ckpt = torch.load(arm_ckpt_path(arm, epoch), map_location="cpu", weights_only=False)
        model = FER(len(ARM_FEATURES__21[arm])).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        sp, dp = [], []
        with torch.inference_mode():
            for s in range(0, len(x), PRED_BS__21):
                bx = torch.from_numpy(x[s:s+PRED_BS__21]).to(device)
                bb = torch.from_numpy(base_logit[s:s+PRED_BS__21]).to(device)
                lg, d = model(bx, bb)
                sp.append(torch.sigmoid(lg).float().cpu().numpy())
                dp.append(d.float().cpu().numpy())
        scores = np.concatenate(sp).astype(np.float64)
        delta = np.concatenate(dp).astype(np.float64)
        out[epoch] = (scores, delta)
        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()
    return out


def build_shared_samples(labels):
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(SEED__21)
    return [np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)]).astype(np.int64)
            for _ in range(REPEATS)]


def build_final_order(full, global_order, pool, cand_scores):
    n = len(full)
    ids = full[ID__21].astype(str).to_numpy()
    pool_ids = pool[ID__21].astype(str).to_numpy()
    id_to_idx = {v: i for i, v in enumerate(ids)}
    cand_idx = np.array([id_to_idx[v] for v in pool_ids], dtype=np.int64)
    local = np.lexsort((pool_ids, pool[ROW__21].to_numpy(np.int64),
                        -pool[GLOBAL_SCORE__21].to_numpy(np.float64),
                        -np.asarray(cand_scores, np.float64)))
    cand_block = cand_idx[local]
    mask = np.zeros(n, dtype=bool)
    mask[cand_idx] = True
    order = np.concatenate([cand_block, global_order[~mask[global_order]]])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    return order, rank, (n - rank + 1).astype(np.float64) / n


def gold_metrics(labels, order, rank_score, samples):
    n = len(labels)
    prev = float(labels.mean())
    m = {}
    for frac, name in [(0.005, "top0_5"), (0.01, "top1"), (0.05, "top5")]:
        k = int(math.floor(n * frac))
        hits = int(labels[order[:k]].sum())
        m[f"{name}_hits"] = hits
        m[f"{name}_fe"] = float(hits / (k * prev))
    ap = [average_precision_score(labels[i], rank_score[i]) for i in samples]
    au = [roc_auc_score(labels[i], rank_score[i]) for i in samples]
    m["auprc"], m["auroc"] = float(np.mean(ap)), float(np.mean(au))
    m["auprc_std"], m["auroc_std"] = float(np.std(ap, ddof=1)), float(np.std(au, ddof=1))
    return m


def step_21():
    global_manifest = MODELING__21 / "output/17_audit_select_final_global_ranker/17_frozen_final_global_ranker_manifest.json"
    global_candidate = json.loads(global_manifest.read_text(encoding="utf-8"))["selected_candidate_id"]

    OUT__21.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    full, pool, global_order = load_data()
    labels = full[LABEL__21].to_numpy(np.int8)
    p = np.clip(pool[GLOBAL_SCORE__21].to_numpy(np.float64), 1e-6, 1 - 1e-6)
    base_logit = np.log(p / (1 - p)).astype(np.float32)
    samples = build_shared_samples(labels)

    arm_scores = {arm: reinfer_arm(arm, pool, base_logit, device) for arm in TRAINED_ARMS}

    rows, candidate_orders = [], {}

    order, rank, rscore = build_final_order(
        full, global_order, pool, pool[GLOBAL_SCORE__21].to_numpy(np.float64))
    m = gold_metrics(labels, order, rscore, samples)
    rows.append({"candidate_id": A0_ID, "arm": A0_ID, "epoch": 0, **m,
                 "formal_eligible": True, "model_type": "identity"})
    candidate_orders[A0_ID] = (order, rank, rscore)

    for arm in TRAINED_ARMS:
        eligible = arm == FORMAL_ARM
        for epoch, (scores, _) in arm_scores[arm].items():
            order, rank, rscore = build_final_order(full, global_order, pool, scores)
            m = gold_metrics(labels, order, rscore, samples)
            cid = f"{arm}_epoch_{epoch:02d}"
            rows.append({"candidate_id": cid, "arm": arm, "epoch": epoch, **m,
                         "formal_eligible": eligible, "model_type": "fer"})
            candidate_orders[cid] = (order, rank, rscore)

    metrics = pd.DataFrame(rows)
    wp__21(metrics, OUT__21 / "21_all_candidate_gold_metrics.tsv")

    formal = metrics[metrics["formal_eligible"]].copy()
    formal["complexity"] = np.where(formal["arm"] == A0_ID, 0, 1)
    formal = formal.sort_values(
        ["top1_hits", "top5_hits", "auprc", "top0_5_hits",
         "auroc", "complexity", "epoch", "candidate_id"],
        ascending=[False, False, False, False, False, True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)
    formal.insert(0, "formal_rank", np.arange(1, len(formal) + 1))
    formal["selected"] = formal["formal_rank"].eq(1)
    selected = formal.iloc[0]
    wp__21(formal, OUT__21 / "21_validation_selection.tsv")

    arm_best = []
    for arm in [A0_ID, *TRAINED_ARMS]:
        sub = metrics[metrics["arm"] == arm].sort_values(
            ["top1_hits", "top5_hits", "auprc", "top0_5_hits", "auroc", "epoch"],
            ascending=[False, False, False, False, False, True],
            kind="mergesort").iloc[0]
        arm_best.append({
            "arm": arm, "best_candidate_id": sub["candidate_id"], "epoch": int(sub["epoch"]),
            "top0_5_hits": int(sub["top0_5_hits"]), "top1_hits": int(sub["top1_hits"]),
            "top5_hits": int(sub["top5_hits"]), "auprc": float(sub["auprc"]),
            "auroc": float(sub["auroc"]), "formal_eligible": arm in (A0_ID, FORMAL_ARM)
        })
    wp__21(pd.DataFrame(arm_best), OUT__21 / "21_arm_best_summary.tsv")

    sel_id = str(selected["candidate_id"])
    order, rank, rscore = candidate_orders[sel_id]
    if sel_id == A0_ID:
        score_cand = pool[GLOBAL_SCORE__21].to_numpy(np.float64)
        delta_cand = np.zeros(len(pool))
    else:
        score_cand, delta_cand = arm_scores[str(selected["arm"])][int(selected["epoch"])]

    id_to_full = {v: i for i, v in enumerate(full[ID__21].astype(str).to_numpy())}
    pool_ids = pool[ID__21].astype(str).to_numpy()
    is_cand = full[ID__21].astype(str).isin(set(pool_ids)).to_numpy(bool)
    score_full = np.full(len(full), np.nan)
    delta_full = np.full(len(full), np.nan)
    for i, pid in enumerate(pool_ids):
        j = id_to_full[pid]
        score_full[j] = score_cand[i]
        delta_full[j] = delta_cand[i]

    wp__21(pd.DataFrame({
        ID__21: full[ID__21].astype(str), SPLIT__21: "valid",
        ROW__21: full[ROW__21].to_numpy(np.int64), LABEL__21: labels,
        GLOBAL_SCORE__21: full[GLOBAL_SCORE__21].to_numpy(np.float64),
        "is_candidate": is_cand, "score_fer_candidate": score_full,
        "delta_logit_candidate": delta_full, "final_rank": rank,
        "final_rank_score": rscore, "selected_candidate_id": sel_id
    }), OUT__21 / "21_selected_validation_predictions.parquet")

    wp__21(pd.DataFrame({
        "repeat": np.arange(1, REPEATS + 1),
        "auprc": [average_precision_score(labels[i], rscore[i]) for i in samples],
        "auroc": [roc_auc_score(labels[i], rscore[i]) for i in samples]
    }), OUT__21 / "21_selected_validation_random1to1_metrics.tsv")

    topk = []
    for frac in [0.005, 0.01, 0.05]:
        k = int(math.floor(len(labels) * frac))
        hits = int(labels[order[:k]].sum())
        topk.append({"fraction": frac, "K": k, "hits": hits,
                     "fe": float(hits / (k * labels.mean()))})
    wp__21(pd.DataFrame(topk), OUT__21 / "21_selected_validation_topk_metrics.tsv")

    shutil.copy2(
        M17_FROZEN / "17_frozen_final_global_ranker.pt",
        FROZEN / "21_frozen_final_global_ranker.pt")
    shutil.copy2(
        M17_FROZEN / "17_frozen_final_global_preprocessor.joblib",
        FROZEN / "21_frozen_final_global_preprocessor.joblib")
    if str(selected["arm"]) == FORMAL_ARM:
        shutil.copy2(
            arm_ckpt_path(FORMAL_ARM, int(selected["epoch"])),
            FROZEN / "21_frozen_primary_fer_component.pt")
        shutil.copy2(
            M20_RUNS / FORMAL_ARM / f"20_{FORMAL_ARM}_preprocessor.joblib",
            FROZEN / "21_frozen_primary_fer_preprocessor.joblib")

    wj({
        "status": "FROZEN_FINAL_DEVELOPMENT_PRIMARY",
        "selected_candidate_id": sel_id, "selected_arm": str(selected["arm"]),
        "selected_epoch": int(selected["epoch"]),
        "selected_model_type": str(selected["model_type"]),
        "A0_safe_fallback_selected": str(selected["arm"]) == A0_ID,
        "A3_FER_accepted": str(selected["arm"]) == FORMAL_ARM,
        "formal_primary_route": FORMAL_ARM, "A1_A2_descriptive_only": True,
        "validation_gold": {k: float(selected[k]) for k in [
            "top0_5_hits", "top0_5_fe", "top1_hits", "top1_fe",
            "top5_hits", "top5_fe", "auprc", "auprc_std", "auroc", "auroc_std"
        ]},
        "final_global_candidate": global_candidate,
        "test_used": False
    }, OUT__21 / "21_final_primary_model_manifest.json")


def main() -> None:
    step_19()
    step_20()
    step_21()


if __name__ == "__main__":
    main()
