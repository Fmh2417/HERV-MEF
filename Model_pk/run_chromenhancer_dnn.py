#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CHROMENHANCER_MODEL,
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import importlib.util
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from pk_common import (
    SEEDS,
    balanced_samples,
    metrics,
    seed_all,
    summarize,
    train_sample_rows,
)



ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

CHRNET1 = CHROMENHANCER_MODEL
CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
RANK23 = (
    PROJECT / "Modeling" / "output" / "23_generate_genomewide_herv_functional_ranking"
    / "23_herv_genomewide_functional_ranking.parquet"
)
PKMASTER = ROOT / "output" / "00_sequence_pk_master" / "sequence_pk_master.parquet"
OUT = ROOT / "output" / "04_chromenhancer_dnn"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
MODEL_NAME = "ChromEnhancer-DNN (Model 1, HERV task adaptation)"

EPOCHS, BATCH, LR, HIDDEN, DROPOUT = 20, 1024, 1e-4, 256, 0.5

BASE_NUMERIC = [
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
GCTX = [
    "gctx_roadmap_active_mark_presence_count", "gctx_roadmap_repressive_mark_presence_count",
    "gctx_roadmap_active_sample_count_sum", "gctx_roadmap_repressive_sample_count_sum",
    "gctx_roadmap_active_max_overlap_bp_max", "gctx_roadmap_repressive_max_overlap_bp_max",
    "gctx_roadmap_active_total_overlap_bp_log1p", "gctx_roadmap_repressive_total_overlap_bp_log1p",
    "gctx_roadmap_active_minus_repressive_logbp",
    "gctx_mpra_is_overlap", "gctx_mpra_is_active_supported",
    "gctx_mpra_total_overlap_bp_log1p", "gctx_mpra_active_total_overlap_bp_log1p",
    "gctx_mpra_active_count_fraction", "gctx_mpra_active_bp_fraction",
    "gctx_ccre_is_ctcf_overlap", "gctx_ccre_is_pls_overlap",
    "gctx_ccre_els_overlap_count_log1p", "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_annotation_count_log1p", "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_overlap_count_log1p", "gctx_serve_noncoding_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]
CATEGORICAL = ["repClass", "repFamily", "nearest_gene_type"]
FEATURES = BASE_NUMERIC + GCTX + CATEGORICAL
FROM_RANK23 = [
    c for c in FEATURES
    if c not in ("is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR",
                 "is_GYPSY", "log_length", "log_abs_dist_to_tss")
]
FROM_CANON = ["is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY"]


def load_official_model():
    spec = importlib.util.spec_from_file_location("_chrnet1_official", CHRNET1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Chromatin_Network1


def feature_whitelist() -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        rows.append({
            "feature": feature,
            "type": "categorical" if feature in CATEGORICAL else "numeric",
            "source": ("step23" if feature in FROM_RANK23
                       else "canonical" if feature in FROM_CANON else "derived"),
            "in_global_ranker": True,
            "in_fer": feature in ("is_ccre_ca_overlap", "is_ccre_enhancer_like",
                                  "is_serve_gtex_supported", "serve_gtex_tissue_count"),
        })
    return pd.DataFrame(rows)


def load_tables() -> pd.DataFrame:
    canon = pq.read_table(CANON, columns=[ID, SPLIT, LABEL, *FROM_CANON]).to_pandas()
    if not canon[ID].is_unique:
        raise RuntimeError("canonical locus identifiers are not unique")
    rank = pq.read_table(RANK23, columns=[ID, *FROM_RANK23]).to_pandas()
    if not rank[ID].is_unique:
        raise RuntimeError("step-23 locus identifiers are not unique")
    master = pq.read_table(PKMASTER, columns=[ID, SPLIT, LABEL]).to_pandas()
    if not master[ID].is_unique:
        raise RuntimeError("PK master locus identifiers are not unique")

    df = canon.merge(rank, on=ID, how="left", validate="one_to_one")
    df = df.sort_values(ID, kind="stable").reset_index(drop=True)
    if len(df) != len(canon) or not df[FROM_RANK23].notna().all().all():
        raise RuntimeError("step-23 join is incomplete")
    df["log_length"] = np.log1p(df["length"].to_numpy(dtype=np.float64))
    df["log_abs_dist_to_tss"] = np.log1p(np.abs(df["dist_to_tss"].to_numpy(dtype=np.float64)))

    chk = master.merge(df[[ID, SPLIT, LABEL]], on=ID, how="inner", suffixes=("_pk", "_canon"))
    if len(chk) != len(master) or len(chk) != len(df):
        raise RuntimeError("locus sets differ between the joined tables")
    if not (chk[f"{SPLIT}_pk"] == chk[f"{SPLIT}_canon"]).all():
        raise RuntimeError("split disagrees between the joined tables")
    if not (chk[f"{LABEL}_pk"] == chk[f"{LABEL}_canon"]).all():
        raise RuntimeError("label disagrees between the joined tables")
    return df


def build_matrix(df: pd.DataFrame) -> tuple:
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    train = df[SPLIT].to_numpy() == "train"
    scaler = StandardScaler().fit(df.loc[train, numeric].to_numpy(dtype=np.float64))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    enc.fit(df.loc[train, CATEGORICAL].astype(str))
    joblib.dump({"scaler": scaler, "encoder": enc, "numeric": numeric,
                 "categorical": CATEGORICAL, "features": FEATURES},
                OUT / "preprocessor.joblib", compress=3)
    a = scaler.transform(df[numeric].to_numpy(dtype=np.float64)).astype(np.float32)
    b = enc.transform(df[CATEGORICAL].astype(str)).astype(np.float32)
    X = np.concatenate([a, b], axis=1).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise RuntimeError("non-finite value in the feature matrix")
    cat_names = [f"{col}={value}" for col, values in zip(CATEGORICAL, enc.categories_)
                 for value in values]
    return X, numeric, cat_names


def train_seed(X, y, idx, seed: int, model_cls, device: torch.device):
    seed_all(seed)
    model = model_cls(name=f"ChromEnhancer_DNN_Model1_HERV_seed{seed}",
                      input_size=X.shape[1], output_size=1, dnn_hidden_size=HIDDEN).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    Xs = torch.from_numpy(X[idx])
    ys = torch.from_numpy(y[idx].astype(np.float32)).reshape(-1, 1)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total, n = 0.0, 0
        for start in range(0, len(idx), BATCH):
            xb = Xs[start:start + BATCH].to(device)
            yb = ys[start:start + BATCH].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(xb)
            n += len(xb)
        history.append(total / n)
    return model, history


@torch.no_grad()
def predict(model, X, device: torch.device, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch):
        xb = torch.from_numpy(X[start:start + batch]).to(device)
        out[start:start + batch] = model(xb).squeeze(-1).cpu().numpy()
    return out


def main() -> None:
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_whitelist().to_csv(OUT / "feature_whitelist.tsv", sep="\t", index=False)
    df = load_tables()
    X, numeric, cat_names = build_matrix(df)
    y = df[LABEL].to_numpy(np.int8)

    model_cls = load_official_model()
    samples = balanced_samples(y[df[SPLIT].to_numpy() == "test"])
    train_mask = df[SPLIT].to_numpy() == "train"

    per_seed, prediction_frames = [], []
    for seed in SEEDS:
        idx = train_sample_rows(train_mask, y, seed)
        sub_y = y[idx]
        pd.DataFrame({
            ID: df[ID].to_numpy()[idx], SPLIT: df[SPLIT].to_numpy()[idx],
            LABEL: sub_y, "role": np.where(sub_y == 1, "positive", "background"),
            "seed": seed,
        }).to_csv(OUT / f"train_sample_ids_seed{seed}.tsv", sep="\t", index=False)

        model, history = train_seed(X, y, idx, seed, model_cls, device)
        torch.save({"state_dict": model.state_dict(), "seed": seed,
                    "input_size": X.shape[1], "hidden": HIDDEN, "epochs": EPOCHS,
                    "features": FEATURES, "numeric": numeric, "categories": cat_names,
                    "train_loss_history": history},
                   OUT / "weights" / f"chromenhancer_dnn_seed{seed}.pt")

        scores = predict(model, X, device)
        prediction_frames.append(pd.DataFrame({
            ID: df[ID].to_numpy(), SPLIT: df[SPLIT].to_numpy(), LABEL: y,
            "score": scores, "seed": seed}))

        valid = df[SPLIT].to_numpy() == "valid"
        test = df[SPLIT].to_numpy() == "test"
        row = {"model": MODEL_NAME, "seed": seed, "train_samples": int(len(idx)),
               "train_positives": int(sub_y.sum()), "input_dim": int(X.shape[1]),
               "valid_auroc_full": float(roc_auc_score(y[valid], scores[valid])),
               "valid_auprc_full": float(average_precision_score(y[valid], scores[valid])),
               "test_rows": int(test.sum()), "test_positives": int(y[test].sum()),
               "final_train_loss": history[-1]}
        row.update(metrics(y[test], scores[test], samples))
        per_seed.append(row)

    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(OUT / "per_seed.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        OUT / "per_locus_predictions.parquet", index=False)

    parameters = int(sum(p.numel() for p in model.parameters()))
    summary = {"model": MODEL_NAME, "seeds": list(SEEDS), "seed_count": len(SEEDS),
               "input_dim": int(X.shape[1]), "n_features": len(FEATURES),
               "train_samples_per_seed": int(per_seed_df["train_samples"].iloc[0]),
               "parameters": parameters, "trainable_parameters": parameters,
               "evaluation": "1:1 resampling, 100 repeats, seed 99999999, shared across seeds"}
    summary.update(summarize(per_seed_df))
    pd.DataFrame([summary]).to_csv(OUT / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
