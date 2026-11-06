#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from modeling_common import (
    write_parquet,
    write_tsv,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import shutil
import numpy as np
import pandas as pd
import gc
import math
import time
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# --------------------------------------------------------------------------
# step 13: Build the global ranker training table
# --------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELING = ROOT

M00_DIR = MODELING / "output" / "00_build_canonical_data"
M00_TABLE = M00_DIR / "00_canonical_herv_mef_table.parquet"

M08_DIR = MODELING / "output" / "08_grid_search_biological_prior"
M08_FOLD_MANIFEST = M08_DIR / "08_train_fold_manifest.tsv"

M12_DIR = MODELING / "output" / "12_export_bio_refinement_predictions"
M12_TRAIN_OOF = M12_DIR / "12_train_oof_representation_predictions.parquet"
M12_VALID_FULL = M12_DIR / "12_valid_full_train_representation_predictions.parquet"

M12_COLUMNS = [
    "herv_internal_id",
    "split",
    "score_ccre_enhancer",
    "score_serve_gtex_support",
    "score_sequence_regulatory",
    "score_final_representation",
    "prediction_origin",
    "is_fantom5_enhancer",
]

OUT_DIR = MODELING / "output" / "13_build_global_candidate_training_table"
OUT_TRAIN = OUT_DIR / "13_global_candidate_train.parquet"
OUT_VALID = OUT_DIR / "13_global_candidate_valid.parquet"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
ID_COL = "herv_internal_id"
SPLIT_COL = "split"
LABEL_COL = "is_fantom5_enhancer"
ROW_COL = "canonical_split_row_index"
FOLD_COL = "fold_id"

SCORE_CCRE = "score_ccre_enhancer"
SCORE_SERVE = "score_serve_gtex_support"
SCORE_SEQUENCE = "score_sequence_regulatory"
SCORE_FINAL = "score_final_representation"
SCORE_COLUMNS = [SCORE_CCRE, SCORE_SERVE, SCORE_SEQUENCE, SCORE_FINAL]
RANK_COLUMNS = [f"rank_pct_{c}" for c in SCORE_COLUMNS]
CALIBRATION_DELTA = "calibration_delta"

REPRESENTATION_NUMERIC = [*SCORE_COLUMNS, CALIBRATION_DELTA, *RANK_COLUMNS]
REPEAT_GENOMIC_NUMERIC = [
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
]
GLOBAL_CONTEXT_NUMERIC = [
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
NUMERIC_FEATURES = [
    *REPRESENTATION_NUMERIC,
    *REPEAT_GENOMIC_NUMERIC,
    *GLOBAL_CONTEXT_NUMERIC,
]
CATEGORICAL_FEATURES = ["repClass", "repFamily", "nearest_gene_type"]
SOURCE_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

POOL_COLUMNS = [
    "neg_pool_random", "neg_pool_evidence_rich",
    "neg_pool_hard", "neg_pool_boundary",
]

HARD_NEGATIVE_QUANTILE = 0.90
BOUNDARY_LOWER = 0.980
BOUNDARY_UPPER = 0.995

TRAIN_ORIGIN = "train_oof"
VALID_ORIGIN = "valid_full_train"

M00_CONTEXT_COLUMNS = [
    ID_COL, SPLIT_COL, LABEL_COL, "repName", "repClass", "repFamily",
    "repeat_divergence", "length", "gc_content", "n_content", "dist_to_tss",
    "nearest_gene_type", "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR",
    "is_GYPSY",
    "is_ccre_ca_overlap",
    "is_ccre_enhancer_like",
    "is_mpra_active_supported",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]

DIRECT_LABELS = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
M08_MANIFEST_COLUMNS = [ID_COL, "row_index", FOLD_COL]

OUTPUT_COLUMNS = [
    ID_COL, SPLIT_COL, ROW_COL, FOLD_COL, LABEL_COL,
    *SOURCE_FEATURES,
    *POOL_COLUMNS, "negative_pool_defined",
    "prediction_origin",
    "deep_model_fit_excluded_row", "calibrator_fit_excluded_row",
]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def rank_percentile(values):
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_m00_development_context() -> pd.DataFrame:
    frame = pd.read_parquet(
        M00_TABLE,
        columns=M00_CONTEXT_COLUMNS,
        filters=[(SPLIT_COL, "in", ["train", "valid"])],
        engine="pyarrow",
    )
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame[SPLIT_COL] = frame[SPLIT_COL].astype(str)

    for column in [
        LABEL_COL,
        "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
        "is_ccre_ca_overlap",
        *DIRECT_LABELS,
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.int8
        )

    for column in [
        "repeat_divergence", "length", "gc_content", "n_content",
        "dist_to_tss", "serve_gtex_tissue_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )

    for column in ["repName", "repClass", "repFamily", "nearest_gene_type"]:
        frame[column] = frame[column].astype(str)

    return frame.reset_index(drop=True)


def load_m08_fold_manifest() -> pd.DataFrame:
    frame = pd.read_csv(M08_FOLD_MANIFEST, sep="\t")
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame["row_index"] = pd.to_numeric(
        frame["row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    frame[FOLD_COL] = pd.to_numeric(
        frame[FOLD_COL], errors="raise"
    ).to_numpy(dtype=np.int64)
    return frame[M08_MANIFEST_COLUMNS].reset_index(drop=True)


def load_m12_prediction(path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=M12_COLUMNS)
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame[SPLIT_COL] = frame[SPLIT_COL].astype(str)

    frame[LABEL_COL] = pd.to_numeric(frame[LABEL_COL], errors="raise").to_numpy(
        dtype=np.int8
    )

    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )

    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def align_and_build(canonical_dev: pd.DataFrame,
                    prediction: pd.DataFrame,
                    fold_manifest: pd.DataFrame,
                    split: str) -> pd.DataFrame:
    context = canonical_dev.loc[canonical_dev[SPLIT_COL] == split].copy()
    context[ID_COL] = context[ID_COL].astype(str)
    pred = prediction.copy()

    left = pred[[
        ID_COL, SPLIT_COL, LABEL_COL, *SCORE_COLUMNS, "prediction_origin",
    ]].copy()

    left = left.merge(
        fold_manifest, on=ID_COL, how="left", validate="one_to_one", sort=False,
    )
    left["row_index"] = left["row_index"].fillna(-1).to_numpy(dtype=np.int64)
    left[FOLD_COL] = left[FOLD_COL].fillna(-1).to_numpy(dtype=np.int64)
    missing_row_index = int((left["row_index"] < 0).sum())
    if missing_row_index:
        fill = np.arange(missing_row_index, dtype=np.int64)
        left.loc[left["row_index"] < 0, "row_index"] = fill

    right = context[M00_CONTEXT_COLUMNS].drop(columns=[SPLIT_COL]).copy()
    right = right.rename(columns={
        LABEL_COL: f"{LABEL_COL}__m00",
        **{c: f"{c}__m00" for c in DIRECT_LABELS},
    })

    merged = left.merge(
        right, on=ID_COL, how="left", validate="one_to_one", sort=False,
    )

    merged = merged.drop(columns=[f"{LABEL_COL}__m00"])
    for column in DIRECT_LABELS:
        merged[column] = pd.to_numeric(
            merged[f"{column}__m00"], errors="raise"
        ).to_numpy(dtype=np.int8)
        merged = merged.drop(columns=[f"{column}__m00"])

    is_train = merged[SPLIT_COL].to_numpy(dtype=str) == "train"
    merged["deep_model_fit_excluded_row"] = ~is_train
    merged["calibrator_fit_excluded_row"] = ~is_train

    merged[ROW_COL] = merged["row_index"].to_numpy(dtype=np.int64)
    for column in SCORE_COLUMNS:
        merged[f"rank_pct_{column}"] = rank_percentile(
            merged[column].to_numpy(dtype=np.float64)
        )
    merged[CALIBRATION_DELTA] = (
        merged[SCORE_FINAL].to_numpy(dtype=np.float64)
        - merged[SCORE_SEQUENCE].to_numpy(dtype=np.float64)
    )
    merged["log_length"] = np.log1p(
        merged["length"].to_numpy(dtype=np.float64)
    )
    merged["log_abs_dist_to_tss"] = np.log1p(
        np.abs(merged["dist_to_tss"].to_numpy(dtype=np.float64))
    )

    for column in NUMERIC_FEATURES:
        merged[column] = pd.to_numeric(
            merged[column], errors="raise"
        ).to_numpy(dtype=np.float64)

    for column in CATEGORICAL_FEATURES:
        merged[column] = merged[column].astype(str)

    return merged


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_train_negative_pools(train: pd.DataFrame):
    output = train.copy()
    labels = output[LABEL_COL].to_numpy(dtype=np.int8)
    negatives = labels == 0
    negative_count = int(negatives.sum())

    evidence_rich = (
        (output["is_ccre_ca_overlap"].to_numpy(dtype=np.float64) > 0)
        | (output["is_ccre_enhancer_like"].to_numpy(dtype=np.float64) > 0)
        | (output["is_serve_gtex_supported"].to_numpy(dtype=np.float64) > 0)
    )

    final_score = output[SCORE_FINAL].to_numpy(dtype=np.float64)
    hard_threshold = float(
        np.quantile(final_score[negatives], HARD_NEGATIVE_QUANTILE)
    )
    final_rank = output[f"rank_pct_{SCORE_FINAL}"].to_numpy(dtype=np.float64)

    output["neg_pool_random"] = negatives
    output["neg_pool_evidence_rich"] = negatives & evidence_rich
    output["neg_pool_hard"] = negatives & (final_score >= hard_threshold)
    output["neg_pool_boundary"] = (
        negatives
        & (final_rank >= BOUNDARY_LOWER)
        & (final_rank < BOUNDARY_UPPER)
    )
    output["negative_pool_defined"] = True

    for column in POOL_COLUMNS:
        output[column] = output[column].astype(bool)

    manifest = pd.DataFrame([
        {
            "pool": "random", "column": "neg_pool_random",
            "rows": int(output["neg_pool_random"].sum()),
            "rate_among_train_negatives": float(
                output["neg_pool_random"].sum() / negative_count
            ),
            "definition": "all train FANTOM5 negatives",
        },
        {
            "pool": "evidence_rich", "column": "neg_pool_evidence_rich",
            "rows": int(output["neg_pool_evidence_rich"].sum()),
            "rate_among_train_negatives": float(
                output["neg_pool_evidence_rich"].sum() / negative_count
            ),
            "definition": (
                "negative AND (is_ccre_ca_overlap>0 OR "
                "is_ccre_enhancer_like>0 OR is_serve_gtex_supported>0)"
            ),
        },
        {
            "pool": "hard", "column": "neg_pool_hard",
            "rows": int(output["neg_pool_hard"].sum()),
            "rate_among_train_negatives": float(
                output["neg_pool_hard"].sum() / negative_count
            ),
            "definition": (
                f"negative AND score_final_representation >= "
                f"train-negative P{int(HARD_NEGATIVE_QUANTILE*100)} "
                f"(threshold={hard_threshold:.6f})"
            ),
        },
        {
            "pool": "boundary", "column": "neg_pool_boundary",
            "rows": int(output["neg_pool_boundary"].sum()),
            "rate_among_train_negatives": float(
                output["neg_pool_boundary"].sum() / negative_count
            ),
            "definition": (
                f"negative AND rank_pct_score_final_representation in "
                f"[{BOUNDARY_LOWER},{BOUNDARY_UPPER})"
            ),
        },
    ])

    overlap_rows = []
    for left in POOL_COLUMNS:
        left_mask = output[left].to_numpy(dtype=bool)
        left_n = int(left_mask.sum())
        for right in POOL_COLUMNS:
            right_mask = output[right].to_numpy(dtype=bool)
            intersection = int((left_mask & right_mask).sum())
            overlap_rows.append({
                "pool_a": left,
                "pool_b": right,
                "intersection_rows": intersection,
                "fraction_of_pool_a": (
                    float(intersection / left_n) if left_n else float("nan")
                ),
            })
    return output, manifest, pd.DataFrame(overlap_rows)


def build_valid_pool_schema(valid: pd.DataFrame) -> pd.DataFrame:
    output = valid.copy()
    for column in POOL_COLUMNS:
        output[column] = False
    output["negative_pool_defined"] = False
    return output


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def finalize_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    output = frame[OUTPUT_COLUMNS].copy()

    output[LABEL_COL] = pd.to_numeric(
        output[LABEL_COL], errors="raise"
    ).to_numpy(dtype=np.int8)

    for column in NUMERIC_FEATURES:
        output[column] = pd.to_numeric(
            output[column], errors="raise"
        ).to_numpy(dtype=np.float64)

    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].astype(str)

    return output.reset_index(drop=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def step_13() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical_dev = load_m00_development_context()
    fold_manifest = load_m08_fold_manifest()
    train_pred = load_m12_prediction(M12_TRAIN_OOF, "train")
    valid_pred = load_m12_prediction(M12_VALID_FULL, "valid")

    train = align_and_build(canonical_dev, train_pred, fold_manifest, "train")
    valid = align_and_build(canonical_dev, valid_pred, fold_manifest, "valid")

    train, pool_manifest, pool_overlap = build_train_negative_pools(train)
    valid = build_valid_pool_schema(valid)

    train_out = finalize_table(train, "train")
    valid_out = finalize_table(valid, "valid")

    atomic_write_parquet(train_out, OUT_TRAIN)
    atomic_write_parquet(valid_out, OUT_VALID)


# --------------------------------------------------------------------------
# step 14: Train the global candidate ranker
# --------------------------------------------------------------------------



ALLOWED_BATCH_SIZES = (16, 32, 64, 128)
PHYSICAL_GPU = "0"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
MODELING__14 = Path(__file__).resolve().parent
OUT_DIR__14 = MODELING__14 / "output" / "14_train_global_candidate_ranker"
COMMON_DIR = OUT_DIR__14 / "14_common"
RUNS_DIR = OUT_DIR__14 / "14_runs"

M13_DIR = MODELING__14 / "output" / "13_build_global_candidate_training_table"
M13_TRAIN = M13_DIR / "13_global_candidate_train.parquet"
M13_VALID = M13_DIR / "13_global_candidate_valid.parquet"

OUT_PREPROCESS = COMMON_DIR / "14_gcr_preprocessor.joblib"
OUT_VALID_SHARED_SAMPLES = COMMON_DIR / "14_valid_random1to1_samples.npz"

OUT_CANDIDATE_REGISTRY = OUT_DIR__14 / "14_candidate_checkpoint_registry.tsv"
OUT_TRAINING_HISTORY = OUT_DIR__14 / "14_training_history.tsv"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SEED = 99999999
MAX_EPOCHS = 30
DROPOUT = 0.20
LR = 8e-4
WEIGHT_DECAY = 1e-4
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 1.0
PREDICT_BATCH_SIZE = 8192
RANDOM_1TO1_REPEATS = 100
RANDOM_1TO1_SEED = 99999999
TOPK_NAMES = {0.005: "top0_5", 0.01: "top1", 0.05: "top5"}

ID_COL__14 = "herv_internal_id"
SPLIT_COL__14 = "split"
ROW_COL__14 = "canonical_split_row_index"
LABEL_COL__14 = "is_fantom5_enhancer"

NUMERIC_FEATURES__14 = [
    "score_ccre_enhancer", "score_serve_gtex_support",
    "score_sequence_regulatory", "score_final_representation",
    "calibration_delta",
    "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
    "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
CATEGORICAL_FEATURES__14 = ["repClass", "repFamily", "nearest_gene_type"]
SOURCE_FEATURES__14 = [*NUMERIC_FEATURES__14, *CATEGORICAL_FEATURES__14]

POOL_COLUMNS__14 = {
    "random": "neg_pool_random",
    "evidence_rich": "neg_pool_evidence_rich",
    "hard": "neg_pool_hard",
    "boundary": "neg_pool_boundary",
}

CURRICULUM = [
    {
        "stage": 1,
        "name": "random_only",
        "epochs": 6,
        "ratios": {"random": 1.0},
    },
    {
        "stage": 2,
        "name": "random_plus_evidence_rich",
        "epochs": 8,
        "ratios": {"random": 1.0, "evidence_rich": 0.5},
    },
    {
        "stage": 3,
        "name": "soft_hard_negative",
        "epochs": 8,
        "ratios": {"random": 1.0, "evidence_rich": 0.5, "hard": 0.1},
    },
    {
        "stage": 4,
        "name": "soft_boundary_negative",
        "epochs": 8,
        "ratios": {"random": 1.0, "evidence_rich": 0.5, "boundary": 0.1},
    },
]

TOPK = (0.005, 0.01, 0.05)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def atomic_write_tsv__14(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def run_dir(bs):
    return RUNS_DIR / f"bs{bs}_seed{SEED}"


def checkpoint_dir(bs):
    return run_dir(bs) / "checkpoints"


def history_path(bs):
    return run_dir(bs) / f"14_gcr_bs{bs}_training_history.tsv"


def manifest_path(bs):
    return run_dir(bs) / f"14_gcr_bs{bs}_checkpoint_manifest.tsv"


def epoch_ckpt_path(epoch, bs):
    return checkpoint_dir(bs) / f"14_gcr_bs{bs}_epoch_{epoch:02d}.pt"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_m13_tables():
    return pd.read_parquet(M13_TRAIN), pd.read_parquet(M13_VALID)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore",
                             sparse=False, dtype=np.float32)


def create_or_load_common(train, valid):
    COMMON_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    encoder = make_onehot_encoder()
    scaler.fit(train[NUMERIC_FEATURES__14].to_numpy(dtype=np.float64))
    encoder.fit(train[CATEGORICAL_FEATURES__14].astype(str))

    preprocess = {
        "scaler": scaler,
        "encoder": encoder,
        "numeric_features": NUMERIC_FEATURES__14,
        "categorical_features": CATEGORICAL_FEATURES__14,
        "source_features": SOURCE_FEATURES__14,
        "fit_scope": "M13 train only",
    }

    tmp_jl = OUT_PREPROCESS.with_name(
        OUT_PREPROCESS.name + f".tmp.{os.getpid()}"
    )
    joblib.dump(preprocess, tmp_jl)
    os.replace(tmp_jl, OUT_PREPROCESS)

    valid_labels = valid[LABEL_COL__14].to_numpy(dtype=np.int8)
    sample_matrix = build_1to1_samples(
        valid_labels,
        repeats=RANDOM_1TO1_REPEATS,
        seed=RANDOM_1TO1_SEED,
    )

    tmp_npz = OUT_VALID_SHARED_SAMPLES.with_name(
        OUT_VALID_SHARED_SAMPLES.name + f".tmp.{os.getpid()}.npz"
    )
    np.savez_compressed(tmp_npz, indices=sample_matrix)
    os.replace(tmp_npz, OUT_VALID_SHARED_SAMPLES)

    return preprocess, sample_matrix


def transform_table(frame, preprocess):
    scaler = preprocess["scaler"]
    encoder = preprocess["encoder"]
    numeric = scaler.transform(
        frame[NUMERIC_FEATURES__14].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    categorical = encoder.transform(
        frame[CATEGORICAL_FEATURES__14].astype(str)
    ).astype(np.float32)
    return np.concatenate([numeric, categorical],
                          axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_1to1_samples(labels, repeats, seed):
    labels = np.asarray(labels, dtype=np.int8)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    sample_matrix = np.empty(
        (repeats, len(positives) * 2), dtype=np.int64
    )
    for repeat in range(repeats):
        sampled_negatives = rng.choice(
            negatives, size=len(positives), replace=False
        )
        sample_matrix[repeat] = np.concatenate(
            [positives, sampled_negatives]
        )
    return sample_matrix


def eval_1to1(labels, scores, samples):
    auprc_values = []
    auroc_values = []
    for indices in samples:
        y = np.asarray(labels)[indices]
        s = np.asarray(scores)[indices]
        auprc_values.append(float(average_precision_score(y, s)))
        auroc_values.append(float(roc_auc_score(y, s)))
    return float(np.mean(auprc_values)), float(np.mean(auroc_values))


def topk_hits_fe(labels, scores, fracs=TOPK, row_order=None):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.asarray(row_order, dtype=np.int64), -scores))
    prevalence = float(labels.mean())
    result = {}
    for fraction in fracs:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        result[f"top{fraction}_hits"] = hits
        result[f"top{fraction}_fe"] = float(hits / (k * prevalence))
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class GCRMLP(nn.Module):
    def __init__(self, input_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating = torch.pow(
            torch.clamp(1.0 - p_t, min=0.0, max=1.0), self.gamma
        )
        return torch.mean(alpha_t * modulating * bce)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def stage_for_epoch(epoch):
    cursor = 0
    for stage in CURRICULUM:
        start = cursor + 1
        end = cursor + int(stage["epochs"])
        if start <= epoch <= end:
            return {
                "stage": int(stage["stage"]),
                "name": str(stage["name"]),
                "epoch_in_stage": epoch - cursor,
                "ratios": dict(stage["ratios"]),
            }
        cursor = end
    raise ValueError(f"epoch {epoch} out of curriculum range")


def make_curriculum_indices(pools, ratios, seed):
    rng = np.random.default_rng(seed)
    positive = pools["positive"]
    n_positive = len(positive)
    pieces = [positive.copy()]
    for pool_name, ratio in ratios.items():
        count = int(round(n_positive * float(ratio)))
        if count <= 0:
            continue
        pieces.append(
            rng.choice(pools[pool_name], size=count, replace=False).astype(
                np.int64
            )
        )
    indices = np.concatenate(pieces).astype(np.int64)
    rng.shuffle(indices)
    return indices


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_train_pools(train):
    labels = train[LABEL_COL__14].to_numpy(dtype=np.int8)
    pools = {"positive": np.flatnonzero(labels == 1).astype(np.int64)}
    negatives = labels == 0
    for pool_name, column in POOL_COLUMNS__14.items():
        flags = train[column].astype(bool).to_numpy() & negatives
        pools[pool_name] = np.flatnonzero(flags).astype(np.int64)
    return pools


def make_batch_index_list(indices, batch_size):
    indices = np.asarray(indices, dtype=np.int64)
    batches = [indices[s:s + batch_size]
               for s in range(0, len(indices), batch_size)]
    if len(batches[-1]) == 1:
        previous = batches[-2]
        singleton = batches[-1]
        batches[-2] = previous[:-1]
        batches[-1] = np.concatenate([previous[-1:], singleton])
    return batches


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, criterion,
                    train_x, train_y, indices, device, batch_size):
    model.train()
    batches = make_batch_index_list(indices, batch_size)
    loss_sum, sample_sum = 0.0, 0
    for step, batch_indices in enumerate(batches, start=1):
        x = torch.from_numpy(train_x[batch_indices]).to(
            device, non_blocking=True)
        y = torch.from_numpy(
            train_y[batch_indices].astype(np.float32, copy=False)
        ).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_n = int(len(batch_indices))
        loss_sum += float(loss.detach().cpu().item()) * batch_n
        sample_sum += batch_n

    return {
        "samples": sample_sum,
        "steps": len(batches),
        "mean_loss": float(loss_sum / sample_sum),
    }


@torch.inference_mode()
def predict_scores(model, features, device):
    model.eval()
    parts = []
    for start in range(0, len(features), PREDICT_BATCH_SIZE):
        x = torch.from_numpy(
            features[start:start + PREDICT_BATCH_SIZE]
        ).to(device, non_blocking=True)
        parts.append(torch.sigmoid(model(x)).detach().float().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def gold_metrics(valid, scores, shared_samples):
    labels = valid[LABEL_COL__14].to_numpy(dtype=np.int8)
    row_order = valid[ROW_COL__14].to_numpy(dtype=np.int64)

    auprc, auroc = eval_1to1(labels, scores, shared_samples)

    result = {
        "random1to1_auprc_mean": float(auprc),
        "random1to1_auroc_mean": float(auroc),
        "full_global_auprc_computed": False,
        "full_global_auroc_computed": False,
    }
    topk = topk_hits_fe(labels, scores, fracs=TOPK, row_order=row_order)
    for frac, name in TOPK_NAMES.items():
        result[f"{name}_hits"] = int(topk[f"top{frac}_hits"])
        result[f"{name}_fe"] = float(topk[f"top{frac}_fe"])
    return result


# ---------------------------------------------------------------------------
# checkpoint / recovery
# ---------------------------------------------------------------------------
def save_epoch_checkpoint(epoch, model, metric, bs):
    path = epoch_ckpt_path(epoch, bs)
    payload = {
        "batch_size": bs,
        "epoch": epoch,
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "validation_gold_metrics_diagnostic_only": metric,
    }
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return {
        "batch_size": bs,
        "epoch": epoch,
        "path": str(path),
        "eligible_for_M15_selection": True,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def arm_complete(bs):
    path = history_path(bs)
    if not path.is_file():
        return False
    frame = pd.read_csv(path, sep="\t")
    return int(frame["epoch"].max()) >= MAX_EPOCHS


def finalize_if_all_arms_complete():
    if not all(arm_complete(bs) for bs in ALLOWED_BATCH_SIZES):
        return False

    histories, registries, arm_rows = [], [], []
    for bs in ALLOWED_BATCH_SIZES:
        h = pd.read_csv(history_path(bs), sep="\t")
        r = pd.read_csv(manifest_path(bs), sep="\t")
        histories.append(h)
        registries.append(r)
        arm_rows.append({
            "batch_size": bs,
            "status": "PASSED",
            "epochs": MAX_EPOCHS,
            "candidate_count": MAX_EPOCHS,
        })

    all_registry = pd.concat(registries, ignore_index=True).sort_values(
        ["batch_size", "epoch"], kind="mergesort")
    all_history = pd.concat(histories, ignore_index=True).sort_values(
        ["batch_size", "epoch"], kind="mergesort")

    atomic_write_tsv__14(all_registry, OUT_CANDIDATE_REGISTRY)
    atomic_write_tsv__14(all_history, OUT_TRAINING_HISTORY)


    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_arm(bs, train, valid, preprocess, shared_samples):

    OUT_DIR__14.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir(bs).mkdir(parents=True, exist_ok=True)
    checkpoint_dir(bs).mkdir(parents=True, exist_ok=True)


    train_x = transform_table(train, preprocess)
    valid_x = transform_table(valid, preprocess)
    train_y = train[LABEL_COL__14].to_numpy(dtype=np.int8)

    device = torch.device("cuda:0")
    seed_everything(SEED)

    model = GCRMLP(input_dim=train_x.shape[1], dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    pools = build_train_pools(train)
    start_epoch, history_rows, checkpoint_rows = 1, [], []


    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        stage = stage_for_epoch(epoch)
        stage_id = int(stage["stage"])
        stage_name = stage["name"]
        ratios = stage["ratios"]

        indices = make_curriculum_indices(pools, ratios, seed=SEED + epoch)

        train_result = train_one_epoch(
            model, optimizer, criterion, train_x, train_y, indices, device, bs)
        valid_scores = predict_scores(model, valid_x, device)
        metric = gold_metrics(valid, valid_scores, shared_samples)
        epoch_seconds = time.perf_counter() - epoch_started

        history_row = {
            "batch_size": bs,
            "physical_gpu": PHYSICAL_GPU,
            "epoch": epoch,
            "stage": stage_id,
            "stage_name": stage_name,
            "train_samples": train_result["samples"],
            "train_steps": train_result["steps"],
            "train_loss": train_result["mean_loss"],
            "top0_5_hits": metric["top0_5_hits"],
            "top0_5_fe": metric["top0_5_fe"],
            "top1_hits": metric["top1_hits"],
            "top1_fe": metric["top1_fe"],
            "top5_hits": metric["top5_hits"],
            "top5_fe": metric["top5_fe"],
            "random1to1_auprc_mean": metric["random1to1_auprc_mean"],
            "random1to1_auroc_mean": metric["random1to1_auroc_mean"],
            "full_global_auprc_computed": False,
            "full_global_auroc_computed": False,
            "epoch_seconds": epoch_seconds,
        }
        checkpoint_row = save_epoch_checkpoint(epoch, model, metric, bs)

        history_rows.append(history_row)
        checkpoint_rows.append(checkpoint_row)
        atomic_write_tsv__14(pd.DataFrame(history_rows), history_path(bs))
        atomic_write_tsv__14(pd.DataFrame(checkpoint_rows), manifest_path(bs))


        del indices, valid_scores
        gc.collect()
        torch.cuda.empty_cache()


def step_14():
    train, valid = load_m13_tables()
    preprocess, shared_samples = create_or_load_common(train, valid)
    for bs in ALLOWED_BATCH_SIZES:
        run_arm(bs, train, valid, preprocess, shared_samples)
    finalize_if_all_arms_complete()


# --------------------------------------------------------------------------
# step 15: Audit and select the global candidate ranker
# --------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
MODELING__15 = Path(__file__).resolve().parent
OUT_DIR__15 = MODELING__15 / "output" / "15_audit_select_global_candidate_ranker"
FROZEN_DIR = OUT_DIR__15 / "15_frozen"

M13_DIR__15 = MODELING__15 / "output" / "13_build_global_candidate_training_table"
M13_VALID__15 = M13_DIR__15 / "13_global_candidate_valid.parquet"

M14_DIR = MODELING__15 / "output" / "14_train_global_candidate_ranker"
M14_COMMON_DIR = M14_DIR / "14_common"
M14_REGISTRY = M14_DIR / "14_candidate_checkpoint_registry.tsv"
M14_HISTORY = M14_DIR / "14_training_history.tsv"
M14_PREPROCESS = M14_COMMON_DIR / "14_gcr_preprocessor.joblib"
M14_VALID_SAMPLES = M14_COMMON_DIR / "14_valid_random1to1_samples.npz"


OUT_FROZEN_MODEL = FROZEN_DIR / "15_frozen_global_candidate_ranker.pt"
OUT_FROZEN_PREPROCESS = FROZEN_DIR / "15_frozen_gcr_preprocessor.joblib"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
EXPECTED_BATCH_SIZES = (16, 32, 64, 128)
EXPECTED_EPOCHS_PER_ARM = 30
EXPECTED_CANDIDATES = len(EXPECTED_BATCH_SIZES) * EXPECTED_EPOCHS_PER_ARM

ID_COL__15 = "herv_internal_id"
SPLIT_COL__15 = "split"
ROW_COL__15 = "canonical_split_row_index"
LABEL_COL__15 = "is_fantom5_enhancer"

NUMERIC_FEATURES__15 = [
    "score_ccre_enhancer", "score_serve_gtex_support",
    "score_sequence_regulatory", "score_final_representation",
    "calibration_delta",
    "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
    "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
CATEGORICAL_FEATURES__15 = ["repClass", "repFamily", "nearest_gene_type"]
SOURCE_FEATURES__15 = [*NUMERIC_FEATURES__15, *CATEGORICAL_FEATURES__15]

RANDOM_1TO1_REPEATS__15 = 100
RANDOM_1TO1_SEED__15 = 99999999
TOPK__15 = (0.005, 0.01, 0.05)
TOPK_NAMES__15 = {0.005: "top0_5", 0.01: "top1", 0.05: "top5"}
PREDICT_BATCH_SIZE__15 = 8192
DROPOUT__15 = 0.20


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def atomic_write_tsv__15(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet__15(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def eval_1to1__15(labels, scores, samples):
    auprc_values = []
    auroc_values = []
    for indices in samples:
        y = np.asarray(labels)[indices]
        s = np.asarray(scores)[indices]
        auprc_values.append(float(average_precision_score(y, s)))
        auroc_values.append(float(roc_auc_score(y, s)))
    return float(np.mean(auprc_values)), float(np.mean(auroc_values))


def topk_hits_fe__15(labels, scores, fracs=TOPK__15, row_order=None):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.asarray(row_order, dtype=np.int64), -scores))
    prevalence = float(labels.mean())
    result = {}
    for fraction in fracs:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        result[f"top{fraction}_hits"] = hits
        result[f"top{fraction}_fe"] = float(hits / (k * prevalence))
    return result

SELECTION_RULE = [
    "Top1% Hits descending",
    "Top5% Hits descending",
    "random-negative 1:1 AUPRC mean descending",
    "Top0.5% Hits descending",
    "random-negative 1:1 AUROC mean descending",
    "earlier epoch ascending on exact five-metric tie",
    "smaller batch size ascending on the same exact tie",
]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class GCRMLP__15(nn.Module):
    def __init__(self, input_dim, dropout=DROPOUT__15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_candidate_table():
    registry = pd.read_csv(M14_REGISTRY, sep="\t")
    history = pd.read_csv(M14_HISTORY, sep="\t")

    registry["batch_size"] = registry["batch_size"].astype(int)
    registry["epoch"] = registry["epoch"].astype(int)
    history["batch_size"] = history["batch_size"].astype(int)
    history["epoch"] = history["epoch"].astype(int)

    if "candidate_id" not in registry.columns:
        registry["candidate_id"] = [
            f"GCR_R2_bs{bs}_epoch{ep:02d}"
            for bs, ep in zip(registry["batch_size"], registry["epoch"])
        ]

    metric_cols = [c for c in [
        "top0_5_hits", "top0_5_fe",
        "top1_hits", "top1_fe",
        "top5_hits", "top5_fe",
        "random1to1_auprc_mean",
        "random1to1_auroc_mean",
    ] if c in history.columns]

    history_slim = history[
        ["batch_size", "epoch", "stage", "stage_name", *metric_cols]
    ]

    merged = registry.merge(
        history_slim,
        on=["batch_size", "epoch"],
        how="inner",
        validate="one_to_one",
    )

    return merged.sort_values(
        ["batch_size", "epoch"], kind="mergesort"
    ).reset_index(drop=True)


def load_and_transform_validation(valid):
    preprocess = joblib.load(M14_PREPROCESS)
    numeric = preprocess["scaler"].transform(
        valid[NUMERIC_FEATURES__15].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    categorical = preprocess["encoder"].transform(
        valid[CATEGORICAL_FEATURES__15].astype(str)
    ).astype(np.float32)
    features = np.concatenate([numeric, categorical], axis=1).astype(
        np.float32, copy=False)
    samples = np.load(
        M14_VALID_SAMPLES, allow_pickle=False
    )["indices"].astype(np.int64)
    return preprocess, features, samples


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@torch.inference_mode()
def predict_checkpoint(path, valid_x, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    if "training_invariants" in ckpt:
        input_dim = int(ckpt["training_invariants"]["model"]["input_dim"])
    else:
        input_dim = int(state["net.0.weight"].shape[1])


    model = GCRMLP__15(input_dim=input_dim, dropout=DROPOUT__15).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    parts = []
    for start in range(0, len(valid_x), PREDICT_BATCH_SIZE__15):
        x = torch.from_numpy(valid_x[start:start + PREDICT_BATCH_SIZE__15]).to(
            device, non_blocking=True)
        parts.append(torch.sigmoid(model(x)).detach().float().cpu().numpy())
    scores = np.concatenate(parts).astype(np.float64)
    diagnostics = ckpt.get("validation_gold_metrics_diagnostic_only")

    del model, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return scores, diagnostics


def compute_gold_metrics(valid, scores, shared_samples):
    labels = valid[LABEL_COL__15].to_numpy(dtype=np.int8)
    row_order = valid[ROW_COL__15].to_numpy(dtype=np.int64)

    auprc, auroc = eval_1to1__15(labels, scores, shared_samples)

    result = {
        "random1to1_auprc_mean": float(auprc),
        "random1to1_auroc_mean": float(auroc),
    }
    topk = topk_hits_fe__15(labels, scores, fracs=TOPK__15, row_order=row_order)
    for frac, name in TOPK_NAMES__15.items():
        result[f"{name}_hits"] = int(topk[f"top{frac}_hits"])
        result[f"{name}_fe"] = float(topk[f"top{frac}_fe"])
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def reinfer_all(candidates, valid, valid_x, shared_samples, device):
    rows = []
    for i, row in enumerate(candidates.itertuples(index=False), start=1):
        path = Path(str(row.path))
        scores, _ = predict_checkpoint(path, valid_x, device)
        m = compute_gold_metrics(valid, scores, shared_samples)
        rows.append({
            "candidate_id": str(row.candidate_id),
            "batch_size": int(row.batch_size),
            "epoch": int(row.epoch),
            "stage": int(row.stage),
            "stage_name": str(row.stage_name),
            **m,
            "checkpoint_path": str(path),
        })
        del scores
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def select_candidate(metrics_df):
    ranked = metrics_df.sort_values(
        ["top1_hits", "top5_hits", "random1to1_auprc_mean",
         "top0_5_hits", "random1to1_auroc_mean", "epoch", "batch_size"],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked["selected"] = ranked["selection_rank"].eq(1)
    return ranked, ranked.iloc[0].copy()


def build_stage_summary(metrics_df):
    rows = []
    for (bs, stage, name), group in metrics_df.groupby(
        ["batch_size", "stage", "stage_name"], sort=True
    ):
        best = group.loc[group["top1_hits"].idxmax()]
        rows.append({
            "batch_size": int(bs),
            "stage": int(stage),
            "stage_name": str(name),
            "candidate_count": len(group),
            "epoch_min": int(group["epoch"].min()),
            "epoch_max": int(group["epoch"].max()),
            "max_top0_5_hits": int(group["top0_5_hits"].max()),
            "max_top1_hits": int(group["top1_hits"].max()),
            "max_top5_hits": int(group["top5_hits"].max()),
            "max_1to1_auprc": float(group["random1to1_auprc_mean"].max()),
            "max_1to1_auroc": float(group["random1to1_auroc_mean"].max()),
            "top1_best_candidate_id": str(best["candidate_id"]),
            "top1_best_epoch": int(best["epoch"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def freeze_selected(selected, valid, valid_x, shared_samples, device):
    src = Path(str(selected["checkpoint_path"]))
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    scores, _ = predict_checkpoint(src, valid_x, device)

    labels = valid[LABEL_COL__15].to_numpy(dtype=np.int8)
    row_order = valid[ROW_COL__15].to_numpy(dtype=np.int64)
    ranking = np.lexsort((row_order, -scores))
    rank_position = np.empty(len(valid), dtype=np.int64)
    rank_position[ranking] = np.arange(1, len(valid) + 1, dtype=np.int64)


    repeat_rows = []
    for i, idx in enumerate(shared_samples):
        y = labels[idx]
        s = scores[idx]
        repeat_rows.append({
            "repeat": i, "seed": RANDOM_1TO1_SEED__15,
            "rows": len(idx), "positives": int(y.sum()),
            "negatives": int(len(idx) - y.sum()),
            "AUPRC": float(average_precision_score(y, s)),
            "AUROC": float(roc_auc_score(y, s)),
        })

    prev = float(labels.mean())
    topk_rows = []
    for frac in TOPK_NAMES__15:
        k = int(math.floor(len(labels) * frac))
        hits = int(labels[ranking[:k]].sum())
        topk_rows.append({
            "fraction": frac, "K": k, "Hits": hits,
            "FE": float(hits / (k * prev)),
        })

    invariants = {
        "model": {
            "type": "GCRMLP",
            "input_dim": int(ckpt["model_state_dict"]["net.0.weight"].shape[1]),
            "hidden_dims": [128, 64],
            "dropout": DROPOUT__15,
        },
        "source_features": SOURCE_FEATURES__15,
        "batch_size": int(selected["batch_size"]),
        "epoch": int(selected["epoch"]),
    }

    frozen = {
        "status": "FROZEN",
        "source_M14_candidate_id": str(selected["candidate_id"]),
        "source_M14_checkpoint_path": str(src),
        "selected_batch_size": int(selected["batch_size"]),
        "selected_epoch": int(selected["epoch"]),
        "selected_stage": int(selected["stage"]),
        "selected_stage_name": str(selected["stage_name"]),
        "model_state_dict": ckpt["model_state_dict"],
        "training_invariants": invariants,
        "source_features": SOURCE_FEATURES__15,
    }
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FROZEN_MODEL.with_name(
        OUT_FROZEN_MODEL.name + f".tmp.{os.getpid()}")
    torch.save(frozen, tmp)
    os.replace(tmp, OUT_FROZEN_MODEL)
    shutil.copy2(M14_PREPROCESS, OUT_FROZEN_PREPROCESS)

    del ckpt, scores
    gc.collect()
    torch.cuda.empty_cache()
    return frozen


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def step_15():

    if OUT_DIR__15.exists():
        shutil.rmtree(OUT_DIR__15)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)


    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    candidates = load_candidate_table()
    valid = pd.read_parquet(M13_VALID__15).reset_index(drop=True)
    _, valid_x, shared_samples = load_and_transform_validation(valid)


    all_metrics = reinfer_all(
        candidates, valid, valid_x, shared_samples, device)

    ranked, selected = select_candidate(all_metrics)

    freeze_selected(selected, valid, valid_x, shared_samples, device)


# --------------------------------------------------------------------------
# step 16: Train the global feature expansion probe
# --------------------------------------------------------------------------



ROUTES = ("R1_ROADMAP", "R2_MPRA", "R3_CCRE_SERVE", "R4_ALL_COMPACT")


MODELING__16 = Path(__file__).resolve().parent
M00_TABLE__16 = MODELING__16 / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M13_TRAIN__16 = MODELING__16 / "output/13_build_global_candidate_training_table/13_global_candidate_train.parquet"
M13_VALID__16 = MODELING__16 / "output/13_build_global_candidate_training_table/13_global_candidate_valid.parquet"
M14_SAMPLES = MODELING__16 / "output/14_train_global_candidate_ranker/14_common/14_valid_random1to1_samples.npz"
OUT, COMMON, RUNS = (MODELING__16 / "output/16_train_global_feature_expansion_probe",
                     MODELING__16 / "output/16_train_global_feature_expansion_probe/16_common",
                     MODELING__16 / "output/16_train_global_feature_expansion_probe/16_runs")
EXP_TRAIN, EXP_VALID = COMMON / "16_expanded_train.parquet", COMMON / "16_expanded_valid.parquet"

ID, SPLIT, ROW, LABEL = "herv_internal_id", "split", "canonical_split_row_index", "is_fantom5_enhancer"
BASE_NUMERIC = [
    "score_ccre_enhancer", "score_serve_gtex_support",
    "score_sequence_regulatory", "score_final_representation",
    "calibration_delta",
    "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
    "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss", "is_ERV1", "is_ERVK", "is_ERVL",
    "is_ERVL_MALR", "is_GYPSY", "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
CATEGORICAL = ["repClass", "repFamily", "nearest_gene_type"]
BASE_SOURCE = [*BASE_NUMERIC, *CATEGORICAL]
POOLS = {"random": "neg_pool_random", "evidence_rich": "neg_pool_evidence_rich",
         "hard": "neg_pool_hard", "boundary": "neg_pool_boundary"}
ACTIVE_MARKS = ("DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3")
REPRESSIVE_MARKS = ("H3K27me3", "H3K9me3")
ALL_MARKS = (*ACTIVE_MARKS, *REPRESSIVE_MARKS)

ROADMAP18 = [f"roadmap_{m}_{s}" for m in ALL_MARKS
             for s in ("sample_count", "max_overlap_bp", "total_overlap_bp")]
MPRA8 = ["mpra_overlap_count", "mpra_max_overlap_bp", "mpra_total_overlap_bp",
         "mpra_active_overlap_count", "mpra_active_max_overlap_bp",
         "mpra_active_total_overlap_bp", "is_mpra_overlap", "is_mpra_active_supported"]
CCRE6 = ["is_ccre_ca_overlap", "is_ccre_enhancer_like", "is_ccre_ctcf_overlap",
         "is_ccre_pls_overlap", "ccre_els_overlap_count", "ccre_els_total_overlap_bp"]
SERVE6 = ["serve_gtex_tissue_count", "serve_gtex_total_overlap_bp",
          "is_serve_gtex_supported", "serve_gtex_annotation_count",
          "serve_noncoding_overlap_count", "serve_noncoding_total_overlap_bp"]
RAW38 = [*ROADMAP18, *MPRA8, *CCRE6, *SERVE6]

ROADMAP9 = [
    "gctx_roadmap_active_mark_presence_count", "gctx_roadmap_repressive_mark_presence_count",
    "gctx_roadmap_active_sample_count_sum", "gctx_roadmap_repressive_sample_count_sum",
    "gctx_roadmap_active_max_overlap_bp_max", "gctx_roadmap_repressive_max_overlap_bp_max",
    "gctx_roadmap_active_total_overlap_bp_log1p", "gctx_roadmap_repressive_total_overlap_bp_log1p",
    "gctx_roadmap_active_minus_repressive_logbp",
]
MPRA6 = ["gctx_mpra_is_overlap", "gctx_mpra_is_active_supported",
         "gctx_mpra_total_overlap_bp_log1p", "gctx_mpra_active_total_overlap_bp_log1p",
         "gctx_mpra_active_count_fraction", "gctx_mpra_active_bp_fraction"]
CCRE_SERVE9 = [
    "gctx_ccre_is_ctcf_overlap", "gctx_ccre_is_pls_overlap",
    "gctx_ccre_els_overlap_count_log1p", "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_annotation_count_log1p", "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_overlap_count_log1p", "gctx_serve_noncoding_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]
ALL_COMPACT = [*ROADMAP9, *MPRA6, *CCRE_SERVE9]
EXTRAS = {"R1_ROADMAP": ROADMAP9, "R2_MPRA": MPRA6,
          "R3_CCRE_SERVE": CCRE_SERVE9, "R4_ALL_COMPACT": ALL_COMPACT}

SEED__16, BATCH, EPOCHS, LR__16, WD = 99999999, 64, 30, 8e-4, 1e-4
DROPOUT__16, ALPHA, GAMMA, PRED_BATCH = 0.20, 0.50, 1.0, 8192
CURRICULUM__16 = [
    (1, "random_only", 6, {"random": 1.0}),
    (2, "random_plus_evidence_rich", 8, {"random": 1.0, "evidence_rich": 0.5}),
    (3, "soft_hard_negative", 8, {"random": 1.0, "evidence_rich": 0.5, "hard": 0.1}),
    (4, "soft_boundary_negative", 8, {"random": 1.0, "evidence_rich": 0.5, "boundary": 0.1}),
]

def run_dir__16(route): return RUNS / f"{route}_bs64_seed{SEED__16}"
def hist_path(route): return run_dir__16(route) / f"16_{route}_training_history.tsv"
def reg_path(route): return run_dir__16(route) / f"16_{route}_checkpoint_manifest.tsv"
def pre_path(route): return run_dir__16(route) / f"16_{route}_preprocessor.joblib"
def ckpt_path(route, epoch): return run_dir__16(route) / "checkpoints" / f"16_{route}_epoch_{epoch:02d}.pt"


def write_atomic(df, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, **kwargs) if path.suffix == ".tsv" else df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)

def seed_all():
    np.random.seed(SEED__16); torch.manual_seed(SEED__16)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED__16)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def frac(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.divide(n, d, out=np.zeros_like(n), where=d > 0)

def derive(raw):
    out = pd.DataFrame(index=raw.index)
    A = lambda suffix: raw[[f"roadmap_{m}_{suffix}" for m in ACTIVE_MARKS]].to_numpy(float)
    R = lambda suffix: raw[[f"roadmap_{m}_{suffix}" for m in REPRESSIVE_MARKS]].to_numpy(float)
    AS, RS, AM, RM, AT, RT = A("sample_count"), R("sample_count"), A("max_overlap_bp"), R("max_overlap_bp"), A("total_overlap_bp"), R("total_overlap_bp")
    at, rt = AT.sum(1), RT.sum(1)
    vals = [
        (AS > 0).sum(1), (RS > 0).sum(1), AS.sum(1), RS.sum(1), AM.max(1), RM.max(1),
        np.log1p(at), np.log1p(rt), np.log1p(at) - np.log1p(rt),
        raw["is_mpra_overlap"].to_numpy(float), raw["is_mpra_active_supported"].to_numpy(float),
        np.log1p(raw["mpra_total_overlap_bp"].to_numpy(float)),
        np.log1p(raw["mpra_active_total_overlap_bp"].to_numpy(float)),
        frac(raw["mpra_active_overlap_count"], raw["mpra_overlap_count"]),
        frac(raw["mpra_active_total_overlap_bp"], raw["mpra_total_overlap_bp"]),
        raw["is_ccre_ctcf_overlap"].to_numpy(float), raw["is_ccre_pls_overlap"].to_numpy(float),
        np.log1p(raw["ccre_els_overlap_count"].to_numpy(float)),
        np.log1p(np.maximum(raw["ccre_els_total_overlap_bp"].to_numpy(float), 0)),
        np.log1p(raw["serve_gtex_annotation_count"].to_numpy(float)),
        np.log1p(raw["serve_gtex_total_overlap_bp"].to_numpy(float)),
        np.log1p(raw["serve_noncoding_overlap_count"].to_numpy(float)),
        np.log1p(raw["serve_noncoding_total_overlap_bp"].to_numpy(float)),
        frac(raw["serve_noncoding_total_overlap_bp"], raw["serve_gtex_total_overlap_bp"]),
    ]
    for name, value in zip(ALL_COMPACT, vals): out[name] = value
    return out

def build_common():
    COMMON.mkdir(parents=True, exist_ok=True)
    tr, va = pd.read_parquet(M13_TRAIN__16), pd.read_parquet(M13_VALID__16)
    m00 = pd.read_parquet(M00_TABLE__16, columns=[ID, SPLIT, *RAW38],
                           filters=[(SPLIT, "in", ["train", "valid"])])
    def attach(base, split):
        ev = m00.loc[m00[SPLIT] == split].copy()
        merged = base.merge(ev, on=[ID, SPLIT], how="left", validate="one_to_one",
                            sort=False, suffixes=("", "__m00"))
        raw = {f: pd.to_numeric(merged[f + "__m00" if f in BASE_SOURCE else f],
                                errors="raise").to_numpy(float) for f in RAW38}
        compact = derive(pd.DataFrame(raw, index=merged.index))
        merged[ALL_COMPACT] = compact[ALL_COMPACT]
        keep = [ID, SPLIT, ROW, LABEL, *BASE_SOURCE, *ALL_COMPACT, *POOLS.values(),
                "negative_pool_defined", "prediction_origin",
                "deep_model_fit_excluded_row", "calibrator_fit_excluded_row"]
        return merged[keep].copy()
    etr, eva = attach(tr, "train"), attach(va, "valid")
    write_atomic(etr, EXP_TRAIN); write_atomic(eva, EXP_VALID)
    return etr, eva

def encoder():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

def build_pre(tr, route):
    nums = [*BASE_NUMERIC, *EXTRAS[route]]
    obj = {"route": route, "numeric_features": nums, "categorical_features": CATEGORICAL,
           "scaler": StandardScaler().fit(tr[nums].to_numpy(float)),
           "encoder": encoder().fit(tr[CATEGORICAL].astype(str))}
    joblib.dump(obj, pre_path(route)); return obj

def transform(df, pre):
    a = pre["scaler"].transform(df[pre["numeric_features"]].to_numpy(float)).astype(np.float32)
    b = pre["encoder"].transform(df[CATEGORICAL].astype(str)).astype(np.float32)
    return np.concatenate([a, b], axis=1).astype(np.float32, copy=False)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT__16),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT__16),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def focal(logits, y):
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits); pt = y * p + (1 - y) * (1 - p)
    at = y * ALPHA + (1 - y) * (1 - ALPHA)
    return torch.mean(at * torch.pow(1 - pt, GAMMA) * bce)

def make_pools(tr):
    y = tr[LABEL].to_numpy(np.int8); neg = y == 0
    return {"positive": np.flatnonzero(y == 1).astype(np.int64),
            **{n: np.flatnonzero(tr[c].astype(bool).to_numpy() & neg).astype(np.int64)
               for n, c in POOLS.items()}}

def stage_for(epoch):
    cur = 0
    for sid, name, n, ratios in CURRICULUM__16:
        if cur < epoch <= cur + n: return sid, name, epoch - cur, ratios
        cur += n
    raise ValueError(f"epoch {epoch} out of range")

def epoch_indices(pools, ratios, epoch):
    rng = np.random.default_rng(SEED__16 + epoch); pos = pools["positive"]
    idx = np.concatenate([pos.copy(), *[
        rng.choice(pools[name], size=int(round(len(pos) * ratio)), replace=False)
        for name, ratio in ratios.items()]])
    rng.shuffle(idx); return idx.astype(np.int64)

def batches(idx):
    b = [idx[s:s + BATCH] for s in range(0, len(idx), BATCH)]
    if len(b[-1]) == 1:
        b[-2], b[-1] = b[-2][:-1], np.concatenate([b[-2][-1:], b[-1]])
    return b

def train_epoch(model, opt, x, y, idx, device):
    model.train(); total = n = 0
    bs = batches(idx)
    for step, bi in enumerate(bs, start=1):
        bx = torch.from_numpy(x[bi]).to(device)
        by = torch.from_numpy(y[bi].astype(np.float32, copy=False)).to(device)
        opt.zero_grad(set_to_none=True); loss = focal(model(bx), by)
        loss.backward(); opt.step()
        total += float(loss.detach().cpu()) * len(bi); n += len(bi)
    return total / n

@torch.inference_mode()
def predict(model, x, device):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(torch.from_numpy(x[s:s + PRED_BATCH]).to(device))).float().cpu().numpy()
        for s in range(0, len(x), PRED_BATCH)]).astype(float)

def gold_metrics__16(valid, scores, samples):
    y, ro = valid[LABEL].to_numpy(np.int8), valid[ROW].to_numpy(np.int64)
    ap, au = [], []
    for idx in samples:
        ap.append(float(average_precision_score(y[idx], scores[idx])))
        au.append(float(roc_auc_score(y[idx], scores[idx])))
    out = {"random1to1_auprc_mean": float(np.mean(ap)),
           "random1to1_auprc_std": float(np.std(ap, ddof=1)),
           "random1to1_auroc_mean": float(np.mean(au)),
           "random1to1_auroc_std": float(np.std(au, ddof=1))}
    order, prev = np.lexsort((ro, -scores)), float(y.mean())
    for f, name in [(0.005, "top0_5"), (0.01, "top1"), (0.05, "top5")]:
        k = int(math.floor(len(y) * f)); hits = int(y[order[:k]].sum())
        out |= {name + "_hits": hits, name + "_fe": float(hits / (k * prev))}
    return out

def run_route(route, tr, va):
    OUT.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    run_dir__16(route).mkdir(parents=True, exist_ok=True)
    (run_dir__16(route) / "checkpoints").mkdir(parents=True, exist_ok=True)

    pre = build_pre(tr, route)
    tx, vx = transform(tr, pre), transform(va, pre)
    y = tr[LABEL].to_numpy(np.int8)
    samples = np.load(M14_SAMPLES, allow_pickle=False)["indices"].astype(np.int64)
    device = torch.device("cuda:0"); seed_all()
    model = MLP(tx.shape[1]).to(device); opt = torch.optim.AdamW(model.parameters(), lr=LR__16, weight_decay=WD)
    pools = make_pools(tr); history, registry = [], []

    for epoch in range(1, EPOCHS + 1):
        et = time.perf_counter(); sid, sname, local, ratios = stage_for(epoch)
        idx = epoch_indices(pools, ratios, epoch)
        loss = train_epoch(model, opt, tx, y, idx, device)
        sc = predict(model, vx, device); m = gold_metrics__16(va, sc, samples)
        cid, cp = f"{route}_bs64_epoch_{epoch:02d}", ckpt_path(route, epoch)
        torch.save({"candidate_id": cid, "route": route, "batch_size": 64, "epoch": epoch,
                    "stage": sid, "stage_name": sname,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "validation_gold_metrics_diagnostic_only": m}, cp)
        history.append({
            "route": route, "batch_size": 64, "epoch": epoch, "stage": sid, "stage_name": sname,
            "local_epoch": local, "extra_feature_count": len(EXTRAS[route]),
            "source_feature_count": 26 + len(EXTRAS[route]), "transformed_dim": tx.shape[1],
            "train_samples": len(idx), "train_loss": loss,
            "top0_5_hits": m["top0_5_hits"], "top0_5_fe": m["top0_5_fe"],
            "top1_hits": m["top1_hits"], "top1_fe": m["top1_fe"],
            "top5_hits": m["top5_hits"], "top5_fe": m["top5_fe"],
            "random1to1_auprc_mean": m["random1to1_auprc_mean"],
            "random1to1_auprc_std": m["random1to1_auprc_std"],
            "random1to1_auroc_mean": m["random1to1_auroc_mean"],
            "random1to1_auroc_std": m["random1to1_auroc_std"],
            "epoch_seconds": time.perf_counter() - et})
        registry.append({"candidate_id": cid, "route": route, "batch_size": 64, "epoch": epoch,
                         "stage": sid, "stage_name": sname, "checkpoint_path": str(cp)})
        write_atomic(pd.DataFrame(history), hist_path(route), sep="\t", index=False)
        write_atomic(pd.DataFrame(registry), reg_path(route), sep="\t", index=False)
        del idx, sc; gc.collect(); torch.cuda.empty_cache()


def step_16():
    tr, va = build_common()
    for route in ROUTES:
        run_route(route, tr, va)


# --------------------------------------------------------------------------
# step 17: Select the final global ranker
# --------------------------------------------------------------------------



MODELING__17 = Path(__file__).resolve().parent
M13_VALID__17 = MODELING__17 / "output/13_build_global_candidate_training_table/13_global_candidate_valid.parquet"
M14_SAMPLES__17 = MODELING__17 / "output/14_train_global_candidate_ranker/14_common/14_valid_random1to1_samples.npz"
M15_DIR = MODELING__17 / "output/15_audit_select_global_candidate_ranker"
M15_MODEL = M15_DIR / "15_frozen/15_frozen_global_candidate_ranker.pt"
M15_PRE = M15_DIR / "15_frozen/15_frozen_gcr_preprocessor.joblib"
M16_RUNS = MODELING__17 / "output/16_train_global_feature_expansion_probe/16_runs"
M16_EXP_VALID = MODELING__17 / "output/16_train_global_feature_expansion_probe/16_common/16_expanded_valid.parquet"

OUT__17 = MODELING__17 / "output/17_audit_select_final_global_ranker"
FROZEN = OUT__17 / "17_frozen"

ID__17, SPLIT__17, ROW__17, LABEL__17 = "herv_internal_id", "split", "canonical_split_row_index", "is_fantom5_enhancer"
VALID_N = 72_048
REPEATS, SAMPLE_SEED = 100, 99999999
PRED_BATCH__17, DROPOUT__17 = 8192, 0.20
SEED16 = 99999999

ROUTES__17 = ("R1_ROADMAP", "R2_MPRA", "R3_CCRE_SERVE", "R4_ALL_COMPACT")

BASE_NUMERIC__17 = [
    "score_ccre_enhancer", "score_serve_gtex_support",
    "score_sequence_regulatory", "score_final_representation",
    "calibration_delta",
    "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
    "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
CATEGORICAL__17 = ["repClass", "repFamily", "nearest_gene_type"]
BASE_SOURCE__17 = [*BASE_NUMERIC__17, *CATEGORICAL__17]

ROADMAP9__17 = [
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
MPRA6__17 = [
    "gctx_mpra_is_overlap", "gctx_mpra_is_active_supported",
    "gctx_mpra_total_overlap_bp_log1p", "gctx_mpra_active_total_overlap_bp_log1p",
    "gctx_mpra_active_count_fraction", "gctx_mpra_active_bp_fraction",
]
CCRE_SERVE9__17 = [
    "gctx_ccre_is_ctcf_overlap", "gctx_ccre_is_pls_overlap",
    "gctx_ccre_els_overlap_count_log1p", "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_annotation_count_log1p", "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_overlap_count_log1p",
    "gctx_serve_noncoding_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]
ALL_COMPACT__17 = [*ROADMAP9__17, *MPRA6__17, *CCRE_SERVE9__17]
ROUTE_EXTRAS = {
    "R1_ROADMAP": ROADMAP9__17, "R2_MPRA": MPRA6__17,
    "R3_CCRE_SERVE": CCRE_SERVE9__17, "R4_ALL_COMPACT": ALL_COMPACT__17,
}






def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_m16_tables():
    regs, hists = [], []
    for route in ROUTES__17:
        rd = M16_RUNS / f"{route}_bs64_seed{SEED16}"
        regs.append(pd.read_csv(rd / f"16_{route}_checkpoint_manifest.tsv", sep="\t"))
        hists.append(pd.read_csv(rd / f"16_{route}_training_history.tsv", sep="\t"))
    return pd.concat(regs, ignore_index=True), pd.concat(hists, ignore_index=True)


def build_inventory(m16_reg):
    m15 = torch.load(M15_MODEL, map_location="cpu", weights_only=False)
    rows = [{
        "candidate_id": str(m15.get("source_M14_candidate_id", "M15_BASELINE")),
        "origin": "M15_BASELINE", "route": "BASELINE_M15",
        "batch_size": int(m15.get("selected_batch_size", 0)),
        "epoch": int(m15.get("selected_epoch", 0)),
        "stage": int(m15.get("selected_stage", -1)),
        "stage_name": str(m15.get("selected_stage_name", "")),
        "extra_feature_count": 0,
        "source_feature_count": 26,
        "checkpoint_path": str(M15_MODEL),
        "preprocessor_path": str(M15_PRE),
    }]
    for row in m16_reg.itertuples(index=False):
        route = str(row.route)
        rd = M16_RUNS / f"{route}_bs64_seed{SEED16}"
        rows.append({
            "candidate_id": str(row.candidate_id),
            "origin": "M16_EXPANSION", "route": route,
            "batch_size": int(row.batch_size), "epoch": int(row.epoch),
            "stage": int(row.stage), "stage_name": str(row.stage_name),
            "extra_feature_count": len(ROUTE_EXTRAS[route]),
            "source_feature_count": 26 + len(ROUTE_EXTRAS[route]),
            "checkpoint_path": str(row.checkpoint_path),
            "preprocessor_path": str(rd / f"16_{route}_preprocessor.joblib"),
        })
    return pd.DataFrame(rows)


def input_dim_of(ckpt):
    if "training_invariants" in ckpt:
        return int(ckpt["training_invariants"]["model"]["input_dim"])
    return int(ckpt["model_state_dict"]["net.0.weight"].shape[1])


def transform_table__17(df, pre):
    a = pre["scaler"].transform(df[list(pre["numeric_features"])].to_numpy(float)).astype(np.float32)
    b = pre["encoder"].transform(df[CATEGORICAL__17].astype(str)).astype(np.float32)
    return np.concatenate([a, b], axis=1).astype(np.float32, copy=False)


class GCRMLP__17(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT__17),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT__17),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.inference_mode()
def predict__17(model, x, device):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(torch.from_numpy(x[s:s + PRED_BATCH__17]).to(device))).float().cpu().numpy()
        for s in range(0, len(x), PRED_BATCH__17)]).astype(float)


def gold_metrics__17(valid_df, scores, samples, with_details=False):
    y = valid_df[LABEL__17].to_numpy(np.int8)
    ro = valid_df[ROW__17].to_numpy(np.int64)
    order = np.lexsort((ro, -scores))
    ap, au, rep = [], [], []
    for i, idx in enumerate(samples):
        yy, ss = y[idx], scores[idx]
        a, b = float(average_precision_score(yy, ss)), float(roc_auc_score(yy, ss))
        ap.append(a)
        au.append(b)
        if with_details:
            rep.append({"repeat": i, "seed": SAMPLE_SEED, "rows": len(idx),
                        "positives": int(yy.sum()),
                        "negatives": int(len(yy) - yy.sum()),
                        "AUPRC": a, "AUROC": b})
    out = {"random1to1_auprc_mean": float(np.mean(ap)),
           "random1to1_auprc_std": float(np.std(ap, ddof=1)),
           "random1to1_auroc_mean": float(np.mean(au)),
           "random1to1_auroc_std": float(np.std(au, ddof=1))}
    prev = float(y.mean())
    topk = []
    for f, name in [(0.005, "top0_5"), (0.01, "top1"), (0.05, "top5")]:
        k = int(math.floor(len(y) * f))
        hits = int(y[order[:k]].sum())
        fe = float(hits / (k * prev))
        out[name + "_hits"] = hits
        out[name + "_fe"] = fe
        if with_details:
            topk.append({"fraction": f, "K": k, "Hits": hits, "FE": fe})
    if with_details:
        return out, pd.DataFrame(rep), pd.DataFrame(topk)
    return out


def reinfer_all__17(inventory, x_by_route, valid_df_by_route, samples, device):
    rows = []
    for i, row in enumerate(inventory.itertuples(index=False), start=1):
        ckpt = torch.load(Path(str(row.checkpoint_path)), map_location="cpu", weights_only=False)
        route = str(row.route)
        model = GCRMLP__17(input_dim_of(ckpt)).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        scores = predict__17(model, x_by_route[route], device)
        m = gold_metrics__17(valid_df_by_route[route], scores, samples)
        rows.append({
            "candidate_id": str(row.candidate_id), "origin": str(row.origin),
            "route": route, "batch_size": int(row.batch_size),
            "epoch": int(row.epoch), "stage": int(row.stage),
            "stage_name": str(row.stage_name),
            "extra_feature_count": int(row.extra_feature_count),
            "source_feature_count": int(row.source_feature_count),
            **m, "checkpoint_path": str(row.checkpoint_path),
            "preprocessor_path": str(row.preprocessor_path),
        })
        del model, ckpt, scores
        gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def select_final(metrics):
    ranked = metrics.sort_values(
        ["top1_hits", "top5_hits", "random1to1_auprc_mean", "top0_5_hits",
         "random1to1_auroc_mean", "epoch", "extra_feature_count", "candidate_id"],
        ascending=[False, False, False, False, False, True, True, True],
        kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked["selected"] = ranked["selection_rank"].eq(1)
    return ranked, ranked.iloc[0].copy()


def build_route_summary(metrics):
    rows = []
    for route, group in metrics.groupby("route", sort=False):
        best = group.sort_values(
            ["top1_hits", "top5_hits", "random1to1_auprc_mean", "top0_5_hits",
             "random1to1_auroc_mean", "epoch"],
            ascending=[False, False, False, False, False, True],
            kind="mergesort").iloc[0]
        rows.append({
            "route": route, "candidate_count": len(group),
            "extra_feature_count": int(best["extra_feature_count"]),
            "formal_best_candidate_id": str(best["candidate_id"]),
            "formal_best_epoch": int(best["epoch"]),
            "formal_best_top0_5_hits": int(best["top0_5_hits"]),
            "formal_best_top1_hits": int(best["top1_hits"]),
            "formal_best_top5_hits": int(best["top5_hits"]),
            "formal_best_1to1_auprc": float(best["random1to1_auprc_mean"]),
            "formal_best_1to1_auroc": float(best["random1to1_auroc_mean"]),
            "max_1to1_auprc": float(group["random1to1_auprc_mean"].max()),
            "max_1to1_auroc": float(group["random1to1_auroc_mean"].max()),
        })
    return pd.DataFrame(rows)


def source_features_of(route):
    if route == "BASELINE_M15":
        return BASE_SOURCE__17
    return [*BASE_NUMERIC__17, *ROUTE_EXTRAS[route], *CATEGORICAL__17]


def freeze_final(selected, x_by_route, valid_df_by_route, samples, device):
    route = str(selected["route"])
    src = Path(str(selected["checkpoint_path"]))
    pre_src = M15_PRE if route == "BASELINE_M15" else Path(str(selected["preprocessor_path"]))

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model = GCRMLP__17(input_dim_of(ckpt)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    scores = predict__17(model, x_by_route[route], device)
    m, rep, topk = gold_metrics__17(valid_df_by_route[route], scores, samples, with_details=True)

    valid = valid_df_by_route[route]
    ranking = np.lexsort((valid[ROW__17].to_numpy(np.int64), -scores))
    rp = np.empty(VALID_N, dtype=np.int64)
    rp[ranking] = np.arange(1, VALID_N + 1, dtype=np.int64)

    pred = pd.DataFrame({
        ID__17: valid[ID__17].astype(str).to_numpy(), SPLIT__17: "valid",
        ROW__17: valid[ROW__17].to_numpy(np.int64), LABEL__17: valid[LABEL__17].to_numpy(np.int8),
        "score_final_global": scores, "rank_final_global": rp,
        "rank_pct_final_global": 1.0 - (rp.astype(float) - 1.0) / float(VALID_N),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_route": route, "selected_epoch": int(selected["epoch"]),
    })
    write_parquet(pred, OUT__17 / "17_selected_validation_predictions.parquet")
    write_tsv(rep, OUT__17 / "17_selected_validation_random1to1_metrics.tsv")
    write_tsv(topk, OUT__17 / "17_selected_validation_topk_metrics.tsv")

    features = source_features_of(route)
    write_tsv(pd.DataFrame({"feature_index": range(len(features)), "feature": features}),
              OUT__17 / "17_final_global_feature_manifest.tsv")

    frozen = {
        "status": "FROZEN_FINAL_GLOBAL",
        "source_candidate_id": str(selected["candidate_id"]),
        "source_origin": str(selected["origin"]),
        "source_route": route,
        "selected_batch_size": int(selected["batch_size"]),
        "selected_epoch": int(selected["epoch"]),
        "selected_stage": int(selected["stage"]),
        "selected_stage_name": str(selected["stage_name"]),
        "additional_compact_feature_count": int(selected["extra_feature_count"]),
        "source_feature_count": len(features),
        "source_features": features,
        "model_state_dict": ckpt["model_state_dict"],
        "validation_gold_metrics": {
            "Top0.5% Hits": int(m["top0_5_hits"]), "Top0.5% FE": float(m["top0_5_fe"]),
            "Top1% Hits": int(m["top1_hits"]), "Top1% FE": float(m["top1_fe"]),
            "Top5% Hits": int(m["top5_hits"]), "Top5% FE": float(m["top5_fe"]),
            "1:1 AUPRC": float(m["random1to1_auprc_mean"]),
            "1:1 AUPRC std": float(m["random1to1_auprc_std"]),
            "1:1 AUROC": float(m["random1to1_auroc_mean"]),
            "1:1 AUROC std": float(m["random1to1_auroc_std"]),
        },
    }
    FROZEN.mkdir(parents=True, exist_ok=True)
    tmp = FROZEN / f"17_frozen_final_global_ranker.pt.tmp.{os.getpid()}"
    torch.save(frozen, tmp)
    os.replace(tmp, FROZEN / "17_frozen_final_global_ranker.pt")
    shutil.copy2(pre_src, FROZEN / "17_frozen_final_global_preprocessor.joblib")

    write_json({
        "status": "FROZEN_FINAL_GLOBAL",
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_route": route, "selected_epoch": int(selected["epoch"]),
        "additional_compact_feature_count": int(selected["extra_feature_count"]),
        "source_features": features,
        "validation_gold_metrics": frozen["validation_gold_metrics"],
    }, OUT__17 / "17_frozen_final_global_ranker_manifest.json")

    del model, ckpt, scores
    gc.collect(); torch.cuda.empty_cache()
    return frozen["validation_gold_metrics"]


def step_17():
    OUT__17.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)


    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    m13_valid = pd.read_parquet(M13_VALID__17).reset_index(drop=True)
    exp_valid = pd.read_parquet(M16_EXP_VALID).reset_index(drop=True)
    m16_reg, _ = load_m16_tables()
    inventory = build_inventory(m16_reg)

    samples = np.load(M14_SAMPLES__17, allow_pickle=False)["indices"].astype(np.int64)

    pre_m15 = joblib.load(M15_PRE)
    x_by_route = {"BASELINE_M15": transform_table__17(m13_valid, pre_m15)}
    valid_df_by_route = {"BASELINE_M15": m13_valid}
    for route in ROUTES__17:
        pre = joblib.load(
            M16_RUNS / f"{route}_bs64_seed{SEED16}/16_{route}_preprocessor.joblib"
        )
        x_by_route[route] = transform_table__17(exp_valid, pre)
        valid_df_by_route[route] = exp_valid


    metrics = reinfer_all__17(inventory, x_by_route, valid_df_by_route, samples, device)

    ranked, selected = select_final(metrics)
    write_tsv(build_route_summary(metrics), OUT__17 / "17_route_summary.tsv")

    freeze_final(selected, x_by_route, valid_df_by_route, samples, device)


# --------------------------------------------------------------------------
# step 18: Export the strict out-of-fold global predictions
# --------------------------------------------------------------------------



MODELING__18 = Path(__file__).resolve().parent

M08_FOLD_MANIFEST__18 = MODELING__18 / "output/08_grid_search_biological_prior/08_train_fold_manifest.tsv"
M16_EXP_TRAIN = MODELING__18 / "output/16_train_global_feature_expansion_probe/16_common/16_expanded_train.parquet"
M16_EXP_VALID__18 = MODELING__18 / "output/16_train_global_feature_expansion_probe/16_common/16_expanded_valid.parquet"
M17_PRE = MODELING__18 / "output/17_audit_select_final_global_ranker/17_frozen/17_frozen_final_global_preprocessor.joblib"
M17_VALID = MODELING__18 / "output/17_audit_select_final_global_ranker/17_selected_validation_predictions.parquet"
M17_MANIFEST = MODELING__18 / "output/17_audit_select_final_global_ranker/17_frozen_final_global_ranker_manifest.json"

OUT__18 = MODELING__18 / "output/18_export_final_global_strict_oof"
FOLD_ROOT = OUT__18 / "18_fold_models"

ID__18, SPLIT__18, ROW__18, M12_ROW, FOLD, LABEL__18 = "herv_internal_id", "split", "canonical_split_row_index", "row_index", "fold_id", "is_fantom5_enhancer"
EXPECTED_TRAIN_N, EXPECTED_VALID_N = 336_217, 72_048
EXPECTED_FOLDS = 5

BASE_NUMERIC__18 = [
    "score_ccre_enhancer", "score_serve_gtex_support",
    "score_sequence_regulatory", "score_final_representation",
    "calibration_delta",
    "rank_pct_score_ccre_enhancer", "rank_pct_score_serve_gtex_support",
    "rank_pct_score_sequence_regulatory", "rank_pct_score_final_representation",
    "repeat_divergence", "length", "log_length", "gc_content", "n_content",
    "dist_to_tss", "log_abs_dist_to_tss",
    "is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY",
    "is_ccre_ca_overlap", "is_ccre_enhancer_like",
    "is_serve_gtex_supported", "serve_gtex_tissue_count",
]
CATEGORICAL__18 = ["repClass", "repFamily", "nearest_gene_type"]

ROADMAP9__18 = [
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
MPRA6__18 = [
    "gctx_mpra_is_overlap", "gctx_mpra_is_active_supported",
    "gctx_mpra_total_overlap_bp_log1p", "gctx_mpra_active_total_overlap_bp_log1p",
    "gctx_mpra_active_count_fraction", "gctx_mpra_active_bp_fraction",
]
CCRE_SERVE9__18 = [
    "gctx_ccre_is_ctcf_overlap", "gctx_ccre_is_pls_overlap",
    "gctx_ccre_els_overlap_count_log1p", "gctx_ccre_els_total_overlap_bp_log1p",
    "gctx_serve_annotation_count_log1p", "gctx_serve_total_overlap_bp_log1p",
    "gctx_serve_noncoding_overlap_count_log1p",
    "gctx_serve_noncoding_total_overlap_bp_log1p",
    "gctx_serve_noncoding_bp_fraction",
]
COMPACT24 = [*ROADMAP9__18, *MPRA6__18, *CCRE_SERVE9__18]


def load_selection():
    with open(M17_MANIFEST, encoding="utf-8") as handle:
        selection = json.load(handle)
    route = str(selection["selected_route"])
    numeric = [*BASE_NUMERIC__18, *ROUTE_EXTRA[route]]
    source_features = [*numeric, *CATEGORICAL__18]
    return route, int(selection["selected_epoch"]), numeric, source_features


ROUTE_EXTRA = {
    "BASELINE_M15": [],
    "R1_ROADMAP": ROADMAP9__18,
    "R2_MPRA": MPRA6__18,
    "R3_CCRE_SERVE": CCRE_SERVE9__18,
    "R4_ALL_COMPACT": COMPACT24,
}

BATCH__18, SEED__18, DROPOUT__18, LR__18, WD__18 = 64, 99999999, 0.20, 8e-4, 1e-4
FOCAL_ALPHA__18, FOCAL_GAMMA__18, PRED_BATCH__18 = 0.50, 1.0, 8192
HARD_Q, BOUNDARY_LOWER__18, BOUNDARY_UPPER__18 = 0.90, 0.980, 0.995

CURRICULUM__18 = [
    {"stage": 1, "name": "random_only", "epoch_start": 1, "epoch_end": 6,
     "ratios": {"random": 1.0}},
    {"stage": 2, "name": "random_plus_evidence_rich", "epoch_start": 7, "epoch_end": 14,
     "ratios": {"random": 1.0, "evidence_rich": 0.5}},
    {"stage": 3, "name": "soft_hard_negative", "epoch_start": 15, "epoch_end": 22,
     "ratios": {"random": 1.0, "evidence_rich": 0.5, "hard": 0.1}},
    {"stage": 4, "name": "soft_boundary_negative", "epoch_start": 23, "epoch_end": 30,
     "ratios": {"random": 1.0, "evidence_rich": 0.5, "boundary": 0.1}},
]


def stage_for__18(epoch):
    for s in CURRICULUM__18:
        if s["epoch_start"] <= epoch <= s["epoch_end"]:
            return s
    raise ValueError(f"epoch {epoch} out of curriculum range")


def write_table(df, path, kind="tsv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if kind == "tsv":
        df.to_csv(tmp, sep="\t", index=False)
    elif kind == "parquet":
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_torch(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_joblib(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def seed_all__18(seed=SEED__18):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class GlobalMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT__18),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT__18),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def focal__18(logits, y):
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = y * p + (1 - y) * (1 - p)
    at = y * FOCAL_ALPHA__18 + (1 - y) * (1 - FOCAL_ALPHA__18)
    return torch.mean(at * torch.pow(1 - pt, FOCAL_GAMMA__18) * bce)


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def fit_fold_preprocessor(fit, target_names, numeric, source_features):
    scaler = StandardScaler().fit(fit[numeric].to_numpy(float))
    encoder = make_encoder().fit(fit[CATEGORICAL__18].astype(str))
    native_names = [*numeric,
                    *encoder.get_feature_names_out(CATEGORICAL__18).astype(str).tolist()]
    target_index = {n: i for i, n in enumerate(target_names)}

    def transform(df):
        num = scaler.transform(df[numeric].to_numpy(float)).astype(np.float32)
        cat = encoder.transform(df[CATEGORICAL__18].astype(str)).astype(np.float32)
        native = np.concatenate([num, cat], axis=1).astype(np.float32, copy=False)
        aligned = np.zeros((len(df), len(target_names)), dtype=np.float32)
        for j, name in enumerate(native_names):
            if name in target_index:
                aligned[:, target_index[name]] = native[:, j]
        return aligned

    pre = {
        "numeric_features": numeric, "categorical_features": CATEGORICAL__18,
        "source_features": source_features, "scaler": scaler, "encoder": encoder,
        "native_transformed_feature_names": native_names,
        "frozen_target_transformed_feature_names": target_names,
        "native_transformed_dim": len(native_names),
        "transformed_dim": len(target_names),
        "fit_scope": "four-fold fitting subset only",
    }
    return pre, transform


def build_fold_pools(fit):
    y = fit[LABEL__18].to_numpy(np.int8)
    positive = np.flatnonzero(y == 1).astype(np.int64)
    negative = np.flatnonzero(y == 0).astype(np.int64)
    neg = fit.iloc[negative]
    evidence = negative[
        (neg["is_ccre_ca_overlap"].to_numpy(float) > 0)
        | (neg["is_ccre_enhancer_like"].to_numpy(float) > 0)
        | (neg["is_serve_gtex_supported"].to_numpy(float) > 0)
    ]
    neg_scores = neg["score_final_representation"].to_numpy(np.float64)
    hard_thr = float(np.quantile(neg_scores, HARD_Q))
    hard = negative[neg_scores >= hard_thr]
    rank_pct = fit["score_final_representation"].rank(method="average", pct=True).to_numpy()
    boundary = np.flatnonzero(
        (y == 0) & (rank_pct >= BOUNDARY_LOWER__18) & (rank_pct < BOUNDARY_UPPER__18)
    ).astype(np.int64)
    return {
        "positive": positive, "random": negative,
        "evidence_rich": evidence.astype(np.int64),
        "hard": hard.astype(np.int64), "boundary": boundary,
    }, hard_thr


def sample_epoch(pools, epoch):
    stage = stage_for__18(epoch)
    rng = np.random.default_rng(SEED__18 + epoch)
    n_pos = len(pools["positive"])
    pieces = [pools["positive"].copy()]
    for name, ratio in stage["ratios"].items():
        k = int(round(n_pos * ratio))
        pieces.append(rng.choice(pools[name], size=k, replace=False))
    idx = np.concatenate(pieces).astype(np.int64)
    rng.shuffle(idx)
    return idx, stage


def safe_batches(idx):
    b = [idx[s:s + BATCH__18] for s in range(0, len(idx), BATCH__18)]
    if len(b[-1]) == 1:
        prev = b[-2]
        b[-2] = prev[:-1]
        b[-1] = np.concatenate([prev[-1:], b[-1]])
    return b


def train_epoch__18(model, opt, x, y, idx, device):
    model.train()
    total, n = 0.0, 0
    for bi in safe_batches(idx):
        bx = torch.from_numpy(x[bi]).to(device, non_blocking=True)
        by = torch.from_numpy(y[bi].astype(np.float32, copy=False)).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = focal__18(model(bx), by)
        loss.backward()
        opt.step()
        total += float(loss.detach().cpu()) * len(bi)
        n += len(bi)
    return total / n


@torch.inference_mode()
def predict__18(model, x, device):
    model.eval()
    out = []
    for s in range(0, len(x), PRED_BATCH__18):
        z = torch.from_numpy(x[s:s + PRED_BATCH__18]).to(device, non_blocking=True)
        out.append(torch.sigmoid(model(z)).detach().float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def run_fold(fold, train, manifest, target_names, device, numeric, source_features, selected_epoch):
    rd = FOLD_ROOT / f"fold_{fold}"
    rd.mkdir(parents=True, exist_ok=True)
    pred_path = rd / f"18_fold_{fold}_heldout_predictions.parquet"
    hist_path = rd / f"18_fold_{fold}_training_history.tsv"

    fold_arr = manifest[FOLD].astype(int).to_numpy()
    held_idx = np.flatnonzero(fold_arr == fold)
    fit_idx = np.flatnonzero(fold_arr != fold)
    fit = train.iloc[fit_idx].reset_index(drop=True)
    held = train.iloc[held_idx].reset_index(drop=True)

    pre, transform = fit_fold_preprocessor(fit, target_names, numeric, source_features)
    write_joblib(pre, rd / f"18_fold_{fold}_preprocessor.joblib")
    fit_x = transform(fit)
    held_x = transform(held)

    pools, hard_thr = build_fold_pools(fit)
    fit_y = fit[LABEL__18].to_numpy(np.int8)

    seed_all__18(SEED__18)
    model = GlobalMLP(len(target_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR__18, weight_decay=WD__18)

    history = []
    for epoch in range(1, selected_epoch + 1):
        idx, stage = sample_epoch(pools, epoch)
        loss = train_epoch__18(model, opt, fit_x, fit_y, idx, device)
        history.append({
            "fold_id": fold, "epoch": epoch,
            "stage": stage["stage"], "stage_name": stage["name"],
            "sampled_rows": len(idx), "train_loss": loss,
            "hard_threshold_fit_only": hard_thr,
        })
        write_table(pd.DataFrame(history), hist_path)

    held_scores = predict__18(model, held_x, device)
    pred = pd.DataFrame({
        ID__18: held[ID__18].astype(str).to_numpy(),
        SPLIT__18: "train",
        ROW__18: held[ROW__18].to_numpy(np.int64),
        FOLD: np.full(len(held), fold, dtype=np.int8),
        LABEL__18: held[LABEL__18].to_numpy(np.int8),
        "score_final_global": held_scores,
        "prediction_origin": "train_oof_final_global",
        "model_fit_excluded_row": True,
        "preprocess_fit_excluded_row": True,
        "pool_definition_excluded_row": True,
    })
    write_table(pred, pred_path, kind="parquet")

    write_torch({
        "fold_id": fold, "epoch": selected_epoch,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "source_features": source_features, "transformed_dim": len(target_names),
    }, rd / f"18_fold_{fold}_final_global.pt")

    del model, fit_x, held_x, held_scores
    gc.collect()
    torch.cuda.empty_cache()


def run_folds(train, manifest, target_names, device, numeric, source_features, selected_epoch):
    for fold in range(EXPECTED_FOLDS):
        run_fold(int(fold), train, manifest, target_names, device, numeric, source_features, selected_epoch)


def consolidate(train, valid, started):
    frames, histories, registry = [], [], []
    for fold in range(EXPECTED_FOLDS):
        rd = FOLD_ROOT / f"fold_{fold}"
        pred = pd.read_parquet(rd / f"18_fold_{fold}_heldout_predictions.parquet")
        hist = pd.read_csv(rd / f"18_fold_{fold}_training_history.tsv", sep="\t")
        frames.append(pred)
        histories.append(hist)
        registry.append({
            "fold_id": fold, "heldout_rows": len(pred),
            "heldout_positives": int(pred[LABEL__18].sum()),
            "model_path": str(rd / f"18_fold_{fold}_final_global.pt"),
            "predictions_path": str(rd / f"18_fold_{fold}_heldout_predictions.parquet"),
        })

    train_oof = pd.concat(frames, ignore_index=True).sort_values(ROW__18).reset_index(drop=True)
    score = train_oof["score_final_global"].to_numpy(np.float64)
    rank = np.empty(len(train_oof), dtype=np.int64)
    rank[np.lexsort((train_oof[ROW__18].to_numpy(np.int64), -score))] = np.arange(1, len(train_oof) + 1)
    train_oof["rank_final_global"] = rank
    train_oof["rank_pct_final_global"] = 1.0 - (rank.astype(float) - 1.0) / len(train_oof)

    saved_valid = pd.read_parquet(M17_VALID).sort_values(ROW__18).reset_index(drop=True)
    valid_out = pd.DataFrame({
        ID__18: valid[ID__18].astype(str).to_numpy(),
        SPLIT__18: "valid",
        ROW__18: valid[ROW__18].to_numpy(np.int64),
        FOLD: np.full(len(valid), -1, dtype=np.int8),
        LABEL__18: valid[LABEL__18].to_numpy(np.int8),
        "score_final_global": saved_valid["score_final_global"].to_numpy(float),
        "prediction_origin": "valid_full_train_final_global",
        "model_fit_excluded_row": False,
        "preprocess_fit_excluded_row": False,
        "pool_definition_excluded_row": False,
    })
    vscore = valid_out["score_final_global"].to_numpy(float)
    vrank = np.empty(len(valid_out), dtype=np.int64)
    vrank[np.lexsort((valid_out[ROW__18].to_numpy(np.int64), -vscore))] = np.arange(1, len(valid_out) + 1)
    valid_out["rank_final_global"] = vrank
    valid_out["rank_pct_final_global"] = 1.0 - (vrank.astype(float) - 1.0) / len(valid_out)

    development = pd.concat([train_oof, valid_out], ignore_index=True, sort=False)

    write_table(train_oof, OUT__18 / "18_train_oof_final_global_predictions.parquet", kind="parquet")
    write_table(valid_out, OUT__18 / "18_valid_full_train_final_global_predictions.parquet", kind="parquet")
    write_table(development, OUT__18 / "18_development_final_global_predictions.parquet", kind="parquet")
    write_table(pd.concat(histories, ignore_index=True), OUT__18 / "18_training_history.tsv")
    write_table(pd.DataFrame(registry), OUT__18 / "18_fold_model_registry.tsv")


def step_18():
    started = time.perf_counter()
    OUT__18.mkdir(parents=True, exist_ok=True)
    FOLD_ROOT.mkdir(parents=True, exist_ok=True)

    selected_route, selected_epoch, numeric, source_features = load_selection()


    train = pd.read_parquet(M16_EXP_TRAIN)
    valid = pd.read_parquet(M16_EXP_VALID__18)

    manifest = pd.read_csv(M08_FOLD_MANIFEST__18, sep="\t").sort_values(M12_ROW).reset_index(drop=True)
    full_pre = joblib.load(M17_PRE)
    target_names = [*numeric, *full_pre["encoder"].get_feature_names_out(CATEGORICAL__18).astype(str).tolist()]
    device = torch.device("cuda:0")

    run_folds(train, manifest, target_names, device, numeric, source_features, selected_epoch)
    consolidate(train, valid, started)


def main() -> None:
    step_13()
    step_14()
    step_15()
    step_16()
    step_17()
    step_18()


if __name__ == "__main__":
    main()
