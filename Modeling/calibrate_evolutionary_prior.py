#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from modeling_common import (
    safe_ap,
    safe_auc,
    stable_order,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import gc
import json
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import shutil
from datetime import (
    datetime,
    timezone,
)
from typing import Any
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (
    DataLoader,
    Dataset,
)

# --------------------------------------------------------------------------
# step 08: Grid-search the evolutionary prior
# --------------------------------------------------------------------------



# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELING_DIR = (PROJECT_ROOT / "Modeling").resolve()

SCRIPT_NAME = "08_grid_search_biological_prior"
SCRIPT_ID = SCRIPT_NAME


M07_DIR = MODELING_DIR / "output" / "07_audit_partial_unfreeze"
M07_FROZEN_CHECKPOINT = M07_DIR / '07_frozen' / '07_frozen_sequence_reference.pt'

M01_DIR = MODELING_DIR / "output" / "01_build_dual_tensors"
TRAIN_TENSOR = M01_DIR / "01_train_dual_tensors.pt"
CANONICAL_TABLE = MODELING_DIR / 'output' / '00_build_canonical_data' / '00_canonical_herv_mef_table.parquet'

OUT_DIR = MODELING_DIR / "output" / SCRIPT_NAME
OUT_FOLD_MANIFEST = OUT_DIR / "08_train_fold_manifest.tsv"
OUT_BASE_SCORES = OUT_DIR / "08_train_m07_sequence_scores.parquet"


OUT_SELECTED_PRIOR = OUT_DIR / "08_selected_biological_prior.json"


INFERENCE_BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 99999999
N_FOLDS = 5

META_EMBED_DIM = 16
META_HIDDEN_DIM = 64
K6_LENGTH = 996
BPE_LENGTH = 256

ID_COL = "herv_internal_id"
SPLIT_COL = "split"
FANTOM_LABEL = "is_fantom5_enhancer"
DIRECT_TARGETS = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
TARGET_NAMES = list(DIRECT_TARGETS)
ACTIVE_HEAD_INDICES = (0, 1)
SCORE_HEAD_INDICES = (0, 1)

DIVERGENCE_COLUMN = "repeat_divergence"
FAMILY_COLUMN = "repFamily"
FAMILIES = ("ERV1", "ERVK", "ERVL", "ERVL-MALR", "GYPSY")

C_VALUES = (0.01, 0.1, 1.0, 10.0)
PRIOR_FEATURES = {
    "P1": ["sequence_logit", "repeat_divergence_z"],
    "P_fam5_int_noglobal": (
        ["sequence_logit"]
        + [f"is_{name}" for name in FAMILIES]
        + [f"is_{name}_x_div_z" for name in FAMILIES]
    ),
}
FORMAL_METRIC_ORDER = ["top1", "top5", "auprc", "top0_5", "auroc"]

RANDOM_1TO1_REPEATS = 100
RANDOM_1TO1_SEED = 99999999
TOPK_FRACTIONS = (0.005, 0.01, 0.05)
TOPK_NAMES = {0.005: "top0_5", 0.01: "top1", 0.05: "top5"}

LOGISTIC_MAX_ITER = 5000
LOGIT_EPSILON = 1e-6


# =============================================================================
# 2. Utilities
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_output() -> None:
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)








def probability_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        LOGIT_EPSILON,
        1.0 - LOGIT_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


# =============================================================================
# 3. Frozen M07 architecture (must match the 07 checkpoint exactly)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class SequenceReferenceModel(nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.config = dict(config)
        self.cat_features = list(config["cat_features"])

        self.emb_k6 = nn.Embedding(
            int(config["vocab_k6"]), d_model, padding_idx=0
        )
        self.cnn_k6 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_k6 = PositionalEncoding(
            d_model, K6_LENGTH, float(config["dropout"])
        )
        self.trans_k6 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.emb_bpe = nn.Embedding(
            int(config["vocab_bpe"]), d_model, padding_idx=0
        )
        self.cnn_bpe = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_bpe = PositionalEncoding(
            d_model, BPE_LENGTH, float(config["dropout"])
        )
        self.trans_bpe = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(config["cat_cardinalities"][feature]),
                    META_EMBED_DIM,
                    padding_idx=0,
                )
                for feature in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM
                + int(config["num_num_features"]),
                META_HIDDEN_DIM,
            ),
            nn.LayerNorm(META_HIDDEN_DIM),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
                for _ in range(3)
            ]
        )

    def forward(
        self,
        k6_ids: torch.Tensor,
        k6_mask: torch.Tensor,
        bpe_ids: torch.Tensor,
        bpe_mask: torch.Tensor,
        cat_ids: torch.Tensor,
        num_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_k6 = self.emb_k6(k6_ids)
        f_k6 = self.cnn_k6(f_k6.transpose(1, 2)).transpose(1, 2)
        f_k6 = self.pos_k6(f_k6)
        f_k6 = self.trans_k6(
            f_k6, src_key_padding_mask=(k6_mask == 0)
        )[:, 0, :]

        f_bpe = self.emb_bpe(bpe_ids)
        f_bpe = self.cnn_bpe(f_bpe.transpose(1, 2)).transpose(1, 2)
        f_bpe = self.pos_bpe(f_bpe)
        f_bpe = self.trans_bpe(
            f_bpe, src_key_padding_mask=(bpe_mask == 0)
        )[:, 0, :]

        categorical = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(categorical + [num_values], dim=1)
        )

        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6
            + gate_bpe * f_bpe
            + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)

        head_outputs = [head(fused).squeeze(-1) for head in self.heads]
        return head_outputs[0], head_outputs[1]


class TrainDataset(Dataset):
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["rows"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


# =============================================================================
# 4. Frozen inference
# =============================================================================

@torch.inference_mode()
def infer_m07_train(
    checkpoint: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model = SequenceReferenceModel(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    ccre_parts: list[np.ndarray] = []
    serve_parts: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader, start=1):
        with torch.amp.autocast(device_type="cuda"):
            ccre_logit, serve_logit = model(
                batch["k6_ids"].to(device, non_blocking=True),
                batch["k6_mask"].to(device, non_blocking=True),
                batch["bpe_ids"].to(device, non_blocking=True),
                batch["bpe_mask"].to(device, non_blocking=True),
                batch["cat_ids"].to(device, non_blocking=True),
                batch["num_values"].to(device, non_blocking=True),
            )
        ccre_parts.append(
            torch.sigmoid(ccre_logit).detach().float().cpu().numpy()
        )
        serve_parts.append(
            torch.sigmoid(serve_logit).detach().float().cpu().numpy()
        )

    ccre = np.concatenate(ccre_parts).astype(np.float64)
    serve = np.concatenate(serve_parts).astype(np.float64)
    sequence = np.mean(np.column_stack([ccre, serve]), axis=1)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "score_ccre_enhancer": ccre,
        "score_serve_gtex_support": serve,
        "score_sequence_regulatory": sequence,
    }


# =============================================================================
# 5. Fold manifest
# =============================================================================

def build_stratification_token(info: pd.DataFrame) -> pd.Series:
    candidates = [
        (
            "F" + info[FANTOM_LABEL].astype(str)
            + "_C" + info[DIRECT_TARGETS[0]].astype(str)
            + "_M" + info[DIRECT_TARGETS[1]].astype(str)
            + "_S" + info[DIRECT_TARGETS[2]].astype(str)
        ),
        (
            "F" + info[FANTOM_LABEL].astype(str)
            + "_M" + info[DIRECT_TARGETS[1]].astype(str)
        ),
        "F" + info[FANTOM_LABEL].astype(str),
    ]
    for level, token in enumerate(candidates):
        token = token.astype(str)
        counts = token.value_counts()
        if len(counts) > 0 and int(counts.min()) >= N_FOLDS:
            return token
    return candidates[-1].astype(str)


def build_fold_manifest(info: pd.DataFrame) -> pd.DataFrame:
    token = build_stratification_token(info)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    fold_id = np.full(len(info), -1, dtype=np.int8)
    placeholder = np.zeros(len(info), dtype=np.int8)
    for fold, (_, held_out) in enumerate(
        splitter.split(placeholder, token.to_numpy(dtype=str))
    ):
        fold_id[held_out] = np.int8(fold)

    manifest = info[
        [
            ID_COL,
            SPLIT_COL,
            "chrom",
            FAMILY_COLUMN,
            DIVERGENCE_COLUMN,
            "is_ERV1",
            "is_ERVK",
            "is_ERVL",
            "is_ERVL_MALR",
            "is_GYPSY",
            FANTOM_LABEL,
            *DIRECT_TARGETS,
        ]
    ].copy()
    manifest["fold_id"] = fold_id
    manifest["stratification_token"] = token.to_numpy(dtype=str)
    manifest["row_index"] = np.arange(len(manifest), dtype=np.int64)
    return manifest


# =============================================================================
# 6. Candidate fitting
# =============================================================================

def build_features(
    prior_type: str,
    sequence_logit: np.ndarray,
    divergence: np.ndarray,
    divergence_mean: float,
    divergence_std: float,
    family: np.ndarray,
) -> np.ndarray:
    if divergence_std <= 0.0:
        divergence_std = 1.0
    divergence_z = (divergence - divergence_mean) / divergence_std
    family = np.asarray(family, dtype=str)
    inds = {name: (family == name).astype(np.float64) for name in FAMILIES}
    if prior_type == "P1":
        matrix = np.column_stack([sequence_logit, divergence_z])
    else:
        matrix = np.column_stack(
            [sequence_logit]
            + [inds[name] for name in FAMILIES]
            + [inds[name] * divergence_z for name in FAMILIES]
        )
    return matrix.astype(np.float64, copy=False)


def build_candidate_inventory() -> pd.DataFrame:
    rows = [
        {
            "candidate_id": "P0_identity",
            "prior_type": "P0",
            "C": np.nan,
            "complexity_order": 0,
            "fold_fit_required": False,
        }
    ]
    for prior_type in ("P1", "P_fam5_int_noglobal"):
        complexity = 1 if prior_type == "P1" else 2
        for c_value in C_VALUES:
            rows.append(
                {
                    "candidate_id": (
                        f"{prior_type}_C{c_value}".replace(".", "p")
                    ),
                    "prior_type": prior_type,
                    "C": c_value,
                    "complexity_order": complexity,
                    "fold_fit_required": True,
                }
            )
    return pd.DataFrame(rows)


def fit_candidates(
    info: pd.DataFrame,
    manifest: pd.DataFrame,
    sequence_score: np.ndarray,
    inventory: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    labels = info[FANTOM_LABEL].to_numpy(dtype=np.int8)
    sequence_logit = probability_logit(sequence_score)
    divergence = info[DIVERGENCE_COLUMN].to_numpy(dtype=np.float64)
    family = info[FAMILY_COLUMN].astype(str).str.upper().to_numpy(dtype=str)
    fold_ids = manifest["fold_id"].to_numpy(dtype=np.int8)

    predictions: dict[str, np.ndarray] = {
        "P0_identity": sequence_score.astype(np.float64, copy=True)
    }
    fit_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    fitted = inventory.loc[inventory["fold_fit_required"].astype(bool)]
    for _, candidate in fitted.iterrows():
        candidate_id = str(candidate["candidate_id"])
        prior_type = str(candidate["prior_type"])
        c_value = float(candidate["C"])
        oof = np.full(len(info), np.nan, dtype=np.float64)


        for fold in range(N_FOLDS):
            train_index = np.flatnonzero(fold_ids != fold)
            held_index = np.flatnonzero(fold_ids == fold)

            fold_mean = float(divergence[train_index].mean())
            fold_std = float(divergence[train_index].std(ddof=0))
            if fold_std <= 0.0:
                fold_std = 1.0

            x_train = build_features(
                prior_type,
                sequence_logit[train_index],
                divergence[train_index],
                fold_mean,
                fold_std,
                family[train_index],
            )
            x_held = build_features(
                prior_type,
                sequence_logit[held_index],
                divergence[held_index],
                fold_mean,
                fold_std,
                family[held_index],
            )
            y_train = labels[train_index]

            model = LogisticRegression(
                penalty="l2",
                C=c_value,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=LOGISTIC_MAX_ITER,
                fit_intercept=True,
                random_state=SEED,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)

            oof[held_index] = model.predict_proba(x_held)[:, 1]

            fit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "prior_type": prior_type,
                    "C": c_value,
                    "fold_id": fold,
                    "fit_rows": int(len(train_index)),
                    "held_out_rows": int(len(held_index)),
                    "fit_positives": int(y_train.sum()),
                    "held_out_positives": int(labels[held_index].sum()),
                    "divergence_fit_mean": fold_mean,
                    "divergence_fit_std": fold_std,
                    "n_iter": int(np.max(model.n_iter_)),
                    "converged": True,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "penalty": "l2",
                }
            )

            feature_names = PRIOR_FEATURES[prior_type]
            for feature_name, coefficient in zip(
                feature_names, model.coef_.reshape(-1), strict=True
            ):
                coefficient_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "prior_type": prior_type,
                        "C": c_value,
                        "fold_id": fold,
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                        "intercept": float(model.intercept_[0]),
                        "divergence_fit_mean": fold_mean,
                        "divergence_fit_std": fold_std,
                    }
                )

        predictions[candidate_id] = oof

    return (
        predictions,
        pd.DataFrame(fit_rows),
        pd.DataFrame(coefficient_rows),
    )


# =============================================================================
# 7. Metrics
# =============================================================================

def build_random_samples(labels: np.ndarray) -> list[np.ndarray]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED)
    samples: list[np.ndarray] = []
    for _ in range(RANDOM_1TO1_REPEATS):
        sampled = rng.choice(
            negatives, size=len(positives), replace=False
        )
        indices = np.concatenate([positives, sampled])
        indices.sort()
        samples.append(indices.astype(np.int64))
    return samples


def evaluate_candidates(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    samples = build_random_samples(labels)
    rows: list[dict[str, Any]] = []

    for _, candidate in inventory.iterrows():
        candidate_id = str(candidate["candidate_id"])
        scores = predictions[candidate_id]

        auprcs: list[float] = []
        aurocs: list[float] = []
        for indices in samples:
            auprcs.append(safe_ap(labels[indices], scores[indices]))
            aurocs.append(safe_auc(labels[indices], scores[indices]))

        total = len(labels)
        prevalence = float(labels.sum()) / total
        order = stable_order(scores)
        row = candidate.to_dict()
        row["random1to1_auprc_mean"] = float(np.mean(auprcs))
        row["random1to1_auprc_std"] = float(np.std(auprcs, ddof=1))
        row["random1to1_auroc_mean"] = float(np.mean(aurocs))
        row["random1to1_auroc_std"] = float(np.std(aurocs, ddof=1))

        for fraction in TOPK_FRACTIONS:
            name = TOPK_NAMES[fraction]
            k = int(math.ceil(total * fraction))
            hits = int(labels[order[:k]].sum())
            expected = k * prevalence
            row[f"fantom_{name}_k"] = k
            row[f"fantom_{name}_hits"] = hits
            row[f"fantom_{name}_fe"] = (
                hits / expected if expected > 0 else float("nan")
            )
        rows.append(row)


    return pd.DataFrame(rows)


def candidate_raw_key(row: pd.Series) -> tuple:
    c_value = float(row["C"]) if pd.notna(row["C"]) else -math.inf
    return (
        -int(row["fantom_top1_hits"]),
        -int(row["fantom_top5_hits"]),
        -float(row["random1to1_auprc_mean"]),
        -int(row["fantom_top0_5_hits"]),
        -float(row["random1to1_auroc_mean"]),
        int(row["complexity_order"]),
        c_value,
        str(row["candidate_id"]),
    )


def select_prior(metrics: pd.DataFrame) -> str:
    ranked = metrics.copy()
    ranked["_key"] = ranked.apply(candidate_raw_key, axis=1)
    ranked = ranked.sort_values("_key").reset_index(drop=True)
    best = ranked.iloc[0]
    p0 = metrics.loc[metrics["candidate_id"] == "P0_identity"].iloc[0]

    # Identity fallback: keep P0 unless a fitted prior strictly beats it on
    # the first formal metric.
    if int(best["fantom_top1_hits"]) <= int(p0["fantom_top1_hits"]):
        return "P0_identity"
    return str(best["candidate_id"])


# =============================================================================
# 8. Main
# =============================================================================

def step_08() -> None:
    seed_everything()
    prepare_output()


    bundle = torch.load(TRAIN_TENSOR, map_location="cpu", weights_only=False)
    info = bundle["info_df"].reset_index(drop=True).copy()
    canonical = pd.read_parquet(
        CANONICAL_TABLE, columns=[ID_COL, FAMILY_COLUMN, DIVERGENCE_COLUMN]
    )
    info = info.merge(canonical, on=ID_COL, how="left")

    for column in [FANTOM_LABEL, *DIRECT_TARGETS]:
        info[column] = pd.to_numeric(
            info[column], errors="raise"
        ).astype(np.int8)
    info[DIVERGENCE_COLUMN] = pd.to_numeric(
        info[DIVERGENCE_COLUMN], errors="raise"
    ).astype(np.float64)
    info[FAMILY_COLUMN] = info[FAMILY_COLUMN].astype(str)
    info["chrom"] = info["chrom"].astype(str)
    family_upper = info[FAMILY_COLUMN].str.upper()
    info["is_ERV1"] = family_upper.eq("ERV1").astype(np.int8)
    info["is_ERVK"] = family_upper.eq("ERVK").astype(np.int8)
    info["is_ERVL"] = family_upper.eq("ERVL").astype(np.int8)
    info["is_ERVL_MALR"] = family_upper.eq("ERVL-MALR").astype(np.int8)
    info["is_GYPSY"] = family_upper.eq("GYPSY").astype(np.int8)

    checkpoint = torch.load(
        M07_FROZEN_CHECKPOINT, map_location="cpu", weights_only=False
    )

    device = torch.device("cuda:0")
    loader = DataLoader(
        TrainDataset(bundle),
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )
    base_scores = infer_m07_train(checkpoint, loader, device)
    sequence_score = base_scores["score_sequence_regulatory"]

    base_frame = pd.DataFrame({ID_COL: info[ID_COL].to_numpy()})
    for column, values in base_scores.items():
        base_frame[column] = values.astype(np.float32)
    base_frame["prediction_origin"] = "m07_full_train_frozen_inference"
    atomic_write_parquet(base_frame, OUT_BASE_SCORES)

    manifest = build_fold_manifest(info)
    atomic_write_tsv(manifest, OUT_FOLD_MANIFEST)

    inventory = build_candidate_inventory()
    predictions, fit_frame, coefficient_frame = fit_candidates(
        info, manifest, sequence_score, inventory
    )

    oof_frame = pd.DataFrame({ID_COL: info[ID_COL].to_numpy()})
    for candidate_id, values in predictions.items():
        oof_frame[f"score_{candidate_id}"] = values.astype(np.float64)

    metrics = evaluate_candidates(
        info[FANTOM_LABEL].to_numpy(dtype=np.int8), predictions, inventory
    )

    selected_id = select_prior(metrics)
    selected_row = metrics.loc[
        metrics["candidate_id"] == selected_id
    ].iloc[0]


    selected_prior = {
        "script": SCRIPT_ID,
        "created_at_utc": utc_now(),
        "selected_candidate_id": selected_id,
        "selected_prior_type": str(selected_row["prior_type"]),
        "selected_C": (
            float(selected_row["C"])
            if pd.notna(selected_row["C"])
            else None
        ),
        "selected_score_column_in_m08_oof": f"score_{selected_id}",
        "prior_definition": (
            "five family indicators and family x fold-standardized divergence"
        ),
        "ervk_definition": "upper(repFamily) == 'ERVK'",
        "sequence_score_input": (
            "mean(sigmoid(cCRE head), sigmoid(SERVE/GTEx head))"
        ),
        "formal_metric_order": FORMAL_METRIC_ORDER,
        "biological_prior_selected": selected_id != "P0_identity",
        "identity_fallback_used": selected_id == "P0_identity",
        "selected_metrics": selected_row.to_dict(),
        "m08_oof_scope": (
            "Only the lightweight FANTOM5 calibrator is cross-fitted. "
            "M12 retains strict deep-model OOF construction."
        ),
        "valid_used": False,
        "test_used": False,
    }
    atomic_write_json(OUT_SELECTED_PRIOR, selected_prior)


# --------------------------------------------------------------------------
# step 09: Train the biological refinement model
# --------------------------------------------------------------------------



# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_ROOT__09 = Path(__file__).resolve().parent.parent
MODELING_DIR__09 = (PROJECT_ROOT__09 / "Modeling").resolve()

SCRIPT_NAME__09 = "09_train_bio_refinement_B"
SCRIPT_ID__09 = SCRIPT_NAME__09


M01_DIR__09 = MODELING_DIR__09 / "output" / "01_build_dual_tensors"
TRAIN_TENSOR__09 = M01_DIR__09 / "01_train_dual_tensors.pt"
VALID_TENSOR = M01_DIR__09 / "01_valid_dual_tensors.pt"
CANONICAL_TABLE__09 = MODELING_DIR__09 / 'output' / '00_build_canonical_data' / '00_canonical_herv_mef_table.parquet'

M07_DIR__09 = MODELING_DIR__09 / "output" / "07_audit_partial_unfreeze"
M07_FROZEN_CHECKPOINT__09 = M07_DIR__09 / '07_frozen' / '07_frozen_sequence_reference.pt'

M08_DIR = MODELING_DIR__09 / "output" / "08_grid_search_biological_prior"
M08_SELECTED_PRIOR = M08_DIR / "08_selected_biological_prior.json"
M08_BASE_TRAIN_SCORES = M08_DIR / "08_train_m07_sequence_scores.parquet"

OUT_DIR__09 = MODELING_DIR__09 / "output" / SCRIPT_NAME__09
FROZEN_DIR = OUT_DIR__09 / "09_frozen"

OUT_VALID_PREDICTIONS = OUT_DIR__09 / '09_valid_biological_score_predictions.parquet'
OUT_FINAL_PARAMETERS = OUT_DIR__09 / "09_final_calibrator_parameters.json"


OUT_FROZEN_CALIBRATOR = FROZEN_DIR / '09_frozen_candidate_biological_calibrator.joblib'

OUT_MODEL_CONTRACT = OUT_DIR__09 / "09_bio_refinement_candidate_contract.json"


INFERENCE_BATCH_SIZE__09 = 128
NUM_WORKERS__09 = 0
SEED__09 = 99999999

META_EMBED_DIM__09 = 16
META_HIDDEN_DIM__09 = 64
K6_LENGTH__09 = 996
BPE_LENGTH__09 = 256

ID_COL__09 = "herv_internal_id"
FANTOM_LABEL__09 = "is_fantom5_enhancer"
ACTIVE_HEAD_INDICES__09 = (0, 1)

DIVERGENCE_COLUMN__09 = "repeat_divergence"
FAMILY_COLUMN__09 = "repFamily"
FAMILIES__09 = ("ERV1", "ERVK", "ERVL", "ERVL-MALR", "GYPSY")
FEATURE_ORDER = (
    ["sequence_logit"]
    + [f"is_{name}" for name in FAMILIES__09]
    + [f"is_{name}_x_div_z" for name in FAMILIES__09]
)

LOGISTIC_MAX_ITER__09 = 5000
LOGIT_EPSILON__09 = 1e-6


# =============================================================================
# 2. Utilities
# =============================================================================

def utc_now__09() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything__09(seed: int = SEED__09) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_output__09() -> None:
    import shutil
    if OUT_DIR__09.exists():
        shutil.rmtree(OUT_DIR__09)
    OUT_DIR__09.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json__09(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_tsv__09(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet__09(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_joblib_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def probability_logit__09(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        LOGIT_EPSILON__09,
        1.0 - LOGIT_EPSILON__09,
    )
    return np.log(clipped / (1.0 - clipped))


# =============================================================================
# 3. Frozen M07 architecture (must match the 07 checkpoint exactly)
# =============================================================================

class PositionalEncoding__09(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class SequenceReferenceModel__09(nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.config = dict(config)
        self.cat_features = list(config["cat_features"])

        self.emb_k6 = nn.Embedding(
            int(config["vocab_k6"]), d_model, padding_idx=0
        )
        self.cnn_k6 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_k6 = PositionalEncoding__09(
            d_model, K6_LENGTH__09, float(config["dropout"])
        )
        self.trans_k6 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.emb_bpe = nn.Embedding(
            int(config["vocab_bpe"]), d_model, padding_idx=0
        )
        self.cnn_bpe = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_bpe = PositionalEncoding__09(
            d_model, BPE_LENGTH__09, float(config["dropout"])
        )
        self.trans_bpe = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(config["cat_cardinalities"][feature]),
                    META_EMBED_DIM__09,
                    padding_idx=0,
                )
                for feature in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM__09
                + int(config["num_num_features"]),
                META_HIDDEN_DIM__09,
            ),
            nn.LayerNorm(META_HIDDEN_DIM__09),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM__09, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM__09, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
                for _ in range(3)
            ]
        )

    def forward(
        self,
        k6_ids: torch.Tensor,
        k6_mask: torch.Tensor,
        bpe_ids: torch.Tensor,
        bpe_mask: torch.Tensor,
        cat_ids: torch.Tensor,
        num_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_k6 = self.emb_k6(k6_ids)
        f_k6 = self.cnn_k6(f_k6.transpose(1, 2)).transpose(1, 2)
        f_k6 = self.pos_k6(f_k6)
        f_k6 = self.trans_k6(
            f_k6, src_key_padding_mask=(k6_mask == 0)
        )[:, 0, :]

        f_bpe = self.emb_bpe(bpe_ids)
        f_bpe = self.cnn_bpe(f_bpe.transpose(1, 2)).transpose(1, 2)
        f_bpe = self.pos_bpe(f_bpe)
        f_bpe = self.trans_bpe(
            f_bpe, src_key_padding_mask=(bpe_mask == 0)
        )[:, 0, :]

        categorical = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(categorical + [num_values], dim=1)
        )

        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6
            + gate_bpe * f_bpe
            + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)

        head_outputs = [head(fused).squeeze(-1) for head in self.heads]
        return head_outputs[0], head_outputs[1]


class TensorDatasetRows(Dataset):
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["rows"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
        }


# =============================================================================
# 4. Calibrator
# =============================================================================

def build_feature_matrix(
    sequence_score: np.ndarray,
    divergence: np.ndarray,
    divergence_mean: float,
    divergence_std: float,
    family: np.ndarray,
) -> np.ndarray:
    if divergence_std <= 0.0:
        divergence_std = 1.0
    sequence_logit = probability_logit__09(sequence_score)
    divergence_z = (divergence - divergence_mean) / divergence_std
    family = np.asarray(family, dtype=str)
    inds = {name: (family == name).astype(np.float64) for name in FAMILIES__09}
    return np.column_stack(
        [sequence_logit]
        + [inds[name] for name in FAMILIES__09]
        + [inds[name] * divergence_z for name in FAMILIES__09]
    ).astype(np.float64, copy=False)


def fit_final_calibrator(
    base: pd.DataFrame,
    candidate_id: str,
    prior_type: str,
    c_value: float,
) -> tuple[LogisticRegression, float, float, pd.DataFrame, pd.DataFrame]:
    labels = base[FANTOM_LABEL__09].to_numpy(dtype=np.int8)
    sequence = base["score_sequence_regulatory"].to_numpy(dtype=np.float64)
    divergence = base[DIVERGENCE_COLUMN__09].to_numpy(dtype=np.float64)
    family = base[FAMILY_COLUMN__09].astype(str).str.upper().to_numpy(dtype=str)

    divergence_mean = float(divergence.mean())
    divergence_std = float(divergence.std(ddof=0))
    x_train = build_feature_matrix(
        sequence, divergence, divergence_mean, divergence_std, family
    )

    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER__09,
        fit_intercept=True,
        random_state=SEED__09,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, labels)

    coefficient_frame = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "prior_type": prior_type,
                "C": c_value,
                "feature_order": index,
                "feature": feature,
                "coefficient": float(coefficient),
                "intercept": float(model.intercept_[0]),
                "divergence_fit_mean": divergence_mean,
                "divergence_fit_std": divergence_std,
            }
            for index, (feature, coefficient) in enumerate(
                zip(FEATURE_ORDER, model.coef_.reshape(-1), strict=True)
            )
        ]
    )
    fit_frame = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "prior_type": prior_type,
                "C": c_value,
                "fit_rows": int(len(base)),
                "fit_positives": int(labels.sum()),
                "divergence_fit_mean": divergence_mean,
                "divergence_fit_std": divergence_std,
                "n_iter": int(np.max(model.n_iter_)),
                "class_weight": "balanced",
                "solver": "lbfgs",
                "penalty": "l2",
            }
        ]
    )
    return model, divergence_mean, divergence_std, coefficient_frame, fit_frame


@torch.inference_mode()
def infer_m07_valid(
    checkpoint: dict[str, Any],
    bundle: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model = SequenceReferenceModel__09(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loader = DataLoader(
        TensorDatasetRows(bundle),
        batch_size=INFERENCE_BATCH_SIZE__09,
        shuffle=False,
        num_workers=NUM_WORKERS__09,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(NUM_WORKERS__09 > 0),
    )

    ccre_parts: list[np.ndarray] = []
    serve_parts: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        with torch.amp.autocast(device_type="cuda"):
            ccre_logit, serve_logit = model(
                batch["k6_ids"].to(device, non_blocking=True),
                batch["k6_mask"].to(device, non_blocking=True),
                batch["bpe_ids"].to(device, non_blocking=True),
                batch["bpe_mask"].to(device, non_blocking=True),
                batch["cat_ids"].to(device, non_blocking=True),
                batch["num_values"].to(device, non_blocking=True),
            )
        ccre_parts.append(
            torch.sigmoid(ccre_logit).detach().float().cpu().numpy()
        )
        serve_parts.append(
            torch.sigmoid(serve_logit).detach().float().cpu().numpy()
        )

    ccre = np.concatenate(ccre_parts).astype(np.float64)
    serve = np.concatenate(serve_parts).astype(np.float64)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(np.column_stack([ccre, serve]), axis=1)


# =============================================================================
# 5. Main
# =============================================================================

def step_09() -> None:
    seed_everything__09()
    prepare_output__09()


    selected = json.loads(M08_SELECTED_PRIOR.read_text(encoding="utf-8"))
    candidate_id = str(selected["selected_candidate_id"])
    prior_type = str(selected["selected_prior_type"])
    c_raw = selected.get("selected_C")
    c_value = float(c_raw) if c_raw is not None else 1.0

    train_bundle = torch.load(
        TRAIN_TENSOR__09, map_location="cpu", weights_only=False
    )
    train_info = train_bundle["info_df"].reset_index(drop=True).copy()
    canonical = pd.read_parquet(
        CANONICAL_TABLE__09, columns=[ID_COL__09, FAMILY_COLUMN__09, DIVERGENCE_COLUMN__09]
    )
    train_info = train_info.merge(canonical, on=ID_COL__09, how="left")
    train_info[FANTOM_LABEL__09] = pd.to_numeric(
        train_info[FANTOM_LABEL__09], errors="raise"
    ).astype(np.int8)
    train_info[DIVERGENCE_COLUMN__09] = pd.to_numeric(
        train_info[DIVERGENCE_COLUMN__09], errors="raise"
    ).astype(np.float64)

    base_scores = pd.read_parquet(M08_BASE_TRAIN_SCORES)
    train_info["score_sequence_regulatory"] = base_scores[
        "score_sequence_regulatory"
    ].to_numpy(dtype=np.float64)

    (
        model,
        divergence_mean,
        divergence_std,
        coefficient_frame,
        fit_frame,
    ) = fit_final_calibrator(train_info, candidate_id, prior_type, c_value)

    valid_bundle = torch.load(
        VALID_TENSOR, map_location="cpu", weights_only=False
    )
    valid_info = valid_bundle["info_df"].reset_index(drop=True).copy()
    canonical = pd.read_parquet(
        CANONICAL_TABLE__09, columns=[ID_COL__09, FAMILY_COLUMN__09, DIVERGENCE_COLUMN__09]
    )
    valid_info = valid_info.merge(canonical, on=ID_COL__09, how="left")
    valid_info[DIVERGENCE_COLUMN__09] = pd.to_numeric(
        valid_info[DIVERGENCE_COLUMN__09], errors="raise"
    ).astype(np.float64)
    valid_info[FAMILY_COLUMN__09] = valid_info[FAMILY_COLUMN__09].astype(str)

    checkpoint = torch.load(
        M07_FROZEN_CHECKPOINT__09, map_location="cpu", weights_only=False
    )
    device = torch.device("cuda:0")
    valid_sequence = infer_m07_valid(checkpoint, valid_bundle, device)

    x_valid = build_feature_matrix(
        valid_sequence,
        valid_info[DIVERGENCE_COLUMN__09].to_numpy(dtype=np.float64),
        divergence_mean,
        divergence_std,
        valid_info[FAMILY_COLUMN__09].str.upper().to_numpy(dtype=str),
    )
    valid_prediction = model.predict_proba(x_valid)[:, 1]

    valid_output = pd.DataFrame(
        {
            ID_COL__09: valid_info[ID_COL__09].to_numpy(),
            "score_sequence_regulatory": valid_sequence.astype(np.float64),
            "score_biological_prior_candidate": valid_prediction.astype(
                np.float64
            ),
            "prediction_origin": "m09_final_calibrator",
        }
    )
    atomic_write_parquet__09(valid_output, OUT_VALID_PREDICTIONS)

    atomic_joblib_dump(model, OUT_FROZEN_CALIBRATOR)

    coefficients = {
        feature: float(coefficient)
        for feature, coefficient in zip(
            FEATURE_ORDER, model.coef_.reshape(-1), strict=True
        )
    }

    atomic_write_json__09(
        OUT_FINAL_PARAMETERS,
        {
            "script": SCRIPT_ID__09,
            "created_at_utc": utc_now__09(),
            "candidate_id": candidate_id,
            "prior_type": prior_type,
            "C": c_value,
            "feature_order": list(FEATURE_ORDER),
            "divergence_mean": divergence_mean,
            "divergence_std": divergence_std,
            "intercept": float(model.intercept_[0]),
            "coefficients": coefficients,
            "n_iter": int(np.max(model.n_iter_)),
            "class_weight": "balanced",
            "solver": "lbfgs",
            "penalty": "l2",
        },
    )


    atomic_write_json__09(
        OUT_MODEL_CONTRACT,
        {
            "script": SCRIPT_ID__09,
            "created_at_utc": utc_now__09(),
            "candidate_id": candidate_id,
            "prior_type": prior_type,
            "C": c_value,
            "feature_order": list(FEATURE_ORDER),
            "active_head_indices": list(ACTIVE_HEAD_INDICES__09),
            "valid_fantom5_label_accessed": False,
            "valid_metrics_computed": False,
            "accepted_as_final_representation": False,
            "acceptance_decided_by": "M10",
            "test_used": False,
            "second_deep_network_trained": False,
        },
    )


# --------------------------------------------------------------------------
# step 10: Select and freeze the biological calibrator
# --------------------------------------------------------------------------



# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_ROOT__10 = Path(__file__).resolve().parent.parent
MODELING_DIR__10 = (PROJECT_ROOT__10 / "Modeling").resolve()

SCRIPT_NAME__10 = "10_eval_bio_refinement_B"
SCRIPT_ID__10 = SCRIPT_NAME__10


M01_DIR__10 = MODELING_DIR__10 / "output" / "01_build_dual_tensors"
VALID_TENSOR__10 = M01_DIR__10 / "01_valid_dual_tensors.pt"

M08_DIR__10 = MODELING_DIR__10 / "output" / "08_grid_search_biological_prior"
M08_SELECTED_PRIOR__10 = M08_DIR__10 / "08_selected_biological_prior.json"

M09_DIR = MODELING_DIR__10 / "output" / "09_train_bio_refinement_B"
M09_VALID_PREDICTIONS = M09_DIR / '09_valid_biological_score_predictions.parquet'
M09_FINAL_PARAMETERS = M09_DIR / "09_final_calibrator_parameters.json"
M09_CANDIDATE_CALIBRATOR = M09_DIR / '09_frozen' / '09_frozen_candidate_biological_calibrator.joblib'
M09_CONTRACT = M09_DIR / "09_bio_refinement_candidate_contract.json"

OUT_DIR__10 = MODELING_DIR__10 / "output" / SCRIPT_NAME__10
FROZEN_DIR__10 = OUT_DIR__10 / "10_frozen"

OUT_ROUTE_DECISION = OUT_DIR__10 / "10_final_representation_decision.json"
OUT_VALID_PREDICTIONS__10 = OUT_DIR__10 / '10_final_representation_validation_predictions.parquet'
OUT_FROZEN_CONTRACT_JSON = FROZEN_DIR__10 / '10_frozen_final_representation_contract.json'
OUT_FROZEN_CONTRACT_JOBLIB = FROZEN_DIR__10 / '10_frozen_final_representation_contract.joblib'
OUT_FROZEN_CALIBRATOR__10 = FROZEN_DIR__10 / "10_frozen_biological_calibrator.joblib"


SEED__10 = 99999999
ID_COL__10 = "herv_internal_id"
SPLIT_COL__10 = "split"
FANTOM_LABEL__10 = "is_fantom5_enhancer"

BASELINE_ID = "B0_identity"
CANDIDATE_ID = "B1_prior_candidate"
BASELINE_SCORE_COLUMN = "score_sequence_regulatory"
CANDIDATE_SCORE_COLUMN = "score_biological_prior_candidate"
FINAL_SCORE_COLUMN = "score_final_representation"

RANDOM_1TO1_REPEATS__10 = 100
RANDOM_1TO1_SEED__10 = 99999999
TOPK_FRACTIONS__10 = (0.005, 0.01, 0.05)
TOPK_NAMES__10 = {0.005: "top0_5", 0.01: "top1", 0.05: "top5"}
FORMAL_METRIC_ORDER__10 = ["top1", "top5", "auprc", "top0_5", "auroc"]


# =============================================================================
# 2. Utilities
# =============================================================================

def utc_now__10() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output__10() -> None:
    if OUT_DIR__10.exists():
        shutil.rmtree(OUT_DIR__10)
    OUT_DIR__10.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR__10.mkdir(parents=True, exist_ok=True)


def atomic_write_json__10(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_tsv__10(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet__10(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_joblib_dump__10(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)








# =============================================================================
# 3. Metrics
# =============================================================================

def build_random_samples__10(labels: np.ndarray) -> list[np.ndarray]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED__10)
    samples: list[np.ndarray] = []
    for _ in range(RANDOM_1TO1_REPEATS__10):
        sampled = rng.choice(negatives, size=len(positives), replace=False)
        indices = np.concatenate([positives, sampled])
        indices.sort()
        samples.append(indices.astype(np.int64))
    return samples


def score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    samples: list[np.ndarray],
) -> dict[str, Any]:
    auprcs: list[float] = []
    aurocs: list[float] = []
    for indices in samples:
        auprcs.append(safe_ap(labels[indices], scores[indices]))
        aurocs.append(safe_auc(labels[indices], scores[indices]))

    total = len(labels)
    prevalence = float(labels.sum()) / total
    order = stable_order(scores)
    row: dict[str, Any] = {
        "random1to1_auprc_mean": float(np.mean(auprcs)),
        "random1to1_auprc_std": float(np.std(auprcs, ddof=1)),
        "random1to1_auroc_mean": float(np.mean(aurocs)),
        "random1to1_auroc_std": float(np.std(aurocs, ddof=1)),
    }
    for fraction in TOPK_FRACTIONS__10:
        name = TOPK_NAMES__10[fraction]
        k = int(math.ceil(total * fraction))
        hits = int(labels[order[:k]].sum())
        expected = k * prevalence
        row[f"fantom_{name}_k"] = k
        row[f"fantom_{name}_hits"] = hits
        row[f"fantom_{name}_fe"] = (
            hits / expected if expected > 0 else float("nan")
        )
    return row


# =============================================================================
# 4. Main
# =============================================================================

def step_10() -> None:
    np.random.seed(SEED__10)
    prepare_output__10()


    bundle = torch.load(VALID_TENSOR__10, map_location="cpu", weights_only=False)
    info = bundle["info_df"].reset_index(drop=True).copy()
    info[FANTOM_LABEL__10] = pd.to_numeric(
        info[FANTOM_LABEL__10], errors="raise"
    ).astype(np.int8)
    labels = info[FANTOM_LABEL__10].to_numpy(dtype=np.int8)

    predictions = pd.read_parquet(M09_VALID_PREDICTIONS)
    baseline = predictions[BASELINE_SCORE_COLUMN].to_numpy(dtype=np.float64)
    candidate = predictions[CANDIDATE_SCORE_COLUMN].to_numpy(dtype=np.float64)


    samples = build_random_samples__10(labels)
    baseline_row = score_metrics(labels, baseline, samples)
    candidate_row = score_metrics(labels, candidate, samples)


    # Accept the biological prior only when it improves the leading formal
    # metric (Top1 hits); otherwise keep the frozen sequence score.
    accepted = int(candidate_row["fantom_top1_hits"]) > int(
        baseline_row["fantom_top1_hits"]
    )
    accepted_id = CANDIDATE_ID if accepted else BASELINE_ID
    final_score = candidate if accepted else baseline

    selected_prior = json.loads(
        M08_SELECTED_PRIOR__10.read_text(encoding="utf-8")
    )
    final_parameters = json.loads(
        M09_FINAL_PARAMETERS.read_text(encoding="utf-8")
    )
    candidate_contract = json.loads(
        M09_CONTRACT.read_text(encoding="utf-8")
    )

    valid_output = pd.DataFrame(
        {
            ID_COL__10: info[ID_COL__10].to_numpy(),
            SPLIT_COL__10: info[SPLIT_COL__10].to_numpy(),
            FANTOM_LABEL__10: labels,
            BASELINE_SCORE_COLUMN: baseline,
            CANDIDATE_SCORE_COLUMN: candidate,
            FINAL_SCORE_COLUMN: final_score,
        }
    )
    atomic_write_parquet__10(valid_output, OUT_VALID_PREDICTIONS__10)

    decision = {
        "script": SCRIPT_ID__10,
        "created_at_utc": utc_now__10(),
        "accepted_score_id": accepted_id,
        "biological_prior_accepted": bool(accepted),
        "final_score_column": FINAL_SCORE_COLUMN,
        "baseline_score_id": BASELINE_ID,
        "candidate_score_id": CANDIDATE_ID,
        "selected_candidate_id": selected_prior.get(
            "selected_candidate_id"
        ),
        "prior_type": final_parameters.get("prior_type"),
        "C": final_parameters.get("C"),
        "formal_metric_order": FORMAL_METRIC_ORDER__10,
        "baseline_metrics": baseline_row,
        "candidate_metrics": candidate_row,
        "test_used": False,
        "model_selection_performed": True,
    }
    atomic_write_json__10(OUT_ROUTE_DECISION, decision)

    contract = {
        "script": SCRIPT_ID__10,
        "created_at_utc": utc_now__10(),
        "accepted_score_id": accepted_id,
        "biological_prior_accepted": bool(accepted),
        "final_score_column": FINAL_SCORE_COLUMN,
        "baseline_score_column": BASELINE_SCORE_COLUMN,
        "candidate_score_column": CANDIDATE_SCORE_COLUMN,
        "feature_order": final_parameters.get("feature_order"),
        "divergence_mean": final_parameters.get("divergence_mean"),
        "divergence_std": final_parameters.get("divergence_std"),
        "intercept": final_parameters.get("intercept"),
        "coefficients": final_parameters.get("coefficients"),
        "prior_type": final_parameters.get("prior_type"),
        "C": final_parameters.get("C"),
        "candidate_contract": candidate_contract,
        "test_used": False,
    }
    atomic_write_json__10(OUT_FROZEN_CONTRACT_JSON, contract)
    atomic_joblib_dump__10(contract, OUT_FROZEN_CONTRACT_JOBLIB)

    calibrator = joblib.load(M09_CANDIDATE_CALIBRATOR)
    atomic_joblib_dump__10(calibrator, OUT_FROZEN_CALIBRATOR__10)


# --------------------------------------------------------------------------
# step 12: Export the biological refinement predictions
# --------------------------------------------------------------------------



# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_ROOT__12 = Path(__file__).resolve().parent.parent
MODELING_DIR__12 = (PROJECT_ROOT__12 / "Modeling").resolve()

SCRIPT_NAME__12 = "12_export_bio_refinement_predictions"
SCRIPT_ID__12 = SCRIPT_NAME__12


M01_DIR__12 = MODELING_DIR__12 / "output" / "01_build_dual_tensors"
TRAIN_TENSOR__12 = M01_DIR__12 / "01_train_dual_tensors.pt"
VALID_TENSOR__12 = M01_DIR__12 / "01_valid_dual_tensors.pt"
CANONICAL_TABLE__12 = MODELING_DIR__12 / 'output' / '00_build_canonical_data' / '00_canonical_herv_mef_table.parquet'

M07_DIR__12 = MODELING_DIR__12 / "output" / "07_audit_partial_unfreeze"
M07_FROZEN_CHECKPOINT__12 = M07_DIR__12 / '07_frozen' / '07_frozen_sequence_reference.pt'

M08_DIR__12 = MODELING_DIR__12 / "output" / "08_grid_search_biological_prior"
M08_FOLD_MANIFEST = M08_DIR__12 / "08_train_fold_manifest.tsv"

M10_DIR = MODELING_DIR__12 / "output" / "10_eval_bio_refinement_B"
M10_DECISION = M10_DIR / "10_final_representation_decision.json"
M10_CONTRACT_JSON = M10_DIR / '10_frozen' / '10_frozen_final_representation_contract.json'
M10_CALIBRATOR = M10_DIR / '10_frozen' / '10_frozen_biological_calibrator.joblib'

OUT_DIR__12 = MODELING_DIR__12 / "output" / SCRIPT_NAME__12
OUT_TRAIN_OOF = OUT_DIR__12 / "12_train_oof_representation_predictions.parquet"
OUT_VALID_FULL = OUT_DIR__12 / '12_valid_full_train_representation_predictions.parquet'

OUT_MODEL_CONTRACT__12 = OUT_DIR__12 / "12_representation_oof_contract.json"


INFERENCE_BATCH_SIZE__12 = 128
NUM_WORKERS__12 = 0
SEED__12 = 99999999
LOGISTIC_MAX_ITER__12 = 5000

META_EMBED_DIM__12 = 16
META_HIDDEN_DIM__12 = 64
K6_LENGTH__12 = 996
BPE_LENGTH__12 = 256

ID_COL__12 = "herv_internal_id"
SPLIT_COL__12 = "split"
FANTOM_LABEL__12 = "is_fantom5_enhancer"
DIVERGENCE_COLUMN__12 = "repeat_divergence"
FAMILY_COLUMN__12 = "repFamily"
FAMILIES__12 = ("ERV1", "ERVK", "ERVL", "ERVL-MALR", "GYPSY")

SCORE_CCRE = "score_ccre_enhancer"
SCORE_SERVE = "score_serve_gtex_support"
SCORE_SEQUENCE = "score_sequence_regulatory"
SCORE_FINAL = "score_final_representation"
LOGIT_EPSILON__12 = 1e-6


# =============================================================================
# 2. Utilities
# =============================================================================

def utc_now__12() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything__12(seed: int = SEED__12) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output__12() -> None:
    if OUT_DIR__12.exists():
        shutil.rmtree(OUT_DIR__12)
    OUT_DIR__12.mkdir(parents=True, exist_ok=True)


def atomic_write_json__12(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_parquet__12(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def probability_logit__12(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        LOGIT_EPSILON__12,
        1.0 - LOGIT_EPSILON__12,
    )
    return np.log(clipped / (1.0 - clipped))


# =============================================================================
# 3. Frozen M07 architecture (must match the 07 checkpoint exactly)
# =============================================================================

class PositionalEncoding__12(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class SequenceReferenceModel__12(nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.config = dict(config)
        self.cat_features = list(config["cat_features"])

        self.emb_k6 = nn.Embedding(
            int(config["vocab_k6"]), d_model, padding_idx=0
        )
        self.cnn_k6 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_k6 = PositionalEncoding__12(
            d_model, K6_LENGTH__12, float(config["dropout"])
        )
        self.trans_k6 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.emb_bpe = nn.Embedding(
            int(config["vocab_bpe"]), d_model, padding_idx=0
        )
        self.cnn_bpe = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_bpe = PositionalEncoding__12(
            d_model, BPE_LENGTH__12, float(config["dropout"])
        )
        self.trans_bpe = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config["num_layers"]),
        )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(config["cat_cardinalities"][feature]),
                    META_EMBED_DIM__12,
                    padding_idx=0,
                )
                for feature in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM__12
                + int(config["num_num_features"]),
                META_HIDDEN_DIM__12,
            ),
            nn.LayerNorm(META_HIDDEN_DIM__12),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM__12, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM__12, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
                for _ in range(3)
            ]
        )

    def forward(
        self,
        k6_ids: torch.Tensor,
        k6_mask: torch.Tensor,
        bpe_ids: torch.Tensor,
        bpe_mask: torch.Tensor,
        cat_ids: torch.Tensor,
        num_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_k6 = self.emb_k6(k6_ids)
        f_k6 = self.cnn_k6(f_k6.transpose(1, 2)).transpose(1, 2)
        f_k6 = self.pos_k6(f_k6)
        f_k6 = self.trans_k6(
            f_k6, src_key_padding_mask=(k6_mask == 0)
        )[:, 0, :]

        f_bpe = self.emb_bpe(bpe_ids)
        f_bpe = self.cnn_bpe(f_bpe.transpose(1, 2)).transpose(1, 2)
        f_bpe = self.pos_bpe(f_bpe)
        f_bpe = self.trans_bpe(
            f_bpe, src_key_padding_mask=(bpe_mask == 0)
        )[:, 0, :]

        categorical = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(categorical + [num_values], dim=1)
        )

        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6
            + gate_bpe * f_bpe
            + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)

        head_outputs = [head(fused).squeeze(-1) for head in self.heads]
        return head_outputs[0], head_outputs[1]


class TensorDatasetRows__12(Dataset):
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["rows"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
        }


# =============================================================================
# 4. Inference and calibration
# =============================================================================

def build_feature_matrix__12(
    sequence_score: np.ndarray,
    divergence: np.ndarray,
    divergence_mean: float,
    divergence_std: float,
    family: np.ndarray,
) -> np.ndarray:
    if divergence_std <= 0.0:
        divergence_std = 1.0
    sequence_logit = probability_logit__12(sequence_score)
    divergence_z = (divergence - divergence_mean) / divergence_std
    family = np.asarray(family, dtype=str)
    inds = {name: (family == name).astype(np.float64) for name in FAMILIES__12}
    return np.column_stack(
        [sequence_logit]
        + [inds[name] for name in FAMILIES__12]
        + [inds[name] * divergence_z for name in FAMILIES__12]
    ).astype(np.float64, copy=False)


def fit_fold_calibrator(
    fit_sequence: np.ndarray,
    fit_divergence: np.ndarray,
    fit_family: np.ndarray,
    fit_labels: np.ndarray,
    held_sequence: np.ndarray,
    held_divergence: np.ndarray,
    held_family: np.ndarray,
    c_value: float,
) -> np.ndarray:
    fit_mean = float(fit_divergence.mean())
    fit_std = float(fit_divergence.std(ddof=0))
    x_fit = build_feature_matrix__12(
        fit_sequence, fit_divergence, fit_mean, fit_std, fit_family
    )
    x_held = build_feature_matrix__12(
        held_sequence, held_divergence, fit_mean, fit_std, held_family
    )
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER__12,
        fit_intercept=True,
        random_state=SEED__12,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit, fit_labels)
    return model.predict_proba(x_held)[:, 1].astype(np.float64)


@torch.inference_mode()
def infer_sequence_score(
    checkpoint: dict[str, Any],
    bundle: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model = SequenceReferenceModel__12(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loader = DataLoader(
        TensorDatasetRows__12(bundle),
        batch_size=INFERENCE_BATCH_SIZE__12,
        shuffle=False,
        num_workers=NUM_WORKERS__12,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(NUM_WORKERS__12 > 0),
    )

    ccre_parts: list[np.ndarray] = []
    serve_parts: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        with torch.amp.autocast(device_type="cuda"):
            ccre_logit, serve_logit = model(
                batch["k6_ids"].to(device, non_blocking=True),
                batch["k6_mask"].to(device, non_blocking=True),
                batch["bpe_ids"].to(device, non_blocking=True),
                batch["bpe_mask"].to(device, non_blocking=True),
                batch["cat_ids"].to(device, non_blocking=True),
                batch["num_values"].to(device, non_blocking=True),
            )
        ccre_parts.append(
            torch.sigmoid(ccre_logit).detach().float().cpu().numpy()
        )
        serve_parts.append(
            torch.sigmoid(serve_logit).detach().float().cpu().numpy()
        )

    ccre = np.concatenate(ccre_parts).astype(np.float64)
    serve = np.concatenate(serve_parts).astype(np.float64)
    sequence = np.mean(np.column_stack([ccre, serve]), axis=1)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ccre, serve, sequence


def build_split_frame(
    info: pd.DataFrame,
    ccre_score: np.ndarray,
    serve_score: np.ndarray,
    sequence_score: np.ndarray,
    final_score: np.ndarray,
    origin: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL__12: info[ID_COL__12].to_numpy(),
            SPLIT_COL__12: info[SPLIT_COL__12].to_numpy(),
            SCORE_CCRE: ccre_score.astype(np.float64),
            SCORE_SERVE: serve_score.astype(np.float64),
            SCORE_SEQUENCE: sequence_score.astype(np.float64),
            SCORE_FINAL: final_score.astype(np.float64),
            "prediction_origin": origin,
        }
    )


# =============================================================================
# 5. Main
# =============================================================================

def step_12() -> None:
    seed_everything__12()
    prepare_output__12()


    decision = json.loads(M10_DECISION.read_text(encoding="utf-8"))
    contract = json.loads(M10_CONTRACT_JSON.read_text(encoding="utf-8"))
    calibrator = joblib.load(M10_CALIBRATOR)
    accepted = bool(decision.get("biological_prior_accepted", False))
    divergence_mean = float(contract.get("divergence_mean", 0.0))
    divergence_std = float(contract.get("divergence_std", 1.0))
    c_value = float(contract.get("C", 1.0))
    fold_manifest = pd.read_csv(M08_FOLD_MANIFEST, sep="\t")
    fold_manifest[ID_COL__12] = fold_manifest[ID_COL__12].astype(str)

    checkpoint = torch.load(
        M07_FROZEN_CHECKPOINT__12, map_location="cpu", weights_only=False
    )
    device = torch.device("cuda:0")

    frames: dict[str, pd.DataFrame] = {}
    for split_name, tensor_path, out_path, origin in (
        ("train", TRAIN_TENSOR__12, OUT_TRAIN_OOF, "m12_frozen_full_train"),
        ("valid", VALID_TENSOR__12, OUT_VALID_FULL, "m12_frozen_full_train"),
    ):
        bundle = torch.load(
            tensor_path, map_location="cpu", weights_only=False
        )
        info = bundle["info_df"].reset_index(drop=True).copy()
        canonical = pd.read_parquet(
            CANONICAL_TABLE__12, columns=[ID_COL__12, FAMILY_COLUMN__12, DIVERGENCE_COLUMN__12]
        )
        info = info.merge(canonical, on=ID_COL__12, how="left")
        info[DIVERGENCE_COLUMN__12] = pd.to_numeric(
            info[DIVERGENCE_COLUMN__12], errors="raise"
        ).astype(np.float64)
        info[FAMILY_COLUMN__12] = info[FAMILY_COLUMN__12].astype(str)

        ccre_score, serve_score, sequence_score = infer_sequence_score(
            checkpoint, bundle, device
        )
        if split_name == "train":
            info = info.merge(
                fold_manifest[[ID_COL__12, "fold_id"]], on=ID_COL__12, how="left"
            )
            fold_id = info["fold_id"].to_numpy(dtype=np.int8)
            divergence = info[DIVERGENCE_COLUMN__12].to_numpy(dtype=np.float64)
            family = info[FAMILY_COLUMN__12].str.upper().to_numpy(dtype=str)
            labels = info[FANTOM_LABEL__12].to_numpy(dtype=np.int8)
            final_score = np.zeros(len(info), dtype=np.float64)
            for fold in range(5):
                fit_mask = fold_id != fold
                held_mask = fold_id == fold
                if held_mask.sum() == 0:
                    continue
                final_score[held_mask] = fit_fold_calibrator(
                    sequence_score[fit_mask],
                    divergence[fit_mask],
                    family[fit_mask],
                    labels[fit_mask],
                    sequence_score[held_mask],
                    divergence[held_mask],
                    family[held_mask],
                    c_value,
                )
        else:
            x_split = build_feature_matrix__12(
                sequence_score,
                info[DIVERGENCE_COLUMN__12].to_numpy(dtype=np.float64),
                divergence_mean,
                divergence_std,
                info[FAMILY_COLUMN__12].str.upper().to_numpy(dtype=str),
            )
            if accepted:
                final_score = calibrator.predict_proba(x_split)[:, 1]
            else:
                final_score = sequence_score

        frame = build_split_frame(
            info, ccre_score, serve_score, sequence_score, final_score, origin
        )
        if FANTOM_LABEL__12 in info.columns:
            frame[FANTOM_LABEL__12] = pd.to_numeric(
                info[FANTOM_LABEL__12], errors="raise"
            ).astype(np.int8)
        atomic_write_parquet__12(frame, out_path)
        frames[split_name] = frame


    atomic_write_json__12(
        OUT_MODEL_CONTRACT__12,
        {
            "script": SCRIPT_ID__12,
            "created_at_utc": utc_now__12(),
            "accepted_score_id": decision.get("accepted_score_id"),
            "biological_prior_accepted": accepted,
            "score_sequence_column": SCORE_SEQUENCE,
            "score_final_column": SCORE_FINAL,
            "divergence_mean": divergence_mean,
            "divergence_std": divergence_std,
            "prediction_origin": "m12_frozen_full_train_inference",
            "strict_deep_oof": True,
            "test_used": False,
        },
    )


def main() -> None:
    step_08()
    step_09()
    step_10()
    step_12()


if __name__ == "__main__":
    main()
