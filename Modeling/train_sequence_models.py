#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from modeling_common import (
    safe_ap,
    safe_auc,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

# --------------------------------------------------------------------------
# step 02: Train the dual K6/BPE sequence model
# --------------------------------------------------------------------------



ROOT = Path(__file__).resolve().parent
M01_DIR = ROOT / "output" / "01_build_dual_tensors"
TRAIN_TENSOR = M01_DIR / "01_train_dual_tensors.pt"
VALID_TENSOR = M01_DIR / "01_valid_dual_tensors.pt"

OUT_DIR = ROOT / "output" / "02_train_dual"
CKPT_DIR = OUT_DIR / "02_checkpoints"
OUT_BEST = CKPT_DIR / "02_dual_best.pt"
OUT_LAST = CKPT_DIR / "02_dual_last.pt"


TARGETS = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
HEADS = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]

ACTIVE_HEAD_INDICES = [0, 1, 2]
ACTIVE_HEADS = [HEADS[index] for index in ACTIVE_HEAD_INDICES]
SCORE_HEAD_INDICES = [0, 1]

K6_LENGTH = 996
BPE_LENGTH = 256

BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
NUM_WORKERS = 0
MAX_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
SEED = 99999999

BALANCE_WARMUP_EPOCHS = 2
GRADNORM_ALPHA = 1.5
GRADNORM_WEIGHT_LR = 0.025
GRADNORM_ADAM_BETA1 = 0.90
GRADNORM_ADAM_BETA2 = 0.999
GRADNORM_EPS = 1e-8
TASK_WEIGHT_POSITIVE_FLOOR = 1e-4
CALIBRATION_BATCH_SIZE = 64
CALIBRATION_SEED = 99999999

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
META_EMBED_DIM = 16
META_HIDDEN_DIM = 64


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualDataset(Dataset):

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["labels"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
            "targets": self.bundle["labels"][index].float(),
        }


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class DualModel(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.cat_features = list(config["cat_features"])

        def branch(vocab_size: int, max_len: int) -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "emb": nn.Embedding(
                        int(vocab_size), d_model, padding_idx=0
                    ),
                    "cnn": nn.Sequential(
                        nn.Conv1d(d_model, d_model, 3, padding=1),
                        nn.BatchNorm1d(d_model),
                        nn.GELU(),
                    ),
                    "pos": PositionalEncoding(
                        d_model,
                        max_len=max_len,
                        dropout=float(config["dropout"]),
                    ),
                    "trans": nn.TransformerEncoder(
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
                    ),
                }
            )

        self.k6 = branch(int(config["vocab_k6"]), K6_LENGTH)
        self.bpe = branch(int(config["vocab_bpe"]), BPE_LENGTH)

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
                nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1)
                )
                for _ in HEADS
            ]
        )

    def encode_branch(
        self, branch: nn.ModuleDict, ids: torch.Tensor, mask: torch.Tensor
    ):
        values = branch["emb"](ids)
        values = branch["cnn"](values.transpose(1, 2)).transpose(1, 2)
        values = branch["pos"](values)
        return branch["trans"](values, src_key_padding_mask=(mask == 0))[
            :, 0, :
        ]

    def forward(
        self,
        k6_ids,
        k6_mask,
        bpe_ids,
        bpe_mask,
        cat_ids,
        num_values,
    ) -> list[torch.Tensor]:
        f_k6 = self.encode_branch(self.k6, k6_ids, k6_mask)
        f_bpe = self.encode_branch(self.bpe, bpe_ids, bpe_mask)

        cat_embeddings = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(cat_embeddings + [num_values], dim=1)
        )

        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)

        return [head(fused).squeeze(-1) for head in self.heads]






def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> list:
    return [
        batch["k6_ids"].to(device, non_blocking=True),
        batch["k6_mask"].to(device, non_blocking=True),
        batch["bpe_ids"].to(device, non_blocking=True),
        batch["bpe_mask"].to(device, non_blocking=True),
        batch["cat_ids"].to(device, non_blocking=True),
        batch["num_values"].to(device, non_blocking=True),
    ]


def pos_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    labels = labels.float()
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return (negatives / positives).to(device=device, dtype=torch.float32)


def normalize_positive_task_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, TASK_WEIGHT_POSITIVE_FLOOR)
    values /= values.sum()
    return values


def build_gradnorm_calibration_sets(
    train_bundle: dict, train_dataset: DualDataset
) -> tuple[dict, list]:
    labels = train_bundle["labels"].detach().cpu().numpy()
    binary = (labels >= 0.5).astype(np.int8)
    positive_counts = binary.sum(axis=0).astype(int)
    per_class = int(positive_counts.min())

    loaders: dict[str, DataLoader] = {}
    manifest_rows: list[dict] = []
    for task_index, head in enumerate(HEADS):
        positive_indices = np.flatnonzero(binary[:, task_index] == 1)
        negative_indices = np.flatnonzero(binary[:, task_index] == 0)
        rng = np.random.default_rng(CALIBRATION_SEED + task_index)
        chosen_positive = np.sort(
            rng.choice(positive_indices, size=per_class, replace=False)
        )
        chosen_negative = np.sort(
            rng.choice(negative_indices, size=per_class, replace=False)
        )
        selected = np.concatenate([chosen_positive, chosen_negative])
        subset = Subset(train_dataset, selected.tolist())
        loaders[head] = DataLoader(
            subset,
            batch_size=CALIBRATION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        for class_name, class_value, indices in [
            ("positive", 1, chosen_positive),
            ("negative", 0, chosen_negative),
        ]:
            for index in indices:
                manifest_rows.append(
                    {
                        "task_index": task_index,
                        "task": head,
                        "class": class_name,
                        "class_value": class_value,
                        "train_index": int(index),
                    }
                )
    return loaders, manifest_rows


def flatten_gradients(gradients: tuple) -> torch.Tensor:
    return torch.cat([g.detach().float().reshape(-1) for g in gradients])


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.detach().cpu()) <= 0:
        return float("nan")
    return float((torch.dot(left, right) / denominator).detach().cpu())


def compute_calibration_statistics(
    model: nn.Module,
    calibration_loaders: dict[str, DataLoader],
    device: torch.device,
    shared_parameters: list,
) -> dict:
    was_training = model.training
    model.eval()
    losses = np.zeros(len(HEADS), dtype=np.float64)
    gradient_vectors: list[torch.Tensor] = []

    for task_index, head in enumerate(HEADS):
        loader = calibration_loaders[head]
        task_loss_sum = 0.0
        task_gradient_sum: torch.Tensor | None = None
        task_count = 0
        for batch in loader:
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                logits = model(*move_inputs(batch, device))
                task_loss = F.binary_cross_entropy_with_logits(
                    logits[task_index].float(),
                    targets[:, task_index].float(),
                    reduction="mean",
                )
            gradients = torch.autograd.grad(
                task_loss,
                shared_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            vector = flatten_gradients(gradients)
            batch_rows = int(targets.shape[0])
            task_loss_sum += float(task_loss.detach().cpu()) * batch_rows
            if task_gradient_sum is None:
                task_gradient_sum = vector * batch_rows
            else:
                task_gradient_sum = task_gradient_sum + vector * batch_rows
            task_count += batch_rows
        losses[task_index] = task_loss_sum / max(task_count, 1)
        gradient_vectors.append(task_gradient_sum / max(task_count, 1))

    if was_training:
        model.train()
    return {
        "losses": losses,
        "raw_grad_norms": np.asarray(
            [float(torch.linalg.vector_norm(v).detach().cpu()) for v in gradient_vectors],
            dtype=np.float64,
        ),
    }


class CalibrationGradNormController:
    def __init__(self) -> None:
        self.weights = np.full(len(HEADS), 1.0 / len(HEADS), dtype=np.float64)
        self.initial_losses: np.ndarray | None = None
        self.adam_m = np.zeros(len(HEADS), dtype=np.float64)
        self.adam_v = np.zeros(len(HEADS), dtype=np.float64)
        self.adam_step = 0

    def initialize(self, calibration_losses: np.ndarray) -> None:
        self.initial_losses = np.asarray(calibration_losses, dtype=np.float64).copy()

    def update_from_calibration(
        self, epoch: int, calibration_losses: np.ndarray, raw_grad_norms: np.ndarray
    ) -> dict:
        losses = np.asarray(calibration_losses, dtype=np.float64)
        raw = np.asarray(raw_grad_norms, dtype=np.float64)
        relative_loss = losses / np.maximum(self.initial_losses, GRADNORM_EPS)
        relative_rates = relative_loss / np.maximum(relative_loss.mean(), GRADNORM_EPS)
        weighted_grad = self.weights * raw
        mean_weighted_grad = float(weighted_grad.mean())
        target_grad = mean_weighted_grad * np.power(relative_rates, GRADNORM_ALPHA)

        if epoch >= BALANCE_WARMUP_EPOCHS:
            weight_gradient = np.sign(weighted_grad - target_grad) * raw
            self.adam_step += 1
            self.adam_m = (
                GRADNORM_ADAM_BETA1 * self.adam_m
                + (1.0 - GRADNORM_ADAM_BETA1) * weight_gradient
            )
            self.adam_v = (
                GRADNORM_ADAM_BETA2 * self.adam_v
                + (1.0 - GRADNORM_ADAM_BETA2) * np.square(weight_gradient)
            )
            m_hat = self.adam_m / (1.0 - GRADNORM_ADAM_BETA1 ** self.adam_step)
            v_hat = self.adam_v / (1.0 - GRADNORM_ADAM_BETA2 ** self.adam_step)
            proposed = self.weights - GRADNORM_WEIGHT_LR * m_hat / (
                np.sqrt(v_hat) + GRADNORM_EPS
            )
            self.weights = normalize_positive_task_weights(proposed)

        return {"weights": self.weights.copy()}

    def tensor_weights(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.weights, device=device, dtype=torch.float32)


def train_one_epoch(
    model: DualModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pos_weights_tensor: torch.Tensor,
    task_weights_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for step, batch in enumerate(loader, start=1):
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(*move_inputs(batch, device))
            head_losses = [
                F.binary_cross_entropy_with_logits(
                    logits[index],
                    targets[:, index],
                    pos_weight=pos_weights_tensor[index],
                )
                for index in ACTIVE_HEAD_INDICES
            ]
            loss = sum(
                task_weights_tensor[index] * head_losses[i]
                for i, index in enumerate(ACTIVE_HEAD_INDICES)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        batch_count += 1
    return total_loss / max(batch_count, 1)


def evaluate(
    model: DualModel,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[list[np.ndarray]] = [[] for _ in HEADS]
    target_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(*move_inputs(batch, device))
            for index in range(len(HEADS)):
                predictions[index].append(
                    torch.sigmoid(logits[index]).detach().cpu().numpy()
                )
            target_parts.append(targets.detach().cpu().numpy())

    labels = np.concatenate(target_parts, axis=0).astype(np.int8)
    predictions = [
        np.concatenate(parts, axis=0).astype(np.float64)
        for parts in predictions
    ]

    metrics: dict[str, float] = {}
    active_aps = []
    active_aucs = []
    for index, head in enumerate(HEADS):
        ap = safe_ap(labels[:, index], predictions[index])
        auc = safe_auc(labels[:, index], predictions[index])
        metrics[f"valid_{head}_ap"] = ap
        metrics[f"valid_{head}_auc"] = auc
        if index in ACTIVE_HEAD_INDICES:
            active_aps.append(ap)
            active_aucs.append(auc)
    metrics["valid_macro_mean_ap"] = float(np.mean(active_aps))
    metrics["valid_macro_mean_auc"] = float(np.mean(active_aucs))
    return metrics


def build_config(train_bundle: dict) -> dict:
    return {
        "vocab_k6": int(train_bundle["k6_vocab_size"]),
        "vocab_bpe": int(train_bundle["bpe_vocab_size"]),
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "cat_features": list(train_bundle["cat_features"]),
        "cat_cardinalities": {
            str(key): int(value)
            for key, value in train_bundle["cat_cardinalities"].items()
        },
        "num_num_features": len(train_bundle["num_features"]),
        "meta_embed_dim": META_EMBED_DIM,
        "meta_hidden_dim": META_HIDDEN_DIM,
        "head_order": HEADS,
        "target_order": TARGETS,
        "active_heads": ACTIVE_HEADS,
        "score_heads": [HEADS[index] for index in SCORE_HEAD_INDICES],
        "sequence_score_contract": (
            "mean of sigmoid probabilities from ccre_enhancer and "
            "serve_gtex_support heads; mpra_activity is trained but closed "
            "for the final sequence score"
        ),
    }


def step_02() -> None:
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_bundle = torch.load(TRAIN_TENSOR, weights_only=False)
    valid_bundle = torch.load(VALID_TENSOR, weights_only=False)

    config = build_config(train_bundle)
    train_dataset = DualDataset(train_bundle)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        DualDataset(valid_bundle),
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    weights = pos_weights(train_bundle["labels"], device)

    model = DualModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")
    calibration_loaders, _ = build_gradnorm_calibration_sets(
        train_bundle, train_dataset
    )
    balance_controller = CalibrationGradNormController()
    shared_balance_parameters = [model.fuse_norm.weight, model.fuse_norm.bias]
    initial_stats = compute_calibration_statistics(
        model, calibration_loaders, device, shared_balance_parameters
    )
    balance_controller.initialize(initial_stats["losses"])

    history = []
    best_ap = -np.inf
    for epoch in range(1, MAX_EPOCHS + 1):
        task_weights_tensor = balance_controller.tensor_weights(device)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, weights, task_weights_tensor, device
        )
        metrics = evaluate(model, valid_loader, weights, device)
        calibration_stats = compute_calibration_statistics(
            model, calibration_loaders, device, shared_balance_parameters
        )
        balance_controller.update_from_calibration(
            epoch, calibration_stats["losses"], calibration_stats["raw_grad_norms"]
        )
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        row.update(
            {
                f"task_weight_{head}": float(balance_controller.weights[i])
                for i, head in enumerate(HEADS)
            }
        )
        history.append(row)

        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "metrics": metrics,
            },
            OUT_LAST,
        )
        if metrics["valid_macro_mean_ap"] > best_ap:
            best_ap = metrics["valid_macro_mean_ap"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                OUT_BEST,
            )


# --------------------------------------------------------------------------
# step 03: Audit the dual sequence model
# --------------------------------------------------------------------------



ROOT__03 = Path(__file__).resolve().parent
M01_DIR__03 = ROOT__03 / "output" / "01_build_dual_tensors"
VALID_TENSOR__03 = M01_DIR__03 / "01_valid_dual_tensors.pt"
M02_DIR = ROOT__03 / "output" / "02_train_dual" / "02_checkpoints"
BEST_CKPT = M02_DIR / "02_dual_best.pt"
LAST_CKPT = M02_DIR / "02_dual_last.pt"

OUT_DIR__03 = ROOT__03 / "output" / "03_audit_dual"

OUT_PREDICTIONS = OUT_DIR__03 / "03_dual_validation_predictions.parquet"

OUT_GATE_DIMENSION = OUT_DIR__03 / "03_selected_gate_dimension_summary.tsv"
OUT_GATE_GROUP = OUT_DIR__03 / "03_selected_gate_group_summary.tsv"

TARGETS__03 = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
HEADS__03 = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
SCORE_HEAD_INDICES__03 = [0, 1]

FANTOM_LABEL = "is_fantom5_enhancer"
ID_COL = "herv_internal_id"

K6_LENGTH__03 = 996
BPE_LENGTH__03 = 256
VALID_BATCH_SIZE__03 = 128
NUM_WORKERS__03 = 0
SEED__03 = 99999999

RANDOM_1TO1_REPEATS = 100
RANDOM_1TO1_SEED = 99999999
TOPK_FRACTIONS = (0.005, 0.010, 0.050)
TOPK_TAGS = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}


class PositionalEncoding__03(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class DualModel__03(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.cat_features = list(config["cat_features"])

        def branch(vocab_size: int, max_len: int) -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "emb": nn.Embedding(
                        int(vocab_size), d_model, padding_idx=0
                    ),
                    "cnn": nn.Sequential(
                        nn.Conv1d(d_model, d_model, 3, padding=1),
                        nn.BatchNorm1d(d_model),
                        nn.GELU(),
                    ),
                    "pos": PositionalEncoding__03(
                        d_model,
                        max_len=max_len,
                        dropout=float(config["dropout"]),
                    ),
                    "trans": nn.TransformerEncoder(
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
                    ),
                }
            )

        self.k6 = branch(int(config["vocab_k6"]), K6_LENGTH__03)
        self.bpe = branch(int(config["vocab_bpe"]), BPE_LENGTH__03)

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(config["cat_cardinalities"][feature]),
                    int(config["meta_embed_dim"]),
                    padding_idx=0,
                )
                for feature in self.cat_features
            ]
        )
        meta_embed_dim = int(config["meta_embed_dim"])
        meta_hidden_dim = int(config["meta_hidden_dim"])
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * meta_embed_dim
                + int(config["num_num_features"]),
                meta_hidden_dim,
            ),
            nn.LayerNorm(meta_hidden_dim),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + meta_hidden_dim, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(meta_hidden_dim, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1)
                )
                for _ in HEADS__03
            ]
        )

    def encode_branch(self, branch: nn.ModuleDict, ids, mask):
        values = branch["emb"](ids)
        values = branch["cnn"](values.transpose(1, 2)).transpose(1, 2)
        values = branch["pos"](values)
        return branch["trans"](values, src_key_padding_mask=(mask == 0))[
            :, 0, :
        ]

    def forward(self, k6_ids, k6_mask, bpe_ids, bpe_mask, cat_ids, num_values):
        f_k6 = self.encode_branch(self.k6, k6_ids, k6_mask)
        f_bpe = self.encode_branch(self.bpe, bpe_ids, bpe_mask)
        cat_embeddings = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(cat_embeddings + [num_values], dim=1)
        )
        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)
        return [head(fused).squeeze(-1) for head in self.heads]

    def forward_with_audit(
        self, k6_ids, k6_mask, bpe_ids, bpe_mask, cat_ids, num_values
    ):
        f_k6 = self.encode_branch(self.k6, k6_ids, k6_mask)
        f_bpe = self.encode_branch(self.bpe, bpe_ids, bpe_mask)
        cat_embeddings = [
            self.cat_embeddings[index](cat_ids[:, index])
            for index in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(
            torch.cat(cat_embeddings + [num_values], dim=1)
        )
        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)
        logits = [head(fused).squeeze(-1) for head in self.heads]
        return (*logits, gate_bpe, gate_meta, fused)


class ValidDataset(Dataset):

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["input_ids_k6"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
        }






def random_1to1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED)
    aps = []
    aucs = []
    for _ in range(RANDOM_1TO1_REPEATS):
        picked = rng.choice(negatives, size=len(positives), replace=False)
        indices = np.concatenate([positives, picked])
        aps.append(safe_ap(labels[indices], scores[indices]))
        aucs.append(safe_auc(labels[indices], scores[indices]))
    return {
        "auprc_mean": float(np.mean(aps)),
        "auprc_std": float(np.std(aps)),
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
    }


def topk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = int((labels == 1).sum())
    prevalence = positives / len(labels)
    out: dict[str, float] = {}
    for fraction in TOPK_FRACTIONS:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        tag = TOPK_TAGS[fraction]
        out[f"{tag}_k"] = k
        out[f"{tag}_hits"] = hits
        out[f"{tag}_fe"] = (
            float((hits / k) / prevalence) if prevalence > 0 else float("nan")
        )
    return out


def score_model(
    model: DualModel__03, loader: DataLoader, device: torch.device, d_model: int
) -> tuple[np.ndarray, dict, dict]:
    model.eval()
    parts = [[] for _ in HEADS__03]
    sample_gate = {
        "bpe_mean": [], "bpe_std": [],
        "meta_mean": [], "meta_std": [],
        "fused_norm": [],
    }
    gate_sums = {
        "bpe": np.zeros(d_model, dtype=np.float64),
        "bpe_sq": np.zeros(d_model, dtype=np.float64),
        "meta": np.zeros(d_model, dtype=np.float64),
        "meta_sq": np.zeros(d_model, dtype=np.float64),
        "rows": 0,
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            with torch.amp.autocast(device_type="cuda"):
                outputs = model.forward_with_audit(
                    batch["k6_ids"].to(device, non_blocking=True),
                    batch["k6_mask"].to(device, non_blocking=True),
                    batch["bpe_ids"].to(device, non_blocking=True),
                    batch["bpe_mask"].to(device, non_blocking=True),
                    batch["cat_ids"].to(device, non_blocking=True),
                    batch["num_values"].to(device, non_blocking=True),
                )
            logits = outputs[: len(HEADS__03)]
            for index in range(len(HEADS__03)):
                parts[index].append(
                    torch.sigmoid(logits[index]).detach().cpu().numpy()
                )
            gate_bpe = outputs[len(HEADS__03)].detach().float().cpu().numpy()
            gate_meta = outputs[len(HEADS__03) + 1].detach().float().cpu().numpy()
            fused = outputs[len(HEADS__03) + 2].detach().float().cpu().numpy()
            sample_gate["bpe_mean"].append(gate_bpe.mean(axis=1))
            sample_gate["bpe_std"].append(gate_bpe.std(axis=1, ddof=0))
            sample_gate["meta_mean"].append(gate_meta.mean(axis=1))
            sample_gate["meta_std"].append(gate_meta.std(axis=1, ddof=0))
            sample_gate["fused_norm"].append(np.linalg.norm(fused, axis=1))
            gate_sums["bpe"] += gate_bpe.sum(axis=0)
            gate_sums["bpe_sq"] += np.square(gate_bpe).sum(axis=0)
            gate_sums["meta"] += gate_meta.sum(axis=0)
            gate_sums["meta_sq"] += np.square(gate_meta).sum(axis=0)
            gate_sums["rows"] += int(gate_bpe.shape[0])
    probabilities = np.stack(
        [np.concatenate(part, axis=0).astype(np.float64) for part in parts],
        axis=1,
    )
    sample_gate = {
        key: np.concatenate(value, axis=0).astype(np.float64)
        for key, value in sample_gate.items()
    }
    return probabilities, sample_gate, gate_sums


def build_gate_dimension_summary(gate_sums: dict) -> pd.DataFrame:
    rows = int(gate_sums["rows"])
    records = []
    for gate_name in ("bpe", "meta"):
        mean = np.asarray(gate_sums[gate_name]) / rows
        second = np.asarray(gate_sums[f"{gate_name}_sq"]) / rows
        std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
        for dimension in range(len(mean)):
            records.append(
                {
                    "gate": gate_name,
                    "dimension": dimension,
                    "mean": float(mean[dimension]),
                    "std": float(std[dimension]),
                    "validation_rows": rows,
                }
            )
    return pd.DataFrame(records)


def build_gate_group_summary(
    labels: np.ndarray, sample_gate: dict
) -> pd.DataFrame:
    frame = pd.DataFrame({"fantom5": labels.astype(np.int8)})
    for key, values in sample_gate.items():
        frame[key] = values
    records = []
    for group_value, group in frame.groupby("fantom5", sort=True):
        records.append(
            {
                "fantom5": int(group_value),
                "rows": int(len(group)),
                "bpe_gate_mean": float(group["bpe_mean"].mean()),
                "meta_gate_mean": float(group["meta_mean"].mean()),
                "fused_embedding_norm_mean": float(group["fused_norm"].mean()),
            }
        )
    return pd.DataFrame(records)


def step_03() -> None:
    OUT_DIR__03.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    bundle = torch.load(VALID_TENSOR__03, weights_only=False)
    info = bundle["info_df"]
    labels = info[FANTOM_LABEL].to_numpy(dtype=np.int8)
    loader = DataLoader(
        ValidDataset(bundle),
        batch_size=VALID_BATCH_SIZE__03,
        shuffle=False,
        num_workers=NUM_WORKERS__03,
        pin_memory=False,
    )

    checkpoint = torch.load(BEST_CKPT, weights_only=False)
    config = checkpoint["config"]
    model = DualModel__03(config).to(device)
    model.load_state_dict(checkpoint["model"])

    probabilities, sample_gate, gate_sums = score_model(
        model, loader, device, int(config["d_model"])
    )
    score_ccre = probabilities[:, 0]
    score_serve = probabilities[:, 1]
    score_mpra = probabilities[:, 2]
    score_mean = probabilities[:, SCORE_HEAD_INDICES__03].mean(axis=1)

    frames = {
        "score_ccre_enhancer": score_ccre,
        "score_serve_gtex_support": score_serve,
        "score_mpra_activity": score_mpra,
        "score_sequence_mean": score_mean,
    }

    rows = []
    for name, scores in frames.items():
        row = {"score": name}
        row.update(random_1to1(labels, scores))
        row["full_auroc"] = safe_auc(labels, scores)
        row["full_auprc"] = safe_ap(labels, scores)
        row.update(topk_metrics(labels, scores))
        rows.append(row)

    gate_dimension = build_gate_dimension_summary(gate_sums)
    gate_dimension.to_csv(OUT_GATE_DIMENSION, sep="\t", index=False)
    gate_group = build_gate_group_summary(labels, sample_gate)
    gate_group.to_csv(OUT_GATE_GROUP, sep="\t", index=False)

    predictions = pd.DataFrame(
        {
            ID_COL: info[ID_COL].to_numpy(),
            FANTOM_LABEL: labels,
            **frames,
        }
    )
    predictions.to_parquet(OUT_PREDICTIONS, index=False)


# --------------------------------------------------------------------------
# step 04: Train the single-branch sequence ablations
# --------------------------------------------------------------------------



ARMS = {
    "k6_meta": {
        "gpu": "1",
        "sequence_mode": "k6",
        "active_head_indices": (0, 1, 2),
        "score_head_indices": (0, 1, 2),
    },
    "bpe_meta": {
        "gpu": "2",
        "sequence_mode": "bpe",
        "active_head_indices": (0, 1, 2),
        "score_head_indices": (0, 1, 2),
    },
    "full_dual_meta": {
        "gpu": "3",
        "sequence_mode": "dual",
        "active_head_indices": (0, 1, 2),
        "score_head_indices": (0, 1, 2),
    },
    "dual_no_mpra_meta": {
        "gpu": "4",
        "sequence_mode": "dual",
        "active_head_indices": (0, 1),
        "score_head_indices": (0, 1),
    },
}

ROOT__04 = Path(__file__).resolve().parent
M01_DIR__04 = ROOT__04 / "output" / "01_build_dual_tensors"
TRAIN_TENSOR__04 = M01_DIR__04 / "01_train_dual_tensors.pt"
VALID_TENSOR__04 = M01_DIR__04 / "01_valid_dual_tensors.pt"

OUT_DIR__04 = ROOT__04 / "output" / "04_train_sequence_branch_ablations"
CKPT_DIR__04 = OUT_DIR__04 / "04_checkpoints"
PRED_DIR = OUT_DIR__04 / "04_predictions"
OUT_PREDICTIONS__04 = OUT_DIR__04 / "04_validation_predictions.parquet"
OUT_METRICS = OUT_DIR__04 / "04_ablation_metrics.tsv"


TARGETS__04 = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
HEADS__04 = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
FANTOM_LABEL__04 = "is_fantom5_enhancer"
ID_COL__04 = "herv_internal_id"

K6_LENGTH__04 = 996
BPE_LENGTH__04 = 256

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE__04 = 128
NUM_WORKERS__04 = 0
MAX_EPOCHS__04 = 2
LR__04 = 1e-4
WEIGHT_DECAY__04 = 0.05
GRAD_CLIP__04 = 1.0
SEED__04 = 99999999

D_MODEL__04 = 128
NHEAD__04 = 8
NUM_LAYERS__04 = 4
DIM_FEEDFORWARD__04 = 512
DROPOUT__04 = 0.1
META_EMBED_DIM__04 = 16
META_HIDDEN_DIM__04 = 64

RANDOM_1TO1_REPEATS__04 = 100
RANDOM_1TO1_SEED__04 = 99999999
TOPK_FRACTIONS__04 = (0.005, 0.010, 0.050)
TOPK_TAGS__04 = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}


def seed_everything__04(seed: int = SEED__04) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualDataset__04(Dataset):

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["labels"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][index].long(),
            "k6_mask": self.bundle["mask_k6"][index].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][index].long(),
            "bpe_mask": self.bundle["mask_bpe"][index].long(),
            "cat_ids": self.bundle["cat_ids"][index].long(),
            "num_values": self.bundle["num_values"][index].float(),
            "targets": self.bundle["labels"][index].float(),
        }


class PositionalEncoding__04(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class BranchAblationModel(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.cat_features = list(config["cat_features"])
        self.sequence_mode = str(config["sequence_mode"])
        self.active_head_indices = tuple(config["active_head_indices"])
        self.score_head_indices = tuple(config["score_head_indices"])

        self.emb_k6 = nn.Embedding(
            int(config["vocab_k6"]), d_model, padding_idx=0
        )
        self.cnn_k6 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_k6 = PositionalEncoding__04(
            d_model, K6_LENGTH__04, float(config["dropout"])
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
        self.pos_bpe = PositionalEncoding__04(
            d_model, BPE_LENGTH__04, float(config["dropout"])
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
                    int(config["cat_cardinalities"][f]),
                    META_EMBED_DIM__04,
                    padding_idx=0,
                )
                for f in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM__04
                + int(config["num_num_features"]),
                META_HIDDEN_DIM__04,
            ),
            nn.LayerNorm(META_HIDDEN_DIM__04),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM__04, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM__04, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1)
                )
                for _ in HEADS__04
            ]
        )
        self._freeze_inactive()

    def _freeze(self, module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_inactive(self) -> None:
        if self.sequence_mode == "k6":
            for module in (
                self.emb_bpe,
                self.cnn_bpe,
                self.trans_bpe,
                self.bpe_gate,
            ):
                self._freeze(module)
        elif self.sequence_mode == "bpe":
            for module in (
                self.emb_k6,
                self.cnn_k6,
                self.trans_k6,
                self.bpe_gate,
            ):
                self._freeze(module)
        for index, head in enumerate(self.heads):
            if index not in self.active_head_indices:
                self._freeze(head)

    def _encode(
        self, prefix: str, ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values = getattr(self, f"emb_{prefix}")(ids)
        values = getattr(self, f"cnn_{prefix}")(values.transpose(1, 2)).transpose(
            1, 2
        )
        values = getattr(self, f"pos_{prefix}")(values)
        return getattr(self, f"trans_{prefix}")(
            values, src_key_padding_mask=(mask == 0)
        )[:, 0, :]

    def _encode_metadata(self, cat_ids, num_values) -> torch.Tensor:
        parts = [
            self.cat_embeddings[i](cat_ids[:, i])
            for i in range(len(self.cat_features))
        ]
        return self.metadata_mlp(torch.cat(parts + [num_values], dim=1))

    def forward(self, k6_ids, k6_mask, bpe_ids, bpe_mask, cat_ids, num_values):
        f_meta = self._encode_metadata(cat_ids, num_values)
        if self.sequence_mode == "k6":
            anchor = self._encode("k6", k6_ids, k6_mask)
            gate_meta = self.meta_gate(torch.cat([anchor, f_meta], dim=1))
            fused = self.fuse_norm(
                anchor + gate_meta * self.project_meta(f_meta)
            )
        elif self.sequence_mode == "bpe":
            anchor = self._encode("bpe", bpe_ids, bpe_mask)
            gate_meta = self.meta_gate(torch.cat([anchor, f_meta], dim=1))
            fused = self.fuse_norm(
                anchor + gate_meta * self.project_meta(f_meta)
            )
        else:
            f_k6 = self._encode("k6", k6_ids, k6_mask)
            f_bpe = self._encode("bpe", bpe_ids, bpe_mask)
            gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
            gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
            fused = self.fuse_norm(
                f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta)
            )
        fused = self.fuse_dropout(fused)
        return [head(fused).squeeze(-1) for head in self.heads]






def random_1to1__04(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED__04)
    aps, aucs = [], []
    for _ in range(RANDOM_1TO1_REPEATS__04):
        picked = rng.choice(negatives, size=len(positives), replace=False)
        idx = np.concatenate([positives, picked])
        aps.append(safe_ap(labels[idx], scores[idx]))
        aucs.append(safe_auc(labels[idx], scores[idx]))
    return {
        "auroc_1to1_mean": float(np.mean(aucs)),
        "auroc_1to1_std": float(np.std(aucs)),
        "auprc_1to1_mean": float(np.mean(aps)),
        "auprc_1to1_std": float(np.std(aps)),
    }


def topk_metrics__04(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = int((labels == 1).sum())
    prevalence = positives / len(labels)
    out = {}
    for fraction in TOPK_FRACTIONS__04:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        tag = TOPK_TAGS__04[fraction]
        out[f"{tag}_hits"] = hits
        out[f"{tag}_fe"] = (
            float((hits / k) / prevalence) if prevalence > 0 else float("nan")
        )
    return out


def build_config__04(train_bundle: dict, arm: str) -> dict:
    spec = ARMS[arm]
    return {
        "arm": arm,
        "sequence_mode": spec["sequence_mode"],
        "active_head_indices": list(spec["active_head_indices"]),
        "active_heads": [HEADS__04[i] for i in spec["active_head_indices"]],
        "score_head_indices": list(spec["score_head_indices"]),
        "score_heads": [HEADS__04[i] for i in spec["score_head_indices"]],
        "vocab_k6": int(train_bundle["k6_vocab_size"]),
        "vocab_bpe": int(train_bundle["bpe_vocab_size"]),
        "d_model": D_MODEL__04,
        "nhead": NHEAD__04,
        "num_layers": NUM_LAYERS__04,
        "dim_feedforward": DIM_FEEDFORWARD__04,
        "dropout": DROPOUT__04,
        "cat_features": list(train_bundle["cat_features"]),
        "cat_cardinalities": {
            str(k): int(v) for k, v in train_bundle["cat_cardinalities"].items()
        },
        "num_num_features": len(train_bundle["num_features"]),
        "head_order": HEADS__04,
        "target_order": TARGETS__04,
        "sequence_score_contract": (
            "mean of sigmoid probabilities over score heads"
        ),
    }


def infer(model, loader, device, active_head_indices, score_head_indices):
    model.eval()
    parts = [[] for _ in HEADS__04]
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(
                    batch["k6_ids"].to(device, non_blocking=True),
                    batch["k6_mask"].to(device, non_blocking=True),
                    batch["bpe_ids"].to(device, non_blocking=True),
                    batch["bpe_mask"].to(device, non_blocking=True),
                    batch["cat_ids"].to(device, non_blocking=True),
                    batch["num_values"].to(device, non_blocking=True),
                )
            for i in range(len(HEADS__04)):
                parts[i].append(
                    torch.sigmoid(logits[i]).detach().cpu().numpy()
                )
    probabilities = np.stack(
        [np.concatenate(p, axis=0).astype(np.float64) for p in parts], axis=1
    )
    for index in range(len(HEADS__04)):
        if index not in active_head_indices:
            probabilities[:, index] = np.nan
    return probabilities, probabilities[:, list(score_head_indices)].mean(
        axis=1
    )


def run_arm(arm: str) -> None:
    seed_everything__04()
    device = torch.device("cuda")
    spec = ARMS[arm]
    active = tuple(spec["active_head_indices"])
    score = tuple(spec["score_head_indices"])

    train_bundle = torch.load(TRAIN_TENSOR__04, weights_only=False)
    valid_bundle = torch.load(VALID_TENSOR__04, weights_only=False)
    config = build_config__04(train_bundle, arm)

    train_loader = DataLoader(
        DualDataset__04(train_bundle),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS__04,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        DualDataset__04(valid_bundle),
        batch_size=VALID_BATCH_SIZE__04,
        shuffle=False,
        num_workers=NUM_WORKERS__04,
        pin_memory=False,
    )

    labels = train_bundle["labels"].float()
    weights = ((labels.shape[0] - labels.sum(dim=0)) / labels.sum(dim=0)).to(
        device
    )

    model = BranchAblationModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR__04, weight_decay=WEIGHT_DECAY__04
    )
    scaler = torch.amp.GradScaler("cuda")

    history = []
    for epoch in range(1, MAX_EPOCHS__04 + 1):
        model.train()
        total, batches = 0.0, 0
        for batch in train_loader:
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(*move_inputs__04(batch, device))
                loss = sum(
                    F.binary_cross_entropy_with_logits(
                        logits[i], targets[:, i], pos_weight=weights[i]
                    )
                    for i in active
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP__04)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        history.append(
            {
                "arm": arm,
                "epoch": epoch,
                "train_loss": total / max(batches, 1),
            }
        )

    probabilities, sequence_score = infer(
        model, valid_loader, device, active, score
    )
    info = valid_bundle["info_df"].reset_index(drop=True)
    

    frame = pd.DataFrame(
        {
            ID_COL__04: info[ID_COL__04].to_numpy(),
            FANTOM_LABEL__04: info[FANTOM_LABEL__04].to_numpy(np.int8),
        }
    )
    for index, head in enumerate(HEADS__04):
        frame[f"score_{head}"] = probabilities[:, index]
    frame["score_sequence_mean"] = sequence_score
    frame.to_parquet(
        PRED_DIR / f"04_{arm}_validation_predictions.parquet", index=False
    )

    
    torch.save(
        {"model": model.state_dict(), "config": config, "epoch": MAX_EPOCHS__04},
        CKPT_DIR__04 / f"04_{arm}.pt",
    )
    pd.DataFrame(history).to_csv(
        PRED_DIR / f"04_{arm}_history.tsv", sep="\t", index=False
    )


def move_inputs__04(batch: dict, device: torch.device) -> tuple:
    return (
        batch["k6_ids"].to(device, non_blocking=True),
        batch["k6_mask"].to(device, non_blocking=True),
        batch["bpe_ids"].to(device, non_blocking=True),
        batch["bpe_mask"].to(device, non_blocking=True),
        batch["cat_ids"].to(device, non_blocking=True),
        batch["num_values"].to(device, non_blocking=True),
    )


def aggregate() -> None:
    frames = []
    for arm in ARMS:
        part = pd.read_parquet(
            PRED_DIR / f"04_{arm}_validation_predictions.parquet"
        )
        part.insert(0, "arm", arm)
        frames.append(part)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(OUT_PREDICTIONS__04, index=False)

    labels = predictions.loc[
        predictions["arm"] == "k6_meta", FANTOM_LABEL__04
    ].to_numpy(np.int8)
    rows = []
    for arm in ARMS:
        sub = predictions[predictions["arm"] == arm]
        spec = ARMS[arm]
        score_columns = [
            f"score_{HEADS__04[index]}" for index in spec["score_head_indices"]
        ] + ["score_sequence_mean"]
        for column in score_columns:
            scores = sub[column].to_numpy(np.float64)
            row = {"arm": arm, "score": column}
            row.update(random_1to1__04(labels, scores))
            row["full_auroc"] = safe_auc(labels, scores)
            row["full_auprc"] = safe_ap(labels, scores)
            row.update(topk_metrics__04(labels, scores))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_METRICS, sep="\t", index=False)


def step_04() -> None:
    OUT_DIR__04.mkdir(parents=True, exist_ok=True)
    CKPT_DIR__04.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        run_arm(arm)
    aggregate()


# --------------------------------------------------------------------------
# step 05: Train the K6/BPE complementarity controls
# --------------------------------------------------------------------------



ROOT__05 = Path(__file__).resolve().parent
M01_DIR__05 = ROOT__05 / "output/01_build_dual_tensors"
TRAIN_TENSOR__05 = M01_DIR__05 / "01_train_dual_tensors.pt"
VALID_TENSOR__05 = M01_DIR__05 / "01_valid_dual_tensors.pt"
M04_DIR = ROOT__05 / "output/04_train_sequence_branch_ablations"
M04_PREDICTIONS = M04_DIR / "04_validation_predictions.parquet"
M04_CKPT_DIR = M04_DIR / "04_checkpoints"

OUT_DIR__05 = ROOT__05 / "output/05_k6_bpe_complementarity_controls"
CKPT_DIR__05 = OUT_DIR__05 / "05_checkpoints"
FROZEN_DIR = OUT_DIR__05 / "05_frozen"
FROZEN_SEQUENCE = FROZEN_DIR / "05_frozen_sequence_reference.pt"
OUT_PREDICTIONS__05 = OUT_DIR__05 / "05_validation_predictions.parquet"


OUT_ROUTE_DECISION = OUT_DIR__05 / "05_route_decision.json"


TARGETS__05 = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
HEADS__05 = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
FANTOM_LABEL__05, ID_COL__05 = "is_fantom5_enhancer", "herv_internal_id"
K6_LENGTH__05, BPE_LENGTH__05 = 996, 256

SHUFFLED_MODEL, SHUFFLED_SEQUENCE_MODE = "k6_shuffled_bpe_meta", "dual"
SHUFFLED_ACTIVE_HEAD_INDICES, SHUFFLED_SCORE_HEAD_INDICES = (0, 1), (0, 1)

ELIGIBLE_ROUTE_MODELS = [
    "dual_no_mpra_meta",
    "full_dual_meta",
    "bpe_meta",
    "k6_meta",
]
ALL_MODELS = ELIGIBLE_ROUTE_MODELS + [SHUFFLED_MODEL]

TRAIN_BATCH_SIZE__05, VALID_BATCH_SIZE__05, NUM_WORKERS__05, MAX_EPOCHS__05 = 32, 128, 0, 2
LR__05, WEIGHT_DECAY__05, GRAD_CLIP__05, SEED__05 = 1e-4, 0.05, 1.0, 99999999
D_MODEL__05, NHEAD__05, NUM_LAYERS__05, DIM_FEEDFORWARD__05, DROPOUT__05 = 128, 8, 4, 512, 0.1
META_EMBED_DIM__05, META_HIDDEN_DIM__05 = 16, 64

RANDOM_1TO1_REPEATS__05, RANDOM_1TO1_SEED__05 = 100, 99999999
TOPK_FRACTIONS__05 = (0.005, 0.010, 0.050)
TOPK_TAGS__05 = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}
TRAIN_BPE_SHUFFLE_SEED, VALID_BPE_SHUFFLE_SEED = 2026080501, 2026080502

FORMAL_METRIC_ORDER = [
    "top1_hits",
    "top5_hits",
    "auprc_1to1_mean",
    "top0_5_hits",
    "auroc_1to1_mean",
]
MODEL_TIE_ORDER = {
    "full_dual_meta": 0,
    "dual_no_mpra_meta": 1,
    "bpe_meta": 2,
    "k6_meta": 3,
}


def seed_everything__05(seed: int = SEED__05) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualDataset__05(Dataset):

    def __init__(self, bundle: dict, bpe_permutation: np.ndarray | None = None):
        self.bundle, self.bpe_permutation = bundle, bpe_permutation

    def __len__(self) -> int:
        return int(self.bundle["labels"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        bpe_idx = (
            idx
            if self.bpe_permutation is None
            else int(self.bpe_permutation[idx])
        )
        return {
            "k6_ids": self.bundle["input_ids_k6"][idx].long(),
            "k6_mask": self.bundle["mask_k6"][idx].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][bpe_idx].long(),
            "bpe_mask": self.bundle["mask_bpe"][bpe_idx].long(),
            "cat_ids": self.bundle["cat_ids"][idx].long(),
            "num_values": self.bundle["num_values"][idx].float(),
            "targets": self.bundle["labels"][idx].float(),
        }


class PositionalEncoding__05(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ComplementarityModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        d_model, self.cat_features = int(cfg["d_model"]), list(
            cfg["cat_features"]
        )

        def make_branch(vocab, max_len):
            return nn.ModuleDict(
                {
                    "emb": nn.Embedding(int(vocab), d_model, padding_idx=0),
                    "cnn": nn.Sequential(
                        nn.Conv1d(d_model, d_model, 3, padding=1),
                        nn.BatchNorm1d(d_model),
                        nn.GELU(),
                    ),
                    "pos": PositionalEncoding__05(
                        d_model, max_len, float(cfg["dropout"])
                    ),
                    "trans": nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=int(cfg["nhead"]),
                            dim_feedforward=int(cfg["dim_feedforward"]),
                            dropout=float(cfg["dropout"]),
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        ),
                        num_layers=int(cfg["num_layers"]),
                    ),
                }
            )

        self.k6 = make_branch(cfg["vocab_k6"], K6_LENGTH__05)
        self.bpe = make_branch(cfg["vocab_bpe"], BPE_LENGTH__05)
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(cfg["cat_cardinalities"][f]),
                    META_EMBED_DIM__05,
                    padding_idx=0,
                )
                for f in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM__05
                + int(cfg["num_num_features"]),
                META_HIDDEN_DIM__05,
            ),
            nn.LayerNorm(META_HIDDEN_DIM__05),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM__05, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM__05, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(cfg["dropout"]))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1)
                )
                for _ in HEADS__05
            ]
        )

        for idx, head in enumerate(self.heads):
            if idx not in SHUFFLED_ACTIVE_HEAD_INDICES:
                for p in head.parameters():
                    p.requires_grad = False

    def _encode(self, branch, ids, mask):
        x = branch["emb"](ids)
        x = branch["cnn"](x.transpose(1, 2)).transpose(1, 2)
        x = branch["pos"](x)
        return branch["trans"](x, src_key_padding_mask=(mask == 0))[:, 0, :]

    def forward(self, k6_ids, k6_mask, bpe_ids, bpe_mask, cat_ids, num_values):
        parts = [
            self.cat_embeddings[i](cat_ids[:, i])
            for i in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(torch.cat(parts + [num_values], dim=1))
        f_k6, f_bpe = self._encode(self.k6, k6_ids, k6_mask), self._encode(
            self.bpe, bpe_ids, bpe_mask
        )
        g_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        g_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6 + g_bpe * f_bpe + g_meta * self.project_meta(f_meta)
        )
        return [h(self.fuse_dropout(fused)).squeeze(-1) for h in self.heads]






def random_1to1__05(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED__05)
    aps, aucs = [], []
    for _ in range(RANDOM_1TO1_REPEATS__05):
        idx = np.concatenate(
            [pos, rng.choice(neg, size=len(pos), replace=False)]
        )
        aps.append(safe_ap(labels[idx], scores[idx]))
        aucs.append(safe_auc(labels[idx], scores[idx]))
    return {
        "auroc_1to1_mean": float(np.mean(aucs)),
        "auroc_1to1_std": float(np.std(aucs)),
        "auprc_1to1_mean": float(np.mean(aps)),
        "auprc_1to1_std": float(np.std(aps)),
    }


def topk_metrics__05(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order, pos = np.argsort(-scores, kind="stable"), int((labels == 1).sum())
    prev = pos / len(labels)
    out = {}
    for frac in TOPK_FRACTIONS__05:
        k = int(math.floor(len(labels) * frac))
        hits = int(labels[order[:k]].sum())
        tag = TOPK_TAGS__05[frac]
        out[f"{tag}_hits"] = hits
        out[f"{tag}_fe"] = (
            float((hits / k) / prev) if prev > 0 else float("nan")
        )
    return out


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    row = dict(random_1to1__05(labels, scores))
    row["full_auroc"], row["full_auprc"] = safe_auc(labels, scores), safe_ap(
        labels, scores
    )
    row.update(topk_metrics__05(labels, scores))
    return row


def build_derangement(rows: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(rows).astype(np.int64)
    p = np.empty(rows, dtype=np.int64)
    p[order] = np.roll(order, 1)
    return p


def build_config__05(bundle: dict) -> dict:
    return {
        "sequence_mode": SHUFFLED_SEQUENCE_MODE,
        "vocab_k6": int(bundle["k6_vocab_size"]),
        "vocab_bpe": int(bundle["bpe_vocab_size"]),
        "d_model": D_MODEL__05,
        "nhead": NHEAD__05,
        "num_layers": NUM_LAYERS__05,
        "dim_feedforward": DIM_FEEDFORWARD__05,
        "dropout": DROPOUT__05,
        "cat_features": list(bundle["cat_features"]),
        "cat_cardinalities": {
            str(k): int(v) for k, v in bundle["cat_cardinalities"].items()
        },
        "num_num_features": len(bundle["num_features"]),
        "head_order": HEADS__05,
        "target_order": TARGETS__05,
        "active_heads": [HEADS__05[i] for i in SHUFFLED_ACTIVE_HEAD_INDICES],
        "score_heads": [HEADS__05[i] for i in SHUFFLED_SCORE_HEAD_INDICES],
        "sequence_score_contract": (
            "mean of sigmoid probabilities over score heads"
        ),
    }


def move_inputs__05(b: dict, dev: torch.device) -> tuple:
    return (
        b["k6_ids"].to(dev, non_blocking=True),
        b["k6_mask"].to(dev, non_blocking=True),
        b["bpe_ids"].to(dev, non_blocking=True),
        b["bpe_mask"].to(dev, non_blocking=True),
        b["cat_ids"].to(dev, non_blocking=True),
        b["num_values"].to(dev, non_blocking=True),
    )


def train_shuffled_control(
    train_bundle: dict, valid_bundle: dict, device: torch.device
):
    config = build_config__05(train_bundle)
    train_loader = DataLoader(
        DualDataset__05(
            train_bundle,
            build_derangement(
                len(train_bundle["labels"]), TRAIN_BPE_SHUFFLE_SEED
            ),
        ),
        batch_size=TRAIN_BATCH_SIZE__05,
        shuffle=True,
        num_workers=NUM_WORKERS__05,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        DualDataset__05(
            valid_bundle,
            build_derangement(
                len(valid_bundle["labels"]), VALID_BPE_SHUFFLE_SEED
            ),
        ),
        batch_size=VALID_BATCH_SIZE__05,
        shuffle=False,
        num_workers=NUM_WORKERS__05,
        pin_memory=False,
    )

    y = train_bundle["labels"].float()
    w = ((y.shape[0] - y.sum(dim=0)) / y.sum(dim=0)).to(device)

    model = ComplementarityModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR__05, weight_decay=WEIGHT_DECAY__05
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, MAX_EPOCHS__05 + 1):
        model.train()
        total, batches = 0.0, 0
        for b in train_loader:
            tg = b["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                out = model(*move_inputs__05(b, device))
                loss = sum(
                    F.binary_cross_entropy_with_logits(
                        out[i], tg[:, i], pos_weight=w[i]
                    )
                    for i in SHUFFLED_ACTIVE_HEAD_INDICES
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP__05)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1

    model.eval()
    parts = [[] for _ in HEADS__05]
    with torch.no_grad():
        for b in valid_loader:
            with torch.amp.autocast(device_type="cuda"):
                out = model(*move_inputs__05(b, device))
            for i in range(len(HEADS__05)):
                parts[i].append(torch.sigmoid(out[i]).detach().cpu().numpy())
    probs = np.stack(
        [np.concatenate(p, axis=0).astype(np.float64) for p in parts], axis=1
    )
    for idx in range(len(HEADS__05)):
        if idx not in SHUFFLED_ACTIVE_HEAD_INDICES:
            probs[:, idx] = np.nan
    torch.save(
        {"model": model.state_dict(), "config": config, "epoch": MAX_EPOCHS__05},
        CKPT_DIR__05 / f"05_{SHUFFLED_MODEL}.pt",
    )
    return probs


def select_route(metrics: pd.DataFrame) -> dict:
    tbl = (
        metrics[metrics["score"] == "score_sequence_mean"]
        .set_index("model")
        .loc[[m for m in ALL_MODELS if m in metrics["model"].values]]
    )
    el = [m for m in ELIGIBLE_ROUTE_MODELS if m in tbl.index]
    winners = set(el)
    trace = []
    for m in FORMAL_METRIC_ORDER:
        val = {mod: float(tbl.loc[mod, m]) for mod in el}
        best = max(val.values())
        keep = {mod for mod, v in val.items() if v == best}
        winners &= keep
        trace.append({"metric": m, "best_value": best, "tied_models": list(keep)})
    if not winners:
        winners = set(trace[-1]["tied_models"])
    selected = (
        winners.pop()
        if len(winners) == 1
        else sorted(winners, key=lambda x: MODEL_TIE_ORDER[x])[0]
    )
    return {
        "selected_model": selected,
        "reason": (
            f"highest {trace[0]['metric']}"
            if len(winners) == 1
            else "tied; simplest arm"
        ),
        "metric_order": FORMAL_METRIC_ORDER,
        "trace": trace,
        "sequence_score_by_model": {
            mod: {m: float(tbl.loc[mod, m]) for m in FORMAL_METRIC_ORDER}
            for mod in tbl.index
        },
    }


def step_05() -> None:
    seed_everything__05()
    OUT_DIR__05.mkdir(parents=True, exist_ok=True)
    CKPT_DIR__05.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_b, valid_b = torch.load(TRAIN_TENSOR__05, weights_only=False), torch.load(
        VALID_TENSOR__05, weights_only=False
    )
    info, labels = valid_b["info_df"].reset_index(drop=True), valid_b["info_df"][
        FANTOM_LABEL__05
    ].to_numpy(np.int8)

    if OUT_PREDICTIONS__05.is_file():
        preds = pd.read_parquet(OUT_PREDICTIONS__05)
    else:
        shuf = train_shuffled_control(train_b, valid_b, device)
        shuf_df = pd.DataFrame(
            {ID_COL__05: info[ID_COL__05].to_numpy(), FANTOM_LABEL__05: labels}
        )
        for idx, h in enumerate(HEADS__05):
            shuf_df[f"score_{h}"] = shuf[:, idx]
        shuf_df["score_sequence_mean"] = shuf[
            :, list(SHUFFLED_SCORE_HEAD_INDICES)
        ].mean(axis=1)
        shuf_df.insert(0, "model", SHUFFLED_MODEL)

        preds = pd.concat(
            [
                pd.read_parquet(M04_PREDICTIONS).rename(columns={"arm": "model"}),
                shuf_df,
            ],
            ignore_index=True,
        )
        preds.to_parquet(OUT_PREDICTIONS__05, index=False)

    rows = []
    for mod in ALL_MODELS:
        sub = preds[preds["model"] == mod]
        score_cols = [f"score_{h}" for h in HEADS__05] + ["score_sequence_mean"]
        if mod in {"dual_no_mpra_meta", SHUFFLED_MODEL}:
            score_cols = [
                "score_ccre_enhancer",
                "score_serve_gtex_support",
                "score_sequence_mean",
            ]
        for col in score_cols:
            sc = sub[col].to_numpy(np.float64)
            row = {"model": mod, "score": col}
            row.update(score_metrics(labels, sc))
            rows.append(row)
    metrics = pd.DataFrame(rows)

    mean_tbl = metrics[metrics["score"] == "score_sequence_mean"].set_index(
        "model"
    )

    mpra_rows = []
    for m in ["auroc_1to1_mean", "auprc_1to1_mean", "top1_hits"]:
        th = (
            float(mean_tbl.loc["full_dual_meta", m])
            if "full_dual_meta" in mean_tbl.index
            else float("nan")
        )
        tw = (
            float(mean_tbl.loc["dual_no_mpra_meta", m])
            if "dual_no_mpra_meta" in mean_tbl.index
            else float("nan")
        )
        mpra_rows.append(
            {
                "metric": m,
                "full_dual_three_head": th,
                "dual_no_mpra_two_head": tw,
                "delta_three_minus_two": th - tw,
            }
        )

    decision = select_route(metrics)
    OUT_ROUTE_DECISION.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    ckpt = torch.load(
        M04_CKPT_DIR / f"04_{decision['selected_model']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(
        {
            "model": ckpt["model"],
            "config": ckpt["config"],
            "selected_model": decision["selected_model"],
            "source": "04_train_sequence_branch_ablations",
        },
        FROZEN_SEQUENCE,
    )


def main() -> None:
    step_02()
    step_03()
    step_04()
    step_05()


if __name__ == "__main__":
    main()
