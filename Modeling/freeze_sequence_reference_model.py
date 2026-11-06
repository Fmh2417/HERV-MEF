#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from modeling_common import (
    HEADS as HEADS__07,
    SequenceReferenceModel,
    safe_ap,
    safe_auc,
)

# --------------------------------------------------------------------------
# step 06: Train the partially unfrozen sequence model
# --------------------------------------------------------------------------



ARMS = {
    "u1_fusion_heads": {
        "gpu": "1",
        "unfreeze_last_transformer_blocks": False,
        "description": "fusion gates + cCRE and SERVE heads; encoders frozen",
    },
    "u2_last_blocks": {
        "gpu": "2",
        "unfreeze_last_transformer_blocks": True,
        "description": "U1 plus the final K6 and BPE Transformer blocks",
    },
}

ROOT = Path(__file__).resolve().parent
M01_DIR = ROOT / "output/01_build_dual_tensors"
TRAIN_TENSOR = M01_DIR / "01_train_dual_tensors.pt"
VALID_TENSOR = M01_DIR / "01_valid_dual_tensors.pt"
M05_DIR = ROOT / "output/05_k6_bpe_complementarity_controls"
FROZEN_SEQUENCE = M05_DIR / "05_frozen/05_frozen_sequence_reference.pt"

OUT_DIR = ROOT / "output/06_train_partial_unfreeze"
CKPT_DIR = OUT_DIR / "06_checkpoints"


TARGETS = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]
HEADS = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
ID_COL = "herv_internal_id"

K6_LENGTH, BPE_LENGTH = 996, 256
TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS = 32, 128, 0, 15
COMMON_LR, LAST_BLOCK_LR, WEIGHT_DECAY, GRAD_CLIP, SEED = (
    2e-5,
    5e-6,
    0.01,
    1.0,
    99999999,
)
META_EMBED_DIM, META_HIDDEN_DIM = 16, 64


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualDataset(Dataset):

    def __init__(self, bundle: dict):
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle["labels"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "k6_ids": self.bundle["input_ids_k6"][idx].long(),
            "k6_mask": self.bundle["mask_k6"][idx].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][idx].long(),
            "bpe_mask": self.bundle["mask_bpe"][idx].long(),
            "cat_ids": self.bundle["cat_ids"][idx].long(),
            "num_values": self.bundle["num_values"][idx].float(),
            "targets": self.bundle["labels"][idx].float(),
        }


class PositionalEncoding(nn.Module):

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


class PartialUnfreezeModel(nn.Module):

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
                    "pos": PositionalEncoding(
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

        self.k6 = make_branch(cfg["vocab_k6"], K6_LENGTH)
        self.bpe = make_branch(cfg["vocab_bpe"], BPE_LENGTH)
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(cfg["cat_cardinalities"][f]),
                    META_EMBED_DIM,
                    padding_idx=0,
                )
                for f in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM
                + int(cfg["num_num_features"]),
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
        self.fuse_dropout = nn.Dropout(float(cfg["dropout"]))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1)
                )
                for _ in HEADS
            ]
        )

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






def remap_flat_branch_state(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        parts = key.split(".", 1)
        if len(parts) == 2 and parts[0].startswith(("emb_", "cnn_", "pos_", "trans_")):
            prefix, suffix = parts
            module_type, branch = prefix.split("_", 1)
            if branch == "k6":
                out[f"k6.{module_type}.{suffix}"] = value
            elif branch == "bpe":
                out[f"bpe.{module_type}.{suffix}"] = value
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def configure_unfreeze(
    model: PartialUnfreezeModel, worker: str, active_head_indices
):
    for p in model.parameters():
        p.requires_grad = False
    common = {
        "bpe_gate": model.bpe_gate,
        "meta_gate": model.meta_gate,
        "metadata_projection": model.project_meta,
        "fusion_norm": model.fuse_norm,
    }
    for idx in active_head_indices:
        common[f"head_{HEADS[idx]}"] = model.heads[idx]
    for m in common.values():
        for p in m.parameters():
            p.requires_grad = True

    last_blocks = {}
    if ARMS[worker]["unfreeze_last_transformer_blocks"]:
        last_blocks = {
            "k6_last": model.k6["trans"].layers[-1],
            "bpe_last": model.bpe["trans"].layers[-1],
        }
        for m in last_blocks.values():
            for p in m.parameters():
                p.requires_grad = True
    return common, last_blocks


def keep_frozen_modules_in_eval(
    model: PartialUnfreezeModel, worker: str
) -> None:
    for m in (
        model.k6["emb"],
        model.k6["cnn"],
        model.k6["pos"],
        model.bpe["emb"],
        model.bpe["cnn"],
        model.bpe["pos"],
        model.cat_embeddings,
        model.metadata_mlp,
    ):
        m.eval()
    model.k6["trans"].eval()
    model.bpe["trans"].eval()
    if ARMS[worker]["unfreeze_last_transformer_blocks"]:
        model.k6["trans"].layers[-1].train(True)
        model.bpe["trans"].layers[-1].train(True)


def build_optimizer(common, last_blocks):
    groups = [
        {
            "params": [p for m in common.values() for p in m.parameters()],
            "lr": COMMON_LR,
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    last_params = [p for m in last_blocks.values() for p in m.parameters()]
    if last_params:
        groups.append(
            {
                "params": last_params,
                "lr": LAST_BLOCK_LR,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    return torch.optim.AdamW(groups)


def move_inputs(b: dict, dev: torch.device) -> tuple:
    return (
        b["k6_ids"].to(dev, non_blocking=True),
        b["k6_mask"].to(dev, non_blocking=True),
        b["bpe_ids"].to(dev, non_blocking=True),
        b["bpe_mask"].to(dev, non_blocking=True),
        b["cat_ids"].to(dev, non_blocking=True),
        b["num_values"].to(dev, non_blocking=True),
    )


def infer(model, loader, device, active_head_indices):
    model.eval()
    parts = [[] for _ in HEADS]
    target_parts = []
    with torch.no_grad():
        for b in loader:
            with torch.amp.autocast(device_type="cuda"):
                out = model(*move_inputs(b, device))
            for i in range(len(HEADS)):
                parts[i].append(torch.sigmoid(out[i]).detach().cpu().numpy())
            target_parts.append(b["targets"].numpy())
    probs = np.stack(
        [np.concatenate(p, axis=0).astype(np.float64) for p in parts], axis=1
    )
    for idx in range(len(HEADS)):
        if idx not in active_head_indices:
            probs[:, idx] = (
                np.nan
            )
    return probs, np.concatenate(target_parts, axis=0).astype(np.int8)


def run_arm(worker: str) -> None:
    seed_everything()
    device = torch.device("cuda")

    ref = torch.load(FROZEN_SEQUENCE, map_location="cpu", weights_only=False)
    config = ref["config"]
    active_idx = tuple(config["active_head_indices"])
    score_idx = tuple(config["score_head_indices"])

    train_b, valid_b = torch.load(TRAIN_TENSOR, weights_only=False), torch.load(
        VALID_TENSOR, weights_only=False
    )
    train_loader = DataLoader(
        DualDataset(train_b),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        DualDataset(valid_b),
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    y = train_b["labels"].float()
    w = ((y.shape[0] - y.sum(dim=0)) / y.sum(dim=0)).to(device)

    model = PartialUnfreezeModel(config).to(device)
    model.load_state_dict(remap_flat_branch_state(ref["model"]))
    common, last_blocks = configure_unfreeze(model, worker, active_idx)
    optimizer = build_optimizer(common, last_blocks)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = CKPT_DIR / worker
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        keep_frozen_modules_in_eval(model, worker)
        total, batches = 0.0, 0
        for step, b in enumerate(train_loader, start=1):
            tg = b["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                out = model(*move_inputs(b, device))
                loss = sum(
                    F.binary_cross_entropy_with_logits(
                        out[i], tg[:, i], pos_weight=w[i]
                    )
                    for i in active_idx
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1

        probs, targets = infer(model, valid_loader, device, active_idx)
        row = {
            "worker": worker,
            "epoch": epoch,
            "train_loss": total / max(batches, 1),
        }
        for i in active_idx:
            row[f"valid_{HEADS[i]}_ap"] = safe_ap(targets[:, i], probs[:, i])
            row[f"valid_{HEADS[i]}_auc"] = safe_auc(targets[:, i], probs[:, i])
        row["valid_mean_ap"] = float(
            np.mean([row[f"valid_{HEADS[i]}_ap"] for i in active_idx])
        )
        history.append(row)

        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "worker": worker,
                "epoch": epoch,
                "active_heads": [HEADS[i] for i in active_idx],
                "score_heads": [HEADS[i] for i in score_idx],
                "active_head_indices": list(active_idx),
                "score_head_indices": list(score_idx),
                "source_model": ref["selected_model"],
                "valid_mean_ap": row["valid_mean_ap"],
            },
            ckpt_dir / f"06_{worker}_epoch_{epoch:02d}.pt",
        )
    pd.DataFrame(history).to_csv(
        ckpt_dir / f"06_{worker}_history.tsv", sep="\t", index=False
    )


def step_06():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for worker in ARMS:
        run_arm(worker)


# --------------------------------------------------------------------------
# step 07: Audit and freeze the final sequence reference model
# --------------------------------------------------------------------------



ROOT__07 = Path(__file__).resolve().parent
M01_DIR__07 = ROOT__07 / "output" / "01_build_dual_tensors"
VALID_TENSOR__07 = M01_DIR__07 / "01_valid_dual_tensors.pt"
M05_DIR__07 = ROOT__07 / "output" / "05_k6_bpe_complementarity_controls"
M05_FROZEN = M05_DIR__07 / "05_frozen" / "05_frozen_sequence_reference.pt"
M06_DIR = ROOT__07 / "output" / "06_train_partial_unfreeze"
CKPT_DIR__07 = M06_DIR / "06_checkpoints"

OUT_DIR__07 = ROOT__07 / "output" / "07_audit_partial_unfreeze"
FROZEN_DIR = OUT_DIR__07 / "07_frozen"
FROZEN_SEQUENCE__07 = FROZEN_DIR / "07_frozen_sequence_reference.pt"


TARGETS__07 = ["is_ccre_enhancer_like", "is_serve_gtex_supported", "is_mpra_active_supported"]
FANTOM_LABEL = "is_fantom5_enhancer"
ID_COL__07 = "herv_internal_id"


INFERENCE_BATCH_SIZE = 128
NUM_WORKERS__07 = 0
SEED__07 = 99999999


RANDOM_1TO1_REPEATS = 100
RANDOM_1TO1_SEED = 99999999
TOPK_FRACTIONS = (0.005, 0.010, 0.050)
TOPK_TAGS = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}

SCORE_COLUMNS = ["score_ccre_enhancer", "score_serve_gtex_support", "score_sequence_mean"]
FORMAL_METRIC_ORDER = [
    "top1_hits",
    "top5_hits",
    "auprc_1to1_mean",
    "top0_5_hits",
    "auroc_1to1_mean",
]
BASELINE_CANDIDATE = "m05_frozen_baseline"


_LOG_HANDLES = {}


def seed_everything__07(seed: int = SEED__07) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ValidationDataset(Dataset):
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






def normalize_state_for_flat_model(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key.startswith("k6.") or key.startswith("bpe."):
            branch, submodule, rest = key.split(".", 2)
            prefix_map = {
                "emb": "emb",
                "cnn": "cnn",
                "pos": "pos",
                "trans": "trans",
            }
            new_key = f"{prefix_map[submodule]}_{branch}.{rest}"
            out[new_key] = value
        else:
            out[key] = value
    return out


def random_1to1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED)
    aps, aucs = [], []
    for _ in range(RANDOM_1TO1_REPEATS):
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


def topk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = int((labels == 1).sum())
    prevalence = positives / len(labels)
    out = {}
    for fraction in TOPK_FRACTIONS:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        tag = TOPK_TAGS[fraction]
        out[f"{tag}_hits"] = hits
        out[f"{tag}_fe"] = float((hits / k) / prevalence) if prevalence > 0 else float("nan")
    return out


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    row = dict(random_1to1(labels, scores))
    row["full_auroc"] = safe_auc(labels, scores)
    row["full_auprc"] = safe_ap(labels, scores)
    row.update(topk_metrics(labels, scores))
    return row


def probe(model, loader, device, score_head_indices) -> dict[str, np.ndarray]:
    model.eval()
    parts = [[] for _ in HEADS__07]
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
            for i in range(len(HEADS__07)):
                parts[i].append(torch.sigmoid(logits[i]).detach().cpu().numpy())
    probabilities = np.stack(
        [np.concatenate(part, axis=0).astype(np.float64) for part in parts], axis=1
    )
    return {
        "score_ccre_enhancer": probabilities[:, 0],
        "score_serve_gtex_support": probabilities[:, 1],
        "score_sequence_mean": probabilities[:, list(score_head_indices)].mean(axis=1),
    }


def load_candidates() -> list[dict]:
    baseline = torch.load(M05_FROZEN, map_location="cpu", weights_only=False)
    candidates = [
        {
            "name": BASELINE_CANDIDATE,
            "path": M05_FROZEN,
            "state_dict": normalize_state_for_flat_model(baseline["model"]),
            "config": baseline["config"],
            "epoch": 0,
            "active_heads": baseline["config"]["active_heads"],
            "score_heads": baseline["config"]["score_heads"],
        }
    ]
    for path in sorted(CKPT_DIR__07.glob("*/*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append(
            {
                "name": f"{path.parent.name}/epoch_{int(checkpoint['epoch']):02d}",
                "path": path,
                "state_dict": normalize_state_for_flat_model(checkpoint["model"]),
                "config": checkpoint["config"],
                "epoch": int(checkpoint["epoch"]),
                "active_heads": checkpoint["active_heads"],
                "score_heads": checkpoint["score_heads"],
            }
        )
    return candidates


def select_route(metrics: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    table = metrics[metrics["score"] == "score_sequence_mean"].set_index("candidate")
    trace_rows = []
    winners = set(table.index)
    for metric in FORMAL_METRIC_ORDER:
        values = {name: float(table.loc[name, metric]) for name in table.index}
        best = max(values.values())
        keep = {name for name, value in values.items() if value == best}
        winners &= keep
        trace_rows.append(
            {"metric": metric, "best_value": best, "tied_candidates": sorted(keep)}
        )
    if not winners:
        # Different metrics favoured different candidates; fall back to the last round.
        winners = set(trace_rows[-1]["tied_candidates"])
    selected = sorted(winners)[0]
    decision = {
        "selected_candidate": selected,
        "metric_order": FORMAL_METRIC_ORDER,
        "reason": (
            f"highest {trace_rows[0]['metric']}"
            if len(winners) == 1
            else "all formal metrics tied; first candidate in sorted order"
        ),
        "baseline_candidate": BASELINE_CANDIDATE,
        "baseline_beaten": selected != BASELINE_CANDIDATE,
    }
    return decision, pd.DataFrame(trace_rows)


def step_07() -> None:
    seed_everything__07()
    OUT_DIR__07.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    bundle = torch.load(VALID_TENSOR__07, weights_only=False)
    info = bundle["info_df"].reset_index(drop=True)
    labels = info[FANTOM_LABEL].to_numpy(np.int8)
    loader = DataLoader(
        ValidationDataset(bundle),
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS__07,
        pin_memory=False,
    )

    candidates = load_candidates()
    rows = []
    scores_by_candidate = {}
    for candidate in candidates:
        model = SequenceReferenceModel(candidate["config"]).to(device)
        model.load_state_dict(candidate["state_dict"])
        scores = probe(
            model, loader, device, tuple(candidate["config"]["score_head_indices"])
        )
        scores_by_candidate[candidate["name"]] = scores
        for column in SCORE_COLUMNS:
            row = {"candidate": candidate["name"], "score": column}
            row.update(score_metrics(labels, scores[column]))
            rows.append(row)
        del model
        torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)

    decision, trace = select_route(metrics)


    selected = next(c for c in candidates if c["name"] == decision["selected_candidate"])
    torch.save(
        {
            "model": selected["state_dict"],
            "config": selected["config"],
            "selected_candidate": selected["name"],
            "active_heads": selected["active_heads"],
            "score_heads": selected["score_heads"],
            "source": str(selected["path"]),
        },
        FROZEN_SEQUENCE__07,
    )


def main() -> None:
    step_06()
    step_07()


if __name__ == "__main__":
    main()
