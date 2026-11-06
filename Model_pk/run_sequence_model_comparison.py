#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling"))

import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from pk_common import (
    SEEDS,
    balanced_samples,
    metrics,
    seed_all,
    METRIC_COLS,
    sequence_model_parameters,
)
from modeling_common import SequenceReferenceModel
from config import (
    CADUCEUS_CACHE,
    CADUCEUS_WEIGHTS,
    DNABERT2_CACHE,
    DNABERT2_WEIGHTS,
    HYENADNA_CACHE,
    HYENADNA_WEIGHTS,
)

# --------------------------------------------------------------------------
# step 00: Build the sequence PK master table
# --------------------------------------------------------------------------



ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

CANONICAL = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
OUT_DIR = ROOT / "output" / "00_sequence_pk_master"
MASTER = OUT_DIR / "sequence_pk_master.parquet"

COLUMNS = ["herv_internal_id", "sequence", "split", "is_fantom5_enhancer"]


def step_00() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(CANONICAL, columns=COLUMNS)
    source = source.sort_values("herv_internal_id", kind="mergesort").reset_index(drop=True)
    source["split"] = source["split"].astype(str)
    source["is_fantom5_enhancer"] = pd.to_numeric(
        source["is_fantom5_enhancer"], errors="raise"
    ).astype("int8")
    source.to_parquet(MASTER, index=False)


# --------------------------------------------------------------------------
# step 01: Count the backbone model parameters
# --------------------------------------------------------------------------



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT__01 = Path(__file__).resolve().parent

OUT = ROOT__01 / "output" / "01_backbone_parameter_counts" / "backbone_param_counts.json"

os.environ.setdefault("HF_HOME", "/tmp/hf_model_pk")


BACKBONES = {
    "Caduceus-Ph": CADUCEUS_WEIGHTS,
    "DNABERT-2": DNABERT2_WEIGHTS,
    "HyenaDNA": HYENADNA_WEIGHTS,
}


def step_01() -> None:
    from transformers import AutoModel

    counts = {}
    for name, weights in BACKBONES.items():
        net = AutoModel.from_pretrained(
            weights,
            trust_remote_code=True,
            local_files_only=True,
        )
        counts[name] = {
            "total_parameters": int(sum(p.numel() for p in net.parameters())),
            "total_buffers": int(sum(b.numel() for b in net.buffers())),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"backbones": counts}, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# step 02: Run the unified forward-only probe comparison
# --------------------------------------------------------------------------



ROOT__02 = Path(__file__).resolve().parent
PROJECT__02 = ROOT__02.parent

MASTER__02 = ROOT__02 / "output" / "00_sequence_pk_master" / "sequence_pk_master.parquet"
FROZEN = (
    PROJECT__02 / "Modeling" / "output" / "07_audit_partial_unfreeze" / "07_frozen"
    / "07_frozen_sequence_reference.pt"
)
TENSORS = PROJECT__02 / "Modeling" / "output" / "01_build_dual_tensors"
TENSOR_FILES = {
    "train": "01_train_dual_tensors.pt",
    "valid": "01_valid_dual_tensors.pt",
    "test": "01_test_dual_inputs.pt",
}
OUT__02 = ROOT__02 / "output" / "02_forward_only_probe"

SPLITS = ("train", "valid", "test")
ID = "herv_internal_id"
LABEL = "is_fantom5_enhancer"
HIDDEN_DIM = 128
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EXTRACT_BATCH = 256
PREDICT_BATCH = 4096


class Probe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, emb: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(emb), PREDICT_BATCH):
        stop = start + PREDICT_BATCH
        p = torch.sigmoid(model(emb[start:stop].to(device)))
        scores.append(p.cpu().numpy())
    return np.concatenate(scores)


def train_probe(emb, labels, device: torch.device, seed: int):
    seed_all(seed)
    y = labels["train"]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    model = Probe(emb["train"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(len(neg) / len(pos), dtype=torch.float32, device=device)
    )
    neg_rng = np.random.default_rng(seed)
    shuffle_rng = torch.Generator().manual_seed(seed + 100000)
    best_auc, best_epoch, best_state = -1.0, -1, None

    for epoch in range(EPOCHS):
        idx = np.concatenate([
            pos, neg_rng.choice(neg, size=len(pos), replace=False)
        ])
        x = emb["train"][torch.from_numpy(idx)]
        batch_labels = torch.from_numpy(y[idx].astype(np.float32))
        loader = DataLoader(
            TensorDataset(x, batch_labels),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=shuffle_rng,
            num_workers=0,
        )
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        valid_scores = predict(model, emb["valid"], device)
        auc = float(roc_auc_score(labels["valid"], valid_scores))
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_auc


def evaluate(name, slug, group, emb, labels, test_ids, device, samples) -> list:
    rows = []
    for seed in SEEDS:
        model, epoch, valid_auc = train_probe(emb, labels, device, seed)
        scores = predict(model, emb["test"], device)
        row = {
            "model": name,
            "group": group,
            "seed": seed,
            "embed_dim": int(emb["train"].shape[1]),
            "best_epoch": epoch,
            "best_valid_auroc": valid_auc,
            "queue_loci": int(len(labels["test"])),
            "queue_positives": int(labels["test"].sum()),
            **metrics(labels["test"], scores, samples),
        }
        rows.append(row)
        pd.DataFrame({
            ID: test_ids,
            LABEL: labels["test"],
            "score": scores,
        }).to_parquet(OUT__02 / "predictions" / f"{slug}_seed{seed}.parquet", index=False)
        del model
    return rows


def split_embedding(values: torch.Tensor, split_indices: dict) -> dict:
    values = values.detach().cpu().float()
    return {
        split: values[torch.from_numpy(split_indices[split])]
        for split in SPLITS
    }


def align_embedding(values: torch.Tensor, source_ids, target_ids) -> torch.Tensor:
    source = pd.Index(np.asarray(source_ids).astype(str))
    target = pd.Index(np.asarray(target_ids).astype(str))
    if not source.is_unique or not target.is_unique or len(source) != len(target):
        raise RuntimeError("representation and queue identifiers are not aligned")
    order = source.get_indexer(target)
    if (order < 0).any():
        raise RuntimeError("representation identifiers do not cover the queue")
    return values.detach().cpu()[torch.from_numpy(order)].float()


@torch.no_grad()
def extract_split(net, split: str, device: torch.device) -> dict:
    bundle = torch.load(TENSORS / TENSOR_FILES[split], map_location="cpu", weights_only=False)
    n = len(bundle["input_ids_k6"])
    dim = net.fuse_norm.normalized_shape[0]
    sequence = torch.empty(n, dim)
    full = torch.empty(n, dim)

    for start in range(0, n, EXTRACT_BATCH):
        stop = start + EXTRACT_BATCH
        k6 = bundle["input_ids_k6"][start:stop].long().to(device)
        km = bundle["mask_k6"][start:stop].long().to(device)
        bpe = bundle["input_ids_bpe"][start:stop].long().to(device)
        bm = bundle["mask_bpe"][start:stop].long().to(device)
        cat = bundle["cat_ids"][start:stop].long().to(device)
        num = bundle["num_values"][start:stop].float().to(device)
        parts = [net.cat_embeddings[i](cat[:, i]) for i in range(len(net.cat_features))]
        meta = net.metadata_mlp(torch.cat(parts + [num], dim=1))
        fk = net._encode("k6", k6, km)
        fb = net._encode("bpe", bpe, bm)
        gb = net.bpe_gate(torch.cat([fk, fb], dim=1))
        gm = net.meta_gate(torch.cat([fk, meta], dim=1))
        raw = fk + gb * fb
        sequence[start:stop] = raw.float().cpu()
        full[start:stop] = net.fuse_norm(raw + gm * net.project_meta(meta)).float().cpu()

    return {
        "sequence_repr": sequence,
        "full_repr": full,
        "ids": bundle["info_df"][ID].astype(str).to_numpy(),
    }


def load_herv(device: torch.device):
    checkpoint = torch.load(FROZEN, map_location="cpu", weights_only=False)
    net = SequenceReferenceModel(checkpoint["config"]).to(device)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()
    blob = {"splits": {split: extract_split(net, split, device) for split in SPLITS}}
    del net

    norm = nn.LayerNorm(checkpoint["config"]["d_model"])
    norm.load_state_dict({
        "weight": checkpoint["model"]["fuse_norm.weight"],
        "bias": checkpoint["model"]["fuse_norm.bias"],
    })
    norm.eval()
    return blob, norm


def step_02() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (OUT__02 / "predictions").mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(MASTER__02)
    master[ID] = master[ID].astype(str)
    master = master.sort_values(ID, kind="mergesort").reset_index(drop=True)
    if not master[ID].is_unique:
        raise RuntimeError("locus identifiers are not unique in the master table")

    split_indices, labels, ids = {}, {}, {}
    for split in SPLITS:
        idx = np.flatnonzero(master["split"].to_numpy() == split)
        split_indices[split] = idx
        labels[split] = master.loc[idx, LABEL].to_numpy(np.int8)
        ids[split] = master.loc[idx, ID].to_numpy()
    samples = balanced_samples(labels["test"])
    rows = []

    settings = [
        ("Caduceus-Ph", "caduceus", CADUCEUS_CACHE, "forward"),
        ("DNABERT-2", "dnabert2", DNABERT2_CACHE, None),
        ("HyenaDNA", "hyena", HYENADNA_CACHE, "embeddings"),
    ]
    for name, slug, path, key in settings:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        values = cache if key is None else cache[key]
        emb = split_embedding(values, split_indices)
        del values, cache
        rows.extend(evaluate(
            name, slug, "sequence", emb, labels, ids["test"], device, samples
        ))
        del emb

    blob, norm = load_herv(device)
    for key, name, slug, group in [
        ("sequence_repr", "HERV-MEF K6+BPE", "herv_sequence", "sequence"),
        ("full_repr", "HERV-MEF + metadata", "herv_metadata", "metadata"),
    ]:
        emb = {}
        for split in SPLITS:
            part = blob["splits"][split]
            values = align_embedding(part[key], part["ids"], ids[split])
            if key == "sequence_repr":
                with torch.no_grad():
                    values = torch.cat([
                        norm(chunk) for chunk in values.split(PREDICT_BATCH)
                    ])
            emb[split] = values
        rows.extend(evaluate(
            name, slug, group, emb, labels, ids["test"], device, samples
        ))
        del emb

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT__02 / "forward_only_per_seed.csv", index=False)
    metric_columns = ["auroc_1to1", "auprc_1to1", "top0_5_hits", "top0_5_fe",
                      "top1_hits", "top1_fe", "top5_hits", "top5_fe"]
    summary = frame.groupby(["model", "group"], sort=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{'seed_std' if stat == 'std' else stat}"
        for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(OUT__02 / "forward_only_comparison.csv", index=False)


# --------------------------------------------------------------------------
# step 03: Collect the sequence PK comparison
# --------------------------------------------------------------------------



ROOT__03 = Path(__file__).resolve().parent

UNIFIED = ROOT__03 / "output" / "02_forward_only_probe"
BACKBONE_COUNTS = (
    ROOT__03 / "output" / "01_backbone_parameter_counts" / "backbone_param_counts.json"
)
OUT__03 = ROOT__03 / "output" / "03_seq_pk_comparison"

SEEDS__03 = (42, 43, 44)
HIDDEN_DIM__03 = 128
EXPECTED_TEST = (72041, 1033)

VARIANTS = [
    {"model": "DNABERT-2", "group": "sequence only",
     "model_family": "DNA foundation model", "input": "forward strand only"},
    {"model": "HyenaDNA", "group": "sequence only",
     "model_family": "DNA foundation model", "input": "forward strand only"},
    {"model": "Caduceus-Ph", "group": "sequence only",
     "model_family": "DNA foundation model", "input": "forward strand only"},
    {"model": "HERV-MEF K6+BPE", "group": "sequence only",
     "model_family": "project sequence model", "input": "forward strand only"},
    {"model": "HERV-MEF + metadata", "group": "sequence + metadata",
     "model_family": "project sequence model", "input": "forward strand + metadata"},
]


def step_03() -> None:
    OUT__03.mkdir(parents=True, exist_ok=True)

    per_seed = pd.read_csv(UNIFIED / "forward_only_per_seed.csv")
    if len(per_seed) != len(VARIANTS) * len(SEEDS__03):
        raise RuntimeError("the unified probe run does not cover every model and seed")
    if sorted(per_seed["seed"].unique().tolist()) != sorted(SEEDS__03):
        raise RuntimeError("the unified probe run does not use the fixed seeds")
    queue_loci = int(per_seed["queue_loci"].iloc[0])
    queue_positives = int(per_seed["queue_positives"].iloc[0])
    if (queue_loci, queue_positives) != EXPECTED_TEST:
        raise RuntimeError("the test queue does not match the expected cohort")

    backbones = json.loads(BACKBONE_COUNTS.read_text(encoding="utf-8"))["backbones"]
    backbone = {name: int(block["total_parameters"]) for name, block in backbones.items()}
    backbone["HERV-MEF K6+BPE"] = sequence_model_parameters()
    backbone["HERV-MEF + metadata"] = backbone["HERV-MEF K6+BPE"]

    summary_rows = []
    for variant in VARIANTS:
        block = per_seed[per_seed["model"] == variant["model"]]
        embed_dim = int(block["embed_dim"].iloc[0])
        head = embed_dim * HIDDEN_DIM__03 + 4 * HIDDEN_DIM__03 + 1
        entry = {
            "model": variant["model"],
            "group": variant["group"],
            "model_family": variant["model_family"],
            "input": variant["input"],
            "seeds": ";".join(str(s) for s in SEEDS__03),
            "seed_count": len(block),
            "backbone_parameters": backbone[variant["model"]],
            "probe_head_parameters": head,
            "total_parameters": backbone[variant["model"]] + head,
            "trainable_parameters": head,
            "embedding_dim": embed_dim,
        }
        for col in METRIC_COLS:
            entry[f"{col}_mean"] = float(block[col].mean())
            entry[f"{col}_std"] = float(block[col].std(ddof=1))
        entry["split"] = "test"
        entry["queue_loci"] = queue_loci
        entry["queue_positives"] = queue_positives
        summary_rows.append(entry)

    front = ["model", "group", "model_family", "input", "seeds", "seed_count",
             "backbone_parameters", "probe_head_parameters", "total_parameters",
             "trainable_parameters", "embedding_dim",
             *[f"{col}_{stat}" for col in METRIC_COLS for stat in ("mean", "std")],
             "split", "queue_loci", "queue_positives"]
    summary = pd.DataFrame(summary_rows)[front]
    summary = summary.sort_values("auprc_1to1_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(OUT__03 / "seq_pk_final_comparison_test.csv", index=False)


def main() -> None:
    step_00()
    step_01()
    step_02()
    step_03()


if __name__ == "__main__":
    main()
