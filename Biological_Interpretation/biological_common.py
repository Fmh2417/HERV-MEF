#!/usr/bin/env python3

import json
import math
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any
from numpy.lib.stride_tricks import sliding_window_view
from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL_OUTPUT = PROJECT_ROOT / "Modeling/output"
M01_DIR = MODEL_OUTPUT / "01_build_dual_tensors"
M07_DIR = MODEL_OUTPUT / "07_audit_partial_unfreeze"

MODEL_CHECKPOINT = M07_DIR / "07_frozen/07_frozen_sequence_reference.pt"

TOKENIZER_PATH = M01_DIR / "01_herv_bpe_tokenizer.json"

ENCODER_PATH = M01_DIR / "01_metadata_encoder.joblib"

SCALER_PATH = M01_DIR / "01_metadata_scaler.joblib"


K6_K = 6

K6_LENGTH = 996

BPE_LENGTH = 256

K6_PAD_ID = 0

K6_CLS_ID = 1

K6_UNK_ID = 2

K6_OFFSET = 3

BYTE_TO_VAL = np.full(256, -1, dtype=np.int16)

BYTE_TO_VAL[ord("A")] = 0

BYTE_TO_VAL[ord("C")] = 1

BYTE_TO_VAL[ord("G")] = 2

BYTE_TO_VAL[ord("T")] = 3

K6_POWERS = (4 ** np.arange(K6_K - 1, -1, -1)).astype(np.int32)

META_EMBED_DIM = 16

META_HIDDEN_DIM = 64



def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def write_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)


def write_json(obj: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_text(content: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1), :])


class FrozenSequenceModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.cat_features = list(config["cat_features"])
        self.emb_k6 = nn.Embedding(int(config["vocab_k6"]), d_model, padding_idx=0)
        self.cnn_k6 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU())
        self.pos_k6 = PositionalEncoding(d_model, K6_LENGTH, float(config["dropout"]))
        self.trans_k6 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]), dropout=float(config["dropout"]),
                activation="gelu", batch_first=True, norm_first=True),
            num_layers=int(config["num_layers"]),
        )
        self.emb_bpe = nn.Embedding(int(config["vocab_bpe"]), d_model, padding_idx=0)
        self.cnn_bpe = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU())
        self.pos_bpe = PositionalEncoding(d_model, BPE_LENGTH, float(config["dropout"]))
        self.trans_bpe = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]), dropout=float(config["dropout"]),
                activation="gelu", batch_first=True, norm_first=True),
            num_layers=int(config["num_layers"]),
        )
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(int(config["cat_cardinalities"][feature]), META_EMBED_DIM, padding_idx=0)
            for feature in self.cat_features
        ])
        self.metadata_mlp = nn.Sequential(
            nn.Linear(len(self.cat_features) * META_EMBED_DIM + int(config["num_num_features"]), META_HIDDEN_DIM),
            nn.LayerNorm(META_HIDDEN_DIM), nn.GELU(),
        )
        self.bpe_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.meta_gate = nn.Sequential(nn.Linear(d_model + META_HIDDEN_DIM, d_model), nn.Sigmoid())
        self.project_meta = nn.Linear(META_HIDDEN_DIM, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))
        self.head_order = list(config["head_order"])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in self.head_order
        ])

    def forward(self, k6_ids: torch.Tensor, k6_mask: torch.Tensor, bpe_ids: torch.Tensor,
                bpe_mask: torch.Tensor, cat_ids: torch.Tensor, num_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f_k6 = self.emb_k6(k6_ids)
        f_k6 = self.cnn_k6(f_k6.transpose(1, 2)).transpose(1, 2)
        f_k6 = self.pos_k6(f_k6)
        f_k6 = self.trans_k6(f_k6, src_key_padding_mask=(k6_mask == 0))[:, 0, :]
        f_bpe = self.emb_bpe(bpe_ids)
        f_bpe = self.cnn_bpe(f_bpe.transpose(1, 2)).transpose(1, 2)
        f_bpe = self.pos_bpe(f_bpe)
        f_bpe = self.trans_bpe(f_bpe, src_key_padding_mask=(bpe_mask == 0))[:, 0, :]
        categorical = [self.cat_embeddings[i](cat_ids[:, i]) for i in range(len(self.cat_features))]
        f_meta = self.metadata_mlp(torch.cat(categorical + [num_values], dim=1))
        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta))
        fused = self.fuse_dropout(fused)
        return tuple(head(fused).squeeze(-1) for head in self.heads)


def encode_k6(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    sequence = sequence.upper()
    ids = np.zeros(K6_LENGTH, dtype=np.int64)
    mask = np.ones(K6_LENGTH, dtype=np.int64)
    ids[0] = K6_CLS_ID
    values = BYTE_TO_VAL[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    windows = sliding_window_view(values, K6_K)
    valid = (windows >= 0).all(axis=1)
    codes = windows.astype(np.int32) @ K6_POWERS
    ids[1:] = np.where(valid, K6_OFFSET + codes, K6_UNK_ID)
    return ids, mask


def encode_bpe(sequence: str, tokenizer: Tokenizer) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    encoded = tokenizer.encode(sequence)
    ids = np.zeros(BPE_LENGTH, dtype=np.int64)
    mask = np.zeros(BPE_LENGTH, dtype=np.int64)
    keep_ids = encoded.ids[:BPE_LENGTH]
    offsets = encoded.offsets[:BPE_LENGTH]
    ids[: len(keep_ids)] = keep_ids
    mask[: len(keep_ids)] = 1
    offsets = list(offsets) + [(0, 0)] * (BPE_LENGTH - len(offsets))
    return ids, mask, offsets


def transform_metadata(frame: pd.DataFrame, config: dict[str, Any], encoder: Any, scaler: Any) -> tuple[np.ndarray, np.ndarray]:
    cat_features = list(config["cat_features"])
    num_features = list(scaler.feature_names_in_)
    raw_cat = encoder.transform(frame[cat_features].astype(str))
    cat_ids = (raw_cat + 1).astype(np.int64)
    cat_ids[cat_ids < 0] = 0
    numeric = frame[num_features].copy()
    numeric["dist_to_tss"] = np.sign(numeric["dist_to_tss"]) * np.log10(np.abs(numeric["dist_to_tss"]) + 1.0)
    num_values = scaler.transform(numeric).astype(np.float32)
    return cat_ids, num_values


def tensors_for_rows(frame: pd.DataFrame, tokenizer: Tokenizer, config: dict[str, Any], encoder: Any, scaler: Any,
                     sequences: list[str] | None = None) -> tuple[dict[str, torch.Tensor], list[list[tuple[int, int]]]]:
    seqs = frame["sequence"].astype(str).tolist() if sequences is None else sequences
    k6_ids, k6_masks, bpe_ids, bpe_masks, offsets = [], [], [], [], []
    for sequence in seqs:
        ki, km = encode_k6(sequence)
        bi, bm, off = encode_bpe(sequence, tokenizer)
        k6_ids.append(ki); k6_masks.append(km); bpe_ids.append(bi); bpe_masks.append(bm); offsets.append(off)
    cat, num = transform_metadata(frame, config, encoder, scaler)
    result = {
        "k6_ids": torch.as_tensor(np.stack(k6_ids), dtype=torch.long),
        "k6_mask": torch.as_tensor(np.stack(k6_masks), dtype=torch.long),
        "bpe_ids": torch.as_tensor(np.stack(bpe_ids), dtype=torch.long),
        "bpe_mask": torch.as_tensor(np.stack(bpe_masks), dtype=torch.long),
        "cat_ids": torch.as_tensor(cat, dtype=torch.long),
        "num_values": torch.as_tensor(num, dtype=torch.float32),
    }
    return result, offsets


def sequence_score(logits: tuple[torch.Tensor, ...], score_head_indices: list[int]) -> torch.Tensor:
    probabilities = [torch.sigmoid(logits[i]) for i in score_head_indices]
    return torch.stack(probabilities).mean(dim=0)


def score_rows(model: FrozenSequenceModel, tensors: dict[str, torch.Tensor], device: torch.device,
               score_head_indices: list[int], batch_size: int = 4) -> np.ndarray:
    scores = []
    n = tensors["k6_ids"].shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        args = [tensors[k][start:end].to(device) for k in
                ["k6_ids", "k6_mask", "bpe_ids", "bpe_mask", "cat_ids", "num_values"]]
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(*args)
        scores.append(sequence_score(logits, score_head_indices).detach().float().cpu().numpy())
    return np.concatenate(scores)


def map_token_attribution(k6_attr: np.ndarray, bpe_attr: np.ndarray,
                          offsets: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k6_base = np.zeros(1000, dtype=np.float64)
    for token_position in range(1, K6_LENGTH):
        start = token_position - 1
        k6_base[start : start + K6_K] += float(k6_attr[token_position]) / K6_K
    bpe_base = np.zeros(1000, dtype=np.float64)
    for token_position, (start, end) in enumerate(offsets):
        if token_position == 0 or end <= start or start >= 1000:
            continue
        start = max(0, int(start)); end = min(1000, int(end))
        bpe_base[start:end] += float(bpe_attr[token_position]) / max(1, end - start)
    combined = k6_base + bpe_base
    return k6_base, bpe_base, combined / combined.sum()


def attribute_one(model: FrozenSequenceModel, row: pd.DataFrame, tokenizer: Tokenizer,
                  config: dict[str, Any], encoder: Any, scaler: Any, device: torch.device,
                  score_head_indices: list[int]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    tensors, offsets = tensors_for_rows(row, tokenizer, config, encoder, scaler)
    args = [tensors[k].to(device) for k in ["k6_ids", "k6_mask", "bpe_ids", "bpe_mask", "cat_ids", "num_values"]]
    activations: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            output.retain_grad()
            activations[name] = output
        return hook

    handles = [model.emb_k6.register_forward_hook(capture("k6")), model.emb_bpe.register_forward_hook(capture("bpe"))]
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(*args)
    score = sequence_score(logits, score_head_indices)
    score.sum().backward()
    k6_attr = torch.abs((activations["k6"] * activations["k6"].grad).sum(dim=-1))[0].detach().cpu().numpy()
    bpe_attr = torch.abs((activations["bpe"] * activations["bpe"].grad).sum(dim=-1))[0].detach().cpu().numpy()
    for handle in handles:
        handle.remove()
    k6_base, bpe_base, combined = map_token_attribution(k6_attr, bpe_attr, offsets[0])
    return float(score.detach().cpu().item()), k6_base, bpe_base, combined


def genomic_interval(row: pd.Series, sequence_start: int, sequence_end: int) -> tuple[int, int]:
    if str(row["strand"]) == "-":
        return int(row["seq_end"] - sequence_end), int(row["seq_end"] - sequence_start)
    return int(row["seq_start"] + sequence_start), int(row["seq_start"] + sequence_end)


def build_frozen_scorer() -> dict[str, Any]:
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenSequenceModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return {
        "checkpoint": checkpoint,
        "config": config,
        "device": device,
        "model": model,
        "score_head_indices": [int(i) for i in config["score_head_indices"]],
        "tokenizer": Tokenizer.from_file(str(TOKENIZER_PATH)),
        "encoder": joblib.load(ENCODER_PATH),
        "scaler": joblib.load(SCALER_PATH),
    }
