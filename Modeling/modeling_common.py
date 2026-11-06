#!/usr/bin/env python3

import math
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


HEADS = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
K6_LENGTH = 996
BPE_LENGTH = 256
META_EMBED_DIM = 16
META_HIDDEN_DIM = 64


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def stable_order(scores: np.ndarray) -> np.ndarray:
    row_order = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((row_order, -scores))


def write_tsv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


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


class SequenceReferenceModel(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.cat_features = list(config["cat_features"])

        self.emb_k6 = nn.Embedding(int(config["vocab_k6"]), d_model, padding_idx=0)
        self.cnn_k6 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU()
        )
        self.pos_k6 = PositionalEncoding(d_model, K6_LENGTH, float(config["dropout"]))
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

        self.emb_bpe = nn.Embedding(int(config["vocab_bpe"]), d_model, padding_idx=0)
        self.cnn_bpe = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU()
        )
        self.pos_bpe = PositionalEncoding(d_model, BPE_LENGTH, float(config["dropout"]))
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
                nn.Embedding(int(config["cat_cardinalities"][f]), META_EMBED_DIM, padding_idx=0)
                for f in self.cat_features
            ]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM + int(config["num_num_features"]),
                META_HIDDEN_DIM,
            ),
            nn.LayerNorm(META_HIDDEN_DIM),
            nn.GELU(),
        )

        self.bpe_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM, d_model), nn.Sigmoid()
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(config["dropout"]))

        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
                for _ in HEADS
            ]
        )

    def _encode(self, prefix: str, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = getattr(self, f"emb_{prefix}")(ids)
        values = getattr(self, f"cnn_{prefix}")(values.transpose(1, 2)).transpose(1, 2)
        values = getattr(self, f"pos_{prefix}")(values)
        return getattr(self, f"trans_{prefix}")(values, src_key_padding_mask=(mask == 0))[
            :, 0, :
        ]

    def forward(self, k6_ids, k6_mask, bpe_ids, bpe_mask, cat_ids, num_values):
        parts = [
            self.cat_embeddings[i](cat_ids[:, i]) for i in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(torch.cat(parts + [num_values], dim=1))
        f_k6 = self._encode("k6", k6_ids, k6_mask)
        f_bpe = self._encode("bpe", bpe_ids, bpe_mask)
        gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
        gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
        fused = self.fuse_norm(
            f_k6 + gate_bpe * f_bpe + gate_meta * self.project_meta(f_meta)
        )
        fused = self.fuse_dropout(fused)
        return [head(fused).squeeze(-1) for head in self.heads]
