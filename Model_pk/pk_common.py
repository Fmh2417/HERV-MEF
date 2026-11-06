#!/usr/bin/env python3
import gzip
import math
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

PROJECT = Path(__file__).resolve().parent.parent

SEEDS = (42, 43, 44)
EVAL_SEED = 99999999
EVAL_REPEATS = 100
TOPKS = ((0.005, "top0_5"), (0.01, "top1"), (0.05, "top5"))
METRIC_COLS = ["auroc_1to1", "auprc_1to1", "top0_5_hits", "top0_5_fe",
               "top1_hits", "top1_fe", "top5_hits", "top5_fe"]

FROZEN_SEQUENCE = (
    PROJECT / "Modeling" / "output" / "07_audit_partial_unfreeze" / "07_frozen"
    / "07_frozen_sequence_reference.pt"
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_samples(y: np.ndarray) -> np.ndarray:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(EVAL_SEED)
    return np.stack([
        np.r_[pos, rng.choice(neg, len(pos), replace=False)]
        for _ in range(EVAL_REPEATS)
    ])


def metrics(y: np.ndarray, scores: np.ndarray, samples: np.ndarray) -> dict:
    result = {
        "auroc_1to1": float(np.mean([roc_auc_score(y[i], scores[i]) for i in samples])),
        "auprc_1to1": float(np.mean([
            average_precision_score(y[i], scores[i]) for i in samples
        ])),
    }
    order = np.argsort(-scores, kind="stable")
    prevalence = float(y.mean())
    for fraction, tag in TOPKS:
        k = int(math.floor(len(y) * fraction))
        hits = int(y[order[:k]].sum())
        result[f"{tag}_hits"] = hits
        result[f"{tag}_fe"] = float((hits / k) / prevalence)
    return result


def train_sample_rows(train_mask: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    pos = np.flatnonzero(train_mask & (y == 1))
    neg = np.flatnonzero(train_mask & (y == 0))
    rng = np.random.default_rng(seed)
    rows = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
    return rows[rng.permutation(len(rows))]


def summarize(per_seed: pd.DataFrame) -> dict:
    result = {}
    for col in METRIC_COLS:
        result[f"{col}_mean"] = float(per_seed[col].mean())
        result[f"{col}_std"] = float(per_seed[col].std(ddof=1))
    return result


def sequence_model_parameters() -> int:
    checkpoint = torch.load(FROZEN_SEQUENCE, map_location="cpu", weights_only=False)
    total = 0
    for name, tensor in checkpoint["model"].items():
        if not hasattr(tensor, "shape"):
            continue
        if name.endswith(("running_mean", "running_var", "num_batches_tracked", ".pe")):
            continue
        total += tensor.numel()
    return int(total)


def bed_intervals(path: Path) -> dict:
    intervals = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            col = line.split()
            intervals.setdefault(col[0], []).append((int(col[1]), int(col[2])))
    merged = {}
    for chrom, pairs in intervals.items():
        blocks = []
        for start, end in sorted(pairs):
            if blocks and start <= blocks[-1][1]:
                blocks[-1][1] = max(blocks[-1][1], end)
            else:
                blocks.append([start, end])
        merged[chrom] = np.asarray(blocks, dtype=np.int64)
    return merged


def build_tracks(paths, dtype=np.float32) -> dict:
    events = {}
    for path in paths:
        for chrom, intervals in bed_intervals(path).items():
            events.setdefault(chrom, []).append(intervals)
    tracks = {}
    for chrom, blocks in events.items():
        a = np.concatenate(blocks)
        positions, inverse = np.unique(np.r_[a[:, 0], a[:, 1]], return_inverse=True)
        levels = np.cumsum(np.bincount(
            inverse, weights=np.r_[np.ones(len(a)), -np.ones(len(a))]))
        tracks[chrom] = (positions, (levels / len(paths)).astype(dtype))
    return tracks


def track_values(track: dict, chrom: str, start: int, end: int) -> np.ndarray:
    if chrom not in track:
        return np.zeros(end - start, np.float32)
    positions, levels = track[chrom]
    lo = np.searchsorted(positions, start, side="right")
    hi = np.searchsorted(positions, end, side="left")
    base = levels[lo - 1] if lo else 0.0
    return np.repeat(
        np.r_[base, levels[lo:hi]], np.diff(np.r_[start, positions[lo:hi], end])
    ).astype(np.float32)


def track_mean(track: dict, chrom: str, start: int, end: int) -> float:
    if chrom not in track or end <= start:
        return 0.0
    positions, levels = track[chrom]
    area = np.r_[0.0, np.cumsum(np.diff(positions) * levels[:-1])]

    def integral(x):
        i = np.searchsorted(positions, x, side="right") - 1
        return 0.0 if i < 0 else area[i] + levels[i] * (x - positions[i])

    return float((integral(end) - integral(start)) / (end - start))


class Fasta:
    def __init__(self, path: Path, base_order: str = "ACGT"):
        rows = pd.read_csv(str(path) + ".fai", sep="\t", header=None, usecols=range(5))
        self.index = {r[0]: tuple(int(v) for v in r[1:])
                      for r in rows.itertuples(index=False, name=None)}
        self.file = open(path, "rb")
        self.codes = np.full(256, 4, np.uint8)
        for i, base in enumerate(base_order):
            self.codes[ord(base)] = i

    def fetch(self, chrom: str, start: int, end: int) -> np.ndarray:
        size, offset, bases, width = self.index[chrom]
        start, end = int(start), int(end)
        lo, hi = max(0, start), min(end, size)
        result = np.full(end - start, 4, np.uint8)
        if lo < hi:
            b0 = offset + lo // bases * width + lo % bases
            b1 = offset + (hi - 1) // bases * width + (hi - 1) % bases + 1
            self.file.seek(b0)
            seq = self.file.read(b1 - b0).replace(b"\n", b"").replace(b"\r", b"").upper()
            result[lo - start:hi - start] = self.codes[np.frombuffer(seq, np.uint8)]
        return result

    def fetch_center(self, chrom: str, center: int, length: int) -> np.ndarray:
        return self.fetch(chrom, int(center) - length // 2, int(center) + length // 2)

    def close(self) -> None:
        self.file.close()
