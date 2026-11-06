#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
    DATASET,
    UNIVERSALEPI_CHECKPOINT,
    UNIVERSALEPI_REPO,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import ast
import gc
import importlib
import importlib.util
import math
import types
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from pk_common import (
    EVAL_REPEATS,
    EVAL_SEED,
    SEEDS,
    Fasta,
    balanced_samples,
    build_tracks,
    metrics,
    seed_all,
    summarize,
    track_values,
    train_sample_rows,
)



ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REFERENCE = DATASET / "reference"

REPO = UNIVERSALEPI_REPO
CHECKPOINT = UNIVERSALEPI_CHECKPOINT
CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
FASTA = REFERENCE / "hg38.fa"
MANIFEST = (
    PROJECT / "Data_Preprocessing" / "output" / "04_roadmap_manifests"
    / "non_tumor_roadmap_manifest.csv"
)
PEAKS = PROJECT / "Data_Preprocessing" / "output" / "05_separated_roadmap_features" / "mapped"
OUT = ROOT / "output" / "08_universalepi"
CACHE = OUT / "cache"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
METHOD = "UniversalEPI (HERV task adaptation)"
BP, FLANK, TOKENS = 1000, 200, 401
DIMS = (7500, 3000, 600, 180, 90)
LUT = np.vstack([np.eye(4), np.zeros((1, 4))]).astype(np.float32)
EPOCHS, BATCH_SIZE, MICRO_BATCH_SIZE, FEATURE_BATCH_SIZE, LR = 30, 64, 8, 256, 1e-4


def load_inputs():
    cols = [ID, SPLIT, LABEL, "chrom", "seq_start", "seq_end"]
    df = pq.read_table(CANON, columns=cols).to_pandas()
    if not df[ID].is_unique or not df[cols].notna().all().all():
        raise RuntimeError("canonical table has missing or duplicated values")
    df = df.sort_values(ID, kind="stable").reset_index(drop=True)
    df["herv_center"] = (df.seq_start.astype("int64") + df.seq_end.astype("int64")) // 2
    train, y = df[SPLIT].eq("train").to_numpy(), df[LABEL].to_numpy(np.int64)
    samples = {seed: train_sample_rows(train, y, seed) for seed in SEEDS}
    valid_rows = np.flatnonzero(df[SPLIT] == "valid")
    keep = np.unique(np.concatenate(list(samples.values()) + [valid_rows]))
    loci = df.iloc[keep].copy().reset_index(drop=True)
    index = pd.Index(loci[ID])
    samples = {seed: index.get_indexer(df.iloc[rows][ID]) for seed, rows in samples.items()}
    training_windows = df.loc[train, ["chrom", "herv_center"]].copy()
    man = pd.read_csv(MANIFEST, dtype={"eid": str, "mark": str})
    man["eid"], man["mark"] = man.eid.str.strip(), man.mark.str.strip()
    man = man.loc[man.mark.isin(["DNase_fdr001", "DNase"])].copy()
    man["path"] = [PEAKS / f"{e}_{m}_non_tumor.hg38.bed.gz"
                   for e, m in zip(man.eid, man.mark)]
    return loci, samples, training_windows, man


def peak_neighbors(tracks: dict) -> dict:
    neighbors = {}
    for chrom, (positions, levels) in tracks.items():
        active = levels > 0
        starts = positions[active & ~np.r_[False, active[:-1]]]
        ends = positions[~active & np.r_[False, active[:-1]]]
        neighbors[chrom] = (starts + ends) // 2
    return neighbors


def train_signal_stats(tracks, training_windows, fasta_index):
    total, square, count = 0.0, 0.0, len(training_windows) * BP
    for chrom, group in training_windows.groupby("chrom"):
        if chrom not in tracks:
            continue
        p, v = tracks[chrom]
        start = np.maximum(group.herv_center.to_numpy(np.int64) - BP // 2, 0)
        end = np.minimum(group.herv_center.to_numpy(np.int64) + BP // 2, fasta_index[chrom][0])
        for power in (1, 2):
            level = v.astype(np.float64) ** power
            prefix = np.r_[0.0, np.cumsum(np.diff(p) * level[:-1])]

            def integral(x):
                i = np.searchsorted(p, x, side="right") - 1
                safe = np.maximum(i, 0)
                return np.where(i >= 0, prefix[safe] + (x - p[safe]) * level[safe], 0.0)

            value = float((integral(end) - integral(start)).sum())
            if power == 1:
                total += value
            else:
                square += value
    mean = total / count
    std = math.sqrt(max(0.0, square / count - mean ** 2))
    if std <= 1e-8:
        raise RuntimeError("the training windows carry no variable DNase signal")
    return {"mean": mean, "std": std}


def import_official():
    package = "_herv_universalepi_models"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str((REPO / "Stage2/models").resolve())]
        sys.modules[package] = module
    model = importlib.import_module(package + ".transformer_encoder_model")
    layer_path = REPO / "Stage2/deepc/components/layers.py"
    spec = importlib.util.spec_from_file_location("_herv_universalepi_layers", layer_path)
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    path = REPO / "Stage2/deepc/trepii_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "deepc.components")]
    namespace = {"__name__": "_herv_universalepi_deepc",
                 "ConvLayer": layers.ConvLayer, "DenseLayer": layers.DenseLayer}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["DeepC"], model.Transformer_Encoder


def load_backbone(device):
    DeepC, _ = import_official()
    cnn = DeepC([11, 11, 11, 5, 5], [5, 30, 60, 60, 90, 90, 90, 4], [4, 5, 5, 4, 2])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    cnn.load_state_dict(state, strict=True)
    cnn.requires_grad_(False)
    return cnn.eval().to(device)


def build_contexts(loci, peak_centers):
    context = np.zeros((len(loci), TOKENS), np.int32)
    blocks, offset = [], 1
    for chrom, group in loci.groupby("chrom", sort=True):
        centers = group.herv_center.to_numpy(np.int64)
        peaks = np.unique(peak_centers.get(chrom, np.empty(0, np.int64)))
        bank = np.union1d(peaks, centers)
        blocks.append(pd.DataFrame({"chrom": chrom, "center": bank}))
        rows = group.index.to_numpy()
        context[rows, FLANK] = np.searchsorted(bank, centers) + offset
        if len(peaks):
            left = np.searchsorted(peaks, centers, side="left")[:, None] + np.arange(-FLANK, 0)
            right = np.searchsorted(peaks, centers, side="right")[:, None] + np.arange(FLANK)
            candidate = np.concatenate([left, right], axis=1)
            good = (candidate >= 0) & (candidate < len(peaks))
            ids = np.searchsorted(bank, peaks[np.clip(candidate, 0, len(peaks) - 1)]) + offset
            ids[~good] = 0
            context[rows[:, None], np.r_[np.arange(FLANK), np.arange(FLANK + 1, TOKENS)]] = ids
        offset += len(bank)
    regions = pd.concat(blocks, ignore_index=True)
    used = np.unique(context)
    used = used[used > 0]
    mapping = np.zeros(len(regions) + 1, np.int32)
    mapping[used] = np.arange(1, len(used) + 1)
    context = mapping[context]
    regions = regions.iloc[used - 1].reset_index(drop=True)
    return regions, context


def prepare_cache(loci, training_windows, man, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    ref = Fasta(FASTA, base_order="ACTG")
    tracks = build_tracks(man.path.tolist())
    peaks = peak_neighbors(tracks)
    for chrom in peaks:
        if chrom in ref.index:
            peaks[chrom] = peaks[chrom][(peaks[chrom] >= 0)
                                       & (peaks[chrom] < ref.index[chrom][0])]
    stats = train_signal_stats(tracks, training_windows, ref.index)
    regions, context = build_contexts(loci, peaks)
    loci.to_parquet(CACHE / "loci.parquet", index=False)
    regions.to_parquet(CACHE / "regions.parquet", index=False)
    np.save(CACHE / "context.npy", context)
    np.save(CACHE / "positions.npy", np.r_[0, regions.center.to_numpy(np.int64)])
    features = np.lib.format.open_memmap(
        CACHE / "features.npy", mode="w+", dtype="float16", shape=(len(regions) + 1, sum(DIMS)))
    dnase = np.lib.format.open_memmap(
        CACHE / "dnase.npy", mode="w+", dtype="float16", shape=(len(regions) + 1, BP))
    features[0], dnase[0] = 0, 0
    cnn = load_backbone(device)
    with torch.no_grad():
        for begin in range(0, len(regions), FEATURE_BATCH_SIZE):
            part = regions.iloc[begin:begin + FEATURE_BATCH_SIZE]
            x = np.empty((len(part), 5, BP), np.float32)
            for j, row in enumerate(part.itertuples(index=False)):
                x[j, 1:] = LUT[ref.fetch_center(row.chrom, row.center, BP)].T
                start, end, size = (int(row.center) - BP // 2, int(row.center) + BP // 2,
                                    ref.index[row.chrom][0])
                lo, hi = max(0, start), min(end, size)
                x[j, 0] = 0
                if lo < hi:
                    raw = track_values(tracks, row.chrom, lo, hi)
                    x[j, 0, lo - start:hi - start] = (raw - stats["mean"]) / stats["std"]
            f = cnn(torch.from_numpy(x).to(device), feat=True)
            values = torch.cat(f, dim=1).cpu().numpy().astype(np.float16)
            features[begin + 1:begin + len(part) + 1] = values
            dnase[begin + 1:begin + len(part) + 1] = x[:, 0].astype(np.float16)
    features.flush()
    dnase.flush()
    meta = {"regions": len(regions), "backbone_parameters": sum(p.numel() for p in cnn.parameters())}
    del cnn, features, dnase
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ref.close()
    return meta


class GenomicPosition(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        pos = torch.arange(12001, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(20000.0) / dim))
        table = torch.zeros(12001, dim)
        table[:, 0::2], table[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("table", table)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pos, embed_dim, peu_flg):
        relative = pos[:, :, -1].long() - pos[:, FLANK:FLANK + 1, -1].long()
        index = torch.div(relative, 500, rounding_mode="floor").clamp(-6000, 6000) + 6000
        scale = math.sqrt(embed_dim) if peu_flg else 1.0
        return self.dropout(x + self.table[index] * scale)


class MaskedEncoder(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core, self.padding = core, None

    def forward(self, x, meta=None):
        return self.core(x, src_key_padding_mask=self.padding, meta=meta)


class HERVModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        _, Encoder = import_official()
        self.encoder = Encoder(input_dim=180, batch_size=1, input_feat_dims=list(DIMS),
                               stg_flg=True, map_flg=False, atac_flg=True, var_flg=False,
                               peu_flg=True, seq_len=TOKENS, binning=BP, num_heads=4,
                               num_layers=4, embed_dim=32, hidden_dim=64, dropout=0.1,
                               sigma=0.5, device=device)
        self.encoder.decoder = nn.Identity()
        self.encoder.pos_encoder = GenomicPosition(64)
        self.encoder.transformer_encoder = MaskedEncoder(self.encoder.transformer_encoder)
        self.head = nn.Linear(64, 1)

    def forward(self, features, dnase, positions, padding):
        self.encoder.transformer_encoder.padding = padding
        layers = list(torch.split(features, DIMS, dim=-1))
        encoded, _, regularizer = self.encoder.encode(
            layers, None, positions.unsqueeze(-1), dnase)
        return self.head(encoded[:, FLANK]).reshape(-1), regularizer


class CachedDataset(Dataset):
    def __init__(self, rows, labels):
        self.context = np.load(CACHE / "context.npy", mmap_mode="r")
        self.features = np.load(CACHE / "features.npy", mmap_mode="r")
        self.dnase = np.load(CACHE / "dnase.npy", mmap_mode="r")
        self.positions = np.load(CACHE / "positions.npy", mmap_mode="r")
        self.rows, self.labels = np.asarray(rows), np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = np.asarray(self.context[self.rows[i]])
        return self.features[ids], self.dnase[ids], self.positions[ids], ids == 0, self.labels[i]


def move_batch(batch, device):
    features, dnase, positions, padding, labels = batch
    return (features.to(device=device, dtype=torch.float32),
            dnase.to(device=device, dtype=torch.float32),
            positions.to(device), padding.to(device), labels.to(device))


def run_seed(seed, rows, valid_rows, loci, device):
    seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    net = HERVModel(device).to(device)
    y = loci[LABEL].to_numpy(np.int64)
    loader = DataLoader(CachedDataset(rows, y[rows]), batch_size=MICRO_BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(seed),
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-6)
    history = []
    for epoch in range(EPOCHS):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        total, bce_total, count, in_group = 0.0, 0.0, 0, 0
        for batch in loader:
            features, dnase, positions, padding, labels = move_batch(batch, device)
            logits, reg = net(features, dnase, positions, padding)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss = bce + 0.01 * reg
            group_size = min(BATCH_SIZE, len(rows) - (count - in_group))
            (loss * len(labels) / group_size).backward()
            in_group += len(labels)
            total += loss.item() * len(labels)
            bce_total += bce.item() * len(labels)
            count += len(labels)
            if in_group == group_size:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                in_group = 0
        history.append({"epoch": epoch + 1, "train_loss": total / count,
                        "train_bce": bce_total / count})
    pd.DataFrame(history).to_csv(OUT / f"history_seed{seed}.csv", index=False)
    torch.save({"state_dict": net.state_dict(), "seed": seed, "method": METHOD},
               OUT / "weights" / f"universalepi_seed{seed}.pt")
    net.eval()
    loader = DataLoader(CachedDataset(valid_rows, y[valid_rows]),
                        batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for batch in loader:
            f, d, p, m, _ = move_batch(batch, device)
            logits, _ = net(f, d, p, m)
            scores.append(logits.cpu().numpy())
    return np.concatenate(scores).astype(np.float64), sum(p.numel() for p in net.parameters())


def main():
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    loci, training, training_windows, man = load_inputs()
    meta = prepare_cache(loci, training_windows, man, device)

    valid_rows = np.flatnonzero(loci[SPLIT].eq("valid").to_numpy())
    valid = loci.iloc[valid_rows][[ID, SPLIT, LABEL]].reset_index(drop=True)
    y = valid[LABEL].to_numpy(np.int64)
    draws = balanced_samples(y)
    np.savez_compressed(OUT / "valid_eval_sample_indices.npz", indices=draws,
                        valid_ids=valid[ID].astype(str).to_numpy(dtype=str), labels=y,
                        eval_seed=EVAL_SEED, repeats=EVAL_REPEATS)

    rows, predictions, trainable = [], [], 0
    for seed in SEEDS:
        loci.iloc[training[seed]][[ID, SPLIT, LABEL]].to_csv(
            OUT / f"train_sample_ids_seed{seed}.tsv", sep="\t", index=False)
        scores, trainable = run_seed(seed, training[seed], valid_rows, loci, device)
        pred = valid.copy()
        pred["seed"], pred["score"] = seed, scores
        pred.to_parquet(OUT / f"valid_predictions_seed{seed}.parquet", index=False)
        predictions.append(pred)
        rows.append({"method": METHOD, "split": "valid", "seed": seed,
                     **metrics(y, scores, draws)})
        pd.DataFrame(rows).to_csv(OUT / "per_seed.csv", index=False)

    per_seed = pd.DataFrame(rows)
    summary = {"method": METHOD, "split": "valid", "seeds": "42;43;44", "seed_count": 3,
               "queue_loci": len(y), "queue_positives": int(y.sum()), "tokens": TOKENS,
               "train_samples_per_seed": len(training[42]), "eval_repeats": EVAL_REPEATS,
               "eval_seed": EVAL_SEED,
               "parameters": meta["backbone_parameters"] + trainable,
               "trainable_parameters": trainable}
    summary.update(summarize(per_seed))
    pd.DataFrame([summary]).to_csv(OUT / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUT / "per_locus_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
