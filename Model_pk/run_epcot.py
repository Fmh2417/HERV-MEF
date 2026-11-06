#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
    DATASET,
    EPCOT_CHECKPOINT,
    EPCOT_REPO,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import importlib
import importlib.util
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from types import SimpleNamespace
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

REPO = EPCOT_REPO
CHECKPOINT = EPCOT_CHECKPOINT
CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
MANIFEST = (
    PROJECT / "Data_Preprocessing" / "output" / "04_roadmap_manifests"
    / "non_tumor_roadmap_manifest.csv"
)
PEAKS = PROJECT / "Data_Preprocessing" / "output" / "05_separated_roadmap_features" / "mapped"
FASTA = DATASET / "reference" / "hg38.fa"
OUT = ROOT / "output" / "06_epcot"
CACHE = OUT / "cache"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
BIN_BP, FLANK, BINS = 1000, 300, 3
LENGTH, OFFSETS = 1600, (-1000, 0, 1000)
METHOD = "EPCOT-EAP (HERV task adaptation)"
LUT = np.vstack([np.eye(4), np.zeros((1, 4))]).astype(np.float32)
EPOCHS, BATCH_SIZE, LR, HEAD_DIM = 30, 64, 1e-4, 256


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
    man = pd.read_csv(MANIFEST, dtype={"eid": str, "mark": str})
    man["eid"], man["mark"] = man.eid.str.strip(), man.mark.str.strip()
    man = man.loc[man.mark.isin(["DNase_fdr001", "DNase"])].copy()
    man["path"] = [PEAKS / f"{e}_{m}_non_tumor.hg38.bed.gz"
                   for e, m in zip(man.eid, man.mark)]
    return loci, samples, man


def prepare_cache(loci, man):
    CACHE.mkdir(parents=True, exist_ok=True)
    ref = Fasta(FASTA)
    track = build_tracks(man.path.tolist())
    dna = np.lib.format.open_memmap(
        CACHE / "dna.npy", mode="w+", dtype="uint8", shape=(len(loci), BINS, LENGTH))
    dnase = np.lib.format.open_memmap(
        CACHE / "dnase.npy", mode="w+", dtype="float16", shape=(len(loci), BINS, LENGTH))
    for i, row in enumerate(loci.itertuples(index=False)):
        size = ref.index[row.chrom][0]
        dnase[i] = 0
        for b, offset in enumerate(OFFSETS):
            center = int(row.herv_center) + offset
            start, end = center - LENGTH // 2, center + LENGTH // 2
            dna[i, b] = ref.fetch_center(row.chrom, center, LENGTH)
            lo, hi = max(0, start), min(size, end)
            if lo < hi:
                dnase[i, b, lo - start:hi - start] = track_values(track, row.chrom, lo, hi)
    dna.flush()
    dnase.flush()
    loci.to_parquet(CACHE / "loci.parquet", index=False)
    ref.close()


def normalize_state(state):
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint does not contain a state dictionary")
    result = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod.", "pretrain_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        result[key] = value
    return result


def infer_config(state):
    key = "transformer.encoder.layers.0.linear1.weight"
    if "input_proj.weight" not in state or key not in state:
        raise RuntimeError("pretrained checkpoint lacks the input projection or encoder")
    hidden, ff = int(state["input_proj.weight"].shape[0]), int(state[key].shape[0])
    layers = sorted({int(m.group(1)) for k in state
                     for m in [re.match(r"transformer\.encoder\.layers\.(\d+)\.", k)] if m})
    return SimpleNamespace(hidden_dim=hidden, dim_feedforward=ff, enc_layers=len(layers),
                            nheads=4, dropout=0.2, dec_layers=2, bins=BINS, embed_dim=HEAD_DIM,
                            fine_tune=False, pretrain_path="none")


def complete_backbone_state(state, required):
    matched = {k: state[k] for k in required if k in state}
    if len(matched) != len(required):
        raise RuntimeError("pretrained backbone tensors are incomplete")
    for k in required:
        if tuple(matched[k].shape) != tuple(required[k].shape):
            raise RuntimeError(f"pretrained backbone tensor has a wrong shape: {k}")
    return matched


def make_model(device):
    repo = REPO.resolve()
    sys.path.insert(0, str(repo))
    module = importlib.import_module("EAP.model")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    state = normalize_state(state)
    config = infer_config(state)
    net = module.build_pretrain_model_enhancer(config)
    matched = complete_backbone_state(state, net.pretrain_model.state_dict())
    net.pretrain_model.load_state_dict(matched, strict=True)
    net.pretrain_model.requires_grad_(False)
    meta = {"parameters": sum(p.numel() for p in net.parameters()),
            "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)}
    return net.to(device), meta


class CachedDataset(Dataset):
    def __init__(self, rows, labels):
        self.dna = np.load(CACHE / "dna.npy", mmap_mode="r")
        self.dnase = np.load(CACHE / "dnase.npy", mmap_mode="r")
        self.rows, self.y = np.asarray(rows), np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = np.empty((BINS, 5, LENGTH), np.float32)
        x[:, :4] = LUT[self.dna[row]].transpose(0, 2, 1)
        x[:, 4] = self.dnase[row]
        return x, self.y[i]


def run_seed(seed, rows, valid_rows, loci, device):
    seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    net, meta = make_model(device)
    y = loci[LABEL].to_numpy(np.int64)
    loader = DataLoader(CachedDataset(rows, y[rows]), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(seed),
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=1e-6)
    criterion, history = nn.BCEWithLogitsLoss(), []
    for epoch in range(EPOCHS):
        net.train()
        net.pretrain_model.eval()
        total, count = 0.0, 0
        for x, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(x.to(device)).reshape(-1), label.to(device))
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * len(label), count + len(label)
        history.append({"epoch": epoch + 1, "train_loss": total / count})
    pd.DataFrame(history).to_csv(OUT / f"history_seed{seed}.csv", index=False)
    torch.save({"state_dict": net.state_dict(), "seed": seed, "method": METHOD},
               OUT / f"epcot_seed{seed}.pt")
    net.eval()
    loader = DataLoader(CachedDataset(valid_rows, y[valid_rows]), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            scores.append(net(x.to(device)).reshape(-1).sigmoid().cpu().numpy())
    return np.concatenate(scores).astype(np.float64), meta


def main():
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    loci, training, man = load_inputs()
    prepare_cache(loci, man)

    valid_rows = np.flatnonzero(loci[SPLIT].eq("valid").to_numpy())
    valid = loci.iloc[valid_rows][[ID, SPLIT, LABEL]].reset_index(drop=True)
    y = valid[LABEL].to_numpy(np.int64)
    draws = balanced_samples(y)
    np.savez_compressed(OUT / "valid_eval_sample_indices.npz", indices=draws,
                        valid_ids=valid[ID].astype(str).to_numpy(dtype=str), labels=y,
                        eval_seed=EVAL_SEED, repeats=EVAL_REPEATS)

    rows, predictions, meta = [], [], {}
    for seed in SEEDS:
        loci.iloc[training[seed]][[ID, SPLIT, LABEL]].to_csv(
            OUT / f"train_sample_ids_seed{seed}.tsv", sep="\t", index=False)
        scores, meta = run_seed(seed, training[seed], valid_rows, loci, device)
        pred = valid.copy()
        pred["seed"], pred["score"] = seed, scores
        pred.to_parquet(OUT / f"valid_predictions_seed{seed}.parquet", index=False)
        predictions.append(pred)
        rows.append({"method": METHOD, "split": "valid", "seed": seed,
                     **metrics(y, scores, draws)})
        pd.DataFrame(rows).to_csv(OUT / "per_seed.csv", index=False)

    per_seed = pd.DataFrame(rows)
    summary = {"method": METHOD, "split": "valid", "seeds": "42;43;44", "seed_count": 3,
               "queue_loci": len(y), "queue_positives": int(y.sum()),
               "input_channels": 5, "bins": BINS,
               "train_samples_per_seed": len(training[42]), "eval_repeats": EVAL_REPEATS,
               "eval_seed": EVAL_SEED, "parameters": meta["parameters"],
               "trainable_parameters": meta["trainable_parameters"]}
    summary.update(summarize(per_seed))
    pd.DataFrame([summary]).to_csv(OUT / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUT / "per_locus_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
