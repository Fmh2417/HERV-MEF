#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
    DATASET,
    HEAP_MODEL,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import ast
import copy
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
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

HEAP_MAIN = HEAP_MODEL
CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
MANIFEST = (
    PROJECT / "Data_Preprocessing" / "output" / "04_roadmap_manifests"
    / "non_tumor_roadmap_manifest.csv"
)
PEAKS = PROJECT / "Data_Preprocessing" / "output" / "05_separated_roadmap_features" / "mapped"
FASTA = DATASET / "hg38.fa"
OUT = ROOT / "output" / "05_heap"
CACHE = OUT / "cache"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
MARKS = ("DNase", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3", "H3K9me3")
LENGTH, BATCH, EPOCHS = 1001, 64, 15
LUT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32)


def load_inputs():
    cols = [ID, SPLIT, LABEL, "chrom", "seq_start", "seq_end"]
    df = pq.read_table(CANON, columns=cols).to_pandas()
    if not df[ID].is_unique or not df[cols].notna().all().all():
        raise RuntimeError("canonical table has missing or duplicated values")
    df = df.sort_values(ID, kind="stable").reset_index(drop=True)
    df["window_start"] = (df.seq_start.astype("int64") + df.seq_end.astype("int64")) // 2 - 500
    train = df[SPLIT].eq("train").to_numpy()
    y = df[LABEL].to_numpy(np.int64)
    samples = {seed: train_sample_rows(train, y, seed) for seed in SEEDS}
    valid_rows = np.flatnonzero(df[SPLIT] == "valid")
    keep = np.unique(np.concatenate(list(samples.values()) + [valid_rows]))
    loci = df.iloc[keep].copy().reset_index(drop=True)
    index = pd.Index(loci[ID])
    samples = {s: index.get_indexer(df.iloc[rows][ID]) for s, rows in samples.items()}
    man = pd.read_csv(MANIFEST, dtype={"eid": str, "mark": str})
    man["eid"], man["mark"] = man.eid.str.strip(), man.mark.str.strip()
    man["track"] = man.mark.replace({"DNase_fdr001": "DNase"})
    man = man[man.track.isin(MARKS)].copy()
    man["path"] = [PEAKS / f"{e}_{m}_non_tumor.hg38.bed.gz"
                   for e, m in zip(man.eid, man.mark)]
    return loci, samples, man


def prepare_cache(loci, man):
    CACHE.mkdir(parents=True, exist_ok=True)
    ref = Fasta(FASTA)
    tracks = {mark: build_tracks(man.loc[man.track == mark, "path"].tolist())
              for mark in MARKS}
    dna = np.lib.format.open_memmap(
        CACHE / "dna.npy", mode="w+", dtype="uint8", shape=(len(loci), LENGTH))
    ep = np.lib.format.open_memmap(
        CACHE / "chromatin.npy", mode="w+", dtype="float16", shape=(len(loci), 6, LENGTH))
    for chrom, group in loci.groupby("chrom", sort=True):
        size = ref.index[chrom][0]
        for row in group.sort_values("window_start").itertuples():
            start = int(row.window_start)
            end = start + LENGTH
            dna[row.Index] = ref.fetch(chrom, start, end)
            ep[row.Index] = 0
            lo, hi = max(start, 0), min(end, size)
            if lo < hi:
                for j, mark in enumerate(MARKS):
                    ep[row.Index, j, lo - start:hi - start] = track_values(
                        tracks[mark], chrom, lo, hi)
    dna.flush()
    ep.flush()
    loci.to_parquet(CACHE / "loci.parquet", index=False)
    ref.close()


def model_from_official(device):
    names = {"upsample", "PositionalEncoding", "Encoder", "attention", "make_model",
             "SequenceEncoding", "ChromatinEncoding", "HEAP_W", "Mlp_Freeze", "freeze_conv"}
    tree = ast.parse(HEAP_MAIN.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    ns = {"__name__": "_heap_official", "torch": torch, "nn": nn, "F": F, "np": np,
          "math": math, "copy": copy, "Variable": torch.autograd.Variable, "device": device}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HEAP_MAIN), "exec"), ns)
    net = ns["HEAP_W"]()
    old = net.chromatin_encoding.conv1
    conv = ns["freeze_conv"](6, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, dilation=old.dilation,
                              groups=old.groups, bias=old.bias is not None)
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    net.chromatin_encoding.conv1, net.softmax = conv, nn.Identity()
    return net.to(device)


class CachedDataset(Dataset):
    def __init__(self, rows, y):
        self.dna = np.load(CACHE / "dna.npy", mmap_mode="r")
        self.ep = np.load(CACHE / "chromatin.npy", mmap_mode="r")
        self.rows, self.y = np.asarray(rows), np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return LUT[self.dna[row]], self.ep[row].astype(np.float32), self.y[i]


def train_and_predict(seed, rows, valid_rows, loci, device):
    seed_all(seed)
    net = model_from_official(device)
    y = loci[LABEL].to_numpy(np.int64)
    loader = DataLoader(CachedDataset(rows, y[rows]), batch_size=BATCH,
                        shuffle=True, generator=torch.Generator().manual_seed(seed),
                        num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion, history = nn.CrossEntropyLoss(), []
    for epoch in range(EPOCHS):
        net.train()
        total, count = 0.0, 0
        for seq, ep, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(seq.to(device), ep.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * len(label), count + len(label)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total / count})
    pd.DataFrame(history).to_csv(OUT / f"history_seed{seed}.csv", index=False)
    torch.save({"state_dict": net.state_dict(), "seed": seed, "marks": MARKS,
                "window": LENGTH}, OUT / "weights" / f"heap_seed{seed}.pt")
    loader = DataLoader(CachedDataset(valid_rows, y[valid_rows]), batch_size=BATCH,
                        shuffle=False, num_workers=0)
    net.eval()
    scores = []
    with torch.no_grad():
        for seq, ep, _ in loader:
            scores.append(net(seq.to(device), ep.to(device)).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(scores).astype(np.float64), net


def main():
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    loci, training, man = load_inputs()
    prepare_cache(loci, man)

    valid_rows = np.flatnonzero(loci[SPLIT].eq("valid").to_numpy())
    valid = loci.iloc[valid_rows][[ID, SPLIT, LABEL]].reset_index(drop=True)
    y = valid[LABEL].to_numpy(np.int64)
    samples = balanced_samples(y)
    np.savez_compressed(OUT / "valid_eval_sample_indices.npz", indices=samples,
                        valid_ids=valid[ID].astype(str).to_numpy(dtype=str), labels=y,
                        eval_seed=EVAL_SEED, repeats=EVAL_REPEATS)

    rows, predictions, parameters = [], [], []
    for seed in SEEDS:
        sample = loci.iloc[training[seed]][[ID, SPLIT, LABEL]].copy()
        sample["seed"] = seed
        sample.to_csv(OUT / f"train_sample_ids_seed{seed}.tsv", sep="\t", index=False)
        scores, net = train_and_predict(seed, training[seed], valid_rows, loci, device)
        pred = valid.copy()
        pred["seed"], pred["score"] = seed, scores
        pred.to_parquet(OUT / f"valid_predictions_seed{seed}.parquet", index=False)
        predictions.append(pred)
        parameters.append(int(sum(p.numel() for p in net.parameters())))
        rows.append({"method": "HEAP-W (HERV task adaptation)", "split": "valid",
                     "seed": seed, **metrics(y, scores, samples)})
        pd.DataFrame(rows).to_csv(OUT / "per_seed.csv", index=False)

    per_seed = pd.DataFrame(rows)
    summary = {"method": "HEAP-W (HERV task adaptation)", "split": "valid",
               "seeds": "42;43;44", "seed_count": 3, "queue_loci": len(y),
               "queue_positives": int(y.sum()),
               "train_samples_per_seed": len(training[42]),
               "eval_repeats": EVAL_REPEATS, "eval_seed": EVAL_SEED,
               "parameters": parameters[0],
               "trainable_parameters": parameters[0]}
    summary.update(summarize(per_seed))
    pd.DataFrame([summary]).to_csv(OUT / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUT / "per_locus_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
