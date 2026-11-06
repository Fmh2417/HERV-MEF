#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
    DATASET,
    EPINFORMER_REPO,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import gzip
import importlib.util
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
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
    track_mean,
    train_sample_rows,
)



ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REFERENCE = DATASET / "reference"

REPO = EPINFORMER_REPO
CHECKPOINT = (REPO / "trained_models" / "pretrained_enhancer_encoder"
              / "fold_8_best_K562_seq2activityLog2_leaveChrOut_combinedRS_2bins_bs64"
              "_H3K27ac_adamW_erisxdl_r0_checkpoint.pt")
CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
GTF = REFERENCE / "gencode.v38.annotation.gtf.gz"
FASTA = REFERENCE / "hg38.fa"
MANIFEST = (
    PROJECT / "Data_Preprocessing" / "output" / "04_roadmap_manifests"
    / "non_tumor_roadmap_manifest.csv"
)
PEAKS = PROJECT / "Data_Preprocessing" / "output" / "05_separated_roadmap_features" / "mapped"
OUT = ROOT / "output" / "07_epinformer"
CACHE = OUT / "cache"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
LENGTH = 2000
MARKS = ("DNase", "H3K27ac")
METHOD = "EPInformer-v2 (HERV/nearest-promoter task adaptation)"
LUT = np.vstack([np.eye(4), np.zeros((1, 4))]).astype(np.float32)
EPOCHS, BATCH_SIZE, LR = 30, 64, 1e-4


def gene_key(value):
    return re.sub(r"\.\d+(?=_|$)", "", str(value).strip())


def read_genes():
    records = {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            col = line.rstrip("\r\n").split("\t")
            if len(col) != 9 or col[2] != "gene":
                continue
            match = re.search(r'(?:^|;\s*)gene_id\s+"([^"]+)"', col[8])
            if match is None or col[6] not in ("+", "-"):
                continue
            start, end = int(col[3]), int(col[4])
            record = (col[0], start - 1 if col[6] == "+" else end - 1, col[6])
            key = gene_key(match.group(1))
            if key in records and records[key] != record:
                raise RuntimeError(f"ambiguous GTF gene identifier: {key}")
            records[key] = record
    return records


def load_inputs():
    cols = [ID, SPLIT, LABEL, "chrom", "seq_start", "seq_end",
            "nearest_gene_id", "has_nearest_gene"]
    df = pq.read_table(CANON, columns=cols).to_pandas()
    if not df[ID].is_unique:
        raise RuntimeError("canonical locus identifiers are not unique")
    df = df.sort_values(ID, kind="stable").reset_index(drop=True)
    genes = read_genes()
    df["gene_key"] = df.nearest_gene_id.map(gene_key)
    present = df.has_nearest_gene.eq(1)
    if (present & ~df.gene_key.isin(genes)).any():
        raise RuntimeError("assigned gene identifiers are absent from the GTF")
    mapping = [genes.get(k) if has else None for k, has in zip(df.gene_key, present)]
    df["tss_chrom"] = [r[0] if r else "" for r in mapping]
    df["tss0"] = [r[1] if r else -1 for r in mapping]
    df["gene_strand"] = [r[2] if r else "." for r in mapping]
    df["herv_center"] = (df.seq_start.astype("int64") + df.seq_end.astype("int64")) // 2
    train, y = df[SPLIT].eq("train").to_numpy(), df[LABEL].to_numpy(np.int64)
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
    tracks = {mark: build_tracks(man.loc[man.track.eq(mark), "path"].tolist(),
                                  dtype=np.float64)
              for mark in MARKS}
    seq = np.lib.format.open_memmap(
        CACHE / "sequences.npy", mode="w+", dtype="uint8", shape=(len(loci), 2, LENGTH))
    feat = np.lib.format.open_memmap(
        CACHE / "features.npy", mode="w+", dtype="float32", shape=(len(loci), 2, 2))
    signal = np.lib.format.open_memmap(
        CACHE / "promoter_signal.npy", mode="w+", dtype="float32", shape=(len(loci), 1))

    def activity(chrom, center):
        start, end = int(center) - LENGTH // 2, int(center) + LENGTH // 2
        a, b = [track_mean(tracks[m], chrom, start, end) for m in MARKS]
        return float(np.log1p(np.sqrt(max(0.0, a) * max(0.0, b))))

    promoter_cache = {}
    for i, row in enumerate(loci.itertuples(index=False)):
        seq[i], feat[i], signal[i] = 4, 0, 0
        seq[i, 1] = ref.fetch_center(row.chrom, row.herv_center, LENGTH)
        feat[i, 1, 1] = activity(row.chrom, row.herv_center)
        if row.has_nearest_gene:
            key = (row.tss_chrom, row.tss0)
            if key not in promoter_cache:
                promoter_cache[key] = (ref.fetch_center(key[0], key[1], LENGTH),
                                       activity(*key))
            seq[i, 0], prm_act = promoter_cache[key]
            signal[i, 0] = prm_act
            feat[i, 0] = (1.0, prm_act)
            feat[i, 1, 0] = abs(row.herv_center - row.tss0) / 100000.0
    for array in (seq, feat, signal):
        array.flush()
    loci.to_parquet(CACHE / "loci.parquet", index=False)
    ref.close()


def encoder_weights(state, required):
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    for prefix in ("encoder.", "seq_encoder.", "model.encoder.", "model.seq_encoder.", ""):
        candidate = {k: state[prefix + k] for k in required if prefix + k in state}
        if set(candidate) == set(required) and all(candidate[k].shape == required[k].shape
                                                    for k in required):
            return candidate
    raise RuntimeError("checkpoint does not match the official sequence encoder")


def make_model(device):
    source = REPO / "EPInformer" / "models.py"
    spec = importlib.util.spec_from_file_location("_epinformer_pk_official", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    encoder = module.seq_256bp_encoder()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    if isinstance(state, nn.Module):
        state = state.state_dict()
    encoder.load_state_dict(encoder_weights(state, encoder.state_dict()), strict=True)
    encoder.requires_grad_(False)
    net = module.EPInformer_v2(pre_trained_encoder=encoder, n_enhancer=1,
                                n_encoder=3, out_dim=128, head=4, device=str(device),
                                useBN=True, useLN=True, useFeat=False,
                                usePromoterSignal=True, n_extraFeat=2)
    return net.to(device)


class PairedDataset(Dataset):
    def __init__(self, rows, labels):
        self.seq = np.load(CACHE / "sequences.npy", mmap_mode="r")
        self.features = np.load(CACHE / "features.npy", mmap_mode="r")
        self.signal = np.load(CACHE / "promoter_signal.npy", mmap_mode="r")
        self.rows, self.y = np.asarray(rows), np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return LUT[self.seq[row]], self.signal[row].copy(), self.features[row].copy(), self.y[i]


def run_seed(seed, rows, valid_rows, loci, device):
    seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    net = make_model(device)
    y = loci[LABEL].to_numpy(np.int64)
    loader = DataLoader(PairedDataset(rows, y[rows]), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(seed),
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=1e-6)
    criterion, history = nn.BCEWithLogitsLoss(), []
    for epoch in range(EPOCHS):
        net.train()
        net.seq_encoder.eval()
        total, count = 0.0, 0
        for seq, signal, features, label in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = net(seq.to(device), signal.to(device), features.to(device))
            loss = criterion(logits, label.to(device))
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * len(label), count + len(label)
        history.append({"epoch": epoch + 1, "train_loss": total / count})
    pd.DataFrame(history).to_csv(OUT / f"history_seed{seed}.csv", index=False)
    torch.save({"state_dict": net.state_dict(), "seed": seed, "method": METHOD},
               OUT / "weights" / f"epinformer_seed{seed}.pt")
    net.eval()
    loader = DataLoader(PairedDataset(valid_rows, y[valid_rows]), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for seq, signal, features, _ in loader:
            logits, _ = net(seq.to(device), signal.to(device), features.to(device))
            scores.append(logits.sigmoid().cpu().numpy())
    parameters = {"parameters": sum(p.numel() for p in net.parameters()),
                  "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)}
    return np.concatenate(scores).astype(np.float64), parameters


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

    rows, predictions, parameters = [], [], {}
    for seed in SEEDS:
        loci.iloc[training[seed]][[ID, SPLIT, LABEL]].to_csv(
            OUT / f"train_sample_ids_seed{seed}.tsv", sep="\t", index=False)
        scores, parameters = run_seed(seed, training[seed], valid_rows, loci, device)
        pred = valid.copy()
        pred["seed"], pred["score"] = seed, scores
        pred.to_parquet(OUT / f"valid_predictions_seed{seed}.parquet", index=False)
        predictions.append(pred)
        rows.append({"method": METHOD, "split": "valid", "seed": seed,
                     **metrics(y, scores, samples)})
        pd.DataFrame(rows).to_csv(OUT / "per_seed.csv", index=False)

    per_seed = pd.DataFrame(rows)
    summary = {"method": METHOD, "split": "valid", "seeds": "42;43;44", "seed_count": 3,
               "queue_loci": len(y), "queue_positives": int(y.sum()),
               "train_samples_per_seed": len(training[42]), "eval_repeats": EVAL_REPEATS,
               "eval_seed": EVAL_SEED, **parameters}
    summary.update(summarize(per_seed))
    pd.DataFrame([summary]).to_csv(OUT / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUT / "per_locus_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
