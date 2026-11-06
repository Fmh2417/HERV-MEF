#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CADUCEUS_CACHE,
    CADUCEUS_WEIGHTS,
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
    DNABERT2_CACHE,
    DNABERT2_WEIGHTS,
    HYENADNA_CACHE,
    HYENADNA_WEIGHTS,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "output" / "00_sequence_pk_master" / "sequence_pk_master.parquet"
CANONICAL = (
    ROOT.parent / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)

ID = "herv_internal_id"
SEQUENCE_COL = "sequence"

EXPECTED_TOTAL_ROWS = 480_306
CADUCEUS_DIM = 256
DNABERT2_DIM = 768
HYENADNA_DIM = 256

CADUCEUS_BATCH = 128
DNABERT2_BATCH = 32
HYENADNA_BATCH = 64

RC_TRANSLATION = str.maketrans(
    {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TRANSLATION)[::-1]


def load_master() -> pd.DataFrame:
    build_master = not MASTER.is_file()
    columns = [ID, SEQUENCE_COL]
    if build_master:
        columns += ["split", "is_fantom5_enhancer"]
    master = pd.read_parquet(CANONICAL if build_master else MASTER, columns=columns)
    master[ID] = master[ID].astype(str)
    master = master.sort_values(ID, kind="mergesort").reset_index(drop=True)
    if not master[ID].is_unique:
        raise RuntimeError("locus identifiers are not unique in the master table")
    if len(master) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"master table has {len(master):,} rows, expected {EXPECTED_TOTAL_ROWS:,}"
        )
    if build_master:
        master["split"] = master["split"].astype(str)
        master["is_fantom5_enhancer"] = pd.to_numeric(
            master["is_fantom5_enhancer"], errors="raise"
        ).astype("int8")
        MASTER.parent.mkdir(parents=True, exist_ok=True)
        master.to_parquet(MASTER, index=False)
    return master[[ID, SEQUENCE_COL]]


def load_backbone(weights: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        weights, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        weights, trust_remote_code=True, local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model.eval()


def encode(tokenizer, model, device, sequences):
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    special_tokens_mask = encoded.get("special_tokens_mask")
    if special_tokens_mask is None:
        special_tokens_mask = torch.tensor(
            [
                tokenizer.get_special_tokens_mask(
                    row.tolist(), already_has_special_tokens=True
                )
                for row in input_ids
            ],
            dtype=torch.long,
        )

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key != "special_tokens_mask" and isinstance(value, torch.Tensor)
    }
    attention_mask = attention_mask.to(device)
    special_tokens_mask = special_tokens_mask.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(**model_inputs, return_dict=True)
        hidden_states = outputs.last_hidden_state

    effective_mask = attention_mask.to(dtype=torch.bool) & ~special_tokens_mask.to(
        dtype=torch.bool
    )
    effective_counts = effective_mask.sum(dim=1)
    if not bool((effective_counts > 0).all().item()):
        raise RuntimeError("a sequence has no effective non-special tokens")
    weights = effective_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    pooled = (
        (hidden_states * weights).sum(dim=1)
        / effective_counts.unsqueeze(-1).to(dtype=hidden_states.dtype)
    )
    return pooled.detach().to(dtype=torch.float32).cpu()


def encode_dnabert2(tokenizer, model, device, sequences):
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(**model_inputs, output_all_encoded_layers=False)
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise RuntimeError("unexpected DNABERT-2 forward output contract")
        hidden_states = outputs[0]

    effective_mask = model_inputs["attention_mask"].to(dtype=torch.bool) & ~(
        special_tokens_mask.to(dtype=torch.bool)
    )
    effective_counts = effective_mask.sum(dim=1)
    if not bool((effective_counts > 0).all().item()):
        raise RuntimeError("a sequence has no effective non-special tokens")
    weights = effective_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    pooled = (
        (hidden_states * weights).sum(dim=1)
        / effective_counts.unsqueeze(-1).to(dtype=hidden_states.dtype)
    )
    return pooled.detach().to(dtype=torch.float32).cpu()


def build_caduceus(master, device):
    tokenizer, model = load_backbone(CADUCEUS_WEIGHTS)
    model = model.to(device)
    sequences = master[SEQUENCE_COL].astype(str)
    forward = torch.empty((len(master), CADUCEUS_DIM), dtype=torch.float32)
    reverse = torch.empty((len(master), CADUCEUS_DIM), dtype=torch.float32)
    for start in range(0, len(master), CADUCEUS_BATCH):
        end = min(start + CADUCEUS_BATCH, len(master))
        block = sequences.iloc[start:end].tolist()
        rc_block = [reverse_complement(seq) for seq in block]
        pooled = encode(tokenizer, model, device, block + rc_block)
        size = end - start
        forward[start:end] = pooled[:size]
        reverse[start:end] = pooled[size:]
    if not bool(torch.isfinite(forward).all()):
        raise RuntimeError("Caduceus-Ph produced non-finite embeddings")
    return {"forward": forward, "reverse_complement": reverse}


def build_dnabert2(master, device):
    tokenizer, model = load_backbone(DNABERT2_WEIGHTS)
    model = model.to(device)
    sequences = master[SEQUENCE_COL].astype(str)
    embeddings = torch.empty((len(master), DNABERT2_DIM), dtype=torch.float32)
    for start in range(0, len(master), DNABERT2_BATCH):
        end = min(start + DNABERT2_BATCH, len(master))
        block = sequences.iloc[start:end].tolist()
        embeddings[start:end] = encode_dnabert2(tokenizer, model, device, block)
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("DNABERT-2 produced non-finite embeddings")
    return embeddings


def build_hyenadna(master, device):
    tokenizer, model = load_backbone(HYENADNA_WEIGHTS)
    model = model.to(device)
    sequences = master[SEQUENCE_COL].astype(str)
    embeddings = torch.empty((len(master), HYENADNA_DIM), dtype=torch.float32)
    for start in range(0, len(master), HYENADNA_BATCH):
        end = min(start + HYENADNA_BATCH, len(master))
        block = sequences.iloc[start:end].tolist()
        embeddings[start:end] = encode(tokenizer, model, device, block)
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("HyenaDNA produced non-finite embeddings")
    return {"embeddings": embeddings}


def save(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backbone embedding extraction")
    device = torch.device("cuda:0")
    master = load_master()

    save(build_caduceus(master, device), CADUCEUS_CACHE)
    save(build_dnabert2(master, device), DNABERT2_CACHE)
    save(build_hyenadna(master, device), HYENADNA_CACHE)


if __name__ == "__main__":
    main()
