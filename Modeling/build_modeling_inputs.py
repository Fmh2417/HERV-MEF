#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import numpy as np
import pandas as pd
import gc
import joblib
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
)
from tokenizers import (
    Tokenizer,
    models,
    processors,
    trainers,
)

# --------------------------------------------------------------------------
# step 00: Build the canonical HERV locus table
# --------------------------------------------------------------------------



ROOT = Path(__file__).resolve().parent
DATAHUB = ROOT.parent / "Data_Preprocessing" / "output" / "08_non_tumor_datahub" / "herv_non_tumor_datahub_model_universe.parquet"
OUT = ROOT / "output" / "00_build_canonical_data"

SPLITS = ["train", "valid", "test"]

CORE = [
    "chrom",
    "start",
    "end",
    "strand",
    "repName",
    "repClass",
    "repFamily",
    "repeat_divergence",
    "length",
    "sequence",
    "seq_start",
    "seq_end",
    "seq_len",
    "gc_content",
    "n_content",
    "dist_to_tss",
    "nearest_gene_id",
    "nearest_gene_name",
    "nearest_gene_type",
    "nearest_gene_strand",
    "has_nearest_gene",
]

FANTOM_LABELS = [
    "is_fantom5_enhancer",
    "is_fantom5_enhancer_10bp",
    "is_fantom5_enhancer_50bp",
    "is_fantom5_enhancer_100bp",
]

ROADMAP_MARKS = [
    "DNase_fdr001",
    "H3K27ac",
    "H3K4me1",
    "H3K4me3",
    "H3K27me3",
    "H3K9me3",
]

ROADMAP_RAW = [
    f"roadmap_{mark}_{stat}"
    for mark in ROADMAP_MARKS
    for stat in ("sample_count", "max_overlap_bp", "total_overlap_bp")
]

ROADMAP_DERIVED = [
    "roadmap_any_active",
    "roadmap_enhancer_like",
    "regulatory_evidence_count",
    "enhancer_evidence_score",
    *[f"roadmap_{mark}_sample_fraction" for mark in ROADMAP_MARKS],
    "roadmap_active_score_normalized",
    "roadmap_repressive_score_normalized",
    "roadmap_active_minus_repressive_normalized",
]

SERVE = [
    "serve_gtex_tissue_count",
    "serve_gtex_total_overlap_bp",
    "is_serve_gtex_supported",
    "serve_gtex_annotation_count",
    "serve_noncoding_overlap_count",
    "serve_noncoding_total_overlap_bp",
]

CCRE_FLAGS = [
    "is_ccre_ca_overlap",
    "is_ccre_enhancer_like",
    "is_ccre_ctcf_overlap",
    "is_ccre_pls_overlap",
]

CCRE_NUMERIC = [
    "ccre_els_overlap_count",
    "ccre_els_total_overlap_bp",
]

CCRE = [*CCRE_FLAGS, *CCRE_NUMERIC]

MPRA = [
    "mpra_overlap_count",
    "mpra_max_overlap_bp",
    "mpra_total_overlap_bp",
    "mpra_active_overlap_count",
    "mpra_active_max_overlap_bp",
    "mpra_active_total_overlap_bp",
    "is_mpra_overlap",
    "is_mpra_active_supported",
]

TARGETS = ["is_ccre_enhancer_like", "is_serve_gtex_supported", "is_mpra_active_supported"]

FAMILY_FLAGS = ["is_ERV1", "is_ERVK", "is_ERVL", "is_ERVL_MALR", "is_GYPSY"]

OUTPUT_COLUMNS = [
    "herv_internal_id",
    "split",
    *CORE,
    *FANTOM_LABELS,
    *ROADMAP_RAW,
    *ROADMAP_DERIVED,
    *SERVE,
    *CCRE,
    *MPRA,
    *FAMILY_FLAGS,
    "repeat_name",
    "repeat_class",
    "repeat_family",
    "distance_to_TSS",
]


def family_flags(frame: pd.DataFrame) -> pd.DataFrame:
    family = frame["repFamily"].astype(str).str.upper()
    out = pd.DataFrame(index=frame.index)
    out["is_ERV1"] = family.eq("ERV1").astype(np.int8)
    out["is_ERVK"] = family.eq("ERVK").astype(np.int8)
    out["is_ERVL"] = family.eq("ERVL").astype(np.int8)
    out["is_ERVL_MALR"] = family.eq("ERVL-MALR").astype(np.int8)
    out["is_GYPSY"] = family.isin(["GYPSY", "GYPSY?"]).astype(np.int8)
    return out


def build() -> pd.DataFrame:
    source = pd.read_parquet(DATAHUB)
    frame = source.copy()
    frame["split"] = frame["datahub_split"].astype(str)
    frame["repeat_name"] = frame["repName"].astype(str)
    frame["repeat_class"] = frame["repClass"].astype(str)
    frame["repeat_family"] = frame["repFamily"].astype(str)
    frame["distance_to_TSS"] = pd.to_numeric(frame["dist_to_tss"], errors="raise")
    frame["has_nearest_gene"] = pd.to_numeric(
        frame["has_nearest_gene"], errors="raise"
    ).astype(np.int8)

    for column in FANTOM_LABELS + TARGETS + CCRE_FLAGS + ["is_mpra_overlap"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int8)

    for column in CCRE_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)


    flags = family_flags(frame)
    frame = pd.concat([frame, flags], axis=1)
    return frame[OUTPUT_COLUMNS].copy()


def split_summary(canonical: pd.DataFrame) -> dict:
    rows = []
    for split in SPLITS:
        sub = canonical[canonical["split"] == split]
        row = {
            "split": split,
            "rows": int(len(sub)),
            "fantom5_positives": int(sub["is_fantom5_enhancer"].sum()),
        }
        for target in TARGETS:
            row[f"{target}_positives"] = int(sub[target].sum())
        rows.append(row)
    return rows


def write_outputs(canonical: pd.DataFrame, summary: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(OUT / "00_canonical_herv_mef_table.parquet", index=False)
    canonical[
        ["herv_internal_id", "split", "is_fantom5_enhancer", "is_fantom5_enhancer_50bp"]
    ].to_csv(OUT / "00_canonical_split_label.tsv", sep="\t", index=False)


def step_00() -> None:
    canonical = build()
    summary = {
        "rows": int(len(canonical)),
        "columns": int(len(canonical.columns)),
        "split_summary": split_summary(canonical),
        "targets": TARGETS,
        "fantom5_is_evaluation_only": True,
        "cCRE_source": "non_tumor_cCRE_noccl",
        "cCRE_included_in_canonical": True,
    }
    write_outputs(canonical, summary)


# --------------------------------------------------------------------------
# step 01: Build the dual-branch sequence and metadata tensors
# --------------------------------------------------------------------------



ROOT__01 = Path(__file__).resolve().parent
M00_DIR = ROOT__01 / "output" / "00_build_canonical_data"
CANONICAL_TABLE = M00_DIR / "00_canonical_herv_mef_table.parquet"

OUT_DIR = ROOT__01 / "output" / "01_build_dual_tensors"
OUT_TRAIN = OUT_DIR / "01_train_dual_tensors.pt"
OUT_VALID = OUT_DIR / "01_valid_dual_tensors.pt"
OUT_TEST = OUT_DIR / "01_test_dual_inputs.pt"
OUT_ENCODER = OUT_DIR / "01_metadata_encoder.joblib"
OUT_SCALER = OUT_DIR / "01_metadata_scaler.joblib"
OUT_TOKENIZER = OUT_DIR / "01_herv_bpe_tokenizer.json"


ID_COL = "herv_internal_id"
SPLIT_COL = "split"
FANTOM_LABEL_COL = "is_fantom5_enhancer"
FANTOM_COMPAT_COL = "is_fantom5_enhancer_50bp"
SPLITS__01 = ["train", "valid", "test"]
DEVELOPMENT_SPLITS = ["train", "valid"]

TARGETS__01 = [
    "is_ccre_enhancer_like",
    "is_serve_gtex_supported",
    "is_mpra_active_supported",
]

CAT_FEATURES = ["repClass", "repFamily", "repName", "nearest_gene_type"]
NUM_FEATURES = [
    "length",
    "repeat_divergence",
    "gc_content",
    "n_content",
    "dist_to_tss",
]

BASE_COLUMNS = [
    ID_COL,
    SPLIT_COL,
    "chrom",
    "start",
    "end",
    "strand",
    *CAT_FEATURES,
    *NUM_FEATURES,
    "sequence",
]
DEV_LABEL_COLUMNS = [
    ID_COL,
    SPLIT_COL,
    FANTOM_LABEL_COL,
    FANTOM_COMPAT_COL,
    *TARGETS__01,
]

BPE_VOCAB_TARGET = 8192
BPE_MIN_FREQUENCY = 2
BPE_MAX_LEN = 256
BPE_PAD_TOKEN = "[PAD]"
BPE_CLS_TOKEN = "[CLS]"
BPE_UNK_TOKEN = "[UNK]"
BPE_SPECIAL_TOKENS = [BPE_PAD_TOKEN, BPE_CLS_TOKEN, BPE_UNK_TOKEN]
BPE_TRAIN_BATCH_SIZE = 4096
BPE_ENCODE_BATCH_SIZE = 2048

K6_K = 6
K6_MAX_LEN = 996
K6_PAD_ID = 0
K6_CLS_ID = 1
K6_UNK_ID = 2
K6_OFFSET = 3
K6_VOCAB_SIZE = K6_OFFSET + (4**K6_K)

TOKEN_DTYPE = np.int16
MASK_DTYPE = np.uint8
CAT_DTYPE = np.int32

BYTE_TO_VAL = np.full(256, -1, dtype=np.int16)
BYTE_TO_VAL[ord("A")] = 0
BYTE_TO_VAL[ord("C")] = 1
BYTE_TO_VAL[ord("G")] = 2
BYTE_TO_VAL[ord("T")] = 3
K6_POWERS = (4 ** np.arange(K6_K - 1, -1, -1)).astype(np.int32)


def load_split_frames() -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    base = pd.read_parquet(CANONICAL_TABLE, columns=BASE_COLUMNS)
    base[ID_COL] = base[ID_COL].astype(str)
    base[SPLIT_COL] = base[SPLIT_COL].astype(str)

    split_counts = {s: int((base[SPLIT_COL] == s).sum()) for s in SPLITS__01}
    for column in CAT_FEATURES:
        base[column] = base[column].astype(str)
    for column in NUM_FEATURES:
        base[column] = pd.to_numeric(base[column], errors="raise")

    labels = pd.read_parquet(
        CANONICAL_TABLE,
        columns=DEV_LABEL_COLUMNS,
        filters=[(SPLIT_COL, "in", DEVELOPMENT_SPLITS)],
    )
    labels[ID_COL] = labels[ID_COL].astype(str)
    for column in [FANTOM_LABEL_COL, FANTOM_COMPAT_COL, *TARGETS__01]:
        labels[column] = pd.to_numeric(labels[column], errors="raise").astype(
            np.int8
        )

    split_frames: dict[str, pd.DataFrame] = {}
    for split in DEVELOPMENT_SPLITS:
        part = base.loc[base[SPLIT_COL] == split].reset_index(drop=True)
        ordered = (
            labels.set_index(ID_COL).loc[part[ID_COL].tolist()].reset_index()
        )
        split_frames[split] = part.merge(
            ordered.drop(columns=[SPLIT_COL]), on=ID_COL, how="left", sort=False
        )
    split_frames["test"] = base.loc[base[SPLIT_COL] == "test"].reset_index(
        drop=True
    )
    return split_frames, split_counts


def transformed_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[NUM_FEATURES].copy()
    numeric["dist_to_tss"] = np.sign(numeric["dist_to_tss"]) * np.log10(
        np.abs(numeric["dist_to_tss"]) + 1.0
    )
    return numeric


def fit_metadata(train_frame: pd.DataFrame):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64
    )
    encoder.fit(train_frame[CAT_FEATURES].astype(str))
    scaler = StandardScaler()
    scaler.fit(transformed_numeric(train_frame))
    cardinalities = {
        feature: int(len(categories) + 1)
        for feature, categories in zip(CAT_FEATURES, encoder.categories_)
    }
    return encoder, scaler, cardinalities


def transform_metadata(
    frame, encoder, scaler
) -> tuple[np.ndarray, np.ndarray]:
    raw = encoder.transform(frame[CAT_FEATURES].astype(str))
    cat_ids = (raw + 1).astype(CAT_DTYPE)
    cat_ids[cat_ids < 0] = 0
    num_values = scaler.transform(transformed_numeric(frame)).astype(np.float32)
    return cat_ids, num_values


def train_bpe(train_frame: pd.DataFrame) -> Tokenizer:
    sequences = train_frame["sequence"].astype(str).tolist()

    def batches():
        for start in range(0, len(sequences), BPE_TRAIN_BATCH_SIZE):
            yield sequences[start : start + BPE_TRAIN_BATCH_SIZE]

    tokenizer = Tokenizer(models.BPE(unk_token=BPE_UNK_TOKEN))
    tokenizer.train_from_iterator(
        batches(),
        trainer=trainers.BpeTrainer(
            vocab_size=BPE_VOCAB_TARGET,
            min_frequency=BPE_MIN_FREQUENCY,
            special_tokens=BPE_SPECIAL_TOKENS,
            initial_alphabet=list("ACGTN"),
            show_progress=True,
        ),
        length=len(sequences),
    )
    cls_id = int(tokenizer.token_to_id(BPE_CLS_TOKEN))
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BPE_CLS_TOKEN} $A", special_tokens=[(BPE_CLS_TOKEN, cls_id)]
    )
    tokenizer.save(str(OUT_TOKENIZER))
    return Tokenizer.from_file(str(OUT_TOKENIZER))


def bpe_encode(
    frame: pd.DataFrame, tokenizer: Tokenizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(frame)
    input_ids = np.zeros((n_rows, BPE_MAX_LEN), dtype=TOKEN_DTYPE)
    attention_mask = np.zeros((n_rows, BPE_MAX_LEN), dtype=MASK_DTYPE)
    lengths = np.zeros(n_rows, dtype=np.int32)
    sequences = frame["sequence"].astype(str).tolist()
    for start in range(0, n_rows, BPE_ENCODE_BATCH_SIZE):
        end = min(start + BPE_ENCODE_BATCH_SIZE, n_rows)
        for offset, encoded in enumerate(
            tokenizer.encode_batch(sequences[start:end])
        ):
            row = start + offset
            keep = encoded.ids[:BPE_MAX_LEN]
            lengths[row] = len(encoded.ids)
            input_ids[row, : len(keep)] = np.asarray(keep, dtype=TOKEN_DTYPE)
            attention_mask[row, : len(keep)] = 1
    return input_ids, attention_mask, lengths


def encode_one_k6(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    ids = np.zeros(K6_MAX_LEN, dtype=TOKEN_DTYPE)
    mask = np.zeros(K6_MAX_LEN, dtype=MASK_DTYPE)
    ids[0] = K6_CLS_ID
    mask[0] = 1
    values = BYTE_TO_VAL[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    windows = sliding_window_view(values, K6_K)
    valid = (windows >= 0).all(axis=1)
    codes = windows.astype(np.int32) @ K6_POWERS
    kmer_ids = np.where(valid, K6_OFFSET + codes, K6_UNK_ID).astype(TOKEN_DTYPE)
    n = min(len(kmer_ids), K6_MAX_LEN - 1)
    ids[1 : 1 + n] = kmer_ids[:n]
    mask[1 : 1 + n] = 1
    return ids, mask


def k6_encode(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(frame)
    input_ids = np.zeros((n_rows, K6_MAX_LEN), dtype=TOKEN_DTYPE)
    attention_mask = np.zeros((n_rows, K6_MAX_LEN), dtype=MASK_DTYPE)
    for row, sequence in enumerate(frame["sequence"].astype(str).values):
        ids, mask = encode_one_k6(sequence)
        input_ids[row] = ids
        attention_mask[row] = mask
    return input_ids, attention_mask


def save_bundle(
    split: str,
    frame: pd.DataFrame,
    tokenizer: Tokenizer,
    encoder,
    scaler,
    cardinalities: dict[str, int],
) -> dict:
    development = split in DEVELOPMENT_SPLITS
    cat_ids, num_values = transform_metadata(frame, encoder, scaler)
    bpe_ids, bpe_mask, bpe_lengths = bpe_encode(frame, tokenizer)
    k6_ids, k6_mask = k6_encode(frame)

    tensors = {
        "input_ids_bpe": torch.from_numpy(bpe_ids),
        "mask_bpe": torch.from_numpy(bpe_mask),
        "input_ids_k6": torch.from_numpy(k6_ids),
        "mask_k6": torch.from_numpy(k6_mask),
        "cat_ids": torch.from_numpy(cat_ids),
        "num_values": torch.from_numpy(num_values),
    }
    if development:
        tensors["labels"] = torch.from_numpy(
            frame[TARGETS__01].to_numpy(dtype=np.float32, copy=True)
        )

    info_columns = [ID_COL, SPLIT_COL, "chrom", "start", "end", "strand"]
    if development:
        info_df = frame[
            info_columns + [FANTOM_LABEL_COL, FANTOM_COMPAT_COL, *TARGETS__01]
        ].reset_index(drop=True)
    else:
        info_df = frame[info_columns].reset_index(drop=True)

    bundle = {
        **tensors,
        "info_df": info_df,
        "target_names": TARGETS__01 if development else [],
        "targets_present": development,
        "cat_cardinalities": cardinalities,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "bpe_vocab_size": int(tokenizer.get_vocab_size()),
        "k6_vocab_size": K6_VOCAB_SIZE,
        "split": split,
        "rows": len(frame),
    }
    path = {"train": OUT_TRAIN, "valid": OUT_VALID, "test": OUT_TEST}[split]
    torch.save(bundle, path)

    row = {
        "split": split,
        "path": str(path),
        "rows": len(frame),
        "bpe_shape": str(tuple(bpe_ids.shape)),
        "k6_shape": str(tuple(k6_ids.shape)),
        "mean_bpe_length": float(bpe_lengths.mean()),
        "bpe_truncation_rate": float(np.mean(bpe_lengths > BPE_MAX_LEN)),
        "targets_present": development,
    }
    if development:
        for target in TARGETS__01:
            row[f"{target}_positive_rate"] = float(frame[target].mean())
    return row


def step_01() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_frames, split_counts = load_split_frames()

    encoder, scaler, cardinalities = fit_metadata(split_frames["train"])
    joblib.dump(encoder, OUT_ENCODER)
    joblib.dump(scaler, OUT_SCALER)

    tokenizer = train_bpe(split_frames["train"])

    manifest_rows = []
    for split in SPLITS__01:
        manifest_rows.append(
            save_bundle(
                split,
                split_frames[split],
                tokenizer,
                encoder,
                scaler,
                cardinalities,
            )
        )
        del split_frames[split]
        gc.collect()


def main() -> None:
    step_00()
    step_01()


if __name__ == "__main__":
    main()
