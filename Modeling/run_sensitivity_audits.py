#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from modeling_common import (
    safe_ap,
    safe_auc,
    stable_order,
)


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import json
import math
import shutil
import numpy as np
import pandas as pd
import torch
import gc
import itertools
import torch.nn as nn
from datetime import (
    datetime,
    timezone,
)
from typing import Any
from torch.utils.data import (
    DataLoader,
    Dataset,
)

# --------------------------------------------------------------------------
# step 11: Audit the ERVK prior generalization
# --------------------------------------------------------------------------



# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELING_DIR = (PROJECT_ROOT / "Modeling").resolve()

SCRIPT_NAME = "11_audit_ervk_prior_generalization"
SCRIPT_ID = SCRIPT_NAME


M01_DIR = MODELING_DIR / "output" / "01_build_dual_tensors"
VALID_TENSOR = M01_DIR / "01_valid_dual_tensors.pt"
CANONICAL_TABLE = MODELING_DIR / 'output' / '00_build_canonical_data' / '00_canonical_herv_mef_table.parquet'

M10_DIR = MODELING_DIR / "output" / "10_eval_bio_refinement_B"
M10_DECISION = M10_DIR / "10_final_representation_decision.json"
M10_VALID_PREDICTIONS = M10_DIR / '10_final_representation_validation_predictions.parquet'

OUT_DIR = MODELING_DIR / "output" / SCRIPT_NAME


SEED = 99999999
ID_COL = "herv_internal_id"
SPLIT_COL = "split"
FANTOM_LABEL = "is_fantom5_enhancer"
FAMILY_COLUMN = "repFamily"
DIVERGENCE_COLUMN = "repeat_divergence"

BASELINE_SCORE_COLUMN = "score_sequence_regulatory"
CANDIDATE_SCORE_COLUMN = "score_biological_prior_candidate"
FINAL_SCORE_COLUMN = "score_final_representation"

AUDIT_FAMILIES = ("ERV1", "ERVK", "ERVL", "ERVL-MALR", "GYPSY")
RANDOM_1TO1_REPEATS = 100
RANDOM_1TO1_SEED = 99999999
TOPK_FRACTIONS = (0.01, 0.05)
TOPK_NAMES = {0.01: "top1", 0.05: "top5"}


# =============================================================================
# 2. Utilities
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)








def subgroup_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_1TO1_SEED)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) == 0 or len(negatives) == 0:
        return {
            "rows": int(len(labels)),
            "positives": int(labels.sum()),
            "auprc": float("nan"),
            "auroc": float("nan"),
            "top1_hits": 0,
            "top5_hits": 0,
        }

    auprcs: list[float] = []
    aurocs: list[float] = []
    for _ in range(RANDOM_1TO1_REPEATS):
        sampled = rng.choice(negatives, size=len(positives), replace=False)
        indices = np.concatenate([positives, sampled])
        indices.sort()
        auprcs.append(safe_ap(labels[indices], scores[indices]))
        aurocs.append(safe_auc(labels[indices], scores[indices]))

    total = len(labels)
    prevalence = float(labels.sum()) / total
    order = stable_order(scores)
    row: dict[str, Any] = {
        "rows": int(total),
        "positives": int(labels.sum()),
        "auprc": float(np.mean(auprcs)),
        "auroc": float(np.mean(aurocs)),
    }
    for fraction in TOPK_FRACTIONS:
        name = TOPK_NAMES[fraction]
        k = int(math.ceil(total * fraction))
        row[f"{name}_hits"] = int(labels[order[:k]].sum())
        row[f"{name}_fe"] = (
            row[f"{name}_hits"] / (k * prevalence)
            if k * prevalence > 0
            else float("nan")
        )
    return row


# =============================================================================
# 3. Main
# =============================================================================

def step_11() -> None:
    np.random.seed(SEED)
    prepare_output()


    decision = json.loads(M10_DECISION.read_text(encoding="utf-8"))
    score_column = str(decision.get("final_score_column", FINAL_SCORE_COLUMN))

    bundle = torch.load(VALID_TENSOR, map_location="cpu", weights_only=False)
    info = bundle["info_df"].reset_index(drop=True).copy()
    canonical = pd.read_parquet(
        CANONICAL_TABLE, columns=[ID_COL, FAMILY_COLUMN]
    )
    info = info.merge(canonical, on=ID_COL, how="left")
    info[FANTOM_LABEL] = pd.to_numeric(
        info[FANTOM_LABEL], errors="raise"
    ).astype(np.int8)
    info[FAMILY_COLUMN] = info[FAMILY_COLUMN].astype(str)
    labels = info[FANTOM_LABEL].to_numpy(dtype=np.int8)
    family_upper = info[FAMILY_COLUMN].str.upper()

    predictions = pd.read_parquet(M10_VALID_PREDICTIONS)
    final_scores = predictions[score_column].to_numpy(dtype=np.float64)
    baseline_scores = predictions[BASELINE_SCORE_COLUMN].to_numpy(
        dtype=np.float64
    )

    definitions = [
        {
            "subgroup": "ALL_VALID",
            "rule": "all validation rows",
            "mask": np.ones(len(info), dtype=bool),
        },
        {
            "subgroup": "ERVK",
            "rule": "upper(repFamily) == 'ERVK'",
            "mask": family_upper.eq("ERVK").to_numpy(dtype=bool),
        },
        {
            "subgroup": "NON_ERVK",
            "rule": "upper(repFamily) != 'ERVK'",
            "mask": ~family_upper.eq("ERVK").to_numpy(dtype=bool),
        },
    ]
    for name in AUDIT_FAMILIES:
        definitions.append(
            {
                "subgroup": name,
                "rule": f"upper(repFamily) == '{name}'",
                "mask": family_upper.eq(name).to_numpy(dtype=bool),
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for definition in definitions:
        mask = definition["mask"]
        if int(mask.sum()) == 0:
            continue
        row: dict[str, Any] = {
            "subgroup": definition["subgroup"],
            "rule": definition["rule"],
        }
        row.update(
            {
                f"final_{key}": value
                for key, value in subgroup_metrics(
                    labels[mask], final_scores[mask]
                ).items()
            }
        )
        row.update(
            {
                f"baseline_{key}": value
                for key, value in subgroup_metrics(
                    labels[mask], baseline_scores[mask]
                ).items()
            }
        )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)

    # Family-level findings: is the final score's AUROC preserved per family
    # relative to the frozen sequence baseline?
    findings: dict[str, Any] = {}
    for _, row in metrics.iterrows():
        subgroup = str(row["subgroup"])
        if subgroup == "ALL_VALID":
            continue
        final_auroc = float(row["final_auroc"])
        baseline_auroc = float(row["baseline_auroc"])
        findings[subgroup] = {
            "rows": int(row["final_rows"]),
            "positives": int(row["final_positives"]),
            "final_auroc": final_auroc,
            "baseline_auroc": baseline_auroc,
            "auroc_delta": final_auroc - baseline_auroc,
            "auroc_preserved": bool(final_auroc >= baseline_auroc - 0.02),
        }


# --------------------------------------------------------------------------
# step 29: Audit the FANTOM5 overlap threshold sensitivity
# --------------------------------------------------------------------------



# =============================================================================
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
MODELING = (ROOT / "Modeling").resolve()

SCRIPT = "29_fantom5_overlap_threshold_sensitivity"
VERSION = "round1_v2"

OUT = MODELING / "output" / "29"

M00_TABLE = MODELING / "output/00_build_canonical_data/00_canonical_herv_mef_table.parquet"
M03_PRED = MODELING / "output/03_audit_dual/03_dual_validation_predictions.parquet"
M23_ATLAS = MODELING / "output/23_generate_genomewide_herv_functional_ranking" / "23_herv_genomewide_functional_ranking.parquet"

OUT_LABEL_SUMMARY = OUT / "29_label_definition_summary.tsv"
OUT_SENSITIVITY = OUT / "29_threshold_sensitivity_metrics.tsv"
OUT_TIER_PROFILE = OUT / "29_overlap_tier_score_profile.tsv"
OUT_BOOTSTRAP = OUT / "29_threshold_delta_bootstrap.tsv"
OUT_PREDICTIONS = OUT / "29_validation_scores_and_threshold_labels.parquet"

ID = "herv_internal_id"
SPLIT = "split"
VALID = "valid"
EXPECTED_VALID_N = 72_048

THRESHOLDS = [
    ("ge1_bp_main", "is_fantom5_enhancer", 1, True),
    ("ge10_bp", "is_fantom5_enhancer_10bp", 10, False),
    ("ge50_bp", "is_fantom5_enhancer_50bp", 50, False),
    ("ge100_bp", "is_fantom5_enhancer_100bp", 100, False),
]
MAIN_TAG = "ge1_bp_main"
LABEL_COLS = [item[1] for item in THRESHOLDS]
STRICT_TAGS = [item[0] for item in THRESHOLDS if not item[3]]
ALL_TAGS = [item[0] for item in THRESHOLDS]

SCORE_SPECS = [
    ("stage1_score_sequence_mean", "M03_stage1_dual", "score_sequence_mean", False),
    ("stage1_score_ccre_enhancer", "M03_stage1_dual", "score_ccre_enhancer", False),
    ("stage1_score_serve_gtex_support", "M03_stage1_dual", "score_serve_gtex_support", False),
    ("stage1_score_mpra_activity", "M03_stage1_dual", "score_mpra_activity", False),
    ("frozen_score_sequence_regulatory", "M23_frozen_chain", "score_sequence_regulatory", False),
    ("frozen_score_final_representation", "M23_frozen_chain", "score_final_representation", False),
    ("frozen_score_final_global", "M23_frozen_chain", "score_final_global", True),
]
PRIMARY_SCORE_ID = next(s[0] for s in SCORE_SPECS if s[3])
PRIMARY_SCORE_COL = next(c for _, _, c, flag in SCORE_SPECS if flag)

TOPK_FRACTIONS__29 = (0.005, 0.010, 0.050)
TOPK_TAGS = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}

RANDOM_1TO1_REPEATS__29 = 100
RANDOM_1TO1_SEED__29 = 99999999
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 99999999

TIER_ORDER = ["MARGINAL_1_9BP", "LOW_10_49BP", "MID_50_99BP", "HIGH_GE100BP"]
NEGATIVE_TIER = "NEGATIVE_NO_OVERLAP"
TIER = "cage_overlap_tier"


# =============================================================================
# =============================================================================


def atomic_tsv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)






def random_1to1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_1TO1_SEED__29)
    aps, aucs = [], []
    for _ in range(RANDOM_1TO1_REPEATS__29):
        picked = rng.choice(negatives, size=len(positives), replace=False)
        idx = np.concatenate([positives, picked])
        aps.append(safe_ap(labels[idx], scores[idx]))
        aucs.append(safe_auc(labels[idx], scores[idx]))
    return {
        "auprc_1to1_mean": float(np.mean(aps)),
        "auprc_1to1_std": float(np.std(aps)),
        "auroc_1to1_mean": float(np.mean(aucs)),
        "auroc_1to1_std": float(np.std(aucs)),
    }


def topk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = int((labels == 1).sum())
    prevalence = positives / len(labels)
    out: dict[str, float] = {}
    for fraction in TOPK_FRACTIONS__29:
        k = int(math.floor(len(labels) * fraction))
        hits = int(labels[order[:k]].sum())
        tag = TOPK_TAGS[fraction]
        out[f"{tag}_k"] = k
        out[f"{tag}_hits"] = hits
        out[f"{tag}_fe"] = float((hits / k) / prevalence) if prevalence > 0 else float("nan")
    return out


# =============================================================================
# =============================================================================

def load_valid_panel() -> pd.DataFrame:
    canonical = pd.read_parquet(M00_TABLE, columns=[ID, SPLIT] + LABEL_COLS)
    valid_labels = canonical[canonical[SPLIT] == VALID]

    s1_cols = [c for _, s, c, _ in SCORE_SPECS if s == "M03_stage1_dual"]
    stage1 = pd.read_parquet(M03_PRED, columns=[ID, "is_fantom5_enhancer"] + s1_cols).rename(
        columns={"is_fantom5_enhancer": "m03_is_fantom5_enhancer"}
    )

    m23_cols = [c for _, s, c, _ in SCORE_SPECS if s == "M23_frozen_chain"]
    atlas = pd.read_parquet(M23_ATLAS, columns=[ID, SPLIT] + m23_cols)
    atlas = atlas[atlas[SPLIT] == VALID].drop(columns=[SPLIT])

    panel = valid_labels.merge(stage1, on=ID, how="inner").merge(atlas, on=ID, how="inner")
    panel = panel.sort_values(ID, kind="mergesort").reset_index(drop=True)

    return panel


def add_overlap_tier(panel: pd.DataFrame) -> pd.Series:
    conds = [
        panel["is_fantom5_enhancer_100bp"] == 1,
        panel["is_fantom5_enhancer_50bp"] == 1,
        panel["is_fantom5_enhancer_10bp"] == 1,
        panel["is_fantom5_enhancer"] == 1,
    ]
    choices = ["HIGH_GE100BP", "MID_50_99BP", "LOW_10_49BP", "MARGINAL_1_9BP"]
    return pd.Series(np.select(conds, choices, default=NEGATIVE_TIER), index=panel.index, name=TIER)


# =============================================================================
# =============================================================================

def build_label_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = len(panel)
    main_pos = int(panel["is_fantom5_enhancer"].sum())
    records = []
    for tag, col, min_bp, is_main in THRESHOLDS:
        pos = int(panel[col].sum())
        records.append({
            "label_tag": tag,
            "label_column": col,
            "min_overlap_bp": min_bp,
            "is_main_definition": is_main,
            "valid_rows": rows,
            "positives": pos,
            "negatives": rows - pos,
            "prevalence": pos / rows,
            "retained_vs_main": pos / main_pos,
            "positives_lost_vs_main": main_pos - pos,
        })
    return pd.DataFrame(records)


def build_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for score_id, source, col, is_primary in SCORE_SPECS:
        scores = panel[col].to_numpy(dtype=np.float64)
        for tag, label_col, min_bp, is_main in THRESHOLDS:
            labels = panel[label_col].to_numpy(dtype=np.int8)
            row = {
                "score_id": score_id,
                "score_source": source,
                "score_column": col,
                "is_primary_score": is_primary,
                "label_tag": tag,
                "label_column": label_col,
                "min_overlap_bp": min_bp,
                "is_main_definition": is_main,
                "valid_rows": len(labels),
                "positives": int(labels.sum()),
                "prevalence": float(labels.mean()),
            }
            row.update(random_1to1(labels, scores))
            row["auroc"] = safe_auc(labels, scores)
            row["auprc"] = safe_ap(labels, scores)
            row.update(topk_metrics(labels, scores))
            records.append(row)
    return pd.DataFrame(records)


def build_tier_profile(panel: pd.DataFrame) -> pd.DataFrame:
    tier = panel[TIER].to_numpy()
    records = []
    for score_id, source, col, is_primary in SCORE_SPECS:
        scores = panel[col].to_numpy(dtype=np.float64)
        rank_pct = pd.Series(scores).rank(pct=True, method="average").to_numpy()
        for tier_name in TIER_ORDER + [NEGATIVE_TIER]:
            mask = tier == tier_name
            record = {
                "score_id": score_id,
                "score_source": source,
                "score_column": col,
                "is_primary_score": is_primary,
                TIER: tier_name,
                "tier_rows": int(mask.sum()),
                "tier_fraction": float(mask.mean()),
                "mean_score": float(scores[mask].mean()),
                "median_score": float(np.median(scores[mask])),
                "mean_rank_pct": float(rank_pct[mask].mean()),
                "tier_vs_rest_auroc": safe_auc(mask.astype(np.int8), scores) if tier_name != NEGATIVE_TIER else float("nan"),
                "tier_vs_rest_auprc": safe_ap(mask.astype(np.int8), scores) if tier_name != NEGATIVE_TIER else float("nan"),
            }
            records.append(record)
    return pd.DataFrame(records)


def build_bootstrap(panel: pd.DataFrame) -> pd.DataFrame:
    scores = panel[PRIMARY_SCORE_COL].to_numpy(dtype=np.float64)
    main_labels = panel["is_fantom5_enhancer"].to_numpy(dtype=np.int8)
    rows = len(panel)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    records = []
    for tag, label_col, min_bp, _ in THRESHOLDS:
        if tag == MAIN_TAG:
            continue
        strict_labels = panel[label_col].to_numpy(dtype=np.int8)
        observed_auc = safe_auc(main_labels, scores) - safe_auc(strict_labels, scores)
        observed_ap = safe_ap(main_labels, scores) - safe_ap(strict_labels, scores)

        auc_deltas = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
        ap_deltas = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
        for r in range(BOOTSTRAP_REPEATS):
            idx = rng.integers(0, rows, size=rows)
            sc = scores[idx]
            auc_deltas[r] = safe_auc(main_labels[idx], sc) - safe_auc(strict_labels[idx], sc)
            ap_deltas[r] = safe_ap(main_labels[idx], sc) - safe_ap(strict_labels[idx], sc)

        for metric, obs, deltas in (("auroc", observed_auc, auc_deltas), ("auprc", observed_ap, ap_deltas)):
            low, high = np.percentile(deltas, [2.5, 97.5])
            records.append({
                "score_id": PRIMARY_SCORE_ID,
                "score_column": PRIMARY_SCORE_COL,
                "metric": metric,
                "main_label_tag": MAIN_TAG,
                "strict_label_tag": tag,
                "strict_min_overlap_bp": min_bp,
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "observed_delta_main_minus_strict": float(obs),
                "bootstrap_delta_mean": float(deltas.mean()),
                "bootstrap_delta_std": float(deltas.std(ddof=1)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
            })
    return pd.DataFrame(records)


def md_table(df: pd.DataFrame, columns: list[str], floats: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.{floats}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =============================================================================
# =============================================================================

def step_29() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


    panel = load_valid_panel()
    panel[TIER] = add_overlap_tier(panel)

    labels = build_label_summary(panel)
    sensitivity = build_sensitivity(panel)
    tier = build_tier_profile(panel)
    bootstrap = build_bootstrap(panel)

    atomic_tsv(labels, OUT_LABEL_SUMMARY)
    atomic_tsv(sensitivity, OUT_SENSITIVITY)
    atomic_tsv(tier, OUT_TIER_PROFILE)
    atomic_tsv(bootstrap, OUT_BOOTSTRAP)
    panel.to_parquet(OUT_PREDICTIONS, index=False)


# --------------------------------------------------------------------------
# step 30: Audit the K6/BPE complementarity enhancement
# --------------------------------------------------------------------------



# =============================================================================
# =============================================================================

ROOT__30 = Path(__file__).resolve().parent
SCRIPT__30 = "30_k6_bpe_complementarity_enhancement"
VERSION__30 = "v1"

OUT__30 = ROOT__30 / "output" / "30"

M01_DIR__30 = ROOT__30 / "output/01_build_dual_tensors"
VALID_TENSOR__30 = M01_DIR__30 / "01_valid_dual_tensors.pt"

M04_DIR = ROOT__30 / "output/04_train_sequence_branch_ablations"
M04_CKPT_DIR = M04_DIR / "04_checkpoints"

M05_PRED = ROOT__30 / "output/05_k6_bpe_complementarity_controls/05_validation_predictions.parquet"


TARGETS = ["is_ccre_enhancer_like", "is_serve_gtex_supported", "is_mpra_active_supported"]
HEADS = ["ccre_enhancer", "serve_gtex_support", "mpra_activity"]
FANTOM_LABEL__30, ID_COL__30 = "is_fantom5_enhancer", "herv_internal_id"

K6_LENGTH, BPE_LENGTH = 996, 256
SHUFFLED_MODEL = "k6_shuffled_bpe_meta"
ELIGIBLE_ROUTE_MODELS = ["dual_no_mpra_meta", "full_dual_meta", "bpe_meta", "k6_meta"]
ALL_MODELS = ELIGIBLE_ROUTE_MODELS + [SHUFFLED_MODEL]

SEED__30 = 99999999
VALID_BPE_SHUFFLE_SEED = 2026080502
RANDOM_1TO1_REPEATS__30 = 100
BOOTSTRAP_REPEATS__30 = 1000
STABILITY_PROBABILITY = 0.95

TOPK_FRACTIONS__30 = (0.005, 0.010, 0.050)
TOPK_NAMES__30 = {0.005: "top0_5", 0.010: "top1", 0.050: "top5"}

VALID_BATCH_SIZE = 128
NUM_WORKERS = 0

D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT = 128, 8, 4, 512, 0.1
META_EMBED_DIM, META_HIDDEN_DIM = 16, 64

# =============================================================================
# =============================================================================


def atomic_tsv__30(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def enh_safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")






def build_derangement(rows: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(rows).astype(np.int64)
    permutation = np.empty(rows, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    return permutation


def remap_branch_state_dict(state: dict) -> dict:
    # M04 uses emb_k6.*, whereas M05/M30 use k6.emb.*.
    for branch in ("k6", "bpe"):
        for block in ("emb", "cnn", "pos", "trans"):
            old_prefix = f"{block}_{branch}."
            new_prefix = f"{branch}.{block}."
            state = {
                new_prefix + key[len(old_prefix):] if key.startswith(old_prefix) else key: value
                for key, value in state.items()
            }
    return state


# =============================================================================
# =============================================================================

class DualDataset(Dataset):
    def __init__(self, bundle: dict, bpe_permutation: np.ndarray | None = None):
        self.bundle = bundle
        self.bpe_permutation = bpe_permutation

    def __len__(self) -> int:
        return int(self.bundle["labels"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        bpe_idx = idx if self.bpe_permutation is None else int(self.bpe_permutation[idx])
        return {
            "k6_ids": self.bundle["input_ids_k6"][idx].long(),
            "k6_mask": self.bundle["mask_k6"][idx].long(),
            "bpe_ids": self.bundle["input_ids_bpe"][bpe_idx].long(),
            "bpe_mask": self.bundle["mask_bpe"][bpe_idx].long(),
            "cat_ids": self.bundle["cat_ids"][idx].long(),
            "num_values": self.bundle["num_values"][idx].float(),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ComplementarityModel(nn.Module):
    def __init__(
        self,
        cfg: dict,
        sequence_mode: str = "dual",
        active_heads: tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        self.sequence_mode = sequence_mode
        d_model = int(cfg["d_model"])
        self.cat_features = list(cfg["cat_features"])

        def make_branch(vocab, max_len):
            return nn.ModuleDict({
                "emb": nn.Embedding(int(vocab), d_model, padding_idx=0),
                "cnn": nn.Sequential(
                    nn.Conv1d(d_model, d_model, 3, padding=1),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                ),
                "pos": PositionalEncoding(d_model, max_len, float(cfg["dropout"])),
                "trans": nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=int(cfg["nhead"]),
                        dim_feedforward=int(cfg["dim_feedforward"]),
                        dropout=float(cfg["dropout"]),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=int(cfg["num_layers"]),
                ),
            })

        if sequence_mode in ("k6", "dual"):
            self.k6 = make_branch(cfg["vocab_k6"], K6_LENGTH)
        if sequence_mode in ("bpe", "dual"):
            self.bpe = make_branch(cfg["vocab_bpe"], BPE_LENGTH)

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(
                int(cfg["cat_cardinalities"][f]),
                META_EMBED_DIM,
                padding_idx=0,
            )
            for f in self.cat_features
        ])
        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                len(self.cat_features) * META_EMBED_DIM + int(cfg["num_num_features"]),
                META_HIDDEN_DIM,
            ),
            nn.LayerNorm(META_HIDDEN_DIM),
            nn.GELU(),
        )

        if sequence_mode == "dual":
            self.bpe_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        self.meta_gate = nn.Sequential(
            nn.Linear(d_model + META_HIDDEN_DIM, d_model),
            nn.Sigmoid(),
        )
        self.project_meta = nn.Linear(META_HIDDEN_DIM, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(float(cfg["dropout"]))
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            for _ in HEADS
        ])

    def _encode(self, branch, ids, mask):
        x = branch["emb"](ids)
        x = branch["cnn"](x.transpose(1, 2)).transpose(1, 2)
        x = branch["pos"](x)
        return branch["trans"](x, src_key_padding_mask=(mask == 0))[:, 0, :]

    def forward_with_gates(
        self,
        k6_ids,
        k6_mask,
        bpe_ids,
        bpe_mask,
        cat_ids,
        num_values,
    ):
        parts = [
            self.cat_embeddings[i](cat_ids[:, i])
            for i in range(len(self.cat_features))
        ]
        f_meta = self.metadata_mlp(torch.cat(parts + [num_values], dim=1))
        gate_bpe = None

        if self.sequence_mode == "k6":
            anchor = self._encode(self.k6, k6_ids, k6_mask)
            gate_meta = self.meta_gate(torch.cat([anchor, f_meta], dim=1))
            fused = self.fuse_norm(anchor + gate_meta * self.project_meta(f_meta))
        elif self.sequence_mode == "bpe":
            anchor = self._encode(self.bpe, bpe_ids, bpe_mask)
            gate_meta = self.meta_gate(torch.cat([anchor, f_meta], dim=1))
            fused = self.fuse_norm(anchor + gate_meta * self.project_meta(f_meta))
        else:
            f_k6 = self._encode(self.k6, k6_ids, k6_mask)
            f_bpe = self._encode(self.bpe, bpe_ids, bpe_mask)
            gate_bpe = self.bpe_gate(torch.cat([f_k6, f_bpe], dim=1))
            gate_meta = self.meta_gate(torch.cat([f_k6, f_meta], dim=1))
            fused = self.fuse_norm(
                f_k6
                + gate_bpe * f_bpe
                + gate_meta * self.project_meta(f_meta)
            )

        out_heads = [
            h(self.fuse_dropout(fused)).squeeze(-1)
            for h in self.heads
        ]
        gates = {
            "bpe_gate_mean": (
                gate_bpe.mean(dim=1)
                if gate_bpe is not None
                else torch.full(
                    (fused.size(0),),
                    float("nan"),
                    device=fused.device,
                )
            ),
            "bpe_gate_std": (
                gate_bpe.std(dim=1)
                if gate_bpe is not None
                else torch.full(
                    (fused.size(0),),
                    float("nan"),
                    device=fused.device,
                )
            ),
            "metadata_gate_mean": gate_meta.mean(dim=1),
            "metadata_gate_std": gate_meta.std(dim=1),
        }
        return out_heads, gates


# =============================================================================
# =============================================================================

def build_pairwise_random1to1(
    repeats: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    rows = []
    for target, reference in itertools.permutations(ALL_MODELS, 2):
        for metric in ["random1to1_auprc", "random1to1_auroc"]:
            m_key = "auprc" if "auprc" in metric else "auroc"
            target_vals = np.array(repeats[target][m_key], dtype=np.float64)
            ref_vals = np.array(repeats[reference][m_key], dtype=np.float64)
            delta = target_vals - ref_vals
            rows.append({
                "target_model": target,
                "reference_model": reference,
                "metric": metric,
                "observed_mean_delta": float(delta.mean()),
                "delta_std": float(delta.std(ddof=0)),
                "delta_ci_low": float(np.quantile(delta, 0.025)),
                "delta_ci_high": float(np.quantile(delta, 0.975)),
                "probability_target_better": float(np.mean(delta > 0)),
                "probability_reference_better": float(np.mean(delta < 0)),
                "stable_target_better": bool(
                    np.mean(delta > 0) >= STABILITY_PROBABILITY
                ),
                "stable_reference_better": bool(
                    np.mean(delta < 0) >= STABILITY_PROBABILITY
                ),
                "paired_repeats": RANDOM_1TO1_REPEATS__30,
            })
    return pd.DataFrame(rows)


def build_topk_sets_and_overlap(
    scores_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    top_sets: dict[str, dict[str, np.ndarray]] = {}
    for model, sc in scores_by_model.items():
        order = np.argsort(-sc, kind="stable")
        top_sets[model] = {}
        for frac, name in TOPK_NAMES__30.items():
            k = int(math.floor(len(labels) * frac))
            top_sets[model][name] = order[:k]

    overlap_rows = []
    for left, right in itertools.combinations(ALL_MODELS, 2):
        for threshold in TOPK_NAMES__30.values():
            l_set = set(top_sets[left][threshold].tolist())
            r_set = set(top_sets[right][threshold].tolist())
            intersection = l_set & r_set
            union = l_set | r_set
            l_pos = {i for i in l_set if labels[i] == 1}
            r_pos = {i for i in r_set if labels[i] == 1}
            pos_inter = l_pos & r_pos
            pos_union = l_pos | r_pos
            overlap_rows.append({
                "left_model": left,
                "right_model": right,
                "threshold": threshold,
                "left_candidates": len(l_set),
                "right_candidates": len(r_set),
                "candidate_overlap": len(intersection),
                "candidate_jaccard": len(intersection) / max(len(union), 1),
                "left_positive_hits": len(l_pos),
                "right_positive_hits": len(r_pos),
                "positive_hit_overlap": len(pos_inter),
                "positive_hit_jaccard": len(pos_inter) / max(len(pos_union), 1),
                "left_unique_positive_hits": len(l_pos - r_pos),
                "right_unique_positive_hits": len(r_pos - l_pos),
            })
    return top_sets, pd.DataFrame(overlap_rows)


def build_topk_positive_bootstrap(
    top_sets: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
) -> pd.DataFrame:
    positives = np.flatnonzero(labels == 1)
    pos_map = {int(row): idx for idx, row in enumerate(positives)}
    memberships: dict[str, dict[str, np.ndarray]] = {
        m: {} for m in ALL_MODELS
    }

    for model in ALL_MODELS:
        for threshold, indices in top_sets[model].items():
            vec = np.zeros(len(positives), dtype=np.int8)
            for row in indices:
                if int(row) in pos_map:
                    vec[pos_map[int(row)]] = 1
            memberships[model][threshold] = vec

    rows = []
    pair_counter = 0
    for target, reference in itertools.permutations(ALL_MODELS, 2):
        for threshold in TOPK_NAMES__30.values():
            pair_counter += 1
            tgt_v = memberships[target][threshold]
            ref_v = memberships[reference][threshold]
            delta_vals = tgt_v.astype(np.int8) - ref_v.astype(np.int8)

            rng = np.random.default_rng(SEED__30 + pair_counter)
            sampled = rng.integers(
                0,
                len(delta_vals),
                size=(BOOTSTRAP_REPEATS__30, len(delta_vals)),
                endpoint=False,
            )
            delta = delta_vals[sampled].sum(axis=1)

            observed = int(tgt_v.sum() - ref_v.sum())
            rows.append({
                "target_model": target,
                "reference_model": reference,
                "metric": threshold,
                "observed_hit_delta": observed,
                "bootstrap_mean_delta": float(delta.mean()),
                "delta_ci_low": float(np.quantile(delta, 0.025)),
                "delta_ci_high": float(np.quantile(delta, 0.975)),
                "probability_target_better": float(np.mean(delta > 0)),
                "probability_reference_better": float(np.mean(delta < 0)),
                "probability_tie": float(np.mean(delta == 0)),
                "stable_target_better": bool(
                    np.mean(delta > 0) >= STABILITY_PROBABILITY
                ),
                "stable_reference_better": bool(
                    np.mean(delta < 0) >= STABILITY_PROBABILITY
                ),
                "positive_rows_resampled": len(positives),
                "bootstrap_repeats": BOOTSTRAP_REPEATS__30,
            })
    return pd.DataFrame(rows)


def infer_all_gate_values(
    valid_bundle: dict,
    info: pd.DataFrame,
    labels: np.ndarray,
    score_seq_mean: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_frames = []

    m05_ckpt = ROOT__30 / "output/05_k6_bpe_complementarity_controls/05_checkpoints" / f"05_{SHUFFLED_MODEL}.pt"
    checkpoints = {
        "dual_no_mpra_meta": M04_CKPT_DIR / "04_dual_no_mpra_meta.pt",
        "full_dual_meta": M04_CKPT_DIR / "04_full_dual_meta.pt",
        SHUFFLED_MODEL: m05_ckpt,
    }

    for model_name, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = ComplementarityModel(
            cfg,
            sequence_mode="dual",
            active_heads=(0, 1) if "no_mpra" in model_name else (0, 1, 2),
        )
        state = remap_branch_state_dict(ckpt["model"])
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

        permutation = (
            build_derangement(len(info), VALID_BPE_SHUFFLE_SEED)
            if model_name == SHUFFLED_MODEL else None
        )
        valid_loader = DataLoader(
            DualDataset(valid_bundle, bpe_permutation=permutation),
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        bpe_mean, bpe_std, meta_mean, meta_std = [], [], [], []
        with torch.no_grad():
            for b in valid_loader:
                k6_ids = b["k6_ids"].to(device)
                k6_mask = b["k6_mask"].to(device)
                bpe_ids = b["bpe_ids"].to(device)
                bpe_mask = b["bpe_mask"].to(device)
                cat_ids = b["cat_ids"].to(device)
                num_val = b["num_values"].to(device)

                _, gates = model.forward_with_gates(
                    k6_ids,
                    k6_mask,
                    bpe_ids,
                    bpe_mask,
                    cat_ids,
                    num_val,
                )

                bpe_mean.append(
                    gates["bpe_gate_mean"].detach().cpu().numpy()
                )
                bpe_std.append(
                    gates["bpe_gate_std"].detach().cpu().numpy()
                )
                meta_mean.append(
                    gates["metadata_gate_mean"].detach().cpu().numpy()
                )
                meta_std.append(
                    gates["metadata_gate_std"].detach().cpu().numpy()
                )

        df_gate = pd.DataFrame({
            "model": model_name,
            "row_index": np.arange(len(info), dtype=np.int64),
            ID_COL__30: info[ID_COL__30].to_numpy(),
            FANTOM_LABEL__30: labels,
            "score": (
                score_seq_mean[model_name]
                if model_name in score_seq_mean else np.nan
            ),
            "bpe_gate_mean": np.concatenate(bpe_mean),
            "bpe_gate_std": np.concatenate(bpe_std),
            "metadata_gate_mean": np.concatenate(meta_mean),
            "metadata_gate_std": np.concatenate(meta_std),
        })
        value_frames.append(df_gate)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    values = (
        pd.concat(value_frames, ignore_index=True)
        if value_frames else pd.DataFrame()
    )

    summary_rows = []
    if not values.empty:
        for model_name in values["model"].unique():
            m_vals = values[values["model"] == model_name].copy()
            order = np.argsort(-m_vals["score"].to_numpy(), kind="stable")
            masks = {
                "all": np.ones(len(m_vals), dtype=bool),
                "fantom5_positive": m_vals[FANTOM_LABEL__30].to_numpy() == 1,
            }
            for frac, name in TOPK_NAMES__30.items():
                k = int(math.floor(len(m_vals) * frac))
                mask = np.zeros(len(m_vals), dtype=bool)
                mask[order[:k]] = True
                masks[name] = mask

            for group, mask in masks.items():
                for g_col in ["bpe_gate_mean", "metadata_gate_mean"]:
                    arr = m_vals.loc[mask, g_col].to_numpy()
                    finite = arr[np.isfinite(arr)]
                    if len(finite) == 0:
                        continue
                    summary_rows.append({
                        "model": model_name,
                        "group": group,
                        "gate": g_col,
                        "rows": len(finite),
                        "mean": float(finite.mean()),
                        "std": float(finite.std(ddof=0)),
                        "p05": float(np.quantile(finite, 0.05)),
                        "p25": float(np.quantile(finite, 0.25)),
                        "median": float(np.quantile(finite, 0.50)),
                        "p75": float(np.quantile(finite, 0.75)),
                        "p95": float(np.quantile(finite, 0.95)),
                    })
    return values, pd.DataFrame(summary_rows)


def build_absorption_and_specific_candidates(
    top_sets: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    info: pd.DataFrame,
    scores_by_model: dict[str, np.ndarray],
    gate_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absorption_summaries = []
    threshold_labels = {
        "top0_5": "Top0.5%",
        "top1": "Top1%",
        "top5": "Top5%",
    }
    pos_indices = set(np.flatnonzero(labels == 1))

    for threshold in ["top0_5", "top1", "top5"]:
        k6 = set(top_sets["k6_meta"][threshold]) & pos_indices
        bpe = set(top_sets["bpe_meta"][threshold]) & pos_indices
        dual = set(top_sets["dual_no_mpra_meta"][threshold]) & pos_indices

        shared = k6 & bpe
        k6_specific = k6 - bpe
        bpe_specific = bpe - k6
        single_union = k6 | bpe

        absorption_summaries.append({
            "threshold": threshold,
            "threshold_label": threshold_labels[threshold],
            "k6_positive_hits": len(k6),
            "bpe_positive_hits": len(bpe),
            "dual_positive_hits": len(dual),
            "k6_bpe_shared_positive_hits": len(shared),
            "k6_specific_positive_hits": len(k6_specific),
            "bpe_specific_positive_hits": len(bpe_specific),
            "single_branch_union_positive_hits": len(single_union),
            "dual_retained_shared_positive_hits": len(shared & dual),
            "dual_absorbed_k6_specific_positive_hits": len(k6_specific & dual),
            "dual_absorbed_bpe_specific_positive_hits": len(bpe_specific & dual),
            "dual_covered_single_union_positive_hits": len(single_union & dual),
            "dual_unique_positive_hits": len(dual - single_union),
            "dual_lost_shared_positive_hits": len(shared - dual),
            "dual_lost_k6_specific_positive_hits": len(k6_specific - dual),
            "dual_lost_bpe_specific_positive_hits": len(bpe_specific - dual),
            "k6_bpe_positive_jaccard": enh_safe_ratio(
                len(shared), len(single_union)
            ),
            "dual_coverage_of_single_union": enh_safe_ratio(
                len(single_union & dual), len(single_union)
            ),
            "dual_retention_of_shared": enh_safe_ratio(
                len(shared & dual), len(shared)
            ),
            "dual_absorption_of_k6_specific": enh_safe_ratio(
                len(k6_specific & dual), len(k6_specific)
            ),
            "dual_absorption_of_bpe_specific": enh_safe_ratio(
                len(bpe_specific & dual), len(bpe_specific)
            ),
            "dual_unique_fraction": enh_safe_ratio(
                len(dual - single_union), len(dual)
            ),
            "dual_minus_k6_positive_hits": len(dual) - len(k6),
            "dual_minus_bpe_positive_hits": len(dual) - len(bpe),
        })

    ranks_by_model = {
        m: np.argsort(
            np.argsort(-sc, kind="stable"),
            kind="stable",
        ) + 1
        for m, sc in scores_by_model.items()
    }

    gates_dual = pd.DataFrame()
    if (
        not gate_values.empty
        and "dual_no_mpra_meta" in gate_values["model"].values
    ):
        gates_dual = gate_values[
            gate_values["model"] == "dual_no_mpra_meta"
        ].set_index("row_index")

    candidate_rows = []
    for threshold in ["top0_5", "top1", "top5"]:
        k6 = set(top_sets["k6_meta"][threshold]) & pos_indices
        bpe = set(top_sets["bpe_meta"][threshold]) & pos_indices
        dual = set(top_sets["dual_no_mpra_meta"][threshold]) & pos_indices
        all_ids = sorted(k6 | bpe | dual)

        for idx in all_ids:
            in_k6 = idx in k6
            in_bpe = idx in bpe
            in_dual = idx in dual

            if in_k6 and in_bpe:
                source = "k6_bpe_shared"
                action = (
                    "dual_retained_shared"
                    if in_dual else "dual_lost_shared"
                )
            elif in_k6:
                source = "k6_specific"
                action = (
                    "dual_absorbed_k6_specific"
                    if in_dual else "dual_lost_k6_specific"
                )
            elif in_bpe:
                source = "bpe_specific"
                action = (
                    "dual_absorbed_bpe_specific"
                    if in_dual else "dual_lost_bpe_specific"
                )
            else:
                source = "neither_single_branch"
                action = (
                    "dual_unique_positive"
                    if in_dual else "not_in_any_topk"
                )

            candidate_rows.append({
                "threshold": threshold,
                "row_index": idx,
                ID_COL__30: info.loc[idx, ID_COL__30],
                "source_category": source,
                "dual_action": action,
                "in_k6_topk": in_k6,
                "in_bpe_topk": in_bpe,
                "in_dual_topk": in_dual,
                "rank_k6_meta": ranks_by_model["k6_meta"][idx],
                "rank_bpe_meta": ranks_by_model["bpe_meta"][idx],
                "rank_dual_no_mpra_meta": ranks_by_model[
                    "dual_no_mpra_meta"
                ][idx],
                "score_k6_meta": scores_by_model["k6_meta"][idx],
                "score_bpe_meta": scores_by_model["bpe_meta"][idx],
                "score_dual_no_mpra_meta": scores_by_model[
                    "dual_no_mpra_meta"
                ][idx],
                "bpe_gate_mean": (
                    gates_dual.loc[idx, "bpe_gate_mean"]
                    if idx in gates_dual.index else np.nan
                ),
                "metadata_gate_mean": (
                    gates_dual.loc[idx, "metadata_gate_mean"]
                    if idx in gates_dual.index else np.nan
                ),
            })

    return pd.DataFrame(absorption_summaries), pd.DataFrame(candidate_rows)


def md_table__30(
    df: pd.DataFrame,
    columns: list[str],
    floats: int = 4,
) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(
                    "" if not np.isfinite(v)
                    else f"{v:.{floats}f}"
                )
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =============================================================================
# =============================================================================

def step_30() -> None:
    OUT__30.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    preds = pd.read_parquet(M05_PRED)

    valid_b = torch.load(VALID_TENSOR__30, weights_only=False)
    info = valid_b["info_df"].reset_index(drop=True)
    info[ID_COL__30] = info[ID_COL__30].astype(str)
    preds[ID_COL__30] = preds[ID_COL__30].astype(str)
    labels = info[FANTOM_LABEL__30].to_numpy(np.int8)

    scores_by_model: dict[str, np.ndarray] = {}
    repeats_record: dict[str, dict[str, list[float]]] = {
        m: {"auprc": [], "auroc": []}
        for m in ALL_MODELS
    }

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(SEED__30)
    samples_1to1 = [
        np.concatenate([
            pos_idx,
            rng.choice(neg_idx, size=len(pos_idx), replace=False),
        ])
        for _ in range(RANDOM_1TO1_REPEATS__30)
    ]

    for mod in ALL_MODELS:
        sub = (
            preds.loc[preds["model"].eq(mod)]
            .set_index(ID_COL__30)
            .loc[info[ID_COL__30].to_numpy()]
        )
        sc = sub["score_sequence_mean"].to_numpy(dtype=np.float64)
        scores_by_model[mod] = sc

        for idx in samples_1to1:
            repeats_record[mod]["auprc"].append(
                safe_ap(labels[idx], sc[idx])
            )
            repeats_record[mod]["auroc"].append(
                safe_auc(labels[idx], sc[idx])
            )

    top_sets, overlap_df = build_topk_sets_and_overlap(
        scores_by_model,
        labels,
    )


    gate_values, gate_summary = infer_all_gate_values(
        valid_b,
        info,
        labels,
        scores_by_model,
        device,
    )

    absorption_df, candidates_df = build_absorption_and_specific_candidates(
        top_sets,
        labels,
        info,
        scores_by_model,
        gate_values,
    )


def main() -> None:
    step_11()
    step_29()
    step_30()


if __name__ == "__main__":
    main()
