#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDA_VISIBLE_DEVICES,
)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from pk_common import (
    EVAL_REPEATS,
    EVAL_SEED,
    METRIC_COLS,
    SEEDS,
    balanced_samples,
    metrics,
    sequence_model_parameters,
)


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

CANON = (
    PROJECT / "Modeling" / "output" / "00_build_canonical_data"
    / "00_canonical_herv_mef_table.parquet"
)
RANK23 = (
    PROJECT / "Modeling" / "output" / "23_generate_genomewide_herv_functional_ranking"
    / "23_herv_genomewide_functional_ranking.parquet"
)
FROZEN = PROJECT / "Modeling" / "output" / "22_finalize_development_model_registry" / "22_frozen"
GLOBAL_CKPT = FROZEN / "22_final_global_ranker.pt"
FER_CKPT = FROZEN / "22_primary_fer_component.pt"
RUNS = {
    "ChromEnhancer-DNN": ROOT / "output" / "04_chromenhancer_dnn",
    "HEAP": ROOT / "output" / "05_heap",
    "EPCOT": ROOT / "output" / "06_epcot",
    "EPInformer": ROOT / "output" / "07_epinformer",
    "UniversalEPI": ROOT / "output" / "08_universalepi",
}
OUT = ROOT / "output" / "09_final_comparison"

ID, SPLIT, LABEL = "herv_internal_id", "split", "is_fantom5_enhancer"
SPLIT_NAME = "valid"

METHODS = {
    "HERV-MEF": "HERV-MEF final model (M17 -> M21 -> M22, frozen)",
    "ChromEnhancer-DNN": "ChromEnhancer-DNN (Model 1, HERV task adaptation)",
    "HEAP": "HEAP (HERV task adaptation)",
    "EPCOT": "EPCOT-EAP (HERV task adaptation)",
    "EPInformer": "EPInformer-v2 (HERV/nearest-promoter task adaptation)",
    "UniversalEPI": "UniversalEPI (HERV task adaptation)",
}
ROLES = {
    "HERV-MEF": "this work",
    "ChromEnhancer-DNN": "published baseline",
    "HEAP": "published baseline",
    "EPCOT": "published baseline",
    "EPInformer": "published baseline",
    "UniversalEPI": "published baseline",
}
PARAMETER_NOTES = {
    "HERV-MEF": "full stack = frozen K6+BPE sequence model + global ranker "
                "(95 encoded columns from 52 source features) + FER (38); only the "
                "ranking stage is fitted to the FANTOM5 label",
    "ChromEnhancer-DNN": "official ChrNet1 DNN, 86 encoded columns, trained from scratch",
    "HEAP": "official HEAP_W with 6 chromatin tracks, all weights trained",
    "EPCOT": "official EAP backbone frozen, head retrained",
    "EPInformer": "official encoder frozen, remaining main-model layers retrained",
    "UniversalEPI": "official Stage1 CNN frozen, Stage2 STG and transformer layers retrained",
}


class _Net(nn.Module):
    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net


def load_queue() -> pd.DataFrame:
    canon = pq.read_table(CANON, columns=[ID, SPLIT, LABEL]).to_pandas()
    if canon[ID].isna().any() or not canon[ID].is_unique:
        raise RuntimeError("canonical table has missing or duplicated locus identifiers")
    queue = canon[canon[SPLIT] == SPLIT_NAME].copy()
    return queue.sort_values(ID, kind="stable").reset_index(drop=True)


def load_herv_mef(queue: pd.DataFrame) -> np.ndarray:
    rank = pq.read_table(RANK23, columns=[ID, "herv_functional_priority_rank"]).to_pandas()
    if not rank[ID].is_unique:
        raise RuntimeError("ranking table has duplicated locus identifiers")
    merged = queue.merge(rank, on=ID, how="left", validate="one_to_one")
    if len(merged) != len(queue) or merged["herv_functional_priority_rank"].isna().any():
        raise RuntimeError("queued loci are missing their final rank")
    ranks = merged["herv_functional_priority_rank"].to_numpy(dtype=np.float64)
    if not np.isfinite(ranks).all():
        raise RuntimeError("final ranks contain non-finite values")
    return -ranks


def load_seed_scores(queue: pd.DataFrame, pred_path: Path, label: str) -> dict:
    pred = pd.read_parquet(pred_path)
    if not {"seed", "score"} <= set(pred.columns):
        raise RuntimeError(f"{label} predictions lack the seed or score column")
    if pred.duplicated(["seed", ID]).any():
        raise RuntimeError(f"{label} predictions duplicate loci within a seed")
    scores = {}
    for seed in SEEDS:
        part = pred[pred["seed"] == seed]
        if part.empty:
            continue
        merged = queue.merge(part[[ID, SPLIT, LABEL, "score"]], on=ID, how="left",
                             suffixes=("_canon", "_pred"))
        if len(merged) != len(queue) or merged["score"].isna().any():
            raise RuntimeError(f"{label} seed {seed} does not cover the queue")
        if not (merged[f"{SPLIT}_canon"] == merged[f"{SPLIT}_pred"]).all():
            raise RuntimeError(f"{label} seed {seed} split disagrees with the canonical table")
        if not (merged[f"{LABEL}_canon"] == merged[f"{LABEL}_pred"]).all():
            raise RuntimeError(f"{label} seed {seed} label disagrees with the canonical table")
        s = merged["score"].to_numpy(dtype=np.float64)
        if not np.isfinite(s).all():
            raise RuntimeError(f"{label} seed {seed} has non-finite scores")
        scores[seed] = s
    return scores


def herv_mef_parameters() -> tuple:
    global_state = torch.load(GLOBAL_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    global_ranker = _Net(nn.Sequential(
        nn.Linear(global_state["net.0.weight"].shape[1], 128), nn.BatchNorm1d(128),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 1),
    ))
    global_ranker.load_state_dict(global_state)

    fer_state = torch.load(FER_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    fer = _Net(nn.Sequential(
        nn.Linear(fer_state["net.0.weight"].shape[1], 64), nn.LayerNorm(64), nn.GELU(),
        nn.Linear(64, 16), nn.LayerNorm(16), nn.GELU(),
        nn.Linear(16, 1),
    ))
    fer.load_state_dict(fer_state)

    ranking = (sum(p.numel() for p in global_ranker.parameters())
               + sum(p.numel() for p in fer.parameters()))
    return ranking + sequence_model_parameters(), ranking


def recorded_parameters(model: str) -> tuple:
    table = pd.read_csv(RUNS[model] / "comparison.csv")
    return int(table["parameters"].iloc[0]), int(table["trainable_parameters"].iloc[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue = load_queue()
    y = queue[LABEL].to_numpy(np.int8)
    samples = balanced_samples(y)

    runs = []
    herv_score = load_herv_mef(queue)
    runs.append({"method": METHODS["HERV-MEF"], "model": "HERV-MEF", "seed": "frozen",
                 "score_source": "step-23 final ranking",
                 **metrics(y, herv_score, samples)})
    seed_scores = {}
    for model, run_dir in RUNS.items():
        seed_scores[model] = load_seed_scores(
            queue, run_dir / "per_locus_predictions.parquet", model)
        for seed, score in seed_scores[model].items():
            runs.append({"method": METHODS[model], "model": model, "seed": seed,
                         "score_source": "per_locus_predictions.parquet",
                         **metrics(y, score, samples)})

    per_run = pd.DataFrame(runs)
    per_run.insert(0, "split", SPLIT_NAME)
    per_run["queue_loci"] = len(queue)
    per_run["queue_positives"] = int(y.sum())
    per_run.to_csv(OUT / "per_run_valid.csv", index=False)

    aligned = pd.DataFrame({
        ID: queue[ID].to_numpy(), SPLIT: queue[SPLIT].to_numpy(), LABEL: y,
        "herv_mef_priority_rank": (-herv_score).astype(np.int64),
        "herv_mef_score": herv_score,
    })
    for model, scores in seed_scores.items():
        for seed, score in scores.items():
            aligned[f"{model.lower()}_score_seed{seed}"] = score
    aligned.to_parquet(OUT / "aligned_predictions_valid.parquet", index=False)

    np.savez_compressed(OUT / "eval_samples_valid.npz", indices=samples,
                        eval_seed=EVAL_SEED, repeats=EVAL_REPEATS, split=SPLIT_NAME)

    full, ranking = herv_mef_parameters()
    parameters = {"HERV-MEF": (full, ranking)}
    for model in RUNS:
        parameters[model] = recorded_parameters(model)

    rows = []
    for model, note in PARAMETER_NOTES.items():
        block = per_run[per_run["model"] == model]
        entry = {
            "model": model,
            "role": ROLES[model],
            "method": METHODS[model],
            "seeds": "frozen" if len(block) == 1
            else ";".join(str(s) for s in block["seed"].tolist()),
            "parameters": parameters[model][0],
            "trainable_parameters": parameters[model][1],
            "parameters_note": note,
        }
        for col in METRIC_COLS:
            entry[f"{col}_mean"] = float(block[col].mean())
            entry[f"{col}_std"] = float(block[col].std(ddof=1)) if len(block) > 1 else None
        entry["queue_loci"] = len(queue)
        entry["queue_positives"] = int(y.sum())
        entry["split"] = SPLIT_NAME
        rows.append(entry)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("auprc_1to1_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(OUT / "final_comparison_valid.csv", index=False)


if __name__ == "__main__":
    main()
