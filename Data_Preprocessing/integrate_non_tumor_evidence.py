#!/usr/bin/env python3


import pandas as pd
from pathlib import Path
from locus_contract import (
    LOCUS_KEY_COLUMNS,
    normalize_locus_columns,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/08_non_tumor_datahub"
INPUTS = {
    "herv_catalog": ROOT / "output/02_herv_clean_catalog/herv_clean_catalog_full.parquet",
    "fantom5": ROOT / "output/03_fantom5_rebuild/herv_fantom5_features_full.parquet",
    "non_tumor_roadmap": ROOT / "output/05_separated_roadmap_features/herv_non_tumor_roadmap_features_full.parquet",
    "serve_gtex": ROOT / "output/06_serve_gtex_features/herv_serve_gtex_features_full.parquet",
    "ccre_mpra": ROOT / "output/07_ccre_enhancer_features/herv_ccre_enhancer_features_full.parquet",
}

# Project contract: only Roadmap is split by tumor status. Rebuilt cCRE fields
# remain in the integrated DataHub only as archived evidence for old-baseline
# and leakage-case comparisons; they must NOT be consumed by the clean final
# model or by these Roadmap-derived count/score features. MPRA (iPSC+NPC),
# FANTOM5, and SERVE/GTEx non-malignant evidence are retained for training.

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = normalize_locus_columns(pd.read_parquet(INPUTS["herv_catalog"]))
    anchor_keys = set(map(tuple, anchor[LOCUS_KEY_COLUMNS].to_numpy()))
    master = anchor.copy()
    module_audit = []
    for name in ["fantom5", "non_tumor_roadmap", "serve_gtex", "ccre_mpra"]:
        frame = normalize_locus_columns(pd.read_parquet(INPUTS[name]))
        frame_keys = set(map(tuple, frame[LOCUS_KEY_COLUMNS].to_numpy()))
        if frame_keys != anchor_keys:
            missing_keys = sorted(anchor_keys - frame_keys)
            extra_keys = sorted(frame_keys - anchor_keys)
            raise ValueError(
                f"{name} locus-key set differs from catalog: "
                f"missing={missing_keys[:3]}, extra={extra_keys[:3]}"
            )
        drop = [c for c in ["is_model_universe", "datahub_split"] if c in frame]
        frame = frame.drop(columns=drop)
        frame = frame.drop(columns=["herv_internal_id"])
        master = master.merge(frame, on=LOCUS_KEY_COLUMNS, how="left",
                              validate="one_to_one", sort=False)
        new_columns = [col for col in frame.columns if col not in LOCUS_KEY_COLUMNS]
        module_audit.append({"module": name, "rows": len(frame),
                             "feature_columns": len(new_columns),
                             "locus_key_columns": "|".join(LOCUS_KEY_COLUMNS),
                             "input_path": str(INPUTS[name].resolve())})

    roadmap_any_active = (
        (master["roadmap_DNase_fdr001_sample_count"] > 0)
        | (master["roadmap_H3K27ac_sample_count"] > 0)
        | (master["roadmap_H3K4me1_sample_count"] > 0)
        | (master["roadmap_H3K4me3_sample_count"] > 0)
    ).astype("int8")
    roadmap_enhancer_like = (
        (master["roadmap_H3K27ac_sample_count"] > 0)
        | (master["roadmap_H3K4me1_sample_count"] > 0)
    ).astype("int8")
    master["roadmap_any_active"] = roadmap_any_active
    master["roadmap_enhancer_like"] = roadmap_enhancer_like
    master["regulatory_evidence_count"] = (
        roadmap_any_active +
        master["is_serve_gtex_supported"] + master["is_mpra_active_supported"]
    )
    master["enhancer_evidence_score"] = (
        roadmap_enhancer_like
    )
    roadmap_mark_totals = {
        "DNase_fdr001": 39,
        "H3K27ac": 93,
        "H3K4me1": 122,
        "H3K4me3": 122,
        "H3K27me3": 122,
        "H3K9me3": 122,
    }
    for mark, total in roadmap_mark_totals.items():
        master[f"roadmap_{mark}_sample_fraction"] = (
            master[f"roadmap_{mark}_sample_count"].astype(float) / total
        )
    master["roadmap_active_score"] = (
        master["roadmap_DNase_fdr001_sample_count"] + master["roadmap_H3K27ac_sample_count"] +
        master["roadmap_H3K4me1_sample_count"] + master["roadmap_H3K4me3_sample_count"]
    )
    master["roadmap_repressive_score"] = (
        master["roadmap_H3K27me3_sample_count"] + master["roadmap_H3K9me3_sample_count"]
    )
    master["roadmap_active_minus_repressive"] = master.roadmap_active_score - master.roadmap_repressive_score
    master["roadmap_active_score_normalized"] = (
        master["roadmap_DNase_fdr001_sample_fraction"]
        + master["roadmap_H3K27ac_sample_fraction"]
        + master["roadmap_H3K4me1_sample_fraction"]
        + master["roadmap_H3K4me3_sample_fraction"]
    )
    master["roadmap_repressive_score_normalized"] = (
        master["roadmap_H3K27me3_sample_fraction"]
        + master["roadmap_H3K9me3_sample_fraction"]
    )
    master["roadmap_active_minus_repressive_normalized"] = (
        master["roadmap_active_score_normalized"]
        - master["roadmap_repressive_score_normalized"]
    )
    # DP08 is the final eligibility gate: sequence QC plus complete module joins.
    # Evidence positivity is deliberately not required; all-zero evidence can be a valid negative.
    master["is_model_universe"] = master["sequence_qc_pass"].astype("int8")
    model = master[master.is_model_universe == 1].copy()

    full_path = OUT / "herv_non_tumor_datahub_full.parquet"
    model_path = OUT / "herv_non_tumor_datahub_model_universe.parquet"
    master.to_parquet(full_path, index=False)
    model.to_parquet(model_path, index=False)
    pd.DataFrame(module_audit).to_csv(OUT / "datahub_module_input_audit.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
