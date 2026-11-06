#!/usr/bin/env python3


import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import DATASET


PROJECT_DIR = Path(__file__).resolve().parent
ROADMAP_ROOT = DATASET / "roadmap"
METADATA_PATH = ROADMAP_ROOT / "EID_metadata.tab.txt"
OUT_DIR = PROJECT_DIR / "output" / "00_sample_split"

MARKS = [
    "DNase_fdr001",
    "H3K27ac",
    "H3K4me1",
    "H3K4me3",
    "H3K27me3",
    "H3K9me3",
]
TUMOR_EIDS = {"E114", "E115", "E117", "E118", "E123"}
IMMORTALIZED_EIDS = {"E116"}


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def scan_peaks() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^(E\d{3})")
    for mark in MARKS:
        mark_dir = ROADMAP_ROOT / mark
        files = sorted(set(mark_dir.glob("*.bed.gz")) | set(mark_dir.glob("*.narrowPeak.gz")))
        for path in files:
            match = pattern.match(path.name)
            rows.append({
                "mark": mark,
                "eid": match.group(1),
                "peak_file": str(path.resolve()),
                "file_name": path.name,
                "file_size": int(path.stat().st_size),
            })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["mark", "eid"], kind="mergesort").reset_index(drop=True)


def count_by_mark(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby("mark").size().to_dict()
    return {mark: int(counts.get(mark, 0)) for mark in MARKS}


def main() -> None:
    setup()

    meta = pd.read_csv(METADATA_PATH, sep="\t", dtype=str).fillna("")
    peaks = scan_peaks()
    peak_eids = set(peaks["eid"])

    meta = meta.loc[meta["EID"].isin(peak_eids)].copy()
    meta["is_explicit_tumor"] = meta["EID"].isin(TUMOR_EIDS)
    meta["is_immortalized_background"] = meta["EID"].isin(IMMORTALIZED_EIDS)
    meta["analysis_set"] = meta["is_explicit_tumor"].map(
        {True: "tumor_external_projection", False: "non_tumor_training"}
    )
    meta["decision_reason"] = "non-tumor Roadmap reference sample"
    meta.loc[meta["is_explicit_tumor"], "decision_reason"] = (
        "explicit carcinoma/leukemia cell line; held out from model training"
    )
    meta.loc[meta["is_immortalized_background"], "decision_reason"] = (
        "GM12878 lymphoblastoid background; retained as non-tumor but flagged"
    )

    presence = (
        peaks.assign(present=1)
        .pivot_table(index="eid", columns="mark", values="present", aggfunc="max", fill_value=0)
        .reindex(columns=MARKS, fill_value=0)
        .add_prefix("has_")
        .reset_index()
    )
    audit = meta.merge(presence, left_on="EID", right_on="eid", how="left", validate="one_to_one")
    audit = audit.drop(columns="eid").rename(columns={"EID": "eid"})
    audit = audit.sort_values("eid", kind="mergesort").reset_index(drop=True)

    non_tumor = peaks.loc[~peaks["eid"].isin(TUMOR_EIDS)].copy()

    audit_path = OUT_DIR / "roadmap_sample_audit.tsv"
    non_tumor_path = OUT_DIR / "non_tumor_eids.txt"
    tumor_path = OUT_DIR / "tumor_eids.txt"

    audit.to_csv(audit_path, sep="\t", index=False)
    pd.Series(sorted(set(non_tumor["eid"]))).to_csv(non_tumor_path, index=False, header=False)
    pd.Series(sorted(TUMOR_EIDS)).to_csv(tumor_path, index=False, header=False)


if __name__ == "__main__":
    main()
