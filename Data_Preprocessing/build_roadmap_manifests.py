#!/usr/bin/env python3


import json
import re
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "output/01_config/preprocessing_config.json"
SPLIT_DIR = ROOT / "output/00_sample_split"
OUT = ROOT / "output/04_roadmap_manifests"
MARKS = ["DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3"]
EXPECTED_NON_TUMOR = dict(DNase_fdr001=39, H3K27ac=93, H3K4me1=122,
                          H3K4me3=122, H3K27me3=122, H3K9me3=122)
EXPECTED_TUMOR = dict(DNase_fdr001=0, H3K27ac=5, H3K4me1=5,
                      H3K4me3=5, H3K27me3=5, H3K9me3=5)


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def counts(frame: pd.DataFrame) -> dict[str, int]:
    observed = frame.groupby("mark").size().to_dict()
    return {mark: int(observed.get(mark, 0)) for mark in MARKS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    peak_root = Path(cfg["inputs"]["roadmap"]["peak_root"])
    meta_path = Path(cfg["inputs"]["roadmap"]["eid_metadata"])
    non_tumor_ids = read_ids(SPLIT_DIR / "non_tumor_eids.txt")
    tumor_ids = read_ids(SPLIT_DIR / "tumor_eids.txt")

    meta = pd.read_csv(meta_path, sep="\t", dtype=str).fillna("")
    meta = meta.rename(columns={"EID": "eid", "GROUP": "sample_group", "ANATOMY": "sample_anatomy",
                                "TYPE": "sample_type", "STD_NAME": "sample_name"})
    meta = meta[["eid", "sample_group", "sample_anatomy", "sample_type", "sample_name"]]
    rows = []
    pattern = re.compile(r"^(E\d{3})")
    for mark in MARKS:
        files = sorted(set((peak_root / mark).glob("*.bed.gz")) |
                       set((peak_root / mark).glob("*.narrowPeak.gz")))
        for path in files:
            match = pattern.match(path.name)
            rows.append({"mark": mark, "eid": match.group(1), "peak_file": str(path.resolve()),
                         "file_basename": path.name,
                         "source_assembly": "GRCh37/hg19",
                         "target_assembly": "GRCh38/hg38", "liftover_required": 1})
    full = pd.DataFrame(rows).merge(meta, on="eid", how="left", validate="many_to_one")
    non_tumor = full[full.eid.isin(non_tumor_ids)].copy()
    tumor = full[full.eid.isin(tumor_ids)].copy()

    non_tumor.to_csv(OUT / "non_tumor_roadmap_manifest.csv", index=False)
    tumor.to_csv(OUT / "tumor_only_roadmap_manifest.csv", index=False)
    pd.DataFrame([
        {"analysis_set": label, "mark": mark, "peak_files": value}
        for label, frame in [("non_tumor_training", non_tumor), ("tumor_external_projection", tumor)]
        for mark, value in counts(frame).items()
    ]).to_csv(OUT / "roadmap_manifest_summary_by_set.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
