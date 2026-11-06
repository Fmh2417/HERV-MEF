#!/usr/bin/env python3


import json
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from liftover_utils import (
    lift_complete_bed,
    load_liftover,
)
from locus_contract import (
    LOCUS_KEY_COLUMNS,
    normalize_locus_columns,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "output/01_config/preprocessing_config.json"
MANIFEST_DIR = ROOT / "output/04_roadmap_manifests"
OUT = ROOT / "output/05_separated_roadmap_features"
CATALOG = ROOT / "output/02_herv_clean_catalog/herv_clean_catalog_full.parquet"
HERV_BED = ROOT / "output/02_herv_clean_catalog/herv_clean_catalog.bed"
MARKS = ["DNase_fdr001", "H3K27ac", "H3K4me1", "H3K4me3", "H3K27me3", "H3K9me3"]


def intersect(herv_bed: Path, mapped_bed: Path, tmp: Path) -> pd.DataFrame:
    raw = tmp / "intersect.tsv"
    with raw.open("w") as handle:
        subprocess.run(["bedtools", "intersect", "-a", str(herv_bed), "-b", str(mapped_bed), "-wo"],
                       stdout=handle, check=True)
    if raw.stat().st_size == 0:
        return pd.DataFrame(columns=["herv_internal_id", "overlap_bp"])
    ncols = len(raw.open().readline().rstrip("\n").split("\t"))
    frame = pd.read_csv(raw, sep="\t", header=None, usecols=[3, ncols - 1],
                        names=["herv_internal_id", "overlap_bp"])
    frame["herv_internal_id"] = frame.herv_internal_id.astype(str)
    frame["overlap_bp"] = frame.overlap_bp.astype(int)
    return frame[frame.overlap_bp > 0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text())
    chain = Path(cfg["inputs"]["hg19_to_hg38_chain"])
    lo = load_liftover(chain)
    catalog = pd.read_parquet(CATALOG, columns=["herv_internal_id", *LOCUS_KEY_COLUMNS,
                                                "is_model_universe", "datahub_split"])
    catalog = normalize_locus_columns(catalog)
    non_manifest = pd.read_csv(MANIFEST_DIR / "non_tumor_roadmap_manifest.csv")
    tumor_manifest = pd.read_csv(MANIFEST_DIR / "tumor_only_roadmap_manifest.csv")
    non_acc = {mark: {} for mark in MARKS}
    tumor_rows = []
    tumor_file_rows = []
    audits = []
    with tempfile.TemporaryDirectory(prefix="herv_roadmap2_") as temp_name:
        tmp = Path(temp_name)
        for analysis_set, manifest in [("non_tumor", non_manifest), ("tumor", tumor_manifest)]:
            for row in manifest.itertuples(index=False):
                mapped = OUT / "mapped" / f"{row.eid}_{row.mark}_{analysis_set}.hg38.bed.gz"
                unmapped = OUT / "unmapped" / f"{row.eid}_{row.mark}_{analysis_set}.unmapped.bed.gz"
                stats = lift_complete_bed(Path(row.peak_file), chain, mapped,
                                          output_unmapped=unmapped, lo=lo)
                hits = intersect(HERV_BED, mapped, tmp)
                audits.append({"analysis_set": analysis_set, "eid": row.eid, "mark": row.mark,
                               "source_file": row.peak_file, "source_assembly": "GRCh37/hg19",
                               "target_assembly": "GRCh38/hg38",
                               "mapped_file": str(mapped.resolve()),
                               "unmapped_file": str(unmapped.resolve()),
                               **stats})
                grouped = hits.groupby("herv_internal_id").overlap_bp.agg(["size", "max", "sum"])
                if analysis_set == "non_tumor":
                    non_acc[row.mark][row.eid] = grouped
                else:
                    tumor_file_rows.append({
                        "eid": row.eid, "sample_name": row.sample_name, "mark": row.mark,
                        "source_file": row.peak_file, "supported_hervs": int(len(grouped)),
                        "zero_overlap_hervs": int(len(catalog) - len(grouped)),
                        "total_catalog_hervs": int(len(catalog)),
                    })
                    for hid, values in grouped.iterrows():
                        tumor_rows.append({"herv_internal_id": hid, "eid": row.eid,
                                           "sample_name": row.sample_name, "mark": row.mark,
                                           "overlap_count": int(values["size"]),
                                           "max_overlap_bp": int(values["max"]),
                                           "total_overlap_bp": int(values["sum"]), "has_overlap": 1})

    ids = pd.Index(catalog.herv_internal_id.astype(str))
    for mark in MARKS:
        sample_count = pd.Series(0, index=ids, dtype="int32")
        max_overlap = pd.Series(0, index=ids, dtype="int32")
        total_overlap = pd.Series(0, index=ids, dtype="int64")
        for grouped in non_acc[mark].values():
            common = grouped.index.intersection(ids)
            sample_count.loc[common] += 1
            max_overlap.loc[common] = pd.concat([max_overlap.loc[common], grouped.loc[common, "max"]], axis=1).max(axis=1)
            total_overlap.loc[common] += grouped.loc[common, "sum"].astype("int64")
        catalog[f"roadmap_{mark}_sample_count"] = catalog.herv_internal_id.astype(str).map(sample_count).astype("int32")
        catalog[f"roadmap_{mark}_max_overlap_bp"] = catalog.herv_internal_id.astype(str).map(max_overlap).astype("int32")
        catalog[f"roadmap_{mark}_total_overlap_bp"] = catalog.herv_internal_id.astype(str).map(total_overlap).astype("int64")

    catalog.to_parquet(OUT / "herv_non_tumor_roadmap_features_full.parquet", index=False)
    tumor_long = pd.DataFrame(tumor_rows)
    if tumor_long.empty:
        tumor_long = pd.DataFrame(columns=["herv_internal_id", "eid", "sample_name", "mark", "overlap_count", "max_overlap_bp", "total_overlap_bp", "has_overlap"])
    tumor_long.to_parquet(OUT / "herv_tumor_cell_line_roadmap_evidence_long.parquet", index=False)
    tumor_file_audit = pd.DataFrame(tumor_file_rows).sort_values(["eid", "mark"])
    tumor_file_audit.to_csv(OUT / "tumor_evidence_summary_by_cell_line_mark.tsv", sep="\t", index=False)
    audit = pd.DataFrame(audits)
    audit.to_csv(OUT / "roadmap_hg19_to_hg38_liftover_audit.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
