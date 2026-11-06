import os
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from locus_contract import (
    LOCUS_KEY_COLUMNS,
    normalize_locus_columns,
)


# ============================================================
# ============================================================

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")

DATA_SUMMARY_DIR = Path(__file__).resolve().parent
CONFIG_PATH = DATA_SUMMARY_DIR / "output/01_config/preprocessing_config.json"

STEP01_DIR = DATA_SUMMARY_DIR / "output/02_herv_clean_catalog"
HERV_CATALOG_PARQUET = STEP01_DIR / "herv_clean_catalog_full.parquet"
HERV_BED = STEP01_DIR / "herv_clean_catalog.bed"

OUT_DIR = DATA_SUMMARY_DIR / "output/06_serve_gtex_features"
TMP_RUN_DIR = OUT_DIR / "tmp" / str(os.getpid())



ROADMAP_MARKS = [
    "DNase_fdr001",
    "H3K27ac",
    "H3K4me1",
    "H3K4me3",
    "H3K27me3",
    "H3K9me3"
]


# ============================================================
# ============================================================


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# ============================================================

def convert_gtf_to_sorted_bed(raw_gtf, sorted_bed_out):
    df_gtf = pd.read_csv(raw_gtf, sep="\t", header=None, comment="#")

    df_trans = df_gtf[df_gtf[2] == "transcript"].copy()

    df_bed = pd.DataFrame()
    df_bed[0] = df_trans[0].astype(str)
    df_bed[1] = pd.to_numeric(df_trans[3], errors="raise").astype(int) - 1
    df_bed[2] = pd.to_numeric(df_trans[4], errors="raise").astype(int)
    df_bed[3] = df_trans[8].astype(str)
    df_bed[4] = 0
    df_bed[5] = df_trans[6].astype(str)

    df_bed = df_bed.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
    df_bed.to_csv(sorted_bed_out, sep="\t", header=False, index=False)


# ============================================================
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_RUN_DIR, exist_ok=True)


    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)


    serve_ann_dir = Path(cfg["inputs"]["serve_annotation_dir"])

    catalog = pd.read_parquet(HERV_CATALOG_PARQUET, columns=["herv_internal_id", *LOCUS_KEY_COLUMNS,
                                                             "is_model_universe", "datahub_split"])
    catalog = normalize_locus_columns(catalog)
    all_herv_ids = catalog["herv_internal_id"].tolist()

    df_herv_bed = pd.read_csv(HERV_BED, sep="\t", header=None)
    df_herv_bed[0] = df_herv_bed[0].astype(str)
    df_herv_bed[1] = pd.to_numeric(df_herv_bed[1], errors="raise").astype(int)
    df_herv_bed = df_herv_bed.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")

    sorted_herv_bed = TMP_RUN_DIR / "herv_clean_catalog.sorted.bed"
    df_herv_bed.to_csv(sorted_herv_bed, sep="\t", header=False, index=False)

    # ============================================================
    # ============================================================
    global_noncoding_gz = serve_ann_dir / "HERV_noncoding.bed.gz"


    sorted_noncoding_bed = TMP_RUN_DIR / "herv_noncoding.sorted.bed"
    sort_noncoding_cmd = (
        f"bash -o pipefail -c 'gzip -cd {global_noncoding_gz} "
        f"| LC_ALL=C sort -k1,1 -k2,2n > {sorted_noncoding_bed}'"
    )
    subprocess.run(sort_noncoding_cmd, shell=True, check=True)

    noncoding_out_raw = TMP_RUN_DIR / "noncoding_intersect.wo.tsv"
    cmd_noncoding = [
        "bedtools", "intersect",
        "-a", str(sorted_herv_bed),
        "-b", str(sorted_noncoding_bed),
        "-wo",
        "-sorted",
        "-split"
    ]

    with open(noncoding_out_raw, "w") as f:
        subprocess.run(cmd_noncoding, stdout=f, check=True)

    herv_noncoding_counts = {hid: 0 for hid in all_herv_ids}
    herv_noncoding_max_bp = {hid: 0 for hid in all_herv_ids}
    herv_noncoding_total_bp = {hid: 0 for hid in all_herv_ids}

    if os.path.exists(noncoding_out_raw) and os.path.getsize(noncoding_out_raw) > 0:
        with open(noncoding_out_raw, "r") as f:
            first_line = f.readline().rstrip("\n").split("\t")
        overlap_col_idx = len(first_line) - 1

        df_nc_isct = pd.read_csv(
            noncoding_out_raw, sep="\t", header=None,
            usecols=[3, overlap_col_idx],
            names=["herv_id", "overlap_bp"]
        )
        df_nc_isct["herv_id"] = df_nc_isct["herv_id"].astype(str)
        df_nc_isct["overlap_bp"] = pd.to_numeric(df_nc_isct["overlap_bp"], errors="raise").astype(int)

        pos_nc = df_nc_isct[df_nc_isct["overlap_bp"] > 0].copy()
        if not pos_nc.empty:
            grouped_nc = pos_nc.groupby("herv_id").agg(
                max_bp=("overlap_bp", "max"),
                sum_bp=("overlap_bp", "sum"),
                count_val=("overlap_bp", "size")
            )
            for hid, r in grouped_nc.iterrows():
                herv_noncoding_counts[hid] = int(r["count_val"])
                herv_noncoding_max_bp[hid] = int(r["max_bp"])
                herv_noncoding_total_bp[hid] = int(r["sum_bp"])

    for p in [sorted_noncoding_bed, noncoding_out_raw]:
        if os.path.exists(p):
            os.remove(p)

    # ============================================================
    # ============================================================
    gtf_paths = sorted(list(serve_ann_dir.glob("*_HERV.gtf")))

    manifest_rows = []

    herv_gtex_tissue_counts = {hid: 0 for hid in all_herv_ids}
    herv_gtex_annotation_counts = {hid: 0 for hid in all_herv_ids}
    herv_gtex_max_overlap = {hid: 0 for hid in all_herv_ids}
    herv_gtex_total_overlap = {hid: 0 for hid in all_herv_ids}


    for gtf_file in gtf_paths:
        file_basename = gtf_file.name

        tissue_name = file_basename.replace("_HERV.gtf", "")

        manifest_rows.append({
            "source_type": "GTEx_tissue_annotation",
            "tissue": tissue_name,
            "source_file": str(gtf_file.resolve()),
            "file_basename": file_basename,
            "file_format": "GTF",
            "is_compressed": 0
        })

        sorted_gtex_bed = TMP_RUN_DIR / f"{tissue_name}.sorted.bed"
        convert_gtf_to_sorted_bed(gtf_file, sorted_gtex_bed)

        gtex_out_raw = TMP_RUN_DIR / f"{tissue_name}_intersect.wo.tsv"
        cmd_gtex = [
            "bedtools", "intersect",
            "-a", str(sorted_herv_bed),
            "-b", str(sorted_gtex_bed),
            "-wo",
            "-sorted"
        ]

        with open(gtex_out_raw, "w") as f:
            subprocess.run(cmd_gtex, stdout=f, check=True)

        if not os.path.exists(gtex_out_raw) or os.path.getsize(gtex_out_raw) == 0:
            if os.path.exists(gtex_out_raw):
                os.remove(gtex_out_raw)
            if os.path.exists(sorted_gtex_bed):
                os.remove(sorted_gtex_bed)
            continue

        with open(gtex_out_raw, "r") as f:
            first_line = f.readline().rstrip("\n").split("\t")
        overlap_col_idx = len(first_line) - 1

        df_gt_isct = pd.read_csv(
            gtex_out_raw, sep="\t", header=None,
            usecols=[3, overlap_col_idx],
            names=["herv_id", "overlap_bp"]
        )
        df_gt_isct["herv_id"] = df_gt_isct["herv_id"].astype(str)
        df_gt_isct["overlap_bp"] = pd.to_numeric(df_gt_isct["overlap_bp"], errors="raise").astype(int)

        pos_gt = df_gt_isct[df_gt_isct["overlap_bp"] > 0].copy()
        if not pos_gt.empty:
            grouped_gt = pos_gt.groupby("herv_id").agg(
                max_bp=("overlap_bp", "max"),
                sum_bp=("overlap_bp", "sum"),
                count_val=("overlap_bp", "size")
            )
            for hid, r in grouped_gt.iterrows():
                herv_gtex_tissue_counts[hid] += 1
                herv_gtex_annotation_counts[hid] += int(r["count_val"])
                herv_gtex_total_overlap[hid] += int(r["sum_bp"])
                if r["max_bp"] > herv_gtex_max_overlap[hid]:
                    herv_gtex_max_overlap[hid] = int(r["max_bp"])

        for p in [sorted_gtex_bed, gtex_out_raw]:
            if os.path.exists(p):
                os.remove(p)


    # ============================================================
    # ============================================================
    feature_rows = []
    for hid in all_herv_ids:
        feature_rows.append({
            "herv_internal_id": hid,

            "serve_noncoding_overlap_count": herv_noncoding_counts[hid],
            "serve_noncoding_max_overlap_bp": herv_noncoding_max_bp[hid],
            "serve_noncoding_total_overlap_bp": herv_noncoding_total_bp[hid],
            "is_serve_noncoding_supported": 1 if herv_noncoding_counts[hid] > 0 else 0,

            "serve_gtex_tissue_count": herv_gtex_tissue_counts[hid],
            "serve_gtex_annotation_count": herv_gtex_annotation_counts[hid],
            "serve_gtex_max_overlap_bp": herv_gtex_max_overlap[hid],
            "serve_gtex_total_overlap_bp": herv_gtex_total_overlap[hid],
            "is_serve_gtex_supported": 1 if herv_gtex_tissue_counts[hid] > 0 else 0
        })

    df_features = pd.DataFrame(feature_rows)

    all_ids = catalog[["herv_internal_id", *LOCUS_KEY_COLUMNS,
                       "is_model_universe", "datahub_split"]].copy()

    df_final = pd.merge(all_ids, df_features, on="herv_internal_id", how="left", validate="one_to_one")

    numeric_cols = [c for c in df_final.columns
                    if c not in ["herv_internal_id", *LOCUS_KEY_COLUMNS, "datahub_split"]]
    for c in numeric_cols:
        df_final[c] = pd.to_numeric(df_final[c], errors="raise").fillna(0).astype(int)

    df_model_features = df_final[df_final["is_model_universe"] == 1].copy()


    df_source_manifest = pd.DataFrame(manifest_rows)
    df_source_manifest = pd.concat([
        pd.DataFrame([{
            "source_type": "SERVE_global_noncoding",
            "tissue": "Global_Noncoding",
            "source_file": str(global_noncoding_gz.resolve()),
            "file_basename": global_noncoding_gz.name,
            "file_format": "BED",
            "is_compressed": 1
        }]),
        df_source_manifest
    ], ignore_index=True)

    summary_by_tissue_rows = []
    df_nc_pos = df_final[df_final["is_serve_noncoding_supported"] == 1]
    summary_by_tissue_rows.append({
        "source": "SERVE_global_noncoding",
        "file_count": 1,
        "herv_with_any_overlap": len(df_nc_pos),
        "total_overlap_records": int(df_final["serve_noncoding_overlap_count"].sum()),
        "max_overlap_bp": int(df_final["serve_noncoding_max_overlap_bp"].max())
    })
    df_gt_pos = df_final[df_final["is_serve_gtex_supported"] == 1]
    summary_by_tissue_rows.append({
        "source": "GTEx_tissue_specific_total",
        "file_count": len(gtf_paths),
        "herv_with_any_overlap": len(df_gt_pos),
        "total_overlap_records": int(df_final["serve_gtex_annotation_count"].sum()),
        "max_overlap_bp": int(df_final["serve_gtex_max_overlap_bp"].max())
    })

    df_final = df_final.sort_values(by="herv_internal_id", kind="mergesort").reset_index(drop=True)
    df_model_features = df_model_features.sort_values(by="herv_internal_id", kind="mergesort").reset_index(drop=True)

    full_parquet = OUT_DIR / "herv_serve_gtex_features_full.parquet"
    manifest_csv = OUT_DIR / "serve_gtex_source_manifest.csv"

    df_final.to_parquet(full_parquet, index=False)
    df_source_manifest.to_csv(manifest_csv, index=False)

    if os.path.exists(sorted_herv_bed):
        os.remove(sorted_herv_bed)
    shutil.rmtree(TMP_RUN_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
