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
from liftover_utils import lift_bed_with_fields


# ============================================================
# ============================================================

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")

DATA_SUMMARY_DIR = Path(__file__).resolve().parent
CONFIG_PATH = DATA_SUMMARY_DIR / "output/01_config/preprocessing_config.json"

STEP01_DIR = DATA_SUMMARY_DIR / "output/02_herv_clean_catalog"
HERV_CATALOG_PARQUET = STEP01_DIR / "herv_clean_catalog_full.parquet"
HERV_BED = STEP01_DIR / "herv_clean_catalog.bed"

OUT_DIR = DATA_SUMMARY_DIR / "output/07_ccre_enhancer_features"
TMP_RUN_DIR = OUT_DIR / "tmp" / str(os.getpid())



# ============================================================
# ============================================================


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# ============================================================

def run_sorted_intersect(sorted_herv_bed, raw_bed_or_tsv, input_type="BED"):

    sorted_temp_ref = TMP_RUN_DIR / f"temp_{input_type}_sorted.bed"

    if input_type == "BED":
        df = pd.read_csv(raw_bed_or_tsv, sep="\t", header=None)
        df[0] = df[0].astype(str)
        df[1] = pd.to_numeric(df[1], errors="raise").astype(int)
        df = df.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
        df.to_csv(sorted_temp_ref, sep="\t", header=False, index=False)
    else:
        sorted_temp_ref = raw_bed_or_tsv

    raw_out = TMP_RUN_DIR / f"temp_{input_type}_intersect.wo.tsv"

    cmd = [
        "bedtools", "intersect",
        "-a", str(sorted_herv_bed),
        "-b", str(sorted_temp_ref),
        "-wo",
        "-sorted"
    ]

    with open(raw_out, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    if input_type == "BED" and os.path.exists(sorted_temp_ref):
        os.remove(sorted_temp_ref)

    return raw_out


# ============================================================
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_RUN_DIR, exist_ok=True)


    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)


    catalog = pd.read_parquet(HERV_CATALOG_PARQUET, columns=["herv_internal_id", *LOCUS_KEY_COLUMNS,
                                                             "is_model_universe", "datahub_split"])
    catalog = normalize_locus_columns(catalog)
    all_herv_ids = catalog["herv_internal_id"].tolist()

    ccre_dir = Path(cfg["inputs"]["ccre_root"])
    mpra_ipsc_counts_path = Path(cfg["inputs"]["mpra_ipsc_counts"])
    mpra_npc_counts_path = Path(cfg["inputs"]["mpra_npc_counts"])

    ccre_files = {
        "ca": ccre_dir / "GRCh38-cCREs.CA.bed",
        "els": ccre_dir / "GRCh38-cCREs.ELS.bed",
        "pls": ccre_dir / "GRCh38-cCREs.PLS.bed",
        "ctcf": ccre_dir / "GRCh38-cCREs.CA-CTCF.bed"
    }

    df_herv_bed = pd.read_csv(HERV_BED, sep="\t", header=None)
    df_herv_bed[0] = df_herv_bed[0].astype(str)
    df_herv_bed[1] = pd.to_numeric(df_herv_bed[1], errors="raise").astype(int)
    df_herv_bed = df_herv_bed.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")

    sorted_herv_bed = TMP_RUN_DIR / "herv_clean_catalog.sorted.bed"
    df_herv_bed.to_csv(sorted_herv_bed, sep="\t", header=False, index=False)

    feature_dict = {hid: {} for hid in all_herv_ids}

    # ============================================================
    # ============================================================

    for ccre_type, bed_path in ccre_files.items():

        herv_counts = {hid: 0 for hid in all_herv_ids}
        herv_max_bp = {hid: 0 for hid in all_herv_ids}
        herv_tot_bp = {hid: 0 for hid in all_herv_ids}

        raw_out = run_sorted_intersect(sorted_herv_bed, bed_path, input_type="BED")

        if not os.path.exists(raw_out) or os.path.getsize(raw_out) == 0:
            if os.path.exists(raw_out):
                os.remove(raw_out)
        else:
            with open(raw_out, "r") as f:
                first_line = f.readline().rstrip("\n").split("\t")
            overlap_col_idx = len(first_line) - 1

            df_isct = pd.read_csv(
                raw_out, sep="\t", header=None,
                usecols=[3, overlap_col_idx],
                names=["herv_id", "overlap_bp"]
            )
            df_isct["herv_id"] = df_isct["herv_id"].astype(str)
            df_isct["overlap_bp"] = pd.to_numeric(df_isct["overlap_bp"], errors="raise").astype(int)

            pos_isct = df_isct[df_isct["overlap_bp"] > 0].copy()
            if not pos_isct.empty:
                grouped = pos_isct.groupby("herv_id").agg(
                    max_bp=("overlap_bp", "max"),
                    sum_bp=("overlap_bp", "sum"),
                    count_val=("overlap_bp", "size")
                )
                for hid, r in grouped.iterrows():
                    herv_counts[hid] = int(r["count_val"])
                    herv_max_bp[hid] = int(r["max_bp"])
                    herv_tot_bp[hid] = int(r["sum_bp"])

            if os.path.exists(raw_out):
                os.remove(raw_out)

        col_prefix = f"ccre_{ccre_type}"
        for hid in all_herv_ids:
            feature_dict[hid][f"{col_prefix}_overlap_count"] = herv_counts[hid]
            feature_dict[hid][f"{col_prefix}_max_overlap_bp"] = herv_max_bp[hid]
            feature_dict[hid][f"{col_prefix}_total_overlap_bp"] = herv_tot_bp[hid]
            feature_dict[hid][f"is_{col_prefix}_overlap" if ccre_type != "els" else "is_ccre_enhancer_like"] = 1 if herv_counts[hid] > 0 else 0


    # ============================================================
    # ============================================================

    ipsc_counts = pd.read_csv(mpra_ipsc_counts_path, sep="\t")
    npc_counts = pd.read_csv(mpra_npc_counts_path, sep="\t")
    nonmalignant = pd.concat([ipsc_counts, npc_counts], ignore_index=True)
    nonmalignant = nonmalignant[nonmalignant["name"].astype(str).str.startswith("hg19_")].copy()
    nonmalignant["log2"] = pd.to_numeric(nonmalignant["log2"], errors="raise")
    nonmalignant["n_obs_bc"] = pd.to_numeric(nonmalignant["n_obs_bc"], errors="raise")

    nonmalignant["weighted_log2"] = nonmalignant["log2"] * nonmalignant["n_obs_bc"]
    grouped = nonmalignant.groupby("name").agg(
        n_reps=("log2", "size"),
        bc_sum=("n_obs_bc", "sum"),
        weighted_sum=("weighted_log2", "sum"),
    ).reset_index()
    grouped["activity"] = grouped["weighted_sum"] / grouped["bc_sum"]

    grouped = grouped[grouped["n_reps"] >= 2].copy()

    coord = grouped["name"].str.extract(r"_(chr[^:]+):(\d+)_(\d+)")
    grouped["chrom"] = coord[0]
    grouped["g_start"] = pd.to_numeric(coord[1], errors="raise").astype(int)
    grouped["g_end"] = pd.to_numeric(coord[2], errors="raise").astype(int)

    df_mpra_raw = grouped[["chrom", "g_start", "g_end", "name", "activity"]].copy()
    mpra_source_table = OUT_DIR / "mpra_hg19_nonmalignant_source.tsv"
    df_mpra_raw.to_csv(mpra_source_table, sep="\t", index=False)


    # Convert complete intervals with pyliftover. The source row ID is retained so
    # records rejected by an unmapped/ambiguous chain block cannot shift activity.
    source = df_mpra_raw.copy().reset_index(drop=True)
    source["g_start"] = pd.to_numeric(source["g_start"], errors="raise").astype(int)
    source["g_end"] = pd.to_numeric(source["g_end"], errors="raise").astype(int)
    named_source = TMP_RUN_DIR / "mpra_source_named.hg19.bed"
    named_mapped = OUT_DIR / "mpra_hg19_to_hg38_mapped.bed"
    named_unmapped = OUT_DIR / "mpra_hg19_to_hg38_unmapped.bed"
    named = source[["chrom", "g_start", "g_end"]].copy()
    named["name"] = [f"mpra_{i}" for i in range(len(named))]
    named.to_csv(named_source, sep="\t", header=False, index=False)
    lift_stats = lift_bed_with_fields(
        named_source, Path(cfg["inputs"]["hg19_to_hg38_chain"]), named_mapped,
        output_unmapped=named_unmapped,
    )
    lifted_named = pd.read_csv(named_mapped, sep="\t", header=None,
                               names=["chrom", "start", "end", "name"])
    lifted_named["source_index"] = lifted_named.name.str.slice(start=5).astype(int)
    lifted_named = lifted_named.merge(source[["activity"]], left_on="source_index", right_index=True,
                                      validate="one_to_one")
    hg38_mpra_rows = lifted_named[["chrom", "start", "end", "name"]].assign(score=0, strand="+").values.tolist()
    hg38_mpra_active_rows = lifted_named[lifted_named.activity.astype(float) > 1.0][
        ["chrom", "start", "end", "name"]].assign(score=0, strand="+").values.tolist()


    mpra_all_sorted_bed = TMP_RUN_DIR / "mpra_all_hg38.sorted.bed"
    df_m_all = pd.DataFrame(hg38_mpra_rows, columns=range(6))
    df_m_all = df_m_all.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
    df_m_all.to_csv(mpra_all_sorted_bed, sep="\t", header=False, index=False)

    mpra_act_sorted_bed = TMP_RUN_DIR / "mpra_active_hg38.sorted.bed"
    df_m_act = pd.DataFrame(hg38_mpra_active_rows, columns=range(6))
    df_m_act = df_m_act.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
    df_m_act.to_csv(mpra_act_sorted_bed, sep="\t", header=False, index=False)

    herv_mpra_counts = {hid: 0 for hid in all_herv_ids}
    herv_mpra_max_bp = {hid: 0 for hid in all_herv_ids}
    herv_mpra_total_bp = {hid: 0 for hid in all_herv_ids}

    raw_out_m_all = run_sorted_intersect(sorted_herv_bed, mpra_all_sorted_bed, input_type="MPRA_ALL")
    if os.path.exists(raw_out_m_all) and os.path.getsize(raw_out_m_all) > 0:
        with open(raw_out_m_all, "r") as f:
            first_line = f.readline().rstrip("\n").split("\t")
        overlap_col_idx = len(first_line) - 1

        df_m_isct = pd.read_csv(raw_out_m_all, sep="\t", header=None, usecols=[3, overlap_col_idx], names=["herv_id", "overlap_bp"])
        df_m_isct["herv_id"] = df_m_isct["herv_id"].astype(str)
        df_m_isct["overlap_bp"] = pd.to_numeric(df_m_isct["overlap_bp"], errors="raise").astype(int)

        pos_m = df_m_isct[df_m_isct["overlap_bp"] > 0].copy()
        if not pos_m.empty:
            grouped_m = pos_m.groupby("herv_id").agg(max_bp=("overlap_bp", "max"), sum_bp=("overlap_bp", "sum"), count_val=("overlap_bp", "size"))
            for hid, r in grouped_m.iterrows():
                herv_mpra_counts[hid] = int(r["count_val"])
                herv_mpra_max_bp[hid] = int(r["max_bp"])
                herv_mpra_total_bp[hid] = int(r["sum_bp"])

    herv_mpra_act_counts = {hid: 0 for hid in all_herv_ids}
    herv_mpra_act_max_bp = {hid: 0 for hid in all_herv_ids}
    herv_mpra_act_total_bp = {hid: 0 for hid in all_herv_ids}

    raw_out_m_act = run_sorted_intersect(sorted_herv_bed, mpra_act_sorted_bed, input_type="MPRA_ACT")
    if os.path.exists(raw_out_m_act) and os.path.getsize(raw_out_m_act) > 0:
        with open(raw_out_m_act, "r") as f:
            first_line = f.readline().rstrip("\n").split("\t")
        overlap_col_idx = len(first_line) - 1

        df_ma_isct = pd.read_csv(raw_out_m_act, sep="\t", header=None, usecols=[3, overlap_col_idx], names=["herv_id", "overlap_bp"])
        df_ma_isct["herv_id"] = df_ma_isct["herv_id"].astype(str)
        df_ma_isct["overlap_bp"] = pd.to_numeric(df_ma_isct["overlap_bp"], errors="raise").astype(int)

        pos_ma = df_ma_isct[df_ma_isct["overlap_bp"] > 0].copy()
        if not pos_ma.empty:
            grouped_ma = pos_ma.groupby("herv_id").agg(max_bp=("overlap_bp", "max"), sum_bp=("overlap_bp", "sum"), count_val=("overlap_bp", "size"))
            for hid, r in grouped_ma.iterrows():
                herv_mpra_act_counts[hid] = int(r["count_val"])
                herv_mpra_act_max_bp[hid] = int(r["max_bp"])
                herv_mpra_act_total_bp[hid] = int(r["sum_bp"])

    for p in [mpra_all_sorted_bed, mpra_act_sorted_bed, raw_out_m_all, raw_out_m_act]:
        if os.path.exists(p):
            os.remove(p)

    for hid in all_herv_ids:
        feature_dict[hid]["mpra_overlap_count"] = herv_mpra_counts[hid]
        feature_dict[hid]["mpra_max_overlap_bp"] = herv_mpra_max_bp[hid]
        feature_dict[hid]["mpra_total_overlap_bp"] = herv_mpra_total_bp[hid]
        feature_dict[hid]["is_mpra_overlap"] = 1 if herv_mpra_counts[hid] > 0 else 0

        feature_dict[hid]["mpra_active_overlap_count"] = herv_mpra_act_counts[hid]
        feature_dict[hid]["mpra_active_max_overlap_bp"] = herv_mpra_act_max_bp[hid]
        feature_dict[hid]["mpra_active_total_overlap_bp"] = herv_mpra_act_total_bp[hid]
        feature_dict[hid]["is_mpra_active_supported"] = 1 if herv_mpra_act_counts[hid] > 0 else 0


    # ============================================================
    # ============================================================
    df_features = pd.DataFrame.from_dict(feature_dict, orient="index").reset_index().rename(columns={"index": "herv_internal_id"})

    all_ids = catalog[["herv_internal_id", *LOCUS_KEY_COLUMNS,
                       "is_model_universe", "datahub_split"]].copy()

    df_final = pd.merge(all_ids, df_features, on="herv_internal_id", how="left", validate="one_to_one")

    numeric_cols = [c for c in df_final.columns
                    if c not in ["herv_internal_id", *LOCUS_KEY_COLUMNS, "datahub_split"]]
    for c in numeric_cols:
        df_final[c] = pd.to_numeric(df_final[c], errors="raise").fillna(0).astype(int)

    df_model_features = df_final[df_final["is_model_universe"] == 1].copy()

    # ============================================================
    # ============================================================

    manifest_rows = []
    for name, p in ccre_files.items():
        manifest_rows.append({
            "source_type": "cCRE_annotation",
            "source_name": f"GRCh38-cCREs.{name.upper()}",
            "source_file": str(p.resolve()),
            "file_format": "BED",
            "assembly": "GRCh38"
        })
    manifest_rows.append({
        "source_type": "MPRA_experimental",
        "source_name": "GSE245662_hg19_iPSC_NPC_nonmalignant",
        "source_file": str(mpra_source_table.resolve()),
        "file_format": "TSV",
        "assembly": "GRCh37_Lifted_to_GRCh38"
    })
    df_source_manifest = pd.DataFrame(manifest_rows)

    df_final = df_final.sort_values(by="herv_internal_id", kind="mergesort").reset_index(drop=True)
    df_model_features = df_model_features.sort_values(by="herv_internal_id", kind="mergesort").reset_index(drop=True)

    full_parquet = OUT_DIR / "herv_ccre_enhancer_features_full.parquet"
    manifest_csv = OUT_DIR / "ccre_enhancer_source_manifest.csv"
    liftover_audit_json = OUT_DIR / "mpra_hg19_to_hg38_liftover_audit.json"

    df_final.to_parquet(full_parquet, index=False)
    df_source_manifest.to_csv(manifest_csv, index=False)
    liftover_audit = {
        "engine": "pyliftover strict same-chain/same-block intervals",
        "source_assembly": "GRCh37/hg19", "target_assembly": "GRCh38/hg38",
        "source_file": str(mpra_source_table.resolve()),
        "chain_file": str(Path(cfg["inputs"]["hg19_to_hg38_chain"]).resolve()),
        "mapped_file": str(named_mapped.resolve()), "unmapped_file": str(named_unmapped.resolve()),
        **lift_stats,
    }
    save_json(liftover_audit, liftover_audit_json)

    if os.path.exists(sorted_herv_bed):
        os.remove(sorted_herv_bed)
    shutil.rmtree(TMP_RUN_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
