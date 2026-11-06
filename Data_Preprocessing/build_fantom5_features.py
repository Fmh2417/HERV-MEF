import json
import gzip
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from locus_contract import (
    LOCUS_KEY_COLUMNS,
    normalize_locus_columns,
)


DATA_SUMMARY_DIR = Path(__file__).resolve().parent

CONFIG_PATH = DATA_SUMMARY_DIR / "output/01_config/preprocessing_config.json"

STEP01_DIR = DATA_SUMMARY_DIR / "output/02_herv_clean_catalog"
HERV_CATALOG_CSV = STEP01_DIR / "herv_clean_catalog_full.csv"
HERV_BED = STEP01_DIR / "herv_clean_catalog.bed"

OUT_DIR = DATA_SUMMARY_DIR / "output/03_fantom5_rebuild"
TMP_DIR = OUT_DIR / "tmp"


HERV_BED_ID_COL_0BASED = 3
FANTOM5_BED_ID_COL_0BASED = 3
SDRF_GROUPS = [
    "human.cell_line.LQhCAGE", "human.cell_line.hCAGE",
    "human.fractionation.hCAGE",
    "human.primary_cell.LQhCAGE", "human.primary_cell.hCAGE",
    "human.timecourse.LQhCAGE", "human.timecourse.hCAGE",
    "human.tissue.hCAGE", "unclassified.hCAGE",
]
TUMOR_RE = re.compile(
    r"tumou?r|cancer|carcinoma|adenocarcinoma|leuk(?:a|e)mia|lymphoma|melanoma|"
    r"sarcoma|malignan|blastoma|myeloma|teratoma|mesothelioma|metast|glioma|"
    r"glioblastoma|myelodysplastic|mycosis[ _-]?fungoides|hodgkin|burkitt|"
    r"hepatoma|neuroblastoma|retinoblastoma|medulloblastoma|choriocarcinoma|"
    r"teratocarcinoma|neoplastic|astrocytoma|carcinoid|histiocytoma|"
    r"keratoacanthoma|leiomyoma|meningioma|neuroepithelioma|neurofibroma|"
    r"osteoclastoma|schwannoma|somatostatinoma|hep-?2", re.I
)
CNHS_RE = re.compile(r"CNhs\d+")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_official_sdrf(sdrf_dir, group):
    candidates = sorted(Path(sdrf_dir).glob(f"00_{group}*.assay_sdrf.txt"))
    return candidates[0]


def build_non_tumor_sample_manifest(matrix_samples, sdrf_dir):
    records = {}
    source_rows = []
    for group in SDRF_GROUPS:
        path = find_official_sdrf(sdrf_dir, group)
        frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        for row in frame.itertuples(index=False, name=None):
            metadata_text = " | ".join(str(x) for x in row)
            for sample_id in sorted(set(CNHS_RE.findall(metadata_text))):
                if sample_id in records and records[sample_id]["metadata_text"] != metadata_text:
                    records[sample_id]["metadata_text"] += " || " + metadata_text
                else:
                    records.setdefault(sample_id, {"assay_group": group,
                                                   "metadata_text": metadata_text})
        source_rows.append({"assay_group": group, "source_file": str(path.resolve()),
                            "rows": len(frame)})

    rows = []
    for sample_id in matrix_samples:
        info = records.get(sample_id)
        if info is None:
            rows.append({"sample_id": sample_id, "mapping_status": "unmapped",
                         "assay_group": "", "metadata_text": "",
                         "is_tumor": 0, "include_non_tumor_activity": 0,
                         "decision_reason": "excluded_unmapped_fail_closed"})
            continue
        tumor = bool(TUMOR_RE.search(info["metadata_text"]))
        include = not tumor
        reason = "excluded_explicit_tumor" if tumor else "included_non_tumor"
        rows.append({"sample_id": sample_id, "mapping_status": "mapped",
                     "assay_group": info["assay_group"],
                     "metadata_text": info["metadata_text"],
                     "is_tumor": int(tumor),
                     "include_non_tumor_activity": int(include),
                     "decision_reason": reason})
    manifest = pd.DataFrame(rows)
    if set(manifest.mapping_status) != {"mapped"}:
        missing = manifest.loc[manifest.mapping_status != "mapped", "sample_id"].tolist()
        raise RuntimeError(f"Unmapped FANTOM5 CNhs samples (fail closed): {missing[:20]}")
    sources_path = OUT_DIR / "fantom5_official_sdrf_sources.tsv"
    pd.DataFrame(source_rows).to_csv(sources_path, sep="\t", index=False)
    return manifest, sources_path


def open_text(path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_first_fields(path):
    with open_text(path) as f:
        line = f.readline().rstrip("\n")
    return line.split("\t")


def count_cols_first_data_line(path):
    with open_text(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                return len(line.rstrip("\n").split("\t"))
    raise RuntimeError(f"No data line found: {path}")


def normalize_fantom5_id_series(s):
    return s.astype(str).str.strip().str.replace("..", "-", regex=False)


def run_bedtools_intersect(herv_bed, f5_bed):


    df_herv = pd.read_csv(herv_bed, sep="\t", header=None)
    df_herv[0] = df_herv[0].astype(str)
    df_herv[1] = pd.to_numeric(df_herv[1], errors="raise").astype(int)
    df_herv = df_herv.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
    sorted_herv_bed = TMP_DIR / "herv_clean_catalog.sorted.bed"
    df_herv.to_csv(sorted_herv_bed, sep="\t", header=False, index=False)

    df_f5 = pd.read_csv(f5_bed, sep="\t", header=None)
    df_f5[0] = df_f5[0].astype(str)
    df_f5[1] = pd.to_numeric(df_f5[1], errors="raise").astype(int)
    df_f5 = df_f5.sort_values(by=[0, 1], ascending=[True, True], kind="mergesort")
    sorted_f5_bed = TMP_DIR / "F5.hg38.enhancers.sorted.bed"
    df_f5.to_csv(sorted_f5_bed, sep="\t", header=False, index=False)


    raw_out = TMP_DIR / "herv_x_fantom5.wao.tsv"

    cmd = [
        "bedtools", "intersect",
        "-a", str(sorted_herv_bed),
        "-b", str(sorted_f5_bed),
        "-wao",
        "-sorted"
    ]

    with open(raw_out, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    return raw_out


def parse_intersect(raw_out, herv_bed, f5_bed):
    n_a = count_cols_first_data_line(herv_bed)
    n_b = count_cols_first_data_line(f5_bed)

    herv_id_col = HERV_BED_ID_COL_0BASED
    f5_id_col = n_a + FANTOM5_BED_ID_COL_0BASED
    overlap_col = n_a + n_b


    df = pd.read_csv(
        raw_out,
        sep="\t",
        header=None,
        usecols=[herv_id_col, f5_id_col, overlap_col],
        names=["herv_internal_id", "fantom5_enhancer_id", "overlap_bp"],
    )

    df["herv_internal_id"] = df["herv_internal_id"].astype(str)
    df["fantom5_enhancer_id"] = normalize_fantom5_id_series(df["fantom5_enhancer_id"])
    df["overlap_bp"] = pd.to_numeric(df["overlap_bp"], errors="coerce").fillna(0).astype(int)

    pos = df[
        (df["overlap_bp"] > 0)
        & (df["fantom5_enhancer_id"] != ".")
        & (df["fantom5_enhancer_id"] != "-1")
    ].copy()

    return df, pos


def build_overlap_features(all_overlap, pos_overlap):
    out = all_overlap.groupby("herv_internal_id", as_index=False).agg(
        f5_max_overlap_bp=("overlap_bp", "max")
    )

    if len(pos_overlap) > 0:
        x = pos_overlap.groupby("herv_internal_id", as_index=False).agg(
            f5_overlap_count=("fantom5_enhancer_id", "size"),
            f5_enhancer_count=("fantom5_enhancer_id", "nunique"),
        )
        out = out.merge(x, on="herv_internal_id", how="left")
    else:
        out["f5_overlap_count"] = 0
        out["f5_enhancer_count"] = 0

    for c in ["f5_overlap_count", "f5_enhancer_count"]:
        out[c] = out[c].fillna(0).astype(int)

    out["is_fantom5_enhancer"] = (out["f5_max_overlap_bp"] > 0).astype(int)
    out["is_fantom5_enhancer_10bp"] = (out["f5_max_overlap_bp"] >= 10).astype(int)
    out["is_fantom5_enhancer_50bp"] = (out["f5_max_overlap_bp"] >= 50).astype(int)
    out["is_fantom5_enhancer_100bp"] = (out["f5_max_overlap_bp"] >= 100).astype(int)

    return out


def get_usage_sample_cols(usage_matrix):
    return read_first_fields(usage_matrix)


def get_tpm_sample_cols(tpm_matrix):
    return read_first_fields(tpm_matrix)[1:]


def load_usage_subset(usage_matrix, enhancer_ids, chunksize=2000):
    sample_cols = get_usage_sample_cols(usage_matrix)
    names = ["fantom5_enhancer_id"] + sample_cols

    enhancer_ids = set(str(x) for x in enhancer_ids)
    chunks = []


    for chunk in pd.read_csv(
        usage_matrix,
        sep="\t",
        header=None,
        names=names,
        skiprows=1,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["fantom5_enhancer_id"] = normalize_fantom5_id_series(chunk["fantom5_enhancer_id"])
        sub = chunk[chunk["fantom5_enhancer_id"].isin(enhancer_ids)].copy()
        if len(sub) > 0:
            chunks.append(sub)

    if not chunks:
        mat = pd.DataFrame(columns=names).set_index("fantom5_enhancer_id")
    else:
        mat = pd.concat(chunks, ignore_index=True)
        mat = mat.drop_duplicates("fantom5_enhancer_id", keep="first")
        mat = mat.set_index("fantom5_enhancer_id")

    for c in sample_cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0)

    return mat, sample_cols


def load_tpm_subset(tpm_matrix, enhancer_ids, chunksize=2000):
    sample_cols = get_tpm_sample_cols(tpm_matrix)
    names = ["fantom5_enhancer_id"] + sample_cols

    enhancer_ids = set(str(x) for x in enhancer_ids)
    chunks = []


    for chunk in pd.read_csv(
        tpm_matrix,
        sep="\t",
        header=None,
        names=names,
        skiprows=1,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["fantom5_enhancer_id"] = normalize_fantom5_id_series(chunk["fantom5_enhancer_id"])
        sub = chunk[chunk["fantom5_enhancer_id"].isin(enhancer_ids)].copy()
        if len(sub) > 0:
            chunks.append(sub)

    if not chunks:
        mat = pd.DataFrame(columns=names).set_index("fantom5_enhancer_id")
    else:
        mat = pd.concat(chunks, ignore_index=True)
        mat = mat.drop_duplicates("fantom5_enhancer_id", keep="first")
        mat = mat.set_index("fantom5_enhancer_id")

    for c in sample_cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0)

    return mat, sample_cols


def compute_cnhs_activity_features(pos_overlap, overlap_features, usage_mat, tpm_mat, common_samples):
    mapping = (
        pos_overlap[["herv_internal_id", "fantom5_enhancer_id"]]
        .drop_duplicates()
        .groupby("herv_internal_id")["fantom5_enhancer_id"]
        .apply(list)
        .to_dict()
    )

    usage_arr = usage_mat[common_samples].to_numpy(dtype=np.float32)
    tpm_arr = tpm_mat[common_samples].to_numpy(dtype=np.float32)

    usage_id_to_rowidx = {
        str(enhancer_id): idx
        for idx, enhancer_id in enumerate(usage_mat.index.astype(str))
    }

    tpm_id_to_rowidx = {
        str(enhancer_id): idx
        for idx, enhancer_id in enumerate(tpm_mat.index.astype(str))
    }

    rows = []


    for i, (hid, enhancers) in enumerate(mapping.items(), start=1):

        enhancers = [str(x) for x in enhancers]

        row = {"herv_internal_id": hid}

        u_row_indices = [
            usage_id_to_rowidx[e]
            for e in enhancers
            if e in usage_id_to_rowidx
        ]

        row["f5_usage_matched_enhancer_count"] = len(u_row_indices)

        if u_row_indices:
            u = usage_arr[u_row_indices, :]
            active_cnhs = (u > 0).any(axis=0)
            row["f5_active_cnhs_count"] = int(active_cnhs.sum())
        else:
            row["f5_active_cnhs_count"] = 0

        t_row_indices = [
            tpm_id_to_rowidx[e]
            for e in enhancers
            if e in tpm_id_to_rowidx
        ]

        row["f5_tpm_matched_enhancer_count"] = len(t_row_indices)

        if t_row_indices:
            t = tpm_arr[t_row_indices, :]
            positive = t[t > 0]

            row["f5_max_tpm"] = float(np.max(t)) if t.size else 0.0
            row["f5_mean_tpm_all"] = float(np.mean(t)) if t.size else 0.0
            row["f5_mean_tpm_positive"] = float(np.mean(positive)) if positive.size else 0.0
            row["f5_tpm_positive_cnhs_count"] = int((t > 0).any(axis=0).sum())
        else:
            row["f5_max_tpm"] = 0.0
            row["f5_mean_tpm_all"] = 0.0
            row["f5_mean_tpm_positive"] = 0.0
            row["f5_tpm_positive_cnhs_count"] = 0

        rows.append(row)

    activity = pd.DataFrame(rows)
    out = overlap_features.merge(activity, on="herv_internal_id", how="left")

    fill_cols = [
        "f5_usage_matched_enhancer_count",
        "f5_active_cnhs_count",
        "f5_tpm_matched_enhancer_count",
        "f5_max_tpm",
        "f5_mean_tpm_all",
        "f5_mean_tpm_positive",
        "f5_tpm_positive_cnhs_count",
    ]

    for c in fill_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


    cfg = load_json(CONFIG_PATH)
    f5_cfg = cfg["inputs"]["fantom5"]

    f5_bed = Path(f5_cfg["enhancer_bed"])
    usage_matrix = Path(f5_cfg["usage_matrix"])
    tpm_matrix = Path(f5_cfg["tpm_matrix"])
    sdrf_dir = Path(cfg["inputs"]["fantom5_official_sdrf_dir"])


    catalog = pd.read_csv(HERV_CATALOG_CSV, low_memory=False)
    catalog = normalize_locus_columns(catalog)

    raw_intersect = run_bedtools_intersect(HERV_BED, f5_bed)

    all_overlap, pos_overlap = parse_intersect(raw_intersect, HERV_BED, f5_bed)


    enhancer_ids = sorted(pos_overlap["fantom5_enhancer_id"].astype(str).unique())


    usage_mat, usage_samples = load_usage_subset(usage_matrix, enhancer_ids)
    tpm_mat, tpm_samples = load_tpm_subset(tpm_matrix, enhancer_ids)

    sample_filter, sdrf_sources_path = build_non_tumor_sample_manifest(usage_samples, sdrf_dir)
    included_ids = set(sample_filter.loc[
        sample_filter.include_non_tumor_activity.astype(int) == 1, "sample_id"
    ])
    common_samples = [s for s in usage_samples if s in set(tpm_samples) and s in included_ids]

    # Remove enhancer atlas elements supported only by tumor-derived CNhs samples.
    non_tumor_active_mask = ((usage_mat[common_samples] > 0).any(axis=1) |
                             (tpm_mat[common_samples] > 0).any(axis=1))
    non_tumor_active_enhancers = set(usage_mat.index[non_tumor_active_mask].astype(str))
    non_tumor_pos_overlap = pos_overlap[
        pos_overlap.fantom5_enhancer_id.astype(str).isin(non_tumor_active_enhancers)
    ].copy()
    non_tumor_all_overlap = all_overlap.copy()
    unsupported = ~non_tumor_all_overlap.fantom5_enhancer_id.astype(str).isin(
        non_tumor_active_enhancers
    )
    non_tumor_all_overlap.loc[unsupported, "overlap_bp"] = 0
    overlap_features = build_overlap_features(non_tumor_all_overlap, non_tumor_pos_overlap)


    features = compute_cnhs_activity_features(
        pos_overlap=non_tumor_pos_overlap,
        overlap_features=overlap_features,
        usage_mat=usage_mat,
        tpm_mat=tpm_mat,
        common_samples=common_samples,
    )

    all_ids = catalog[["herv_internal_id", *LOCUS_KEY_COLUMNS,
                       "is_model_universe", "datahub_split"]].copy()
    features = all_ids.merge(features, on="herv_internal_id", how="left", validate="one_to_one")

    numeric_cols = [
        c for c in features.columns
        if c not in ["herv_internal_id", *LOCUS_KEY_COLUMNS, "datahub_split"]
    ]

    for c in numeric_cols:
        features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0)


    sample_manifest = sample_filter.copy()
    sample_manifest["used_for_activity_aggregation"] = sample_manifest.sample_id.isin(common_samples).astype(int)

    sample_manifest_path = OUT_DIR / "fantom5_cnhs_sample_manifest.tsv"
    sample_manifest.to_csv(sample_manifest_path, sep="\t", index=False)

    full_parquet = OUT_DIR / "herv_fantom5_features_full.parquet"

    features.to_parquet(full_parquet, index=False)


if __name__ == "__main__":
    main()
