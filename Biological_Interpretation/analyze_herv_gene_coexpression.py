#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import gzip
import json
import re
import numpy as np
import pandas as pd
import scipy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
# step 20: Analyze HERV-gene coexpression in LUAD
# --------------------------------------------------------------------------



PROJECT = Path(__file__).resolve().parent.parent
DATASET = config.DATASET


def analysis_config() -> SimpleNamespace:
    biology = PROJECT / "Biological_Interpretation"
    module18 = biology / "output/18_luad_transcriptional_realization"
    tcga = biology / "TCGA_lung"
    telescope = DATASET / "Telescope/LUAD/merged"
    return SimpleNamespace(
        project_root=PROJECT,
        tcga_dir=tcga,
        rna_dir=tcga / "RNAseq",
        metadata=one_file(tcga, "metadata.cart*.json"),
        manifest=one_file(tcga, "gdc_manifest*.txt"),
        loci=module18 / "18_luad_locus_transcriptional_realization.tsv.gz",
        size_factors=module18 / "18_sample_size_factors.tsv",
        counts=tsv_path(telescope, "LUAD_HERV_locus_counts_merged.tsv"),
        samples=tsv_path(telescope, "LUAD_sample_metadata.tsv"),
        out_dir=biology / "output/20_luad_herv_gene_coexpression",
        gtf=DATASET / "reference/gencode.v38.annotation.gtf.gz",
        window_bp=100_000,
        min_prevalence=0.10,
        herv_min_count=1.0,
        gene_min_tpm=1.0,
        min_samples=30,
        rho_cutoff=0.30,
        fdr_cutoff=0.05,
        workers=4,
    )


def one_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    return matches[0]


def tsv_path(directory, name):
    path = directory / name
    return path if path.exists() else directory / (name + ".gz")


def boolean(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "t"])


def stable_gene_id(value):
    # Preserve the distinct PAR_Y suffix while removing the Ensembl version.
    return re.sub(r"\.\d+(?=_PAR_Y$|$)", "", value)


def write_table(frame, directory, name):
    frame.to_csv(directory / ("20_" + name), sep="\t", index=False, na_rep="NA")


def match_primary_samples(args, samples):
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    file_rows = []
    for entry in metadata:
        aliquots = [entity["entity_submitter_id"] for entity in entry["associated_entities"]
                    if entity["entity_type"] == "aliquot"]
        file_rows.append(dict(sample_id=aliquots[0], file_id=entry["file_id"],
                              file_name=entry["file_name"]))
    files = pd.DataFrame(file_rows)
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"id": str})
    files = files.merge(manifest[["id", "filename", "size"]],
                        left_on="file_id", right_on="id", validate="one_to_one")

    paths = list(args.rna_dir.rglob("*.rna_seq.augmented_star_gene_counts.tsv"))
    by_name = {path.name: path for path in paths}
    primary = samples.loc[boolean(samples.is_primary_tumor),
                          ["sample_id", "patient_id", "sample_type_code"]].copy()
    matched = primary.merge(files, on="sample_id", how="left", validate="one_to_one")
    matched["file_path"] = matched.file_name.map(by_name)
    matched["match_status"] = np.where(matched.file_id.isna(), "no_exact_aliquot",
                                      np.where(matched.file_path.isna(), "file_missing", "matched"))
    write_table(matched, args.out_dir, "sample_matching.tsv")
    available = matched.loc[matched.match_status.eq("matched")].copy()
    return available.reset_index(drop=True), len(primary), len(paths)


def read_gtf(path):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            attr = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
            if attr.get("gene_type") != "protein_coding":
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            rows.append(dict(gene_id=stable_gene_id(attr["gene_id"]),
                             gtf_gene_id=attr["gene_id"], gene_name=attr["gene_name"],
                             gene_chrom=fields[0], gene_start_0based=start,
                             gene_end_halfopen=end, gene_strand=fields[6],
                             gene_tss_0based=start if fields[6] == "+" else end - 1))
    genes = pd.DataFrame(rows)
    return genes


def nearby_pairs(loci, genes, window):
    groups = {chrom: frame for chrom, frame in genes.groupby("gene_chrom")}
    frames = []
    for locus in loci.itertuples(index=False):
        if locus.mef_chrom not in groups:
            continue
        chrom_genes = groups[locus.mef_chrom]
        start, end = int(locus.mef_start_0based), int(locus.mef_end_halfopen)
        selected = chrom_genes.loc[chrom_genes.gene_tss_0based.between(
            start - window, end - 1 + window)].copy()
        if selected.empty:
            continue
        tss = selected.gene_tss_0based.to_numpy()
        selected["signed_tss_distance_bp"] = np.where(
            tss < start, tss - start, np.where(tss >= end, tss - (end - 1), 0))
        selected["tss_distance_bp"] = selected.signed_tss_distance_bp.abs()
        selected["mef_gene_body_overlap_bp"] = np.maximum(
            0, np.minimum(end, selected.gene_end_halfopen)
            - np.maximum(start, selected.gene_start_0based))
        selected["telescope_gene_body_overlap_bp"] = np.maximum(
            0, np.minimum(int(locus.telescope_end_inclusive), selected.gene_end_halfopen)
            - np.maximum(int(locus.telescope_start_1based) - 1, selected.gene_start_0based))
        selected["gene_body_overlap"] = (
            (selected.mef_gene_body_overlap_bp > 0) | (selected.telescope_gene_body_overlap_bp > 0))
        selected["same_strand_as_mef"] = selected.gene_strand.eq(locus.mef_strand)
        selected["herv_internal_id"] = locus.herv_internal_id
        frames.append(selected)
    columns = list(genes.columns) + ["signed_tss_distance_bp", "tss_distance_bp",
               "mef_gene_body_overlap_bp", "telescope_gene_body_overlap_bp", "gene_body_overlap",
               "same_strand_as_mef", "herv_internal_id"]
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return pairs.merge(loci, on="herv_internal_id", validate="many_to_one")


def read_gene_expression(cohort, gene_ids, workers):
    wanted = set(gene_ids)

    def read_one(row):
        with open(row.file_path, encoding="utf-8") as handle:
            gene_model = handle.readline().strip()
        data = pd.read_csv(row.file_path, sep="\t", comment="#",
                           usecols=["gene_id", "tpm_unstranded"])
        data["gene_id"] = data.gene_id.map(stable_gene_id)
        data = data.loc[data.gene_id.isin(wanted)]
        series = data.set_index("gene_id").tpm_unstranded.rename(row.sample_id)
        return series, gene_model

    records, models = [], set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (series, gene_model) in enumerate(
                executor.map(read_one, cohort.itertuples(index=False)), 1):
            records.append(series)
            models.add(gene_model)
    matrix = pd.concat(records, axis=1).reindex(index=gene_ids,
                                              columns=cohort.sample_id.tolist())
    return matrix, sorted(models)


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan)
    positions = np.flatnonzero(np.isfinite(values))
    order = np.argsort(values[positions], kind="stable")
    ranked = values[positions][order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result[positions[order]] = np.minimum(adjusted, 1.0)
    return result


def correlate(pairs, raw_herv, log_herv, gene_tpm, args):
    rows = []
    for pair in pairs.itertuples(index=False):
        raw = raw_herv.loc[pair.telescope_id].to_numpy(float)
        x = log_herv.loc[pair.telescope_id].to_numpy(float)
        tpm = gene_tpm.loc[pair.gene_id].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(tpm)
        n = int(mask.sum())
        hprev = float((raw[mask] >= args.herv_min_count).mean()) if n else np.nan
        gprev = float((tpm[mask] >= args.gene_min_tpm).mean()) if n else np.nan
        rho, p_value, status = np.nan, np.nan, "tested"
        if n < args.min_samples:
            status = "gene_expression_missing_or_insufficient_samples"
        elif hprev < args.min_prevalence:
            status = "low_herv_prevalence"
        elif gprev < args.min_prevalence:
            status = "low_gene_prevalence"
        elif np.ptp(x[mask]) == 0 or np.ptp(tpm[mask]) == 0:
            status = "constant_expression"
        else:
            # Log2(TPM+1) preserves ranks; include zero-expression patients.
            rho, p_value = spearmanr(x[mask], np.log2(tpm[mask] + 1))
        rows.append(dict(n_patients=n, herv_detection_prevalence=hprev,
                         gene_tpm_prevalence=gprev, test_status=status,
                         spearman_rho=rho, p_value=p_value))
    result_columns = ["n_patients", "herv_detection_prevalence", "gene_tpm_prevalence",
                      "test_status", "spearman_rho", "p_value"]
    results = pd.concat([pairs.reset_index(drop=True),
                         pd.DataFrame(rows, columns=result_columns)], axis=1)
    results["BH_adjusted_p"] = bh_adjust(results.p_value)
    results["bh_significant"] = results.BH_adjusted_p.lt(args.fdr_cutoff)
    results["positive_link"] = results.bh_significant & results.spearman_rho.ge(args.rho_cutoff)
    results["negative_link"] = results.bh_significant & results.spearman_rho.le(-args.rho_cutoff)
    return results.sort_values(["BH_adjusted_p", "p_value", "herv_internal_id", "gene_id"],
                               na_position="last").reset_index(drop=True)


def step_20():
    args = analysis_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.samples, sep="\t", dtype={"sample_type_code": str})
    cohort, n_primary, n_files = match_primary_samples(args, samples)
    sample_ids = cohort.sample_id.tolist()

    all_loci = pd.read_csv(args.loci, sep="\t")
    loci = all_loci.loc[boolean(all_loci.frozen_top1)].copy()
    for column in ["frozen_top1", "a549_active_support", "de_tested", "tumor_upregulated"]:
        loci[column] = boolean(loci[column])
    counts = pd.read_csv(args.counts, sep="\t", usecols=["transcript"] + sample_ids)
    counts = counts.set_index("transcript")
    raw_herv = counts.loc[loci.telescope_id, sample_ids].astype(float)
    factors = pd.read_csv(args.size_factors, sep="\t").set_index("sample_id")
    factors = factors.loc[sample_ids, "sparse_median_ratio_size_factor"].astype(float)
    log_herv = np.log2(raw_herv.div(factors, axis="columns") + 1)
    cohort["herv_size_factor"] = factors.to_numpy()
    write_table(cohort, args.out_dir, "matched_primary_samples.tsv")
    genes = read_gtf(args.gtf)
    pairs = nearby_pairs(loci, genes, args.window_bp)
    gene_ids = sorted(pairs.gene_id.unique())
    gene_tpm, gene_models = read_gene_expression(cohort, gene_ids, args.workers)
    results = correlate(pairs, raw_herv, log_herv, gene_tpm, args)
    write_table(results, args.out_dir, "nearby_gene_all_pairs.tsv")
    links = results.loc[results.positive_link | results.negative_link].copy()
    write_table(links, args.out_dir, "coexpression_links.tsv")

    counts_by_locus = results.assign(tested=results.test_status.eq("tested")).groupby(
        "herv_internal_id").agg(n_nearby_genes=("gene_id", "size"),
                                n_tested_genes=("tested", "sum"),
                                n_bh_significant=("bh_significant", "sum"),
                                n_positive_links=("positive_link", "sum"),
                                n_negative_links=("negative_link", "sum"))
    locus_summary = loci.merge(counts_by_locus, on="herv_internal_id", how="left",
                               validate="one_to_one")
    count_columns = list(counts_by_locus.columns)
    locus_summary[count_columns] = locus_summary[count_columns].fillna(0).astype(int)
    prevalence = (raw_herv >= args.herv_min_count).mean(axis=1)
    locus_summary["matched_tumour_prevalence"] = locus_summary.telescope_id.map(prevalence)
    write_table(locus_summary, args.out_dir, "locus_summary.tsv")
    write_table(log_herv.rename_axis("telescope_id").reset_index(), args.out_dir,
                "herv_log2_normalized_count1.tsv.gz")
    write_table(gene_tpm.rename_axis("gene_id").reset_index(), args.out_dir,
                "nearby_gene_tpm.tsv.gz")

    params = {name: getattr(args, name) for name in ["window_bp", "min_prevalence",
              "herv_min_count", "gene_min_tpm", "min_samples", "rho_cutoff", "fdr_cutoff"]}
    inputs = {name: str(getattr(args, name)) for name in ["project_root", "tcga_dir", "rna_dir",
              "metadata", "manifest", "loci", "counts", "samples", "size_factors", "gtf"]}
    tested = results.loc[results.test_status.eq("tested")]
    summary = dict(module="20_luad_herv_gene_coexpression", inputs=inputs, parameters=params,
                   primary_telescope_samples=n_primary, matched_primary_patients=len(cohort),
                   downloaded_gene_files=n_files, mapped_top1_loci=len(loci),
                   mapped_top1_tumour_upregulated_loci=int(loci.tumor_upregulated.sum()),
                   herv_loci_prevalence_eligible=int((prevalence >= args.min_prevalence).sum()),
                   n_nearby_genes=len(gene_ids), n_nearby_pairs=len(results),
                   n_tested_pairs=len(tested), n_tested_herv_loci=int(tested.herv_internal_id.nunique()),
                   n_nominal_p_lt_005=int(tested.p_value.lt(0.05).sum()),
                   n_bh_significant_pairs=int(tested.bh_significant.sum()),
                   n_positive_links=int(results.positive_link.sum()),
                   n_negative_links=int(results.negative_link.sum()),
                   n_herv_loci_with_positive_links=int(links.loc[links.positive_link,
                                                               "herv_internal_id"].nunique()),
                   n_positive_nonoverlapping_links=int((results.positive_link & ~results.gene_body_overlap).sum()),
                   test_status_counts=results.test_status.value_counts().astype(int).to_dict(),
                   gdc_gene_models=gene_models, coordinate_gtf=str(args.gtf),
                   normalization="HERV: Module 18 size factors; gene: GDC unstranded TPM",
                   multiple_testing="One BH family: all valid tested Top1-nearby-gene pairs",
                   interpretation="Nearby-gene coexpression; not evidence of direct cis regulation",
                   versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                                 pandas=pd.__version__, scipy=scipy.__version__))
    (args.out_dir / "20_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")


def main() -> None:
    step_20()


if __name__ == "__main__":
    main()
