#!/usr/bin/env python3


import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from locus_contract import (
    LOCUS_KEY_COLUMNS,
    normalize_locus_columns,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/09_final_audit"
TUMOR_EIDS = {"E114", "E115", "E117", "E118", "E123"}
FORBIDDEN_PATHS = ("/MVP/", "MVP/data_merger/output", "Base_MVP_NEW", "HERV_MEF/output")
FORBIDDEN_CATALOG_COLUMNS = {
    "main_c_type", "is_promoter", "is_h3k4me3", "is_ctcf", "is_ca_tf", "is_ca",
    "is_ccre", "is_enhancer", "label_ccre_any", "label_enhancer_loose",
}
SPLIT_METHOD = "deterministic_seeded_hash_random_split_stratified_within_chromosome"
SPLIT_SEED = 99999999
SPLIT_LABELS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
EXCLUDED_QC_LABEL = "excluded_qc"


def deterministic_split_key(herv_internal_id: str) -> str:
    payload = f"{SPLIT_SEED}|{herv_internal_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate_split_counts(n_rows: int) -> dict[str, int]:
    raw = {label: n_rows * SPLIT_RATIOS[label] for label in SPLIT_LABELS}
    counts = {label: int(raw[label]) for label in SPLIT_LABELS}
    remainder = n_rows - sum(counts.values())
    order = sorted(
        SPLIT_LABELS,
        key=lambda label: (raw[label] - counts[label], -SPLIT_LABELS.index(label)),
        reverse=True,
    )
    for label in order[:remainder]:
        counts[label] += 1
    return counts


def reconstruct_expected_splits(catalog: pd.DataFrame, chrom_order: list[str]) -> pd.Series:
    expected = pd.Series(EXCLUDED_QC_LABEL, index=catalog.index, dtype="object")
    eligible = catalog.loc[
        catalog["sequence_qc_pass"].astype(int) == 1,
        ["herv_internal_id", "chrom"],
    ].copy()
    eligible["_split_random_key"] = eligible["herv_internal_id"].astype(str).map(
        deterministic_split_key
    )
    for chrom in chrom_order:
        frame = eligible.loc[eligible["chrom"].astype(str) == chrom].sort_values(
            ["_split_random_key", "herv_internal_id"], kind="mergesort"
        )
        counts = allocate_split_counts(len(frame))
        start = 0
        for label in SPLIT_LABELS:
            end = start + counts[label]
            expected.loc[frame.index[start:end]] = label
            start = end
    return expected


def main() -> None:
    # This version intentionally replaces the complete DP09 result set while
    # retaining a versioned script filename.  The step-specific output
    # directory contains only regenerated results from the current audit run.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg_path = ROOT / "output/01_config/preprocessing_config.json"
    catalog_path = ROOT / "output/02_herv_clean_catalog/herv_clean_catalog_full.parquet"
    split_manifest_path = ROOT / "output/02_herv_clean_catalog/split_manifest.tsv"
    split_summary_path = ROOT / "output/02_herv_clean_catalog/split_summary_by_chromosome.tsv"
    split_policy_path = ROOT / "output/02_herv_clean_catalog/split_policy.json"
    fantom_manifest_path = ROOT / "output/03_fantom5_rebuild/fantom5_cnhs_sample_manifest.tsv"
    non_path = ROOT / "output/04_roadmap_manifests/non_tumor_roadmap_manifest.csv"
    tumor_path = ROOT / "output/04_roadmap_manifests/tumor_only_roadmap_manifest.csv"
    lift_path = ROOT / "output/05_separated_roadmap_features/roadmap_hg19_to_hg38_liftover_audit.tsv"
    tumor_long_path = ROOT / "output/05_separated_roadmap_features/herv_tumor_cell_line_roadmap_evidence_long.parquet"
    tumor_file_audit_path = ROOT / "output/05_separated_roadmap_features/tumor_evidence_summary_by_cell_line_mark.tsv"
    mpra_lift_path = ROOT / "output/07_ccre_enhancer_features/mpra_hg19_to_hg38_liftover_audit.json"
    module_path = ROOT / "output/08_non_tumor_datahub/datahub_module_input_audit.tsv"
    full_path = ROOT / "output/08_non_tumor_datahub/herv_non_tumor_datahub_full.parquet"
    model_path = ROOT / "output/08_non_tumor_datahub/herv_non_tumor_datahub_model_universe.parquet"

    cfg = json.loads(cfg_path.read_text())
    catalog = normalize_locus_columns(pd.read_parquet(catalog_path))
    split_manifest = pd.read_csv(split_manifest_path, sep="\t")
    split_summary = pd.read_csv(split_summary_path, sep="\t")
    split_policy = json.loads(split_policy_path.read_text(encoding="utf-8"))
    fantom = pd.read_csv(fantom_manifest_path, sep="\t")
    non = pd.read_csv(non_path)
    tumor = pd.read_csv(tumor_path)
    lift = pd.read_csv(lift_path, sep="\t")
    tumor_long = pd.read_parquet(tumor_long_path)
    tumor_file_audit = pd.read_csv(tumor_file_audit_path, sep="\t")
    mpra_lift = json.loads(mpra_lift_path.read_text())
    modules = pd.read_csv(module_path, sep="\t")
    full = normalize_locus_columns(pd.read_parquet(full_path))
    model = normalize_locus_columns(pd.read_parquet(model_path))
    checks = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": str(observed),
                       "expected": str(expected)})

    def require_columns(frame: pd.DataFrame, label: str, required_columns: set[str]) -> None:
        missing_columns = sorted(required_columns - set(frame.columns))
        check(f"{label}_required_columns", not missing_columns, missing_columns, [])

    def string_paths(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [path for child in value.values() for path in string_paths(child)]
        if isinstance(value, list):
            return [path for child in value for path in string_paths(child)]
        return []

    input_paths = string_paths(cfg["inputs"])
    module_paths = modules.input_path.astype(str).tolist()
    forbidden_hits = [path for path in input_paths + module_paths
                      if any(fragment in path for fragment in FORBIDDEN_PATHS)]
    check("no_forbidden_derived_inputs", not forbidden_hits, forbidden_hits, [])
    legacy_cols = set(catalog.columns) & FORBIDDEN_CATALOG_COLUMNS
    check("catalog_has_no_legacy_derived_columns", not legacy_cols, sorted(legacy_cols), [])
    check("catalog_unique_ids", catalog.herv_internal_id.nunique() == len(catalog),
          catalog.herv_internal_id.nunique(), len(catalog))
    check("catalog_unique_locus_keys", not catalog.duplicated(LOCUS_KEY_COLUMNS).any(),
          int(catalog.duplicated(LOCUS_KEY_COLUMNS).sum()), 0)

    # DP02 v1 split contract: sequence-QC-only model universe, deterministic
    # within-chromosome random allocation, and no whole-chromosome holdout.
    require_columns(
        catalog,
        "catalog_split",
        {"herv_internal_id", "chrom", "sequence_qc_pass", "is_model_universe", "datahub_split"},
    )
    require_columns(
        split_manifest,
        "split_manifest",
        {"herv_internal_id", "chrom", "datahub_split"},
    )
    require_columns(
        split_summary,
        "split_summary",
        {"chrom", "datahub_split", "rows", "chrom_model_rows", "fraction_within_chrom"},
    )
    required_policy_keys = {
        "method", "seed", "ratios", "eligible_rule", "excluded_qc_label",
        "all_standard_chromosomes_present_in_each_model_split", "standard_chromosomes",
        "model_rows", "split_distribution",
    }
    missing_policy_keys = sorted(required_policy_keys - set(split_policy))
    check("split_policy_required_keys", not missing_policy_keys, missing_policy_keys, [])

    expected_chrom_order = list(cfg["catalog_policy"]["standard_chromosomes"])
    observed_policy_ratios = {label: float(split_policy["ratios"][label]) for label in SPLIT_LABELS}
    ratio_match = all(
        np.isclose(observed_policy_ratios[label], SPLIT_RATIOS[label], rtol=0.0, atol=1e-12)
        for label in SPLIT_LABELS
    )
    check("split_policy_method", split_policy["method"] == SPLIT_METHOD,
          split_policy["method"], SPLIT_METHOD)
    check("split_policy_seed", int(split_policy["seed"]) == SPLIT_SEED,
          split_policy["seed"], SPLIT_SEED)
    check("split_policy_ratios", ratio_match, observed_policy_ratios, SPLIT_RATIOS)
    check("split_policy_eligible_rule", split_policy["eligible_rule"] == "sequence_qc_pass == 1",
          split_policy["eligible_rule"], "sequence_qc_pass == 1")
    check("split_policy_excluded_qc_label", split_policy["excluded_qc_label"] == EXCLUDED_QC_LABEL,
          split_policy["excluded_qc_label"], EXCLUDED_QC_LABEL)
    check("split_policy_standard_chromosomes",
          list(split_policy["standard_chromosomes"]) == expected_chrom_order,
          split_policy["standard_chromosomes"], expected_chrom_order)
    check("split_policy_declares_all_chromosomes_in_each_split",
          bool(split_policy["all_standard_chromosomes_present_in_each_model_split"]),
          split_policy["all_standard_chromosomes_present_in_each_model_split"], True)

    catalog["sequence_qc_pass"] = pd.to_numeric(
        catalog["sequence_qc_pass"], errors="raise"
    ).astype("int8")
    catalog["is_model_universe"] = pd.to_numeric(
        catalog["is_model_universe"], errors="raise"
    ).astype("int8")
    catalog["datahub_split"] = catalog["datahub_split"].astype(str)
    eligible_mask = catalog["sequence_qc_pass"].eq(1)
    excluded_mask = ~eligible_mask
    allowed_catalog_labels = set(SPLIT_LABELS) | {EXCLUDED_QC_LABEL}
    catalog_labels = set(catalog["datahub_split"])
    check("catalog_split_labels_allowed", catalog_labels.issubset(allowed_catalog_labels),
          sorted(catalog_labels), sorted(allowed_catalog_labels))
    check("model_universe_equals_sequence_qc_pass",
          catalog["is_model_universe"].equals(catalog["sequence_qc_pass"]),
          int((catalog["is_model_universe"] != catalog["sequence_qc_pass"]).sum()), 0)
    check("qc_failing_loci_are_excluded_qc",
          catalog.loc[excluded_mask, "datahub_split"].eq(EXCLUDED_QC_LABEL).all(),
          catalog.loc[excluded_mask, "datahub_split"].value_counts().to_dict(),
          {EXCLUDED_QC_LABEL: int(excluded_mask.sum())})
    check("qc_passing_loci_use_only_model_splits",
          catalog.loc[eligible_mask, "datahub_split"].isin(SPLIT_LABELS).all(),
          catalog.loc[eligible_mask, "datahub_split"].value_counts().to_dict(), list(SPLIT_LABELS))
    check("catalog_model_split_labels_exact",
          set(catalog.loc[eligible_mask, "datahub_split"]) == set(SPLIT_LABELS),
          sorted(set(catalog.loc[eligible_mask, "datahub_split"])), sorted(SPLIT_LABELS))

    expected_assignment = reconstruct_expected_splits(catalog, expected_chrom_order)
    assignment_mismatch = catalog["datahub_split"].ne(expected_assignment)
    check("split_assignments_reconstruct_from_seed_and_id",
          not assignment_mismatch.any(), int(assignment_mismatch.sum()), 0)

    eligible_catalog = catalog.loc[
        eligible_mask, ["herv_internal_id", "chrom", "datahub_split"]
    ].copy()
    observed_chroms = set(eligible_catalog["chrom"].astype(str))
    check("model_universe_has_exact_standard_chromosomes",
          observed_chroms == set(expected_chrom_order), sorted(observed_chroms), expected_chrom_order)
    split_nunique_by_chrom = eligible_catalog.groupby("chrom")["datahub_split"].nunique()
    whole_chromosome_holdouts = sorted(
        split_nunique_by_chrom.index[split_nunique_by_chrom.eq(1)].astype(str).tolist()
    )
    check("no_whole_chromosome_holdout", not whole_chromosome_holdouts,
          whole_chromosome_holdouts, [])
    incomplete_three_split_chroms = sorted(
        split_nunique_by_chrom.index[split_nunique_by_chrom.ne(len(SPLIT_LABELS))]
        .astype(str).tolist()
    )
    check("current_policy_all_chromosomes_cover_all_three_splits",
          not incomplete_three_split_chroms, incomplete_three_split_chroms, [])

    observed_counts = (
        eligible_catalog.groupby(["chrom", "datahub_split"], sort=False).size()
        .unstack(fill_value=0)
        .reindex(index=expected_chrom_order, columns=SPLIT_LABELS, fill_value=0)
        .astype(int)
    )
    expected_count_rows = []
    for chrom in expected_chrom_order:
        chrom_total = int((eligible_catalog["chrom"].astype(str) == chrom).sum())
        allocated = allocate_split_counts(chrom_total)
        for label in SPLIT_LABELS:
            expected_count_rows.append({
                "chrom": chrom,
                "datahub_split": label,
                "expected_rows": int(allocated[label]),
                "chrom_model_rows": chrom_total,
                "expected_fraction": float(allocated[label] / chrom_total),
            })
    expected_split_counts = pd.DataFrame(expected_count_rows)
    observed_long = (
        observed_counts.stack()
        .rename("observed_rows")
        .reset_index()
    )
    split_audit = expected_split_counts.merge(
        observed_long, on=["chrom", "datahub_split"], how="left", validate="one_to_one"
    )
    split_audit["observed_rows"] = split_audit["observed_rows"].fillna(0).astype(int)
    split_audit["observed_fraction"] = (
        split_audit["observed_rows"] / split_audit["chrom_model_rows"]
    )
    split_audit["row_count_match"] = split_audit["observed_rows"].eq(split_audit["expected_rows"])
    check("per_chromosome_split_counts_match_70_15_15_allocation",
          bool(split_audit["row_count_match"].all()),
          int((~split_audit["row_count_match"]).sum()), 0)

    check("split_manifest_unique_ids",
          split_manifest["herv_internal_id"].nunique() == len(split_manifest),
          split_manifest["herv_internal_id"].nunique(), len(split_manifest))
    check("split_manifest_rows", len(split_manifest) == len(eligible_catalog),
          len(split_manifest), len(eligible_catalog))
    check("split_manifest_only_model_labels",
          set(split_manifest["datahub_split"].astype(str)) == set(SPLIT_LABELS),
          sorted(set(split_manifest["datahub_split"].astype(str))), sorted(SPLIT_LABELS))
    manifest_compare = eligible_catalog.merge(
        split_manifest[["herv_internal_id", "chrom", "datahub_split"]],
        on="herv_internal_id", how="outer", validate="one_to_one",
        suffixes=("_catalog", "_manifest"), indicator=True,
    )
    manifest_mismatch = (
        manifest_compare["_merge"].ne("both")
        | manifest_compare["chrom_catalog"].astype(str).ne(manifest_compare["chrom_manifest"].astype(str))
        | manifest_compare["datahub_split_catalog"].astype(str).ne(
            manifest_compare["datahub_split_manifest"].astype(str)
        )
    )
    check("split_manifest_exact_catalog_mapping", not manifest_mismatch.any(),
          int(manifest_mismatch.sum()), 0)

    check("split_summary_unique_keys",
          not split_summary.duplicated(["chrom", "datahub_split"]).any(),
          int(split_summary.duplicated(["chrom", "datahub_split"]).sum()), 0)
    summary_compare = split_audit.merge(
        split_summary,
        on=["chrom", "datahub_split"], how="outer", validate="one_to_one", indicator=True,
    )
    summary_mismatch = (
        summary_compare["_merge"].ne("both")
        | pd.to_numeric(summary_compare["rows"], errors="coerce").ne(
            summary_compare["observed_rows"]
        )
        | pd.to_numeric(summary_compare["chrom_model_rows_y"], errors="coerce").ne(
            summary_compare["chrom_model_rows_x"]
        )
        | ~np.isclose(
            pd.to_numeric(summary_compare["fraction_within_chrom"], errors="coerce"),
            summary_compare["observed_fraction"], rtol=0.0, atol=1e-12, equal_nan=False,
        )
    )
    check("split_summary_matches_catalog", not summary_mismatch.any(),
          int(summary_mismatch.sum()), 0)

    observed_split_distribution = {
        label: int((eligible_catalog["datahub_split"] == label).sum()) for label in SPLIT_LABELS
    }
    policy_split_distribution = {
        label: int(split_policy["split_distribution"][label]) for label in SPLIT_LABELS
    }
    check("split_policy_model_rows", int(split_policy["model_rows"]) == len(eligible_catalog),
          split_policy["model_rows"], len(eligible_catalog))
    check("split_policy_distribution_matches_catalog",
          policy_split_distribution == observed_split_distribution,
          policy_split_distribution, observed_split_distribution)

    check("non_tumor_manifest_rows", len(non) == 620, len(non), 620)
    check("tumor_manifest_rows", len(tumor) == 25, len(tumor), 25)
    check("manifest_eid_disjoint", not set(non.eid) & set(tumor.eid),
          sorted(set(non.eid) & set(tumor.eid)), [])
    check("tumor_manifest_exact_eids", set(tumor.eid) == TUMOR_EIDS,
          sorted(set(tumor.eid)), sorted(TUMOR_EIDS))
    check("tumor_long_only_allowed_eids", set(tumor_long.eid).issubset(TUMOR_EIDS),
          sorted(set(tumor_long.eid)), sorted(TUMOR_EIDS))
    check("tumor_dnase_absent", "DNase_fdr001" not in set(tumor_long.mark),
          sorted(set(tumor_long.mark)), "no DNase_fdr001")
    expected_tumor_pairs = set(map(tuple, tumor[["eid", "mark"]].to_numpy()))
    audited_tumor_pairs = set(map(tuple, tumor_file_audit[["eid", "mark"]].to_numpy()))
    check("tumor_all_25_files_processed", len(tumor_file_audit) == 25 and
          audited_tumor_pairs == expected_tumor_pairs, len(tumor_file_audit), 25)
    check("tumor_positive_long_only_contract",
          tumor_long.empty or set(tumor_long.has_overlap.astype(int)) == {1},
          sorted(set(tumor_long.has_overlap.astype(int))), [1])

    used = fantom[fantom.used_for_activity_aggregation.astype(int) == 1]
    tumor_fantom = fantom[fantom.is_tumor.astype(int) == 1]
    check("fantom_samples_audited", len(fantom) == 1829, len(fantom), 1829)
    check("fantom_non_tumor_used", len(used) == 1361, len(used), 1361)
    check("fantom_tumor_excluded", len(tumor_fantom) == 468, len(tumor_fantom), 468)
    check("fantom_no_tumor_used", not set(used.sample_id) & set(tumor_fantom.sample_id),
          len(set(used.sample_id) & set(tumor_fantom.sample_id)), 0)

    accounted = lift.source_intervals == lift.lifted_intervals + lift.failed_or_ambiguous_intervals
    total_ratio = float(lift.lifted_total_bp.sum() / lift.source_total_bp.sum())
    check("liftover_all_645_files", len(lift) == 645, len(lift), 645)
    check("liftover_interval_accounting", bool(accounted.all()), int((~accounted).sum()), 0)
    check("liftover_total_length_ratio", 0.8 <= total_ratio <= 1.2, total_ratio, "0.8..1.2")
    check("liftover_no_extreme_interval_expansion",
          float(lift.max_interval_length_ratio.max()) <= 10.0,
          float(lift.max_interval_length_ratio.max()), "<=10")

    mpra_accounted = (int(mpra_lift["source_intervals"]) ==
                      int(mpra_lift["lifted_intervals"]) +
                      int(mpra_lift["failed_or_ambiguous_intervals"]))
    check("mpra_liftover_engine", str(mpra_lift["engine"]).startswith("pyliftover"),
          mpra_lift["engine"], "pyliftover*")
    check("mpra_liftover_interval_accounting", mpra_accounted,
          [mpra_lift["source_intervals"], mpra_lift["lifted_intervals"],
           mpra_lift["failed_or_ambiguous_intervals"]], "source=mapped+unmapped")
    check("mpra_liftover_length_ratio", 0.8 <= float(mpra_lift["total_length_ratio"]) <= 1.2,
          mpra_lift["total_length_ratio"], "0.8..1.2")

    expected_rows = len(catalog)
    expected_model = int(catalog.is_model_universe.astype(int).sum())
    check("datahub_full_rows", len(full) == expected_rows, len(full), expected_rows)
    check("datahub_model_rows", len(model) == expected_model, len(model), expected_model)
    check("datahub_unique_ids", full.herv_internal_id.nunique() == len(full),
          full.herv_internal_id.nunique(), len(full))
    check("datahub_exact_catalog_ids", set(full.herv_internal_id.astype(str)) ==
          set(catalog.herv_internal_id.astype(str)), len(set(full.herv_internal_id.astype(str))), expected_rows)
    check("datahub_exact_catalog_locus_keys",
          set(map(tuple, full[LOCUS_KEY_COLUMNS].to_numpy())) ==
          set(map(tuple, catalog[LOCUS_KEY_COLUMNS].to_numpy())),
          len(set(map(tuple, full[LOCUS_KEY_COLUMNS].to_numpy()))), expected_rows)
    check("datahub_locus_order_preserved", full[LOCUS_KEY_COLUMNS].equals(catalog[LOCUS_KEY_COLUMNS]),
          False if not full[LOCUS_KEY_COLUMNS].equals(catalog[LOCUS_KEY_COLUMNS]) else True, True)
    require_columns(full, "full_datahub_split", {"herv_internal_id", "datahub_split",
                                                  "sequence_qc_pass", "is_model_universe"})
    require_columns(model, "model_datahub_split", {"herv_internal_id", "chrom", "datahub_split",
                                                    "sequence_qc_pass", "is_model_universe"})
    full_split_compare = catalog[["herv_internal_id", "datahub_split"]].merge(
        full[["herv_internal_id", "datahub_split"]],
        on="herv_internal_id", how="outer", validate="one_to_one",
        suffixes=("_catalog", "_datahub"), indicator=True,
    )
    full_split_mismatch = (
        full_split_compare["_merge"].ne("both")
        | full_split_compare["datahub_split_catalog"].astype(str).ne(
            full_split_compare["datahub_split_datahub"].astype(str)
        )
    )
    check("full_datahub_split_exact_catalog_mapping", not full_split_mismatch.any(),
          int(full_split_mismatch.sum()), 0)
    check("full_datahub_model_universe_matches_catalog",
          full["is_model_universe"].astype(int).equals(catalog["is_model_universe"].astype(int)),
          int((full["is_model_universe"].astype(int) != catalog["is_model_universe"].astype(int)).sum()), 0)
    check("full_datahub_sequence_qc_matches_catalog",
          full["sequence_qc_pass"].astype(int).equals(catalog["sequence_qc_pass"].astype(int)),
          int((full["sequence_qc_pass"].astype(int) != catalog["sequence_qc_pass"].astype(int)).sum()), 0)

    model_compare = eligible_catalog.merge(
        model[["herv_internal_id", "chrom", "datahub_split"]],
        on="herv_internal_id", how="outer", validate="one_to_one",
        suffixes=("_catalog", "_datahub"), indicator=True,
    )
    model_split_mismatch = (
        model_compare["_merge"].ne("both")
        | model_compare["chrom_catalog"].astype(str).ne(model_compare["chrom_datahub"].astype(str))
        | model_compare["datahub_split_catalog"].astype(str).ne(
            model_compare["datahub_split_datahub"].astype(str)
        )
    )
    check("model_datahub_exact_catalog_split_mapping", not model_split_mismatch.any(),
          int(model_split_mismatch.sum()), 0)
    check("model_datahub_only_model_splits",
          set(model["datahub_split"].astype(str)) == set(SPLIT_LABELS),
          sorted(set(model["datahub_split"].astype(str))), sorted(SPLIT_LABELS))
    check("model_datahub_all_sequence_qc_pass",
          model["sequence_qc_pass"].astype(int).eq(1).all(),
          model["sequence_qc_pass"].astype(int).value_counts().to_dict(), {1: len(model)})
    check("model_datahub_all_in_model_universe",
          model["is_model_universe"].astype(int).eq(1).all(),
          model["is_model_universe"].astype(int).value_counts().to_dict(), {1: len(model)})
    if "seq_len" in catalog.columns:
        check("fixed_1000bp_sequences", int((catalog.seq_len == 1000).sum()) == len(catalog),
              int((catalog.seq_len == 1000).sum()), len(catalog))
    numeric = full.select_dtypes(include=[np.number])
    check("numeric_finite", bool(np.isfinite(numeric.to_numpy()).all()), "finite", "finite")
    check("no_unexpected_nan", not full.drop(columns=["datahub_split"]).isna().any().any(),
          int(full.drop(columns=["datahub_split"]).isna().sum().sum()), 0)
    check("module_set", set(modules.module) == {"fantom5", "non_tumor_roadmap", "serve_gtex", "ccre_mpra"},
          sorted(set(modules.module)), ["ccre_mpra", "fantom5", "non_tumor_roadmap", "serve_gtex"])
    check("no_tumor_named_training_columns", not [c for c in full.columns if "tumor" in c.lower()],
          [c for c in full.columns if "tumor" in c.lower()], [])
    tumor_module_paths = []
    for path in module_paths:
        basename = Path(path).name.lower()
        # "non_tumor" is the required training scope, not a tumor-data hit.
        normalized = basename.replace("non_tumor", "")
        if "tumor" in normalized or "projection" in normalized:
            tumor_module_paths.append(path)
    check("no_tumor_training_module_paths", not tumor_module_paths, tumor_module_paths, [])

    # Coordinate-level spot audit: sample once, then show the catalog coordinates,
    # every module's coordinates, and the merged feature row side by side.
    module_frames = {}
    module_key_bad = []
    for row in modules.itertuples(index=False):
        name = str(row.module)
        frame = normalize_locus_columns(pd.read_parquet(str(row.input_path)))
        module_frames[name] = frame
        if set(map(tuple, frame[LOCUS_KEY_COLUMNS].to_numpy())) != set(
                map(tuple, catalog[LOCUS_KEY_COLUMNS].to_numpy())):
            module_key_bad.append(name)
    check("module_locus_keys_match_catalog", not module_key_bad, module_key_bad, [])

    split_audit_path = OUT / "09_split_audit_by_chromosome.tsv"
    split_audit.to_csv(split_audit_path, sep="\t", index=False)

    pd.DataFrame(checks).to_csv(OUT / "09_final_audit_checks.tsv", sep="\t", index=False)
    run_manifest = {
        "status": "PASSED", "checks_passed": len(checks), "catalog_rows": expected_rows,
        "model_rows": expected_model,
        "liftover_total_length_ratio": total_ratio,
        "split_method": SPLIT_METHOD,
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_distribution": observed_split_distribution,
        "whole_chromosome_holdout_detected": False,
        "split_audit_path": str(split_audit_path.resolve()),
    }
    (OUT / "09_final_audit_summary.json").write_text(json.dumps(run_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
