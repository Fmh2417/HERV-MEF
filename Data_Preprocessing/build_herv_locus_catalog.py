#!/usr/bin/env python3


import gzip
import hashlib
import json
import re
import pandas as pd
from bisect import bisect_left
from pathlib import Path
from locus_contract import normalize_locus_columns


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "output/01_config/preprocessing_config.json"
OUT = ROOT / "output/02_herv_clean_catalog"
STANDARD_CHROM_ORDER: list[str] = []
STANDARD_CHROMS: set[str] = set()
MIN_LENGTH = 0
MAX_N = 0.0
SEQUENCE_WINDOW_BP = 1000
NO_NEAREST_GENE_DISTANCE = 10_000_000

# Split policy: every chromosome contributes loci to all three model splits.
SPLIT_SEED = 99999999
SPLIT_LABELS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
EXCLUDED_QC_LABEL = "excluded_qc"


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def parse_attrs(text: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'(\w+) "([^"]+)"', text)}


def load_genes(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene" or f[0] not in STANDARD_CHROMS:
                continue
            attrs = parse_attrs(f[8])
            tss = int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1
            rows.append({"chrom": f[0], "tss": tss, "strand": f[6],
                         "gene_id": attrs.get("gene_id", ""),
                         "gene_name": attrs.get("gene_name", ""),
                         "gene_type": attrs.get("gene_type", "")})
    return pd.DataFrame(rows)


def build_gene_index(genes: pd.DataFrame) -> dict[str, tuple[list[int], list[dict[str, str]]]]:
    index = {}
    for chrom, frame in genes.sort_values(["chrom", "tss"]).groupby("chrom", sort=False):
        index[chrom] = (frame.tss.astype(int).tolist(), frame.to_dict("records"))
    return index


def nearest_gene(chrom: str, start: int, end: int,
                 gene_index: dict[str, tuple[list[int], list[dict[str, str]]]]) -> dict[str, object]:
    if chrom not in gene_index:
        return {"dist_to_tss": NO_NEAREST_GENE_DISTANCE, "has_nearest_gene": 0,
                "nearest_gene_id": "", "nearest_gene_name": "",
                "nearest_gene_type": "", "nearest_gene_strand": ""}
    positions, records = gene_index[chrom]
    left = bisect_left(positions, start)
    right = bisect_left(positions, end - 1)
    candidates = sorted({i for pivot in (left, right) for i in (pivot - 1, pivot)
                         if 0 <= i < len(positions)})
    best = min(candidates, key=lambda i: 0 if start <= positions[i] < end else
               min(abs(positions[i] - start), abs(positions[i] - (end - 1))))
    tss = positions[best]
    row = records[best]
    distance = 0 if start <= tss < end else min(abs(tss - start), abs(tss - (end - 1)))
    return {"dist_to_tss": int(distance), "has_nearest_gene": 1,
            "nearest_gene_id": row["gene_id"],
            "nearest_gene_name": row["gene_name"], "nearest_gene_type": row["gene_type"],
            "nearest_gene_strand": row["strand"]}


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


def assign_stratified_random_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    result["datahub_split"] = EXCLUDED_QC_LABEL

    eligible = result.loc[result["sequence_qc_pass"].astype(int) == 1,
                          ["herv_internal_id", "chrom"]].copy()

    eligible["_split_random_key"] = eligible["herv_internal_id"].map(deterministic_split_key)
    split_rows = []

    for chrom in STANDARD_CHROM_ORDER:
        chrom_frame = eligible.loc[eligible["chrom"] == chrom].sort_values(
            ["_split_random_key", "herv_internal_id"], kind="mergesort"
        )
        n_rows = len(chrom_frame)

        counts = allocate_split_counts(n_rows)
        # HERV counts are large on every standard chromosome; require all three
        # splits to be represented so whole-chromosome holdout cannot recur.

        start = 0
        for label in SPLIT_LABELS:
            end = start + counts[label]
            selected_index = chrom_frame.index[start:end]
            result.loc[selected_index, "datahub_split"] = label
            split_rows.append({
                "chrom": chrom,
                "datahub_split": label,
                "rows": int(len(selected_index)),
                "chrom_model_rows": int(n_rows),
                "fraction_within_chrom": float(len(selected_index) / n_rows),
            })
            start = end

    split_by_chrom = pd.DataFrame(split_rows)
    coverage = split_by_chrom.pivot(index="chrom", columns="datahub_split", values="rows")
    coverage = coverage.reindex(index=STANDARD_CHROM_ORDER, columns=SPLIT_LABELS, fill_value=0)
    if (coverage <= 0).any().any():
        bad = coverage.index[(coverage <= 0).any(axis=1)].tolist()
        raise RuntimeError(f"Whole-chromosome holdout detected after random split: {bad}")

    return result, split_by_chrom


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    global STANDARD_CHROM_ORDER, STANDARD_CHROMS, MIN_LENGTH, MAX_N
    cfg = json.loads(CONFIG_PATH.read_text())
    STANDARD_CHROM_ORDER = list(cfg["catalog_policy"]["standard_chromosomes"])
    STANDARD_CHROMS = set(STANDARD_CHROM_ORDER)
    MIN_LENGTH = int(cfg["catalog_policy"]["min_length_bp"])
    MAX_N = float(cfg["catalog_policy"]["max_n_content"])
    inputs = cfg["inputs"]
    rmsk = Path(inputs["repeatmasker"])
    gencode = Path(inputs["gencode_v38"])
    fasta = Path(inputs["hg38_fasta"])
    genes = load_genes(gencode)
    gene_index = build_gene_index(genes)
    import pysam
    rows = []
    with pysam.FastaFile(str(fasta)) as fa, gzip.open(
            rmsk, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in STANDARD_CHROMS:
                continue
            rep_class, rep_family = f[11], f[12]
            if "LTR" not in rep_class and "ERV" not in rep_family:
                continue
            start, end = int(f[6]), int(f[7])
            if end - start < MIN_LENGTH:
                continue
            chrom_length = fa.get_reference_length(f[5])
            center = (start + end) // 2
            seq_start = center - SEQUENCE_WINDOW_BP // 2
            seq_end = seq_start + SEQUENCE_WINDOW_BP
            if seq_start < 0:
                seq_start, seq_end = 0, SEQUENCE_WINDOW_BP
            if seq_end > chrom_length:
                seq_end = chrom_length
                seq_start = chrom_length - SEQUENCE_WINDOW_BP
            if seq_start < 0 or seq_end - seq_start != SEQUENCE_WINDOW_BP:
                continue
            sequence = fa.fetch(f[5], seq_start, seq_end).upper()
            if f[9] == "-":
                sequence = reverse_complement(sequence)
            if len(sequence) != SEQUENCE_WINDOW_BP:
                continue
            n_content = sequence.count("N") / len(sequence)
            gene = nearest_gene(f[5], start, end, gene_index)
            coord_key = f"GRCh38:{f[5]}:{start}-{end}:{f[9]}:{f[10]}:{rep_family}"
            rows.append({"chrom": f[5], "start": start, "end": end, "strand": f[9],
                         "repName": f[10], "repClass": rep_class, "repFamily": rep_family,
                         "milliDiv": int(f[2]), "repeat_divergence": int(f[2]) / 10.0,
                         "milliDel": int(f[3]), "milliIns": int(f[4]), "length": end - start,
                         "coord_key": coord_key, "sequence": sequence, "seq_start": seq_start,
                         "seq_end": seq_end, "seq_len": len(sequence),
                         "gc_content": (sequence.count("G") + sequence.count("C")) / len(sequence),
                         "n_content": n_content, **gene})

    df = pd.DataFrame(rows).sort_values(
        ["chrom", "start", "end", "strand", "repName", "repFamily"], kind="mergesort"
    ).reset_index(drop=True)
    df.insert(0, "herv_internal_id", [f"HERV_HG38_{i + 1:07d}" for i in range(len(df))])
    df = normalize_locus_columns(df)

    # Model eligibility is independent of labels and external functional evidence.
    df["sequence_qc_pass"] = (
        (df.seq_len == SEQUENCE_WINDOW_BP) & (df.n_content <= MAX_N)
    ).astype("int8")
    df["is_model_universe"] = df["sequence_qc_pass"]

    # Remove whole-chromosome holdout. Every chromosome is randomly represented
    # in train/valid/test using a fixed, reproducible seed.
    df, split_by_chrom = assign_stratified_random_splits(df)

    bed = df[["chrom", "start", "end", "herv_internal_id", "strand"]]
    bed.to_csv(OUT / "herv_clean_catalog.bed", sep="\t", header=False, index=False)
    df.to_parquet(OUT / "herv_clean_catalog_full.parquet", index=False)
    df.to_csv(OUT / "herv_clean_catalog_full.csv", index=False)
    model = df[df.is_model_universe == 1].copy()

    model[["herv_internal_id", "chrom", "datahub_split"]].to_csv(
        OUT / "split_manifest.tsv", sep="\t", index=False)
    split_by_chrom.to_csv(OUT / "split_summary_by_chromosome.tsv", sep="\t", index=False)

    split_policy = {
        "method": "deterministic_seeded_hash_random_split_stratified_within_chromosome",
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "eligible_rule": "sequence_qc_pass == 1",
        "excluded_qc_label": EXCLUDED_QC_LABEL,
        "all_standard_chromosomes_present_in_each_model_split": True,
        "standard_chromosomes": STANDARD_CHROM_ORDER,
        "model_rows": int(len(model)),
        "split_distribution": {
            label: int((model.datahub_split == label).sum()) for label in SPLIT_LABELS
        },
    }
    (OUT / "split_policy.json").write_text(
        json.dumps(split_policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
