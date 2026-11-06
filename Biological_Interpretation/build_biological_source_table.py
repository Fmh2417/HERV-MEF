#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from biological_common import atomic_write_tsv


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE_CONFIG

import hashlib
import pandas as pd

# --------------------------------------------------------------------------
# step 01: Build the biological source table
# --------------------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent
Modeling = (PROJECT_ROOT / "Modeling").resolve()

OUT_DIR = (PROJECT_ROOT / "Biological_Interpretation/output/01_build_biological_source_table").resolve()

ID_COL = "herv_internal_id"
EXPECTED_CANONICAL_ROWS = 480_306

INPUTS: dict[str, Path] = {
    "M23": Modeling
    / "output/23_generate_genomewide_herv_functional_ranking/23_herv_genomewide_functional_ranking.parquet",
    "M24": Modeling
    / "output/24_project_frozen_herv_ranking_to_tumor_epigenomes/24_tumor_projection_all_herv.parquet",
}

OUTPUT_TABLE = OUT_DIR / "01_biological_source_table.parquet"
OUTPUT_TSV = OUT_DIR / "01_biological_source_table.tsv.gz"




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def validate_id_table(frame: pd.DataFrame) -> set[str]:
    return set(frame[ID_COL].astype("string").tolist())


def validate_shared_values(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    right_ids: set[str],
) -> None:
    shared_columns = [
        column
        for column in ("chrom", "start", "end", "strand", "repName", "repFamily")
        if column in left.columns and column in right.columns
    ]
    if not shared_columns:
        return

    left_subset = left[left[ID_COL].isin(right_ids)].set_index(ID_COL)
    right_subset = right.set_index(ID_COL)
    common_ids = left_subset.index.intersection(right_subset.index)
    for column in shared_columns:
        left_values = left_subset.loc[common_ids, column].astype("string")
        right_values = right_subset.loc[common_ids, column].astype("string")
        mismatch = (left_values != right_values).fillna(False)
        if bool(mismatch.any()):
            count = int(mismatch.sum())
            raise ValueError(
                f"{left_name}/{right_name} mismatch in {column}: {count} rows"
            )


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)




def step_01() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    tables: dict[str, pd.DataFrame] = {}
    ids: dict[str, set[str]] = {}

    for name, path in INPUTS.items():
        frame = read_table(path)
        table_ids = validate_id_table(frame)

        tables[name] = frame
        ids[name] = table_ids

    for name in ("M24",):
        validate_shared_values("M23", tables["M23"], name, tables[name], ids[name])

    source = tables["M23"].copy()
    for name in ("M24",):
        source = source.merge(
            tables[name],
            on=ID_COL,
            how="left",
            suffixes=("", "_" + name),
            validate="one_to_one",
        )

    for name in INPUTS:
        source[f"source_{name}_present"] = source[ID_COL].isin(ids[name])

    if source.columns.duplicated().any():
        duplicate_columns = source.columns[source.columns.duplicated()].tolist()
        raise RuntimeError(f"Output contains duplicate columns: {duplicate_columns}")


    atomic_write_parquet(source, OUTPUT_TABLE)
    atomic_write_tsv(source, OUTPUT_TSV)


def main() -> None:
    step_01()


if __name__ == "__main__":
    main()
