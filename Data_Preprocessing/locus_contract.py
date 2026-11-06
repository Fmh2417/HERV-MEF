#!/usr/bin/env python3


import pandas as pd


LOCUS_KEY_COLUMNS = ["chrom", "start", "end", "strand", "repName", "repFamily"]
MODULE_LOCATOR_COLUMNS = ["herv_internal_id", *LOCUS_KEY_COLUMNS]


def normalize_locus_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["herv_internal_id", "chrom", "strand", "repName", "repFamily"]:
        result[column] = result[column].astype(str)
    for column in ["start", "end"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    return result
