#!/usr/bin/env python3


import importlib.metadata
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import DATASET


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/01_config"
SPLIT = ROOT / "output/00_sample_split"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        "repeatmasker": DATASET / "reference/rmsk.txt.gz",
        "gencode_v38": DATASET / "reference/gencode.v38.annotation.gtf.gz",
        "hg38_fasta": DATASET / "reference/hg38.fa",
        "hg38_fai": DATASET / "reference/hg38.fa.fai",
        "hg19_to_hg38_chain": DATASET / "reference/hg19ToHg38.over.chain.gz",
        "roadmap_root": DATASET / "roadmap",
        "roadmap_eid_metadata": DATASET / "roadmap/EID_metadata.tab.txt",
        "fantom5_dir": DATASET / "FANTOM5",
        "fantom5_enhancer_bed": DATASET / "FANTOM5/F5.hg38.enhancers.bed",
        "fantom5_expression_matrix": DATASET / "FANTOM5/F5.hg38.enhancers.expression.matrix",
        "fantom5_tpm_matrix": DATASET / "FANTOM5/F5.hg38.enhancers.expression.tpm.matrix",
        "fantom5_usage_matrix": DATASET / "FANTOM5/F5.hg38.enhancers.expression.usage.matrix",
        "fantom5_sample_metadata": DATASET / "FANTOM5/HumanSamples2.0.sdrf.xlsx",
        "fantom5_official_sdrf_dir": DATASET / "FANTOM5/official_sdrf",
        "ccre_root": DATASET / "cCRE",
        "mpra_ipsc_counts": DATASET / "GSE245662_iPSC_allreps.tsv",
        "mpra_npc_counts": DATASET / "GSE245662_NPC_allreps.tsv",
        "serve_annotation_dir": DATASET / "SERVE-main/Annotation",
        "serve_supplementary_tables": DATASET / "SERVE_Supplementary_Tables.xlsx",
        "bpe_raw_sequences": DATASET / "reference/hg38.fa",
    }

    rows = []
    for name, path in inputs.items():
        is_file = path.is_file()
        rows.append({
            "name": name,
            "path": str(path.resolve()),
            "exists": path.exists(),
            "type": "file" if is_file else "directory",
            "size_bytes": path.stat().st_size if is_file else None,
        })

    missing = [row["path"] for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError("Missing preprocessing inputs: " + ", ".join(missing))

    tools = {
        "python": shutil.which("python") or "",
        "bedtools": shutil.which("bedtools") or "",
        "pyliftover_version": importlib.metadata.version("pyliftover"),
    }

    tumor_eids = [line.strip() for line in (SPLIT / "tumor_eids.txt").read_text().splitlines() if line.strip()]
    non_tumor_eids = [line.strip() for line in (SPLIT / "non_tumor_eids.txt").read_text().splitlines() if line.strip()]
    resolved = {name: str(path.resolve()) for name, path in inputs.items()}
    resolved["fantom5"] = {
        "fantom5_dir": resolved["fantom5_dir"],
        "enhancer_bed": resolved["fantom5_enhancer_bed"],
        "expression_matrix": resolved["fantom5_expression_matrix"],
        "tpm_matrix": resolved["fantom5_tpm_matrix"],
        "usage_matrix": resolved["fantom5_usage_matrix"],
        "sample_metadata": resolved["fantom5_sample_metadata"],
    }
    resolved["roadmap"] = {"peak_root": resolved["roadmap_root"],
                           "eid_metadata": resolved["roadmap_eid_metadata"]}
    config = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(ROOT.resolve()),
        "inputs": resolved,
        "tools": tools,
        "roadmap_policy": {
            "training_set": "non_tumor_training",
            "external_projection_set": "tumor_external_projection",
            "explicit_tumor_eids": sorted(tumor_eids),
            "non_tumor_eids": sorted(non_tumor_eids),
            "tumor_data_allowed_in_training": False,
            "source_assembly": "GRCh37/hg19",
            "target_assembly": "GRCh38/hg38",
            "liftover_engine": "pyliftover strict same-chain/same-block intervals",
        },
        "catalog_policy": {
            "source_assembly": "GRCh38/hg38",
            "standard_chromosomes": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
            "min_length_bp": 200,
            "max_n_content": 0.05,
        },
    }
    config_path = OUT / "preprocessing_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "raw_input_manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
