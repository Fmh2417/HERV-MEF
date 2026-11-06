#!/usr/bin/env python3


import gzip
import pandas as pd
from pathlib import Path


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else path.open("r")


def _read_bed3(path: Path) -> list[tuple[str, int, int]]:
    rows = []
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed BED line in {path}: {line[:100]!r}")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid BED interval in {path}: {line[:100]!r}")
            rows.append((chrom, start, end))
    return rows


def load_liftover(chain: Path):
    try:
        from pyliftover import LiftOver
    except ImportError as exc:
        raise RuntimeError("pyliftover is required in the active Python environment") from exc
    return LiftOver(str(chain), use_web=False, write_cache=False)


def strict_lift_interval(lo, chrom: str, start: int, end: int):
    if start < 0 or end <= start:
        return None, "invalid_source"
    left = lo.chain_file.query(chrom, start)
    right = lo.chain_file.query(chrom, end - 1)
    if not left or not right:
        return None, "unmapped_endpoint"

    matches = []
    for l_source_start, l_source_end, l_data in left:
        l_target_start, l_chain = l_data
        for r_source_start, r_source_end, r_data in right:
            r_target_start, r_chain = r_data
            if (l_chain is r_chain and l_source_start == r_source_start and
                    l_source_end == r_source_end and l_target_start == r_target_start):
                matches.append((l_source_start, l_target_start, l_chain))
    if len(matches) != 1:
        return None, "ambiguous_or_cross_block"

    source_block_start, target_block_start, chain = matches[0]
    lpos = target_block_start + (start - source_block_start)
    rpos = target_block_start + (end - 1 - source_block_start)
    if chain.target_strand == "-":
        lpos = chain.target_size - 1 - lpos
        rpos = chain.target_size - 1 - rpos
    lchrom = chain.target_name
    new_start, new_end = min(lpos, rpos), max(lpos, rpos) + 1
    if new_start < 0 or new_end <= new_start:
        return None, "invalid_target"
    if new_end - new_start != end - start:
        return None, "length_changed_within_block"
    return (lchrom, new_start, new_end), "mapped_same_chain_block"


def lift_bed_with_fields(input_bed: Path, chain: Path, output_bed: Path,
                         output_unmapped: Path | None = None, lo=None):
    rows = []
    with _open_text(input_bed) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed BED line in {input_bed}: {line[:100]!r}")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid BED interval in {input_bed}: {line[:100]!r}")
            rows.append((chrom, start, end, fields[3:]))
    if not rows:
        raise RuntimeError(f"No intervals found in {input_bed}")
    lo = lo or load_liftover(chain)
    mapped, unmapped, reasons = [], [], {}
    for chrom, start, end, extra in rows:
        lifted, reason = strict_lift_interval(lo, chrom, start, end)
        reasons[reason] = reasons.get(reason, 0) + 1
        (mapped if lifted else unmapped).append((lifted, extra, chrom, start, end, reason))
    if not mapped:
        raise RuntimeError(f"No intervals survived strict pyliftover for {input_bed}")
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w") as handle:
        for (chrom, start, end), extra, *_ in mapped:
            handle.write("\t".join([chrom, str(start), str(end), *extra]) + "\n")
    if output_unmapped is not None:
        output_unmapped.parent.mkdir(parents=True, exist_ok=True)
        with output_unmapped.open("w") as handle:
            for _, extra, chrom, start, end, reason in unmapped:
                handle.write("\t".join([chrom, str(start), str(end), *extra, reason]) + "\n")
    source_total = sum(end - start for _, start, end, _ in rows)
    mapped_total = sum(item[0][2] - item[0][1] for item in mapped)
    return {
        "source_intervals": len(rows), "lifted_intervals": len(mapped),
        "failed_or_ambiguous_intervals": len(unmapped),
        "source_total_bp": source_total, "lifted_total_bp": mapped_total,
        "length_changed_intervals": 0, "max_abs_length_change_bp": 0,
        "max_interval_length_ratio": 1.0,
        "total_length_ratio": mapped_total / source_total if source_total else 0.0,
        "unmapped_endpoint_count": reasons.get("unmapped_endpoint", 0),
        "cross_block_or_ambiguous_count": reasons.get("ambiguous_or_cross_block", 0),
    }


def lift_complete_bed(input_bed: Path, chain: Path, output_bed: Path,
                      output_unmapped: Path | None = None, lo=None) -> dict[str, int | float]:
    source = _read_bed3(input_bed)
    if not source:
        raise RuntimeError(f"No intervals found in {input_bed}")
    lo = lo or load_liftover(chain)
    mapped_rows = []
    unmapped_rows = []
    reasons: dict[str, int] = {}
    for chrom, start, end in source:
        lifted, reason = strict_lift_interval(lo, chrom, start, end)
        reasons[reason] = reasons.get(reason, 0) + 1
        if lifted is None:
            unmapped_rows.append((chrom, start, end, reason))
        else:
            mapped_rows.append(lifted)
    if not mapped_rows:
        raise RuntimeError(f"No intervals survived strict pyliftover for {input_bed}")

    output_bed.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapped_rows, columns=["chrom", "start", "end"]).sort_values(
        ["chrom", "start", "end"], kind="mergesort").to_csv(
            output_bed, sep="\t", header=False, index=False)
    if output_unmapped is not None:
        output_unmapped.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmapped_rows, columns=["chrom", "start", "end", "reason"]).to_csv(
            output_unmapped, sep="\t", header=False, index=False)

    source_total = sum(end - start for _, start, end in source)
    mapped_total = sum(end - start for _, start, end in mapped_rows)
    # Same-block mapping is length preserving; total mapped/source ratio also
    # reflects the intentionally rejected fraction.
    return {
        "source_intervals": len(source),
        "lifted_intervals": len(mapped_rows),
        "failed_or_ambiguous_intervals": len(unmapped_rows),
        "source_total_bp": source_total,
        "lifted_total_bp": mapped_total,
        "length_changed_intervals": 0,
        "max_abs_length_change_bp": 0,
        "max_interval_length_ratio": 1.0,
        "total_length_ratio": mapped_total / source_total if source_total else 0.0,
        "unmapped_endpoint_count": reasons.get("unmapped_endpoint", 0),
        "cross_block_or_ambiguous_count": reasons.get("ambiguous_or_cross_block", 0),
    }
