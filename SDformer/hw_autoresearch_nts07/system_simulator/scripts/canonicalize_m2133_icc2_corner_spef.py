#!/usr/bin/env python3
"""Fail-closed canonicalization of the single M2133 TT corner SPEF."""

import argparse
import json
import os
import re
from pathlib import Path


RAW_NAME = re.compile(r"^routed\.n28_1p9m_6x1z1u_typ_(25(?:\.0+)?)\.spef$")
CANONICAL_NAME = "routed.spef"


def canonicalize(raw_dir, output_dir, receipt):
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise ValueError("raw_parasitics must be a real directory")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("output must be a real directory")

    # Enumerate all SPEF-shaped entries, not just the expected name.  This
    # rejects duplicate corners and symlink traps.  routed.spef_scenario is
    # metadata only and is deliberately not in this candidate set.
    candidates = sorted(p for p in raw_dir.iterdir() if p.name.endswith(".spef"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one raw corner SPEF, found {[p.name for p in candidates]}")
    source = candidates[0]
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("raw corner SPEF must be a nonempty regular nonsymlink file")
    match = RAW_NAME.fullmatch(source.name)
    if not match or float(match.group(1)) != 25.0:
        raise ValueError(f"wrong raw parasitic technology/corner/temperature identity: {source.name}")

    canonical = output_dir / CANONICAL_NAME
    if canonical.exists() or canonical.is_symlink():
        raise ValueError("canonical routed.spef existed before canonicalization")
    os.replace(str(source), str(canonical))
    if canonical.is_symlink() or not canonical.is_file() or canonical.stat().st_size <= 0:
        raise ValueError("canonical routed.spef missing after atomic rename")

    payload = {
        "schema": "m2133_icc2_corner_spef_canonicalization_r1_v1",
        "status": "PASS_M2133_UNIQUE_TT_CORNER_SPEF_CANONICALIZED",
        "raw_name": source.name,
        "canonical_name": CANONICAL_NAME,
        "parasitic_technology": "n28_1p9m_6x1z1u_typ",
        "corner": "tt_power",
        "temperature_c": 25.0,
        "candidate_count_before_rename": 1,
        "scenario_metadata_is_not_spef": True,
        "atomic_rename": True,
    }
    temporary = receipt.with_name(receipt.name + ".tmp")
    if receipt.exists() or receipt.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("canonicalization receipt target already exists")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(receipt))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = canonicalize(args.raw_dir, args.output_dir, args.receipt)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
