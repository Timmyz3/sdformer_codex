#!/usr/bin/env python3
"""Fail-closed cycle A/B between exact M212 and M214 VCS sweeps."""

import argparse
import collections
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"M(212|214)TAIL blocks=(\d+) mode=(\d+) seed=(\d+) "
    r"descriptors=(\d+) measured=(\d+)")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse(path, expected_design):
    records = {}
    for match in PATTERN.finditer(Path(path).read_text()):
        design, blocks, mode, seed, descriptors, cycles = match.groups()
        require(design == expected_design, "unexpected design in sweep")
        key = (int(blocks), int(mode), int(seed))
        require(key not in records, "duplicate sweep key")
        records[key] = {
            "descriptors": int(descriptors), "cycles": int(cycles)}
    require(len(records) == 256, "sweep extent drift")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m212-log", required=True, type=Path)
    parser.add_argument("--m214-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    m212 = parse(args.m212_log, "212")
    m214 = parse(args.m214_log, "214")
    require(m212.keys() == m214.keys(), "A/B key drift")
    histogram = collections.Counter()
    by_blocks = collections.defaultdict(collections.Counter)
    improved = []
    for key in sorted(m212):
        require(m212[key]["descriptors"] == m214[key]["descriptors"],
                "descriptor identity drift")
        saved = m212[key]["cycles"] - m214[key]["cycles"]
        require(saved >= 0, "M214 cycle regression")
        histogram[saved] += 1
        by_blocks[str(key[0])][saved] += 1
        if saved:
            improved.append({
                "output_blocks": key[0], "mode": key[1], "seed": key[2],
                "descriptors": m212[key]["descriptors"],
                "m212_cycles": m212[key]["cycles"],
                "m214_cycles": m214[key]["cycles"],
                "cycles_saved": saved,
            })
    result = {
        "schema": "m214_m212_exact_vcs_tail_cycle_ab_v1",
        "status": "PASS_NO_REGRESSION",
        "cases": 256,
        "improved_cases": len(improved),
        "unchanged_cases": histogram[0],
        "regressed_cases": 0,
        "total_cycles_saved_across_sweep": sum(
            item["cycles_saved"] for item in improved),
        "cycle_saving_histogram": {
            str(key): value for key, value in sorted(histogram.items())},
        "per_output_blocks_histogram": {
            key: {str(delta): count for delta, count in sorted(value.items())}
            for key, value in sorted(by_blocks.items())},
        "improved_records": improved,
        "claim_boundary": {
            "verification_sweep_only": True,
            "frozen_h67": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "cases", "improved_cases", "unchanged_cases",
        "regressed_cases", "total_cycles_saved_across_sweep")},
        sort_keys=True))


if __name__ == "__main__":
    main()
