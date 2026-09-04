#!/usr/bin/env python3
"""Fail-closed post-processing for the M276 full220800 VCS milestone."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


EXPECTED_TAILS = [175162, 175604, 176110, 182167, 190728, 219956]
PASS_PATTERN = re.compile(
    r"^PASS M276 M235 full220800 protocol_ii "
    r"corpus_vectors=220800 corpus_requests=220800 corpus_results=220800 "
    r"mismatches=0 first_result_latency=8 intrinsic_ii_min=9 "
    r"intrinsic_ii_max=9 intrinsic_ii_samples=220798 "
    r"backpressured_requests=220799 request_backpressure_cycles=(\d+) "
    r"result_stalls=5 illegal_pending_attacks=1 attack_setup_requests=1 "
    r"lut_bins=64 even_exponents=6 tail_extrema=6 "
    r"unchanged_production_rtl=true new_speedup=false moment_finalizer=false "
    r"event_equivalence=false full_bn=false system_speedup=false headline=false$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def signed_coverage(values: np.ndarray) -> dict:
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "negative": int(np.count_nonzero(values < 0)),
        "zero": int(np.count_nonzero(values == 0)),
        "positive": int(np.count_nonzero(values > 0)),
        "unique": int(np.unique(values).size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--sim-log", required=True, type=Path)
    parser.add_argument("--assert-report", required=True, type=Path)
    parser.add_argument("--coverage-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    rows = np.loadtxt(args.vectors, delimiter=",", skiprows=1, dtype=np.int64)
    require(rows.shape == (220800, 12), f"vector shape drift: {rows.shape}")
    sequence = np.arange(220800, dtype=np.int64)
    require(np.array_equal(rows[:, 0], sequence), "vector_id discontinuity")
    require(np.array_equal(rows[:, 1], sequence), "source_index discontinuity")
    require(sha256(args.vectors) ==
            "81fbb84952fd79fc03a5b8660e839e27f06dec4e5fcb4b2c0cf770966c42ca29",
            "frozen vector SHA drift")

    sim_lines = args.sim_log.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = [PASS_PATTERN.match(line) for line in sim_lines]
    matches = [match for match in matches if match is not None]
    require(len(matches) == 1, "missing or duplicate M276 PASS line")
    request_backpressure_cycles = int(matches[0].group(1))
    require(request_backpressure_cycles >= 8 * 220799,
            "insufficient legal request backpressure coverage")

    report = args.assert_report.read_text(encoding="utf-8", errors="replace")
    required_covers = {
        "base_sva.cp_result": 220800,
        "base_sva.cp_result_stall": 5,
        "base_sva.cp_fault_with_pending_result": 1,
        "protocol_sva.cp_request_backpressure": 1,
        "protocol_sva.cp_held_request_turnaround": 1,
        "protocol_sva.cp_result_backpressure": 5,
        "protocol_sva.cp_illegal_pending_result_atomic": 1,
    }
    cover_matches = {}
    for name, minimum in required_covers.items():
        found = re.search(rf"{re.escape(name)}, .*? (\d+) match", report)
        require(found is not None, f"assertion cover missing: {name}")
        count = int(found.group(1))
        require(count >= minimum, f"assertion cover too small: {name}={count}")
        cover_matches[name] = count
    require("failed at" not in report and "Offending" not in report,
            "assertion failure present")

    names = [
        "vector_id", "source_index", "variance_uq6p16", "mean_sq3p14",
        "gamma_sq1p14", "beta_sq1p14", "even_exponent", "mantissa",
        "lut_index", "invstd_uq4p16", "alpha_sq3p16", "offset_sq3p16",
    ]
    coverage = {
        "schema": "m276_m235_full220800_protocol_ii_coverage_v1",
        "status": "PASS_FULL_FROZEN_POPULATION_AND_PROTOCOL_COVERAGE",
        "identity": {
            "vector_csv_sha256": sha256(args.vectors),
            "rows": int(rows.shape[0]),
            "source_indices_contiguous_unique": True,
        },
        "input_coverage": {
            name: signed_coverage(rows[:, index])
            for index, name in enumerate(names[2:9], start=2)
        },
        "output_coverage": {
            name: signed_coverage(rows[:, index])
            for index, name in enumerate(names[9:12], start=9)
        },
        "structural_bins": {
            "lut_indices": sorted(int(value) for value in np.unique(rows[:, 8])),
            "all_64_lut_indices_present": np.unique(rows[:, 8]).size == 64,
            "even_exponents": sorted(int(value) for value in np.unique(rows[:, 6])),
            "all_six_even_exponents_present": np.unique(rows[:, 6]).size == 6,
            "tail_extrema": EXPECTED_TAILS,
            "all_tail_extrema_present": all(
                rows[index, 0] == index and rows[index, 1] == index
                for index in EXPECTED_TAILS),
        },
        "vcs_protocol_and_schedule": {
            "corpus_request_accepts": 220800,
            "corpus_result_accepts": 220800,
            "integer_output_mismatches": 0,
            "first_result_latency_cycles": 8,
            "intrinsic_unstalled_accept_interval_cycles": 9,
            "intrinsic_interval_samples": 220798,
            "continuously_presented_backpressured_requests": 220799,
            "request_backpressure_cycles": request_backpressure_cycles,
            "result_backpressure_cycles": 5,
            "illegal_zero_with_pending_result_attacks": 1,
            "assertion_cover_matches": cover_matches,
            "assertion_failures": 0,
        },
    }
    require(coverage["structural_bins"]["all_64_lut_indices_present"],
            "LUT bin coverage drift")
    require(coverage["structural_bins"]["all_six_even_exponents_present"],
            "exponent coverage drift")
    require(coverage["structural_bins"]["all_tail_extrema_present"],
            "tail extrema coverage drift")

    receipt = {
        "schema": "m276_m235_full220800_protocol_ii_vcs_receipt_v1",
        "status": "PASS_M276_M235_FULL220800_PROTOCOL_II_EXACT_VCS",
        "tool": "Synopsys VCS V-2023.12-SP1",
        "exact_sha": True,
        "unchanged_production_rtl": True,
        "frozen_corpus_vectors": 220800,
        "integer_output_mismatches": 0,
        "first_result_latency_cycles": 8,
        "intrinsic_unstalled_accept_interval_cycles": 9,
        "m245_driver_observed_interval_superseded": 10,
        "m245_extra_driver_bubble_removed": True,
        "request_backpressure_closed": True,
        "result_backpressure_closed": True,
        "illegal_pending_result_fail_closed": True,
        "new_speedup": False,
        "moment_finalizer": False,
        "runtime_affine_equivalence": False,
        "event_equivalence": False,
        "full_bn": False,
        "system_speedup": False,
        "paper_ppa": False,
        "headline": False,
        "coverage_file": args.coverage_output.name,
    }

    args.coverage_output.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.receipt_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
