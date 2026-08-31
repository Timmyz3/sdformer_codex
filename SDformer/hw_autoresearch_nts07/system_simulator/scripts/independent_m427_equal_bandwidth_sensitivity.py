#!/usr/bin/env python3
"""M427 optimistic equal-bandwidth sensitivity for the strong zero baseline."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROWS = 3000
PHASES = 17280
PER_SAMPLE = 1728
EXPECTED_ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(block, lut):
    require(len(block) == ROWS * 9, "phase transport truncation")
    raw = np.frombuffer(block, dtype=np.uint8).reshape(ROWS, 9)
    require(bool(np.all(raw[:, 8] == 10)), "newline drift")
    digits = lut[raw[:, :8]]
    require(not bool(np.any(digits == 255)), "non-hex digit")
    words = np.zeros(ROWS, dtype=np.uint32)
    for column in range(8):
        words = (words << np.uint32(4)) | digits[:, column].astype(np.uint32)
    require(not bool(np.any(words >> np.uint32(29))), "reserved-bit drift")
    return words


def advance(start, compute, first, last):
    preprocess = 3005
    initial = preprocess if first else 0
    exposed = 0 if last else max(0, preprocess - compute)
    end = start + initial + compute + exposed + 2 + (96000 if last else 0)
    return end, initial, exposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing sensitivity overwrite")
    require(args.rows.is_file() and not args.rows.is_symlink(),
            "missing/symlink row transport")
    require(sha256(args.rows) == EXPECTED_ROWS_SHA, "row SHA drift")
    pop16 = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                       dtype=np.uint8)
    lut = np.full(256, 255, dtype=np.uint8)
    for value, digit in zip(b"0123456789abcdef", range(16)):
        lut[value] = digit
    times = {1: 0, 2: 0, 3: 0}
    compute_totals = {1: 0, 2: 0, 3: 0}
    initial_totals = {1: 0, 2: 0, 3: 0}
    exposed_totals = {1: 0, 2: 0, 3: 0}
    minimum_compute = {1: None, 2: None, 3: None}
    with args.rows.open("rb") as handle:
        for phase in range(PHASES):
            words = decode(handle.read(ROWS * 9), lut)
            population = pop16[(words & np.uint32(0xffff)).astype(np.uint16)]
            first = phase % PER_SAMPLE == 0
            last = phase % PER_SAMPLE == PER_SAMPLE - 1
            for issue_width in (1, 2, 3):
                compute = int(((population.astype(np.int64) +
                                issue_width - 1) // issue_width).sum()) * 8
                times[issue_width], initial, exposed = advance(
                    times[issue_width], compute, first, last)
                compute_totals[issue_width] += compute
                initial_totals[issue_width] += initial
                exposed_totals[issue_width] += exposed
                minimum_compute[issue_width] = (
                    compute if minimum_compute[issue_width] is None else
                    min(minimum_compute[issue_width], compute))
        require(handle.read(1) == b"", "transport trailing bytes")
    require(times[1] == 742148386, "K1 strong-baseline reproduction drift")
    candidates = {"current": 641790704, "dualbank": 530606660,
                  "seed_fused": 437640532}
    variants = {}
    for width in (1, 2, 3):
        variants["K{}_zero".format(width)] = {
            "optimistic_source_vectors_per_cycle": width,
            "peak_weight_read_bytes_per_cycle": 96 * width,
            "cycles": times[width],
            "compute_cycles": compute_totals[width],
            "initial_preprocess_cycles": initial_totals[width],
            "next_preprocess_exposed_cycles": exposed_totals[width],
            "tail_cycles": PHASES * 2,
            "commit_cycles": 10 * 96000,
            "minimum_phase_compute_cycles": minimum_compute[width],
            "candidate_speedups": {
                name: times[width] / cycles for name, cycles in candidates.items()
            },
        }
        require(sum((compute_totals[width], initial_totals[width],
                     exposed_totals[width], PHASES * 2, 10 * 96000)) ==
                times[width], "K{} component conservation".format(width))
    result = {
        "schema": "m427_equal_bandwidth_zero_sensitivity_v1",
        "status": "PASS_OPTIMISTIC_RESOURCE_SENSITIVITY",
        "input_rows_sha256": EXPECTED_ROWS_SHA,
        "model": {
            "K1": "current strong zero, one 96-byte source vector/cycle",
            "K2": "optimistic ceil(pop/2) issue, two 96-byte vectors/cycle and a two-source merge datapath",
            "K3": "optimistic ceil(pop/3) issue, three 96-byte vectors/cycle and a three-source merge datapath",
            "scope": "four frozen H67 bottleneck Conv3x3 operators",
            "not_modeled": ["extra adder/mux area", "bank conflicts",
                            "wire timing", "SRAM macro timing", "power"],
        },
        "m426_peak_reads": {
            "dualbank_pwp_logical_bytes_per_cycle": 144,
            "seed_fused_first_positive_residual_logical_bytes_per_cycle": 240,
            "existing_m405_high_port_padding_means_physical_input_signal_bytes":
                {"dualbank": 160, "seed_fused_first_positive_residual": 256},
        },
        "variants": variants,
        "interpretation": {
            "K2_is_lower_peak_bandwidth_than_seed_fused": True,
            "K2_exceeds_dualbank_logical_peak_bandwidth": True,
            "K3_meets_or_exceeds_seed_fused_logical_peak_bandwidth": True,
            "primary_1p6958x_resource_normalized": False,
            "required_presentation": "throughput-area Pareto until RTL plus macro/interconnect PPA",
            "weak_dense_12p507x_15p164x_secondary_only": True,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M427_BW_PASS K1={} K2={} K3={} fused_vs_K2={:.9f} "
          "fused_vs_K3={:.9f}".format(
              times[1], times[2], times[3],
              times[2] / candidates["seed_fused"],
              times[3] / candidates["seed_fused"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
