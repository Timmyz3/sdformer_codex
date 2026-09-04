#!/usr/bin/env python3
"""Fail-closed validator for the canonical M47-r1 conservative ledger."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = HW_ROOT / "contracts/m47_bit_tight_timestep_pair_single_buffer_contract_r1_20260823.json"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m47_bit_tight_timestep_pair_single_buffer.py"
RESULT = HW_ROOT / "results/m47_bit_tight_timestep_pair_single_buffer_r1_20260823/m47_bit_tight_timestep_pair_single_buffer.json"
EXPECTED = {
    "contract": "64319c18c4ac8d2bfe925c39e6d867638c803cbe2546f9781187881ec97c234d",
    "builder": "469fec2d14619d6d71c7103a48fe742233fbb54258813c661ba1f0fdb4110d7e",
    "result": "dc42df25567ad49be863586a3e287c9137a9f41470e83a5ef95bf125aa1734ed",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open("r") as handle:
        return json.load(handle)


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def find_config(result, name):
    matches = [entry for entry in result["configurations"]
               if entry.get("name") == name]
    require(len(matches) == 1, "configuration identity mismatch")
    return matches[0]


def validate(rerun):
    require(sha256_path(CONTRACT) == EXPECTED["contract"], "contract SHA mismatch")
    require(sha256_path(BUILDER) == EXPECTED["builder"], "builder SHA mismatch")
    require(sha256_path(RESULT) == EXPECTED["result"], "result SHA mismatch")
    contract = read_json(CONTRACT)
    result = read_json(RESULT)
    inputs = {}
    for name, identity in contract["inputs"].items():
        path = HW_ROOT / identity["path"]
        require(sha256_path(path) == identity["sha256"],
                "input SHA mismatch: {}".format(name))
        inputs[name] = read_json(path)

    require(result["identity"]["contract_sha256"] == EXPECTED["contract"],
            "embedded contract identity mismatch")
    require(result["identity"]["builder_sha256"] == EXPECTED["builder"],
            "embedded builder identity mismatch")
    require(result["identity"]["inputs_sha256"] == {
        name: identity["sha256"] for name, identity in contract["inputs"].items()
    }, "embedded input identities mismatch")

    capacity = result["capacity"]
    components = capacity["components"]
    independently_rebuilt_components = {
        "single_weight_tile_buffer_bytes": 24576,
        "bit_tight_parent_line_bytes": (20 * 96 * 19 + 7) // 8,
        "support_line_bytes": 20 * 256 // 8,
        "eight_contexts_bytes": 8 * ((96 * 19 + 7) // 8 + 64),
        "two_bit_tight_frames_bytes": 2 * ((300 * 96 * 19 + 7) // 8),
        "ready_frontier_bytes": 20 * 64,
        "response_metadata_fifo_bytes": 16 * 8,
        "complete_fifo_bytes": 16 * ((96 * 19 + 7) // 8 + 16),
    }
    require(components == independently_rebuilt_components,
            "capacity component reconstruction mismatch")
    rebuilt_capacity = sum(independently_rebuilt_components.values())
    require(rebuilt_capacity == 174224, "capacity arithmetic mismatch")
    require(capacity["combined_local_capacity_bytes"] == rebuilt_capacity,
            "reported capacity mismatch")
    require(capacity["frozen_local_residency_bytes"] == 193728 and
            capacity["local_capacity_headroom_bytes"] == 19504 and
            capacity["double_weight_buffer_permitted"] is False and
            capacity["external_accumulator_spill_permitted"] is False,
            "capacity admission mismatch")

    k2 = find_config(inputs["m45_r2_result"], "K2_CTX8_PRIMARY")
    expected_loads = 4 * 8 * 5 * 27
    expected_weight_bytes = expected_loads * 24576
    expected_load_cycles = expected_weight_bytes // 64
    traffic = result["weight_traffic"]
    require(expected_loads == 4320 and expected_weight_bytes == 106168320 and
            expected_load_cycles == 1658880,
            "independent weight load arithmetic mismatch")
    require(k2["traffic_bytes_per_sample"]["weight_dma"] == 212336640,
            "M45 inherited weight bytes mismatch")
    require(traffic["weight_tile_loads_per_sample"] == expected_loads and
            traffic["candidate_bytes_per_sample"] == expected_weight_bytes and
            traffic["m45_r2_bytes_per_sample"] == 2 * expected_weight_bytes and
            traffic["serialized_load_cycles_per_sample"] == expected_load_cycles and
            traffic["exact_two_x_reduction"] is True,
            "weight traffic result mismatch")

    inherited_by_id = {
        entry["sample_id"]: entry for entry in k2["per_sample"]
    }
    upper = result["conservative_cycle_upper_bound"]
    require(len(upper["per_sample"]) == 10, "cycle population mismatch")
    rebuilt_values = []
    for entry in upper["per_sample"]:
        inherited = inherited_by_id[entry["sample_id"]]
        rebuilt = inherited["integrated_cycles"] + expected_load_cycles
        require(entry["inherited_m45_k2_ctx8_integrated_cycles"] ==
                inherited["integrated_cycles"], "inherited cycle mismatch")
        require(entry["inherited_m45_weight_dma_wait_cycles_not_subtracted"] ==
                inherited["weight_dma_wait_cycles"], "wait-cycle disclosure mismatch")
        require(entry["serialized_single_buffer_weight_load_cycles_added"] ==
                expected_load_cycles, "serialized load mismatch")
        require(entry["conservative_integrated_cycle_upper_bound"] == rebuilt,
                "per-sample upper bound mismatch")
        rebuilt_values.append(rebuilt)
    require(sum(rebuilt_values) == 111636472 and
            upper["aggregate_cycles"] == sum(rebuilt_values),
            "aggregate upper bound mismatch")
    require(nearest_rank(rebuilt_values, 0.95) == 11340632 and
            upper["distribution"]["p95_nearest_rank"] == 11340632 and
            upper["distribution"]["p99_nearest_rank"] == 11340632,
            "tail upper bound mismatch")
    require(upper["p95_margin_to_gate_cycles"] == 15495075 - 11340632,
            "p95 headroom mismatch")
    require(upper["new_address_timed_pair_schedule_replayed"] is False,
            "upper bound was promoted to replayed schedule")

    model = result["conditional_frozen_compute_model"]
    denominator = 188824491 + 2636515 + 11340632
    require(denominator == 202801638 and
            model["conditional_total_cycles"] == denominator and
            model["conditional_compute_speedup"] == {
                "numerator": 620868243, "denominator": denominator} and
            620868243 >= 3 * denominator and
            model["three_x_crossing_in_conditional_model"] is True and
            model["system_or_end_to_end_speedup_admitted"] is False,
            "conditional compute-model gate mismatch")

    admission = result["admission"]
    require(admission == {
        "bit_exact_capacity_admitted": True,
        "weight_traffic_reduction_admitted": True,
        "conservative_cycle_upper_bound_admitted": True,
        "conditional_frozen_compute_three_x_admitted": True,
        "new_address_timed_pair_schedule_admitted": False,
        "rtl_vcs_synopsys_admitted": False,
        "full_network_or_system_speedup_admitted": False,
        "ppa_power_energy_admitted": False,
        "date_headline_or_best_paper_admitted": False,
    }, "claim admission mismatch")
    require(result["claim_policy"] == contract["claim_policy"],
            "claim policy mismatch")

    if rerun:
        with tempfile.TemporaryDirectory(prefix="m47_validate_") as directory:
            output = Path(directory) / "rebuilt.json"
            subprocess.check_call([
                sys.executable, str(BUILDER), "--output", str(output)
            ])
            require(sha256_path(output) == EXPECTED["result"],
                    "deterministic rerun SHA mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    validate(args.rerun)
    print("PASS M47-r1 fail-closed validation{}".format(
        " with deterministic rerun" if args.rerun else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("FAIL M47 validation: {}".format(error), file=sys.stderr)
        sys.exit(1)
