#!/usr/bin/env python3
"""Fail-closed validator for the M45-r3 scheduler-state capacity repair."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m45_r3_scheduler_state_capacity_repair_contract_r1_20260823.json")
BUILDER = HW_ROOT / (
    "system_simulator/scripts/build_m45_r3_scheduler_state_capacity_repair.py")
DEFAULT_RESULT = HW_ROOT / (
    "results/m45_scheduler_state_capacity_repair_r3_20260823/"
    "m45_r3_scheduler_state_capacity_repair.json")
EXPECTED_CONTRACT_SHA256 = (
    "466ed800cf4bb5258573a30e1df4f3165a1215e5cc034f55c9f1ef7cd890961e")
EXPECTED_BUILDER_SHA256 = (
    "91a38fb867d74b52e288399e410b6ba22ca522ff6b54000a239670b8c78220b5")
EXPECTED_RESULT_SHA256 = (
    "4e3764f58b5c8b893e9d5b71b6a27adca582aaac43378cfc610e6e1010a0ce72")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def validate(path, require_canonical_sha=True):
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M45-r3 contract SHA drift")
    require(sha256(BUILDER) == EXPECTED_BUILDER_SHA256,
            "M45-r3 builder SHA drift")
    if require_canonical_sha:
        require(sha256(path) == EXPECTED_RESULT_SHA256,
                "M45-r3 canonical result SHA drift")
    contract = read_json(CONTRACT)
    for name, item in contract["inputs"].items():
        source = HW_ROOT / item["path"]
        require(source.is_file() and sha256(source) == item["sha256"],
                "M45-r3 input drift: {}".format(name))
    result = read_json(path)
    require(result["schema"] ==
            "m45_r3_scheduler_state_capacity_repair_result_v1" and
            result["status"] ==
            "PASS_M45_R3_LEDGER_ONLY_CAPACITY_REPAIR_RTL_SYSTEM_UNADMITTED",
            "M45-r3 result schema/status drift")
    require(result["identity"]["contract_sha256"] ==
            EXPECTED_CONTRACT_SHA256 and
            result["identity"]["builder_sha256"] ==
            EXPECTED_BUILDER_SHA256,
            "M45-r3 embedded identity drift")
    repair = result["repair_scope"]
    require(all(repair[name] is False for name in (
        "transaction_schedule_changed", "ready_selection_changed",
        "command_timing_changed", "descriptor_backpressure_added",
        "all10_schedule_rerun_required")),
        "M45-r3 improperly changes the frozen transaction schedule")
    frontier = result["spatial_frontier_proof"]
    require(frontier["width"] == frontier["maximum_raw_ready_entries"] == 20 and
            frontier["height"] == 15 and
            frontier["tasks_per_tile_timestep"] == 300 and
            frontier["mandatory_up_edge_checked_for_parent_policies"] == 3,
            "M45-r3 frontier proof drift")
    require(result["targeted_raw_ready_entries"] ==
            [20, 20, 19, 20, 20, 20, 19, 20],
            "M45-r3 targeted raw-ready evidence drift")
    capacity = result["capacity"]
    require(capacity["ready_descriptor_frontier_bytes"] == 1280 and
            capacity["response_metadata_fifo_bytes"] == 128 and
            capacity["complete_fifo_bytes_unchanged"] == 4864 and
            capacity["scheduler_and_fifo_bytes"] == 6272 and
            capacity["capacity_delta_bytes_vs_r2"] == 384 and
            capacity["combined_local_capacity_bytes"] == 151040 and
            capacity["local_capacity_headroom_bytes"] == 42688 and
            capacity["combined_local_capacity_bytes"] +
            capacity["local_capacity_headroom_bytes"] == 193728,
            "M45-r3 capacity arithmetic drift")
    inherited = result["inherited_transaction_identity"]
    require(inherited == {
        "population": {"samples": 10, "operators": 4, "records": 40},
        "k1_aggregate_integrated_cycles": 122418024,
        "k2_ctx8_aggregate_source_only_cycles": 88269520,
        "k2_ctx8_aggregate_integrated_cycles": 95047672,
        "k2_ctx8_p95_integrated_cycles": 9681752,
        "k2_ctx4_p95_integrated_cycles": 10101880,
        "k4_ctx4_p95_integrated_cycles": 10535896,
        "all_r2_transaction_kill_gates_pass": True},
        "M45-r3 inherited transaction metrics drift")
    admission = result["admission"]
    require(admission["transaction_cycles_unchanged"] is True and
            admission["scheduler_state_capacity_repaired"] is True and
            admission["nominal_local_headroom_admitted_at_transaction_level"]
            is True and
            admission["rtl_atomic_push2_or_response_fifo_occupancy_proved"]
            is False and
            admission["rtl_ppa_system_or_three_x_admitted"] is False,
            "M45-r3 admission boundary drift")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    canonical = validate(args.result)
    if args.rerun:
        with tempfile.TemporaryDirectory(prefix="m45_r3_validate_") as tempdir:
            rerun = Path(tempdir) / "rerun.json"
            subprocess.check_call([sys.executable, str(BUILDER),
                                   "--output", str(rerun)])
            observed = validate(rerun, require_canonical_sha=False)
            require(observed == canonical,
                    "M45-r3 deterministic rerun mismatch")
    print("PASS M45-r3 {}".format(args.result))


if __name__ == "__main__":
    main()
