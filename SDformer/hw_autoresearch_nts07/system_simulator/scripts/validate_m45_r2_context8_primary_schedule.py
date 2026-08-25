#!/usr/bin/env python3
"""Fail-closed validator for the frozen M45-r2 all10 result."""

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
    "contracts/m45_dual_destination_bank_fused_integrated_schedule_contract_r2_20260823.json")
ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_r2_context8_primary_schedule.py")
DEFAULT_RESULT = HW_ROOT / (
    "results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823/"
    "m45_r2_context8_primary_schedule.json")
EXPECTED_CONTRACT_SHA256 = (
    "1c547c3ecd5d82c5dc8217297f19ca730748ac9526663f5449d8f13d867cd6b4")
EXPECTED_ANALYZER_SHA256 = (
    "1b07e6efea778561605f7a89d03505c3610ec96c19c21b278c347c2cf8d90885")
EXPECTED_RESULT_SHA256 = (
    "0f16e75601fdb18f31f9bc36f6aae8a17a9e62a20f5c07e18226562e9ba0d37c")


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


def validate_result(path, require_frozen_sha=True):
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M45-r2 validator contract SHA drift")
    require(sha256(ANALYZER) == EXPECTED_ANALYZER_SHA256,
            "M45-r2 validator analyzer SHA drift")
    if require_frozen_sha:
        require(sha256(path) == EXPECTED_RESULT_SHA256,
                "M45-r2 canonical result SHA drift")
    result = read_json(path)
    require(result["schema"] ==
            "m45_dual_destination_bank_fused_integrated_schedule_result_v2",
            "M45-r2 result schema drift")
    require(result["status"] ==
            "PASS_M45_R2_TRANSACTION_GATES_RTL_AND_SYSTEM_UNADMITTED",
            "M45-r2 result is not transaction-level GO")
    require(result["identity"]["contract_sha256"] ==
            EXPECTED_CONTRACT_SHA256 and
            result["identity"]["analyzer_sha256"] ==
            EXPECTED_ANALYZER_SHA256,
            "M45-r2 embedded identity drift")
    require(result["population"] ==
            {"samples": 10, "operators": 4, "records": 40},
            "M45-r2 population drift")
    capacity = result["capacity"]
    require(capacity["combined_local_capacity_bytes"] == 150656 and
            capacity["local_capacity_headroom_bytes"] == 43072 and
            capacity["extra_state_bytes_vs_four_contexts"] == 1408,
            "M45-r2 result capacity drift")
    by_name = dict((item["name"], item)
                   for item in result["configurations"])
    require(set(by_name) == set((
        "K1_CTX4_REPRODUCTION", "K2_CTX8_PRIMARY",
        "K2_CTX4_CAPACITY_ABLATION", "K4_CTX4_KILLED_ABLATION")),
        "M45-r2 configuration set drift")
    require(by_name["K1_CTX4_REPRODUCTION"][
        "aggregate_source_only_cycles"] == 116376872,
        "M45-r2 K1 reproduction drift")
    for item in by_name.values():
        require(len(item["per_sample"]) == 10 and len(item["records"]) == 40,
                "M45-r2 per-sample/record population drift")
        require(item["aggregate_source_only_cycles"] == sum(
            row["source_only_cycles"] for row in item["per_sample"]),
            "M45-r2 aggregate source reconciliation drift")
        require(item["aggregate_integrated_cycles"] == sum(
            row["integrated_cycles"] for row in item["per_sample"]),
            "M45-r2 aggregate integrated reconciliation drift")
    primary = by_name["K2_CTX8_PRIMARY"]
    for sample in primary["per_sample"]:
        require((sample["integrated_cycles"] - sample["source_only_cycles"]) *
                10 <= sample["source_only_cycles"],
                "M45-r2 per-sample primary overhead gate failure")
        require(sample["parent_wait_cycles"] * 20 <=
                sample["integrated_cycles"],
                "M45-r2 per-sample parent-wait gate failure")
        require(sample["maximum_resident_occupancy"] <= 8 and
                sample["maximum_complete_occupancy"] <= 16 and
                sample["maximum_metadata_occupancy"] <= 16,
                "M45-r2 physical occupancy overflow")
    require(all(result["kill_gates"][name] for name in (
        "primary_all_samples_integrated_over_source_only_le_10pct",
        "primary_all_samples_parent_wait_le_5pct",
        "primary_aggregate_integrated_reduction_vs_k1_ge_15pct",
        "ctx8_p95_improvement_over_ctx4_ge_3pct",
        "primary_p95_integrated_cycles_le_15495075",
        "k4_ctx4_slower_than_k2_ctx8_and_killed",
        "all_kill_gates_pass")),
        "M45-r2 one or more frozen gates failed")
    require(result["kill_gates"]["three_x_target_crossing_admitted"] is False,
            "M45-r2 improperly admits a 3x/system claim")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    canonical = validate_result(args.result)
    if args.rerun:
        with tempfile.TemporaryDirectory(prefix="m45_r2_validate_") as tempdir:
            rerun = Path(tempdir) / "rerun.json"
            subprocess.check_call([sys.executable, str(ANALYZER),
                                   "--output", str(rerun)])
            validate_result(rerun, require_frozen_sha=False)
            require(read_json(rerun) == canonical,
                    "M45-r2 deterministic rerun mismatch")
    print("PASS M45-r2 {}".format(args.result))


if __name__ == "__main__":
    main()
