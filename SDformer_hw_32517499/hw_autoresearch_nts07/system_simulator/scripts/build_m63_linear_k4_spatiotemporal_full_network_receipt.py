#!/usr/bin/env python3
"""Build a durable validator-backed M63 producer receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("nonstandard JSON {}".format(raw))

    def pairs(raw):
        result = {}
        for key, value in raw:
            require(key not in result, "duplicate key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    for name in ("validator", "analyzer", "tamper-runner", "contract",
                 "manifest", "m52-result", "m53-result", "m55-result",
                 "m39-result", "operator-transactions", "dual-line-contract",
                 "result", "tamper-receipt", "stale-result", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "refusing existing receipt")
    result_sha = sha256_path(arguments.result)
    argv = [
        arguments.python, str(arguments.validator),
        "--contract", str(arguments.contract),
        "--analyzer", str(arguments.analyzer),
        "--manifest", str(arguments.manifest),
        "--m52-result", str(arguments.m52_result),
        "--m53-result", str(arguments.m53_result),
        "--m55-result", str(arguments.m55_result),
        "--m39-result", str(arguments.m39_result),
        "--operator-transactions", str(arguments.operator_transactions),
        "--dual-line-contract", str(arguments.dual_line_contract),
        "--result", str(arguments.result),
        "--expected-result-sha256", result_sha,
    ]
    completed = subprocess.run(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
    require(completed.returncode == 0, "canonical validator failed")
    stdout_json = json.loads(completed.stdout)
    result = strict_json(arguments.result)
    tamper = strict_json(arguments.tamper_receipt)
    require(tamper["status"] == "PASS_ALL_SEMANTIC_TAMPERS_REJECTED" and
            tamper["attack_count"] == tamper["rejected_count"] == 22,
            "tamper receipt")
    spatial = result["aggregate_configurations"]["spatial_K4"]
    temporal = result["aggregate_configurations"]["temporal_K4"]
    receipt = {
        "schema": "m63_linear_k4_spatiotemporal_full_network_validation_receipt_v1",
        "status": "PASS_ALL24_LINEAR_OPPORTUNITY_ONLY_TEMPORAL_AND_JOINT_RATIO_KILLED",
        "claim": "ALL24_LINEAR_BANK_EXECUTABLE_OPPORTUNITY_NOT_RTL_NUMERIC_OR_SYSTEM_SPEEDUP",
        "sources": {
            "analyzer_sha256": sha256_path(arguments.analyzer),
            "validator_sha256": sha256_path(arguments.validator),
            "tamper_runner_sha256": sha256_path(arguments.tamper_runner),
            "contract_sha256": sha256_path(arguments.contract),
            "manifest_sha256": sha256_path(arguments.manifest),
            "m52_result_sha256": sha256_path(arguments.m52_result),
            "m53_result_sha256": sha256_path(arguments.m53_result),
            "m55_result_sha256": sha256_path(arguments.m55_result),
            "m39_result_sha256": sha256_path(arguments.m39_result),
            "operator_transactions_sha256":
                sha256_path(arguments.operator_transactions),
            "dual_line_contract_sha256":
                sha256_path(arguments.dual_line_contract),
        },
        "artifacts": {
            arguments.result.name: {
                "bytes": arguments.result.stat().st_size,
                "sha256": result_sha,
            },
            arguments.tamper_receipt.name: {
                "bytes": arguments.tamper_receipt.stat().st_size,
                "sha256": sha256_path(arguments.tamper_receipt),
            },
        },
        "canonical_validator": {
            "argv": argv,
            "exit_code": completed.returncode,
            "stdout_json": stdout_json,
            "stderr": completed.stderr,
        },
        "population": result["population"],
        "m39_category_cycles": result["m39_category_ledger"],
        "headline": {
            "spatial_k4_source_p95_cycles": spatial[
                "source_cycle_distribution"]["p95_nearest_rank"],
            "spatial_k4_serialized_p95_cycles": spatial[
                "serialized_integrated_cycle_distribution"][
                    "p95_nearest_rank"],
            "spatial_k1_over_k4_source_not_system_speedup": spatial[
                "ratios_not_system_speedup"]["k1_over_k_source_issue"],
            "spatial_k1_over_k4_serialized_not_system_speedup": spatial[
                "ratios_not_system_speedup"]["k1_over_k_serialized_integrated"],
            "temporal_k4_source_p95_cycles": temporal[
                "source_cycle_distribution"]["p95_nearest_rank"],
            "temporal_k4_serialized_p95_cycles": temporal[
                "serialized_integrated_cycle_distribution"][
                    "p95_nearest_rank"],
            "temporal_k1_over_k4_source_not_system_speedup": temporal[
                "ratios_not_system_speedup"]["k1_over_k_source_issue"],
            "temporal_k1_over_k4_serialized_not_system_speedup": temporal[
                "ratios_not_system_speedup"]["k1_over_k_serialized_integrated"],
        },
        "amdahl": result["m39_amdahl"],
        "m53_overlap_reconciliation": result["m53_overlap_reconciliation"],
        "kill_gates": result["kill_gates"],
        "claim_boundary": result["claim_boundary"],
        "tamper": {
            "attack_count": 22,
            "rejected_count": 22,
            "receipt_sha256": sha256_path(arguments.tamper_receipt),
            "status": tamper["status"],
        },
        "stale_artifact": {
            "path": arguments.stale_result.name,
            "sha256": sha256_path(arguments.stale_result),
            "status": "STALE_PRE_M53_OVERLAP_GUARD_DO_NOT_CITE",
            "reason": "superseded before exact M53 outside-term overlap guard and temporal capacity kill gate were frozen",
        },
        "method": {
            "raw_a800_payload_read_only": True,
            "open_source_hdl_used": False,
            "synopsys_hdl_launched": False,
            "producer_not_self_review": True,
            "system_speedup_admitted": False,
        },
    }
    temporary = arguments.output.with_name(
        arguments.output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(arguments.output))
    temporary.unlink()
    print(json.dumps({"output_sha256": sha256_path(arguments.output),
                      "status": receipt["status"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
