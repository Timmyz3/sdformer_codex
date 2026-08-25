#!/usr/bin/env python3
"""Validate the M65 non-overlap result and its fail-closed identity chain."""

import argparse
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    require(result["schema"] == "m65_m53_m63_nonoverlap_joint_result_v1",
            "result schema drift")
    require(result["status"] == "PASS_ONE_CYCLE_TIGHT_NONOVERLAP_SPATIAL_K4_NO_GO",
            "result status drift")
    require(result["identity"]["contract_sha256"] == sha256(args.contract),
            "contract identity drift")
    require(result["identity"]["analyzer_sha256"] == sha256(args.analyzer),
            "analyzer identity drift")
    require(result["reconstruction"]["m53_conditional_cycles"] == 201259510,
            "M53 denominator drift")
    captured = result["captured_linear_decomposition"]
    require(captured["modules"] == 24 and captured["raw_eligible_cycles"] == 154631318,
            "captured Linear population drift")
    require(captured["inherited_integral_interval"] == {
        "minimum": 25792603, "maximum": 25792604, "width_cycles": 1},
        "inherited interval drift")
    joint = result["spatial_k4_nonoverlap_joint"]
    require(joint["replacement_p95_cycles"] == 28535569,
            "replacement drift")
    require(joint["joint_conditional_cycle_interval"] == {
        "minimum": 204002475, "maximum": 204002476}, "joint interval drift")
    require(joint["replacement_regression_cycles_interval"] == {
        "minimum": 2742965, "maximum": 2742966}, "regression interval drift")
    require(joint["decision"] == "NO_GO_AS_ADDITIVE_M53_ACCELERATOR",
            "NO-GO decision drift")
    require(result["temporal_k4"] == {
        "all24_joint_admitted": False,
        "capacity_infeasible_modules": 11,
        "decision": "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES"},
        "temporal kill drift")
    policy = result["claim_boundary"]
    require(policy == contract["claim_policy"], "claim policy drift")
    require(policy["headline"] is False and
            policy["system_speedup_admitted"] is False and
            policy["paper_ppa_ready"] is False and
            policy["power_or_energy_admitted"] is False,
            "claim boundary opened")
    print("PASS M65 validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
