#!/usr/bin/env python3
"""Validate the exact-SHA M65 r2 release and independently recompute it."""

from __future__ import print_function

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RELEASE_SHA256 = \
    "cba46273de617bbb4f28f13baf0adc3999ca2c465a9f2308c863f95ca5213185"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("invalid JSON constant " + raw)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m65_release_independent", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import independent validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    require(not args.receipt.exists(), "refusing to overwrite release receipt")
    require(sha256(args.release) == EXPECTED_RELEASE_SHA256,
            "release contract SHA drift")
    release = strict_json(args.release)
    require(release["schema"] == "m65_exact_sha_release_contract_v2" and
            release["status"] == "FROZEN_M65_ARITHMETIC_AND_DUAL_NO_GO_RELEASE",
            "release schema/status drift")
    observed = {}
    for entry in release["entries"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = HW_ROOT / path
        require(path.is_file() and not path.is_symlink(),
                "release entry missing/symlink: " + entry["path"])
        actual = sha256(path)
        require(actual == entry["sha256"],
                "release entry SHA drift: " + entry["path"])
        observed[entry["path"]] = actual
    require(len(observed) == len(release["entries"]),
            "duplicate release entry path")

    review_validator_path = HW_ROOT / (
        "reviews/m65_independent_hammer_r1_20260823/"
        "validate_m65_independent_hammer_review.py")
    independent = load_module(review_validator_path)
    contract = independent.strict_json(HW_ROOT / independent.CONTRACT_REL)
    result = independent.strict_json(HW_ROOT / independent.RESULT_REL)
    m25 = independent.strict_json(HW_ROOT / independent.M25_REL)
    m39 = independent.strict_json(HW_ROOT / independent.M39_REL)
    m53 = independent.strict_json(HW_ROOT / independent.M53_REL)
    m63 = independent.strict_json(HW_ROOT / independent.M63_REL)
    m4 = independent.strict_json(independent.M4_STATEFUL)
    summary = independent.validate_semantics(
        contract, result, m25, m39, m53, m63, m4)
    require(not independent.independent_guard(contract, result),
            "independent semantic guard rejected frozen result")

    conclusion = release["frozen_conclusion"]
    require(conclusion == {
        "exact_m4_speed_numerator": 5158877,
        "exact_m4_speed_denominator": 860504,
        "captured_linear_inherited_cycle_interval": [25792603, 25792604],
        "spatial_k4_joint_cycle_interval": [204002475, 204002476],
        "spatial_k4_conditional_ratio_interval_not_system_speedup": [
            3.0434348404673286, 3.0434348553859456],
        "spatial_k4_regression_cycle_interval": [2742965, 2742966],
        "spatial_k4_decision": "NO_GO_AS_ADDITIVE_M53_ACCELERATOR",
        "temporal_k4_capacity_infeasible_modules": 11,
        "temporal_k4_decision": "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES",
    }, "frozen conclusion drift")
    boundary = release["claim_boundary"]
    require(boundary["headline"] is False and
            boundary["system_speedup_admitted"] is False and
            boundary["paper_ppa_ready"] is False and
            boundary["power_or_energy_admitted"] is False,
            "release claim promotion")

    receipt = {
        "schema": "m65_exact_sha_release_validation_receipt_v2",
        "status": "PASS_EXACT_SHA_INDEPENDENT_RECOMPUTE_DUAL_NO_GO",
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "release_contract_sha256": EXPECTED_RELEASE_SHA256,
        "release_validator_sha256": sha256(Path(__file__).resolve()),
        "entries_sha256": observed,
        "exact_speed": [summary["exact"]["speed_numerator"],
                        summary["exact"]["speed_denominator"]],
        "joint_cycle_interval": summary["joint_cycles"],
        "joint_ratio_interval_not_system_speedup": summary["joint_ratios"],
        "replacement_regression_interval": summary["regression"],
        "spatial_k4_decision": conclusion["spatial_k4_decision"],
        "temporal_k4_decision": conclusion["temporal_k4_decision"],
        "claim_boundary": boundary,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("PASS M65 exact-SHA r2 release receipt_sha256={}".format(
        sha256(args.receipt)))


if __name__ == "__main__":
    main()
