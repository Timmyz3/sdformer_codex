#!/usr/bin/env python3
"""Fail-closed validator for the independent M45-r2 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
RESULT_DIR = HW_ROOT / (
    "results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823")
REVIEW = RESULT_DIR / "m45_r2_independent_hammer_review.json"
REVIEWER = HW_ROOT / (
    "system_simulator/scripts/review_m45_r2_independent_hammer.py")
TARGETED_REPLAY = RESULT_DIR / "m45_r2_independent_targeted_replay_samples3_7.json"
CANONICAL = RESULT_DIR / "m45_r2_context8_primary_schedule.json"

EXPECTED_REVIEW_SHA256 = (
    "cc0110cd9a8e084adf2c6e58224a2a3f52144608c96be3f65bde132a4921d6a8")
EXPECTED_REVIEWER_SHA256 = (
    "8a411aaf9cf707681298c93ea3eb9140506c972bd0f1f4af205705718292460d")
EXPECTED_REPLAY_SHA256 = (
    "3bc28c1cb06cf27e5497c52205e6046ec7bc09b0f5cec537f8876a6500641f47")
EXPECTED_CANONICAL_SHA256 = (
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


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args()
    require(sha256(args.review) == EXPECTED_REVIEW_SHA256,
            "independent review SHA drift")
    require(sha256(REVIEWER) == EXPECTED_REVIEWER_SHA256,
            "independent reviewer SHA drift")
    require(sha256(TARGETED_REPLAY) == EXPECTED_REPLAY_SHA256,
            "targeted replay SHA drift")
    require(sha256(CANONICAL) == EXPECTED_CANONICAL_SHA256,
            "canonical result SHA drift")
    spec = importlib.util.spec_from_file_location(
        "m45_r2_independent_reviewer_validate", str(REVIEWER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    review = strict_json(args.review)
    require(module.build() == review,
            "independent review does not reproduce from frozen inputs")
    require(review["status"] ==
            "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
            "review decision drift")
    require(review["review"] == {
        "decision": "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
        "score_0_to_100": 86, "p0": 0, "p1": 1, "p2": 5},
        "review score/finding count drift")
    require(review["targeted_replay"]["maximum_raw_spatial_dag_ready_depth"] == 20,
            "raw ready depth drift")
    require(review["capacity"]["nominal_headroom_admitted_after_r2"] is False and
            review["repair_gate"]["r3_required"] is True,
            "metadata capacity disposition drift")
    print("PASS M45-r2 independent hammer {}".format(args.review))


if __name__ == "__main__":
    main()
