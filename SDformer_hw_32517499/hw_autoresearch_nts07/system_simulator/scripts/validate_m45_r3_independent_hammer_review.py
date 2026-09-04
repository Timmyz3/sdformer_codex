#!/usr/bin/env python3
"""Fail-closed validator for the independent M45-r3 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEWER = HW_ROOT / (
    "system_simulator/scripts/review_m45_r3_scheduler_state_capacity_repair.py")
REVIEW = HW_ROOT / (
    "results/m45_scheduler_state_capacity_repair_r3_independent_hammer_20260823/"
    "m45_r3_independent_hammer_review.json")
CANONICAL = HW_ROOT / (
    "results/m45_scheduler_state_capacity_repair_r3_20260823/"
    "m45_r3_scheduler_state_capacity_repair.json")

EXPECTED_REVIEWER_SHA256 = (
    "dabf4f3a7294c24dae3b1aebcd3758c0abdd1c085b8ba2d3ba7b22fa19156af0")
EXPECTED_REVIEW_SHA256 = (
    "cd95b19adb7610e134fc64a85ab842a4f3ec6b96d8a92bd9295e2db522101938")
EXPECTED_CANONICAL_SHA256 = (
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
    require(sha256(REVIEWER) == EXPECTED_REVIEWER_SHA256,
            "independent reviewer SHA drift")
    require(sha256(args.review) == EXPECTED_REVIEW_SHA256,
            "independent review SHA drift")
    require(sha256(CANONICAL) == EXPECTED_CANONICAL_SHA256,
            "canonical M45-r3 SHA drift")
    spec = importlib.util.spec_from_file_location(
        "m45_r3_independent_reviewer_validate", str(REVIEWER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    review = read_json(args.review)
    require(module.build() == review,
            "review is not reproducible from frozen inputs")
    require(review["review"] == {
        "decision": "GO_LEDGER_ONLY_SCHEDULER_STATE_CAPACITY_REPAIR",
        "score_0_to_100": 94, "p0": 0, "p1": 0, "p2": 4},
        "review scorecard drift")
    require(review["repair_disposition"]["r2_metadata_capacity_p1_closed"]
            is True, "r2 P1 disposition drift")
    require(review["capacity_reconstruction"][
        "combined_local_capacity_bytes"] == 151040 and
        review["capacity_reconstruction"][
        "local_capacity_headroom_bytes"] == 42688,
        "review capacity drift")
    require(review["ledger_only_legality"][
        "all10_rerun_required_for_this_capacity_only_repair"] is False,
        "ledger-only rerun disposition drift")
    print("PASS M45-r3 independent hammer {}".format(args.review))


if __name__ == "__main__":
    main()
