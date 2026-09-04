#!/usr/bin/env python3
"""Fail-closed validator for the independent M47-r1 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEWER = HW_ROOT / (
    "system_simulator/scripts/review_m47_bit_tight_timestep_pair_single_buffer.py")
REVIEW = HW_ROOT / (
    "results/m47_bit_tight_timestep_pair_single_buffer_r1_independent_hammer_20260823/"
    "m47_r1_independent_hammer_review.json")
CANONICAL = HW_ROOT / (
    "results/m47_bit_tight_timestep_pair_single_buffer_r1_20260823/"
    "m47_bit_tight_timestep_pair_single_buffer.json")

EXPECTED_REVIEWER_SHA256 = (
    "9902da905cd678302d0fba3bc9f8ee89ea93e62719f5b6cf48d5c0d1c25ac5c7")
EXPECTED_REVIEW_SHA256 = (
    "cd7df5b1b2d8ec5cb701759962449ccf1b6a78273229e9653f2eba6a1d335f29")
EXPECTED_CANONICAL_SHA256 = (
    "dc42df25567ad49be863586a3e287c9137a9f41470e83a5ef95bf125aa1734ed")


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
            "reviewer SHA drift")
    require(sha256(args.review) == EXPECTED_REVIEW_SHA256,
            "review SHA drift")
    require(sha256(CANONICAL) == EXPECTED_CANONICAL_SHA256,
            "canonical SHA drift")
    spec = importlib.util.spec_from_file_location(
        "m47_independent_reviewer_validate", str(REVIEWER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    review = read_json(args.review)
    require(module.build() == review,
            "review does not reproduce from frozen inputs")
    require(review["review"] == {
        "decision": "GO_M47_R1_CONSERVATIVE_LEDGER_ONLY",
        "score_0_to_100": 92, "p0": 0, "p1": 0, "p2": 5},
        "review scorecard drift")
    require(review["capacity_reconstruction"][
        "combined_local_capacity_bytes"] == 174224 and
        review["capacity_reconstruction"][
        "local_capacity_headroom_bytes"] == 19504,
        "review capacity drift")
    require(review["weight_reconstruction"][
        "weight_tile_loads_per_sample"] == 4320 and
        review["weight_reconstruction"][
        "serialized_load_cycles_per_sample"] == 1658880,
        "review weight ledger drift")
    require(review["upper_bound_reconstruction"][
        "p95_nearest_rank"] == 11340632 and
        review["upper_bound_reconstruction"][
        "canonical_upper_bound_slack_vs_deduplicated_model_cycles"] == 12288,
        "review upper-bound drift")
    require(review["conditional_model_reconstruction"][
        "three_x_conditional_model_crossing"] is True and
        review["conditional_model_reconstruction"][
        "system_speedup_admitted"] is False,
        "conditional/system claim boundary drift")
    print("PASS M47-r1 independent hammer {}".format(args.review))


if __name__ == "__main__":
    main()
