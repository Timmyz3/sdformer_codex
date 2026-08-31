#!/usr/bin/env python3
from __future__ import print_function

import glob
import hashlib
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNNER = "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_exact_sha_r1.sh"
TCL = "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl"
CONTRACT = "contracts/m884_m528_r21_macro_aware_product_dc_source_only_contract_r1_20260829.json"
CANDIDATE = "contracts/m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_r1_20260829.json"
SOURCE_REVIEW = "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829/review.json"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "b23d53dc45828d3e206d0e37f421f775d585c9cc32c457addeea6b26cc9b4ab2",
    TCL: "f9703e94198f05dbeb9101e12ec4e8dfa993e528212b173fba64cc2a261066e1",
    CONTRACT: "271b6e85119ef0783dc074788c0269a4f5e047c9a2fe572bb8b86fba07fd56fb",
    CANDIDATE: "e89c2d613906412fcf1381ef71261a509f140b2f6d454d3b66e02ad2b5cfe080",
    SOURCE_REVIEW: "607b3898c05ce816b25f8cff26ffe01991d603db5e106707e2b7f8dc80d91b95",
    "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829/SHA256SUMS": "7e8c08587529b574049e2dd5e43bdd9f205bf9cf8e5dbf42397ed1cce6dd3497",
    "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829/SHA256SUMS.seal.sha256": "df48b418dd8c73b4f0e2920517c3144f158a900baa23c38727b0ea4cc53b1c59",
    "reviews/m884_m528_r21_macro_aware_product_dc_source_author_handoff_r1_20260829/SHA256SUMS.seal.sha256": "dacac902cae14427f0b03435e62df5f91972c134b3f7c832fad3c24ba63133e3",
    "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_hammer_REQUEST_r1_20260829/SHA256SUMS.seal.sha256": "016426089501f8dc509f2b71e18cb5fd3009bde7aa1fb55cb37b799d08768db8",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_nonfinite)


def main():
    for rel, expected in EXPECTED.items():
        path = os.path.join(ROOT, rel)
        require(os.path.isfile(path) and not os.path.islink(path), "identity missing/symlink: %s" % rel)
        require(sha(path) == expected, "identity drift: %s" % rel)

    candidate = strict_load(os.path.join(ROOT, CANDIDATE))
    contract = strict_load(os.path.join(ROOT, CONTRACT))
    source_review = strict_load(os.path.join(ROOT, SOURCE_REVIEW))
    require(type(candidate["launch_now"]) is bool and candidate["launch_now"] is False,
            "candidate launch boundary")
    require(type(candidate["authorization"]["max_attempts"]) is int and
            candidate["authorization"]["max_attempts"] == 0,
            "candidate max attempts")
    require(candidate["authorization"]["run_dc"] is False, "candidate DC boundary")
    require(contract["authorization"]["run_dc_now"] is False, "contract DC boundary")
    require(source_review["verdict"] == "PASS", "M885 verdict")
    require(type(source_review["score_out_of_100"]) is int and
            source_review["score_out_of_100"] == 100, "M885 score")
    require([source_review["p0_count"], source_review["p1_count"],
             source_review["p2_count"]] == [0, 0, 0], "M885 severities")

    # The exact production predicate in the byte-frozen runner is intentionally
    # evaluated against the exact M885 object.  It must be false because the
    # runner requests two fields that M885 does not define.
    require("score_100" not in source_review, "unexpected score_100 field")
    require("severity_counts" not in source_review, "unexpected severity_counts field")
    production_predicate = (
        source_review.get("verdict") == "PASS" and
        source_review.get("score_100") == 100 and
        source_review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
    )
    require(production_predicate is False, "production predicate unexpectedly passes")

    runner_text = open(os.path.join(ROOT, RUNNER), "r").read()
    require('.score_100 == 100' in runner_text, "runner score predicate drift")
    require('.severity_counts == {"p0":0,"p1":0,"p2":0}' in runner_text,
            "runner severity predicate drift")
    require('m884_source_review="$(jq -er' in runner_text, "source review binding drift")

    release = os.path.join(ROOT, "contracts/m884_m528_r21_macro_aware_product_dc_launch_release_r1_20260829.json")
    final_review = os.path.join(ROOT, "reviews/m884_m528_r21_macro_aware_product_dc_final_launch_hammer_r1_20260829")
    canonical = os.path.join(ROOT, "dc_handoff/runs/m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829")
    attempt = os.path.join(ROOT, "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_attempt_consumed")
    lock = os.path.join(ROOT, "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_launch_lock")
    for path in (release, final_review, canonical, attempt, lock):
        require(not os.path.lexists(path), "prospective path present: %s" % path)
    require(glob.glob(os.path.join(ROOT, "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_work.*")) == [],
            "work population nonzero")
    require(glob.glob(canonical + ".failed_or_incomplete.*.quarantine") == [],
            "quarantine population nonzero")

    duplicate_negatives = [b'{"x":1,"x":2}', b'{"launch_now":true,"launch_now":false}']
    nonfinite_negatives = [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']
    for payload in duplicate_negatives + nonfinite_negatives:
        try:
            json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_nonfinite)
        except ValueError:
            pass
        else:
            raise AssertionError("strict JSON negative accepted")

    print("PASS_M891_FAIL_CLOSED_RELEASE_AUTHOR_PREFLIGHT")
    print("python=%s" % sys.version.split()[0])
    print("frozen_identities=%d" % len(EXPECTED))
    print("production_source_review_predicate=false")
    print("missing_score_100=true")
    print("missing_severity_counts=true")
    print("duplicate_key_negatives=2")
    print("nonfinite_negatives=3")
    print("release_final_result_attempt_work_quarantine_absent=true")
    print("eda_runs=0")
    print("license_queries=0")
    print("remote_runs=0")


if __name__ == "__main__":
    main()
