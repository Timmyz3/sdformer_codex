#!/usr/bin/env python3
"""Read-only correction hammer for the sealed but schema-invalid M1669 review.

This script never invokes capture, GPU, EDA, SSH, or a production attempt.  It
first proves that the sealed canonical M1669 review is rejected by M1668's own
``validate_future_authorities`` review clause.  It then checks the already
sealed hammer payload as an inert replacement candidate by constructing a
temporary, locally sealed review/release fixture and calling the unmodified
validator end to end.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "successor_r1.py")
CANONICAL = HW / (
    "reviews/m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_"
    "rebind_source_independent_review_r1_20260901")


class CorrectionError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CorrectionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def strict_json(path):
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates)


def load_source():
    spec = importlib.util.spec_from_file_location("m1668_correction_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_recursive_tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "canonical review absent/symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n", "outer seal mismatch")
    sealed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in sealed,
                "malformed/duplicate manifest row")
        member = root / fields[1]
        require(member.is_file() and not member.is_symlink(),
                "sealed member absent/symlink")
        require(sha256(member) == fields[0], "sealed member hash mismatch")
        sealed[fields[1]] = fields[0]
    actual = set(str(path.relative_to(root)) for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(sealed), "canonical recursive population mismatch")
    return {
        "entries": len(sealed),
        "manifest_sha256": sha256(manifest),
        "outer_file_sha256": sha256(outer),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def seal_review(root, payload):
    root.mkdir()
    review = root / "review.json"
    write_json(review, payload)
    manifest = root / "SHA256SUMS"
    manifest.write_text(sha256(review) + "  review.json\n", encoding="ascii")
    outer = root / "SHA256SUMS.seal.sha256"
    outer.write_text(sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")
    return sha256(review), sha256(manifest), sha256(outer)


def seal_file(path, payload):
    write_json(path, payload)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(sha256(path) + "  " + path.name + "\n", encoding="ascii")
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(sha256(sidecar) + "  " + sidecar.name + "\n",
                     encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = load_source()
    seal = verify_recursive_tree(CANONICAL)
    actual = strict_json(CANONICAL / "review.json")
    candidate = strict_json(CANONICAL / "cpython312_hammer.json")

    expected_identity = {
        "source_sha256": sha256(source.SOURCE),
        "test_sha256": sha256(source.TEST),
        "source_contract_sha256": sha256(source.SOURCE_CONTRACT),
        "selection_identity_sha256": source.SELECTION_IDENTITY_SHA256,
        "runtime_tar_sha256": source.RUNTIME_TAR_SHA256,
        "m1647_source_sha256": source.M1647_SOURCE_SHA256,
        "m1648_review_sha256": source.M1648_REVIEW_SHA256,
        "m1649_release_sha256": source.M1649_RELEASE_SHA256,
        "checkpoint_sha256": source.CHECKPOINT_SHA256,
        "config_sha256": source.CONFIG_SHA256,
        "profile_sha256": source.PROFILE_SHA256,
        "docs359_sha256": source.DOCS359_SHA256,
    }
    expected_authorization = {
        "release_authoring": True,
        "capture": False,
        "gpu": False,
        "automatic_retry": False,
    }
    mismatches = {
        "score_key_missing": "score" not in actual,
        "score_seen_by_validator": actual.get("score", 0),
        "identity_exact": actual.get("identity") == expected_identity,
        "identity_missing_keys": sorted(set(expected_identity) -
                                        set(actual.get("identity", {}))),
        "identity_extra_keys": sorted(set(actual.get("identity", {})) -
                                      set(expected_identity)),
        "authorization_exact": actual.get("authorization") ==
                               expected_authorization,
        "authorization_missing_keys": sorted(set(expected_authorization) -
                                             set(actual.get("authorization", {}))),
        "authorization_extra_keys": sorted(set(actual.get("authorization", {})) -
                                           set(expected_authorization)),
    }
    require(mismatches == {
        "score_key_missing": True,
        "score_seen_by_validator": 0,
        "identity_exact": False,
        "identity_missing_keys": [
            "m1647_source_sha256", "m1648_review_sha256",
            "m1649_release_sha256", "profile_sha256"],
        "identity_extra_keys": [
            "author_manifest_sha256", "author_outer_seal_file_sha256"],
        "authorization_exact": False,
        "authorization_missing_keys": ["release_authoring"],
        "authorization_extra_keys": [
            "attempt_write", "m1670_release_authoring", "remote_write"],
    }, "unexpected canonical mismatch signature")

    canonical_error = None
    try:
        source.validate_future_authorities()
    except Exception as error:
        canonical_error = type(error).__name__ + ": " + str(error)
    require(canonical_error is not None and
            "M1669 review mismatch" in canonical_error,
            "canonical review was not rejected at M1669 review clause")

    require(candidate.get("status") == source.REVIEW_STATUS and
            candidate.get("score", 0) >= 95 and
            candidate.get("p0_count") == 0 and
            candidate.get("p1_count") == 0 and
            candidate.get("identity") == expected_identity and
            candidate.get("authorization") == expected_authorization,
            "sealed hammer payload is not an exact replacement candidate")

    old_review = source.FUTURE_REVIEW
    old_release = source.FUTURE_RELEASE
    old_child_python = source.P.P.CHILD_PYTHON
    replacement_validator_passed = False
    with tempfile.TemporaryDirectory(
            prefix=".m1669-correction-fixture-",
            dir=str(HW / "reviews")) as tmp:
        tmp = Path(tmp)
        # The production interpreter exists only on the remote capture host.
        # Rebind the local validator fixture to the current regular interpreter;
        # this checks exact review/release schema consumption only and is never
        # represented as remote interpreter or launch admission evidence.
        source.P.P.CHILD_PYTHON = Path(sys.executable).resolve()
        review_root = tmp / "review"
        review_sha, manifest_sha, outer_sha = seal_review(review_root, candidate)
        release_path = tmp / "release.json"
        release_identity = dict(expected_identity)
        release_identity.update({
            "review_sha256": review_sha,
            "review_manifest_sha256": manifest_sha,
            "review_outer_file_sha256": outer_sha,
        })
        release = {
            "schema": source.RELEASE_SCHEMA,
            "status": source.RELEASE_STATUS,
            "identity": release_identity,
            "authorization": {
                "parent_calls": 1,
                "clean_child_processes": 1,
                "gpu_runs": 1,
                "production_captures": 1,
                "automatic_retry": False,
                "all_other_runs": 0,
            },
            "namespaces": {
                "result": str(source.RESULT.relative_to(source.ROOT)),
                "attempt": str(source.ATTEMPT.relative_to(source.ROOT)),
                "work": str(source.WORK.relative_to(source.ROOT)),
                "failure": str(source.FAILURE.relative_to(source.ROOT)),
            },
            "pre_budget_preflight": {
                "runtime_m1257_canonical": True,
                "current_entity_exact": True,
                "build_runtime_before_parent_subprocess": True,
                "build_runtime_before_child_gpu_attempt": True,
            },
            "claim_boundary": {
                "tsbg_dse": False,
                "aee": False,
                "rtl": False,
                "eda": False,
                "performance": False,
                "paper_result": False,
            },
            "child_interpreter": {
                "path": str(source.P.P.CHILD_PYTHON),
                "sha256": sha256(source.P.P.CHILD_PYTHON),
            },
        }
        seal_file(release_path, release)
        source.FUTURE_REVIEW = review_root
        source.FUTURE_RELEASE = release_path
        try:
            returned = source.validate_future_authorities()
            replacement_validator_passed = returned == release
        finally:
            source.FUTURE_REVIEW = old_review
            source.FUTURE_RELEASE = old_release
            source.P.P.CHILD_PYTHON = old_child_python
    require(replacement_validator_passed,
            "exact replacement candidate failed unmodified validator")

    output = {
        "schema": "m1669_m1668_review_schema_correction_r2_v1",
        "date_cst": "2026-09-01",
        "status": (
            "FAIL_CLOSED_M1669_CANONICAL_REVIEW_SCHEMA_MISMATCH__"
            "SUPERSEDED__NO_M1670_RELEASE"),
        "verdict": "FAIL_CLOSED_SUPERSEDED_NO_M1670_RELEASE",
        "score_out_of_100": 100,
        "p0_count": 1,
        "p1_count": 0,
        "p2_count": 0,
        "p0": [{
            "id": "P0_CANONICAL_REVIEW_NOT_CONSUMABLE_BY_M1668",
            "finding": (
                "The sealed canonical review.json does not satisfy M1668's "
                "exact score/identity/authorization shape and is rejected "
                "before any release may be consumed."),
        }],
        "canonical_seal": seal,
        "canonical_validator_error": canonical_error,
        "canonical_mismatches": mismatches,
        "replacement_candidate": {
            "source": "canonical cpython312_hammer.json",
            "sha256": sha256(CANONICAL / "cpython312_hammer.json"),
            "exact_review_clause_shape": True,
            "unmodified_validate_future_authorities_with_temporary_local_release":
                replacement_validator_passed,
            "local_child_interpreter_fixture_only": True,
            "remote_interpreter_validated": False,
            "published_to_canonical_path": False,
        },
        "authorization": {
            "m1670_release_authoring": False,
            "m1670_release": False,
            "remote_launch": False,
            "capture": False,
            "gpu": False,
            "attempt_write": False,
            "automatic_retry": False,
        },
        "required_repair": [
            "Do not edit the sealed canonical M1669 tree in place.",
            "Publish a separately reviewed exact-shape replacement at the canonical path only under a new explicit authority or a source successor that pins an additive review path.",
            "After replacement, run M1668.validate_future_authorities with the real separately sealed M1670 release before any remote launch.",
        ],
        "review_execution": {
            "remote_connections": 0,
            "remote_writes": 0,
            "capture_runs": 0,
            "gpu_runs": 0,
            "attempt_writes": 0,
            "eda_runs": 0,
            "temporary_local_validator_fixtures": 1,
            "git_commit": False,
            "git_push": False,
        },
    }
    write_json(args.output, output)
    print(output["status"])


if __name__ == "__main__":
    main()
