#!/usr/bin/env python3
"""M1704 authority adapter for the exact M1688/M1681 D0 shard campaign.

M1688 repaired only the reducer topology and deliberately exposed no execution
authority.  Its inherited M1681 executor still points at the now permanently
forbidden M1683 release.  This additive adapter changes only that authority
edge: a future sealed M1705 review and M1706 release are validated here, then
the exact frozen ``M1681.B._run_authorized_shard`` implementation is called.
No scheduler, grid, payload, metric, namespace or result schema is rewritten.
Reduction remains the exact M1688 strong topology reducer.  Source CLI is
preflight/describe only; CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m1704_ep34_decoder_d0_execution_authority_adapter_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1704_ep34_decoder_d0_execution_authority_adapter_"
    "source_contract_r1_20260901.json")
M1688_SOURCE = HERE / (
    "build_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py")
M1688_TEST = HW / (
    "system_simulator/tests/"
    "test_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py")
M1688_CONTRACT = HW / (
    "contracts/m1688_ep34_decoder_d0_reducer_topology_repair_successor_"
    "source_contract_r1_20260901.json")
M1689_REVIEW = HW / (
    "reviews/m1689_m1688_ep34_decoder_d0_reducer_topology_repair_"
    "successor_source_independent_review_r1_20260901")
FUTURE_REVIEW = HW / (
    "reviews/m1705_m1704_ep34_decoder_d0_execution_authority_adapter_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1706_m1705_m1704_ep34_decoder_d0_8700_shard_"
    "campaign_release_r1_20260901.json")
FORBIDDEN_M1683_RELEASE = HW / (
    "contracts/m1683_m1682_m1681_ep34_decoder_d0_shard_execution_"
    "campaign_release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m1704_ep34_decoder_d0_execution_authority_adapter_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M1688_STRONG_REDUCER_PRESERVED__M1683_PERMANENTLY_"
    "FORBIDDEN__M1705_REVIEW_REQUIRED")
REVIEW_STATUS = (
    "PASS_M1705_M1704_DECODER_D0_EXECUTION_AUTHORITY_ADAPTER_SOURCE__"
    "AUTHORIZE_M1706_RELEASE_AUTHORING_ONLY__NO_EXECUTION")
RELEASE_SCHEMA = (
    "m1706_m1705_m1704_ep34_decoder_d0_8700_shard_campaign_release_r1_v1")
RELEASE_STATUS = "AUTHORIZE_M1704_ADAPTER_BACKED_FULL_D0_8700_SHARD_CAMPAIGN"
M1688_SOURCE_SHA256 = (
    "2ae2725e24c46972f46c54ae71260a8fc637e85c4de0b90f9f91bc42da76abba")
M1688_TEST_SHA256 = (
    "7a331143f6d486939ed77eb34eef60610e450d131313f6df3340cd76290662cb")
M1688_CONTRACT_SHA256 = (
    "10f44a589f986c06f560b0353224b83f5ca6f44e5a0ac73599bd40a8dc85271f")
M1689_REVIEW_SHA256 = (
    "227c03b0cf9c16e80780f4889df198e1d0fd50c4e4802be1e24cb596e79655e8")
M1689_MANIFEST_SHA256 = (
    "2a331bfafa358b6285bf3e3da98a4f019a9e2409dea06f5ebe13025f001c04b4")
M1689_OUTER_FILE_SHA256 = (
    "e76e8ce32dd6049eae72df2864c150cdc8a7427cbb8bf4be1f1b466a94dcacc8")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


class M1704Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1704Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m1688():
    regular_exact(M1688_SOURCE, M1688_SOURCE_SHA256, "exact M1688 source")
    spec = importlib.util.spec_from_file_location("m1704_exact_m1688",
                                                  str(M1688_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1688")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_r1_v1" and
            module.B.G.TOTAL_SHARDS == 8700 and
            module.reduce_complete_sealed_shards.__module__ == module.__name__,
            "M1688 reducer/grid identity drift")
    return module


M1688 = load_m1688()
B = M1688.B


def _absent_with_sidecars(path, label):
    paths = (Path(path), Path(str(path) + ".sha256"),
             Path(str(path) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(item)) for item in paths),
            label + " or sidecar exists")


def verify_m1689():
    seal = B.verify_sealed_tree(M1689_REVIEW, M1689_REVIEW_SHA256,
        M1689_MANIFEST_SHA256, M1689_OUTER_FILE_SHA256, False, "M1689")
    row = B.strict_json(M1689_REVIEW / "review.json")
    require(row.get("status") ==
            "PASS_M1689_M1688_DECODER_D0_REDUCER_TOPOLOGY_REPAIR_SOURCE__AUTHORIZE_NEWLY_NUMBERED_RELEASE_AUTHORING_ONLY__NO_EXECUTION" and
            row.get("score") == 100 and row.get("p0_count") == 0 and
            row.get("p1_count") == 0 and row.get("p2_count") == 0 and
            row.get("authorization") == {
                "newly_numbered_release_authoring": True,
                "m1683_release_authoring": False,
                "payload_open": False, "shard_execution": False,
                "reducer_execution": False, "automatic_retry": False,
                "gpu": False, "eda": False},
            "M1689 authority drift")
    return seal


def _review_identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1688_source_sha256": M1688_SOURCE_SHA256,
        "m1688_test_sha256": M1688_TEST_SHA256,
        "m1688_contract_sha256": M1688_CONTRACT_SHA256,
        "m1689_review_sha256": M1689_REVIEW_SHA256,
        "m1689_manifest_sha256": M1689_MANIFEST_SHA256,
        "m1689_outer_file_sha256": M1689_OUTER_FILE_SHA256,
        "checkpoint_sha256": B.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": B.G.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_future_review_and_release():
    """Exact M1705/M1706 gate replacing only inherited M1683 authority."""
    _absent_with_sidecars(FORBIDDEN_M1683_RELEASE,
                          "permanently forbidden M1683 release")
    review_seal = B.verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M1705")
    review = B.strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == _review_identity() and
            review.get("authorization") == {
                "release_authoring": True, "shard_execution": False,
                "payload_open": False, "reducer_execution": False,
                "automatic_retry": False},
            "M1705 semantic authority drift")
    release_sha = B.verify_double_sealed_file(FUTURE_RELEASE, "M1706 release")
    release = B.strict_json(FUTURE_RELEASE)
    identity = dict(_review_identity(),
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=review_seal["manifest_sha256"],
        review_outer_file_sha256=review_seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == identity and
            release.get("authorization") == {
                "shard_runs": 8700, "payload_opens": 8700,
                "attempt_writes": 8700, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            release.get("fixed_grid") == B.G.fixed_grid() and
            release.get("namespace_examples") == {
                "first": B.namespace_strings(0),
                "last": B.namespace_strings(B.G.TOTAL_SHARDS - 1)} and
            release.get("reducer") == {
                "source": "M1688", "strong_exact_sibling_topology": True,
                "attempt_regular_nonsymlink_mode_0400": True} and
            release.get("claim_boundary") == {
                "shard_isolated": True, "monolithic_full_call": False,
                "full_decoder": False, "system_speedup": False,
                "paper_result": False},
            "M1706 release identity/authorization/grid/namespace drift")
    return release_sha


def _run_authorized_shard(ordinal):
    """Run exact inherited shard code with only its authority gate rebound."""
    require(type(ordinal) is int and 0 <= ordinal < B.G.TOTAL_SHARDS,
            "shard ordinal out of range")
    original = B.validate_future_review_and_release
    B.validate_future_review_and_release = validate_future_review_and_release
    try:
        return B._run_authorized_shard(ordinal)
    finally:
        B.validate_future_review_and_release = original


def reduce_complete_sealed_shards():
    """No adapter reducer: delegate only to M1688's strong verifier/reducer."""
    return M1688.reduce_complete_sealed_shards()


def validate_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1688_SOURCE, M1688_SOURCE_SHA256, "exact M1688 source")
    regular_exact(M1688_TEST, M1688_TEST_SHA256, "exact M1688 test")
    regular_exact(M1688_CONTRACT, M1688_CONTRACT_SHA256,
                  "exact M1688 contract")
    B.verify_double_sealed_file(M1688_CONTRACT, "M1688 contract")
    seal = verify_m1689()
    _absent_with_sidecars(FORBIDDEN_M1683_RELEASE,
                          "permanently forbidden M1683 release")
    require(not FUTURE_REVIEW.exists(), "future M1705 review exists")
    _absent_with_sidecars(FUTURE_RELEASE, "future M1706 release")
    require(B.G.validate_grid() == {"calls": 30, "timesteps": 300,
            "destinations": 360000, "shards": 8700,
            "gap_count": 0, "overlap_count": 0}, "fixed grid drift")
    return {"m1689": seal, "grid": B.G.validate_grid(),
        "m1683_release_permanently_forbidden": True,
        "payload_opened": False, "execution": False,
        "reducer_executed": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "authority_adapter": {
            "execution_implementation": "M1688.B._run_authorized_shard",
            "authority_override_only": True,
            "grid_changed": False, "scheduler_changed": False,
            "payload_path_changed": False, "metric_changed": False,
            "namespace_changed": False, "result_schema_changed": False,
            "future_review": "M1705", "future_release": "M1706",
            "forbidden_release": "M1683"},
        "reducer": {"implementation":
            "M1688.reduce_complete_sealed_shards",
            "exact_sibling_topology": {"result": True, "attempt": True,
                "work": False, "failure": False},
            "attempt_regular_nonsymlink_mode_0400": True},
        "fixed_grid": B.G.fixed_grid(),
        "claim_boundary": {"source_only": True, "payload_opened": False,
            "shard_execution": False, "reducer_execution": False,
            "cycles": False, "traffic": False, "speedup": False,
            "energy": False, "rtl": False, "eda": False,
            "full_d0_result": False, "monolithic_full_call": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M1704_SOURCE_PREFLIGHT__NO_PAYLOAD_NO_EXECUTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
