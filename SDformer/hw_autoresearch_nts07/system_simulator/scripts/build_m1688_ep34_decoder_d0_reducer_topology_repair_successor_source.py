#!/usr/bin/env python3
"""M1688 additive repair of the M1682 reducer-topology finding.

M1681 remains immutable.  This source inherits its grid, scheduler, payload
closure, sealed receipt validation and metric checks.  It changes only the
completion arbiter: every accepted shard must have the exact sibling topology
``result=True, attempt=True, work=False, failure=False`` and its attempt must
be a regular non-symlink with mode exactly 0400.  The reducer calls this
strong verifier directly for all 8,700 ordinals.

The failed M1683 release name is permanently forbidden.  M1688 is source-only
and requires a different-author M1689 review before any newly numbered release
may be considered.  No CLI reaches payload, replay or reduction.  CPython 3.6
safe.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1688_ep34_decoder_d0_reducer_topology_repair_successor_"
    "source_contract_r1_20260901.json")
M1681_SOURCE = HERE / (
    "build_m1681_ep34_decoder_d0_shard_execution_closure_successor_source.py")
M1681_TEST = HW / (
    "system_simulator/tests/"
    "test_m1681_ep34_decoder_d0_shard_execution_closure_successor_source.py")
M1681_CONTRACT = HW / (
    "contracts/m1681_ep34_decoder_d0_shard_execution_closure_successor_"
    "source_contract_r1_20260901.json")
M1682_REVIEW = HW / (
    "reviews/m1682_m1681_ep34_decoder_d0_shard_execution_closure_"
    "successor_source_independent_review_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_REVIEW = HW / (
    "reviews/m1689_m1688_ep34_decoder_d0_reducer_topology_repair_"
    "successor_source_independent_review_r1_20260901")
FORBIDDEN_M1683_RELEASE = HW / (
    "contracts/m1683_m1682_m1681_ep34_decoder_d0_shard_execution_"
    "campaign_release_r1_20260901.json")

SCHEMA = "m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M1682_REDUCER_TOPOLOGY_P1_REPAIRED__"
    "M1683_RELEASE_PERMANENTLY_FORBIDDEN__M1689_REVIEW_REQUIRED")
M1681_SOURCE_SHA256 = (
    "006535679b38e2aa207fadde05e9207d2e72dae0464315dceea4a3c96da77a6f")
M1681_TEST_SHA256 = (
    "e80c432a88397dc2c10495f8e019be0452fa0e64c150ee05b74d500de57e5721")
M1681_CONTRACT_SHA256 = (
    "3056b9ab52a24e86a98f565cdfe59f3c15f063aaf346477990190a3a9fedddfb")
M1682_REVIEW_SHA256 = (
    "67b08d3d0a4dc5160da5499bdf1fd72ce912aa505bd934c04102ce4d6625ab3e")
M1682_MANIFEST_SHA256 = (
    "8f9548cc216c83a7868cbe2d5ee678436a12c09f55c465fbbe841eb055c0981f")
M1682_OUTER_FILE_SHA256 = (
    "b7ef378a17c6f0ed7f14d9beffb636ebddb35f410b59820550f639485020f05f")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")


class M1688Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1688Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1688Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_m1681():
    regular_exact(M1681_SOURCE, M1681_SOURCE_SHA256, "exact M1681 source")
    spec = importlib.util.spec_from_file_location("m1688_exact_m1681",
                                                  str(M1681_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1681")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1681_ep34_decoder_d0_shard_execution_closure_successor_source_r1_v1" and
            module.G.TOTAL_SHARDS == 8700 and
            tuple(module.CONFIGS) == CONFIGS,
            "M1681 boundary drift")
    return module


B = load_m1681()


def verify_m1682_disposition():
    seal = B.verify_sealed_tree(M1682_REVIEW, M1682_REVIEW_SHA256,
        M1682_MANIFEST_SHA256, M1682_OUTER_FILE_SHA256, True, "M1682")
    require(all("__pycache__/" in name and name.endswith(".pyc")
                for name in seal["ignored_unsealed_pycache"]),
            "M1682 ignored runtime cache boundary drift")
    row = B.strict_json(M1682_REVIEW / "review.json")
    require(row.get("status") ==
            "FAIL_M1682_M1681_DECODER_D0_EXECUTION_CLOSURE_SOURCE__NO_M1683_RELEASE__SUCCESSOR_REDUCER_TOPOLOGY_REPAIR_REQUIRED" and
            row.get("verdict") == "FAIL_CLOSED_NO_M1683_RELEASE" and
            row.get("p0_count") == 0 and row.get("p1_count") == 1 and
            row.get("p1", [{}])[0].get("id") ==
                "P1_REDUCER_BYPASSES_EXACT_SIBLING_NAMESPACE_TOPOLOGY" and
            row.get("p2", [{}])[0].get("id") ==
                "P2_ATTEMPT_VERIFIER_TYPE_MODE_HARDENING" and
            row.get("authorization", {}).get(
                "successor_reducer_topology_repair_source") is True and
            row.get("authorization", {}).get("release_authoring") is False,
            "M1682 finding/disposition drift")
    return seal


def exact_sibling_topology(ordinal):
    paths = B.namespace_paths(ordinal)
    present = dict((name, os.path.lexists(str(path)))
                   for name, path in paths.items())
    require(present == {"result": True, "attempt": True,
                        "work": False, "failure": False},
            "sealed shard sibling topology must be exact")
    attempt = paths["attempt"]
    try:
        mode = attempt.lstat().st_mode
    except OSError as error:
        raise M1688Error("missing sealed shard attempt") from error
    require(stat.S_ISREG(mode) and not attempt.is_symlink(),
            "sealed shard attempt must be regular non-symlink")
    require(stat.S_IMODE(mode) == 0o400,
            "sealed shard attempt mode must be exactly 0400")
    return paths


def verify_sealed_shard(ordinal):
    """Strong completion arbiter: topology first, then exact M1681 receipt."""
    exact_sibling_topology(ordinal)
    verified = B.verify_sealed_shard(ordinal)
    require(verified.get("ordinal") == ordinal,
            "M1681 sealed shard ordinal drift")
    return verified


def reduce_complete_sealed_shards():
    """Reduce only 8,700 shards accepted by the strong topology verifier."""
    totals = dict((configuration, {"cycles": 0, "requests": 0,
                                   "bytes": {}})
                  for configuration in CONFIGS)
    manifest_chain = hashlib.sha256()
    for ordinal in range(B.G.TOTAL_SHARDS):
        verified = verify_sealed_shard(ordinal)
        row = verified["row"]
        require(row["shard"] == B.G.shard_descriptor(ordinal),
                "reducer shard order drift")
        manifest_chain.update((str(ordinal) + ":" +
            verified["seal"]["manifest_sha256"] + "\n").encode("ascii"))
        for configuration, metric in zip(CONFIGS, row["metrics"]):
            B.validate_metric(metric, configuration, row["shard"])
            target = totals[configuration]
            target["cycles"] += metric["total_cycles"]
            target["requests"] += metric["request_count"]
            for name in B.EXPECTED_BYTE_KINDS:
                value = metric["byte_counts"].get(name, 0)
                require(type(value) is int and value >= 0,
                        "reducer byte ledger drift")
                target["bytes"][name] = target["bytes"].get(name, 0) + value
    dense = totals[CONFIGS[0]]["cycles"]
    equal = totals[CONFIGS[1]]["cycles"]
    typed = totals[CONFIGS[2]]["cycles"]
    return {"schema": SCHEMA,
        "status": "COMPLETE_8700_EXACT_TOPOLOGY_SEALED_SHARDS__INDEPENDENT_HAMMER_REQUIRED",
        "configuration_totals": totals,
        "ratio_of_sums": {
            "dense_to_bit_typed": {"numerator": dense,
                "denominator": typed},
            "bit_equal_to_bit_typed": {"numerator": equal,
                "denominator": typed}},
        "complete_shards": B.G.TOTAL_SHARDS,
        "sealed_manifest_chain_sha256": manifest_chain.hexdigest(),
        "exact_sibling_topology": True,
        "attempt_regular_nonsymlink_mode_0400": True,
        "shard_isolated": True, "monolithic_full_call": False,
        "full_decoder": False, "system_speedup": False,
        "paper_result_pending_independent_hammer": True}


def validate_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1681_SOURCE, M1681_SOURCE_SHA256, "exact M1681 source")
    regular_exact(M1681_TEST, M1681_TEST_SHA256, "exact M1681 test")
    regular_exact(M1681_CONTRACT, M1681_CONTRACT_SHA256,
                  "exact M1681 contract")
    review = verify_m1682_disposition()
    require(not os.path.lexists(str(FORBIDDEN_M1683_RELEASE)) and
            not os.path.lexists(str(Path(str(FORBIDDEN_M1683_RELEASE) +
                                         ".sha256"))) and
            not os.path.lexists(str(Path(str(FORBIDDEN_M1683_RELEASE) +
                                         ".sha256.seal.sha256"))),
            "forbidden M1683 release or sidecar exists")
    require(not FUTURE_REVIEW.exists(),
            "future M1689 review must be absent at source stage")
    return {"m1682": review, "grid": B.G.validate_grid(),
            "M1683_release_permanently_forbidden": True,
            "payload_opened": False, "execution": False,
            "reducer_executed": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "additive_repair": {"m1681_source_sha256": M1681_SOURCE_SHA256,
            "grid_changed": False, "scheduler_changed": False,
            "execution_path_changed": False,
            "completion_arbiter_changed": True,
            "required_topology": {"result": True, "attempt": True,
                "work": False, "failure": False},
            "attempt_type": "regular non-symlink",
            "attempt_mode_octal": "0400",
            "strong_verifier": "verify_sealed_shard",
            "strong_reducer": "reduce_complete_sealed_shards"},
        "numbering": {"source": "M1688", "review": "M1689",
            "forbidden_release": "M1683", "release_created": False},
        "claim_boundary": {"source_only": True,
            "actual_payload": False, "actual_execution": False,
            "reducer_execution": False, "cycles": False,
            "traffic": False, "speedup": False, "energy": False,
            "rtl": False, "eda": False, "full_d0_result": False,
            "monolithic_full_call": False, "full_decoder": False,
            "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M1688_SOURCE_PREFLIGHT__NO_PAYLOAD_NO_EXECUTION",
            "authorities": validate_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
