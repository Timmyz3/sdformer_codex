#!/usr/bin/env python3
"""Additive M1747 schema-identity successor for the consumed M1727 analysis.

The unique M1727/M1729 analysis failed closed before payload replay because
the exact M1721 implementation expected an obsolete sample-order schema.
M1747 imports the exact M1727 implementation and changes only that identity
boundary: it accepts the canonical successor schema at one exact file SHA,
while retaining the forty-sample/checkpoint checks already implemented by the
predecessor.  Every TSBG/S2 algorithm, comparator, resource caveat, gate and
claim boundary remains the exact M1727 implementation.

Production remains unavailable until the exact M1747 failure receipt, M1744
capture review triple, a future different-author M1748 source review and a
future one-shot M1749 release all validate before capture access.  This source
is CPython-3.6 compatible.  Source checks do not touch capture, run analysis,
use a GPU, invoke EDA or access the network.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/test_m1747_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
CONTRACT = HW / (
    "contracts/m1747_m1729_m1727_m1707_ep34_tsbg_schema_identity_"
    "successor_source_contract_r1_20260901.json")
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1727_SOURCE = HW / (
    "system_simulator/scripts/analyze_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
M1727_TEST = HW / (
    "system_simulator/tests/test_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
M1727_CONTRACT = HW / (
    "contracts/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_contract_r1_20260901.json")
M1729_RELEASE = HW / (
    "contracts/m1729_m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_analysis_release_r1_20260901.json")
FAILED_RECEIPT = HW / (
    "results/m1747_m1729_m1727_ep34_tsbg_analysis_failed_attempt_"
    "receipt_r1_20260901.json")
FAILED_RECEIPT_SIDECAR = Path(str(FAILED_RECEIPT) + ".sha256")
FAILED_RECEIPT_OUTER = Path(str(FAILED_RECEIPT) + ".sha256.seal.sha256")
M1744_REVIEW = HW / (
    "reviews/m1744_m1707_ep34_tsbg_capture_result_independent_hammer_"
    "r1_20260901")
FUTURE_REVIEW = HW / (
    "reviews/m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_"
    "source_hammer_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_"
    "successor_analysis_release_r1_20260901.json")
RESULT = HW / (
    "results/m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901")
WORK = HW / (
    "results/.m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901.work")

M1727_SOURCE_SHA256 = (
    "e0d2fc508a835b667b63a8719af3bf4ad883bfccca5b4c388f4e96ac9c6eaed9")
M1727_TEST_SHA256 = (
    "3b68aa96eba68e397a84459cfdc3199a7b8df6bf646236bf9495e0dd9137071c")
M1727_CONTRACT_SHA256 = (
    "efa110402bee236e4f1d2956ccad364a8de2c52e429d1e58a7c3dbe19f1e55f6")
M1729_RELEASE_SHA256 = (
    "440dd2472c6a92d99980d46b36709d88d697f48ad88b1119a36cd20d1d5d439a")
FAILED_RECEIPT_SHA256 = (
    "e07805d95200208c74b817c13f7d100a78cf33d6d7694fb42cc7a2f7c0be1b24")
FAILED_RECEIPT_SIDECAR_SHA256 = (
    "5b2d9e64158db8e015e377cac5108d4482f9f5c224ecceb7860fc186f3e788fe")
FAILED_RECEIPT_OUTER_SHA256 = (
    "a16412bb861fde518a977e1e5c57c524d924f721e7585826813e261343cf21a5")
M1744_REVIEW_SHA256 = (
    "d237b3a64cf47313873a84a4749465b7cc7361bd8cf57dde5a0b6275f336dbc7")
M1744_MANIFEST_SHA256 = (
    "df15fe385bc7f5eccde2fecd19f5fe478dbc0480653cec5aab208c59a8a6b1f4")
M1744_OUTER_SHA256 = (
    "40c3e5f2c4a98be985bf225fe6cf3a3cda88c3a32047a372c84ca0608baaf1d2")
SAMPLE_ORDER_SCHEMA = "m1544_ep34_m1458_sample_order_r1_v1"
LEGACY_SAMPLE_ORDER_SCHEMA = "m1544_ep34_sample_order_r1_v1"
SAMPLE_ORDER_SHA256 = (
    "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773")
SCHEMA = "m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_v1"
STATUS = ("DIAGNOSTIC_SCREENING_ONLY__CANONICAL_SAMPLE_ORDER_IDENTITY_REPAIRED__"
          "M1727_ALGORITHM_AND_CLAIM_BOUNDARY_UNCHANGED__NO_PAPER_RESULT")
REVIEW_SCHEMA = (
    "m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_source_"
    "hammer_r1_v1")
REVIEW_STATUS = (
    "PASS_M1748_M1747_SOURCE_HAMMER__M1749_RELEASE_MAY_BE_CREATED")
RELEASE_SCHEMA = (
    "m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_"
    "analysis_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1747_EP34_TSBG_SCHEMA_IDENTITY_SUCCESSOR_ANALYSIS")


class M1747Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1747Error(message)


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
        raise M1747Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1747Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_sidecar(path, sidecar, outer, label):
    path, sidecar, outer = Path(path), Path(sidecar), Path(outer)
    regular_exact(sidecar, sha256(sidecar), label + " sidecar")
    regular_exact(outer, sha256(outer), label + " outer")
    require(sidecar.read_text(encoding="ascii").split() ==
            [sha256(path), path.name], label + " sidecar drift")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sidecar), sidecar.name], label + " outer drift")


def verify_sealed_directory(root, label):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), label + " missing")
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and not sums.is_symlink() and
            outer.is_file() and not outer.is_symlink(), label + " seal missing")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sums), sums.name], label + " outer drift")
    names = []
    for line in sums.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                label + " malformed manifest")
        name = fields[1].strip().lstrip("*")
        require(name and name not in names and not Path(name).is_absolute() and
                ".." not in Path(name).parts and Path(name).as_posix() == name,
                label + " unsafe member")
        regular_exact(root / name, fields[0], label + " member " + name)
        names.append(name)
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file() and
                    path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
                    "__pycache__" not in path.parts and path.suffix != ".pyc")
    require(sorted(names) == actual and "review.json" in names,
            label + " manifest coverage drift")
    return {"review_sha256": sha256(root / "review.json"),
            "manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer)}


for _path, _digest, _label in (
        (M1727_SOURCE, M1727_SOURCE_SHA256, "exact M1727 source"),
        (M1727_TEST, M1727_TEST_SHA256, "exact M1727 test"),
        (M1727_CONTRACT, M1727_CONTRACT_SHA256, "exact M1727 contract"),
        (M1729_RELEASE, M1729_RELEASE_SHA256, "consumed M1729 release"),
        (FAILED_RECEIPT, FAILED_RECEIPT_SHA256, "M1727 failure receipt"),
        (FAILED_RECEIPT_SIDECAR, FAILED_RECEIPT_SIDECAR_SHA256,
         "M1727 failure receipt sidecar"),
        (FAILED_RECEIPT_OUTER, FAILED_RECEIPT_OUTER_SHA256,
         "M1727 failure receipt outer")):
    regular_exact(_path, _digest, _label)

_SPEC = importlib.util.spec_from_file_location("m1747_exact_m1727", str(M1727_SOURCE))
require(_SPEC is not None and _SPEC.loader is not None,
        "cannot import exact M1727 predecessor")
M1727 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M1727)
regular_exact(M1727_SOURCE, M1727_SOURCE_SHA256,
              "exact M1727 source after import")
BASE = M1727.BASE
_BASE_VERIFY_CAPTURE_IDENTITY = BASE.verify_capture_identity
_BASE_STRICT_JSON = BASE.strict_json
_BASE_CANONICAL_JSON_BYTES = M1727._BASE_CANONICAL_JSON_BYTES
_ACTIVE_AUTHORITY = None


def adapt_sample_order_document(value):
    """Validate the one canonical successor identity, then adapt only schema."""
    require(type(value) is dict and value.get("schema") == SAMPLE_ORDER_SCHEMA and
            len(value.get("samples", [])) == 40 and
            [row.get("global_sample_id") for row in value["samples"]] ==
                list(range(40)) and
            value.get("identity", {}).get("checkpoint_sha256") ==
                BASE.CHECKPOINT_SHA256,
            "canonical M1707 successor sample order drift")
    adapted = dict(value)
    adapted["schema"] = LEGACY_SAMPLE_ORDER_SCHEMA
    return adapted


def verify_capture_identity(root):
    """Run exact predecessor verification with one exact-SHA schema adapter."""
    sample_path = Path(root) / "sample_order.json"
    regular_exact(sample_path, SAMPLE_ORDER_SHA256,
                  "canonical M1707 sample_order.json")
    canonical = _BASE_STRICT_JSON(sample_path)
    adapted = adapt_sample_order_document(canonical)

    def strict_json_adapter(path, root_type=dict):
        value = _BASE_STRICT_JSON(path, root_type)
        if Path(path) == sample_path:
            regular_exact(path, SAMPLE_ORDER_SHA256,
                          "canonical M1707 sample_order.json during verify")
            require(value == canonical, "sample order changed during verify")
            return adapted
        return value

    old = BASE.strict_json
    BASE.strict_json = strict_json_adapter
    try:
        result = _BASE_VERIFY_CAPTURE_IDENTITY(root)
    finally:
        BASE.strict_json = old
    result = list(result)
    result[2] = canonical
    return tuple(result)


def verify_failure_and_capture_review():
    verify_sidecar(FAILED_RECEIPT, FAILED_RECEIPT_SIDECAR,
                   FAILED_RECEIPT_OUTER, "M1727 failure receipt")
    failure = strict_json(FAILED_RECEIPT)
    require(failure.get("status") ==
            "FAILED_CLOSED_BEFORE_PAYLOAD_REPLAY__SAMPLE_ORDER_SCHEMA_FALSE_NEGATIVE__M1727_NO_RETRY" and
            failure.get("observed_failure", {}).get("canonical_actual_schema") ==
                SAMPLE_ORDER_SCHEMA and
            failure.get("observed_failure", {}).get(
                "canonical_sample_order_sha256") == SAMPLE_ORDER_SHA256 and
            failure.get("absence_witness", {}).get(
                "result_absent_after_failure") is True and
            failure.get("absence_witness", {}).get(
                "work_absent_after_failure") is True and
            failure.get("observed_budget", {}).get("analysis_invocations") == 1 and
            failure.get("observed_budget", {}).get("payload_replays") == 0 and
            failure.get("observed_budget", {}).get("automatic_retry") is False and
            failure.get("observed_budget", {}).get(
                "m1729_authority_consumed") is True,
            "M1727 failure receipt semantic drift")
    review_binding = verify_sealed_directory(M1744_REVIEW, "M1744 review")
    require(review_binding == {
                "review_sha256": M1744_REVIEW_SHA256,
                "manifest_sha256": M1744_MANIFEST_SHA256,
                "outer_seal_file_sha256": M1744_OUTER_SHA256},
            "M1744 capture review triple drift")
    review = strict_json(M1744_REVIEW / "review.json")
    hammer = strict_json(M1744_REVIEW / "hammer_output.json")
    require(review.get("status") ==
            "PASS_M1744_M1707_EP34_TSBG_CAPTURE_RESULT__AUTHORIZE_M1727_ANALYSIS_ONLY" and
            review.get("verified", {}).get("samples") == 40 and
            review.get("authorization", {}).get("capture_retry") is False and
            hammer.get("bindings", {}).get("sample_order_sha256") ==
                SAMPLE_ORDER_SHA256 and
            hammer.get("checks", {}).get("sample_order_40_exact") is True,
            "M1744 capture/sample-order authority drift")
    return review_binding


def source_identities():
    return {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "contract_sha256": sha256(CONTRACT),
        "contract_sidecar_sha256": sha256(CONTRACT_SIDECAR),
        "contract_outer_seal_file_sha256": sha256(CONTRACT_OUTER),
        "m1727_source_sha256": M1727_SOURCE_SHA256,
        "consumed_m1729_release_sha256": M1729_RELEASE_SHA256,
        "m1727_failure_receipt_sha256": FAILED_RECEIPT_SHA256,
        "m1727_failure_receipt_outer_seal_file_sha256":
            FAILED_RECEIPT_OUTER_SHA256,
        "m1744_review_sha256": M1744_REVIEW_SHA256,
        "m1744_review_manifest_sha256": M1744_MANIFEST_SHA256,
        "m1744_review_outer_seal_file_sha256": M1744_OUTER_SHA256,
        "canonical_sample_order_sha256": SAMPLE_ORDER_SHA256}


def validate_source_contract():
    verify_sidecar(CONTRACT, CONTRACT_SIDECAR, CONTRACT_OUTER,
                   "M1747 source contract")
    value = strict_json(CONTRACT)
    require(value.get("schema") ==
            "m1747_m1729_m1727_m1707_ep34_tsbg_schema_identity_successor_source_contract_r1_v1" and
            value.get("status") ==
            "SOURCE_ONLY__EXACT_SCHEMA_IDENTITY_REPAIR__NO_CAPTURE_NO_ANALYSIS_NO_RELEASE" and
            value.get("source") == {"path": str(SOURCE.relative_to(ROOT)),
                                     "sha256": sha256(SOURCE)} and
            value.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                   "sha256": sha256(TEST)} and
            value.get("repair", {}).get("actual_schema") == SAMPLE_ORDER_SCHEMA and
            value.get("repair", {}).get("sample_order_sha256") ==
                SAMPLE_ORDER_SHA256 and
            value.get("authorization") == {
                "analysis_run": False, "capture_verify": False,
                "gpu": False, "eda": False, "network": False,
                "release": False, "paper_result": False} and
            value.get("claim_boundary", {}).get("paper_result") is False,
            "M1747 source contract drift")
    return value


def validate_future_review(root, identities):
    binding = verify_sealed_directory(root, "M1748 review")
    review = strict_json(Path(root) / "review.json")
    require(review.get("schema") == REVIEW_SCHEMA and
            review.get("status") == REVIEW_STATUS and
            review.get("identity") == identities and
            review.get("authorization") == {
                "m1749_release_may_be_created": True,
                "analysis_run": False, "capture_verify": False} and
            review.get("claim_boundary", {}).get("paper_result") is False,
            "M1748 review authority drift")
    return binding


def validate_future_release(path, review, identities):
    path = Path(path)
    sidecar, outer = Path(str(path) + ".sha256"), Path(str(path) + ".sha256.seal.sha256")
    verify_sidecar(path, sidecar, outer, "M1749 release")
    release = strict_json(path)
    expected = dict(identities)
    expected.update({"m1748_review_sha256": review["review_sha256"],
        "m1748_review_outer_seal_file_sha256": review["outer_seal_file_sha256"]})
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected and
            release.get("authorization") == {
                "analysis_runs": 1, "capture_verifications": 1,
                "result_publications": 1, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            release.get("claim_boundary", {}).get("paper_result") is False,
            "M1749 release authority drift")
    return {"release_sha256": sha256(path),
            "release_outer_seal_file_sha256": sha256(outer)}


def verify_analysis_authority():
    validate_source_contract()
    verify_failure_and_capture_review()
    identities = source_identities()
    review = validate_future_review(FUTURE_REVIEW, identities)
    release = validate_future_release(FUTURE_RELEASE, review, identities)
    return {"identities": identities,
        "m1748_review_sha256": review["review_sha256"],
        "m1748_review_outer_seal_file_sha256": review["outer_seal_file_sha256"],
        "m1749_release_sha256": release["release_sha256"],
        "m1749_release_outer_seal_file_sha256":
            release["release_outer_seal_file_sha256"]}


def canonical_json_bytes(value):
    if type(value) is dict and value.get("schema") == SCHEMA:
        require(_ACTIVE_AUTHORITY is not None,
                "M1747 result serialization lacks active authority")
        value.setdefault("identity", {}).update({
            "m1747_contract_sha256":
                _ACTIVE_AUTHORITY["identities"]["contract_sha256"],
            "m1727_failure_receipt_sha256": FAILED_RECEIPT_SHA256,
            "m1744_capture_review_sha256": M1744_REVIEW_SHA256,
            "m1748_review_sha256": _ACTIVE_AUTHORITY["m1748_review_sha256"],
            "m1749_release_sha256": _ACTIVE_AUTHORITY["m1749_release_sha256"],
            "canonical_sample_order_sha256": SAMPLE_ORDER_SHA256})
        value["analysis_authority"] = {
            "m1727_failure_receipt_double_sealed": True,
            "m1744_capture_review_double_sealed": True,
            "m1748_different_author_review_double_sealed": True,
            "m1749_one_shot_release_double_sealed": True,
            "capture_verified_only_after_m1749_release": True}
        value["schema_identity_repair"] = {
            "predecessor_expected_schema": LEGACY_SAMPLE_ORDER_SCHEMA,
            "canonical_actual_schema": SAMPLE_ORDER_SCHEMA,
            "sample_order_sha256": SAMPLE_ORDER_SHA256,
            "algorithm_changed": False,
            "gates_changed": False,
            "claim_boundary_changed": False}
    return _BASE_CANONICAL_JSON_BYTES(value)


# Rebind the imported exact implementation in this process only.
BASE.SOURCE = SOURCE
BASE.TEST = TEST
BASE.CONTRACT = CONTRACT
BASE.RESULT = RESULT
BASE.WORK = WORK
BASE.SCHEMA = SCHEMA
BASE.STATUS = STATUS
BASE.verify_capture_identity = verify_capture_identity
BASE.canonical_json_bytes = canonical_json_bytes


def run_analysis():
    global _ACTIVE_AUTHORITY
    require(_ACTIVE_AUTHORITY is None, "M1747 analysis already active")
    authority = verify_analysis_authority()
    require(not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)), "fresh M1747 namespaces required")
    _ACTIVE_AUTHORITY = authority
    try:
        return BASE.run_analysis()
    finally:
        _ACTIVE_AUTHORITY = None


def source_self_check():
    validate_source_contract()
    review = verify_failure_and_capture_review()
    require(review["review_sha256"] == M1744_REVIEW_SHA256 and
            not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)), "M1747 inert boundary drift")
    synthetic = {"schema": SAMPLE_ORDER_SCHEMA,
        "samples": [{"global_sample_id": index} for index in range(40)],
        "identity": {"checkpoint_sha256": BASE.CHECKPOINT_SHA256}}
    require(adapt_sample_order_document(synthetic)["schema"] ==
            LEGACY_SAMPLE_ORDER_SCHEMA, "M1747 schema adapter drift")
    return {"status": "PASS_M1747_SOURCE_SELF_CHECK__NO_CAPTURE_NO_ANALYSIS",
        "predecessor_source_sha256": M1727_SOURCE_SHA256,
        "failure_receipt_double_sealed": True,
        "m1744_capture_review_triple_bound": True,
        "canonical_sample_order_sha256": SAMPLE_ORDER_SHA256,
        "algorithm_changed": False, "gates_changed": False,
        "claim_boundary_changed": False, "capture_touched": False,
        "analysis_executed": False, "gpu_runs": 0, "eda_runs": 0,
        "network_access": False,
        "claim_boundary": {"source_only": True, "cycles": False,
            "traffic": False, "aee": False, "speedup": False,
            "energy": False, "rtl": False, "eda": False,
            "paper_result": False}}


def main(argv=None):
    parser = BASE.argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-self-check", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args(argv)
    value = source_self_check() if args.source_self_check else run_analysis()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
