#!/usr/bin/env python3
"""M1707 deployment-complete successor for the exact M1692 TSBG engine.

M1692 is permanently no-retry after its sealed pre-attempt remote failure: a
selective deployment omitted the child-only M1544 handoff validator.  M1707
uses fresh review/release/result/attempt/work/failure identities and changes
only pre-budget deployment closure.  Before the parent subprocess budget and
again inside the clean child before GPU/attempt/model budgets, it SHA-binds the
missing validator and executes the M1558 dependency closure
``verify_bindings -> frozen_layer_specs -> estimate_from_specs``.  The closure
must prove 32 layers and a 7,598,737,368-byte upper bound.

Capture execution remains the exact M1692 -> M1668 -> M1647 -> M1624 engine:
one parent, one clean child, one GPU lease, one capture and no retry.  This
source is inert until a different-author M1708 review and sealed M1709 release
exist.  Source authoring performs no SSH, GPU, capture, attempt or remote work.
CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "tests/test_m1707_motion_ep34_s2_tsbg_deployment_complete_"
    "successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1707_motion_ep34_s2_tsbg_deployment_complete_"
    "successor_source_contract_r1_20260901.json")
M1692_SOURCE = SOURCE.with_name(
    "capture_m1692_motion_ep34_s2_tsbg_authority_shape_repair_successor_r1.py")
M1692_TEST = HW / (
    "tests/test_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source.py")
M1692_CONTRACT = HW / (
    "contracts/m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source_contract_r1_20260901.json")
M1693_REVIEW = HW / (
    "reviews/m1693_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "source_independent_review_r1_20260901")
M1694_RELEASE = HW / (
    "contracts/m1694_m1693_m1692_motion_ep34_s2_tsbg_authority_shape_"
    "repair_capture_release_r1_20260901.json")
M1692_FAILURE = HW / (
    "results/m1692_motion_ep34_s2_tsbg_capture_failed_pre_attempt_20260901")
RUNTIME_VALIDATOR = HW / (
    "system_handoff/scripts/validate_m1544_ep34_sparse_capture_handoff.py")

FUTURE_REVIEW = HW / (
    "reviews/m1708_m1707_motion_ep34_s2_tsbg_deployment_complete_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1709_m1708_m1707_motion_ep34_s2_tsbg_"
    "deployment_complete_capture_release_r1_20260901.json")
RESULT = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
ATTEMPT = HW / (
    "results/.m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901.work")
FAILURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901.failed_no_retry")

REMOTE_TARGET = {
    "host": "ssh.sd5ai.scnet.cn", "port": 10037, "user": "root",
    "repository_root": "/root/private_data/work/sdformer_codex/SDformer"}
CHILD_INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python3.10")
SOURCE_SCHEMA = "m1707_motion_ep34_s2_tsbg_deployment_complete_source_r1_v1"
SOURCE_STATUS = (
    "SOURCE_ONLY__M1692_PRE_ATTEMPT_FAILURE_BOUND__FULL_RUNTIME_CLOSURE_"
    "BEFORE_PARENT_AND_CHILD_BUDGETS__M1708_REVIEW_REQUIRED__NO_CAPTURE")
REVIEW_STATUS = (
    "PASS_M1708_M1707_TSBG_DEPLOYMENT_COMPLETE_SOURCE__"
    "AUTHORIZE_M1709_RELEASE_AUTHORING__NO_CAPTURE")
RELEASE_SCHEMA = (
    "m1709_m1708_m1707_tsbg_deployment_complete_capture_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1707_EP34_S2_TSBG_DEPLOYMENT_COMPLETE_CAPTURE")
ATTEMPT_TOKEN = (
    "M1707_ATTEMPT_CONSUMED__FULL_RUNTIME_CLOSURE_PARENT_AND_CHILD__"
    "EXACT_REMOTE_AND_INTERPRETER__ONE_CHILD__NO_RETRY\n")
PASS_TOKEN = (
    "PASS_M1707_EP34_S2_TSBG_DEPLOYMENT_COMPLETE_CAPTURE__"
    "FRESH_RESULT_HAMMER_REQUIRED")
RESULT_RECEIPT_NAME = "m1707_clean_child_receipt.json"
RESULT_RECEIPT_SCHEMA = (
    "m1707_ep34_s2_tsbg_deployment_complete_receipt_r1_v1")
RESULT_RECEIPT_STATUS = (
    "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED")

M1692_SOURCE_SHA256 = (
    "ea7b300811a71d63456d16b3c3bfe04e7668266e73613ba426e0c8d6ea5e0e58")
M1692_TEST_SHA256 = (
    "ce720955e8d54d40303222732a2edd836c958d5e7b58178baccead6e0ec1f8ad")
M1692_CONTRACT_SHA256 = (
    "cc38745b2a094d6b31367e60a12211075cbc749a72f611b6ef3030b987aabd70")
M1693_REVIEW_SHA256 = (
    "20522d5eadd307a839c949b4b2980cd7f5faa387e44ce55d5e2064c24939e6c8")
M1693_MANIFEST_SHA256 = (
    "adf85c017536aac009042b370da75b7225ec3a85ea7271b7dafca20b8c24868d")
M1693_OUTER_FILE_SHA256 = (
    "4ede41caf0e11aec796a49d4284a82f363151199e0e1a4aa3313fdaa92358c6c")
M1694_RELEASE_SHA256 = (
    "0c807ad5e2b02cdf9a87cae51a461b41d8b22928ca70edf46f034dcb822f256a")
M1692_FAILURE_RECEIPT_SHA256 = (
    "aba412d6443ac945223872e1c71b27b7ae374fa943d970f9793d9e8a45d1b132")
M1692_FAILURE_LOG_SHA256 = (
    "82f9f9d882d204349380bea6e5b9e66f8265e9e2b6c5ce38232dd68f0e6fd181")
M1692_FAILURE_MANIFEST_SHA256 = (
    "aaa9714e5b02140d41704ca2bde033fc05448a63b256f145bfb3681e5eab0b05")
M1692_FAILURE_OUTER_FILE_SHA256 = (
    "c4abf8e08a9c7c2554acd9f2b40904cd5b7503f84ce5d916843e81777b35bf5d")
RUNTIME_VALIDATOR_SHA256 = (
    "463fa7392fa090eda7fdb298fcc10ff896f91a961a0a529a013be2eec47ec240")
EXPECTED_LAYERS = 32
EXPECTED_RESULT_UPPER_BYTES = 7598737368


class M1707Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1707Error(message)


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
        raise M1707Error("missing " + label) from error
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
            M1707Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1692():
    regular_exact(M1692_SOURCE, M1692_SOURCE_SHA256, "exact M1692 source")
    spec = importlib.util.spec_from_file_location("m1707_exact_m1692",
                                                  str(M1692_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot load exact M1692")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SOURCE_SCHEMA ==
            "m1692_motion_ep34_s2_tsbg_authority_shape_repair_source_r1_v1",
            "M1692 schema drift")
    return module


P = load_m1692()
SELECTION_IDENTITY_SHA256 = P.SELECTION_IDENTITY_SHA256
RUNTIME_TAR_SHA256 = P.RUNTIME_TAR_SHA256
CHECKPOINT_SHA256 = P.CHECKPOINT_SHA256
CONFIG_SHA256 = P.CONFIG_SHA256
PROFILE_SHA256 = P.PROFILE_SHA256
DOCS359_SHA256 = P.DOCS359_SHA256


def _verify_sealed_m1692_failure():
    receipt = M1692_FAILURE / "failure_receipt.json"
    launch_log = M1692_FAILURE / (
        "m1692_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
        "r1_20260901.launch.log")
    regular_exact(receipt, M1692_FAILURE_RECEIPT_SHA256,
                  "M1692 failure receipt")
    regular_exact(launch_log, M1692_FAILURE_LOG_SHA256,
                  "M1692 failure log")
    regular_exact(M1692_FAILURE / "SHA256SUMS", M1692_FAILURE_MANIFEST_SHA256,
                  "M1692 failure manifest")
    regular_exact(M1692_FAILURE / "SHA256SUMS.seal.sha256",
                  M1692_FAILURE_OUTER_FILE_SHA256,
                  "M1692 failure outer seal")
    value = strict_json(receipt)
    require(value.get("status") ==
            "FAILED_PRE_ATTEMPT__MISSING_DEPLOYED_HANDOFF_VALIDATOR__M1692_NO_RETRY" and
            value.get("identity", {}).get("launch_log_sha256") ==
                M1692_FAILURE_LOG_SHA256 and
            value.get("failure", {}).get("missing_remote_path") ==
                str(RUNTIME_VALIDATOR.relative_to(ROOT)) and
            value.get("failure", {}).get("canonical_missing_member_sha256") ==
                RUNTIME_VALIDATOR_SHA256 and
            value.get("observed_budget") == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_leases": 0, "attempt_writes": 0,
                "checkpoint_loads": 0, "production_captures": 0,
                "automatic_retry": False} and
            value.get("authorization", {}).get("m1692_retry") is False and
            value.get("authorization", {}).get(
                "future_additive_successor_source_authoring") is True,
            "M1692 failure/no-retry semantics drift")
    return value


def verify_predecessors():
    regular_exact(M1692_SOURCE, M1692_SOURCE_SHA256, "M1692 source")
    regular_exact(M1692_TEST, M1692_TEST_SHA256, "M1692 test")
    regular_exact(M1692_CONTRACT, M1692_CONTRACT_SHA256, "M1692 contract")
    review, manifest, outer = P.P._verify_review_tree(M1693_REVIEW)
    require(sha256(M1693_REVIEW / "review.json") == M1693_REVIEW_SHA256 and
            manifest == M1693_MANIFEST_SHA256 and
            outer == M1693_OUTER_FILE_SHA256 and
            review.get("status") == P.REVIEW_STATUS,
            "M1693 sealed review drift")
    P.P.P._verify_file_seal(M1694_RELEASE)
    regular_exact(M1694_RELEASE, M1694_RELEASE_SHA256, "M1694 release")
    prior_release = strict_json(M1694_RELEASE)
    require(prior_release.get("authorization", {}).get("automatic_retry") is False,
            "M1694 retry drift")
    failed = _verify_sealed_m1692_failure()
    P.verify_predecessors()
    return {"m1692_source_sha256": M1692_SOURCE_SHA256,
            "m1692_failure_receipt_sha256": M1692_FAILURE_RECEIPT_SHA256,
            "m1692_no_retry": True, "failure": failed["failure"]}


def verify_runtime_closure():
    """Full child dependency closure, safe before parent and child budgets."""
    regular_exact(RUNTIME_VALIDATOR, RUNTIME_VALIDATOR_SHA256,
                  "M1544 handoff validator before closure")
    m1558 = P.P.P.P.load_m1558()
    bindings = m1558.M1552.verify_bindings()
    specs = m1558.frozen_layer_specs()
    estimate = m1558.estimate_from_specs(specs, 40)
    require(type(bindings) is dict and type(specs) is list and
            len(specs) == EXPECTED_LAYERS and
            int(estimate.get("result_upper_bytes", -1)) ==
                EXPECTED_RESULT_UPPER_BYTES,
            "M1558 full runtime closure/estimate drift")
    regular_exact(RUNTIME_VALIDATOR, RUNTIME_VALIDATOR_SHA256,
                  "M1544 handoff validator after closure")
    return {"status": "PASS_M1707_FULL_RUNTIME_CLOSURE_BEFORE_BUDGET",
            "validator_path": str(RUNTIME_VALIDATOR.relative_to(ROOT)),
            "validator_sha256": RUNTIME_VALIDATOR_SHA256,
            "m1558_verify_bindings": True,
            "frozen_layer_specs": len(specs),
            "estimated_result_upper_bytes":
                int(estimate["result_upper_bytes"]),
            "gpu_runs": 0, "attempt_writes": 0}


def expected_review_identity():
    return {"source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1692_source_sha256": M1692_SOURCE_SHA256,
        "m1692_test_sha256": M1692_TEST_SHA256,
        "m1692_contract_sha256": M1692_CONTRACT_SHA256,
        "m1692_failure_receipt_sha256": M1692_FAILURE_RECEIPT_SHA256,
        "m1692_failure_log_sha256": M1692_FAILURE_LOG_SHA256,
        "m1692_failure_manifest_sha256": M1692_FAILURE_MANIFEST_SHA256,
        "runtime_validator_sha256": RUNTIME_VALIDATOR_SHA256,
        "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
        "runtime_tar_sha256": RUNTIME_TAR_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "docs359_sha256": DOCS359_SHA256}


def validate_source_contract():
    value = strict_json(SOURCE_CONTRACT)
    require(value.get("schema") == SOURCE_SCHEMA and
            value.get("status") == SOURCE_STATUS and
            value.get("source") == {"path": str(SOURCE.relative_to(ROOT)),
                "sha256": sha256(SOURCE)} and
            value.get("test") == {"path": str(TEST.relative_to(ROOT)),
                "sha256": sha256(TEST)} and
            value.get("runtime_closure") == {
                "validator_path": str(RUNTIME_VALIDATOR.relative_to(ROOT)),
                "validator_sha256": RUNTIME_VALIDATOR_SHA256,
                "m1558_verify_bindings": True,
                "frozen_layer_specs": EXPECTED_LAYERS,
                "estimated_result_upper_bytes": EXPECTED_RESULT_UPPER_BYTES,
                "before_parent_subprocess_budget": True,
                "before_clean_child_gpu_attempt_budget": True} and
            value.get("future_authority") == {
                "review": str(FUTURE_REVIEW.relative_to(ROOT)),
                "release": str(FUTURE_RELEASE.relative_to(ROOT))},
            "M1707 source contract identity drift")
    require(value.get("authorization") == {
            "different_author_review": True, "release_authoring": False,
            "parent_launch": False, "remote_launch": False,
            "capture": False, "gpu": False, "attempt_creation": False,
            "automatic_retry": False},
            "M1707 source contract authorizes runtime")
    return value


def validate_future_authorities():
    review, manifest_sha, outer_sha = P.P._verify_review_tree(FUTURE_REVIEW)
    expected = expected_review_identity()
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == expected and
            review.get("authorization") == {"release_authoring": True,
                "capture": False, "gpu": False, "automatic_retry": False},
            "M1708 review mismatch")
    P.P.P._verify_file_seal(FUTURE_RELEASE)
    release = strict_json(FUTURE_RELEASE)
    identity = dict(expected,
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=manifest_sha,
        review_outer_file_sha256=outer_sha)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == identity and
            release.get("authorization") == {"parent_calls": 1,
                "clean_child_processes": 1, "gpu_runs": 1,
                "production_captures": 1, "automatic_retry": False,
                "all_other_runs": 0} and
            release.get("namespaces") == {
                "result": str(RESULT.relative_to(ROOT)),
                "attempt": str(ATTEMPT.relative_to(ROOT)),
                "work": str(WORK.relative_to(ROOT)),
                "failure": str(FAILURE.relative_to(ROOT))} and
            release.get("pre_budget_runtime_closure") == {
                "validator_path": str(RUNTIME_VALIDATOR.relative_to(ROOT)),
                "validator_sha256": RUNTIME_VALIDATOR_SHA256,
                "m1558_verify_bindings": True,
                "frozen_layer_specs": EXPECTED_LAYERS,
                "estimated_result_upper_bytes": EXPECTED_RESULT_UPPER_BYTES,
                "before_parent_subprocess_budget": True,
                "before_clean_child_gpu_attempt_budget": True} and
            release.get("remote_target") == REMOTE_TARGET and
            release.get("claim_boundary") == {"tsbg_dse": False,
                "aee": False, "rtl": False, "eda": False,
                "performance": False, "paper_result": False},
            "M1709 release mismatch")
    interpreter = release.get("child_interpreter", {})
    require(interpreter.get("path") == str(CHILD_INTERPRETER),
            "M1709 child interpreter path drift")
    regular_exact(CHILD_INTERPRETER, interpreter.get("sha256"),
                  "M1709 child interpreter")
    return release


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and all("m1707_" in path.name for path in paths),
            "M1707 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1707 namespace is not fresh")


def write_child_receipt(root, release, load_audit, validation):
    receipt = {"schema": RESULT_RECEIPT_SCHEMA,
        "status": RESULT_RECEIPT_STATUS,
        "identity": {"source_sha256": sha256(SOURCE),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "release_sha256": sha256(FUTURE_RELEASE),
            "m1692_source_sha256": M1692_SOURCE_SHA256,
            "m1692_failure_receipt_sha256": M1692_FAILURE_RECEIPT_SHA256,
            "runtime_validator_sha256": RUNTIME_VALIDATOR_SHA256,
            "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
            "runtime_tar_sha256": RUNTIME_TAR_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "profile_sha256": PROFILE_SHA256},
        "checkpoint_load": dict((key, int(load_audit.get(key, -1))) for key in (
            "missing_count", "unexpected_count", "overlay_missing_count",
            "overlay_unexpected_count")),
        "population": {"samples": 40, "frames": int(validation["frames"]),
            "fc_tokens": int(validation["fc_tokens"]),
            "patch_histogram_rows": int(validation["patch_histogram_rows"])},
        "execution": {"full_runtime_closure_before_parent_budget": True,
            "full_runtime_closure_before_clean_child_budget": True,
            "m1558_verify_bindings": True,
            "frozen_layer_specs": EXPECTED_LAYERS,
            "estimated_result_upper_bytes": EXPECTED_RESULT_UPPER_BYTES,
            "exact_remote_target": dict(REMOTE_TARGET),
            "clean_child_processes": 1, "automatic_retry": False},
        "claim_boundary": {"capture_payload_only": True,
            "fresh_result_hammer_required": True,
            "hardware_quantization_authority": False,
            "model_bit_exact": False, "tsbg_dse": False, "aee": False,
            "cycles": False, "traffic": False, "energy": False,
            "speedup": False, "rtl": False, "eda": False,
            "paper_result": False}}
    (root / RESULT_RECEIPT_NAME).write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    P.P.P.P.seal_result(root)
    return receipt


@contextlib.contextmanager
def _bound_exact_m1692():
    replacements = {"SOURCE": SOURCE, "TEST": TEST,
        "SOURCE_CONTRACT": SOURCE_CONTRACT, "FUTURE_REVIEW": FUTURE_REVIEW,
        "FUTURE_RELEASE": FUTURE_RELEASE, "RESULT": RESULT,
        "ATTEMPT": ATTEMPT, "WORK": WORK, "FAILURE": FAILURE,
        "SOURCE_SCHEMA": SOURCE_SCHEMA, "SOURCE_STATUS": SOURCE_STATUS,
        "REVIEW_STATUS": REVIEW_STATUS, "RELEASE_SCHEMA": RELEASE_SCHEMA,
        "RELEASE_STATUS": RELEASE_STATUS, "ATTEMPT_TOKEN": ATTEMPT_TOKEN,
        "PASS_TOKEN": PASS_TOKEN, "RESULT_RECEIPT_NAME": RESULT_RECEIPT_NAME,
        "RESULT_RECEIPT_SCHEMA": RESULT_RECEIPT_SCHEMA,
        "RESULT_RECEIPT_STATUS": RESULT_RECEIPT_STATUS,
        "validate_source_contract": validate_source_contract,
        "validate_future_authorities": validate_future_authorities,
        "require_fresh_namespaces": require_fresh_namespaces,
        "write_child_receipt": write_child_receipt}
    originals = dict((name, getattr(P, name)) for name in replacements)
    try:
        for name, value in replacements.items():
            setattr(P, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(P, name, value)


def fixed_clean_child():
    verify_predecessors()
    verify_runtime_closure()
    with _bound_exact_m1692():
        return P.fixed_clean_child()


def launch_parent():
    verify_predecessors()
    verify_runtime_closure()
    with _bound_exact_m1692():
        return P.launch_parent()


def source_self_check():
    verify_predecessors()
    closure = verify_runtime_closure()
    validate_source_contract()
    require_fresh_namespaces()
    future = (FUTURE_REVIEW, FUTURE_RELEASE,
              Path(str(FUTURE_RELEASE) + ".sha256"),
              Path(str(FUTURE_RELEASE) + ".sha256.seal.sha256"))
    require(all(not os.path.lexists(str(path)) for path in future),
            "future M1708/M1709 authority exists at source stage")
    return {"status":
            "PASS_M1707_SOURCE_SELF_CHECK__DEPLOYMENT_COMPLETE__NO_CAPTURE",
        "source_status": SOURCE_STATUS,
        "m1692_exact_engine_reused": True,
        "m1692_no_retry": True,
        "runtime_closure": closure,
        "parent_pre_budget_runtime_closure": True,
        "clean_child_pre_budget_runtime_closure": True,
        "remote_target": dict(REMOTE_TARGET),
        "child_interpreter_path": str(CHILD_INTERPRETER),
        "result_namespace": str(RESULT.relative_to(ROOT)),
        "remote_connected": False, "checkpoint_loaded": False,
        "parent_processes": 0, "child_processes": 0, "gpu_runs": 0,
        "capture_runs": 0, "attempt_writes": 0, "automatic_retry": False,
        "claim_boundary": {"source_only": True, "capture": False,
            "gpu": False, "aee": False, "cycles": False,
            "traffic": False, "energy": False, "speedup": False,
            "rtl": False, "eda": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-self-check", action="store_true")
    modes.add_argument("--launch-parent", action="store_true")
    modes.add_argument("--fixed-clean-child", action="store_true",
                       help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.source_self_check:
        print(json.dumps(source_self_check(), indent=2, sort_keys=True,
                         allow_nan=False))
        return 0
    if args.launch_parent:
        return launch_parent()
    return fixed_clean_child()


if __name__ == "__main__":
    raise SystemExit(main())
