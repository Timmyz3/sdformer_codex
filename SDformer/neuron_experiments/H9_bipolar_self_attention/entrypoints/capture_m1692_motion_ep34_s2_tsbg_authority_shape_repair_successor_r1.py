#!/usr/bin/env python3
"""M1692 additive source-only repair for the ep34 TSBG authority shape.

M1668 remains immutable.  Its sealed M1669 review is recursively intact but
cannot be consumed by M1668's exact review schema.  M1692 binds that failure
and its additive correction, then gives a fresh M1693/M1694 review/release
namespace an exact validator shape.

All capture semantics continue through the exact M1668 -> M1647 -> M1624
chain.  Runtime handoff, current checkpoint/config/profile entities, complete
``build_runtime`` before both parent and child budgets, exclusive GPU lease,
O_EXCL attempt consumption, one child, one capture, and no retry are preserved.
The release also binds the exact remote target and child interpreter identity.

This source revision is inert.  M1693 review and M1694 release are absent.  It
performs no SSH, remote write, checkpoint load, GPU run, attempt, or capture.
Python syntax is compatible with CPython 3.6.
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
    "tests/test_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source_contract_r1_20260901.json")
M1668_SOURCE = SOURCE.with_name(
    "capture_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "successor_r1.py")
M1668_TEST = HW / (
    "tests/test_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "source.py")
M1668_CONTRACT = HW / (
    "contracts/m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "source_contract_r1_20260901.json")
M1669_INVALID = HW / (
    "reviews/m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_"
    "rebind_source_independent_review_r1_20260901")
M1669_CORRECTION = HW / (
    "reviews/m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_"
    "rebind_source_independent_review_schema_correction_r2_20260901")

FUTURE_REVIEW = HW / (
    "reviews/m1693_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1694_m1693_m1692_motion_ep34_s2_tsbg_authority_shape_"
    "repair_capture_release_r1_20260901.json")
RESULT = HW / (
    "results/m1692_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1692_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1692_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1692_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.failed_no_retry")

REMOTE_TARGET = {
    "host": "ssh.sd5ai.scnet.cn",
    "port": 10037,
    "user": "root",
    "repository_root": "/root/private_data/work/sdformer_codex/SDformer",
}
CHILD_INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python3.10")

SOURCE_SCHEMA = (
    "m1692_motion_ep34_s2_tsbg_authority_shape_repair_source_r1_v1")
SOURCE_STATUS = (
    "SOURCE_ONLY__M1668_RUNTIME_AND_ENTITY_GATES_PRESERVED__"
    "AUTHORITY_SHAPE_REPAIRED__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_CAPTURE")
REVIEW_STATUS = (
    "PASS_M1693_M1692_TSBG_AUTHORITY_SHAPE_REPAIR_SOURCE__"
    "AUTHORIZE_M1694_RELEASE_AUTHORING__NO_CAPTURE")
RELEASE_SCHEMA = (
    "m1694_m1693_m1692_tsbg_authority_shape_repair_capture_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1692_EP34_S2_TSBG_AUTHORITY_SHAPE_REPAIR_CAPTURE")
ATTEMPT_TOKEN = (
    "M1692_ATTEMPT_CONSUMED__M1668_RUNTIME_ENTITY_GATES_PASS__"
    "EXACT_REMOTE_AND_INTERPRETER__ONE_CHILD__NO_RETRY\n")
PASS_TOKEN = (
    "PASS_M1692_EP34_S2_TSBG_AUTHORITY_SHAPE_REPAIR_CAPTURE__"
    "FRESH_RESULT_HAMMER_REQUIRED")
RESULT_RECEIPT_NAME = "m1692_clean_child_receipt.json"
RESULT_RECEIPT_SCHEMA = (
    "m1692_ep34_s2_tsbg_authority_shape_repair_receipt_r1_v1")
RESULT_RECEIPT_STATUS = (
    "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED")

M1668_SOURCE_SHA256 = (
    "7e728162de630da2086dee5a39536fc9c4141d24dcde4f4840549c9aabc77d8b")
M1668_TEST_SHA256 = (
    "ef36f416df749fc646fc901b662dc1fac7de4d9872989e29f5ba21e34c202fee")
M1668_CONTRACT_SHA256 = (
    "723e8797889d231e36dca343281abff7eccb4c3080f4231e2746c4a083100165")
M1669_INVALID_REVIEW_SHA256 = (
    "e8b6f337bda28049942cbe8088bd8953211af37fc296503b88cde6837400db6e")
M1669_INVALID_MANIFEST_SHA256 = (
    "4168cfab358e2c7350c967099f34833daf10287e631cad5247b8a5ab696c7192")
M1669_INVALID_OUTER_SHA256 = (
    "f5425757663ff9d1250c1196419ce0ee0e400def1d6095fd3facb120b664381e")
M1669_CORRECTION_REVIEW_SHA256 = (
    "f8f91dff20c1b5709e6d4224486b1f36d781f9c5359daabb6c67afac3d747c6a")
M1669_CORRECTION_MANIFEST_SHA256 = (
    "045d7f72ab72e815c00a902bc6c9cb01d2542036d64866cceae96a17be661300")
M1669_CORRECTION_OUTER_SHA256 = (
    "94fb9b3bc7bb41309628c6d8e3814c758f910c8c36f9d0f64c141aa9d7796557")


class M1692Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1692Error(message)


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
        raise M1692Error("missing " + label) from error
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
                           M1692Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1668():
    regular_exact(M1668_SOURCE, M1668_SOURCE_SHA256, "M1668 source")
    spec = importlib.util.spec_from_file_location("m1692_exact_m1668", M1668_SOURCE)
    require(spec is not None and spec.loader is not None,
            "cannot load exact M1668 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load_m1668()

# Re-export immutable algorithm/runtime identities for exact review generation.
SELECTION_IDENTITY_SHA256 = P.SELECTION_IDENTITY_SHA256
RUNTIME_TAR_SHA256 = P.RUNTIME_TAR_SHA256
CHECKPOINT_SHA256 = P.CHECKPOINT_SHA256
CONFIG_SHA256 = P.CONFIG_SHA256
PROFILE_SHA256 = P.PROFILE_SHA256
DOCS359_SHA256 = P.DOCS359_SHA256


def _verify_review_tree(root, review_sha, manifest_sha, outer_sha, label):
    review, observed_manifest, observed_outer = P._verify_review_tree(root)
    require(sha256(Path(root) / "review.json") == review_sha and
            observed_manifest == manifest_sha and observed_outer == outer_sha,
            label + " sealed identity drift")
    return review


def verify_predecessors():
    regular_exact(M1668_SOURCE, M1668_SOURCE_SHA256, "M1668 source")
    regular_exact(M1668_TEST, M1668_TEST_SHA256, "M1668 test")
    regular_exact(M1668_CONTRACT, M1668_CONTRACT_SHA256, "M1668 contract")
    invalid = _verify_review_tree(
        M1669_INVALID, M1669_INVALID_REVIEW_SHA256,
        M1669_INVALID_MANIFEST_SHA256, M1669_INVALID_OUTER_SHA256,
        "M1669 invalid canonical review")
    require("score" not in invalid and invalid.get("score_out_of_100") == 98 and
            invalid.get("status") == P.REVIEW_STATUS,
            "M1669 invalid-schema witness drift")
    correction = _verify_review_tree(
        M1669_CORRECTION, M1669_CORRECTION_REVIEW_SHA256,
        M1669_CORRECTION_MANIFEST_SHA256, M1669_CORRECTION_OUTER_SHA256,
        "M1669 correction review")
    require(correction.get("status") ==
            "FAIL_CLOSED_M1669_CANONICAL_REVIEW_SCHEMA_MISMATCH__"
            "SUPERSEDED__NO_M1670_RELEASE" and
            correction.get("canonical_validator_error") ==
            "M1668Error: M1669 review mismatch" and
            correction.get("authorization", {}).get(
                "m1670_release_authoring") is False,
            "M1669 correction semantics drift")
    P.verify_predecessors()
    return {"m1668_source_sha256": M1668_SOURCE_SHA256,
            "invalid_review_bound": True, "correction_bound": True}


def expected_review_identity():
    return {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1668_source_sha256": M1668_SOURCE_SHA256,
        "m1669_invalid_review_sha256": M1669_INVALID_REVIEW_SHA256,
        "m1669_correction_review_sha256": M1669_CORRECTION_REVIEW_SHA256,
        "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
        "runtime_tar_sha256": RUNTIME_TAR_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }


def validate_source_contract():
    value = strict_json(SOURCE_CONTRACT)
    require(value.get("schema") == SOURCE_SCHEMA and
            value.get("status") == SOURCE_STATUS and
            value.get("source") == {
                "path": str(SOURCE.relative_to(ROOT)),
                "sha256": sha256(SOURCE)} and
            value.get("test") == {
                "path": str(TEST.relative_to(ROOT)),
                "sha256": sha256(TEST)} and
            value.get("predecessor") == {
                "m1668_source_sha256": M1668_SOURCE_SHA256,
                "m1668_test_sha256": M1668_TEST_SHA256,
                "m1668_contract_sha256": M1668_CONTRACT_SHA256,
                "m1669_invalid_review_sha256": M1669_INVALID_REVIEW_SHA256,
                "m1669_correction_review_sha256":
                    M1669_CORRECTION_REVIEW_SHA256} and
            value.get("runtime_identity") == {
                "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
                "runtime_tar_sha256": RUNTIME_TAR_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "config_sha256": CONFIG_SHA256,
                "profile_sha256": PROFILE_SHA256} and
            value.get("remote_target") == REMOTE_TARGET and
            value.get("child_interpreter_path") == str(CHILD_INTERPRETER),
            "M1692 source contract identity drift")
    require(value.get("future_authority") == {
                "review": str(FUTURE_REVIEW.relative_to(ROOT)),
                "release": str(FUTURE_RELEASE.relative_to(ROOT))} and
            value.get("capture_consumer") == {
                "result_namespace": str(RESULT.relative_to(ROOT)),
                "receipt_name": RESULT_RECEIPT_NAME,
                "receipt_schema": RESULT_RECEIPT_SCHEMA,
                "receipt_status": RESULT_RECEIPT_STATUS,
                "fresh_different_author_result_hammer_required": True,
                "payload_identity_includes_source_contract_release_runtime":
                    True} and
            value.get("authorization") == {
                "different_author_review": True,
                "release_authoring": False,
                "parent_launch": False,
                "remote_launch": False,
                "capture": False,
                "gpu": False,
                "attempt_creation": False,
                "automatic_retry": False},
            "M1692 source contract authorizes runtime work")
    return value


def validate_future_authorities():
    review, manifest_sha, outer_sha = P._verify_review_tree(FUTURE_REVIEW)
    expected = expected_review_identity()
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == expected and
            review.get("authorization") == {
                "release_authoring": True,
                "capture": False,
                "gpu": False,
                "automatic_retry": False},
            "M1693 review mismatch")
    P.P._verify_file_seal(FUTURE_RELEASE)
    release = strict_json(FUTURE_RELEASE)
    release_identity = dict(expected,
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=manifest_sha,
        review_outer_file_sha256=outer_sha)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == release_identity and
            release.get("authorization") == {
                "parent_calls": 1,
                "clean_child_processes": 1,
                "gpu_runs": 1,
                "production_captures": 1,
                "automatic_retry": False,
                "all_other_runs": 0} and
            release.get("namespaces") == {
                "result": str(RESULT.relative_to(ROOT)),
                "attempt": str(ATTEMPT.relative_to(ROOT)),
                "work": str(WORK.relative_to(ROOT)),
                "failure": str(FAILURE.relative_to(ROOT))} and
            release.get("pre_budget_preflight") == {
                "runtime_m1257_canonical": True,
                "current_entity_exact": True,
                "build_runtime_before_parent_subprocess": True,
                "build_runtime_before_child_gpu_attempt": True,
                "exact_remote_target": True,
                "exact_child_interpreter": True} and
            release.get("remote_target") == REMOTE_TARGET and
            release.get("claim_boundary") == {
                "tsbg_dse": False,
                "aee": False,
                "rtl": False,
                "eda": False,
                "performance": False,
                "paper_result": False},
            "M1694 release mismatch")
    interpreter = release.get("child_interpreter", {})
    require(interpreter.get("path") == str(CHILD_INTERPRETER),
            "M1694 child interpreter path drift")
    regular_exact(CHILD_INTERPRETER, interpreter.get("sha256"),
                  "M1694 child interpreter")
    return release


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and all("m1692_" in path.name for path in paths),
            "M1692 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1692 namespace is not fresh")


def write_child_receipt(root, release, load_audit, validation):
    receipt = {
        "schema": RESULT_RECEIPT_SCHEMA,
        "status": RESULT_RECEIPT_STATUS,
        "identity": {
            "source_sha256": sha256(SOURCE),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "release_sha256": sha256(FUTURE_RELEASE),
            "m1668_source_sha256": M1668_SOURCE_SHA256,
            "m1669_correction_review_sha256": M1669_CORRECTION_REVIEW_SHA256,
            "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
            "runtime_tar_sha256": RUNTIME_TAR_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "profile_sha256": PROFILE_SHA256,
        },
        "checkpoint_load": dict((key, int(load_audit.get(key, -1))) for key in (
            "missing_count", "unexpected_count", "overlay_missing_count",
            "overlay_unexpected_count")),
        "population": {
            "samples": 40,
            "frames": int(validation["frames"]),
            "fc_tokens": int(validation["fc_tokens"]),
            "patch_histogram_rows": int(validation["patch_histogram_rows"]),
        },
        "execution": {
            "runtime_and_entity_build_preflight_before_parent_and_child_budget":
                True,
            "exact_remote_target": dict(REMOTE_TARGET),
            "clean_child_processes": 1,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "capture_payload_only": True,
            "fresh_result_hammer_required": True,
            "hardware_quantization_authority": False,
            "model_bit_exact": False,
            "tsbg_dse": False,
            "aee": False,
            "cycles": False,
            "traffic": False,
            "energy": False,
            "speedup": False,
            "rtl": False,
            "eda": False,
            "paper_result": False,
        },
    }
    (root / RESULT_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    P.P.P.seal_result(root)
    return receipt


@contextlib.contextmanager
def _bound_exact_m1668():
    replacements = {
        "SOURCE": SOURCE,
        "TEST": TEST,
        "SOURCE_CONTRACT": SOURCE_CONTRACT,
        "FUTURE_REVIEW": FUTURE_REVIEW,
        "FUTURE_RELEASE": FUTURE_RELEASE,
        "RESULT": RESULT,
        "ATTEMPT": ATTEMPT,
        "WORK": WORK,
        "FAILURE": FAILURE,
        "SOURCE_SCHEMA": SOURCE_SCHEMA,
        "SOURCE_STATUS": SOURCE_STATUS,
        "REVIEW_STATUS": REVIEW_STATUS,
        "RELEASE_SCHEMA": RELEASE_SCHEMA,
        "RELEASE_STATUS": RELEASE_STATUS,
        "ATTEMPT_TOKEN": ATTEMPT_TOKEN,
        "PASS_TOKEN": PASS_TOKEN,
        "validate_source_contract": validate_source_contract,
        "validate_future_authorities": validate_future_authorities,
        "require_fresh_namespaces": require_fresh_namespaces,
        "write_child_receipt": write_child_receipt,
    }
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
    P.preflight_runtime_binding()
    with _bound_exact_m1668():
        return P.fixed_clean_child()


def launch_parent():
    verify_predecessors()
    P.preflight_runtime_binding()
    with _bound_exact_m1668():
        return P.launch_parent()


def source_self_check():
    verify_predecessors()
    identity = P.selection_identity()
    handoff = P.verify_runtime_handoff_source()
    validate_source_contract()
    require_fresh_namespaces()
    future_paths = (
        FUTURE_REVIEW,
        FUTURE_RELEASE,
        Path(str(FUTURE_RELEASE) + ".sha256"),
        Path(str(FUTURE_RELEASE) + ".sha256.seal.sha256"),
    )
    require(all(not os.path.lexists(str(path)) for path in future_paths),
            "future M1693/M1694 authority must be absent at authoring")
    return {
        "status": "PASS_M1692_SOURCE_SELF_CHECK__AUTHORITY_SHAPE_REPAIRED__NO_CAPTURE",
        "source_status": SOURCE_STATUS,
        "m1668_runtime_and_entity_gates_preserved": True,
        "m1669_invalid_review_bound": True,
        "m1669_correction_bound": True,
        "runtime_handoff_files": handoff["archive_files"],
        "runtime_canonical_files": handoff["canonical_files"],
        "runtime_preflight_status":
            "BOUND_TO_M1668_PARENT_AND_CHILD__NOT_EXECUTED_LOCALLY",
        "selected_candidate_id": "resume_ep34",
        "selected_epoch": 34,
        "configuration_content_unchanged":
            identity["configuration_frozen_selection_entity"]["sha256"] ==
            identity["configuration_current_capture_entity"]["sha256"],
        "remote_target": dict(REMOTE_TARGET),
        "child_interpreter_path": str(CHILD_INTERPRETER),
        "capture_consumer": {
            "result_namespace": str(RESULT.relative_to(ROOT)),
            "receipt_name": RESULT_RECEIPT_NAME,
            "receipt_schema": RESULT_RECEIPT_SCHEMA,
            "receipt_status": RESULT_RECEIPT_STATUS,
            "fresh_different_author_result_hammer_required": True,
        },
        "remote_connected": False,
        "checkpoint_loaded": False,
        "parent_processes": 0,
        "child_processes": 0,
        "gpu_runs": 0,
        "capture_runs": 0,
        "attempt_writes": 0,
        "automatic_retry": False,
        "claim_boundary": {
            "source_only": True,
            "capture": False,
            "gpu": False,
            "aee": False,
            "cycles": False,
            "traffic": False,
            "energy": False,
            "speedup": False,
            "rtl": False,
            "eda": False,
            "paper_result": False,
        },
    }


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
