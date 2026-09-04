#!/usr/bin/env python3
"""M1647 deployment-complete successor of exact M1624/M1640.

The failed A800 deployment reached neither the attempt nor the GPU: a plain
Git archive omitted the M1314 ``author_test.log`` even though M1314's recursive
seal requires it.  This successor adds a complete, machine-readable inventory
of every reachable predecessor seal and makes that inventory the first parent
and child preflight.  The omitted byte-exact member is now tracked, but remains
explicitly recorded as the failure root cause and a regression target.

M1624's clean-child capture implementation is reused byte-for-byte inside an
isolated, reversible binding of only source/authority/namespace globals.  This
authoring revision is inert: the new M1648 review and M1649 release do not
exist.  It does not connect remotely, open capture payload, create an attempt,
load a checkpoint, use a GPU, run capture/EDA, or authorize a retry.

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
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_source.py"
SOURCE_CONTRACT = HW / (
    "contracts/m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_"
    "source_contract_r1_20260901.json")
DEPLOYMENT_MANIFEST = HW / (
    "contracts/m1647_motion_ep34_s2_tsbg_runtime_deployment_completeness_"
    "manifest_r1_20260901.json")
M1624_SOURCE = SOURCE.with_name(
    "capture_m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "successor_r1.py")
M1624_TEST = HW / "tests/test_m1624_motion_ep34_s2_tsbg_clean_child_source.py"
M1624_CONTRACT = HW / (
    "contracts/m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "source_contract_r1_20260901.json")
M1626_RELEASE = HW / (
    "contracts/m1626_m1625_m1624_motion_ep34_s2_tsbg_clean_child_capture_"
    "release_r1_20260901.json")
M1640 = HW / (
    "reviews/m1640_m1626_m1625_m1624_tsbg_s2_clean_child_capture_release_"
    "hammer_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

FUTURE_REVIEW = HW / (
    "reviews/m1648_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_"
    "child_source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1649_m1648_m1647_motion_ep34_s2_tsbg_deployment_complete_"
    "clean_child_capture_release_r1_20260901.json")

RESULT = HW / (
    "results/m1647_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1647_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1647_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1647_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.failed_no_retry")

SOURCE_SCHEMA = (
    "m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_source_r1_v1")
SOURCE_STATUS = (
    "SOURCE_ONLY__DEPLOYMENT_COMPLETE_PREFLIGHT_FIRST__"
    "DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_CAPTURE")
REVIEW_STATUS = (
    "PASS_M1648_M1647_DEPLOYMENT_COMPLETE_CLEAN_CHILD_SOURCE__"
    "AUTHORIZE_RELEASE_AUTHORING__NO_CAPTURE")
RELEASE_SCHEMA = (
    "m1649_m1648_m1647_deployment_complete_clean_child_capture_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1647_EP34_S2_TSBG_DEPLOYMENT_COMPLETE_CLEAN_CHILD_CAPTURE")
ATTEMPT_TOKEN = (
    "M1647_ATTEMPT_CONSUMED__DEPLOYMENT_PREFLIGHT_PASS__"
    "ONE_CLEAN_CHILD__AUTOMATIC_RETRY_FALSE\n")
PASS_TOKEN = (
    "PASS_M1647_EP34_S2_TSBG_DEPLOYMENT_COMPLETE_CLEAN_CHILD_CAPTURE__"
    "FRESH_RESULT_HAMMER_REQUIRED")

M1624_SOURCE_SHA256 = (
    "ad36ab02b598f28458ed226f816b47281b7d388fddfe80bc7ea15155709ba76f")
M1624_TEST_SHA256 = (
    "5b44434df85b2832435ded94258a9a9f038f902ed6e77de1f4b7d690c497891b")
M1624_CONTRACT_SHA256 = (
    "2ba3445c2c40c437124c62f49881db1b8443344aa19afc504f4f45aa1c1eacd9")
M1626_RELEASE_SHA256 = (
    "ce15529bcfceda5be92084bdb411330b0c56c8fe47c7024dd9b35a1a0490e273")
M1640_REVIEW_SHA256 = (
    "dab55414c6a88219cada4d1fe378f42964350c9912b17affba30f154e676efc8")
M1640_MANIFEST_SHA256 = (
    "70534d1d0a3844737ddf4317ba00c14947e7e27e2cd016dea16f63010024c555")
M1640_OUTER_SHA256 = (
    "b94ad5b294a492c8722297eb72d2aa1a26e1d22a9badbb0593289b5aa1c1602e")
DEPLOYMENT_MANIFEST_SHA256 = (
    "a5c8eee213fe30df5a3781ca5f7c1458b49029e5542d315e6505213c2ea4c6bc")
DEPLOYMENT_MANIFEST_SIDECAR_SHA256 = (
    "b4ee26a27ea08b27192a171046d03e9cb7e884bd9f078e734aa74d60b21cdd9c")
DEPLOYMENT_MANIFEST_OUTER_FILE_SHA256 = (
    "3ec9a9727e0e3adea3a3b96c6c8bccea17dcf95c1fbf9e7664e376fa76c10f4c")
CHECKPOINT_SHA256 = (
    "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
M1314_AUTHOR_TEST = HW / (
    "reviews/m1314_m1313_motion_ep34_final_unified_capture_production_launch_"
    "blind_hammer_r1_20260831/author_test.log")
M1314_AUTHOR_TEST_SHA256 = (
    "4581ebeb0ead646e949468bf40f6f1bda9047cc112e899de30b913ca35be6bc5")


class M1647Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1647Error(message)


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
        raise M1647Error("missing " + label) from error
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
                           M1647Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _safe_member(root, name):
    require(type(name) is str and name and "\\" not in name,
            "unsafe sealed member name")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "unsafe sealed member path")
    if relative.parts[0] in ("hw_autoresearch_nts07", "neuron_experiments"):
        path = ROOT / relative
    else:
        path = root / relative
    require(str(path.resolve()).startswith(str(ROOT.resolve()) + os.sep),
            "sealed member escapes repository")
    return path


def _manifest_seals(value):
    rows = value.get("runtime_predecessor_seals")
    require(type(rows) is list and len(rows) == 16,
            "deployment predecessor seal count drift")
    return rows


def verify_deployment_completeness():
    """First-stage filesystem preflight; no subprocess or runtime budget."""
    regular_exact(DEPLOYMENT_MANIFEST, DEPLOYMENT_MANIFEST_SHA256,
                  "M1647 deployment manifest")
    deployment_sidecar = Path(str(DEPLOYMENT_MANIFEST) + ".sha256")
    deployment_outer = Path(str(DEPLOYMENT_MANIFEST) +
                            ".sha256.seal.sha256")
    regular_exact(deployment_sidecar, DEPLOYMENT_MANIFEST_SIDECAR_SHA256,
                  "M1647 deployment manifest sidecar")
    regular_exact(deployment_outer, DEPLOYMENT_MANIFEST_OUTER_FILE_SHA256,
                  "M1647 deployment manifest outer")
    require(deployment_sidecar.read_text(encoding="ascii") ==
            DEPLOYMENT_MANIFEST_SHA256 + "  " + DEPLOYMENT_MANIFEST.name + "\n" and
            deployment_outer.read_text(encoding="ascii") ==
            DEPLOYMENT_MANIFEST_SIDECAR_SHA256 + "  " +
            deployment_sidecar.name + "\n",
            "M1647 deployment manifest double seal drift")
    value = strict_json(DEPLOYMENT_MANIFEST)
    require(value.get("schema") ==
            "m1647_motion_ep34_s2_tsbg_runtime_deployment_completeness_manifest_r1_v1" and
            value.get("status") ==
            "SOURCE_STAGE_DEPLOYMENT_INVENTORY__NO_REMOTE_NO_CAPTURE",
            "deployment manifest schema/status drift")
    exact = value.get("exact_predecessor", {})
    require(exact == {
        "m1624_source_path": str(M1624_SOURCE.relative_to(ROOT)),
        "m1624_source_sha256": M1624_SOURCE_SHA256,
        "m1624_source_contract_sha256": M1624_CONTRACT_SHA256,
        "m1626_release_sha256": M1626_RELEASE_SHA256,
        "m1640_review_sha256": M1640_REVIEW_SHA256,
        "m1640_manifest_sha256": M1640_MANIFEST_SHA256,
        "m1640_outer_seal_file_sha256": M1640_OUTER_SHA256,
    }, "deployment predecessor identity drift")
    previous = value.get(
        "failed_pre_attempt_archive_missing_runtime_required_members")
    require(previous == [{
        "path": str(M1314_AUTHOR_TEST.relative_to(ROOT)),
        "sha256": M1314_AUTHOR_TEST_SHA256,
        "seal_root": str(M1314_AUTHOR_TEST.parent.relative_to(ROOT)),
        "reason": "repository *.log ignore excludes the member, but exact M1314 recursive seal verification reads it",
        "deployment_role": "failure root cause; now present at the exact destination path in commit 016849ec",
    }] and value.get("current_git_archive_missing_runtime_required_members") == [],
            "archive-missing forensic/current inventory drift")
    completeness = value.get("archive_completeness", {})
    require(completeness.get("previous_failed_archive_missing_members") == 1 and
            completeness.get("current_plain_git_archive_is_complete") is True and
            completeness.get("current_required_supplements") == 0 and
            completeness.get(
                "all_previous_and_current_missing_runtime_required_members_enumerated") is True and
            completeness.get(
                "preflight_must_fail_if_any_runtime_member_is_missing_mismatched_or_symlinked") is True and
            completeness.get(
                "parent_or_child_budget_may_be_reached_before_preflight") is False,
            "deployment completeness policy drift")

    total_members = 0
    m1314_member_bound = False
    roots = set()
    for entry in _manifest_seals(value):
        require(type(entry) is dict and set(entry) == {
            "root", "manifest_sha256", "outer_file_sha256", "members"},
            "predecessor seal entry shape drift")
        root = ROOT / entry["root"]
        require(root.parent == HW / "reviews" and
                root.is_dir() and not root.is_symlink() and root not in roots,
                "predecessor seal root invalid/duplicate")
        roots.add(root)
        manifest = root / "SHA256SUMS"
        outer = root / "SHA256SUMS.seal.sha256"
        regular_exact(manifest, entry["manifest_sha256"], root.name + " manifest")
        regular_exact(outer, entry["outer_file_sha256"], root.name + " outer")
        require(outer.read_text(encoding="ascii") ==
                entry["manifest_sha256"] + "  SHA256SUMS\n",
                root.name + " outer content drift")
        seen = set()
        lines = manifest.read_text(encoding="utf-8").splitlines()
        require(len(lines) == entry["members"], root.name + " member count drift")
        for line in lines:
            fields = line.split("  ", 1)
            require(len(fields) == 2 and len(fields[0]) == 64,
                    root.name + " malformed member")
            digest, name = fields
            require(name not in seen, root.name + " duplicate member")
            seen.add(name)
            member = _safe_member(root, name)
            regular_exact(member, digest, root.name + " member " + name)
            total_members += 1
            if member.resolve() == M1314_AUTHOR_TEST.resolve():
                require(digest == M1314_AUTHOR_TEST_SHA256,
                        "M1314 author_test seal digest drift")
                m1314_member_bound = True
    require(total_members == 116 and m1314_member_bound and
            len(roots) == 16, "deployment closure cardinality drift")
    regular_exact(M1314_AUTHOR_TEST, M1314_AUTHOR_TEST_SHA256,
                  "tracked M1314 author_test.log")
    return {
        "status": "PASS_M1647_DEPLOYMENT_COMPLETE_BEFORE_ANY_RUNTIME_BUDGET",
        "runtime_predecessor_seals": 16,
        "sealed_members": 116,
        "previous_archive_missing_members": 1,
        "current_archive_missing_members": 0,
        "m1314_author_test_sha256": M1314_AUTHOR_TEST_SHA256,
        "parent_processes": 0,
        "child_processes": 0,
        "gpu_runs": 0,
        "capture_runs": 0,
        "attempt_writes": 0,
    }


def validate_archive_member_inventory(archive_members):
    """Pure regression gate for a decoded Git-archive member-name set."""
    require(type(archive_members) is set and
            all(type(name) is str for name in archive_members),
            "archive inventory must be a set of names")
    required = str(M1314_AUTHOR_TEST.relative_to(ROOT))
    require(required in archive_members,
            "current archive omits sealed runtime-required M1314 author_test.log")
    return {"current_archive_missing_runtime_required_members": [],
            "m1314_author_test_present": True}


def load_m1624():
    regular_exact(M1624_SOURCE, M1624_SOURCE_SHA256, "exact M1624 source")
    spec = importlib.util.spec_from_file_location("m1647_exact_m1624",
                                                  str(M1624_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1624")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regular_exact(M1624_SOURCE, M1624_SOURCE_SHA256,
                  "exact M1624 source after import")
    return module


P = load_m1624()


def verify_exact_m1624_m1640():
    for path, digest, label in (
            (M1624_SOURCE, M1624_SOURCE_SHA256, "M1624 source"),
            (M1624_TEST, M1624_TEST_SHA256, "M1624 test"),
            (M1624_CONTRACT, M1624_CONTRACT_SHA256, "M1624 contract"),
            (M1626_RELEASE, M1626_RELEASE_SHA256, "M1626 release"),
            (M1640 / "review.json", M1640_REVIEW_SHA256, "M1640 review"),
            (M1640 / "SHA256SUMS", M1640_MANIFEST_SHA256, "M1640 manifest"),
            (M1640 / "SHA256SUMS.seal.sha256", M1640_OUTER_SHA256,
             "M1640 outer seal"),
            (DOCS359, DOCS359_SHA256, "protected docs359")):
        regular_exact(path, digest, label)
    require((M1640 / "SHA256SUMS.seal.sha256").read_text(encoding="ascii") ==
            M1640_MANIFEST_SHA256 + "  SHA256SUMS\n",
            "M1640 outer content drift")
    review = strict_json(M1640 / "review.json")
    require(review.get("status") ==
            "PASS_M1640_M1626_CLEAN_CHILD_CAPTURE_RELEASE_HAMMER__REMOTE_ONE_SHOT_GO" and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("authorization", {}).get("parent_calls") == 1 and
            review.get("authorization", {}).get("clean_child_processes") == 1 and
            review.get("authorization", {}).get("gpu_runs") == 1 and
            review.get("authorization", {}).get("production_captures") == 1 and
            review.get("authorization", {}).get("automatic_retry") is False,
            "M1640 exact release verdict drift")
    return review


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
            value.get("deployment_manifest") == {
                "path": str(DEPLOYMENT_MANIFEST.relative_to(ROOT)),
                "sha256": DEPLOYMENT_MANIFEST_SHA256,
                "sidecar_sha256": DEPLOYMENT_MANIFEST_SIDECAR_SHA256,
                "outer_seal_file_sha256":
                    DEPLOYMENT_MANIFEST_OUTER_FILE_SHA256,
                "double_sealed": True},
            "M1647 source contract identity drift")
    require(value.get("authorization", {}).get("different_author_review") is True and
            value.get("authorization", {}).get("capture") is False and
            value.get("authorization", {}).get("gpu") is False and
            value.get("authorization", {}).get("release") is False,
            "source contract authorizes runtime work")
    return value


def _verify_tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "review root absent/symlink")
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and outer.is_file(), "review double seal absent")
    sums_sha = sha256(sums)
    require(outer.read_text(encoding="ascii") == sums_sha + "  SHA256SUMS\n",
            "review outer mismatch")
    sealed_review = None
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        member = _safe_member(root, name)
        regular_exact(member, digest, "review member")
        if member.resolve() == (root / "review.json").resolve():
            sealed_review = digest
    require(sealed_review == sha256(root / "review.json"),
            "review.json not sealed")
    return strict_json(root / "review.json"), sums_sha, sha256(outer)


def _verify_file_seal(path):
    path = Path(path)
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1647Error("release file absent") from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "release must be regular non-symlink")
    regular_exact(sums, sha256(sums), "release sidecar")
    regular_exact(outer, sha256(outer), "release outer")
    require(sums.read_text(encoding="ascii") ==
            sha256(path) + "  " + path.name + "\n" and
            outer.read_text(encoding="ascii") ==
            sha256(sums) + "  " + sums.name + "\n",
            "release double seal drift")


def validate_future_authorities():
    review, review_manifest_sha, review_outer_file_sha = _verify_tree(FUTURE_REVIEW)
    expected_identity = {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "m1624_source_sha256": M1624_SOURCE_SHA256,
        "m1624_source_contract_sha256": M1624_CONTRACT_SHA256,
        "m1626_release_sha256": M1626_RELEASE_SHA256,
        "m1640_review_sha256": M1640_REVIEW_SHA256,
        "m1640_manifest_sha256": M1640_MANIFEST_SHA256,
        "m1640_outer_file_sha256": M1640_OUTER_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == expected_identity and
            review.get("authorization") == {
                "release_authoring": True, "capture": False,
                "gpu": False, "automatic_retry": False},
            "M1648 review mismatch")
    _verify_file_seal(FUTURE_RELEASE)
    release = strict_json(FUTURE_RELEASE)
    expected_release_identity = dict(
        expected_identity,
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=review_manifest_sha,
        review_outer_file_sha256=review_outer_file_sha)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_release_identity and
            release.get("authorization") == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_runs": 1, "production_captures": 1,
                "automatic_retry": False, "all_other_runs": 0} and
            release.get("namespaces") == {
                "result": str(RESULT.relative_to(ROOT)),
                "attempt": str(ATTEMPT.relative_to(ROOT)),
                "work": str(WORK.relative_to(ROOT)),
                "failure": str(FAILURE.relative_to(ROOT))} and
            release.get("claim_boundary") == {
                "tsbg_dse": False, "aee": False, "rtl": False,
                "eda": False, "performance": False,
                "paper_result": False} and
            release.get("deployment_preflight") == {
                "manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
                "before_parent_subprocess": True,
                "before_child_budget": True,
                "before_gpu_attempt_checkpoint": True},
            "M1649 release mismatch")
    interpreter = release.get("child_interpreter", {})
    require(interpreter.get("path") == str(P.CHILD_PYTHON),
            "M1649 interpreter path drift")
    regular_exact(P.CHILD_PYTHON, interpreter.get("sha256"),
                  "M1649 child interpreter")
    return release


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and
            all("m1647_" in path.name for path in paths),
            "M1647 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1647 result/attempt/work/failure namespace is not fresh")


def write_child_receipt(root, release, load_audit, validation):
    receipt = {
        "schema": "m1647_ep34_s2_tsbg_deployment_complete_capture_receipt_r1_v1",
        "status": "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
        "identity": {
            "source_sha256": sha256(SOURCE),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "release_sha256": sha256(FUTURE_RELEASE),
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
            "m1624_source_sha256": M1624_SOURCE_SHA256,
            "m1640_review_sha256": M1640_REVIEW_SHA256,
            "m1558_source_sha256": P.M1558_SHA256,
            "m1458_manifest_sha256": P.M1458_MANIFEST_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": P.CONFIG_SHA256,
        },
        "checkpoint_load": dict((key, int(load_audit.get(key, -1))) for key in (
            "missing_count", "unexpected_count", "overlay_missing_count",
            "overlay_unexpected_count")),
        "population": {
            "samples": 40, "frames": int(validation["frames"]),
            "fc_tokens": int(validation["fc_tokens"]),
            "patch_histogram_rows": int(validation["patch_histogram_rows"]),
        },
        "execution": {
            "deployment_preflight_before_parent_and_child_budget": True,
            "clean_child_processes": 1, "automatic_retry": False,
            "provider_crossed_parent_boundary": False,
            "permit_crossed_parent_boundary": False,
            "free_space_crossed_parent_boundary": False,
            "provenance_crossed_parent_boundary": False,
            "callable_crossed_parent_boundary": False,
        },
        "claim_boundary": {
            "capture_payload_only": True, "fresh_result_hammer_required": True,
            "hardware_quantization_authority": False,
            "model_bit_exact": False, "tsbg_dse": False, "aee": False,
            "cycles": False, "traffic": False, "energy": False,
            "speedup": False, "rtl": False, "eda": False,
            "paper_result": False,
        },
    }
    (root / "m1647_clean_child_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    P.seal_result(root)
    return receipt


@contextlib.contextmanager
def _bound_exact_m1624():
    """Bind only identities/namespaces inside one isolated process."""
    replacements = {
        "SOURCE": SOURCE, "TEST": TEST, "SOURCE_CONTRACT": SOURCE_CONTRACT,
        "FUTURE_REVIEW": FUTURE_REVIEW, "FUTURE_RELEASE": FUTURE_RELEASE,
        "RESULT": RESULT, "ATTEMPT": ATTEMPT, "WORK": WORK,
        "FAILURE": FAILURE, "SOURCE_SCHEMA": SOURCE_SCHEMA,
        "SOURCE_STATUS": SOURCE_STATUS, "REVIEW_STATUS": REVIEW_STATUS,
        "RELEASE_STATUS": RELEASE_STATUS, "ATTEMPT_TOKEN": ATTEMPT_TOKEN,
        "PASS_TOKEN": PASS_TOKEN, "validate_source_contract": validate_source_contract,
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
    # This must be the first call: no M1624 import chain, lease, attempt,
    # checkpoint or GPU is reachable before the complete deployment passes.
    verify_deployment_completeness()
    verify_exact_m1624_m1640()
    with _bound_exact_m1624():
        return P.fixed_clean_child()


def launch_parent():
    # First parent action, before the only subprocess/child budget.
    verify_deployment_completeness()
    verify_exact_m1624_m1640()
    with _bound_exact_m1624():
        return P.launch_parent()


def source_self_check():
    deployment = verify_deployment_completeness()
    verify_exact_m1624_m1640()
    validate_source_contract()
    require_fresh_namespaces()
    require(not os.path.lexists(str(FUTURE_REVIEW)) and
            not os.path.lexists(str(FUTURE_RELEASE)) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) + ".sha256"))) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                         ".sha256.seal.sha256"))),
            "future M1648/M1649 authority must be absent at authoring")
    deployment.update({
        "source_status": SOURCE_STATUS,
        "future_review_present": False,
        "future_release_present": False,
        "remote_connected": False,
        "payload_opened": False,
        "checkpoint_loaded": False,
        "automatic_retry": False,
        "claim_boundary": {
            "source_only": True, "capture": False, "gpu": False,
            "aee": False, "cycles": False, "traffic": False,
            "energy": False, "speedup": False, "rtl": False,
            "eda": False, "paper_result": False,
        },
    })
    return deployment


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
