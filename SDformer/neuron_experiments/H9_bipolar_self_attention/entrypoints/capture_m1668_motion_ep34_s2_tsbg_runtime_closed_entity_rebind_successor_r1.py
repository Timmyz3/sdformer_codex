#!/usr/bin/env python3
"""M1668 source-only successor for one future ep34 TSBG capture.

M1647 failed before its attempt because its deployment contained the Python
closure but not the non-Git M1257 runtime selection result.  A diagnostic
supplement then exposed a second, independent fact: the selected configuration
was recreated with byte-identical content but a new filesystem entity.

This successor binds the exact M1306 runtime-data handoff and a new read-only
capture-time entity observation.  The checkpoint is not reselected and the
configuration contents do not change.  Before either the parent subprocess or
the child GPU/attempt budget is reachable, it validates the M1257 canonical
runtime result, validates the current checkpoint/config/profile entities, and
executes the complete M1434 ``build_runtime()`` path under a narrowly scoped
configuration-entity rebind.  Both nested identity verifiers are restored in
``finally``.  M1647/M1624 capture semantics remain otherwise exact.

This authoring revision is inert: M1669 review and M1670 release do not exist.
It performs no remote write, checkpoint load, GPU run, capture, or retry.
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
import tarfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "tests/test_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_"
    "source_contract_r1_20260901.json")
SELECTION_IDENTITY = HW / (
    "contracts/m1668_motion_ep34_s2_tsbg_current_selection_entity_"
    "r1_20260901.json")
M1647_SOURCE = SOURCE.with_name(
    "capture_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_"
    "successor_r1.py")
M1647_TEST = HW / (
    "tests/test_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_"
    "source.py")
M1647_CONTRACT = HW / (
    "contracts/m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_"
    "source_contract_r1_20260901.json")
M1648 = HW / (
    "reviews/m1648_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_"
    "child_source_independent_review_r1_20260901")
M1649_RELEASE = HW / (
    "contracts/m1649_m1648_m1647_motion_ep34_s2_tsbg_deployment_complete_"
    "clean_child_capture_release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNTIME_TAR = HW / (
    "system_handoff/incoming/m1306_remote_selection_result_20260830/"
    "m1306_remote_selection_result_20260830.tar")
RUNTIME_TAR_SIDECAR = Path(str(RUNTIME_TAR) + ".sha256")
SOURCE_STAGE_M1257 = HW / (
    "system_handoff/incoming/m1306_remote_selection_result_20260830/"
    "hw_autoresearch_nts07/results/"
    "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")
SOURCE_STAGE_ATTEMPT = SOURCE_STAGE_M1257.parent / (
    ".m1257_motion_cross_run_final_checkpoint_selection_r5_attempt_consumed")
SOURCE_STAGE_LOG = SOURCE_STAGE_M1257.parent / (
    "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830.launch.log")
RUNTIME_M1257 = HW / (
    "results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")
RUNTIME_ATTEMPT = HW / (
    "results/.m1257_motion_cross_run_final_checkpoint_selection_r5_"
    "attempt_consumed")
RUNTIME_LOG = HW / (
    "results/m1257_motion_cross_run_final_checkpoint_selection_r5_"
    "20260830.launch.log")

FUTURE_REVIEW = HW / (
    "reviews/m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_"
    "rebind_source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1670_m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_"
    "entity_rebind_capture_release_r1_20260901.json")
RESULT = HW / (
    "results/m1668_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1668_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1668_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1668_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.failed_no_retry")

SOURCE_SCHEMA = (
    "m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_source_r1_v1")
SOURCE_STATUS = (
    "SOURCE_ONLY__RUNTIME_DATA_CLOSED__CURRENT_ENTITY_BOUND__"
    "BUILD_RUNTIME_BEFORE_BUDGET__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_CAPTURE")
REVIEW_STATUS = (
    "PASS_M1669_M1668_RUNTIME_CLOSED_ENTITY_REBIND_SOURCE__"
    "AUTHORIZE_RELEASE_AUTHORING__NO_CAPTURE")
RELEASE_SCHEMA = (
    "m1670_m1669_m1668_runtime_closed_entity_rebind_capture_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1668_EP34_S2_TSBG_RUNTIME_CLOSED_ENTITY_REBIND_CAPTURE")
ATTEMPT_TOKEN = (
    "M1668_ATTEMPT_CONSUMED__RUNTIME_AND_ENTITY_PREFLIGHT_PASS__"
    "ONE_CLEAN_CHILD__AUTOMATIC_RETRY_FALSE\n")
PASS_TOKEN = (
    "PASS_M1668_EP34_S2_TSBG_RUNTIME_CLOSED_ENTITY_REBIND_CAPTURE__"
    "FRESH_RESULT_HAMMER_REQUIRED")

M1647_SOURCE_SHA256 = (
    "3e16c6f4b740a7a9454ad243de3c128185d3135f7a26ccbdfb7e94ae5505682a")
M1647_TEST_SHA256 = (
    "57d73fb037176458f3c1802ffa3b26a4985c049fbc023ca107150f4005673abf")
M1647_CONTRACT_SHA256 = (
    "c98bf1b0b77c315550d0e618947d8e47d58488c7a8470080ee5a552ade0c719a")
M1648_REVIEW_SHA256 = (
    "c8292001df4481f78e09018a65ce86d1b512983634daffcd7ed1e1f034b9de7c")
M1648_MANIFEST_SHA256 = (
    "ae79205a393b2b8a6e0b91272668da9d58ca25d440cf316f207a2c0b2737c557")
M1648_OUTER_SHA256 = (
    "62377418621d4465c001593d16f414fd0407028715c3c6a02369c8d17fb054d7")
M1649_RELEASE_SHA256 = (
    "64cb869004753d1e9c8aeda3f6533657dd53334f4b3639ee85b3bd64981e555a")
SELECTION_IDENTITY_SHA256 = (
    "e6b3dd82d5d1eb54e605595369bfc8228fd616ab707d58b2e4afd95c159f87c7")
SELECTION_IDENTITY_SIDECAR_SHA256 = (
    "528081ea8dbe938846b571565f5bb65b42fc402b0931e492098e90ba4f4184de")
RUNTIME_TAR_SHA256 = (
    "0524a94ccb36adc7ebc17603dedc322810141d8b14dc743923c5b942a5c6c36f")
RUNTIME_ATTEMPT_SHA256 = (
    "60a55f692eb71374d1029628e52b30a47b0c12e0bc57aec1d78f6d330ffc19f8")
RUNTIME_LOG_SHA256 = (
    "7bef9b9b43e341d6dee703fa78a331aacf151d4b4e56591a48919bd66cb2e51b")
M1257_MANIFEST_SHA256 = (
    "ae4a61f5e79b0d6e308174c00567fff6e25a07a6f065cd7ee3acec2faabcf458")
M1257_OUTER_SHA256 = (
    "d0afaea457958752b9d76c21746c0796145a91466cf93ecd20a56d27bd5ef7e4")
CHECKPOINT_SHA256 = (
    "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
CONFIG_SHA256 = (
    "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39")
PROFILE_SHA256 = (
    "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

M1257_MEMBER_SHA256 = {
    "RUN_COMPLETE.txt": "aac49e55e808ea1759a36ef6a00550c5b5f24709615de83ecde3d4c7d448f0d6",
    "SHA256SUMS": M1257_MANIFEST_SHA256,
    "SHA256SUMS.seal.sha256": M1257_OUTER_SHA256,
    "e0_e8_activation_rebind_targets.json": "2630e67c18b19c8b645d397de2ee98225aaf8ae0caf0ee3d64a12ed967aa5d50",
    "final_checkpoint_selection.json": "4af7b7e1b4a174440331268fcfffda44896d86d02c7d20195e7a49d73eae6cd0",
    "four_checkpoint_metrics.csv": "2e6266ff753f24fec7185eb6f6b9b7b2b93fce1033cc5a46f3dd4ed710a2700f",
    "selected_checkpoint_and_config.json": "154cc413ed5cd029285529e8e1e0677a6031cca94ed16e956ea77d2b090b777b",
}


class M1668Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1668Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1668Error("missing " + label) from error
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
                           M1668Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _verify_file_double_seal(path, digest, sidecar_digest):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular_exact(path, digest, path.name)
    regular_exact(sidecar, sidecar_digest, path.name + " sidecar")
    require(sidecar.read_text(encoding="ascii") ==
            digest + "  " + path.name + "\n",
            path.name + " sidecar content drift")
    require(outer.read_text(encoding="ascii") ==
            sidecar_digest + "  " + sidecar.name + "\n",
            path.name + " outer content drift")


def selection_identity():
    _verify_file_double_seal(SELECTION_IDENTITY, SELECTION_IDENTITY_SHA256,
                             SELECTION_IDENTITY_SIDECAR_SHA256)
    value = strict_json(SELECTION_IDENTITY)
    require(value.get("schema") ==
            "m1668_motion_ep34_s2_tsbg_current_selection_entity_r1_v1" and
            value.get("status") ==
            "SOURCE_STAGE_REMOTE_READ_ONLY_ENTITY_OBSERVATION__NO_CAPTURE_AUTHORITY",
            "M1668 selection entity schema/status drift")
    semantic = value.get("selection_semantics", {})
    require(semantic.get("checkpoint_reselected") is False and
            semantic.get("configuration_semantics_changed") is False and
            semantic.get("profile_reselected") is False and
            semantic.get("selected_candidate_id") == "resume_ep34" and
            semantic.get("selected_epoch") == 34 and
            semantic.get("source_selection_manifest_sha256") ==
                M1257_MANIFEST_SHA256 and
            semantic.get("source_selection_outer_file_sha256") ==
                M1257_OUTER_SHA256,
            "M1668 selection semantics drift")
    frozen = value.get("configuration_frozen_selection_entity", {})
    current = value.get("configuration_current_capture_entity", {})
    require(frozen.get("sha256") == current.get("sha256") == CONFIG_SHA256 and
            frozen.get("size_bytes") == current.get("size_bytes") == 6481 and
            frozen.get("absolute_path") == current.get("absolute_path") and
            (frozen.get("device"), frozen.get("inode"), frozen.get("mode"),
             frozen.get("mtime_ns")) !=
            (current.get("device"), current.get("inode"), current.get("mode"),
             current.get("mtime_ns")),
            "current configuration entity is not a strict content-identical rebind")
    require(value.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA256 and
            value.get("profile", {}).get("sha256") == PROFILE_SHA256 and
            value.get("claim_boundary", {}).get("capture") is False,
            "M1668 checkpoint/profile/claim boundary drift")
    return value


def _verify_m1257_root(root, attempt, launch_log):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "M1257 root absent/symlink")
    require(set(path.name for path in root.iterdir() if path.is_file()) ==
            set(M1257_MEMBER_SHA256), "M1257 member set drift")
    for name, digest in M1257_MEMBER_SHA256.items():
        regular_exact(root / name, digest, "M1257 " + name)
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="ascii") ==
            M1257_MANIFEST_SHA256 + "  SHA256SUMS\n",
            "M1257 outer seal content drift")
    sealed = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in sealed,
                "M1257 malformed/duplicate seal row")
        sealed[fields[1]] = fields[0]
    require(set(sealed) == set(M1257_MEMBER_SHA256) - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "M1257 inner seal population drift")
    require(all(sealed[name] == M1257_MEMBER_SHA256[name] for name in sealed),
            "M1257 inner seal digest drift")
    regular_exact(attempt, RUNTIME_ATTEMPT_SHA256, "M1257 attempt")
    regular_exact(launch_log, RUNTIME_LOG_SHA256, "M1257 launch log")
    return {"canonical_files": 7, "attempt": 1, "launch_log": 1}


def verify_runtime_handoff_source():
    regular_exact(RUNTIME_TAR, RUNTIME_TAR_SHA256, "M1306 runtime handoff tar")
    require(RUNTIME_TAR_SIDECAR.read_text(encoding="ascii") ==
            RUNTIME_TAR_SHA256 + "  " + RUNTIME_TAR.name + "\n",
            "M1306 runtime handoff sidecar drift")
    prefix = ("hw_autoresearch_nts07/results/"
              "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830/")
    expected_files = set(prefix + name for name in M1257_MEMBER_SHA256)
    expected_files.update({
        "hw_autoresearch_nts07/results/"
        ".m1257_motion_cross_run_final_checkpoint_selection_r5_attempt_consumed",
        "hw_autoresearch_nts07/results/"
        "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830.launch.log",
    })
    observed_files = set()
    with tarfile.open(str(RUNTIME_TAR), "r:") as stream:
        for member in stream.getmembers():
            require(not member.issym() and not member.islnk() and
                    not member.name.startswith("/") and
                    ".." not in Path(member.name).parts,
                    "unsafe M1306 tar member")
            if member.isfile():
                observed_files.add(member.name)
                handle = stream.extractfile(member)
                require(handle is not None, "cannot read M1306 tar member")
                payload = handle.read()
                if member.name.startswith(prefix):
                    require(bytes_sha256(payload) ==
                            M1257_MEMBER_SHA256[member.name[len(prefix):]],
                            "M1306 tar M1257 member SHA drift")
                elif member.name.endswith("attempt_consumed"):
                    require(bytes_sha256(payload) == RUNTIME_ATTEMPT_SHA256,
                            "M1306 tar attempt SHA drift")
                else:
                    require(bytes_sha256(payload) == RUNTIME_LOG_SHA256,
                            "M1306 tar launch-log SHA drift")
    require(observed_files == expected_files, "M1306 tar file inventory drift")
    row = _verify_m1257_root(SOURCE_STAGE_M1257, SOURCE_STAGE_ATTEMPT,
                             SOURCE_STAGE_LOG)
    row.update({"archive_sha256": RUNTIME_TAR_SHA256,
                "archive_files": len(expected_files)})
    return row


def verify_runtime_canonical():
    return _verify_m1257_root(RUNTIME_M1257, RUNTIME_ATTEMPT, RUNTIME_LOG)


def verify_entity(path, expected, label):
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise M1668Error("missing " + label) from error
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    digest = sha256(path)
    after = path.lstat()
    observed = {
        "absolute_path": str(path), "device": after.st_dev,
        "inode": after.st_ino, "mode": after.st_mode,
        "mtime_ns": after.st_mtime_ns, "sha256": digest,
        "size_bytes": after.st_size,
    }
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
             before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size,
             after.st_mtime_ns), label + " changed while hashing")
    require(observed == expected, label + " current entity drift")
    return observed


def verify_predecessors():
    for path, digest, label in (
            (M1647_SOURCE, M1647_SOURCE_SHA256, "M1647 source"),
            (M1647_TEST, M1647_TEST_SHA256, "M1647 test"),
            (M1647_CONTRACT, M1647_CONTRACT_SHA256, "M1647 contract"),
            (M1648 / "review.json", M1648_REVIEW_SHA256, "M1648 review"),
            (M1648 / "SHA256SUMS", M1648_MANIFEST_SHA256, "M1648 manifest"),
            (M1648 / "SHA256SUMS.seal.sha256", M1648_OUTER_SHA256,
             "M1648 outer"),
            (M1649_RELEASE, M1649_RELEASE_SHA256, "M1649 release"),
            (DOCS359, DOCS359_SHA256, "protected docs359")):
        regular_exact(path, digest, label)
    require((M1648 / "SHA256SUMS.seal.sha256").read_text(encoding="ascii") ==
            M1648_MANIFEST_SHA256 + "  SHA256SUMS\n",
            "M1648 outer content drift")


def load_m1647():
    regular_exact(M1647_SOURCE, M1647_SOURCE_SHA256, "M1647 source before import")
    spec = importlib.util.spec_from_file_location("m1668_exact_m1647",
                                                  str(M1647_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1647")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1647_SOURCE, M1647_SOURCE_SHA256, "M1647 source after import")
    return module


P = load_m1647()
ORIGINAL_M1624_LOAD_M1434 = P.P.load_m1434


@contextlib.contextmanager
def current_configuration_entity_rebind(m1434, identity):
    m1319 = m1434.M1349.M1327.M1319
    frozen_m1233 = m1319.FROZEN_M1233
    original_extended = m1319.exact_extended_identity
    original_frozen = frozen_m1233.exact_identity
    frozen = identity["configuration_frozen_selection_entity"]
    current = identity["configuration_current_capture_entity"]
    require(m1319.exact_extended_identity is original_extended and
            frozen_m1233.exact_identity is original_frozen,
            "identity verifier already rebound")

    def narrow(value, label):
        if label == "selected configuration":
            require(type(value) is dict and value == frozen,
                    "sealed frozen configuration selection drift")
            verify_entity(Path(current["absolute_path"]), current,
                          "current selected configuration")
            return dict(value)
        return original_extended(value, label)

    def frozen_narrow(value, label):
        if label == "selected configuration":
            require(type(value) is dict and value == frozen,
                    "nested frozen configuration selection drift")
            verify_entity(Path(current["absolute_path"]), current,
                          "current nested selected configuration")
            return dict(value)
        return original_frozen(value, label)

    m1319.exact_extended_identity = narrow
    frozen_m1233.exact_identity = frozen_narrow
    tampered = False
    try:
        yield
    finally:
        tampered = (m1319.exact_extended_identity is not narrow or
                    frozen_m1233.exact_identity is not frozen_narrow)
        m1319.exact_extended_identity = original_extended
        frozen_m1233.exact_identity = original_frozen
        require(not tampered, "identity verifier changed inside rebind scope")


def load_m1434_rebound():
    identity = selection_identity()
    m1434 = ORIGINAL_M1624_LOAD_M1434()
    original_build = m1434.build_runtime

    def build_runtime_rebound():
        with current_configuration_entity_rebind(m1434, identity):
            runtime, binding = original_build()
        current = identity["configuration_current_capture_entity"]
        require(binding.get("identity", {}).get("checkpoint_sha256") ==
                CHECKPOINT_SHA256 and
                binding.get("identity", {}).get("config_sha256") == CONFIG_SHA256,
                "rebound runtime content identity drift")
        binding["identity"]["config_current_entity"] = {
            key: current[key] for key in
            ("device", "inode", "mode", "mtime_ns", "size_bytes", "sha256")}
        binding["identity"]["m1668_selection_entity_sha256"] = (
            SELECTION_IDENTITY_SHA256)
        return runtime, binding

    m1434.build_runtime = build_runtime_rebound
    return m1434


def preflight_runtime_binding():
    """Complete runtime construction without substrate/GPU/attempt access."""
    verify_runtime_canonical()
    identity = selection_identity()
    for key, label in (("checkpoint", "current checkpoint"),
                       ("configuration_current_capture_entity", "current config"),
                       ("profile", "current profile")):
        row = identity[key]
        verify_entity(Path(row["absolute_path"]), row, label)
    m1434 = load_m1434_rebound()
    m1434.verify_predecessors()
    runtime, binding = m1434.build_runtime()
    require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
            runtime.get("capture") == {"attention_windows_per_call": 100} and
            binding.get("identity", {}).get("m1668_selection_entity_sha256") ==
                SELECTION_IDENTITY_SHA256 and
            Path(binding.get("checkpoint_path")) ==
                Path(identity["checkpoint"]["absolute_path"]) and
            Path(binding.get("config_path")) ==
                Path(identity["configuration_current_capture_entity"][
                    "absolute_path"]),
            "M1668 build_runtime projection drift")
    return {"status": "PASS_M1668_BUILD_RUNTIME_BEFORE_ANY_BUDGET",
            "runtime_keys": sorted(runtime), "samples": 40,
            "gpu_runs": 0, "attempt_writes": 0,
            "selection_entity_sha256": SELECTION_IDENTITY_SHA256}


def validate_source_contract():
    value = strict_json(SOURCE_CONTRACT)
    require(value.get("schema") == SOURCE_SCHEMA and
            value.get("status") == SOURCE_STATUS and
            value.get("source") == {
                "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            value.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)} and
            value.get("selection_identity", {}).get("sha256") ==
                SELECTION_IDENTITY_SHA256 and
            value.get("runtime_data_handoff", {}).get("archive_sha256") ==
                RUNTIME_TAR_SHA256,
            "M1668 source contract identity drift")
    authorization = value.get("authorization", {})
    require(authorization.get("different_author_review") is True and
            authorization.get("release") is False and
            authorization.get("parent_launch") is False and
            authorization.get("capture") is False and
            authorization.get("gpu") is False and
            authorization.get("attempt_creation") is False,
            "M1668 source contract authorizes runtime work")
    return value


def _verify_review_tree(root):
    review, manifest_sha, outer_sha = P._verify_tree(root)
    return review, manifest_sha, outer_sha


def validate_future_authorities():
    review, manifest_sha, outer_sha = _verify_review_tree(FUTURE_REVIEW)
    expected = {
        "source_sha256": sha256(SOURCE), "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "selection_identity_sha256": SELECTION_IDENTITY_SHA256,
        "runtime_tar_sha256": RUNTIME_TAR_SHA256,
        "m1647_source_sha256": M1647_SOURCE_SHA256,
        "m1648_review_sha256": M1648_REVIEW_SHA256,
        "m1649_release_sha256": M1649_RELEASE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == expected and
            review.get("authorization") == {
                "release_authoring": True, "capture": False,
                "gpu": False, "automatic_retry": False},
            "M1669 review mismatch")
    P._verify_file_seal(FUTURE_RELEASE)
    release = strict_json(FUTURE_RELEASE)
    release_identity = dict(expected,
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=manifest_sha,
        review_outer_file_sha256=outer_sha)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == release_identity and
            release.get("authorization") == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_runs": 1, "production_captures": 1,
                "automatic_retry": False, "all_other_runs": 0} and
            release.get("namespaces") == {
                "result": str(RESULT.relative_to(ROOT)),
                "attempt": str(ATTEMPT.relative_to(ROOT)),
                "work": str(WORK.relative_to(ROOT)),
                "failure": str(FAILURE.relative_to(ROOT))} and
            release.get("pre_budget_preflight") == {
                "runtime_m1257_canonical": True,
                "current_entity_exact": True,
                "build_runtime_before_parent_subprocess": True,
                "build_runtime_before_child_gpu_attempt": True} and
            release.get("claim_boundary") == {
                "tsbg_dse": False, "aee": False, "rtl": False,
                "eda": False, "performance": False,
                "paper_result": False},
            "M1670 release mismatch")
    interpreter = release.get("child_interpreter", {})
    require(interpreter.get("path") == str(P.P.CHILD_PYTHON),
            "M1670 child interpreter path drift")
    regular_exact(P.P.CHILD_PYTHON, interpreter.get("sha256"),
                  "M1670 child interpreter")
    return release


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and all("m1668_" in path.name for path in paths),
            "M1668 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1668 namespace is not fresh")


def write_child_receipt(root, release, load_audit, validation):
    receipt = {
        "schema": "m1668_ep34_s2_tsbg_runtime_closed_entity_rebind_receipt_r1_v1",
        "status": "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
        "identity": {
            "source_sha256": sha256(SOURCE),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "release_sha256": sha256(FUTURE_RELEASE),
            "selection_entity_sha256": SELECTION_IDENTITY_SHA256,
            "runtime_tar_sha256": RUNTIME_TAR_SHA256,
            "m1647_source_sha256": M1647_SOURCE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "profile_sha256": PROFILE_SHA256,
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
            "runtime_and_entity_build_preflight_before_parent_and_child_budget": True,
            "clean_child_processes": 1, "automatic_retry": False,
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
    (root / "m1668_clean_child_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    P.P.seal_result(root)
    return receipt


@contextlib.contextmanager
def _bound_exact_m1647():
    replacements = {
        "SOURCE": SOURCE, "TEST": TEST, "SOURCE_CONTRACT": SOURCE_CONTRACT,
        "FUTURE_REVIEW": FUTURE_REVIEW, "FUTURE_RELEASE": FUTURE_RELEASE,
        "RESULT": RESULT, "ATTEMPT": ATTEMPT, "WORK": WORK,
        "FAILURE": FAILURE, "SOURCE_SCHEMA": SOURCE_SCHEMA,
        "SOURCE_STATUS": SOURCE_STATUS, "REVIEW_STATUS": REVIEW_STATUS,
        "RELEASE_SCHEMA": RELEASE_SCHEMA, "RELEASE_STATUS": RELEASE_STATUS,
        "ATTEMPT_TOKEN": ATTEMPT_TOKEN, "PASS_TOKEN": PASS_TOKEN,
        "validate_source_contract": validate_source_contract,
        "validate_future_authorities": validate_future_authorities,
        "require_fresh_namespaces": require_fresh_namespaces,
        "write_child_receipt": write_child_receipt,
    }
    originals = dict((name, getattr(P, name)) for name in replacements)
    original_loader = P.P.load_m1434
    try:
        for name, value in replacements.items():
            setattr(P, name, value)
        P.P.load_m1434 = load_m1434_rebound
        yield
    finally:
        P.P.load_m1434 = original_loader
        for name, value in originals.items():
            setattr(P, name, value)


def fixed_clean_child():
    # Complete runtime construction is repeated before GPU lease/attempt/model.
    verify_predecessors()
    preflight_runtime_binding()
    with _bound_exact_m1647():
        return P.fixed_clean_child()


def launch_parent():
    # Complete runtime construction occurs before the only child subprocess.
    verify_predecessors()
    preflight_runtime_binding()
    with _bound_exact_m1647():
        return P.launch_parent()


def source_self_check():
    verify_predecessors()
    identity = selection_identity()
    handoff = verify_runtime_handoff_source()
    validate_source_contract()
    require_fresh_namespaces()
    require(not os.path.lexists(str(FUTURE_REVIEW)) and
            not os.path.lexists(str(FUTURE_RELEASE)) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) + ".sha256"))) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                         ".sha256.seal.sha256"))),
            "future M1669/M1670 authority must be absent at authoring")
    return {
        "status": "PASS_M1668_SOURCE_SELF_CHECK__RUNTIME_HANDOFF_CLOSED__NO_CAPTURE",
        "source_status": SOURCE_STATUS,
        "selected_candidate_id": "resume_ep34", "selected_epoch": 34,
        "selection_entity_sha256": SELECTION_IDENTITY_SHA256,
        "configuration_content_unchanged":
            identity["configuration_frozen_selection_entity"]["sha256"] ==
            identity["configuration_current_capture_entity"]["sha256"],
        "runtime_handoff_files": handoff["archive_files"],
        "runtime_canonical_files": handoff["canonical_files"],
        "build_runtime_before_parent_subprocess": True,
        "build_runtime_before_child_gpu_attempt": True,
        "remote_connected_by_source_check": False,
        "checkpoint_loaded": False, "parent_processes": 0,
        "child_processes": 0, "gpu_runs": 0, "capture_runs": 0,
        "attempt_writes": 0, "automatic_retry": False,
        "claim_boundary": {
            "source_only": True, "capture": False, "gpu": False,
            "aee": False, "cycles": False, "traffic": False,
            "energy": False, "speedup": False, "rtl": False,
            "eda": False, "paper_result": False,
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
