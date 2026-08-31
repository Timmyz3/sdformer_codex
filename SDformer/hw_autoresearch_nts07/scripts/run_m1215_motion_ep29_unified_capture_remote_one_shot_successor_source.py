#!/opt/conda/envs/sdformerflow/bin/python
"""Inert M1215 successor launcher for the unchanged M1208 capture identity.

The wrapper cannot execute until the frozen M1211 admission and a fresh M1216
successor hammer both pass.  It performs the exact40 symlink-root and content check before
the child creates the disjoint M1208 persistent attempt marker.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Sequence


REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
HW = REPO / "hw_autoresearch_nts07"
INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python")
PYTHON_VERSION = "3.10.20"
CAPTURE_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1208_motion_ep29_unified_hardware_symlink_root_successor_r1.py")
SOURCE_CONTRACT_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1208_motion_ep29_unified_capture_symlink_root_successor_source_contract_r1_20260830.json")
TEST_REL = Path(
    "hw_autoresearch_nts07/tests/"
    "test_m1208_motion_ep29_unified_capture_symlink_root_successor_source.py")
LAUNCH_CONTRACT_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json")
SOURCE_HAMMER_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1209_m1208_motion_ep29_unified_capture_source_hammer_r1_20260830")
RELEASE_HAMMER_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_20260830")
SUCCESSOR_HAMMER_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_20260830")
FORENSIC_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
LEASE_REL = Path("hw_autoresearch_nts07/results/gpu_profile_lease.lock")
ATTEMPT_REL = Path(
    "hw_autoresearch_nts07/results/"
    ".m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
RESULT_REL = Path(
    "hw_autoresearch_nts07/results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830")
LOG_REL = Path(
    "hw_autoresearch_nts07/results/"
    ".m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
CAPTURE_SHA = "41b5276c39b613b6568ad7c7486abf150c3d0db86c3a905d6a30cdbbb543a049"
SOURCE_CONTRACT_SHA = "dad36c0a264e3e0d3a478929549431453ced60cba84fc24b2d9de442d29faa20"
TEST_SHA = "69de86545947d3c006dc621ddc0b618a61a8c57aa7e453478f61b56f079b3934"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RELEASE_HAMMER_SCHEMA = "m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_v1"
RELEASE_HAMMER_STATUS = "PASS_M1210_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED"
SUCCESSOR_HAMMER_SCHEMA = "m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_v1"
SUCCESSOR_HAMMER_STATUS = "PASS_M1215_SUCCESSOR_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED"
PASS_TOKEN = "PASS_M1208_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"


class ReleaseError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ReleaseError("non-standard JSON token: " + token)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def load_capture(path: Path) -> Any:
    regular(path, "M1208 capture source")
    require(sha256(path) == CAPTURE_SHA, "M1208 capture source SHA drift")
    spec = importlib.util.spec_from_file_location("m1208_remote_capture", path)
    require(spec is not None and spec.loader is not None, "cannot import M1208 capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(sha256(path) == CAPTURE_SHA, "M1208 capture changed during import")
    return module


def validate_release_hammer(capture: Any, contract_path: Path) -> dict[str, Any]:
    root = REPO / RELEASE_HAMMER_REL
    rows = capture.canonical_verify_double_seal(root)
    require("review.json" in rows, "M1211 release hammer lacks review.json")
    review = strict_json(root / "review.json")
    require(review.get("schema") == RELEASE_HAMMER_SCHEMA and
            review.get("status") == RELEASE_HAMMER_STATUS and
            review.get("verdict") == "GO" and
            isinstance(review.get("score"), int) and review.get("score") >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0,
            "M1211 release hammer semantic admission mismatch")
    bindings = review.get("bindings", {})
    require(bindings.get("source_sha256") ==
            "d3ce4b1e7aa1243b266053ed4f26ba452d25505791a9d328b03b5edda6e8432d" and
            bindings.get("source_contract_sha256") ==
            "b6bea670083eecbf8a6a73b48996fea6f1ee268d01f7da05738a4f986e361267" and
            bindings.get("launch_contract_sha256") == sha256(contract_path) and
            bindings.get("inventory_sha256") ==
            "11483d3c227f28aba8bb8a1dd765db59b8d432383b192f91c967d50f881de0cd" and
            bindings.get("transfer_list_sha256") ==
            "2734d71d8555cb131d77a3bea0057634d93264d0fcda38b050416c506f41e220" and
            bindings.get("release_author_manifest_sha256") ==
            "a7d10fc194fae5efffce5430ac75bc3c57fe4accb6b91ab57d158a396738a8c3" and
            bindings.get("release_author_outer_file_sha256") ==
            "b2a9e6d8dbc90f65435537de8bc4fa3e38ee2e53ffe45bc50f3f658974f0b68f",
            "M1211 release hammer exact binding mismatch")
    require(review.get("authorization", {}).get("exact_remote_launch") is True and
            review.get("authorization", {}).get("launch_count") == 1 and
            review.get("authorization", {}).get("automatic_retry") is False,
            "M1211 does not authorize exact no-retry launch")
    return review


def validate_successor_hammer(capture: Any, contract_path: Path) -> dict[str, Any]:
    forensic_rows = capture.canonical_verify_double_seal(REPO / FORENSIC_REL)
    require(forensic_rows.get("review.json") ==
            "1bc5af3d81d1cc1ee8dd7a91871ba9d31edc31cc442c256d96e23bc1bd828e65" and
            sha256(REPO / FORENSIC_REL / "SHA256SUMS") ==
            "a4cb9a3da26224b4f75bd2b3ea857c262155a81a094eb977e981ce60ea77cacb" and
            sha256(REPO / FORENSIC_REL / "SHA256SUMS.seal.sha256") ==
            "a7868aec49fb721f495e40d80d98f0d5525391ea807608a1363c04465373f995",
            "M1215 forensic identity mismatch")
    forensic = strict_json(REPO / FORENSIC_REL / "review.json")
    require(forensic.get("status") ==
            "PASS_FORENSIC__LOCAL_M1210_CONSUMED__REMOTE_M1208_UNCONSUMED__STATUS_MISMATCH_REPRODUCED" and
            forensic.get("successor_boundary", {}).get("remote_m1208_namespace_is_still_fresh") is True,
            "M1215 forensic semantic boundary mismatch")
    root = REPO / SUCCESSOR_HAMMER_REL
    rows = capture.canonical_verify_double_seal(root)
    require("review.json" in rows, "M1216 successor hammer lacks review.json")
    review = strict_json(root / "review.json")
    require(review.get("schema") == SUCCESSOR_HAMMER_SCHEMA and
            review.get("status") == SUCCESSOR_HAMMER_STATUS and
            review.get("verdict") == "GO" and
            isinstance(review.get("score"), int) and review.get("score") >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0,
            "M1216 successor semantic admission mismatch")
    bindings = review.get("bindings", {})
    require(bindings.get("successor_launcher_sha256") == sha256(Path(__file__).resolve()) and
            bindings.get("launch_contract_sha256") == sha256(contract_path) and
            bindings.get("capture_source_sha256") == CAPTURE_SHA and
            bindings.get("capture_source_contract_sha256") == SOURCE_CONTRACT_SHA and
            bindings.get("capture_source_test_sha256") == TEST_SHA and
            bindings.get("source_hammer_manifest_sha256") ==
            sha256(REPO / SOURCE_HAMMER_REL / "SHA256SUMS") and
            bindings.get("source_hammer_outer_file_sha256") ==
            sha256(REPO / SOURCE_HAMMER_REL / "SHA256SUMS.seal.sha256") and
            bindings.get("m1211_review_sha256") ==
            sha256(REPO / RELEASE_HAMMER_REL / "review.json") and
            bindings.get("m1215_forensic_review_sha256") ==
            "1bc5af3d81d1cc1ee8dd7a91871ba9d31edc31cc442c256d96e23bc1bd828e65" and
            bindings.get("m1210_failure_marker_sha256") ==
            "b60af667912eae9f19fb93aaf201fc342cfdd22e9add4bfeac0e55c09268e5f6",
            "M1216 successor exact binding mismatch")
    auth = review.get("authorization", {})
    require(auth.get("exact_remote_launch") is True and auth.get("launch_count") == 1 and
            auth.get("automatic_retry") is False,
            "M1216 does not authorize exact no-retry launch")
    return review


def gpu_compute_pids(runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[int]:
    completed = runner(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0, "cannot prove GPU idleness")
    try:
        return [int(row.strip()) for row in completed.stdout.splitlines() if row.strip()]
    except ValueError as exc:
        raise ReleaseError("malformed nvidia-smi output") from exc


def prove_lease_available(path: Path) -> None:
    if not os.path.lexists(path):
        return
    regular(path, "canonical GPU lease")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError("canonical GPU lease is busy") from exc
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Policy:
    repo: Path = REPO
    interpreter: Path = INTERPRETER
    python_version: str = PYTHON_VERSION


def preflight(policy: Policy, raw_executable: Path, version: str,
              cwd: Path) -> tuple[list[str], Any]:
    require(policy.repo == REPO and raw_executable == policy.interpreter and
            version == policy.python_version and cwd == policy.repo,
            "remote runtime identity mismatch")
    for relative, expected in {
        CAPTURE_REL: CAPTURE_SHA, SOURCE_CONTRACT_REL: SOURCE_CONTRACT_SHA,
        TEST_REL: TEST_SHA, DOCS359_REL: DOCS359_SHA,
    }.items():
        path = policy.repo / relative
        regular(path, "sealed prerequisite " + str(relative))
        require(sha256(path) == expected, "sealed prerequisite SHA drift: " + str(relative))
    capture = load_capture(policy.repo / CAPTURE_REL)
    contract_path = policy.repo / LAUNCH_CONTRACT_REL
    regular(contract_path, "M1208 launch contract")
    contract = capture.strict_json(contract_path)
    validate_release_hammer(capture, contract_path)
    validate_successor_hammer(capture, contract_path)
    capture.validate_launch_contract(contract, contract_path)
    require(contract["remote_runtime_identity"] == {
        "repository": str(policy.repo), "interpreter": str(policy.interpreter),
        "python_version": policy.python_version,
        "dataset_root": str(capture.PINNED_DSEC_ROOT)},
        "M1208 launch remote identity mismatch")
    require(contract["one_shot"]["attempt_marker"] == str(ATTEMPT_REL) and
            contract["output"]["path"] == str(RESULT_REL) and
            contract["production_log"]["path"] == str(LOG_REL),
            "M1208 disjoint namespace mismatch")
    require(not os.path.lexists(policy.repo / ATTEMPT_REL) and
            not os.path.lexists(policy.repo / RESULT_REL) and
            not os.path.lexists(policy.repo / LOG_REL),
            "M1208 attempt/result/log namespace is not fresh")
    # Critical repair is proved before attempt consumption, including all 40 hashes.
    selected = capture.selected_samples(contract["r1_compatible_binding"])
    require(len(selected) == 40, "M1208 exact40 sample preflight mismatch")
    require(gpu_compute_pids() == [], "GPU has active compute processes")
    require(capture.R1.running_legacy_watchers() == [], "legacy watcher remains present")
    prove_lease_available(policy.repo / LEASE_REL)
    return [str(policy.interpreter), str(policy.repo / CAPTURE_REL),
            "--contract", str(contract_path)], capture


Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(list(command), cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, env=environment)


def execute_once(policy: Policy, raw_executable: Path, version: str, cwd: Path,
                 runner: Runner = default_runner) -> subprocess.CompletedProcess[str]:
    command, capture = preflight(policy, raw_executable, version, cwd)
    completed = runner(command, policy.repo)
    require(completed.returncode == 0,
            "single M1208 child failed after persistent attempt; no retry authorized")
    require(completed.stdout.count(PASS_TOKEN) == 1, "M1208 terminal token mismatch")
    attempt = policy.repo / ATTEMPT_REL
    regular(attempt, "persistent M1208 attempt marker")
    require(attempt.read_text(encoding="ascii") ==
            "M1208_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n",
            "persistent M1208 attempt token mismatch")
    capture.canonical_verify_double_seal(policy.repo / RESULT_REL)
    log = policy.repo / LOG_REL
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        payload = ("M1208_PRODUCTION_LOG__RESULT_SEAL_VERIFIED__RESULT_HAMMER_REQUIRED\n"
                   "automatic_retry=false\nreturncode={}\n--- stdout ---\n{}\n--- stderr ---\n{}"
                   ).format(completed.returncode, completed.stdout, completed.stderr)
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return completed


def main() -> int:
    completed = execute_once(Policy(), Path(sys.executable),
                             "{}.{}.{}".format(*sys.version_info[:3]), Path.cwd())
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
