#!/opt/conda/envs/sdformerflow/bin/python
"""Inert, fail-closed remote launcher for the M1180 ep29 unified capture.

This zero-argument wrapper is unusable until a fresh M1184 release hammer is
present and recursively sealed.  Production performs one exact child launch;
the child owns the canonical GPU lease and persistent M1180 attempt marker.
There is no automatic retry.  Importing this module is side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
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
    "capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py")
CONTRACT_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json")
SOURCE_CONTRACT_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1180_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")
TEST_REL = Path(
    "hw_autoresearch_nts07/tests/"
    "test_m1180_motion_checkpoint_parametric_unified_capture_r2_source.py")
AUTHOR_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1180_motion_checkpoint_parametric_unified_capture_r2_author_r1_20260830")
SOURCE_HAMMER_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1181_m1180_motion_checkpoint_parametric_unified_capture_r2_source_hammer_r1_20260830")
RELEASE_HAMMER_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830")
RELEASE_AUTHOR_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1182_m1180_motion_ep29_unified_capture_launch_release_author_r1_20260830")
DEPENDENCY_INVENTORY_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
TRANSFER_LIST_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1182_m1180_motion_ep29_unified_capture_remote_transfer_files_r1_20260830.txt")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
LEASE_REL = Path("hw_autoresearch_nts07/results/gpu_profile_lease.lock")
ATTEMPT_REL = Path(
    "hw_autoresearch_nts07/results/"
    ".m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
RESULT_REL = Path(
    "hw_autoresearch_nts07/results/"
    "m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
LOG_REL = Path(
    "hw_autoresearch_nts07/results/"
    ".m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")

SOURCE_SHA = "f88426c789c99a0d56c34ffaa742b052c73fcbad600c4ecd5797a62e2cf26479"
SOURCE_CONTRACT_SHA = "bcc91d46cf02b3b3d1011287fb7c4d287431db08dba71eef22d5037b06c1d8df"
SOURCE_TEST_SHA = "6cb33ac3abcbc8678d8a3038afb87a895d7cc65cdd3f5d2fe4307b19f96ad57d"
AUTHOR_MANIFEST_SHA = "1363a7256655b8b64874099b6de7d4ac87a93ffe5712afa5fdfcb94371393547"
AUTHOR_OUTER_FILE_SHA = "d7bc3196af16c8f97fbc07bd11ac477f8b942b222042b372e459843a6cfe7e36"
SOURCE_HAMMER_REVIEW_SHA = "2dc8f5b39c990d67fd73d9f5fc8ff5167b17c6759d93781ee8dbdad128d05330"
SOURCE_HAMMER_MANIFEST_SHA = "8c483b73ee4623f1a1876f55b710e4292e3f21530907b7084f41efa71398c837"
SOURCE_HAMMER_OUTER_FILE_SHA = "9b85611c24595f70d4e08b12522294c1a98c53a0a5981cce91873af4d1c1499b"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RELEASE_HAMMER_SCHEMA = (
    "m1184_m1182_m1180_motion_unified_capture_launch_release_hammer_r1_v1")
PASS_TOKEN = "PASS_M1180_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"


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


def directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "{} must be a non-symlink directory: {}".format(label, path))


def executable(path: Path, label: str) -> None:
    try:
        target = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ReleaseError("missing/broken {}: {}".format(label, path)) from exc
    require(target.is_file() and os.access(target, os.X_OK),
            "{} target is not executable: {}".format(label, target))


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


def load_capture(path: Path, expected_sha: str) -> Any:
    regular(path, "M1180 capture source")
    require(sha256(path) == expected_sha, "M1180 capture source SHA drift")
    spec = importlib.util.spec_from_file_location("m1182_sealed_m1180_capture", path)
    require(spec is not None and spec.loader is not None, "cannot import M1180 capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(sha256(path) == expected_sha, "M1180 capture changed during import")
    return module


def validate_release_hammer(module: Any, policy: "Policy") -> dict[str, Any]:
    root = policy.repo / RELEASE_HAMMER_REL
    rows = module.canonical_verify_double_seal(root)
    require("review.json" in rows, "release hammer lacks review.json")
    review = strict_json(root / "review.json")
    require(review.get("schema") == RELEASE_HAMMER_SCHEMA and
            review.get("status") == "PASS", "release hammer semantic admission mismatch")
    bindings = review.get("bindings", {})
    require(bindings.get("launcher_sha256") == sha256(Path(__file__).resolve()) and
            bindings.get("launch_contract_sha256") == sha256(policy.repo / CONTRACT_REL) and
            bindings.get("capture_source_sha256") == SOURCE_SHA and
            bindings.get("capture_source_contract_sha256") == SOURCE_CONTRACT_SHA and
            bindings.get("capture_source_test_sha256") == SOURCE_TEST_SHA and
            bindings.get("capture_author_manifest_sha256") == AUTHOR_MANIFEST_SHA and
            bindings.get("capture_author_outer_file_sha256") == AUTHOR_OUTER_FILE_SHA and
            bindings.get("source_hammer_review_sha256") == SOURCE_HAMMER_REVIEW_SHA and
            bindings.get("source_hammer_manifest_sha256") == SOURCE_HAMMER_MANIFEST_SHA and
            bindings.get("source_hammer_outer_file_sha256") == SOURCE_HAMMER_OUTER_FILE_SHA and
            bindings.get("dependency_inventory_sha256") ==
            sha256(policy.repo / DEPENDENCY_INVENTORY_REL) and
            bindings.get("transfer_list_sha256") == sha256(policy.repo / TRANSFER_LIST_REL),
            "release hammer exact binding mismatch")
    require(bindings.get("release_author_manifest_sha256") ==
            sha256(policy.repo / RELEASE_AUTHOR_REL / "SHA256SUMS") and
            bindings.get("release_author_outer_file_sha256") ==
            sha256(policy.repo / RELEASE_AUTHOR_REL / "SHA256SUMS.seal.sha256"),
            "release hammer author-seal binding mismatch")
    require(review.get("authorization", {}).get("exact_remote_launch") is True and
            review.get("authorization", {}).get("automatic_retry") is False,
            "release hammer does not authorize exact no-retry launch")
    return review


def validate_dependency_inventory(policy: "Policy", release_review: dict[str, Any]) -> dict[str, Any]:
    inventory_path = policy.repo / DEPENDENCY_INVENTORY_REL
    transfer_path = policy.repo / TRANSFER_LIST_REL
    regular(inventory_path, "remote dependency inventory")
    regular(transfer_path, "exact transfer file list")
    bindings = release_review["bindings"]
    require(sha256(inventory_path) == bindings["dependency_inventory_sha256"] and
            sha256(transfer_path) == bindings["transfer_list_sha256"],
            "dependency inventory/transfer-list release binding mismatch")
    inventory = strict_json(inventory_path)
    require(inventory.get("schema") ==
            "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
            "remote dependency inventory semantic mismatch")
    require(inventory.get("remote_repository") == str(policy.repo) and
            inventory.get("remote_interpreter") == str(policy.interpreter),
            "remote dependency inventory runtime mismatch")
    rows = inventory.get("dependencies")
    require(isinstance(rows, list) and rows, "remote dependency inventory is empty")
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                relative.as_posix() == row["path"] and row["path"] not in observed,
                "unsafe or duplicate dependency path")
        require(row["disposition"] in {"transfer_required", "remote_existing_hash_verify"},
                "invalid dependency disposition")
        path = policy.repo / relative
        regular(path, "remote dependency " + row["label"])
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                "remote dependency identity mismatch: " + row["label"])
        observed[row["path"]] = row
    required_labels = set(inventory["required_labels"])
    require({row["label"] for row in rows} == required_labels,
            "remote dependency required-label population mismatch")
    transfer_lines = [line for line in transfer_path.read_text(encoding="utf-8").splitlines()
                      if line]
    expected_transfer = sorted(
        [row["path"] for row in rows if row["disposition"] == "transfer_required"] +
        [DEPENDENCY_INVENTORY_REL.as_posix(), TRANSFER_LIST_REL.as_posix()])
    require(transfer_lines == expected_transfer and len(transfer_lines) == len(set(transfer_lines)),
            "exact remote transfer file list mismatch")
    return inventory


@dataclass(frozen=True)
class Policy:
    repo: Path = REPO
    interpreter: Path = INTERPRETER
    python_version: str = PYTHON_VERSION


def gpu_compute_pids(runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[int]:
    completed = runner(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0, "cannot prove GPU compute-process idleness")
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ReleaseError("malformed nvidia-smi compute PID output") from exc


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


def preflight(policy: Policy, raw_executable: Path, version: str, cwd: Path) -> tuple[list[str], Any]:
    require(raw_executable == policy.interpreter, "remote interpreter path mismatch")
    require(version == policy.python_version, "remote interpreter version mismatch")
    require(cwd == policy.repo, "remote repository cwd mismatch")
    directory(policy.repo, "remote repository")
    executable(policy.interpreter, "remote interpreter")

    exact = {
        CAPTURE_REL: SOURCE_SHA,
        SOURCE_CONTRACT_REL: SOURCE_CONTRACT_SHA,
        TEST_REL: SOURCE_TEST_SHA,
        DOCS359_REL: DOCS359_SHA,
    }
    for relative, expected in exact.items():
        path = policy.repo / relative
        regular(path, "sealed prerequisite " + str(relative))
        require(sha256(path) == expected, "sealed prerequisite SHA drift: " + str(relative))

    capture = load_capture(policy.repo / CAPTURE_REL, SOURCE_SHA)
    capture.canonical_verify_double_seal(
        policy.repo / AUTHOR_REL, AUTHOR_MANIFEST_SHA, AUTHOR_OUTER_FILE_SHA)
    source_rows = capture.canonical_verify_double_seal(
        policy.repo / SOURCE_HAMMER_REL,
        SOURCE_HAMMER_MANIFEST_SHA, SOURCE_HAMMER_OUTER_FILE_SHA)
    require(source_rows.get("review.json") == SOURCE_HAMMER_REVIEW_SHA,
            "source hammer review member SHA mismatch")
    release_review = validate_release_hammer(capture, policy)
    inventory = validate_dependency_inventory(policy, release_review)

    contract_path = policy.repo / CONTRACT_REL
    regular(contract_path, "M1180 launch contract")
    contract = capture.strict_json(contract_path)
    capture.validate_launch_contract(contract, contract_path)
    require(contract["remote_runtime_identity"] == {
        "repository": str(policy.repo),
        "interpreter": str(policy.interpreter),
        "python_version": policy.python_version,
    }, "launch contract remote runtime mismatch")
    require(contract["one_shot"]["attempt_marker"] == str(ATTEMPT_REL) and
            contract["output"]["path"] == str(RESULT_REL) and
            contract["production_log"]["path"] == str(LOG_REL) and
            contract["gpu_ownership"]["lease_path"] == str(LEASE_REL),
            "launch contract canonical namespace mismatch")
    frozen_rows = {row["path"]: row for row in contract["cohort"]["samples"]}
    dependency_rows = {row["path"]: row for row in inventory["dependencies"]}
    require(len(frozen_rows) == 40 and all(
        path in dependency_rows and
        dependency_rows[path]["size_bytes"] == row["bytes"] and
        dependency_rows[path]["sha256"] == row["sha256"] and
        dependency_rows[path]["disposition"] == "remote_existing_hash_verify"
        for path, row in frozen_rows.items()),
        "forty-source contract/dependency-inventory cross-binding mismatch")
    require(not os.path.lexists(policy.repo / ATTEMPT_REL) and
            not os.path.lexists(policy.repo / RESULT_REL) and
            not os.path.lexists(policy.repo / LOG_REL),
            "M1180 attempt/result/log namespace is not fresh")
    require(gpu_compute_pids() == [], "GPU has active compute processes")
    require(capture.R1.running_legacy_watchers() == [],
            "legacy M511 watcher remains present, including SIGSTOP")
    prove_lease_available(policy.repo / LEASE_REL)

    r1_source = policy.repo / capture.BASE.R1_PATH
    regular(r1_source, "one-load M1174 substrate")
    require(sha256(r1_source) == capture.BASE.R1_SHA256 and
            r1_source.read_text(encoding="utf-8").count("profile.build_model(") == 1,
            "one-model-load substrate rule mismatch")
    command = [str(policy.interpreter), str(policy.repo / CAPTURE_REL),
               "--contract", str(contract_path)]
    return command, capture


Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(list(command), cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, env=environment)


def write_production_log(path: Path, command: Sequence[str], completed: subprocess.CompletedProcess[str]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        payload = (
            "M1180_PRODUCTION_LOG__RESULT_SEAL_VERIFIED__RESULT_HAMMER_REQUIRED\n"
            "automatic_retry=false\n"
            "command_sha256={}\nreturncode={}\n--- stdout ---\n{}\n--- stderr ---\n{}"
        ).format(hashlib.sha256(os.fsencode(chr(0).join(command))).hexdigest(),
                 completed.returncode, completed.stdout, completed.stderr).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_once(policy: Policy, raw_executable: Path, version: str, cwd: Path,
                 runner: Runner = default_runner) -> subprocess.CompletedProcess[str]:
    command, capture = preflight(policy, raw_executable, version, cwd)
    completed = runner(command, policy.repo)
    require(completed.returncode == 0,
            "single M1180 child failed after its persistent attempt; no retry authorized")
    require(completed.stdout.count(PASS_TOKEN) == 1,
            "single M1180 child terminal token mismatch")
    attempt = policy.repo / ATTEMPT_REL
    regular(attempt, "persistent M1180 attempt marker")
    require(attempt.read_text(encoding="ascii") ==
            "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n",
            "persistent M1180 attempt token mismatch")
    capture.canonical_verify_double_seal(policy.repo / RESULT_REL)
    write_production_log(policy.repo / LOG_REL, command, completed)
    return completed


def main() -> int:
    require(len(sys.argv) == 1, "production M1182 launcher accepts zero arguments")
    completed = execute_once(Policy(), Path(sys.executable), platform.python_version(), Path.cwd())
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1182_M1180_REMOTE_ONE_SHOT__SEALED_CAPTURE_PRESENT__FRESH_RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
