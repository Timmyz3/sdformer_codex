#!/usr/bin/env python3
"""M836 publication-boundary repair around frozen M832/M828/M819/M809."""

import argparse
import copy
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M832_PATH = HERE / "execute_m832_m828_decoder_directory_bound_consumption.py"
M832_SHA256 = "6af98828c967ef6bcf7d1324a5fdc3883f3bc47f67c134e90b121dfcac35ff13"
M832_CANDIDATE = HW / "contracts/m832_m785_decoder_directory_bound_consumption_candidate_r1_20260829.json"
M832_CANDIDATE_SHA256 = "55e64e4f293d3cbd171c8ec866d02941bb04bb6bc13716a1fc1fdd8167b46c2e"
M835_REVIEW = HW / "reviews/m835_m832_m785_decoder_directory_bound_consumption_source_fresh_hammer_r1_20260829"
M835_REVIEW_JSON_SHA256 = "f8220b2f9e5b15799bf54965e74bc215bac34d91faa4e4311dcc072fc42b0a9e"
M835_MANIFEST_SHA256 = "ff9ef22e0acedea16c1c1e27dde7b2cad6e8f1af3129a32defa00e64e3771245"
M835_OUTER_SEAL_FILE_SHA256 = "9e682bc00555ab543ba98692859fd8c63aaa39c44496cb24abba8e37c5e6b971"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANDIDATE = HW / "contracts/m836_m785_decoder_publication_boundary_repair_candidate_r1_20260829.json"
CANDIDATE_SCHEMA = "m836_m785_decoder_publication_boundary_repair_candidate_v1"
RELEASE_SCHEMA = "m836_m785_decoder_production_true_release_v1"
SOURCE_HAMMER_DIR = "reviews/m839_m836_m785_decoder_publication_boundary_source_fresh_hammer_r1_20260829"
PARENT_ATTEMPT_STATUS = "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY"
CANONICAL_FAILURE_PREFIX = (
    "m836_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
M832_FAILURE_PREFIX = (
    "m832_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
M828_FAILURE_PREFIX = (
    "m828_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
INHERITED_FAILURE_PREFIX = (
    "m785_h67_decoder_physical_residency_production_r1_20260829"
    ".failed_or_incomplete.")
GUARDED_PREFIXES = (CANONICAL_FAILURE_PREFIX, M832_FAILURE_PREFIX,
                    M828_FAILURE_PREFIX, INHERITED_FAILURE_PREFIX)
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")
ATTEMPT_MEMBERS = ("attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")
RENAME_NOREPLACE = 1


class Failure(RuntimeError):
    pass


class ControlledPreproductionStop(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def _load_exact(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " absent")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            name + " SHA drift")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M832 = _load_exact(M832_PATH, M832_SHA256, "m836_frozen_m832")


def sha256(path: Path) -> str:
    return M832.sha256(Path(path))


def strict_json(path: Path) -> object:
    try:
        return M832.strict_json(Path(path))
    except M832.Failure as error:
        raise Failure(str(error)) from error


def verify_sealed(directory: Path) -> Dict[str, str]:
    try:
        return M832.verify_sealed(Path(directory))
    except M832.Failure as error:
        raise Failure(str(error)) from error


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        M832.regular_exact(Path(path), expected, label)
    except M832.Failure as error:
        raise Failure(str(error)) from error


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {name: HW / entry["path"]
            for name, entry in candidate["source_identity"].items()}


def _canonical_paths(candidate: Mapping[str, object]):
    canonical = candidate["canonical"]
    return (REPO / canonical["result"], REPO / canonical["attempt"],
            REPO / canonical["future_release"])


def _directory_identity(value) -> Tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _member_identity(value) -> Tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _fd_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    return flags | os.O_NOFOLLOW


def _safe_basename(name: str, label: str) -> None:
    require(isinstance(name, str) and name and name not in (".", "..") and
            "/" not in name and "\x00" not in name,
            label + " basename malformed")


def _lstat_at(fd: int, name: str):
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _assert_results_binding(parent_fd: int, results_name: str,
                            identity: Tuple[int, int]) -> None:
    current = _lstat_at(parent_fd, results_name)
    require(current is not None and stat.S_ISDIR(current.st_mode) and
            not stat.S_ISLNK(current.st_mode),
            "current results pathname is absent, symlinked, or non-directory")
    require(_directory_identity(current) == identity,
            "current results pathname no longer binds the opened directory FD")


def _scan_once(results_fd: int, prefixes: Sequence[str]):
    before = _member_identity(os.fstat(results_fd))
    names = tuple(sorted(os.listdir(results_fd)))
    matches = []
    for name in names:
        if any(name.startswith(prefix) for prefix in prefixes):
            observed = _lstat_at(results_fd, name)
            matches.append((name, None if observed is None else
                            int(observed.st_mode)))
    after = _member_identity(os.fstat(results_fd))
    return before, after, tuple(matches)


def _require_stable_absence(results_fd: int, prefixes: Sequence[str],
                            invoke_between_hook: bool) -> None:
    first = _scan_once(results_fd, prefixes)
    require(not first[2], "preexisting failure-prefix artifact: " +
            repr(first[2]))
    if invoke_between_hook:
        _after_first_scan_hook()
    second = _scan_once(results_fd, prefixes)
    require(not second[2], "concurrent failure-prefix artifact: " +
            repr(second[2]))
    require(first[0] == first[1] == second[0] == second[1],
            "results directory changed during protected scan")


def _write_exclusive_at(fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    handle = os.open(name, flags, 0o600, dir_fd=fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(handle, payload[offset:])
        os.fsync(handle)
    finally:
        os.close(handle)


def _read_regular_at(fd: int, name: str,
                     expected_identity: Tuple[int, int, int]) -> bytes:
    observed = _lstat_at(fd, name)
    require(observed is not None and stat.S_ISREG(observed.st_mode) and
            not stat.S_ISLNK(observed.st_mode) and
            _member_identity(observed) == expected_identity,
            "attempt member identity drift: " + name)
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        require(_member_identity(os.fstat(handle)) == expected_identity,
                "attempt member raced during open: " + name)
        chunks = []
        while True:
            block = os.read(handle, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(handle)


def _verify_stage_fd(stage_fd: int, stage_identity: Tuple[int, int],
                     member_identities: Mapping[str, Tuple[int, int, int]],
                     payloads: Mapping[str, bytes]) -> None:
    opened = os.fstat(stage_fd)
    require(_directory_identity(opened) == stage_identity and
            stat.S_ISDIR(opened.st_mode), "attempt stage FD identity drift")
    require(set(os.listdir(stage_fd)) == set(ATTEMPT_MEMBERS),
            "attempt stage population drift")
    require(set(member_identities) == set(ATTEMPT_MEMBERS) and
            set(payloads) == set(ATTEMPT_MEMBERS),
            "attempt stage expected population drift")
    for name in ATTEMPT_MEMBERS:
        require(_read_regular_at(stage_fd, name,
                                 member_identities[name]) == payloads[name],
                "attempt stage sealed content drift: " + name)


def _renameat2_noreplace(directory_fd: int, source: str,
                         destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    rc = renameat2(directory_fd, os.fsencode(source), directory_fd,
                   os.fsencode(destination), RENAME_NOREPLACE)
    if rc != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise Failure("canonical attempt collision")
        raise Failure("renameat2 no-replace failed: " + os.strerror(number))


def _cleanup_exact_owned_directory(
        results_fd: int, name: str, directory_identity: Tuple[int, int],
        member_identities: Mapping[str, Tuple[int, int, int]]) -> None:
    observed = _lstat_at(results_fd, name)
    if observed is None:
        return
    require(stat.S_ISDIR(observed.st_mode) and
            _directory_identity(observed) == directory_identity,
            "owned directory identity changed; cleanup refused")
    owned_fd = os.open(name, _fd_flags(), dir_fd=results_fd)
    try:
        require(_directory_identity(os.fstat(owned_fd)) == directory_identity,
                "owned directory raced during cleanup open")
        names = set(os.listdir(owned_fd))
        require(names.issubset(set(ATTEMPT_MEMBERS)),
                "owned directory contains unknown members; cleanup refused")
        for member in sorted(names):
            current = _lstat_at(owned_fd, member)
            require(current is not None and member in member_identities and
                    _member_identity(current) == member_identities[member] and
                    stat.S_ISREG(current.st_mode) and
                    not stat.S_ISLNK(current.st_mode),
                    "owned member identity changed; cleanup refused: " +
                    member)
        for member in ATTEMPT_MEMBERS:
            if member in names:
                os.unlink(member, dir_fd=owned_fd)
        os.fsync(owned_fd)
    finally:
        os.close(owned_fd)
    os.rmdir(name, dir_fd=results_fd)
    os.fsync(results_fd)


def _after_first_scan_hook() -> None:
    return None


def _before_stage_mkdir_hook() -> None:
    return None


def _after_stage_mkdir_hook() -> None:
    return None


def _before_attempt_publish_hook() -> None:
    return None


def _after_final_rebind_hook() -> None:
    return None


def _after_attempt_publish_hook() -> None:
    return None


def atomic_guard_and_consume(results_directory: Path,
                             prefixes: Sequence[str], stage_name: str,
                             attempt_name: str,
                             receipt: Mapping[str, object]) -> Dict[str, object]:
    """Bind scan, sealed stage, publication, and rollback to pinned FDs."""
    directory = Path(results_directory)
    prefixes = tuple(prefixes)
    require(prefixes and len(set(prefixes)) == len(prefixes),
            "guard prefix set malformed")
    for prefix in prefixes:
        _safe_basename(prefix, "guard prefix")
    _safe_basename(stage_name, "attempt stage")
    _safe_basename(attempt_name, "canonical attempt")
    require(stage_name.startswith(attempt_name + ".stage."),
            "attempt stage namespace drift")
    require(re.match(r"^[A-Za-z0-9_.-]+$", stage_name) is not None,
            "attempt stage contains unsafe characters")
    expected_nonce = hashlib.sha256(stage_name.encode("utf-8")).hexdigest()
    require(receipt.get("publication_nonce") == expected_nonce,
            "attempt publication nonce drift")
    parent = directory.parent
    results_name = directory.name
    _safe_basename(results_name, "results directory")
    parent_fd = os.open(str(parent), _fd_flags())
    results_fd = -1
    stage_fd = -1
    stage_created = False
    published = False
    stage_identity = None
    member_identities = {}
    payloads = {}
    try:
        initial = _lstat_at(parent_fd, results_name)
        require(initial is not None and stat.S_ISDIR(initial.st_mode) and
                not stat.S_ISLNK(initial.st_mode),
                "results pathname absent, symlinked, or non-directory")
        results_fd = os.open(results_name, _fd_flags(), dir_fd=parent_fd)
        opened = os.fstat(results_fd)
        identity = _directory_identity(opened)
        require(_directory_identity(initial) == identity,
                "results pathname raced during open")
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, True)
        _assert_results_binding(parent_fd, results_name, identity)
        _before_stage_mkdir_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, False)
        require(_lstat_at(results_fd, attempt_name) is None,
                "canonical attempt already exists")
        require(_lstat_at(results_fd, stage_name) is None,
                "attempt stage collision")
        os.mkdir(stage_name, 0o700, dir_fd=results_fd)
        stage_created = True
        os.fsync(results_fd)
        created = _lstat_at(results_fd, stage_name)
        require(created is not None and stat.S_ISDIR(created.st_mode),
                "private attempt stage creation failed")
        stage_identity = _directory_identity(created)
        stage_fd = os.open(stage_name, _fd_flags(), dir_fd=results_fd)
        require(_directory_identity(os.fstat(stage_fd)) == stage_identity,
                "attempt stage raced during open")
        _after_stage_mkdir_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, False)
        current_stage = _lstat_at(results_fd, stage_name)
        require(current_stage is not None and
                _directory_identity(current_stage) == stage_identity,
                "attempt stage pathname identity drift")
        payload = (json.dumps(receipt, indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
        manifest = (hashlib.sha256(payload).hexdigest() +
                    "  attempt.json\n").encode("ascii")
        outer = (hashlib.sha256(manifest).hexdigest() +
                 "  SHA256SUMS\n").encode("ascii")
        payloads = {"attempt.json": payload, "SHA256SUMS": manifest,
                    "SHA256SUMS.seal.sha256": outer}
        for name in ATTEMPT_MEMBERS:
            _write_exclusive_at(stage_fd, name, payloads[name])
            member_identities[name] = _member_identity(
                os.stat(name, dir_fd=stage_fd, follow_symlinks=False))
        os.fsync(stage_fd)
        _verify_stage_fd(stage_fd, stage_identity, member_identities,
                         payloads)
        _before_attempt_publish_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, False)
        require(_lstat_at(results_fd, attempt_name) is None,
                "canonical attempt collision before publication")
        current_stage = _lstat_at(results_fd, stage_name)
        require(current_stage is not None and stat.S_ISDIR(current_stage.st_mode)
                and _directory_identity(current_stage) == stage_identity,
                "attempt stage identity drift before publication")
        _verify_stage_fd(stage_fd, stage_identity, member_identities,
                         payloads)
        _assert_results_binding(parent_fd, results_name, identity)
        _after_final_rebind_hook()
        _verify_stage_fd(stage_fd, stage_identity, member_identities,
                         payloads)
        _renameat2_noreplace(results_fd, stage_name, attempt_name)
        published = True
        os.fsync(results_fd)
        _after_attempt_publish_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        require(_lstat_at(results_fd, stage_name) is None,
                "attempt stage survived no-replace publication")
        final = _lstat_at(results_fd, attempt_name)
        require(final is not None and stat.S_ISDIR(final.st_mode) and
                _directory_identity(final) == stage_identity,
                "published attempt identity drift")
        attempt_fd = os.open(attempt_name, _fd_flags(), dir_fd=results_fd)
        try:
            require(_directory_identity(os.fstat(attempt_fd)) ==
                    stage_identity,
                    "published attempt raced during canonical open")
            _verify_stage_fd(attempt_fd, stage_identity, member_identities,
                             payloads)
        finally:
            os.close(attempt_fd)
        _verify_stage_fd(stage_fd, stage_identity, member_identities,
                         payloads)
        _assert_results_binding(parent_fd, results_name, identity)
        return {
            "status": "PASS_M836_PUBLICATION_BOUNDARY_CLOSED_ATTEMPT_CONSUMED",
            "results_dev": identity[0], "results_ino": identity[1],
            "attempt_dev": stage_identity[0],
            "attempt_ino": stage_identity[1],
            "publication_nonce": expected_nonce,
            "attempt_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "attempt_outer_seal_file_sha256":
                hashlib.sha256(outer).hexdigest(),
            "guarded_prefixes": list(prefixes),
            "production_cycles": None,
        }
    except (Failure, OSError, ValueError) as error:
        cleanup_name = attempt_name if published else stage_name
        if (stage_created and results_fd >= 0 and stage_identity is not None):
            try:
                _cleanup_exact_owned_directory(
                    results_fd, cleanup_name, stage_identity,
                    member_identities)
            except Exception as cleanup_error:
                raise Failure(str(error) +
                              "; exact owned rollback failed closed: " +
                              str(cleanup_error)) from cleanup_error
        if isinstance(error, Failure):
            raise
        raise Failure(str(error)) from error
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        if results_fd >= 0:
            os.close(results_fd)
        os.close(parent_fd)


def _verify_m835_negative() -> Dict[str, object]:
    identity = verify_sealed(M835_REVIEW)
    regular_exact(M835_REVIEW / "review.json", M835_REVIEW_JSON_SHA256,
                  "M835 negative review")
    require(identity["manifest_sha256"] == M835_MANIFEST_SHA256 and
            identity["outer_seal_file_sha256"] ==
            M835_OUTER_SEAL_FILE_SHA256, "M835 seal drift")
    review = strict_json(M835_REVIEW / "review.json")
    require(review.get("status") ==
            "NO_GO_M832_SOURCE_CANDIDATE__P1_1__PUBLICATION_BOUNDARY_REPAIR_REQUIRED" and
            review.get("true_release_authorized") is False and
            review.get("production_launch_authorized") is False,
            "M835 negative authority weakened")
    return review


def validate_candidate(candidate_path: Path, require_future_absent: bool = True,
                       attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict) and
            candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("status") ==
            "SOURCE_ONLY_M836_PUBLICATION_BOUNDARY_REPAIR_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "M836 candidate identity drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate authorizes production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "temporary_publication_boundary_attacks": True,
        "temporary_zero_row_parent_traversal": True,
        "fresh_source_hammer": True,
        "production_replay": False,
        "result_directory": False,
        "cycles_or_speedup": False,
        "rtl_vcs_eda_gpu_remote": False,
    }, "candidate authorization drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    paths = _candidate_paths(candidate)
    required = {"driver", "runner", "tests", "repair_contract",
                "m832_parent_driver", "m832_parent_candidate",
                "m828_parent_driver", "m819_parent_driver",
                "m809_parent_driver", "m785_contract", "analyzer",
                "storage_oracle", "m785_tests", "m835_no_go_review"}
    require(set(paths) == required, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate driver path drift")
    require(paths["runner"].resolve() == HERE /
            "run_m836_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() == HERE.parent /
            "tests/test_m836_m832_decoder_publication_boundary_repair.py",
            "candidate tests path drift")
    require(paths["m832_parent_driver"].resolve() == M832_PATH and
            candidate["source_identity"]["m832_parent_driver"]["sha256"] ==
            M832_SHA256, "M832 parent driver drift")
    require(paths["m832_parent_candidate"].resolve() == M832_CANDIDATE and
            candidate["source_identity"]["m832_parent_candidate"]["sha256"] ==
            M832_CANDIDATE_SHA256, "M832 parent candidate drift")
    _verify_m835_negative()
    require(candidate.get("m835_no_go_basis") == {
        "directory":
            "reviews/m835_m832_m785_decoder_directory_bound_consumption_source_fresh_hammer_r1_20260829",
        "review_json_sha256": M835_REVIEW_JSON_SHA256,
        "manifest_sha256": M835_MANIFEST_SHA256,
        "outer_seal_file_sha256": M835_OUTER_SEAL_FILE_SHA256,
        "status":
            "NO_GO_M832_SOURCE_CANDIDATE__P1_1__PUBLICATION_BOUNDARY_REPAIR_REQUIRED",
        "m832_true_release_authorized": False,
        "additive_publication_boundary_repair_required": True,
    }, "M835 basis drift")
    parent_value = M832.validate_candidate(M832_CANDIDATE, True, False)
    require(parent_value["status"] ==
            "PASS_M832_DIRECTORY_BOUND_CONSUMPTION_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "M832 parent validation drift")
    parent = strict_json(M832_CANDIDATE)
    require(candidate["common_resource"] == parent["common_resource"] and
            candidate["production_semantics"] ==
            parent["production_semantics"],
            "frozen runtime semantics drift")
    require(candidate["publication_boundary_repair"] == {
        "results_directory": "hw_autoresearch_nts07/results",
        "basename_prefixes": list(GUARDED_PREFIXES),
        "stage_fd_lifetime": "HELD_THROUGH_PUBLICATION_AND_FINAL_CHECK",
        "stage_member_binding": "EXACT_TYPE_DEV_INO_AND_BYTES",
        "stage_population": list(ATTEMPT_MEMBERS),
        "publication": "RENAMEAT2_NOREPLACE_SAME_RESULTS_FD",
        "postpublication_binding":
            "PARENT_DIRFD_RESULTS_DEV_INO_AND_CANONICAL_ATTEMPT_INODE",
        "rollback":
            "EXACT_RECORDED_DIRECTORY_AND_MEMBER_INODES_ONLY_ON_PINNED_FD",
        "publication_nonce": "SHA256_OF_PRIVATE_STAGE_BASENAME",
        "transient_history_claim": False,
    }, "publication boundary repair drift")
    require(candidate["attempt_compatibility"] == {
        "formal_status": PARENT_ATTEMPT_STATUS,
        "accepted_by_m836_validator": True,
        "accepted_by_frozen_m832_m828_m819_m809": True,
        "outer_schema": "m836_m785_decoder_production_attempt_v1",
    }, "attempt compatibility drift")
    require(candidate["canonical"] == {
        "result": "hw_autoresearch_nts07/results/m836_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt": "hw_autoresearch_nts07/results/.m836_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release": "hw_autoresearch_nts07/contracts/m836_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix": "hw_autoresearch_nts07/results/" +
            CANONICAL_FAILURE_PREFIX,
    }, "canonical path drift")
    require(candidate.get("future_release_required_fields") == {
        "schema": RELEASE_SCHEMA,
        "status":
            "TRUE_RELEASE_AFTER_FRESH_M836_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY",
        "launch_now": True, "release": True, "max_attempts": 1,
        "fresh_source_hammer_directory": SOURCE_HAMMER_DIR,
        "fresh_source_hammer_status":
            "PASS100_M836_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY",
    }, "future release requirements drift")
    result, attempt, future = _canonical_paths(candidate)
    require(not result.exists() and not result.is_symlink(),
            "canonical result exists")
    if attempt_required:
        require(attempt.is_dir() and not attempt.is_symlink(),
                "attempt must be consumed")
        verify_sealed(attempt)
    else:
        require(not attempt.exists() and not attempt.is_symlink(),
                "canonical attempt exists")
    if require_future_absent:
        require(not future.exists() and not future.is_symlink(),
                "future release must be absent")
    else:
        require(future.is_file() and not future.is_symlink(),
                "true release absent or nonregular")
    require(candidate["claim_boundary"] == {
        "source_only": True, "production_replay": False,
        "production_cycles": False, "production_speedup": False,
        "energy": False, "ppa": False, "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False, "system_speedup": False,
        "paper_claim": False, "rtl_vcs_eda_energy_ppa": False,
    }, "claim boundary drift")
    return {
        "status":
            "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
        "candidate_sha256": sha256(candidate_path),
        "parent_status": parent_value["status"],
        "production_cycles": None,
    }


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool):
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    cv = validate_candidate(candidate_path, False, attempt_required)
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    require(isinstance(release, dict) and release.get("schema") ==
            RELEASE_SCHEMA and release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_M836_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release.get("launch_now") is True and
            release.get("release") is True and
            release.get("max_attempts") == 1, "release identity drift")
    result, attempt, future = _canonical_paths(candidate)
    require(release_path == future.resolve(), "release path drift")
    binding = release["candidate_binding"]
    require((HW / binding["path"]).resolve() == candidate_path and
            binding["sha256"] == cv["candidate_sha256"],
            "release candidate binding drift")
    require(release["source_identity"] == candidate["source_identity"] and
            release["canonical"] == candidate["canonical"] and
            release["publication_boundary_repair"] ==
            candidate["publication_boundary_repair"],
            "release source/canonical/repair drift")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR, "hammer path drift")
    hammer_dir = HW / hammer["directory"]
    identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M839 source hammer")
    require(identity["manifest_sha256"] == hammer["manifest_sha256"] and
            identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"], "hammer seal drift")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M836_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("true_release_authorized") is True and
            review.get("production_launch_authorized") is False,
            "source hammer authorization drift")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio":
            "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M832_M828_M819_M809_EXACT_SHA",
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }, "release runtime semantics drift")
    expected = os.environ.get("M836_EXPECTED_RELEASE_SHA256", "")
    require(len(expected) == 64 and sha256(release_path) == expected,
            "caller did not supply exact release SHA")
    sidecar = Path(str(release_path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink() and
            sidecar.read_text(encoding="utf-8") ==
            expected + "  " + release_path.name + "\n" and
            outer.read_text(encoding="utf-8") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            "release sidecar drift")
    return {"release": release, "candidate": candidate,
            "candidate_validation": cv}


def consume_formal_attempt(release_path: Path, candidate_path: Path,
                           runner_path: Path, stage_name: str,
                           expected_runner_sha256: str) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, False)
    candidate = gate["candidate"]
    regular_exact(runner_path, expected_runner_sha256, "M836 runner")
    require(candidate["source_identity"]["runner"]["sha256"] ==
            expected_runner_sha256, "runner binding drift")
    result, attempt, future = _canonical_paths(candidate)
    require(attempt.parent.resolve() == (HW / "results").resolve(),
            "attempt results directory drift")
    nonce = hashlib.sha256(stage_name.encode("utf-8")).hexdigest()
    receipt = {
        "schema": "m836_m785_decoder_production_attempt_v1",
        "status": PARENT_ATTEMPT_STATUS,
        "outer_boundary": "M836_POSTPUBLICATION_BOUND_AND_SEALED",
        "publication_nonce": nonce,
        "runner_sha256": expected_runner_sha256,
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "candidate_sha256": sha256(candidate_path),
        "release_sha256": sha256(release_path),
        "canonical_result": str(result), "max_attempts": 1,
        "claim_boundary": {
            "cycles_before_result_hammer": False,
            "speedup_before_result_hammer": False,
            "decoder_complete": False, "full_network_completion": False,
            "table_a_insertion_allowed": False,
        },
    }
    return atomic_guard_and_consume(attempt.parent, GUARDED_PREFIXES,
                                    stage_name, attempt.name, receipt)


def validate_consumed_attempt(release_path: Path, candidate_path: Path,
                              attempt_path: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, True)
    candidate = gate["candidate"]
    canonical = (REPO / candidate["canonical"]["attempt"]).resolve()
    attempt_path = Path(attempt_path).resolve()
    require(attempt_path == canonical, "attempt path drift")
    identity = verify_sealed(attempt_path)
    receipt = strict_json(attempt_path / "attempt.json")
    require(receipt.get("schema") ==
            "m836_m785_decoder_production_attempt_v1" and
            receipt.get("status") == PARENT_ATTEMPT_STATUS and
            receipt.get("max_attempts") == 1 and
            isinstance(receipt.get("publication_nonce"), str) and
            len(receipt["publication_nonce"]) == 64 and
            receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            receipt.get("runner_sha256") ==
            candidate["source_identity"]["runner"]["sha256"] and
            receipt.get("release_sha256") == sha256(release_path),
            "consumed attempt identity drift")
    return {
        "status":
            "PASS_M836_PUBLICATION_BOUND_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
        "attempt_manifest_sha256": identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256":
            identity["outer_seal_file_sha256"],
        "scheduled_rows": 0, "production_cycles": None,
    }


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, True)
    original = M832.validate_true_release
    M832.validate_true_release = lambda *args, **kwargs: gate
    try:
        result = M832.run_production(release_path, candidate_path,
                                     attempt_path, output)
    finally:
        M832.validate_true_release = original
    result = copy.deepcopy(result)
    result["schema"] = (
        "m836_m785_decoder_physical_residency_production_result_v1")
    result["delegated_publication_boundary_repair"] = {
        "driver_path": str(Path(__file__).resolve().relative_to(HW)),
        "parent_driver_path": str(M832_PATH.relative_to(HW)),
        "parent_driver_sha256": M832_SHA256,
        "semantics_changed": False,
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }
    M832.M828.M819.M809._write_json(Path(output) / "result.m836.json",
                                    result)
    (Path(output) / "result.json").unlink()
    (Path(output) / "result.m836.json").rename(Path(output) / "result.json")
    return result


def publish_no_replace(candidate_path: Path, stage: Path,
                       destination: Path) -> Dict[str, object]:
    candidate = strict_json(candidate_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "publication candidate drift")
    canonical = (REPO / candidate["canonical"]["result"]).resolve()
    stage, destination = Path(stage).resolve(), Path(destination).resolve()
    require(destination == canonical and stage.parent == canonical.parent and
            stage.name.startswith(canonical.name + ".stage."),
            "publication path drift")
    identity = verify_sealed(stage)
    require({entry.name for entry in stage.iterdir()} == {
        "result.json", "detailed_rows.json", "SHA256SUMS",
        "SHA256SUMS.seal.sha256"}, "stage population drift")
    try:
        M832.M828.M819.M809._rename_noreplace(stage, destination)
    except M832.M828.M819.M809.Failure as error:
        raise Failure(str(error)) from error
    require(verify_sealed(destination) == identity and not stage.exists(),
            "publication transition drift")
    return {"status": "PASS_M836_ATOMIC_NOREPLACE_PUBLICATION", **identity}


def publish_failure_receipt(candidate_path: Path, release_path: Path,
                            attempt_path: Path, runner_path: Path,
                            stdout_path: Path, stderr_path: Path,
                            output: Path, expected_runner_sha256: str,
                            expected_release_sha256: str, return_code: int,
                            phase: str, partial_artifact: str):
    require(return_code != 0 and phase, "failure identity malformed")
    gate = validate_true_release(release_path, candidate_path, True)
    candidate = gate["candidate"]
    regular_exact(runner_path, expected_runner_sha256, "failure runner")
    regular_exact(release_path, expected_release_sha256, "failure release")
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    canonical_result = (REPO / candidate["canonical"]["result"]).resolve()
    require(Path(attempt_path).resolve() == canonical_attempt,
            "failure attempt path drift")
    identity = verify_sealed(attempt_path)
    output = Path(output).resolve()
    require(output.parent == canonical_result.parent and
            output.name.startswith(canonical_result.name +
                                   ".failed_or_incomplete."),
            "failure quarantine path drift")
    payload = {
        "schema": "m836_m785_decoder_production_failure_receipt_v1",
        "date": "2026-08-29",
        "status": "FAILED_OR_INCOMPLETE__NO_CYCLES_CITABLE",
        "return_code": int(return_code), "phase": phase,
        "partial_artifact": partial_artifact,
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256": expected_runner_sha256,
        "release_sha256": expected_release_sha256,
        "attempt_manifest_sha256": identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256": identity["outer_seal_file_sha256"],
        "claim_boundary": {"production_complete": False,
                           "cycles_citable": False,
                           "speedup_citable": False,
                           "decoder_complete": False,
                           "table_a_insertion_allowed": False},
    }
    try:
        value = M832.M828._write_failure_receipt(
            output, stdout_path, stderr_path, payload)
    except M832.M828.Failure as error:
        raise Failure(str(error)) from error
    return {"status": "PASS_M836_SEALED_FAILURE_RECEIPT", **value}


def _source_receipt(stage_name: str) -> Dict[str, object]:
    return {
        "schema": "m836_source_test",
        "status": PARENT_ATTEMPT_STATUS,
        "publication_nonce":
            hashlib.sha256(stage_name.encode("utf-8")).hexdigest(),
    }


def _directory_swap_attack(hook_name: str) -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m836_swap_") as directory:
        top = Path(directory)
        current, old = top / "results", top / "old"
        current.mkdir()
        artifact_name = CANONICAL_FAILURE_PREFIX + "replacement"
        original = globals()[hook_name]
        fired = [False]

        def swap():
            if not fired[0]:
                fired[0] = True
                current.rename(old)
                current.mkdir()
                (current / artifact_name).write_text("KEEP", encoding="utf-8")

        globals()[hook_name] = swap
        rejected = False
        try:
            try:
                atomic_guard_and_consume(
                    current, GUARDED_PREFIXES, "attempt.stage.swap", "attempt",
                    _source_receipt("attempt.stage.swap"))
            except Failure:
                rejected = True
        finally:
            globals()[hook_name] = original
        require(rejected and fired[0] and
                (current / artifact_name).read_text(encoding="utf-8") ==
                "KEEP" and not (old / "attempt").exists() and
                not (old / "attempt.stage.swap").exists(),
                "directory replacement rollback attack failed")
        return {"hook": hook_name, "rejected": True,
                "self_publication_rolled_back": True,
                "replacement_unchanged": True}


def _prepublish_content_attack() -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m836_tamper_") as directory:
        root = Path(directory)
        stage_name = "attempt.stage.tamper"
        original = globals()["_before_attempt_publish_hook"]
        fired = [False]

        def tamper():
            fired[0] = True
            (root / stage_name / "attempt.json").write_text(
                "ATTACK\n", encoding="utf-8")

        globals()["_before_attempt_publish_hook"] = tamper
        rejected = False
        try:
            try:
                atomic_guard_and_consume(
                    root, GUARDED_PREFIXES, stage_name, "attempt",
                    _source_receipt(stage_name))
            except Failure:
                rejected = True
        finally:
            globals()["_before_attempt_publish_hook"] = original
        require(rejected and fired[0] and
                not (root / stage_name).exists() and
                not (root / "attempt").exists(),
                "prepublication content mutation was not rejected/rolled back")
        return {"rejected": True, "recorded_inode_stage_removed": True,
                "canonical_attempt_absent": True}


def preproduction_traversal_test() -> Dict[str, object]:
    candidate = strict_json(CANDIDATE)
    with tempfile.TemporaryDirectory(prefix="m836_preproduction_") as directory:
        root = Path(directory)
        results = root / "results"
        results.mkdir()
        candidate_path = root / "candidate.json"
        release_path = root / "release.json"
        output = results / "result.stage.test"
        M832.M828.M819.M809._write_json(candidate_path, {"source_test": True})
        M832.M828.M819.M809._write_json(release_path, {"source_test": True})
        fake = copy.deepcopy(candidate)
        fake["canonical"]["attempt"] = str(results / "attempt")
        fake["canonical"]["result"] = str(results / "result")
        stage_name = "attempt.stage.source-test"
        receipt = {
            "schema": "m836_m785_decoder_production_attempt_v1",
            "status": PARENT_ATTEMPT_STATUS,
            "publication_nonce":
                hashlib.sha256(stage_name.encode("utf-8")).hexdigest(),
            "candidate_sha256": sha256(candidate_path),
            "release_sha256": sha256(release_path),
        }
        consumed = atomic_guard_and_consume(
            results, GUARDED_PREFIXES, stage_name, "attempt", receipt)
        attempt_path = results / "attempt"
        attempt_before = sha256(attempt_path / "attempt.json")
        original_local_validate = globals()["validate_true_release"]
        original_m832_validate = M832.validate_true_release
        original_m828_validate = M832.M828.validate_true_release
        original_m819_validate = M832.M828.M819.validate_true_release
        original_m809_validate = M832.M828.M819.M809.validate_true_release
        original_path = M832.M828.M819.M809.Path
        base_path = type(Path())
        target = os.path.abspath(str(output))

        class GuardedPath(base_path):
            def mkdir(self, *args, **kwargs):
                if os.path.abspath(str(self)) == target:
                    raise ControlledPreproductionStop("AT_M809_OUTPUT_MKDIR")
                return super().mkdir(*args, **kwargs)

        globals()["validate_true_release"] = lambda *args, **kwargs: {
            "candidate": fake}
        M832.M828.M819.M809.Path = GuardedPath
        stopped = False
        try:
            try:
                run_production(release_path, candidate_path,
                               attempt_path, output)
            except ControlledPreproductionStop as error:
                require(str(error) == "AT_M809_OUTPUT_MKDIR",
                        "wrong controlled stop")
                stopped = True
        finally:
            M832.M828.M819.M809.Path = original_path
            globals()["validate_true_release"] = original_local_validate
        require(M832.validate_true_release is original_m832_validate and
                M832.M828.validate_true_release is original_m828_validate and
                M832.M828.M819.validate_true_release is
                original_m819_validate and
                M832.M828.M819.M809.validate_true_release is
                original_m809_validate,
                "delegated validators not restored")
        require(stopped and not output.exists() and
                sha256(attempt_path / "attempt.json") == attempt_before,
                "M836 traversal crossed zero-row boundary")
        return {
            "status":
                "PASS_M836_PUBLICATION_BOUNDARY_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL",
            "consume_status": consumed["status"],
            "entered_exact_m832": True, "entered_exact_m828": True,
            "entered_exact_m819": True, "entered_exact_m809": True,
            "parent_attempt_status_accepted": True,
            "stopped_at": "M809_OUTPUT_MKDIR", "scheduled_rows": 0,
            "output_exists": False,
            "attempt_receipt_identity_drift": False,
            "delegated_validators_restored": True,
            "production_cycles": None,
        }


def self_test() -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m836_selftest_") as directory:
        root = Path(directory)
        stage_name = "attempt.stage.clean"
        value = atomic_guard_and_consume(
            root, GUARDED_PREFIXES, stage_name, "attempt",
            _source_receipt(stage_name))
    tamper = _prepublish_content_attack()
    final_rebind_swap = _directory_swap_attack("_after_final_rebind_hook")
    postpublish_swap = _directory_swap_attack("_after_attempt_publish_hook")
    traversal = preproduction_traversal_test()
    return {
        "status":
            "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SYNTHETIC_SELF_TEST",
        "clean_consume": value["status"],
        "prepublish_content_change": tamper,
        "after_final_rebind_swap": final_rebind_swap,
        "postpublish_swap": postpublish_swap,
        "traversal": traversal["status"], "scheduled_rows": 0,
        "formal_attempt_created": False, "production_cycles": None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preproduction-traversal-test", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--validate-release-preflight", action="store_true")
    parser.add_argument("--guard-and-consume-attempt", action="store_true")
    parser.add_argument("--validate-consumed-attempt", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--write-failure-receipt", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--stage-basename")
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-release-sha256")
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    modes = (args.self_test, args.preproduction_traversal_test,
             args.validate_candidate, args.validate_release_preflight,
             args.guard_and_consume_attempt, args.validate_consumed_attempt,
             args.run_production, args.publish_no_replace,
             args.write_failure_receipt)
    require(sum(bool(value) for value in modes) == 1,
            "select exactly one mode")
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return 0
    if args.preproduction_traversal_test:
        print(json.dumps(preproduction_traversal_test(), sort_keys=True,
                         allow_nan=False))
        return 0
    require(args.candidate is not None, "candidate is required")
    if args.validate_candidate:
        print(json.dumps(validate_candidate(args.candidate), sort_keys=True,
                         allow_nan=False))
        return 0
    if args.validate_release_preflight:
        require(args.release is not None, "release is required")
        value = validate_true_release(args.release, args.candidate, False)
        print(json.dumps({
            "status": "PASS_M836_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
            "candidate_sha256":
                value["candidate_validation"]["candidate_sha256"],
            "release_sha256": sha256(args.release),
            "production_cycles": None}, sort_keys=True, allow_nan=False))
        return 0
    if args.guard_and_consume_attempt:
        require(args.release is not None and args.runner is not None and
                args.stage_basename and args.expected_runner_sha256,
                "consume inputs incomplete")
        print(json.dumps(consume_formal_attempt(
            args.release, args.candidate, args.runner, args.stage_basename,
            args.expected_runner_sha256), sort_keys=True, allow_nan=False))
        return 0
    if args.validate_consumed_attempt:
        require(args.release is not None and args.attempt is not None,
                "release and attempt are required")
        print(json.dumps(validate_consumed_attempt(
            args.release, args.candidate, args.attempt), sort_keys=True,
            allow_nan=False))
        return 0
    if args.publish_no_replace:
        require(args.output is not None and args.publish_to is not None,
                "stage and destination are required")
        print(json.dumps(publish_no_replace(args.candidate, args.output,
                                            args.publish_to), sort_keys=True,
                         allow_nan=False))
        return 0
    if args.write_failure_receipt:
        require(all((args.release, args.attempt, args.runner, args.stdout_log,
                     args.stderr_log, args.output,
                     args.expected_runner_sha256,
                     args.expected_release_sha256,
                     args.return_code is not None, args.phase)),
                "failure receipt inputs incomplete")
        print(json.dumps(publish_failure_receipt(
            args.candidate, args.release, args.attempt, args.runner,
            args.stdout_log, args.stderr_log, args.output,
            args.expected_runner_sha256, args.expected_release_sha256,
            args.return_code, args.phase, args.partial_artifact),
            sort_keys=True, allow_nan=False))
        return 0
    require(args.release is not None and args.attempt is not None and
            args.output is not None, "production inputs incomplete")
    value = run_production(args.release, args.candidate, args.attempt,
                           args.output)
    print(json.dumps({"status": value["status"],
                     "result_sha256": sha256(Path(args.output) /
                                              "result.json"),
                     "detailed_rows": value["detailed_rows"],
                     "fresh_result_hammer_required": True},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
