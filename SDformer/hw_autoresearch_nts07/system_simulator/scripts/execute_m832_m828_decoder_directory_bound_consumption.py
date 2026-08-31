#!/usr/bin/env python3
"""M832 directory-FD-bound one-shot wrapper around frozen M828/M819/M809."""

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
M828_PATH = HERE / "execute_m828_m819_decoder_failure_prefix_guard.py"
M828_SHA256 = "3788a70980092e6eb21394d3b1fad49acd2cbc6059d9c59c6b4d3a02a6beb781"
M828_CANDIDATE = HW / "contracts/m828_m785_decoder_failure_prefix_guard_candidate_r1_20260829.json"
M828_CANDIDATE_SHA256 = "c1aaca4da0380ad0e03bc8d4bebf5ab2fcde55472ae8d2602ba60207e78334a6"
M831_REVIEW = HW / "reviews/m831_m828_m785_decoder_failure_prefix_guard_source_fresh_hammer_r1_20260829"
M831_REVIEW_JSON_SHA256 = "ac39d7dac0d996234594ccb1a5d3993a35f33af19ee59079592081a0bef5928b"
M831_MANIFEST_SHA256 = "3866cb211b03661ac9ab030d8667381a20d8f26c09868d9960a33a3ed59ee724"
M831_OUTER_SEAL_FILE_SHA256 = "32f55f511fe52c3782cf31c64cd053d7c051563dfa8c07667ca67b1c118caf68"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANDIDATE = HW / "contracts/m832_m785_decoder_directory_bound_consumption_candidate_r1_20260829.json"
CANDIDATE_SCHEMA = "m832_m785_decoder_directory_bound_consumption_candidate_v1"
RELEASE_SCHEMA = "m832_m785_decoder_production_true_release_v1"
SOURCE_HAMMER_DIR = "reviews/m835_m832_m785_decoder_directory_bound_consumption_source_fresh_hammer_r1_20260829"
PARENT_ATTEMPT_STATUS = "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY"
CANONICAL_FAILURE_PREFIX = (
    "m832_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
M828_FAILURE_PREFIX = (
    "m828_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
INHERITED_FAILURE_PREFIX = (
    "m785_h67_decoder_physical_residency_production_r1_20260829"
    ".failed_or_incomplete.")
GUARDED_PREFIXES = (CANONICAL_FAILURE_PREFIX, M828_FAILURE_PREFIX,
                    INHERITED_FAILURE_PREFIX)
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


M828 = _load_exact(M828_PATH, M828_SHA256, "m832_frozen_m828")


def sha256(path: Path) -> str:
    return M828.sha256(Path(path))


def strict_json(path: Path) -> object:
    try:
        return M828.strict_json(Path(path))
    except M828.Failure as error:
        raise Failure(str(error)) from error


def verify_sealed(directory: Path) -> Dict[str, str]:
    try:
        return M828.verify_sealed(Path(directory))
    except M828.Failure as error:
        raise Failure(str(error)) from error


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        M828.regular_exact(Path(path), expected, label)
    except M828.Failure as error:
        raise Failure(str(error)) from error


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {name: HW / entry["path"]
            for name, entry in candidate["source_identity"].items()}


def _canonical_paths(candidate: Mapping[str, object]):
    canonical = candidate["canonical"]
    return (REPO / canonical["result"], REPO / canonical["attempt"],
            REPO / canonical["future_release"])


def _directory_token(value) -> Tuple[int, int, int]:
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
    require((int(current.st_dev), int(current.st_ino)) == identity,
            "current results pathname no longer binds the opened directory FD")


def _scan_once(results_fd: int, prefixes: Sequence[str]):
    before = _directory_token(os.fstat(results_fd))
    names = tuple(sorted(os.listdir(results_fd)))
    matches = []
    for name in names:
        if any(name.startswith(prefix) for prefix in prefixes):
            observed = _lstat_at(results_fd, name)
            matches.append((name, None if observed is None else
                            int(observed.st_mode)))
    after = _directory_token(os.fstat(results_fd))
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    handle = os.open(name, flags, 0o600, dir_fd=fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(handle, payload[offset:])
        os.fsync(handle)
    finally:
        os.close(handle)


def _read_at(fd: int, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    handle = os.open(name, flags, dir_fd=fd)
    try:
        chunks = []
        while True:
            block = os.read(handle, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(handle)


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


def _cleanup_private_stage(results_fd: int, stage_name: str,
                           stage_identity: Tuple[int, int]) -> None:
    observed = _lstat_at(results_fd, stage_name)
    if observed is None:
        return
    require(stat.S_ISDIR(observed.st_mode) and
            (int(observed.st_dev), int(observed.st_ino)) == stage_identity,
            "private stage identity changed; cleanup refused")
    stage_fd = os.open(stage_name, _fd_flags(), dir_fd=results_fd)
    try:
        names = set(os.listdir(stage_fd))
        require(names.issubset(set(ATTEMPT_MEMBERS)),
                "private stage contains unknown members; cleanup refused")
        for name in ATTEMPT_MEMBERS:
            if name in names:
                os.unlink(name, dir_fd=stage_fd)
        os.fsync(stage_fd)
    finally:
        os.close(stage_fd)
    os.rmdir(stage_name, dir_fd=results_fd)
    os.fsync(results_fd)


def _after_first_scan_hook() -> None:
    return None


def _before_stage_mkdir_hook() -> None:
    return None


def _after_stage_mkdir_hook() -> None:
    return None


def _before_attempt_publish_hook() -> None:
    return None


def atomic_guard_and_consume(results_directory: Path,
                             prefixes: Sequence[str], stage_name: str,
                             attempt_name: str,
                             receipt: Mapping[str, object]) -> Dict[str, object]:
    """Inspect and consume relative to one verified results FD."""
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
    parent = directory.parent
    results_name = directory.name
    _safe_basename(results_name, "results directory")
    parent_fd = os.open(str(parent), _fd_flags())
    results_fd = -1
    stage_created = False
    published = False
    stage_identity = None
    try:
        initial = _lstat_at(parent_fd, results_name)
        require(initial is not None and stat.S_ISDIR(initial.st_mode) and
                not stat.S_ISLNK(initial.st_mode),
                "results pathname absent, symlinked, or non-directory")
        results_fd = os.open(results_name, _fd_flags(), dir_fd=parent_fd)
        opened = os.fstat(results_fd)
        identity = (int(opened.st_dev), int(opened.st_ino))
        require((int(initial.st_dev), int(initial.st_ino)) == identity,
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
        stage_identity = (int(created.st_dev), int(created.st_ino))
        _after_stage_mkdir_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, False)
        stage_fd = os.open(stage_name, _fd_flags(), dir_fd=results_fd)
        try:
            stage_opened = os.fstat(stage_fd)
            require((int(stage_opened.st_dev), int(stage_opened.st_ino)) ==
                    stage_identity, "attempt stage raced during open")
            payload = (json.dumps(receipt, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n").encode("utf-8")
            _write_exclusive_at(stage_fd, "attempt.json", payload)
            manifest = (hashlib.sha256(payload).hexdigest() +
                        "  attempt.json\n").encode("ascii")
            _write_exclusive_at(stage_fd, "SHA256SUMS", manifest)
            outer = (hashlib.sha256(manifest).hexdigest() +
                     "  SHA256SUMS\n").encode("ascii")
            _write_exclusive_at(stage_fd, "SHA256SUMS.seal.sha256", outer)
            require(set(os.listdir(stage_fd)) == set(ATTEMPT_MEMBERS),
                    "attempt stage population drift")
            require(_read_at(stage_fd, "attempt.json") == payload and
                    _read_at(stage_fd, "SHA256SUMS") == manifest and
                    _read_at(stage_fd, "SHA256SUMS.seal.sha256") == outer,
                    "attempt stage seal readback drift")
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        _before_attempt_publish_hook()
        _assert_results_binding(parent_fd, results_name, identity)
        _require_stable_absence(results_fd, prefixes, False)
        require(_lstat_at(results_fd, attempt_name) is None,
                "canonical attempt collision before publication")
        current_stage = _lstat_at(results_fd, stage_name)
        require(current_stage is not None and stat.S_ISDIR(current_stage.st_mode)
                and (int(current_stage.st_dev), int(current_stage.st_ino)) ==
                stage_identity, "attempt stage identity drift")
        _renameat2_noreplace(results_fd, stage_name, attempt_name)
        published = True
        os.fsync(results_fd)
        require(_lstat_at(results_fd, stage_name) is None,
                "attempt stage survived no-replace publication")
        final = _lstat_at(results_fd, attempt_name)
        require(final is not None and stat.S_ISDIR(final.st_mode) and
                (int(final.st_dev), int(final.st_ino)) == stage_identity,
                "published attempt identity drift")
        return {
            "status": "PASS_M832_DIRECTORY_FD_BOUND_ATTEMPT_CONSUMED",
            "results_dev": identity[0],
            "results_ino": identity[1],
            "attempt_dev": stage_identity[0],
            "attempt_ino": stage_identity[1],
            "attempt_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "attempt_outer_seal_file_sha256":
                hashlib.sha256(outer).hexdigest(),
            "guarded_prefixes": list(prefixes),
            "production_cycles": None,
        }
    except (Failure, OSError, ValueError) as error:
        if (stage_created and not published and results_fd >= 0 and
                stage_identity is not None):
            try:
                _cleanup_private_stage(results_fd, stage_name, stage_identity)
            except Exception as cleanup_error:
                raise Failure(str(error) +
                              "; private stage cleanup failed closed: " +
                              str(cleanup_error)) from cleanup_error
        if isinstance(error, Failure):
            raise
        raise Failure(str(error)) from error
    finally:
        if results_fd >= 0:
            os.close(results_fd)
        os.close(parent_fd)


def _verify_m831_negative() -> Dict[str, object]:
    identity = verify_sealed(M831_REVIEW)
    regular_exact(M831_REVIEW / "review.json", M831_REVIEW_JSON_SHA256,
                  "M831 negative review")
    require(identity["manifest_sha256"] == M831_MANIFEST_SHA256 and
            identity["outer_seal_file_sha256"] ==
            M831_OUTER_SEAL_FILE_SHA256, "M831 seal drift")
    review = strict_json(M831_REVIEW / "review.json")
    require(review.get("status") ==
            "NO_GO_M828_SOURCE_CANDIDATE__P1_1__DIRECTORY_BINDING_TOCTOU_REPAIR_REQUIRED" and
            review.get("true_release_authorized") is False and
            review.get("production_launch_authorized") is False,
            "M831 negative authority weakened")
    return review


def validate_candidate(candidate_path: Path, require_future_absent: bool = True,
                       attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict) and
            candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("status") ==
            "SOURCE_ONLY_M832_DIRECTORY_BOUND_CONSUMPTION_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "M832 candidate identity drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate authorizes production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "temporary_guard_and_consumption_attacks": True,
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
    required = {"driver", "runner", "tests", "consumption_contract",
                "m828_parent_driver", "m828_parent_candidate",
                "m819_parent_driver", "m809_parent_driver",
                "m785_contract", "analyzer", "storage_oracle",
                "m785_tests", "m831_no_go_review"}
    require(set(paths) == required, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate driver path drift")
    require(paths["runner"].resolve() == HERE /
            "run_m832_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() == HERE.parent /
            "tests/test_m832_m828_decoder_directory_bound_consumption.py",
            "candidate tests path drift")
    require(paths["m828_parent_driver"].resolve() == M828_PATH and
            candidate["source_identity"]["m828_parent_driver"]["sha256"] ==
            M828_SHA256, "M828 parent driver drift")
    require(paths["m828_parent_candidate"].resolve() == M828_CANDIDATE and
            candidate["source_identity"]["m828_parent_candidate"]["sha256"] ==
            M828_CANDIDATE_SHA256, "M828 parent candidate drift")
    _verify_m831_negative()
    require(candidate.get("m831_no_go_basis") == {
        "directory":
            "reviews/m831_m828_m785_decoder_failure_prefix_guard_source_fresh_hammer_r1_20260829",
        "review_json_sha256": M831_REVIEW_JSON_SHA256,
        "manifest_sha256": M831_MANIFEST_SHA256,
        "outer_seal_file_sha256": M831_OUTER_SEAL_FILE_SHA256,
        "status":
            "NO_GO_M828_SOURCE_CANDIDATE__P1_1__DIRECTORY_BINDING_TOCTOU_REPAIR_REQUIRED",
        "m828_true_release_authorized": False,
        "additive_directory_bound_consumption_required": True,
    }, "M831 basis drift")
    parent_value = M828.validate_candidate(M828_CANDIDATE, True, False)
    require(parent_value["status"] ==
            "PASS_M828_FAILURE_PREFIX_GUARD_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "M828 parent validation drift")
    parent = strict_json(M828_CANDIDATE)
    require(candidate["common_resource"] == parent["common_resource"] and
            candidate["production_semantics"] == parent["production_semantics"],
            "frozen runtime semantics drift")
    require(candidate["directory_bound_consumption"] == {
        "results_directory": "hw_autoresearch_nts07/results",
        "basename_prefixes": list(GUARDED_PREFIXES),
        "parent_and_results_nofollow": True,
        "path_binding": "PARENT_DIRFD_RELATIVE_DEV_INO",
        "stage_create": "MKDIRAT_SAME_RESULTS_FD",
        "receipt_write": "OPENAT_EXCLUSIVE_SAME_STAGE_FD",
        "attempt_publish": "RENAMEAT2_NOREPLACE_SAME_RESULTS_FD",
        "transient_history_claim": False,
        "persistent_match_rejected": True,
        "wrong_prefix_is_not_a_match": True,
    }, "directory-bound consumption drift")
    require(candidate["attempt_compatibility"] == {
        "formal_status": PARENT_ATTEMPT_STATUS,
        "accepted_by_m832_validator": True,
        "accepted_by_frozen_m828_m819_m809": True,
        "outer_schema": "m832_m785_decoder_production_attempt_v1",
    }, "attempt compatibility drift")
    require(candidate["canonical"] == {
        "result": "hw_autoresearch_nts07/results/m832_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt": "hw_autoresearch_nts07/results/.m832_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release": "hw_autoresearch_nts07/contracts/m832_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix": "hw_autoresearch_nts07/results/" +
            CANONICAL_FAILURE_PREFIX,
    }, "canonical path drift")
    require(candidate.get("future_release_required_fields") == {
        "schema": RELEASE_SCHEMA,
        "status":
            "TRUE_RELEASE_AFTER_FRESH_M832_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY",
        "launch_now": True,
        "release": True,
        "max_attempts": 1,
        "fresh_source_hammer_directory": SOURCE_HAMMER_DIR,
        "fresh_source_hammer_status":
            "PASS100_M832_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY",
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
    return {"status":
            "PASS_M832_DIRECTORY_BOUND_CONSUMPTION_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "candidate_sha256": sha256(candidate_path),
            "parent_status": parent_value["status"],
            "production_cycles": None}


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool):
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    cv = validate_candidate(candidate_path, False, attempt_required)
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    require(isinstance(release, dict) and release.get("schema") ==
            RELEASE_SCHEMA and release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_M832_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release.get("launch_now") is True and release.get("release") is True
            and release.get("max_attempts") == 1, "release identity drift")
    result, attempt, future = _canonical_paths(candidate)
    require(release_path == future.resolve(), "release path drift")
    binding = release["candidate_binding"]
    require((HW / binding["path"]).resolve() == candidate_path and
            binding["sha256"] == cv["candidate_sha256"],
            "release candidate binding drift")
    require(release["source_identity"] == candidate["source_identity"] and
            release["canonical"] == candidate["canonical"] and
            release["directory_bound_consumption"] ==
            candidate["directory_bound_consumption"],
            "release source/canonical/consumption drift")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR, "hammer path drift")
    hammer_dir = HW / hammer["directory"]
    identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M835 source hammer")
    require(identity["manifest_sha256"] == hammer["manifest_sha256"] and
            identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"], "hammer seal drift")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M832_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0}
            and review.get("true_release_authorized") is True and
            review.get("production_launch_authorized") is False,
            "source hammer authorization drift")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule": "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M828_M819_M809_EXACT_SHA",
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }, "release runtime semantics drift")
    expected = os.environ.get("M832_EXPECTED_RELEASE_SHA256", "")
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
    regular_exact(runner_path, expected_runner_sha256, "M832 runner")
    require(candidate["source_identity"]["runner"]["sha256"] ==
            expected_runner_sha256, "runner binding drift")
    result, attempt, future = _canonical_paths(candidate)
    require(attempt.parent.resolve() == (HW / "results").resolve(),
            "attempt results directory drift")
    receipt = {
        "schema": "m832_m785_decoder_production_attempt_v1",
        "status": PARENT_ATTEMPT_STATUS,
        "outer_boundary": "M832_DIRECTORY_FD_BOUND_CONSUMPTION",
        "runner_sha256": expected_runner_sha256,
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "candidate_sha256": sha256(candidate_path),
        "release_sha256": sha256(release_path),
        "canonical_result": str(result),
        "max_attempts": 1,
        "claim_boundary": {
            "cycles_before_result_hammer": False,
            "speedup_before_result_hammer": False,
            "decoder_complete": False,
            "full_network_completion": False,
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
            "m832_m785_decoder_production_attempt_v1" and
            receipt.get("status") == PARENT_ATTEMPT_STATUS and
            receipt.get("max_attempts") == 1 and
            receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            receipt.get("runner_sha256") ==
            candidate["source_identity"]["runner"]["sha256"] and
            receipt.get("release_sha256") == sha256(release_path),
            "consumed attempt identity drift")
    return {"status":
            "PASS_M832_DIRECTORY_BOUND_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
            "attempt_manifest_sha256": identity["manifest_sha256"],
            "attempt_outer_seal_file_sha256":
                identity["outer_seal_file_sha256"],
            "scheduled_rows": 0, "production_cycles": None}


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, True)
    original = M828.validate_true_release
    M828.validate_true_release = lambda *args, **kwargs: gate
    try:
        result = M828.run_production(release_path, candidate_path,
                                     attempt_path, output)
    finally:
        M828.validate_true_release = original
    result = copy.deepcopy(result)
    result["schema"] = (
        "m832_m785_decoder_physical_residency_production_result_v1")
    result["delegated_directory_bound_consumption"] = {
        "driver_path": str(Path(__file__).resolve().relative_to(HW)),
        "parent_driver_path": str(M828_PATH.relative_to(HW)),
        "parent_driver_sha256": M828_SHA256,
        "semantics_changed": False,
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }
    M828.M819.M809._write_json(Path(output) / "result.m832.json", result)
    (Path(output) / "result.json").unlink()
    (Path(output) / "result.m832.json").rename(Path(output) / "result.json")
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
        M828.M819.M809._rename_noreplace(stage, destination)
    except M828.M819.M809.Failure as error:
        raise Failure(str(error)) from error
    require(verify_sealed(destination) == identity and not stage.exists(),
            "publication transition drift")
    return {"status": "PASS_M832_ATOMIC_NOREPLACE_PUBLICATION", **identity}


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
        "schema": "m832_m785_decoder_production_failure_receipt_v1",
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
        value = M828._write_failure_receipt(
            output, stdout_path, stderr_path, payload)
    except M828.Failure as error:
        raise Failure(str(error)) from error
    return {"status": "PASS_M832_SEALED_FAILURE_RECEIPT", **value}


def preproduction_traversal_test() -> Dict[str, object]:
    candidate = strict_json(CANDIDATE)
    with tempfile.TemporaryDirectory(prefix="m832_preproduction_") as directory:
        root = Path(directory)
        results = root / "results"
        results.mkdir()
        candidate_path = root / "candidate.json"
        release_path = root / "release.json"
        output = results / "result.stage.test"
        M828.M819.M809._write_json(candidate_path, {"source_test": True})
        M828.M819.M809._write_json(release_path, {"source_test": True})
        fake = copy.deepcopy(candidate)
        fake["canonical"]["attempt"] = str(results / "attempt")
        fake["canonical"]["result"] = str(results / "result")
        receipt = {
            "schema": "m832_m785_decoder_production_attempt_v1",
            "status": PARENT_ATTEMPT_STATUS,
            "candidate_sha256": sha256(candidate_path),
            "release_sha256": sha256(release_path),
        }
        consumed = atomic_guard_and_consume(
            results, GUARDED_PREFIXES, "attempt.stage.source-test",
            "attempt", receipt)
        attempt_path = results / "attempt"
        attempt_before = sha256(attempt_path / "attempt.json")
        original_local_validate = globals()["validate_true_release"]
        original_m828_validate = M828.validate_true_release
        original_m819_validate = M828.M819.validate_true_release
        original_m809_validate = M828.M819.M809.validate_true_release
        original_path = M828.M819.M809.Path
        base_path = type(Path())
        target = os.path.abspath(str(output))

        class GuardedPath(base_path):
            def mkdir(self, *args, **kwargs):
                if os.path.abspath(str(self)) == target:
                    raise ControlledPreproductionStop("AT_M809_OUTPUT_MKDIR")
                return super().mkdir(*args, **kwargs)

        globals()["validate_true_release"] = lambda *args, **kwargs: {
            "candidate": fake}
        M828.M819.M809.Path = GuardedPath
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
            M828.M819.M809.Path = original_path
            globals()["validate_true_release"] = original_local_validate
        require(M828.validate_true_release is original_m828_validate and
                M828.M819.validate_true_release is original_m819_validate and
                M828.M819.M809.validate_true_release is original_m809_validate,
                "delegated validators not restored")
        require(stopped and not output.exists() and
                sha256(attempt_path / "attempt.json") == attempt_before,
                "M832 traversal crossed zero-row boundary")
        return {
            "status":
                "PASS_M832_DIRECTORY_BOUND_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL",
            "consume_status": consumed["status"],
            "entered_exact_frozen_m828": True,
            "entered_exact_frozen_m819": True,
            "entered_exact_frozen_m809": True,
            "parent_attempt_status_accepted": True,
            "stopped_at": "M809_OUTPUT_MKDIR",
            "scheduled_rows": 0,
            "output_exists": False,
            "attempt_receipt_identity_drift": False,
            "delegated_validators_restored": True,
            "production_cycles": None,
        }


def self_test() -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m832_consume_selftest_") as directory:
        root = Path(directory)
        wrong = root / ("x" + CANONICAL_FAILURE_PREFIX + "wrong")
        wrong.write_text("unrelated", encoding="utf-8")
        value = atomic_guard_and_consume(
            root, GUARDED_PREFIXES, "attempt.stage.selftest", "attempt",
            {"schema": "source_test", "status": PARENT_ATTEMPT_STATUS})
        require(wrong.read_text(encoding="utf-8") == "unrelated",
                "wrong prefix clobbered")
    traversal = preproduction_traversal_test()
    return {
        "status": "PASS_M832_DIRECTORY_BOUND_CONSUMPTION_SYNTHETIC_SELF_TEST",
        "consume": value["status"],
        "traversal": traversal["status"],
        "scheduled_rows": 0,
        "formal_attempt_created": False,
        "production_cycles": None,
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
        print(json.dumps({"status":
                         "PASS_M832_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
                         "candidate_sha256": value["candidate_validation"][
                             "candidate_sha256"],
                         "release_sha256": sha256(args.release),
                         "production_cycles": None}, sort_keys=True,
                         allow_nan=False))
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
