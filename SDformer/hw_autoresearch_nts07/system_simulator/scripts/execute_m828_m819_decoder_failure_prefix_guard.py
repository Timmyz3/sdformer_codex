#!/usr/bin/env python3
"""M828 additive failure-prefix guard around the frozen M819 decoder replay."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M819_PATH = HERE / "execute_m819_m809_decoder_production_delegation_compat.py"
M819_SHA256 = "7832fac849481e0f05417a4eebce489f131ce4c14554679c0b0343e9b1261d54"
M819_CANDIDATE = HW / "contracts/m819_m785_decoder_production_delegation_compat_candidate_r1_20260829.json"
M819_CANDIDATE_SHA256 = "71e61d14a54ed2250aefd50e3239968754326b8aa3fed76371d5634064b233c2"
M825_REVIEW = HW / "reviews/m825_m824_m819_decoder_production_final_release_hammer_r1_20260829"
M825_REVIEW_JSON_SHA256 = "c8a18940ce150052cb19798c26a3498e33e4059ecf87afb65f60094a4305bf41"
M825_MANIFEST_SHA256 = "f7089be8df13ca1812848d971e9cec098c95ff783244e82e05a1a36bd1e5511f"
M825_OUTER_SEAL_FILE_SHA256 = "2f9be8077084c5f3292ef659e3695295152416dad5ab69eaef613fa352721c57"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PARENT_ATTEMPT_STATUS = "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY"
CANDIDATE_SCHEMA = "m828_m785_decoder_failure_prefix_guard_candidate_v1"
RELEASE_SCHEMA = "m828_m785_decoder_production_true_release_v1"
CANDIDATE = HW / "contracts/m828_m785_decoder_failure_prefix_guard_candidate_r1_20260829.json"
SOURCE_HAMMER_DIR = "reviews/m831_m828_m785_decoder_failure_prefix_guard_source_fresh_hammer_r1_20260829"
CANONICAL_FAILURE_PREFIX = (
    "m828_m785_h67_decoder_physical_residency_cycles_r1_20260829"
    ".failed_or_incomplete.")
INHERITED_FAILURE_PREFIX = (
    "m785_h67_decoder_physical_residency_production_r1_20260829"
    ".failed_or_incomplete.")
GUARDED_PREFIXES = (CANONICAL_FAILURE_PREFIX, INHERITED_FAILURE_PREFIX)
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")


class Failure(RuntimeError):
    pass


class ControlledPreproductionStop(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def _load_exact(path: Path, expected: str, module_name: str):
    require(path.is_file() and not path.is_symlink(), module_name + " absent")
    import hashlib
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            module_name + " SHA drift")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None,
            "cannot import " + module_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M819 = _load_exact(M819_PATH, M819_SHA256, "m828_frozen_m819")


def sha256(path: Path) -> str:
    return M819.sha256(Path(path))


def strict_json(path: Path) -> object:
    try:
        return M819.strict_json(Path(path))
    except M819.Failure as error:
        raise Failure(str(error)) from error


def verify_sealed(directory: Path) -> Dict[str, str]:
    try:
        return M819.verify_sealed(Path(directory))
    except M819.Failure as error:
        raise Failure(str(error)) from error


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        M819.regular_exact(Path(path), expected, label)
    except M819.Failure as error:
        raise Failure(str(error)) from error


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {name: HW / entry["path"]
            for name, entry in candidate["source_identity"].items()}


def _canonical_paths(candidate: Mapping[str, object]):
    canonical = candidate["canonical"]
    return (REPO / canonical["result"], REPO / canonical["attempt"],
            REPO / canonical["future_release"])


def _directory_token(value) -> Tuple[int, int, int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode),
            int(value.st_mtime_ns), int(value.st_ctime_ns))


def _classify_mode(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "REGULAR"
    if stat.S_ISDIR(mode):
        return "DIRECTORY"
    if stat.S_ISLNK(mode):
        return "SYMLINK"
    return "OTHER"


def _guard_sample_yield() -> None:
    os.sched_yield()


def _sample_prefixes(fd: int, prefixes: Sequence[str]):
    before = _directory_token(os.fstat(fd))
    names = tuple(sorted(os.listdir(fd)))
    matches = []
    for name in names:
        if any(name.startswith(prefix) for prefix in prefixes):
            try:
                observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
                kind = _classify_mode(observed.st_mode)
                if kind == "SYMLINK":
                    try:
                        os.stat(name, dir_fd=fd, follow_symlinks=True)
                    except FileNotFoundError:
                        kind = "DANGLING_SYMLINK"
            except FileNotFoundError:
                kind = "RACED_AWAY"
            matches.append((name, kind))
    after = _directory_token(os.fstat(fd))
    return {"before": before, "after": after,
            "matches": tuple(matches)}


def guard_failure_prefix_absence(results_directory: Path,
                                 prefixes: Sequence[str]) -> Dict[str, object]:
    """Double-sample one pinned directory FD and reject every prefix type."""
    directory = Path(results_directory)
    prefixes = tuple(prefixes)
    require(prefixes and len(set(prefixes)) == len(prefixes),
            "guard prefix set malformed")
    for prefix in prefixes:
        require(isinstance(prefix, str) and prefix and "/" not in prefix and
                prefix not in (".", ".."), "guard prefix malformed")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(directory), flags)
    except OSError as error:
        raise Failure("results directory is absent, symlinked, or non-directory") from error
    try:
        identity = _directory_token(os.fstat(fd))[:2]
        for _ in range(3):
            first = _sample_prefixes(fd, prefixes)
            if first["matches"]:
                raise Failure("preexisting failure-prefix artifact: " +
                              repr(first["matches"]))
            _guard_sample_yield()
            second = _sample_prefixes(fd, prefixes)
            if second["matches"]:
                raise Failure("concurrent failure-prefix artifact: " +
                              repr(second["matches"]))
            stable = (first["before"] == first["after"] ==
                      second["before"] == second["after"] and
                      first["matches"] == second["matches"] and
                      _directory_token(os.fstat(fd))[:2] == identity)
            if stable:
                return {
                    "status": "PASS_M828_STABLE_FAILURE_PREFIX_ABSENCE",
                    "directory_dev": identity[0],
                    "directory_ino": identity[1],
                    "samples": 2,
                    "guarded_prefixes": list(prefixes),
                    "matches": [],
                    "formal_attempt_created": False,
                    "production_cycles": None,
                }
        raise Failure("results directory enumeration was unstable")
    finally:
        os.close(fd)


def guard_from_candidate(candidate_path: Path) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    require(candidate_path == CANDIDATE.resolve(), "guard candidate path drift")
    candidate = strict_json(candidate_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "guard candidate schema drift")
    guard = candidate.get("failure_prefix_guard", {})
    require(guard == {
        "results_directory": "hw_autoresearch_nts07/results",
        "basename_prefixes": list(GUARDED_PREFIXES),
        "match_rule": "BYTE_EXACT_BASENAME_STARTSWITH",
        "samples": 2,
        "directory_fd_nofollow": True,
        "reject_any_matched_type": True,
        "wrong_prefix_is_not_a_match": True,
    }, "guard contract drift")
    results = REPO / guard["results_directory"]
    require(results.resolve() == (HW / "results").resolve(),
            "guard results path drift")
    return guard_failure_prefix_absence(results, guard["basename_prefixes"])


def _verify_m825_negative() -> Dict[str, object]:
    identity = verify_sealed(M825_REVIEW)
    regular_exact(M825_REVIEW / "review.json", M825_REVIEW_JSON_SHA256,
                  "M825 negative review")
    require(identity["manifest_sha256"] == M825_MANIFEST_SHA256 and
            identity["outer_seal_file_sha256"] ==
            M825_OUTER_SEAL_FILE_SHA256, "M825 seal drift")
    review = strict_json(M825_REVIEW / "review.json")
    require(review.get("status") ==
            "NO_GO_M819_TRUE_RELEASE__P1_1__ADDITIVE_FAILURE_PREFIX_PRECONSUMPTION_GATE_REQUIRED" and
            review.get("production_replay_authorized") is False and
            review.get("formal_runner_invocation_authorized") is False,
            "M825 negative authority weakened")
    return review


def validate_candidate(candidate_path: Path, require_future_absent: bool = True,
                       attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict) and
            candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("status") ==
            "SOURCE_ONLY_M828_FAILURE_PREFIX_GUARD_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "M828 candidate identity drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate authorizes production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "temporary_guard_attacks": True,
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
    require(set(paths) == {
        "driver", "runner", "tests", "guard_contract",
        "m819_parent_driver", "m819_parent_candidate",
        "m809_parent_driver", "m809_parent_candidate",
        "m793_parent_driver", "m793_parent_candidate", "m785_contract",
        "analyzer", "storage_oracle", "m785_tests", "m825_no_go_review",
    }, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate driver path drift")
    require(paths["runner"].resolve() ==
            HERE / "run_m828_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() == HERE.parent /
            "tests/test_m828_m819_decoder_failure_prefix_guard.py",
            "candidate tests path drift")
    require(paths["m819_parent_driver"].resolve() == M819_PATH and
            candidate["source_identity"]["m819_parent_driver"]["sha256"] ==
            M819_SHA256, "M819 parent driver drift")
    require(paths["m819_parent_candidate"].resolve() == M819_CANDIDATE and
            candidate["source_identity"]["m819_parent_candidate"]["sha256"] ==
            M819_CANDIDATE_SHA256, "M819 parent candidate drift")
    require(paths["m825_no_go_review"].resolve() == M825_REVIEW /
            "review.json", "M825 path drift")
    _verify_m825_negative()
    require(candidate["m825_no_go_basis"] == {
        "directory":
            "reviews/m825_m824_m819_decoder_production_final_release_hammer_r1_20260829",
        "review_json_sha256": M825_REVIEW_JSON_SHA256,
        "manifest_sha256": M825_MANIFEST_SHA256,
        "outer_seal_file_sha256": M825_OUTER_SEAL_FILE_SHA256,
        "status":
            "NO_GO_M819_TRUE_RELEASE__P1_1__ADDITIVE_FAILURE_PREFIX_PRECONSUMPTION_GATE_REQUIRED",
        "m819_launch_authorized": False,
        "additive_guard_required": True,
    }, "M825 basis drift")
    parent_value = M819.validate_candidate(M819_CANDIDATE, False, False)
    require(parent_value["status"] ==
            "PASS_M819_DELEGATION_COMPAT_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "M819 parent validation drift")
    parent = strict_json(M819_CANDIDATE)
    expected_frozen = {
        "m809_parent_driver": (
            "system_simulator/scripts/execute_m809_m785_decoder_physical_residency_production.py",
            "2b273d5fb3f68ae7cae16c458b8138ab89c83a677ce9aa8f15b90de2fd6736d0"),
        "m809_parent_candidate": (
            "contracts/m809_m785_decoder_physical_residency_production_recovery_candidate_r1_20260829.json",
            "9742335fb312af4e3d8805c43d44ae5470e0af32a2a1c9026cdb2ef4b0b3e635"),
        "m793_parent_driver": (
            "system_simulator/scripts/execute_m793_m785_decoder_physical_residency_production.py",
            "c868eb5569d856f75d08e01c78d896eee774502df27a07301dc681c65410dd77"),
        "m793_parent_candidate": (
            "contracts/m793_m785_decoder_physical_residency_production_release_candidate_r1_20260828.json",
            "2dc6b6dc0b110f124446cbd8c3b4b10d5e395f32d384be20098c1d8b6b7634aa"),
        "m785_contract": (
            "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json",
            "612a2ba39ceecedc351f2f6550347ad50ca9526fd89ed143bc6362c3e5681810"),
        "analyzer": (
            "system_simulator/scripts/analyze_m785_h67_decoder_physical_residency_repair.py",
            "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1"),
        "storage_oracle": (
            "system_simulator/scripts/oracle_m785_decoder_global_vector_storage.py",
            "422da36ad1414d2dfa70363607c27bb99dee2f2505d1ceee2142a6023c162db5"),
        "m785_tests": (
            "system_simulator/tests/test_m785_h67_decoder_physical_residency_repair.py",
            "1ec8730cde5f91a91af269fb54969c5c6762fe5cb8bc36ba4b36117ce21c6787"),
    }
    for name, expected in expected_frozen.items():
        entry = candidate["source_identity"][name]
        require((entry["path"], entry["sha256"]) == expected,
                name + " frozen identity drift")
    guard_contract = strict_json(paths["guard_contract"])
    require(guard_contract.get("schema") ==
            "m828_m819_decoder_failure_prefix_guard_contract_v1" and
            guard_contract.get("status") ==
            "SOURCE_ONLY_ADDITIVE_FAILURE_PREFIX_GUARD__NO_RELEASE_OR_PRODUCTION" and
            guard_contract["guard"]["basename_prefixes"] ==
            list(GUARDED_PREFIXES), "guard contract identity drift")
    require(candidate["common_resource"] == parent["common_resource"] and
            candidate["production_semantics"] == parent["production_semantics"],
            "frozen runtime semantics drift")
    require(candidate["attempt_compatibility"] == {
        "formal_status": PARENT_ATTEMPT_STATUS,
        "accepted_by_m828_validator": True,
        "accepted_by_frozen_m819_delegate": True,
        "accepted_by_frozen_m809_body": True,
        "outer_schema": "m828_m785_decoder_production_attempt_v1",
    }, "attempt compatibility drift")
    require(candidate["failure_prefix_guard"] == {
        "results_directory": "hw_autoresearch_nts07/results",
        "basename_prefixes": list(GUARDED_PREFIXES),
        "match_rule": "BYTE_EXACT_BASENAME_STARTSWITH",
        "samples": 2,
        "directory_fd_nofollow": True,
        "reject_any_matched_type": True,
        "wrong_prefix_is_not_a_match": True,
    }, "failure-prefix guard drift")
    require(candidate["canonical"] == {
        "result": "hw_autoresearch_nts07/results/m828_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt": "hw_autoresearch_nts07/results/.m828_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release": "hw_autoresearch_nts07/contracts/m828_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix": "hw_autoresearch_nts07/results/" +
            CANONICAL_FAILURE_PREFIX,
    }, "canonical path drift")
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
            "PASS_M828_FAILURE_PREFIX_GUARD_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
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
            "TRUE_RELEASE_AFTER_FRESH_M828_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
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
            release["failure_prefix_guard"] ==
            candidate["failure_prefix_guard"],
            "release source/canonical/guard drift")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR, "hammer path drift")
    hammer_dir = HW / hammer["directory"]
    identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M831 source hammer")
    require(identity["manifest_sha256"] == hammer["manifest_sha256"] and
            identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"], "hammer seal drift")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M828_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0} and
            review.get("true_release_authorized") is True and
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
        "delegated_schedule_body": "FROZEN_M819_THIN_BOUNDARY_AND_M809_EXACT_SHA",
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }, "release runtime semantics drift")
    require(release["claim_boundary"] == {
        "production_replay": False, "production_cycles": False,
        "production_speedup": False, "energy": False, "ppa": False,
        "decoder_complete": False, "full_network_completion": False,
        "table_a_insertion_allowed": False, "system_speedup": False,
        "paper_claim": False, "rtl_vcs_eda_energy_ppa": False,
        "docs359_modified": False,
    }, "release claim boundary drift")
    expected = os.environ.get("M828_EXPECTED_RELEASE_SHA256", "")
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
            "m828_m785_decoder_production_attempt_v1" and
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
            "PASS_M828_PARENT_COMPAT_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
            "attempt_manifest_sha256": identity["manifest_sha256"],
            "attempt_outer_seal_file_sha256":
                identity["outer_seal_file_sha256"],
            "scheduled_rows": 0, "production_cycles": None}


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, True)
    original = M819.validate_true_release
    M819.validate_true_release = lambda *args, **kwargs: gate
    try:
        result = M819.run_production(release_path, candidate_path,
                                     attempt_path, output)
    finally:
        M819.validate_true_release = original
    result = copy.deepcopy(result)
    result["schema"] = (
        "m828_m785_decoder_physical_residency_production_result_v1")
    result["delegated_failure_prefix_guard"] = {
        "driver_path": str(Path(__file__).resolve().relative_to(HW)),
        "parent_driver_path": str(M819_PATH.relative_to(HW)),
        "parent_driver_sha256": M819_SHA256,
        "semantics_changed": False,
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }
    M819.M809._write_json(Path(output) / "result.m828.json", result)
    (Path(output) / "result.json").unlink()
    (Path(output) / "result.m828.json").rename(Path(output) / "result.json")
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
        M819.M809._rename_noreplace(stage, destination)
    except M819.M809.Failure as error:
        raise Failure(str(error)) from error
    require(verify_sealed(destination) == identity and not stage.exists(),
            "publication transition drift")
    return {"status": "PASS_M828_ATOMIC_NOREPLACE_PUBLICATION", **identity}


def write_flat_attempt(directory: Path, receipt: Mapping[str, object]):
    try:
        return M819.write_flat_attempt(directory, receipt)
    except M819.Failure as error:
        raise Failure(str(error)) from error


def _write_failure_receipt(output: Path, stdout_path: Path,
                           stderr_path: Path, payload: Mapping[str, object]):
    try:
        return M819._write_failure_receipt(output, stdout_path, stderr_path,
                                           payload)
    except M819.Failure as error:
        raise Failure(str(error)) from error


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
    require(candidate["source_identity"]["runner"]["sha256"] ==
            expected_runner_sha256, "failure runner drift")
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
        "schema": "m828_m785_decoder_production_failure_receipt_v1",
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
        "scheduled_rows": 0 if phase ==
            "ATTEMPT_PUBLISHED_POSTCHECK" else None,
        "claim_boundary": {"production_complete": False,
                           "cycles_citable": False,
                           "speedup_citable": False,
                           "decoder_complete": False,
                           "table_a_insertion_allowed": False},
    }
    value = _write_failure_receipt(output, stdout_path, stderr_path, payload)
    return {"status": "PASS_M828_SEALED_FAILURE_RECEIPT", **value}


def preproduction_traversal_test() -> Dict[str, object]:
    candidate = strict_json(CANDIDATE)
    with tempfile.TemporaryDirectory(prefix="m828_preproduction_") as directory:
        root = Path(directory)
        guard = guard_failure_prefix_absence(root, GUARDED_PREFIXES)
        candidate_path = root / "candidate.json"
        release_path = root / "release.json"
        attempt_path = root / "attempt"
        output = root / "result.stage.test"
        M819.M809._write_json(candidate_path, {"source_test": True})
        M819.M809._write_json(release_path, {"source_test": True})
        fake = copy.deepcopy(candidate)
        fake["canonical"]["attempt"] = str(attempt_path)
        fake["canonical"]["result"] = str(root / "result")
        write_flat_attempt(attempt_path, {
            "schema": "m828_m785_decoder_production_attempt_v1",
            "status": PARENT_ATTEMPT_STATUS,
            "candidate_sha256": sha256(candidate_path),
            "release_sha256": sha256(release_path),
        })
        attempt_before = sha256(attempt_path / "attempt.json")
        original_local_validate = validate_true_release
        original_m819_validate = M819.validate_true_release
        original_m809_validate = M819.M809.validate_true_release
        original_path = M819.M809.Path
        base_path = type(Path())
        target = os.path.abspath(str(output))

        class GuardedPath(base_path):
            def mkdir(self, *args, **kwargs):
                if os.path.abspath(str(self)) == target:
                    raise ControlledPreproductionStop("AT_M809_OUTPUT_MKDIR")
                return super().mkdir(*args, **kwargs)

        globals()["validate_true_release"] = lambda *args, **kwargs: {
            "candidate": fake}
        M819.M809.Path = GuardedPath
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
            M819.M809.Path = original_path
            globals()["validate_true_release"] = original_local_validate
        require(M819.validate_true_release is original_m819_validate and
                M819.M809.validate_true_release is original_m809_validate,
                "delegated validators not restored")
        require(stopped and not output.exists() and
                sha256(attempt_path / "attempt.json") == attempt_before,
                "M828 traversal crossed the zero-row boundary")
        return {
            "status": "PASS_M828_GUARD_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL",
            "guard_status": guard["status"],
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
    with tempfile.TemporaryDirectory(prefix="m828_guard_selftest_") as directory:
        root = Path(directory)
        clean = guard_failure_prefix_absence(root, GUARDED_PREFIXES)
        wrong = root / ("x" + CANONICAL_FAILURE_PREFIX + "wrong")
        wrong.write_text("unrelated", encoding="utf-8")
        wrong_prefix = guard_failure_prefix_absence(root, GUARDED_PREFIXES)
    traversal = preproduction_traversal_test()
    return {
        "status": "PASS_M828_FAILURE_PREFIX_GUARD_SYNTHETIC_SELF_TEST",
        "clean": clean["status"],
        "wrong_prefix": wrong_prefix["status"],
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
    parser.add_argument("--guard-failure-prefix-absence", action="store_true")
    parser.add_argument("--validate-release-preflight", action="store_true")
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
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-release-sha256")
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    modes = (args.self_test, args.preproduction_traversal_test,
             args.validate_candidate, args.guard_failure_prefix_absence,
             args.validate_release_preflight, args.validate_consumed_attempt,
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
    if args.guard_failure_prefix_absence:
        print(json.dumps(guard_from_candidate(args.candidate), sort_keys=True,
                         allow_nan=False))
        return 0
    if args.validate_release_preflight:
        require(args.release is not None, "release is required")
        value = validate_true_release(args.release, args.candidate, False)
        print(json.dumps({"status":
                         "PASS_M828_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
                         "candidate_sha256": value["candidate_validation"][
                             "candidate_sha256"],
                         "release_sha256": sha256(args.release),
                         "production_cycles": None}, sort_keys=True,
                         allow_nan=False))
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
