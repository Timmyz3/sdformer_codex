#!/usr/bin/env python3
"""M819 additive delegation-compatible boundary for frozen M809 cycles."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M815_PATH = HERE / "execute_m815_m809_decoder_production_runner_recovery.py"
M815_SHA256 = "1c35b106e48c614f466689b86600e7c227fbed9d92d3037b5b28442d401f163e"
M815_CANDIDATE = HW / "contracts/m815_m785_decoder_production_runner_recovery_candidate_r1_20260829.json"
M815_CANDIDATE_SHA256 = "83b7946d972c794c258c0b45a651bb3bb1c4bbbd6a75b2b580ced30eee600ab8"
M809_PATH = HERE / "execute_m809_m785_decoder_physical_residency_production.py"
M809_SHA256 = "2b273d5fb3f68ae7cae16c458b8138ab89c83a677ce9aa8f15b90de2fd6736d0"
M809_CANDIDATE = HW / "contracts/m809_m785_decoder_physical_residency_production_recovery_candidate_r1_20260829.json"
M809_CANDIDATE_SHA256 = "9742335fb312af4e3d8805c43d44ae5470e0af32a2a1c9026cdb2ef4b0b3e635"
M817_REVIEW = HW / "reviews/m817_m815_m785_decoder_production_source_fresh_hammer_r1_20260829"
M817_REVIEW_JSON_SHA256 = "e00e60bbbc5e98a4261de3bf1111d2ae86d5df89d3ec9aee2045e5daacf28149"
M817_MANIFEST_SHA256 = "2bd2f33cd8040f6dcf8a513d2763cb281e1a5983a26a42a2fee54af97306d3c9"
M817_OUTER_SEAL_FILE_SHA256 = "d4c8d26cb1afcbce7f461928e706ac2ad3fabcecf64e7108d8ff13df8f518a8e"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PARENT_ATTEMPT_STATUS = "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY"
CANDIDATE_SCHEMA = "m819_m785_decoder_production_delegation_compat_candidate_v1"
RELEASE_SCHEMA = "m819_m785_decoder_production_true_release_v1"
CANDIDATE = HW / "contracts/m819_m785_decoder_production_delegation_compat_candidate_r1_20260829.json"
SOURCE_HAMMER_DIR = "reviews/m821_m819_m785_decoder_production_source_fresh_hammer_r1_20260829"
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


M815 = _load_exact(M815_PATH, M815_SHA256, "m819_frozen_m815")
M809 = M815.M809
require(M815.sha256(M809_PATH) == M809_SHA256, "M809 SHA drift")


def sha256(path: Path) -> str:
    return M809.sha256(Path(path))


def strict_json(path: Path) -> object:
    try:
        return M809.strict_json(Path(path))
    except M809.Failure as error:
        raise Failure(str(error)) from error


def verify_sealed(directory: Path) -> Dict[str, str]:
    try:
        return M809.verify_sealed(Path(directory))
    except M809.Failure as error:
        raise Failure(str(error)) from error


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        M809.regular_exact(Path(path), expected, label)
    except M809.Failure as error:
        raise Failure(str(error)) from error


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {name: HW / entry["path"]
            for name, entry in candidate["source_identity"].items()}


def _canonical_paths(candidate: Mapping[str, object]):
    canonical = candidate["canonical"]
    return (REPO / canonical["result"], REPO / canonical["attempt"],
            REPO / canonical["future_release"])


def validate_candidate(candidate_path: Path, require_future_absent: bool = True,
                       attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict) and
            candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("status") ==
            "SOURCE_ONLY_M819_DELEGATION_COMPAT_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "M819 candidate identity drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate authorizes production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "synthetic_self_test": True,
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
        "driver", "runner", "tests", "compatibility_contract",
        "m809_parent_driver", "m809_parent_candidate",
        "m815_parent_driver", "m815_parent_candidate",
        "m817_no_go_review", "m785_contract", "analyzer", "storage_oracle",
    }, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate driver path drift")
    require(paths["runner"].resolve() ==
            HERE / "run_m819_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() == HERE.parent /
            "tests/test_m819_m809_decoder_production_delegation_compat.py",
            "candidate tests path drift")
    require(paths["m809_parent_driver"].resolve() == M809_PATH and
            candidate["source_identity"]["m809_parent_driver"]["sha256"] ==
            M809_SHA256, "M809 parent drift")
    require(paths["m815_parent_driver"].resolve() == M815_PATH and
            candidate["source_identity"]["m815_parent_driver"]["sha256"] ==
            M815_SHA256, "M815 parent drift")
    require(paths["m809_parent_candidate"].resolve() == M809_CANDIDATE and
            candidate["source_identity"]["m809_parent_candidate"]["sha256"] ==
            M809_CANDIDATE_SHA256, "M809 candidate drift")
    require(paths["m815_parent_candidate"].resolve() == M815_CANDIDATE and
            candidate["source_identity"]["m815_parent_candidate"]["sha256"] ==
            M815_CANDIDATE_SHA256, "M815 candidate drift")
    parent = M815.validate_candidate(M815_CANDIDATE)
    require(parent["status"] ==
            "PASS_M815_RUNNER_RECOVERY_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "M815 source validation drift")
    identity = verify_sealed(M817_REVIEW)
    regular_exact(M817_REVIEW / "review.json", M817_REVIEW_JSON_SHA256,
                  "M817 NO-GO review")
    require(identity["manifest_sha256"] == M817_MANIFEST_SHA256 and
            identity["outer_seal_file_sha256"] ==
            M817_OUTER_SEAL_FILE_SHA256, "M817 seal drift")
    m817 = strict_json(M817_REVIEW / "review.json")
    require(m817.get("status") ==
            "NO_GO_M815_TRUE_RELEASE__P1_1__ADDITIVE_DELEGATION_ATTEMPT_STATUS_REPAIR_REQUIRED" and
            m817.get("true_release_authorized") is False and
            m817.get("production_launch_authorized") is False,
            "M817 negative provenance weakened")
    require(candidate["m817_no_go_basis"] == {
        "directory":
            "reviews/m817_m815_m785_decoder_production_source_fresh_hammer_r1_20260829",
        "review_json_sha256": M817_REVIEW_JSON_SHA256,
        "manifest_sha256": M817_MANIFEST_SHA256,
        "outer_seal_file_sha256": M817_OUTER_SEAL_FILE_SHA256,
        "true_release_authorized": False,
        "additive_delegation_repair_required": True,
    }, "candidate M817 basis drift")
    contract = strict_json(paths["compatibility_contract"])
    require(contract.get("schema") ==
            "m819_m809_decoder_delegation_compat_contract_v1" and
            contract.get("status") ==
            "SOURCE_ONLY_ADDITIVE_DELEGATION_COMPAT__NO_RELEASE_OR_PRODUCTION" and
            contract["compatibility_repair"]["formal_attempt_status"] ==
            PARENT_ATTEMPT_STATUS, "compatibility contract drift")
    parent_candidate = strict_json(M815_CANDIDATE)
    require(candidate["common_resource"] == parent_candidate["common_resource"],
            "resource drift")
    expected = dict(parent_candidate["production_semantics"])
    expected.pop("delegated_schedule_body")
    actual = dict(candidate["production_semantics"])
    require(actual.pop("delegated_schedule_body") ==
            "FROZEN_M809_EXACT_SHA", "delegation marker drift")
    require(actual == expected and
            candidate["attempt_compatibility"] == {
                "formal_status": PARENT_ATTEMPT_STATUS,
                "accepted_by_m819_validator": True,
                "accepted_by_frozen_m809_body": True,
                "outer_schema": "m819_m785_decoder_production_attempt_v1",
            }, "schedule or attempt compatibility drift")
    require(candidate["canonical"] == {
        "result": "hw_autoresearch_nts07/results/m819_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt": "hw_autoresearch_nts07/results/.m819_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release": "hw_autoresearch_nts07/contracts/m819_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix": "hw_autoresearch_nts07/results/m819_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_incomplete.",
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
        "d1_headline": False, "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False, "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "claim boundary drift")
    return {"status":
            "PASS_M819_DELEGATION_COMPAT_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "candidate_sha256": sha256(candidate_path),
            "parent_status": parent["status"], "production_cycles": None}


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool):
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    cv = validate_candidate(candidate_path, False, attempt_required)
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    require(isinstance(release, dict) and release.get("schema") ==
            RELEASE_SCHEMA and release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_M819_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
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
            release["canonical"] == candidate["canonical"],
            "release source/canonical drift")
    require(release["reviewed_source_identity"] == {
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256": candidate["source_identity"]["runner"]["sha256"],
    }, "release reviewed-source identity drift")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR, "hammer path drift")
    hammer_dir = HW / hammer["directory"]
    identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M821 source hammer")
    require(identity["manifest_sha256"] == hammer["manifest_sha256"] and
            identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"], "hammer seal drift")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M819_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0} and
            review.get("true_release_authorized") is True and
            review.get("production_launch_authorized") is False,
            "source hammer authorization drift")
    target = review["review_target"]
    require(target["candidate_sha256"] == sha256(candidate_path) and
            target["driver_sha256"] ==
            candidate["source_identity"]["driver"]["sha256"] and
            target["runner_sha256"] ==
            candidate["source_identity"]["runner"]["sha256"],
            "hammer source identity drift")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule": "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M809_EXACT_SHA",
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }, "release runtime semantics drift")
    require(release["claim_boundary"] == {
        "decoder_component_cycles_after_result_hammer": True,
        "production_speedup_before_result_hammer": False,
        "d1_headline": False, "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False, "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "release claim boundary drift")
    expected = os.environ.get("M819_EXPECTED_RELEASE_SHA256", "")
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
            "m819_m785_decoder_production_attempt_v1" and
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
            "PASS_M819_PARENT_COMPAT_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
            "attempt_manifest_sha256": identity["manifest_sha256"],
            "attempt_outer_seal_file_sha256":
                identity["outer_seal_file_sha256"],
            "scheduled_rows": 0, "production_cycles": None}


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path, True)
    original = M809.validate_true_release
    M809.validate_true_release = lambda *args, **kwargs: gate
    try:
        result = M809.run_production(release_path, candidate_path,
                                     attempt_path, output)
    finally:
        M809.validate_true_release = original
    result = copy.deepcopy(result)
    result["schema"] = (
        "m819_m785_decoder_physical_residency_production_result_v1")
    result["delegated_schedule_body"] = {
        "path": str(M809_PATH.relative_to(HW)), "sha256": M809_SHA256,
        "semantics_changed": False,
        "attempt_status": PARENT_ATTEMPT_STATUS,
    }
    M809._write_json(Path(output) / "result.m819.json", result)
    (Path(output) / "result.json").unlink()
    (Path(output) / "result.m819.json").rename(Path(output) / "result.json")
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
        M809._rename_noreplace(stage, destination)
    except M809.Failure as error:
        raise Failure(str(error)) from error
    require(verify_sealed(destination) == identity and not stage.exists(),
            "publication transition drift")
    return {"status": "PASS_M819_ATOMIC_NOREPLACE_PUBLICATION", **identity}


def write_flat_attempt(directory: Path, receipt: Mapping[str, object]):
    try:
        return M809.write_flat_attempt(directory, receipt)
    except M809.Failure as error:
        raise Failure(str(error)) from error


def _write_failure_receipt(output: Path, stdout_path: Path,
                           stderr_path: Path, payload: Mapping[str, object]):
    try:
        return M809._write_failure_receipt(output, stdout_path, stderr_path,
                                           payload)
    except M809.Failure as error:
        raise Failure(str(error)) from error


def publish_failure_receipt(candidate_path: Path, release_path: Path,
                            attempt_path: Path, runner_path: Path,
                            stdout_path: Path, stderr_path: Path,
                            output: Path, expected_runner_sha256: str,
                            expected_release_sha256: str, return_code: int,
                            phase: str, partial_artifact: str):
    require(return_code != 0 and phase, "failure identity malformed")
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    regular_exact(runner_path, expected_runner_sha256, "failure runner")
    regular_exact(release_path, expected_release_sha256, "failure release")
    require(candidate["source_identity"]["runner"]["sha256"] ==
            expected_runner_sha256 and
            release.get("candidate_binding", {}).get("sha256") ==
            sha256(candidate_path) and
            release.get("source_identity") == candidate["source_identity"],
            "failure release/source drift")
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    canonical_result = (REPO / candidate["canonical"]["result"]).resolve()
    require(Path(attempt_path).resolve() == canonical_attempt,
            "failure attempt path drift")
    identity = verify_sealed(attempt_path)
    receipt = strict_json(Path(attempt_path) / "attempt.json")
    require(receipt.get("schema") ==
            "m819_m785_decoder_production_attempt_v1" and
            receipt.get("status") == PARENT_ATTEMPT_STATUS and
            receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("runner_sha256") == expected_runner_sha256 and
            receipt.get("release_sha256") == expected_release_sha256,
            "failure attempt identity drift")
    output = Path(output).resolve()
    require(output.parent == canonical_result.parent and
            output.name.startswith(canonical_result.name +
                                   ".failed_or_incomplete."),
            "failure quarantine path drift")
    payload = {
        "schema": "m819_m785_decoder_production_failure_receipt_v1",
        "date": "2026-08-29", "status":
            "FAILED_OR_INCOMPLETE__NO_CYCLES_CITABLE",
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
    return {"status": "PASS_M819_SEALED_FAILURE_RECEIPT", **value}


def injected_postpublish_failure_test() -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m819_postpublish_") as directory:
        root = Path(directory)
        stage, attempt = root / "attempt.stage", root / "attempt.consumed"
        result = root / "result"
        quarantine = root / "result.failed_or_incomplete.injected"
        stdout, stderr = root / "stdout", root / "stderr"
        write_flat_attempt(stage, {"schema":
                           "m819_source_test_attempt_v1",
                           "status": PARENT_ATTEMPT_STATUS})
        M809._rename_noreplace(stage, attempt)
        started, phase, scheduled_rows = \
            True, "ATTEMPT_PUBLISHED_POSTCHECK", 0
        stdout.write_text("attempt publish succeeded\n", encoding="utf-8")
        stderr.write_text("injected postcheck failure\n", encoding="utf-8")
        require(started and phase == "ATTEMPT_PUBLISHED_POSTCHECK",
                "post-consumption trap not armed")
        _write_failure_receipt(quarantine, stdout, stderr, {
            "schema": "m819_source_test_failure_v1",
            "status": "INJECTED_POSTPUBLISH_POSTCHECK_FAILURE",
            "scheduled_rows": scheduled_rows,
            "canonical_result_exists": result.exists()})
        members = sorted(entry.name for entry in quarantine.iterdir())
        require(attempt.is_dir() and not result.exists() and
                scheduled_rows == 0 and set(members) == {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256"}, "failure boundary drift")
        identity = verify_sealed(quarantine)
        before = {entry.name: sha256(entry) for entry in quarantine.iterdir()}
        try:
            _write_failure_receipt(quarantine, stdout, stderr,
                                   {"status": "COLLISION"})
        except Failure:
            pass
        else:
            raise Failure("failure collision accepted")
        require(before == {entry.name: sha256(entry) for entry in
                           quarantine.iterdir()}, "collision clobbered receipt")
        return {"status": "PASS_M819_INJECTED_POSTPUBLISH_FAILURE",
                "scheduled_rows": 0, "canonical_result_exists": False,
                "attempt_consumed": True, "failure_members": members,
                "failure_manifest_sha256": identity["manifest_sha256"],
                "collision_no_clobber": True}


def preproduction_traversal_test() -> Dict[str, object]:
    """Enter frozen M809 and stop exactly when it would create output."""
    candidate = strict_json(CANDIDATE)
    with tempfile.TemporaryDirectory(prefix="m819_preproduction_") as directory:
        root = Path(directory)
        candidate_path, release_path = root / "candidate.json", root / "release.json"
        attempt_path, output = root / "attempt", root / "result.stage.test"
        M809._write_json(candidate_path, {"source_test": True})
        M809._write_json(release_path, {"source_test": True})
        fake = copy.deepcopy(candidate)
        fake["canonical"]["attempt"] = str(attempt_path)
        fake["canonical"]["result"] = str(root / "result")
        write_flat_attempt(attempt_path, {
            "schema": "m819_m785_decoder_production_attempt_v1",
            "status": PARENT_ATTEMPT_STATUS,
            "candidate_sha256": sha256(candidate_path),
            "release_sha256": sha256(release_path)})
        original_validate, original_path = M809.validate_true_release, M809.Path
        base_path = type(Path())
        target = os.path.abspath(str(output))

        class GuardedPath(base_path):
            def mkdir(self, *args, **kwargs):
                if os.path.abspath(str(self)) == target:
                    raise ControlledPreproductionStop("AT_OUTPUT_MKDIR")
                return super().mkdir(*args, **kwargs)

        M809.validate_true_release = lambda *args, **kwargs: {
            "candidate": fake}
        M809.Path = GuardedPath
        stopped = False
        try:
            try:
                M809.run_production(release_path, candidate_path,
                                    attempt_path, output)
            except ControlledPreproductionStop as error:
                require(str(error) == "AT_OUTPUT_MKDIR",
                        "wrong controlled stop")
                stopped = True
        finally:
            M809.Path = original_path
            M809.validate_true_release = original_validate
        require(stopped and not output.exists(),
                "preproduction traversal did not stop before output mkdir")
        return {"status":
                "PASS_M819_PARENT_COMPAT_PREPRODUCTION_TRAVERSAL",
                "parent_attempt_status_accepted": True,
                "stopped_at": "M809_OUTPUT_MKDIR",
                "scheduled_rows": 0, "output_exists": False,
                "attempt_receipt_identity_drift": False}


def self_test() -> Dict[str, object]:
    parent = M815.self_test()
    require(parent["status"] ==
            "PASS_M815_RUNNER_RECOVERY_SYNTHETIC_SELF_TEST",
            "M815 parent self-test drift")
    failure = injected_postpublish_failure_test()
    traversal = preproduction_traversal_test()
    runner = HERE / "run_m819_m785_decoder_physical_residency_one_shot.sh"
    require(runner.is_file() and not runner.is_symlink(), "runner absent")
    text = runner.read_text(encoding="utf-8")
    publish = text.index(
        'mv -T --no-clobber -- "${m819_attempt_stage}" "${m819_attempt}"')
    started = text.index("m819_started=1", publish)
    phase = text.index('m819_phase="ATTEMPT_PUBLISHED_POSTCHECK"', publish)
    postcheck = text.index('[[ -d "${m819_attempt}"', publish)
    consumed = text.index("--validate-consumed-attempt", postcheck)
    production = text.index("--run-production", consumed)
    require(publish < started < phase < postcheck < consumed < production,
            "runner order drift")
    require(PARENT_ATTEMPT_STATUS in text, "parent-compatible token absent")
    return {"status":
            "PASS_M819_DELEGATION_COMPAT_SYNTHETIC_SELF_TEST",
            "parent": parent["status"],
            "postpublish_injection": failure,
            "preproduction_traversal": traversal,
            "runner_order":
                "PUBLISH_LT_STARTED_LT_PHASE_LT_POSTCHECK_LT_PREFLIGHT_LT_RUN",
            "schedule_body_sha256": M809_SHA256,
            "schedule_semantics_changed": False,
            "production_cycles": None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preproduction-traversal-test", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
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
             args.validate_candidate, args.validate_release_preflight,
             args.validate_consumed_attempt, args.run_production,
             args.publish_no_replace, args.write_failure_receipt)
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
                         "PASS_M819_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
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
