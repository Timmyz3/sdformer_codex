#!/usr/bin/env python3
"""Additive M815 production boundary around the frozen M809/M785 schedule.

M815 does not modify or relabel the M809 source candidate.  It binds a new
runner/release identity, keeps the M811 NO-GO as negative provenance, and
delegates only the cycle-generation body to the exact frozen M809 driver.
"""

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
M809_PATH = HERE / "execute_m809_m785_decoder_physical_residency_production.py"
M809_SHA256 = "2b273d5fb3f68ae7cae16c458b8138ab89c83a677ce9aa8f15b90de2fd6736d0"
M809_CANDIDATE = (
    HW / "contracts/m809_m785_decoder_physical_residency_production_recovery_candidate_r1_20260829.json"
)
M809_CANDIDATE_SHA256 = "9742335fb312af4e3d8805c43d44ae5470e0af32a2a1c9026cdb2ef4b0b3e635"
M811_REVIEW = (
    HW / "reviews/m811_m809_m785_decoder_production_source_fresh_hammer_r1_20260829"
)
M811_REVIEW_JSON_SHA256 = "0022d1ca8eb151e30864a2f78b4adac2ac6a23363c4441f64a227ee9cf187c14"
M811_MANIFEST_SHA256 = "1d8f237209b0ffa0f11ae7bcaa4f28f8a4fa61b0ca1a9cce682dca17ec0b7e67"
M811_OUTER_SEAL_FILE_SHA256 = "ebea473d3f876e6c88348849a2822120b639082bbdad582760bb95501a6733ca"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANDIDATE_SCHEMA = "m815_m785_decoder_production_runner_recovery_candidate_v1"
RELEASE_SCHEMA = "m815_m785_decoder_production_true_release_v1"
SOURCE_HAMMER_DIR = (
    "reviews/m817_m815_m785_decoder_production_source_fresh_hammer_r1_20260829"
)
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


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


def load_m809():
    path = M809_PATH
    require(path.is_file() and not path.is_symlink(),
            "frozen M809 driver is absent or nonregular")
    value = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    require(value == M809_SHA256, "frozen M809 driver SHA drift")
    spec = importlib.util.spec_from_file_location("m815_frozen_m809", path)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M809 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M809 = load_m809()


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {
        name: HW / entry["path"]
        for name, entry in candidate["source_identity"].items()
    }


def _canonical_paths(candidate: Mapping[str, object]):
    canonical = candidate["canonical"]
    return (
        REPO / canonical["result"],
        REPO / canonical["attempt"],
        REPO / canonical["future_release"],
    )


def validate_candidate(candidate_path: Path, require_future_absent: bool = True,
                       attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict), "candidate must be an object")
    require(candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("status") ==
            "SOURCE_ONLY_M815_RUNNER_RECOVERY_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "M815 candidate identity drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate must not authorize production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "synthetic_self_test": True,
        "fresh_source_hammer": True,
        "production_replay": False,
        "result_directory": False,
        "cycles_or_speedup": False,
        "rtl_vcs_eda_gpu_remote": False,
    }, "candidate authorization is not closed")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")

    paths = _candidate_paths(candidate)
    require(set(paths) == {
        "driver", "runner", "tests", "recovery_contract",
        "m809_parent_driver", "m809_parent_candidate", "m785_contract",
        "analyzer", "storage_oracle",
    }, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate does not bind this exact M815 driver")
    require(paths["runner"].resolve() ==
            HERE / "run_m815_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() ==
            HERE.parent / "tests/test_m815_m809_decoder_runner_recovery.py",
            "candidate tests path drift")
    require(paths["m809_parent_driver"].resolve() == M809_PATH and
            candidate["source_identity"]["m809_parent_driver"]["sha256"] ==
            M809_SHA256, "M809 parent driver drift")
    require(paths["m809_parent_candidate"].resolve() == M809_CANDIDATE and
            candidate["source_identity"]["m809_parent_candidate"]["sha256"] ==
            M809_CANDIDATE_SHA256, "M809 parent candidate drift")
    parent = M809.validate_candidate(M809_CANDIDATE)
    require(parent["status"] ==
            "PASS_M809_REPAIRED_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
            "frozen M809 parent source validation drift")

    m811_identity = verify_sealed(M811_REVIEW)
    regular_exact(M811_REVIEW / "review.json", M811_REVIEW_JSON_SHA256,
                  "M811 NO-GO review")
    require(m811_identity["manifest_sha256"] == M811_MANIFEST_SHA256 and
            m811_identity["outer_seal_file_sha256"] ==
            M811_OUTER_SEAL_FILE_SHA256, "M811 double seal drift")
    m811 = strict_json(M811_REVIEW / "review.json")
    require(m811.get("status") ==
            "NO_GO_M809_TRUE_RELEASE__P1_1__AUTHOR_ADDITIVE_RUNNER_REPAIR_REQUIRED" and
            m811.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 0} and
            m811.get("true_release_authorized") is False and
            m811.get("production_launch_authorized") is False,
            "M811 negative provenance was weakened")
    require(candidate["m811_no_go_basis"] == {
        "directory":
            "reviews/m811_m809_m785_decoder_production_source_fresh_hammer_r1_20260829",
        "review_json_sha256": M811_REVIEW_JSON_SHA256,
        "manifest_sha256": M811_MANIFEST_SHA256,
        "outer_seal_file_sha256": M811_OUTER_SEAL_FILE_SHA256,
        "true_release_authorized": False,
        "additive_runner_repair_required": True,
    }, "candidate M811 basis drift")

    contract = strict_json(paths["recovery_contract"])
    require(contract.get("schema") ==
            "m815_m809_decoder_runner_recovery_contract_v1" and
            contract.get("status") ==
            "SOURCE_ONLY_ADDITIVE_RUNNER_RECOVERY__NO_RELEASE_OR_PRODUCTION" and
            contract.get("launch_now") is False and
            contract.get("release") is False and
            contract.get("max_attempts") == 0,
            "M815 recovery contract drift")
    require(candidate["common_resource"] == {
        "lanes": 96,
        "onchip_sram_bytes_macro_rounded": 245760,
        "accumulator_bits": 24,
        "clock_ns": 3.0,
        "external_bytes_per_cycle": 192,
        "weight_bytes": 13824,
        "psum_bytes": 221184,
        "descriptor_control_bytes": 8192,
        "reserved_unallocated_bytes": 2560,
        "reserved_borrow_allowed": False,
        "resource_manifest_sha256":
            "a7400bddb174a00875298cd9bd8d2692e636727ff27b22ae580803383fdea0f3",
    }, "common resource drift")
    require(candidate["production_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configurations": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "cold_start_and_drain_per_record_timestep": True,
        "records": "40_PLUS_120",
        "timesteps": 10,
        "only_legal_headline_ratio":
            "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8",
        "headline_total_excludes_modules": [1],
        "all_module_total_cycles_retained": True,
        "k8_vs_a1_headline_allowed": False,
        "d1": "COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M809_EXACT_SHA",
    }, "production semantics drift")
    require(candidate["canonical"] == {
        "result":
            "hw_autoresearch_nts07/results/m815_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt":
            "hw_autoresearch_nts07/results/.m815_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release":
            "hw_autoresearch_nts07/contracts/m815_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix":
            "hw_autoresearch_nts07/results/m815_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_incomplete.",
    }, "canonical path drift")
    result, attempt, future = _canonical_paths(candidate)
    require(not result.exists() and not result.is_symlink(),
            "canonical result already exists")
    if attempt_required:
        require(attempt.is_dir() and not attempt.is_symlink(),
                "attempt must be consumed")
        verify_sealed(attempt)
    else:
        require(not attempt.exists() and not attempt.is_symlink(),
                "canonical attempt already exists")
    if require_future_absent:
        require(not future.exists() and not future.is_symlink(),
                "future release must be absent")
    else:
        require(future.is_file() and not future.is_symlink(),
                "true release absent or nonregular")
    require(candidate["claim_boundary"] == {
        "source_only": True,
        "production_replay": False,
        "production_cycles": False,
        "production_speedup": False,
        "d1_headline": False,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
        "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "claim boundary drift")
    return {
        "status": "PASS_M815_RUNNER_RECOVERY_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
        "candidate_sha256": sha256(candidate_path),
        "parent_status": parent["status"],
        "production_cycles": None,
    }


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool):
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    candidate_validation = validate_candidate(
        candidate_path, require_future_absent=False,
        attempt_required=attempt_required)
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    require(isinstance(release, dict) and
            release.get("schema") == RELEASE_SCHEMA and
            release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_M815_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release.get("launch_now") is True and
            release.get("release") is True and
            release.get("max_attempts") == 1,
            "M815 true-release authorization drift")
    binding = release["candidate_binding"]
    require((HW / binding["path"]).resolve() == Path(candidate_path).resolve() and
            binding["sha256"] == candidate_validation["candidate_sha256"],
            "release candidate binding drift")
    _result, _attempt, future = _canonical_paths(candidate)
    require(release_path == future.resolve(),
            "release path is not the canonical future release")
    require(release["source_identity"] == candidate["source_identity"] and
            release["canonical"] == candidate["canonical"],
            "release changed candidate identity")
    require(release["reviewed_source_identity"] == {
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256":
            candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256":
            candidate["source_identity"]["runner"]["sha256"],
    }, "release reviewed-source identity drift")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR,
            "fresh hammer directory drift")
    hammer_dir = HW / hammer["directory"]
    identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M817 source hammer")
    require(identity["manifest_sha256"] == hammer["manifest_sha256"] and
            identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"], "M817 seal drift")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M815_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0} and
            review.get("true_release_authorized") is True and
            review.get("production_launch_authorized") is False,
            "M817 PASS100 semantics drift")
    target = review["review_target"]
    require(target["candidate_sha256"] == sha256(candidate_path) and
            target["driver_sha256"] ==
            candidate["source_identity"]["driver"]["sha256"] and
            target["runner_sha256"] ==
            candidate["source_identity"]["runner"]["sha256"],
            "M817 did not review exact source")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M809_EXACT_SHA",
    }, "release runtime semantics drift")
    require(release["claim_boundary"] == {
        "decoder_component_cycles_after_result_hammer": True,
        "production_speedup_before_result_hammer": False,
        "d1_headline": False,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
        "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "release claim boundary drift")
    expected = os.environ.get("M815_EXPECTED_RELEASE_SHA256", "")
    require(len(expected) == 64 and sha256(release_path) == expected,
            "caller did not supply exact reviewed release SHA")
    sidecar = Path(str(release_path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink() and
            sidecar.read_text(encoding="utf-8") ==
            expected + "  " + Path(release_path).name + "\n" and
            outer.read_text(encoding="utf-8") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            "true-release sidecar drift")
    return {"release": release, "candidate": candidate,
            "candidate_validation": candidate_validation}


def validate_consumed_attempt(release_path: Path, candidate_path: Path,
                              attempt_path: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path,
                                 attempt_required=True)
    candidate = gate["candidate"]
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    attempt_path = Path(attempt_path).resolve()
    require(attempt_path == canonical_attempt, "attempt path drift")
    identity = verify_sealed(attempt_path)
    receipt = strict_json(attempt_path / "attempt.json")
    require(receipt.get("schema") ==
            "m815_m785_decoder_production_attempt_v1" and
            receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_M815_PRODUCTION_REPLAY" and
            receipt.get("max_attempts") == 1 and
            receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            receipt.get("runner_sha256") ==
            candidate["source_identity"]["runner"]["sha256"] and
            receipt.get("release_sha256") == sha256(release_path),
            "consumed-attempt identity drift")
    return {
        "status": "PASS_M815_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
        "attempt_manifest_sha256": identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256":
            identity["outer_seal_file_sha256"],
        "scheduled_rows": 0,
        "production_cycles": None,
    }


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(release_path, candidate_path,
                                 attempt_required=True)
    original = M809.validate_true_release
    M809.validate_true_release = lambda release, candidate, attempt_required: gate
    try:
        result = M809.run_production(
            release_path, candidate_path, attempt_path, output)
    finally:
        M809.validate_true_release = original
    result = copy.deepcopy(result)
    result["schema"] = (
        "m815_m785_decoder_physical_residency_production_result_v1")
    result["delegated_schedule_body"] = {
        "path": str(M809_PATH.relative_to(HW)),
        "sha256": M809_SHA256,
        "semantics_changed": False,
    }
    M809._write_json(Path(output) / "result.m815.json", result)
    (Path(output) / "result.json").unlink()
    (Path(output) / "result.m815.json").rename(Path(output) / "result.json")
    return result


def publish_no_replace(candidate_path: Path, stage: Path,
                       destination: Path) -> Dict[str, object]:
    candidate = strict_json(candidate_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "publication candidate drift")
    paths = _candidate_paths(candidate)
    regular_exact(paths["driver"],
                  candidate["source_identity"]["driver"]["sha256"],
                  "publication driver")
    regular_exact(paths["runner"],
                  candidate["source_identity"]["runner"]["sha256"],
                  "publication runner")
    canonical = (REPO / candidate["canonical"]["result"]).resolve()
    stage = Path(stage).resolve()
    destination = Path(destination).resolve()
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
    require(destination.is_dir() and not destination.is_symlink() and
            not stage.exists(), "publication transition failed")
    require(verify_sealed(destination) == identity,
            "publication changed sealed identity")
    return {"status": "PASS_M815_ATOMIC_NOREPLACE_PUBLICATION",
            **identity}


def seal_exact_members(directory: Path, member_names: Sequence[str]):
    try:
        return M809.seal_exact_members(directory, member_names)
    except M809.Failure as error:
        raise Failure(str(error)) from error


def write_flat_attempt(directory: Path, receipt: Mapping[str, object]):
    try:
        return M809.write_flat_attempt(directory, receipt)
    except M809.Failure as error:
        raise Failure(str(error)) from error


def _write_failure_receipt(output: Path, stdout_path: Path,
                           stderr_path: Path, payload: Mapping[str, object]):
    try:
        return M809._write_failure_receipt(
            output, stdout_path, stderr_path, payload)
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
    paths = _candidate_paths(candidate)
    regular_exact(paths["driver"],
                  candidate["source_identity"]["driver"]["sha256"],
                  "failure driver")
    regular_exact(runner_path, expected_runner_sha256, "failure runner")
    regular_exact(release_path, expected_release_sha256, "failure release")
    require(paths["runner"].resolve() == Path(runner_path).resolve() and
            release.get("candidate_binding", {}).get("sha256") ==
            sha256(candidate_path) and
            release.get("source_identity") == candidate.get("source_identity"),
            "failure release/source binding drift")
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    canonical_result = (REPO / candidate["canonical"]["result"]).resolve()
    require(Path(attempt_path).resolve() == canonical_attempt,
            "failure attempt path drift")
    attempt_identity = verify_sealed(attempt_path)
    receipt = strict_json(Path(attempt_path) / "attempt.json")
    require(receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            receipt.get("runner_sha256") == expected_runner_sha256 and
            receipt.get("release_sha256") == expected_release_sha256,
            "failure attempt identity drift")
    output = Path(output).resolve()
    require(output.parent == canonical_result.parent and
            output.name.startswith(canonical_result.name +
                                   ".failed_or_incomplete."),
            "failure quarantine path drift")
    payload = {
        "schema": "m815_m785_decoder_production_failure_receipt_v1",
        "date": "2026-08-29",
        "status": "FAILED_OR_INCOMPLETE__NO_CYCLES_CITABLE",
        "return_code": int(return_code),
        "phase": phase,
        "partial_artifact": partial_artifact,
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256": expected_runner_sha256,
        "release_sha256": expected_release_sha256,
        "attempt_manifest_sha256": attempt_identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256":
            attempt_identity["outer_seal_file_sha256"],
        "scheduled_rows": 0 if phase ==
            "ATTEMPT_PUBLISHED_POSTCHECK" else None,
        "claim_boundary": {
            "production_complete": False,
            "cycles_citable": False,
            "speedup_citable": False,
            "decoder_complete": False,
            "table_a_insertion_allowed": False,
        },
    }
    identity = _write_failure_receipt(
        output, stdout_path, stderr_path, payload)
    return {"status": "PASS_M815_SEALED_FAILURE_RECEIPT", **identity}


def injected_postpublish_failure_test() -> Dict[str, object]:
    """Exercise the repaired state transition without a formal path."""
    with tempfile.TemporaryDirectory(prefix="m815_postpublish_") as directory:
        root = Path(directory)
        attempt_stage = root / "attempt.stage"
        attempt = root / "attempt.consumed"
        result = root / "result"
        quarantine = root / "result.failed_or_incomplete.injected"
        stdout = root / "stdout"
        stderr = root / "stderr"
        write_flat_attempt(attempt_stage, {
            "schema": "m815_source_test_attempt_v1",
            "status": "SOURCE_TEST_ONLY",
        })
        M809._rename_noreplace(attempt_stage, attempt)
        started = True
        phase = "ATTEMPT_PUBLISHED_POSTCHECK"
        scheduled_rows = 0
        stdout.write_text("attempt publish succeeded\n", encoding="utf-8")
        stderr.write_text("injected postcheck failure\n", encoding="utf-8")
        require(started and phase == "ATTEMPT_PUBLISHED_POSTCHECK",
                "post-consumption trap was not armed")
        _write_failure_receipt(quarantine, stdout, stderr, {
            "schema": "m815_source_test_failure_v1",
            "status": "INJECTED_POSTPUBLISH_POSTCHECK_FAILURE",
            "scheduled_rows": scheduled_rows,
            "canonical_result_exists": result.exists(),
        })
        require(attempt.is_dir() and not result.exists() and
                scheduled_rows == 0 and
                {entry.name for entry in quarantine.iterdir()} == {
                    "failure.json", "driver.log", "SHA256SUMS",
                    "SHA256SUMS.seal.sha256"},
                "injected failure boundary drift")
        identity = verify_sealed(quarantine)
        before = {entry.name: sha256(entry) for entry in
                  quarantine.iterdir()}
        try:
            _write_failure_receipt(quarantine, stdout, stderr,
                                   {"status": "COLLISION"})
        except Failure:
            pass
        else:
            raise Failure("failure quarantine collision was accepted")
        require(before == {entry.name: sha256(entry) for entry in
                           quarantine.iterdir()},
                "collision modified failure evidence")
        return {
            "status": "PASS_M815_INJECTED_POSTPUBLISH_FAILURE",
            "scheduled_rows": 0,
            "canonical_result_exists": False,
            "attempt_consumed": True,
            "failure_members": sorted(entry.name for entry in
                                      quarantine.iterdir()),
            "failure_manifest_sha256": identity["manifest_sha256"],
            "collision_no_clobber": True,
        }


def self_test() -> Dict[str, object]:
    parent = M809.self_test()
    require(parent["status"] ==
            "PASS_M809_REPAIRED_DRIVER_SYNTHETIC_SELF_TEST",
            "M809 parent self-test drift")
    injected = injected_postpublish_failure_test()
    runner = (HERE /
              "run_m815_m785_decoder_physical_residency_one_shot.sh")
    require(runner.is_file() and not runner.is_symlink(),
            "M815 runner absent")
    text = runner.read_text(encoding="utf-8")
    publish = text.index(
        'mv -T --no-clobber -- "${m815_attempt_stage}" "${m815_attempt}"')
    started = text.index("m815_started=1", publish)
    phase = text.index('m815_phase="ATTEMPT_PUBLISHED_POSTCHECK"', publish)
    postcheck = text.index('[[ -d "${m815_attempt}"', publish)
    consumed = text.index("--validate-consumed-attempt", postcheck)
    production = text.index("--run-production", consumed)
    require(publish < started < phase < postcheck < consumed < production,
            "runner did not arm failure receipt before fallible postcheck")
    return {
        "status": "PASS_M815_RUNNER_RECOVERY_SYNTHETIC_SELF_TEST",
        "parent": parent["status"],
        "postpublish_injection": injected,
        "runner_order":
            "PUBLISH_LT_STARTED_LT_PHASE_LT_POSTCHECK_LT_PREFLIGHT_LT_RUN",
        "schedule_body_sha256": M809_SHA256,
        "schedule_semantics_changed": False,
        "production_cycles": None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
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
    modes = (args.self_test, args.validate_candidate,
             args.validate_release_preflight,
             args.validate_consumed_attempt, args.run_production,
             args.publish_no_replace, args.write_failure_receipt)
    require(sum(bool(value) for value in modes) == 1,
            "select exactly one mode")
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
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
            "status": "PASS_M815_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
            "candidate_sha256": value["candidate_validation"][
                "candidate_sha256"],
            "release_sha256": sha256(args.release),
            "production_cycles": None,
        }, sort_keys=True, allow_nan=False))
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
        print(json.dumps(publish_no_replace(
            args.candidate, args.output, args.publish_to), sort_keys=True,
            allow_nan=False))
        return 0
    if args.write_failure_receipt:
        require(all((args.release, args.attempt, args.runner,
                     args.stdout_log, args.stderr_log, args.output,
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
    value = run_production(args.release, args.candidate,
                           args.attempt, args.output)
    print(json.dumps({
        "status": value["status"],
        "result_sha256": sha256(Path(args.output) / "result.json"),
        "detailed_rows": value["detailed_rows"],
        "fresh_result_hammer_required": True,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
