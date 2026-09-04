#!/opt/conda/envs/sdformerflow/bin/python
"""M1249 final-checkpoint unified-capture one-shot production release.

This source is inert until a future production launch contract supplies a sealed
M1234 selection and its exact M1237 result-hammer entry.  It binds the exact M1243
capture-launch implementation and the recursively sealed M1244 source hammer.
No production launch contract is authored in this revision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1243_SOURCE = Path(__file__).with_name(
    "capture_m1243_motion_final_checkpoint_unified_hardware_launch_authority_r3.py")
M1243_SOURCE_SHA256 = "009c92c22b5429352b0b4dd29c723035744efa828db9c4472d1f4fb4140297e2"
M1243_TEST = HW / "tests/test_m1243_motion_capture_launch_authority_successor_source.py"
M1243_TEST_SHA256 = "7529dd988e48926d683c0ea28c1ca5e9e06a2af617febe796a02e09e38c3ded7"
M1243_CONTRACT = HW / (
    "contracts/m1243_motion_capture_launch_authority_successor_source_contract_r1_20260830.json")
M1243_CONTRACT_SHA256 = "de558985c0f9a64580060dce90675d8ba4ca771a616fe8152b439483663f26ba"
M1244_ROOT = HW / (
    "reviews/m1244_m1243_motion_capture_launch_authority_source_hammer_r1_20260830")
M1244_ENTRY = {
    "path": str(M1244_ROOT.relative_to(ROOT)),
    "manifest_sha256": "8b4e633103098faf140c1660abd1ac6e4745bb7dd3c2838ec9ac88ee6a9adce2",
    "outer_file_sha256": "657af0a531ed95e3abb301f2dd5b5827e3f737dcc34ab3120f3f593ad3ac55f2",
    "review_sha256": "64773f9fc58b67af2caf9cf60642ace071e526ee9de928cfb515c419959edd8a",
}
SOURCE_CONTRACT = HW / (
    "contracts/m1249_motion_final_checkpoint_unified_capture_one_shot_release_source_contract_r1_20260830.json")
TEST = HW / (
    "tests/test_m1249_motion_final_checkpoint_unified_capture_one_shot_release_source.py")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1249_motion_final_checkpoint_unified_capture_one_shot_release_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__FINAL_M1237_RESULT_HAMMER_REQUIRED__NO_PRODUCTION_LAUNCH"
PRODUCTION_SCHEMA = "m1249_motion_final_checkpoint_unified_capture_one_shot_production_launch_r1_v1"
PRODUCTION_STATUS = (
    "M1243_M1244_AND_FINAL_M1237_BOUND__ONE_M1249_GPU_RUN_AUTHORIZED")
PASS_TOKEN = "PASS_M1249_FINAL_CHECKPOINT_UNIFIED_CAPTURE__RESULT_HAMMER_REQUIRED"
ATTEMPT_TOKEN = "M1249_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"

CANONICAL_RESULT = HW / (
    "results/m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830")
CANONICAL_ATTEMPT = HW / (
    "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "production.log")

IDENTITY_KEYS = {"path", "sha256"}
RELEASE_IDENTITY_KEYS = {
    "source_path", "source_sha256", "test_path", "test_sha256",
    "source_contract_path", "source_contract_sha256",
}
M1237_ENTRY_KEYS = {"path", "manifest_sha256", "outer_file_sha256", "review_sha256"}
INPUT_KEYS = {
    "m1243_source", "m1243_test", "m1243_source_contract", "m1244_source_hammer",
    "final_selection_result", "final_selection_result_hammer",
}
TOP_KEYS = {
    "schema", "status", "contract_path", "release_identity", "inputs", "cohort",
    "one_shot", "output", "production_log",
}


class M1249Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1249Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1249Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def _load_m1243():
    regular_exact(M1243_SOURCE, M1243_SOURCE_SHA256, "M1243 source")
    regular_exact(M1243_TEST, M1243_TEST_SHA256, "M1243 test")
    regular_exact(M1243_CONTRACT, M1243_CONTRACT_SHA256, "M1243 contract")
    spec = importlib.util.spec_from_file_location("m1249_sealed_m1243", str(M1243_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1243")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1243 = _load_m1243()
R1 = M1243.R1

# Capture semantics remain exact aliases; this file only owns final admission and namespace.
EXPECTED_STATIC_COUNTS = M1243.EXPECTED_STATIC_COUNTS
EXPECTED_LIVE_COUNTS = M1243.EXPECTED_LIVE_COUNTS
DEAD_SN_V = M1243.DEAD_SN_V
audit_call_matrix = M1243.audit_call_matrix
audit_attention_population = M1243.audit_attention_population
validate_payload_population = M1243.validate_payload_population
atomic_sample_snapshot = M1243.atomic_sample_snapshot
final_validate_and_seal = M1243.final_validate_and_seal


def strict_json(path: Path) -> dict[str, Any]:
    try:
        return M1243.strict_json(path)
    except M1243.M1243Error as exc:
        raise M1249Error(str(exc)) from exc


def exact_identity(value: Any, expected_path: Path, expected_sha: str, label: str) -> None:
    require(isinstance(value, dict) and set(value) == IDENTITY_KEYS,
            label + " identity keys mismatch")
    require(value == {"path": str(expected_path.relative_to(ROOT)), "sha256": expected_sha},
            label + " identity mismatch")
    regular_exact(expected_path, expected_sha, label)


def exact_release_identity(value: Any) -> None:
    require(isinstance(value, dict) and set(value) == RELEASE_IDENTITY_KEYS,
            "M1249 release identity keys mismatch")
    expected = {
        "source_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__).resolve()),
        "test_path": str(TEST.relative_to(ROOT)),
        "test_sha256": sha256(TEST),
        "source_contract_path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
    }
    require(value == expected, "M1249 source/test/contract release identity mismatch")


def ensure_fresh_namespaces() -> None:
    for path, label in (
        (CANONICAL_RESULT, "result"), (CANONICAL_ATTEMPT, "attempt"),
        (CANONICAL_LOG, "production log"),
    ):
        require(not os.path.lexists(str(path)), "M1249 " + label + " namespace is not fresh")


def validate_production_launch(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(isinstance(contract, dict) and set(contract) == TOP_KEYS,
            "production launch top-level keys mismatch")
    require(contract.get("schema") == PRODUCTION_SCHEMA and
            contract.get("status") == PRODUCTION_STATUS,
            "source-only or non-production M1249 contract cannot launch")
    require(contract.get("contract_path") == str(contract_path.relative_to(ROOT)),
            "production launch contract path mismatch")
    exact_release_identity(contract.get("release_identity"))
    inputs = contract.get("inputs")
    require(isinstance(inputs, dict) and set(inputs) == INPUT_KEYS, "launch inputs mismatch")
    exact_identity(inputs["m1243_source"], M1243_SOURCE, M1243_SOURCE_SHA256, "M1243 source")
    exact_identity(inputs["m1243_test"], M1243_TEST, M1243_TEST_SHA256, "M1243 test")
    exact_identity(inputs["m1243_source_contract"], M1243_CONTRACT,
                   M1243_CONTRACT_SHA256, "M1243 contract")
    require(inputs["m1244_source_hammer"] == M1244_ENTRY,
            "M1244 source-hammer exact entry mismatch")
    source_hammer = M1243.verify_source_hammer(inputs["m1244_source_hammer"])
    result_hammer = inputs.get("final_selection_result_hammer")
    require(isinstance(result_hammer, dict) and set(result_hammer) == M1237_ENTRY_KEYS,
            "future M1237 result-hammer exact path/seal entry is required")
    try:
        R1.validate_m1224()
        binding = M1243.validate_final_selection(
            inputs["final_selection_result"], result_hammer)
        samples = R1.validate_cohort(contract["cohort"]["samples"])
        require(R1.safe_repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True)
                == CANONICAL_ATTEMPT and
                R1.safe_repo_path(contract["output"]["path"], missing_leaf=True)
                == CANONICAL_RESULT and
                R1.safe_repo_path(contract["production_log"]["path"], missing_leaf=True)
                == CANONICAL_LOG, "M1249 exact fresh namespace mismatch")
    except (M1243.M1243Error, R1.M1227Error, KeyError, TypeError) as exc:
        raise M1249Error(str(exc)) from exc
    require(contract["one_shot"] == {
        "attempt_marker": str(CANONICAL_ATTEMPT.relative_to(ROOT)),
        "automatic_retry": False,
    }, "M1249 one-shot policy mismatch")
    ensure_fresh_namespaces()
    binding["identity"]["m1244_source_hammer"] = source_hammer
    return dict(binding, verified_samples=samples, policy=R1.strict_json(R1.SOURCE_CONTRACT))


def consume_attempt() -> None:
    descriptor = os.open(
        str(CANONICAL_ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_capture(contract: dict[str, Any], binding: dict[str, Any], substrate=None):
    original = M1243.CANONICAL_RESULT
    try:
        M1243.CANONICAL_RESULT = CANONICAL_RESULT
        return M1243.run_capture(
            contract, binding, predecessor=M1243.P, substrate=substrate)
    finally:
        M1243.CANONICAL_RESULT = original


def execute_once(contract: dict[str, Any], contract_path: Path, substrate):
    """Revalidate all source/checkpoint/config/cohort inputs under lease, then burn one shot."""
    with substrate.exclusive_gpu_lease(R1.CANONICAL_LEASE):
        binding = validate_production_launch(contract, contract_path)
        consume_attempt()
        return run_capture(contract, binding, substrate=substrate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(str(contract_path).startswith(str(ROOT) + os.sep),
            "production launch contract must be in repository")
    contract = strict_json(contract_path)
    # Substrate loading is inert with respect to the one-shot marker.  The marker is
    # created only after execute_once revalidates every admission input under GPU lease.
    substrate = R1.load_substrate()
    output = execute_once(contract, contract_path, substrate)
    R1.verify_double_seal(output)
    print(PASS_TOKEN + " " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
