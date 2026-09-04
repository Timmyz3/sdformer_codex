#!/opt/conda/envs/sdformerflow/bin/python
"""M1243 additive source-hammer launch gate over sealed M1233.

Selection admission and capture are exact aliases of M1233/M1227.  This source
only adds an executable launch-authority gate: a recursively double-sealed,
different-author source hammer must cross-bind this source, contract, and test
and explicitly authorize production_capture=true.  This revision is inert and
source-only; no such hammer or release is authored here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PREDECESSOR = Path(__file__).with_name(
    "capture_m1233_motion_final_checkpoint_unified_hardware_selection_interface_r2.py")
PREDECESSOR_SHA256 = "1227b0746776aff1103937ba5557f325e97e5c8fa751a2593136ece9674f8462"
SOURCE_CONTRACT = HW / (
    "contracts/m1243_motion_capture_launch_authority_successor_source_contract_r1_20260830.json")
TEST = HW / "tests/test_m1243_motion_capture_launch_authority_successor_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1243_motion_capture_launch_authority_successor_source_contract_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1244_DIFFERENT_AUTHOR_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
LAUNCH_SCHEMA = "m1243_motion_final_checkpoint_unified_capture_launch_r1_v1"
LAUNCH_STATUS = (
    "M1234_SELECTION_M1237_RESULT_HAMMER_AND_M1244_SOURCE_HAMMER_BOUND__"
    "ONE_M1243_GPU_RUN_AUTHORIZED")
SOURCE_HAMMER_SCHEMA = "m1244_m1243_motion_capture_launch_authority_source_hammer_r1_v1"
SOURCE_HAMMER_STATUS = (
    "PASS_M1244_M1243_CAPTURE_LAUNCH_AUTHORITY__"
    "PRODUCTION_CAPTURE_RELEASE_AUTHORING_ALLOWED")
SOURCE_AUTHORITY_KEYS = {
    "source_path", "source_sha256", "contract_path", "contract_sha256",
    "test_path", "test_sha256",
}

CANONICAL_RESULT = HW / (
    "results/m1243_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830")
CANONICAL_ATTEMPT = HW / (
    "results/.m1243_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1243_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "production.log")
ATTEMPT_TOKEN = "M1243_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1243_FINAL_CHECKPOINT_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"


class M1243Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1243Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_predecessor():
    require(PREDECESSOR.is_file() and not PREDECESSOR.is_symlink(),
            "missing or symlink M1233 predecessor")
    require(sha256(PREDECESSOR) == PREDECESSOR_SHA256, "M1233 predecessor SHA drift")
    spec = importlib.util.spec_from_file_location("m1243_sealed_m1233", str(PREDECESSOR))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1233")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load_predecessor()
R1 = P.R1

# Frozen selection and capture surface; no reimplementation is permitted here.
validate_final_selection = P.validate_final_selection
EXPECTED_STATIC_COUNTS = P.EXPECTED_STATIC_COUNTS
EXPECTED_LIVE_COUNTS = P.EXPECTED_LIVE_COUNTS
DEAD_SN_V = P.DEAD_SN_V
audit_call_matrix = P.audit_call_matrix
audit_attention_population = P.audit_attention_population
validate_payload_population = P.validate_payload_population
atomic_sample_snapshot = P.atomic_sample_snapshot
final_validate_and_seal = P.final_validate_and_seal
ALLOWED_SELECTION_SCHEMA = P.ALLOWED_SELECTION_SCHEMA
ALLOWED_SELECTION_STATUS = P.ALLOWED_SELECTION_STATUS
SELECTION_RESULT_HAMMER_SCHEMA = P.SELECTION_RESULT_HAMMER_SCHEMA
SELECTION_RESULT_HAMMER_STATUS = P.SELECTION_RESULT_HAMMER_STATUS


def strict_json(path: Path) -> dict[str, Any]:
    try:
        return P.strict_json(path)
    except P.M1233Error as exc:
        raise M1243Error(str(exc)) from exc


def safe_repo_path(relative: str) -> Path:
    try:
        return P.safe_repo_path(relative)
    except P.M1233Error as exc:
        raise M1243Error(str(exc)) from exc


def verify_double_seal(root: Path, manifest_sha: str,
                       outer_file_sha: str) -> dict[str, str]:
    try:
        return P.verify_double_seal(root, manifest_sha, outer_file_sha)
    except P.M1233Error as exc:
        raise M1243Error(str(exc)) from exc


def verify_source_hammer(entry: dict[str, Any]) -> dict[str, Any]:
    required_entry_keys = {"path", "manifest_sha256", "outer_file_sha256", "review_sha256"}
    require(isinstance(entry, dict) and set(entry) == required_entry_keys,
            "source hammer entry keys mismatch")
    root = safe_repo_path(entry["path"])
    require(root.parent == HW / "reviews", "source hammer must be directly under reviews")
    rows = verify_double_seal(root, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "source hammer review member mismatch")
    review = strict_json(root / "review.json")
    require(review.get("schema") == SOURCE_HAMMER_SCHEMA, "source hammer schema mismatch")
    require(review.get("status") == SOURCE_HAMMER_STATUS, "source hammer status mismatch")
    authority = review.get("source_authority")
    require(isinstance(authority, dict) and set(authority) == SOURCE_AUTHORITY_KEYS,
            "source hammer authority keys mismatch")
    expected = {
        "source_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__).resolve()),
        "contract_path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(SOURCE_CONTRACT),
        "test_path": str(TEST.relative_to(ROOT)),
        "test_sha256": sha256(TEST),
    }
    require(authority == expected, "source hammer source/contract/test cross-SHA mismatch")
    require(review.get("independence") == {"different_author": True},
            "source hammer must assert different-author independence")
    require(review.get("authorization") == {"production_capture": True},
            "source hammer production-capture authority mismatch")
    return {
        "path": str(root.relative_to(ROOT)),
        "review_sha256": entry["review_sha256"],
        "manifest_sha256": entry["manifest_sha256"],
        "outer_file_sha256": entry["outer_file_sha256"],
        "production_capture": True,
    }


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == LAUNCH_SCHEMA and contract.get("status") == LAUNCH_STATUS,
            "source-only or unhammered M1243 contract cannot launch")
    require(contract.get("contract_path") == str(contract_path.relative_to(ROOT)),
            "launch contract path mismatch")
    inputs = contract.get("inputs")
    require(isinstance(inputs, dict), "launch inputs missing")
    require(inputs.get("launcher") == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }, "M1243 launcher identity mismatch")
    require(inputs.get("source_contract") == {
        "path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "sha256": sha256(SOURCE_CONTRACT),
    }, "M1243 source contract identity mismatch")
    source_hammer = verify_source_hammer(inputs.get("source_hammer"))
    try:
        R1.validate_m1224()
        binding = P.validate_final_selection(
            inputs["final_selection_result"], inputs["final_selection_result_hammer"])
        samples = R1.validate_cohort(contract["cohort"]["samples"])
        require(R1.safe_repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True)
                == CANONICAL_ATTEMPT and
                R1.safe_repo_path(contract["output"]["path"], missing_leaf=True)
                == CANONICAL_RESULT and
                R1.safe_repo_path(contract["production_log"]["path"], missing_leaf=True)
                == CANONICAL_LOG, "M1243 fresh namespace mismatch")
    except (P.M1233Error, R1.M1227Error, KeyError, TypeError) as exc:
        raise M1243Error(str(exc)) from exc
    binding["identity"]["source_hammer"] = source_hammer
    return dict(binding, verified_samples=samples, policy=R1.strict_json(R1.SOURCE_CONTRACT))


def run_capture(contract: dict[str, Any], binding: dict[str, Any],
                predecessor=None, substrate=None):
    """Delegate selection/capture unchanged; vary only the canonical result namespace."""
    module = P if predecessor is None else predecessor
    original_result = module.CANONICAL_RESULT
    try:
        module.CANONICAL_RESULT = CANONICAL_RESULT
        return module.run_capture(contract, binding, predecessor=R1, substrate=substrate)
    finally:
        module.CANONICAL_RESULT = original_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(str(contract_path).startswith(str(ROOT) + os.sep),
            "launch contract must be in repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    require(not os.path.lexists(str(CANONICAL_ATTEMPT)) and
            not os.path.lexists(str(CANONICAL_RESULT)) and
            not os.path.lexists(str(CANONICAL_LOG)), "fresh M1243 namespace required")
    substrate = R1.load_substrate()
    with substrate.exclusive_gpu_lease(R1.CANONICAL_LEASE):
        descriptor = os.open(
            str(CANONICAL_ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output = run_capture(contract, binding, predecessor=P, substrate=substrate)
    R1.verify_double_seal(output)
    print(PASS_TOKEN + " " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
