#!/opt/conda/envs/sdformerflow/bin/python
"""M1180 namespace-clean, fail-closed Motion unified hardware capture.

The M1177 r2 implementation is retained as a SHA-pinned, source-only
substrate after its independent hammer found no technical defect.  This
successor owns a disjoint M1180 authority namespace: source policy, future
hammer, launch, one-shot marker, result directory, log, and PASS token.
Importing this module never accesses a GPU, checkpoint, remote host, or EDA.
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
SUBSTRATE_PATH = Path(__file__).with_name(
    "capture_m1177_motion_checkpoint_parametric_unified_hardware_r2.py")
SUBSTRATE_SHA256 = "b2b578f7f38643c7e05bd3413101bd5a6eb4f5224e0c0468ff088997f7602184"
SOURCE_CONTRACT = HW / (
    "contracts/m1180_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")
CANONICAL_LEASE = HW / "results/gpu_profile_lease.lock"
CANONICAL_RESULT = HW / "results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830"
CANONICAL_ATTEMPT = HW / "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
CANONICAL_LOG = HW / "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log"
SOURCE_SCHEMA = "m1180_motion_checkpoint_parametric_unified_capture_source_contract_r2_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1180_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
HAMMER_SCHEMA = "m1181_m1180_motion_unified_capture_source_hammer_r1_v1"
LAUNCH_SCHEMA = "m1180_motion_checkpoint_parametric_unified_capture_launch_r1_v1"
LAUNCH_STATUS = "M1175_AND_M1181_BOUND__ONE_M1180_GPU_RUN_AUTHORIZED"
ATTEMPT_TOKEN = "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1180_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"


class M1180Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1180Error(message)


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
        raise M1180Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise M1180Error("non-standard JSON token: " + token)
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


def load_substrate() -> Any:
    regular(SUBSTRATE_PATH, "sealed technical substrate")
    require(sha256(SUBSTRATE_PATH) == SUBSTRATE_SHA256,
            "sealed technical substrate SHA drift")
    spec = importlib.util.spec_from_file_location(
        "m1180_sealed_technical_substrate", SUBSTRATE_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot import sealed technical substrate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_substrate()
R1 = BASE.R1
canonical_write_double_seal = BASE.canonical_write_double_seal
canonical_verify_double_seal = BASE.canonical_verify_double_seal
StrictWriter = BASE.StrictWriter
StrictAttentionWriter = BASE.StrictAttentionWriter
make_strict_attention_writer = BASE.make_strict_attention_writer
frozen_inventory = BASE.frozen_inventory
validate_m1175 = BASE.validate_m1175
validate_fixed_samples = BASE.validate_fixed_samples
CATEGORIES = BASE.CATEGORIES
C1_TARGETS = BASE.C1_TARGETS
DECODER_TARGETS = BASE.DECODER_TARGETS
ATTENTION_ALIASES = BASE.ATTENTION_ALIASES


def validate_m1181_hammer(contract: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    entry = contract["inputs"]["m1180_source_hammer"]
    path = ROOT / entry["path"]
    require(path.is_relative_to(HW / "reviews"),
            "M1181 source hammer must be under reviews")
    rows = canonical_verify_double_seal(
        path, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "M1181 review member SHA mismatch")
    review = strict_json(path / "review.json")
    require(review.get("schema") == HAMMER_SCHEMA and review.get("status") == "PASS",
            "M1181 semantic admission mismatch")
    require(review.get("source_sha256") == sha256(Path(__file__).resolve()) and
            review.get("contract_sha256") == sha256(SOURCE_CONTRACT) and
            review.get("test_sha256") == policy["test_sha256"],
            "M1181 does not bind exact M1180 artifacts")
    require(review.get("authorization", {}).get("production_release") is True,
            "M1181 does not authorize release authoring")
    return review


def load_technical_policy(policy: dict[str, Any]) -> dict[str, Any]:
    entry = policy["sealed_technical_policy"]
    path = ROOT / entry["path"]
    regular(path, "sealed technical policy")
    require(sha256(path) == entry["sha256"],
            "sealed technical policy SHA drift")
    technical = strict_json(path)
    require(technical.get("claim_boundary", {}).get("production_authorized") is False and
            technical.get("claim_boundary", {}).get("source_only") is True,
            "sealed technical policy is not a source-only non-authority")
    return technical


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == LAUNCH_SCHEMA and
            contract.get("status") == LAUNCH_STATUS,
            "source-only or unhammered M1180 contract cannot launch")
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "canonical M1180 source policy mismatch")
    technical = load_technical_policy(policy)
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "launch contract path mismatch")
    require(contract["inputs"]["launcher"]["sha256"] == sha256(Path(__file__).resolve()) and
            contract["inputs"]["source_contract"]["path"] ==
            str(SOURCE_CONTRACT.relative_to(ROOT)) and
            contract["inputs"]["source_contract"]["sha256"] == sha256(SOURCE_CONTRACT),
            "launch source/source-contract identity mismatch")
    require(contract["gpu_ownership"]["lease_path"] ==
            str(CANONICAL_LEASE.relative_to(ROOT)),
            "launch contract cannot redirect canonical GPU lease")
    require(ROOT / contract["one_shot"]["attempt_marker"] == CANONICAL_ATTEMPT and
            ROOT / contract["output"]["path"] == CANONICAL_RESULT and
            ROOT / contract["production_log"]["path"] == CANONICAL_LOG,
            "launch attempt/result/log namespace is not canonical M1180")
    m1175 = validate_m1175()
    hammer = validate_m1181_hammer(contract, policy)
    verified = validate_fixed_samples(contract, technical)
    require(contract["r1_compatible_binding"]["cohort"]["samples"] ==
            contract["cohort"]["samples"],
            "r1 substrate cohort differs from M1180 frozen cohort")
    binding = R1.validate_launch_contract(contract["r1_compatible_binding"], contract_path)
    require(binding["selection"]["selected"]["epoch"] == 29,
            "r1 substrate selected wrong epoch")
    return {**binding, "m1175": m1175, "m1180_source_hammer": hammer,
            "verified_samples": verified, "policy": technical}


def run_capture(contract: dict[str, Any], binding: dict[str, Any]) -> Path:
    inventory = frozen_inventory(binding["policy"])
    StrictWriter.EXPECTED = inventory
    R1.UnifiedHookWriter = StrictWriter
    R1.write_double_seal = canonical_write_double_seal
    R1.verify_double_seal = canonical_verify_double_seal
    original_load_source = R1.load_source
    def strict_load(name: str, path: Path, expected_sha: str) -> Any:
        module = original_load_source(name, path, expected_sha)
        if name == "m1174_bit_writer":
            module.AttentionBitTraceWriter = make_strict_attention_writer(
                module.AttentionBitTraceWriter)
        return module
    R1.load_source = strict_load
    try:
        output = R1.run_capture(contract["r1_compatible_binding"], binding)
        require(output == CANONICAL_RESULT, "substrate returned noncanonical M1180 result")
        return output
    finally:
        R1.load_source = original_load_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path.is_relative_to(ROOT),
            "launch contract must be inside repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    require(not os.path.lexists(CANONICAL_ATTEMPT),
            "fresh canonical M1180 attempt marker required")
    with R1.exclusive_gpu_lease(CANONICAL_LEASE):
        descriptor = os.open(CANONICAL_ATTEMPT,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output = run_capture(contract, binding)
    canonical_verify_double_seal(output)
    print(PASS_TOKEN + " " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
