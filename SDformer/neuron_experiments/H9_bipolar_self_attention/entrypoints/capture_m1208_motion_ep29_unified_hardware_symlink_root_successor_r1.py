#!/opt/conda/envs/sdformerflow/bin/python
"""M1208 one-shot ep29 capture successor with one pinned dataset-root link.

M1180 remains immutable failed evidence.  This additive successor changes only
sample-path admission: the repository component ``data/Datasets/DSEC`` may be
one absolute symlink when both its raw target and resolved target equal the
pinned remote dataset root.  Every other component, including every sample
leaf, remains non-symlink.  Importing this module is inert.
"""
from __future__ import annotations

import argparse
import copy
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
PREDECESSOR_PATH = Path(__file__).with_name(
    "capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py")
PREDECESSOR_SHA256 = "f88426c789c99a0d56c34ffaa742b052c73fcbad600c4ecd5797a62e2cf26479"
SOURCE_CONTRACT = HW / (
    "contracts/m1208_motion_ep29_unified_capture_symlink_root_successor_"
    "source_contract_r1_20260830.json")
M1180_LAUNCH_CONTRACT = HW / (
    "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json")
M1180_LAUNCH_SHA256 = "46450015bcdb3b8c0a32ccd7aaba68a78abf923705a133147202283e7bc7220f"
CANONICAL_LEASE = HW / "results/gpu_profile_lease.lock"
CANONICAL_RESULT = HW / "results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830"
CANONICAL_ATTEMPT = HW / (
    "results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
M1180_ATTEMPT = HW / (
    "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT = HW / "results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830"
M1180_LOG = HW / (
    "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
PINNED_LINK_REL = Path("data/Datasets/DSEC")
PINNED_DSEC_ROOT = Path("/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC")
SOURCE_SCHEMA = "m1208_motion_ep29_unified_capture_symlink_root_successor_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1209_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
HAMMER_SCHEMA = "m1209_m1208_motion_ep29_unified_capture_source_hammer_r1_v1"
LAUNCH_SCHEMA = "m1208_motion_ep29_unified_capture_launch_r1_v1"
LAUNCH_STATUS = "M1175_AND_M1209_BOUND__ONE_M1208_GPU_RUN_AUTHORIZED"
ATTEMPT_TOKEN = "M1208_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1208_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"


class M1208Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1208Error(message)


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
        raise M1208Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise M1208Error("non-standard JSON token: " + token)
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


def load_predecessor() -> Any:
    regular(PREDECESSOR_PATH, "sealed M1180 predecessor")
    require(sha256(PREDECESSOR_PATH) == PREDECESSOR_SHA256,
            "sealed M1180 predecessor SHA drift")
    spec = importlib.util.spec_from_file_location("m1208_sealed_m1180", PREDECESSOR_PATH)
    require(spec is not None and spec.loader is not None, "cannot import M1180 predecessor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_predecessor()
R1 = BASE.R1
canonical_write_double_seal = BASE.canonical_write_double_seal
canonical_verify_double_seal = BASE.canonical_verify_double_seal
StrictWriter = BASE.StrictWriter
make_strict_attention_writer = BASE.make_strict_attention_writer
frozen_inventory = BASE.frozen_inventory
validate_m1175 = BASE.validate_m1175
validate_fixed_samples = BASE.validate_fixed_samples


def _resolve_whitelisted_sample(relative_text: str, expected_bytes: int,
                                expected_sha256: str, *, repo_root: Path = ROOT,
                                pinned_root: Path = PINNED_DSEC_ROOT) -> Path:
    relative = Path(relative_text)
    require(not relative.is_absolute() and ".." not in relative.parts and
            relative.parts[:3] == PINNED_LINK_REL.parts and len(relative.parts) > 3,
            "sample must be below exact pinned DSEC repository component")
    require(pinned_root.is_absolute(), "pinned DSEC root must be absolute")

    cursor = repo_root
    for part in PINNED_LINK_REL.parts[:-1]:
        cursor = cursor / part
        require(os.path.lexists(cursor) and not cursor.is_symlink() and cursor.is_dir(),
                "non-whitelisted repository component must be a real directory: " + str(cursor))
    link = repo_root / PINNED_LINK_REL
    require(os.path.lexists(link) and link.is_symlink(),
            "exact DSEC repository component must be a symlink: " + str(link))
    raw_target = os.readlink(link)
    require(raw_target == str(pinned_root), "pinned DSEC raw symlink target drift")
    try:
        resolved_root = link.resolve(strict=True)
        pinned_resolved = pinned_root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise M1208Error("pinned DSEC root is missing or cyclic") from exc
    require(resolved_root == pinned_root and pinned_resolved == pinned_root,
            "pinned DSEC resolved absolute target drift")

    cursor = link
    suffix = relative.parts[len(PINNED_LINK_REL.parts):]
    for index, part in enumerate(suffix):
        cursor = cursor / part
        require(os.path.lexists(cursor), "missing sample path component: " + str(cursor))
        require(not cursor.is_symlink(), "non-whitelisted symlink component rejected: " + str(cursor))
        if index != len(suffix) - 1:
            require(cursor.is_dir(), "sample parent component is not a directory: " + str(cursor))
    regular(cursor, "cohort source leaf")
    try:
        resolved_sample = cursor.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise M1208Error("cohort source cannot be resolved") from exc
    require(resolved_sample.is_relative_to(pinned_root),
            "resolved sample escapes pinned DSEC root")
    require(cursor.stat().st_size == expected_bytes and sha256(cursor) == expected_sha256,
            "cohort source identity drift: " + relative_text)
    return resolved_sample


def selected_samples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = contract["cohort"]["samples"]
    require(len(rows) == 40 and
            [row["global_sample_id"] for row in rows] == list(range(40)),
            "unified cohort must contain ordered global sample ids 0..39")
    observed = []
    for row in rows:
        path = _resolve_whitelisted_sample(
            row["path"], int(row["bytes"]), row["sha256"])
        observed.append({**row, "resolved_path": str(path)})
    return observed


def validate_m1209_hammer(contract: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    entry = contract["inputs"]["m1209_source_hammer"]
    path = ROOT / entry["path"]
    require(path.is_relative_to(HW / "reviews"), "M1209 source hammer must be under reviews")
    rows = canonical_verify_double_seal(
        path, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "M1209 review member SHA mismatch")
    review = strict_json(path / "review.json")
    require(review.get("schema") == HAMMER_SCHEMA and review.get("status") == "PASS",
            "M1209 semantic admission mismatch")
    require(review.get("source_sha256") == sha256(Path(__file__).resolve()) and
            review.get("contract_sha256") == sha256(SOURCE_CONTRACT) and
            review.get("test_sha256") == policy["test_sha256"],
            "M1209 does not bind exact M1208 artifacts")
    require(review.get("authorization", {}).get("production_release") is True,
            "M1209 does not authorize release authoring")
    return review


def validate_prior_m1180_failure(contract: dict[str, Any]) -> None:
    expected = contract["prior_m1180_failure"]
    require(expected == {
        "attempt_marker": str(M1180_ATTEMPT.relative_to(ROOT)),
        "attempt_token": "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE",
        "result_path": str(M1180_RESULT.relative_to(ROOT)),
        "production_log_path": str(M1180_LOG.relative_to(ROOT)),
        "result_absent": True, "production_log_absent": True,
        "automatic_retry": False,
        "failure_boundary": "R1_SELECTED_SAMPLES_REJECTED_PINNED_DSEC_SYMLINK_PRE_GPU",
    }, "M1180 failure binding mismatch")
    regular(M1180_ATTEMPT, "consumed M1180 attempt marker")
    require(M1180_ATTEMPT.read_text(encoding="ascii") ==
            "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n",
            "M1180 attempt token drift")
    require(not os.path.lexists(M1180_RESULT) and not os.path.lexists(M1180_LOG),
            "M1180 failed namespace unexpectedly contains result/log")


def _expected_r1_successor(old: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    expected = copy.deepcopy(old["r1_compatible_binding"])
    expected["contract_path"] = str(contract_path.relative_to(ROOT))
    expected["one_shot"]["attempt_marker"] = str(CANONICAL_ATTEMPT.relative_to(ROOT))
    expected["output"]["path"] = str(CANONICAL_RESULT.relative_to(ROOT))
    return expected


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == LAUNCH_SCHEMA and
            contract.get("status") == LAUNCH_STATUS,
            "source-only or unhammered M1208 contract cannot launch")
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "canonical M1208 source policy mismatch")
    require(contract["inputs"]["launcher"]["sha256"] == sha256(Path(__file__).resolve()) and
            contract["inputs"]["source_contract"] == {
                "path": str(SOURCE_CONTRACT.relative_to(ROOT)),
                "sha256": sha256(SOURCE_CONTRACT)},
            "M1208 launch source/source-contract identity mismatch")
    require(ROOT / contract["one_shot"]["attempt_marker"] == CANONICAL_ATTEMPT and
            ROOT / contract["output"]["path"] == CANONICAL_RESULT and
            ROOT / contract["production_log"]["path"] == CANONICAL_LOG and
            contract["gpu_ownership"]["lease_path"] == str(CANONICAL_LEASE.relative_to(ROOT)),
            "M1208 canonical attempt/result/log/lease mismatch")
    regular(M1180_LAUNCH_CONTRACT, "sealed M1180 launch evidence")
    require(sha256(M1180_LAUNCH_CONTRACT) == M1180_LAUNCH_SHA256,
            "sealed M1180 launch contract SHA drift")
    old = strict_json(M1180_LAUNCH_CONTRACT)
    require(contract["r1_compatible_binding"] == _expected_r1_successor(old, contract_path),
            "M1208 changed technical capture binding beyond disjoint namespaces")
    validate_prior_m1180_failure(contract)
    m1175 = validate_m1175()
    hammer = validate_m1209_hammer(contract, policy)
    technical = BASE.load_technical_policy(strict_json(BASE.SOURCE_CONTRACT))
    verified = validate_fixed_samples(contract, technical)
    binding = R1.validate_launch_contract(contract["r1_compatible_binding"], contract_path)
    require(binding["selection"]["selected"]["epoch"] == 29,
            "M1208 selected wrong checkpoint epoch")
    return {**binding, "m1175": m1175, "m1209_source_hammer": hammer,
            "verified_samples": verified, "policy": technical}


def run_capture(contract: dict[str, Any], binding: dict[str, Any]) -> Path:
    inventory = frozen_inventory(binding["policy"])
    StrictWriter.EXPECTED = inventory
    R1.UnifiedHookWriter = StrictWriter
    R1.write_double_seal = canonical_write_double_seal
    R1.verify_double_seal = canonical_verify_double_seal
    original_load_source = R1.load_source
    original_selected_samples = R1.selected_samples
    def strict_load(name: str, path: Path, expected_sha: str) -> Any:
        module = original_load_source(name, path, expected_sha)
        if name == "m1174_bit_writer":
            module.AttentionBitTraceWriter = make_strict_attention_writer(
                module.AttentionBitTraceWriter)
        return module
    R1.load_source = strict_load
    R1.selected_samples = selected_samples
    try:
        output = R1.run_capture(contract["r1_compatible_binding"], binding)
        require(output == CANONICAL_RESULT, "substrate returned noncanonical M1208 result")
        return output
    finally:
        R1.selected_samples = original_selected_samples
        R1.load_source = original_load_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path.is_relative_to(ROOT), "launch contract must be inside repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    require(not os.path.lexists(CANONICAL_ATTEMPT), "fresh canonical M1208 attempt required")
    with R1.exclusive_gpu_lease(CANONICAL_LEASE):
        descriptor = os.open(CANONICAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
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
