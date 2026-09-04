#!/opt/conda/envs/sdformerflow/bin/python
"""M1233 additive final-selection interface successor for the M1227 capture.

The sealed M1227 implementation remains the sole authority for capture logic,
module populations, attention/payload gates, per-sample forensic snapshots and
final sealing.  M1233 changes only the final-selection admission boundary:
checkpoint and configuration must come from the same ``selected`` object of a
fixed M1234 schema/status, and a separately double-sealed M1237 result hammer
must cross-bind the exact selected candidate/profile/checkpoint/config tuple.

This checked-in revision is source-only.  Import is lazy with respect to torch,
numpy, the model and the M1174 substrate.  No release is authorized here.
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
PREDECESSOR = Path(__file__).with_name(
    "capture_m1227_motion_final_checkpoint_unified_hardware_r1.py"
)
PREDECESSOR_SHA256 = "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b"
SOURCE_CONTRACT = HW / (
    "contracts/m1233_motion_final_checkpoint_unified_capture_selection_interface_"
    "successor_source_contract_r1_20260830.json"
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1233_motion_final_checkpoint_unified_capture_selection_interface_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1233_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
LAUNCH_SCHEMA = "m1233_motion_final_checkpoint_unified_capture_launch_r1_v1"
LAUNCH_STATUS = "M1234_SELECTION_AND_M1237_RESULT_HAMMER_BOUND__ONE_M1233_GPU_RUN_AUTHORIZED"
ALLOWED_SELECTION_SCHEMA = "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1"
ALLOWED_SELECTION_STATUS = (
    "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
    "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"
)
SELECTION_RESULT_HAMMER_SCHEMA = (
    "m1237_m1234_motion_cross_run_final_checkpoint_binder_result_hammer_r1_v1"
)
SELECTION_RESULT_HAMMER_STATUS = (
    "PASS_M1237_M1234_FINAL_SELECTION__HARDWARE_REBIND_RELEASE_AUTHORING_ALLOWED"
)

CANONICAL_RESULT = HW / (
    "results/m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830"
)
CANONICAL_ATTEMPT = HW / (
    "results/.m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "attempt_consumed"
)
CANONICAL_LOG = HW / (
    "results/.m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830."
    "production.log"
)
ATTEMPT_TOKEN = "M1233_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1233_FINAL_CHECKPOINT_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"

CANDIDATE_EPOCH = {
    "legacy_ep29": 29,
    "resume_ep30": 30,
    "resume_ep32": 32,
    "resume_ep34": 34,
}
IDENTITY_KEYS = {"absolute_path", "size_bytes", "mtime_ns", "sha256"}
SELECTED_KEYS = {
    "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
    "profile", "accuracy_metrics", "activity",
}
PROFILE_MINIMUM_KEYS = IDENTITY_KEYS | {
    "samples", "artifact_identity_exact", "load_audit_exact_zero", "module_counts",
}
HAMMER_AUTHORITY_KEYS = {
    "result_path", "selection_member", "selection_sha256", "selection_manifest_sha256",
    "selection_outer_file_sha256", "selection_schema", "selection_status",
    "selected_candidate_id", "selected_epoch", "selected_profile_sha256",
    "selected_checkpoint_sha256", "selected_config_sha256",
}


class M1233Error(RuntimeError):
    """Fail-closed source/selection/result-hammer admission error."""


def require(value: bool, message: str) -> None:
    if not value:
        raise M1233Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_predecessor():
    require(PREDECESSOR.is_file() and not PREDECESSOR.is_symlink(),
            "missing or symlink M1227 predecessor")
    require(sha256(PREDECESSOR) == PREDECESSOR_SHA256, "M1227 predecessor SHA drift")
    spec = importlib.util.spec_from_file_location("m1233_sealed_m1227", str(PREDECESSOR))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1227")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R1 = _load_predecessor()

# Explicit aliases prove that selection admission is the only changed layer.
EXPECTED_STATIC_COUNTS = R1.EXPECTED_STATIC_COUNTS
EXPECTED_LIVE_COUNTS = R1.EXPECTED_LIVE_COUNTS
DEAD_SN_V = R1.DEAD_SN_V
audit_call_matrix = R1.audit_call_matrix
audit_attention_population = R1.audit_attention_population
validate_payload_population = R1.validate_payload_population
atomic_sample_snapshot = R1.atomic_sample_snapshot
final_validate_and_seal = R1.final_validate_and_seal


def regular(path: Path, label: str) -> None:
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1233Error(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            f"{label} must be a non-symlink regular file: {path}")


def strict_json(path: Path) -> dict[str, Any]:
    regular(path, f"JSON input {path}")

    def reject(token: str) -> None:
        raise M1233Error("non-standard JSON token: " + token)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise M1233Error(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def safe_repo_path(relative: str) -> Path:
    value = Path(relative)
    require(value.parts and not value.is_absolute() and ".." not in value.parts,
            "unsafe repository-relative path")
    cursor = ROOT
    for part in value.parts:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink repository component: " + str(cursor))
    return cursor


def verify_double_seal(root: Path, manifest_sha: str, outer_file_sha: str) -> dict[str, str]:
    try:
        return R1.verify_double_seal(root, manifest_sha, outer_file_sha)
    except R1.M1227Error as exc:
        raise M1233Error(str(exc)) from exc


def exact_identity(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == IDENTITY_KEYS,
            f"{label} identity keys mismatch")
    require(type(value["absolute_path"]) is str and value["absolute_path"],
            f"{label} path must be nonempty string")
    require(type(value["size_bytes"]) is int and value["size_bytes"] > 0,
            f"{label} size must be positive exact integer")
    require(type(value["mtime_ns"]) is int and value["mtime_ns"] > 0,
            f"{label} mtime must be positive exact integer")
    require(type(value["sha256"]) is str and len(value["sha256"]) == 64 and
            all(character in "0123456789abcdef" for character in value["sha256"]),
            f"{label} SHA must be lowercase SHA256")
    path = Path(value["absolute_path"])
    require(path.is_absolute(), f"{label} path must be absolute")
    regular(path, label)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
        == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev),
        f"{label} changed while hashing",
    )
    require(after.st_size == value["size_bytes"] and
            after.st_mtime_ns == value["mtime_ns"] and digest == value["sha256"],
            f"{label} identity drift")
    return value


def _verify_selection_hammer(
    entry: dict[str, Any],
    selection_entry: dict[str, Any],
    selection: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    required_entry_keys = {
        "path", "manifest_sha256", "outer_file_sha256", "review_sha256"
    }
    require(isinstance(entry, dict) and set(entry) == required_entry_keys,
            "selection result hammer entry keys mismatch")
    root = safe_repo_path(entry["path"])
    require(root.parent == HW / "reviews", "selection result hammer must be under reviews")
    rows = verify_double_seal(root, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "selection result hammer review member mismatch")
    review = strict_json(root / "review.json")
    require(review.get("schema") == SELECTION_RESULT_HAMMER_SCHEMA,
            "selection result hammer schema mismatch")
    require(review.get("status") == SELECTION_RESULT_HAMMER_STATUS,
            "selection result hammer status mismatch")
    authority = review.get("selection_authority")
    require(isinstance(authority, dict) and set(authority) == HAMMER_AUTHORITY_KEYS,
            "selection result hammer authority keys mismatch")
    expected = {
        "result_path": selection_entry["result_path"],
        "selection_member": selection_entry["selection_member"],
        "selection_sha256": selection_entry["selection_sha256"],
        "selection_manifest_sha256": selection_entry["manifest_sha256"],
        "selection_outer_file_sha256": selection_entry["outer_file_sha256"],
        "selection_schema": ALLOWED_SELECTION_SCHEMA,
        "selection_status": ALLOWED_SELECTION_STATUS,
        "selected_candidate_id": selected["candidate_id"],
        "selected_epoch": selected["epoch"],
        "selected_profile_sha256": selected["profile"]["sha256"],
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
        "selected_config_sha256": selected["configuration"]["sha256"],
    }
    require(authority == expected, "selection result hammer cross-SHA/pair mismatch")
    require(review.get("independence") == {"different_author": True},
            "selection result hammer must assert different-author independence")
    require(review.get("authorization") == {
        "hardware_rebind_release_authoring": True,
        "production_capture": False,
    }, "selection result hammer authorization mismatch")
    return {
        "path": str(root.relative_to(ROOT)),
        "review_sha256": entry["review_sha256"],
        "manifest_sha256": entry["manifest_sha256"],
        "outer_file_sha256": entry["outer_file_sha256"],
    }


def validate_final_selection(
    selection_entry: dict[str, Any], hammer_entry: dict[str, Any]
) -> dict[str, Any]:
    required_selection_entry_keys = {
        "result_path", "manifest_sha256", "outer_file_sha256",
        "selection_member", "selection_sha256",
    }
    require(isinstance(selection_entry, dict) and
            set(selection_entry) == required_selection_entry_keys,
            "final selection entry keys mismatch")
    root = safe_repo_path(selection_entry["result_path"])
    require(root.parent == HW / "results", "final selection must be under results")
    rows = verify_double_seal(
        root, selection_entry["manifest_sha256"], selection_entry["outer_file_sha256"]
    )
    member = selection_entry["selection_member"]
    require(rows.get(member) == selection_entry["selection_sha256"],
            "final selection member SHA mismatch")
    selection = strict_json(root / member)
    require(selection.get("schema") == ALLOWED_SELECTION_SCHEMA,
            "final selection schema mismatch")
    require(selection.get("status") == ALLOWED_SELECTION_STATUS,
            "final selection status mismatch")
    require("configuration" not in selection,
            "top-level configuration is forbidden; selected pair must be atomic")
    selected = selection.get("selected")
    require(isinstance(selected, dict) and set(selected) == SELECTED_KEYS,
            "selected object keys mismatch")
    candidate_id = selected["candidate_id"]
    require(type(candidate_id) is str and candidate_id in CANDIDATE_EPOCH,
            "selected candidate_id is not allowed")
    require(type(selected["epoch"]) is int and
            selected["epoch"] == CANDIDATE_EPOCH[candidate_id],
            "selected candidate/epoch pair mismatch")
    require(type(selected["run_directory"]) is str and selected["run_directory"],
            "selected run directory must be nonempty string")

    checkpoint = exact_identity(selected["checkpoint"], "selected checkpoint")
    configuration = exact_identity(selected["configuration"], "selected configuration")
    profile = selected["profile"]
    require(isinstance(profile, dict) and PROFILE_MINIMUM_KEYS <= set(profile),
            "selected profile identity keys incomplete")
    exact_identity({key: profile[key] for key in IDENTITY_KEYS}, "selected profile")
    require(type(profile["samples"]) is int and profile["samples"] == 825,
            "selected profile samples must be exact integer 825")
    require(profile["artifact_identity_exact"] is True and
            profile["load_audit_exact_zero"] is True,
            "selected profile load/artifact audit is not exact")
    require(profile["module_counts"] == {
        "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12,
    }, "selected profile topology mismatch")

    hammer = _verify_selection_hammer(
        hammer_entry, selection_entry, selection, selected
    )
    return {
        "selection": selection,
        "checkpoint_path": Path(checkpoint["absolute_path"]),
        "config_path": Path(configuration["absolute_path"]),
        "profile_path": Path(profile["absolute_path"]),
        "identity": {
            "candidate_id": candidate_id,
            "epoch": selected["epoch"],
            "profile_sha256": profile["sha256"],
            "profile_size_bytes": profile["size_bytes"],
            "profile_mtime_ns": profile["mtime_ns"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "checkpoint_mtime_ns": checkpoint["mtime_ns"],
            "config_sha256": configuration["sha256"],
            "config_size_bytes": configuration["size_bytes"],
            "config_mtime_ns": configuration["mtime_ns"],
            "selection_sha256": selection_entry["selection_sha256"],
            "selection_result_hammer": hammer,
        },
    }


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == LAUNCH_SCHEMA and contract.get("status") == LAUNCH_STATUS,
            "source-only or unhammered M1233 contract cannot launch")
    require(contract.get("contract_path") == str(contract_path.relative_to(ROOT)),
            "launch contract path mismatch")
    require(contract.get("inputs", {}).get("launcher") == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }, "M1233 launcher identity mismatch")
    require(contract["inputs"].get("source_contract") == {
        "path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "sha256": sha256(SOURCE_CONTRACT),
    }, "M1233 source contract identity mismatch")
    R1.validate_m1224()
    binding = validate_final_selection(
        contract["inputs"]["final_selection_result"],
        contract["inputs"]["final_selection_result_hammer"],
    )
    samples = R1.validate_cohort(contract["cohort"]["samples"])
    require(R1.safe_repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True)
            == CANONICAL_ATTEMPT and
            R1.safe_repo_path(contract["output"]["path"], missing_leaf=True)
            == CANONICAL_RESULT and
            R1.safe_repo_path(contract["production_log"]["path"], missing_leaf=True)
            == CANONICAL_LOG,
            "M1233 fresh namespace mismatch")
    return dict(binding, verified_samples=samples,
                policy=R1.strict_json(R1.SOURCE_CONTRACT))


def run_capture(
    contract: dict[str, Any], binding: dict[str, Any], predecessor=None, substrate=None
):
    """Delegate unchanged capture/population/snapshot behavior to sealed M1227."""
    module = R1 if predecessor is None else predecessor
    original_result = module.CANONICAL_RESULT
    try:
        module.CANONICAL_RESULT = CANONICAL_RESULT
        return module.run_capture(contract, binding, r1=substrate)
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
            not os.path.lexists(str(CANONICAL_LOG)),
            "fresh M1233 namespace required")
    substrate = R1.load_substrate()
    with substrate.exclusive_gpu_lease(R1.CANONICAL_LEASE):
        descriptor = os.open(
            str(CANONICAL_ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
        )
        try:
            os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output = run_capture(contract, binding, predecessor=R1, substrate=substrate)
    R1.verify_double_seal(output)
    print(PASS_TOKEN + " " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
