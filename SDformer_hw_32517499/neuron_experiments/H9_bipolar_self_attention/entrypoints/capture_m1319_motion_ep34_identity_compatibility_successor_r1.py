#!/opt/conda/envs/sdformerflow/bin/python
"""M1319 additive identity-compatibility successor for sealed M1249.

M1257 deliberately freezes checkpoint and configuration identities with three
entity fields (device, inode and mode) in addition to the four legacy M1233
fields.  Frozen M1233 rejects that stronger map before checking its values.
This source admits exactly that seven-field shape, verifies all seven fields
against one unchanged regular non-symlink, and temporarily widens only frozen
M1233's identity-key set while its unchanged validator runs.  The selected
artifact, sealed selection result, selection hammer, capture implementation
and M1249 canonical namespaces are not rewritten.

This revision is source-only.  It does not provide a production release and
its CLI cannot launch capture.  A future different-author release must bind
this exact source/test/contract, exact M1313 and exact M1314 before calling
``execute_once``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1249_SOURCE = Path(__file__).with_name(
    "capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py")
M1249_SOURCE_SHA256 = "5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219"
SOURCE_CONTRACT = HW / (
    "contracts/m1319_m1249_ep34_identity_compatibility_successor_"
    "source_contract_r1_20260831.json")
TEST = HW / "tests/test_m1319_m1249_ep34_identity_compatibility_successor.py"
M1313_CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")
M1313_CONTRACT_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
M1314_ROOT = HW / (
    "reviews/m1314_m1313_motion_ep34_final_unified_capture_production_launch_"
    "blind_hammer_r1_20260831")
M1314_ENTRY = {
    "path": str(M1314_ROOT.relative_to(ROOT)),
    "manifest_sha256": "1fbd77896e91241df5b1ffa32efdbd76fdc145b5af3823ad79272fc9241db1d5",
    "outer_file_sha256": "44cf8e5f8babf96346878cfbe8efb83929f13fa4c81fe180fd38646b82d3cef2",
    "review_sha256": "26a01134f4089f67ae3c74ca4633939f26d0b3b0d29d5ebf7b31bdb96d0027b6",
}
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1319_m1249_ep34_identity_compatibility_successor_source_r1_v1"
SOURCE_STATUS = (
    "SOURCE_ONLY__EXACT_M1313_M1314_AND_DIFFERENT_AUTHOR_RELEASE_REQUIRED__"
    "NO_GPU_NO_REMOTE_NO_PRODUCTION")
M1314_SCHEMA = (
    "m1314_m1313_motion_ep34_final_unified_capture_production_launch_"
    "blind_hammer_r1_v1")
M1314_STATUS = (
    "PASS_M1314_M1313_BLIND_HAMMER__ROOT_AGENT_SINGLE_REMOTE_CAPTURE_ONLY__NO_RETRY")

LEGACY_KEYS = {"absolute_path", "size_bytes", "mtime_ns", "sha256"}
ENTITY_KEYS = {"device", "inode", "mode"}
EXTENDED_KEYS = LEGACY_KEYS | ENTITY_KEYS
PROFILE_KEYS = EXTENDED_KEYS | {
    "samples", "artifact_identity_exact", "load_audit_exact_zero", "module_counts",
    "descriptor_rooted_no_symlink_components", "hash_and_parse_same_bytes",
    "immutable_single_read", "post_parse_path_identity_frozen",
}
PASS_TOKEN = "PASS_M1319_SOURCE_SELF_CHECK__NO_PRODUCTION"


class M1319Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1319Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1319Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def _load_m1249():
    regular_exact(M1249_SOURCE, M1249_SOURCE_SHA256, "sealed M1249 source")
    spec = importlib.util.spec_from_file_location("m1319_sealed_m1249", str(M1249_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1249")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1249 = _load_m1249()
FROZEN_M1233 = M1249.M1243.P


def strict_json(path: Path) -> dict[str, Any]:
    try:
        return M1249.strict_json(path)
    except Exception as exc:
        raise M1319Error(str(exc)) from exc


def exact_extended_identity(value: Any, label: str) -> dict[str, Any]:
    """Verify exactly legacy4+device/inode/mode against one stable pathname entity."""
    require(isinstance(value, dict) and set(value) == EXTENDED_KEYS,
            label + " must contain exactly legacy4 plus device/inode/mode")
    require(type(value["absolute_path"]) is str and value["absolute_path"],
            label + " absolute_path must be a nonempty string")
    path = Path(value["absolute_path"])
    require(path.is_absolute(), label + " path must be absolute")
    for key in ("size_bytes", "mtime_ns", "device", "inode", "mode"):
        require(type(value[key]) is int and value[key] >= 0,
                label + " " + key + " must be a nonnegative exact integer")
    require(value["size_bytes"] > 0 and value["mtime_ns"] > 0 and value["inode"] > 0,
            label + " size/mtime/inode must be positive")
    require(type(value["sha256"]) is str and len(value["sha256"]) == 64 and
            all(ch in "0123456789abcdef" for ch in value["sha256"]),
            label + " SHA must be lowercase SHA256")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise M1319Error("missing " + label) from exc
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            label + " must resolve at a regular non-symlink leaf")
    digest = sha256(path)
    after = path.lstat()
    observed = {
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": after.st_mode,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
             before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size,
             after.st_mtime_ns), label + " changed while hashing")
    require(all(value[key] == observed[key] for key in observed),
            label + " entity/stat/SHA identity drift")
    return dict(value)


def _selection_path(selection_entry: dict[str, Any]) -> Path:
    try:
        root = FROZEN_M1233.safe_repo_path(selection_entry["result_path"])
        return root / selection_entry["selection_member"]
    except Exception as exc:
        raise M1319Error(str(exc)) from exc


def validate_extended_selection_before_projection(
        selection_entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Read the still-sealed real selection and validate only the new entity fields."""
    required = {"result_path", "manifest_sha256", "outer_file_sha256",
                "selection_member", "selection_sha256"}
    require(isinstance(selection_entry, dict) and set(selection_entry) == required,
            "final selection entry keys mismatch")
    selection_path = _selection_path(selection_entry)
    try:
        rows = FROZEN_M1233.verify_double_seal(
            selection_path.parent, selection_entry["manifest_sha256"],
            selection_entry["outer_file_sha256"])
    except Exception as exc:
        raise M1319Error(str(exc)) from exc
    require(rows.get(selection_path.name) == selection_entry["selection_sha256"],
            "selection member SHA mismatch before projection")
    selection = strict_json(selection_path)
    selected = selection.get("selected")
    require(isinstance(selected, dict), "selected object missing")
    exact_extended_identity(selected.get("checkpoint"), "selected checkpoint")
    exact_extended_identity(selected.get("configuration"), "selected configuration")
    profile = selected.get("profile")
    require(isinstance(profile, dict) and set(profile) == PROFILE_KEYS,
            "selected profile keyset mismatch")
    exact_extended_identity({key: profile[key] for key in EXTENDED_KEYS},
                            "selected profile")
    require(profile["samples"] == 825 and
            profile["module_counts"] == {
                "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12} and
            all(profile[key] is True for key in (
                "artifact_identity_exact", "load_audit_exact_zero",
                "descriptor_rooted_no_symlink_components", "hash_and_parse_same_bytes",
                "immutable_single_read", "post_parse_path_identity_frozen")),
            "selected profile frozen semantics mismatch")
    return selection_path, selection


@contextlib.contextmanager
def _extended_identity_keyset() -> Iterator[None]:
    """Temporarily widen only the frozen identity shape, then always restore it."""
    original = FROZEN_M1233.IDENTITY_KEYS
    require(original == LEGACY_KEYS, "frozen M1233 legacy identity keyset drift")
    FROZEN_M1233.IDENTITY_KEYS = set(EXTENDED_KEYS)
    try:
        yield
    finally:
        FROZEN_M1233.IDENTITY_KEYS = original


def compat_validate_final_selection(selection_entry: dict[str, Any],
                                    hammer_entry: dict[str, Any]) -> dict[str, Any]:
    """Run frozen M1233 after a narrow, already-verified in-memory projection."""
    _selection_path_value, selection = validate_extended_selection_before_projection(selection_entry)
    with _extended_identity_keyset():
        try:
            binding = FROZEN_M1233.validate_final_selection(selection_entry, hammer_entry)
        except Exception as exc:
            raise M1319Error(str(exc)) from exc
    identity = binding["identity"]
    identity["checkpoint_entity"] = {
        key: selection["selected"]["checkpoint"][key] for key in ENTITY_KEYS}
    identity["config_entity"] = {
        key: selection["selected"]["configuration"][key] for key in ENTITY_KEYS}
    identity["profile_entity"] = {
        key: selection["selected"]["profile"][key] for key in ENTITY_KEYS}
    identity["m1319_projection"] = "extended7_verified_then_frozen_keyset_temporarily_extended"
    return binding


@contextlib.contextmanager
def _m1249_validation_hook() -> Iterator[None]:
    original = M1249.M1243.validate_final_selection
    M1249.M1243.validate_final_selection = compat_validate_final_selection
    try:
        yield
    finally:
        M1249.M1243.validate_final_selection = original


def verify_m1314(entry: Any) -> dict[str, Any]:
    require(entry == M1314_ENTRY, "exact M1314 entry required")
    try:
        rows = M1249.M1243.verify_double_seal(
            M1314_ROOT, entry["manifest_sha256"], entry["outer_file_sha256"])
    except Exception as exc:
        raise M1319Error(str(exc)) from exc
    require(rows.get("review.json") == entry["review_sha256"],
            "M1314 review member mismatch")
    review = strict_json(M1314_ROOT / "review.json")
    require(review.get("schema") == M1314_SCHEMA and review.get("status") == M1314_STATUS,
            "M1314 schema/status mismatch")
    require(review.get("verdict") == "GO_ROOT_AGENT_ONE_REMOTE_M1249_CAPTURE_ONLY",
            "M1314 verdict mismatch")
    require(review.get("independence") == {"different_author": True},
            "M1314 independence mismatch")
    require(review.get("authorization") == {
        "authorized_actor": "root_agent", "production_capture": True,
        "remote_capture_runs": 1, "automatic_retry": False,
        "authorization_transferable": False, "exact_M1313_contract_only": True,
        "exact_canonical_namespaces_only": True,
    }, "M1314 authorization mismatch")
    require(review.get("docs359_sha256") == DOCS359_SHA256,
            "M1314 docs359 pin mismatch")
    return review


def validate_exact_m1313_m1314(m1313_path: Path, m1314_entry: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    m1313_path = m1313_path.resolve()
    require(m1313_path == M1313_CONTRACT, "only exact canonical M1313 is allowed")
    regular_exact(M1313_CONTRACT, M1313_CONTRACT_SHA256, "M1313 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    review = verify_m1314(m1314_entry)
    contract = M1249.strict_json(M1313_CONTRACT)
    with _m1249_validation_hook():
        try:
            binding = M1249.validate_production_launch(contract, M1313_CONTRACT)
        except Exception as exc:
            raise M1319Error(str(exc)) from exc
    require(binding["identity"].get("m1319_projection") ==
            "extended7_verified_then_frozen_keyset_temporarily_extended",
            "M1319 compatibility binding absent")
    return contract, dict(binding, m1314_review_status=review["status"])


def execute_once(m1313_path: Path, m1314_entry: Any, substrate: Any):
    """Future-release hook; same M1249 namespaces, lease, attempt and capture."""
    with substrate.exclusive_gpu_lease(M1249.R1.CANONICAL_LEASE):
        contract, binding = validate_exact_m1313_m1314(m1313_path, m1314_entry)
        M1249.consume_attempt()
        output = M1249.run_capture(contract, binding, substrate=substrate)
    M1249.R1.verify_double_seal(output)
    return output


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1319 source policy mismatch")
    require(policy.get("predecessor") == {
        "path": str(M1249_SOURCE.relative_to(ROOT)), "sha256": M1249_SOURCE_SHA256},
        "M1249 predecessor policy mismatch")
    require(policy.get("production_dependencies", {}).get("M1313") == {
        "path": str(M1313_CONTRACT.relative_to(ROOT)), "sha256": M1313_CONTRACT_SHA256},
        "M1313 policy mismatch")
    require(policy.get("production_dependencies", {}).get("M1314") == M1314_ENTRY,
            "M1314 policy mismatch")
    source = policy.get("source")
    test = policy.get("test")
    require(isinstance(source, dict) and set(source) == {"path", "sha256"} and
            source["path"] == str(Path(__file__).resolve().relative_to(ROOT)),
            "source policy identity mismatch")
    require(isinstance(test, dict) and set(test) == {"path", "sha256"} and
            test["path"] == str(TEST.relative_to(ROOT)), "test policy identity mismatch")
    regular_exact(Path(__file__).resolve(), source["sha256"], "M1319 source")
    regular_exact(TEST, test["sha256"], "M1319 test")
    require(policy.get("production_authorized") is False,
            "source policy must not authorize production")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1319 is source-only; production CLI is forbidden")
    validate_source_policy()
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
