#!/opt/conda/envs/sdformerflow/bin/python
"""M1325 source-only additive runtime projection over sealed M1319.

M1319 remains the sole identity-binding authority.  This successor adds only
the four-key runtime object required by frozen M1227 and allocates disjoint
M1325 namespaces.  It contains no attempt consumer and no production CLI.
"""
from __future__ import annotations

import argparse
import ast
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
M1319_SOURCE = Path(__file__).with_name(
    "capture_m1319_motion_ep34_identity_compatibility_successor_r1.py")
M1319_SOURCE_SHA256 = "84a43559c408fcdb0f02a6cbbf76fc2d062d1749224b2302bffd79af609698f2"
M1313_CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_"
    "r1_20260831.json")
M1313_CONTRACT_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
M1314_ENTRY = {
    "path": "hw_autoresearch_nts07/reviews/"
            "m1314_m1313_motion_ep34_final_unified_capture_production_launch_"
            "blind_hammer_r1_20260831",
    "manifest_sha256": "1fbd77896e91241df5b1ffa32efdbd76fdc145b5af3823ad79272fc9241db1d5",
    "outer_file_sha256": "44cf8e5f8babf96346878cfbe8efb83929f13fa4c81fe180fd38646b82d3cef2",
    "review_sha256": "26a01134f4089f67ae3c74ca4633939f26d0b3b0d29d5ebf7b31bdb96d0027b6",
}
M1324_ROOT = HW / (
    "reviews/m1324_m1320_pre_gpu_capture_key_failure_forensic_r1_20260831")
M1324_ENTRY = {
    "path": str(M1324_ROOT.relative_to(ROOT)),
    "manifest_sha256": "e324d27d966a4b5c4c7546a75addd94012299bf745e90e58c5f0dfc356c72c33",
    "outer_file_sha256": "53b7bbcefc8c84a4c349d8aac72db978e63f30d723c48aaa0ceab30023af75c3",
    "review_sha256": "54b46c751160fb4bf6b6023dc0855efa2f91137b7d34c54b7ed0e3e75012e981",
}
SOURCE_CONTRACT = HW / (
    "contracts/m1325_motion_ep34_runtime_projection_successor_source_contract_"
    "r1_20260831.json")
TEST = HW / "tests/test_m1325_motion_ep34_runtime_projection_successor.py"
FUTURE_RUNTIME_CONTRACT = HW / (
    "contracts/m1325_motion_ep34_runtime_projection_production_launch_"
    "r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CANONICAL_RESULT = HW / (
    "results/m1325_motion_ep34_unified_hardware_capture_s40_r1_20260831")
CANONICAL_ATTEMPT = HW / (
    "results/.m1325_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1325_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "production.log")

SOURCE_SCHEMA = "m1325_motion_ep34_runtime_projection_successor_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
FORENSIC_SCHEMA = "m1324_m1320_pre_gpu_capture_key_failure_forensic_r1_v1"
FORENSIC_STATUS = (
    "PASS_M1324_FAILURE_FORENSIC__ADDITIVE_SOURCE_AUTHORING_ALLOWED__"
    "OLD_NAMESPACE_FORBIDDEN")
RUNTIME_KEYS = {"contract_path", "capture", "cohort", "output"}
PASS_TOKEN = "PASS_M1325_SOURCE_SELF_CHECK__NO_ATTEMPT_NO_GPU_NO_CAPTURE"


class M1325Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1325Error(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise M1325Error("missing " + label) from exc
    require(stat.S_ISREG(observed.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), str(path) + " must contain JSON object")
    return value


def _load_m1319():
    regular_exact(M1319_SOURCE, M1319_SOURCE_SHA256, "M1319 source")
    spec = importlib.util.spec_from_file_location("m1325_sealed_m1319", str(M1319_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1319")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1319 = _load_m1319()
M1227 = M1319.M1249.M1243.P.R1


def verify_m1324_forensic(entry: Any = M1324_ENTRY) -> dict[str, Any]:
    require(entry == M1324_ENTRY, "exact M1324 forensic required")
    rows = M1319.M1249.M1243.verify_double_seal(
        M1324_ROOT, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "M1324 review member mismatch")
    review = strict_json(M1324_ROOT / "review.json")
    require(review.get("schema") == FORENSIC_SCHEMA and
            review.get("status") == FORENSIC_STATUS,
            "M1324 forensic schema/status mismatch")
    require(review.get("independence") == {"different_author": True},
            "M1324 independence mismatch")
    require(review.get("authorization", {}).get("additive_successor_source_authoring") is True and
            review.get("authorization", {}).get("production_release") is False and
            review.get("authorization", {}).get("old_M1249_attempt_reuse") is False,
            "M1324 authorization boundary mismatch")
    return review


def _chain(node: ast.AST):
    parts = []
    while isinstance(node, ast.Subscript):
        key = node.slice
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        parts.append(key.value)
        node = node.value
    if isinstance(node, ast.Name):
        return tuple([node.id] + list(reversed(parts)))
    return None


def frozen_m1227_direct_contract_keys() -> set[str]:
    source = Path(M1227.__file__).resolve()
    regular_exact(source, "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b",
                  "frozen M1227")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "run_capture"]
    require(len(functions) == 1, "M1227 run_capture function count mismatch")
    keys = set()
    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.Subscript):
            continue
        value = _chain(node)
        if value and len(value) >= 2 and value[0] == "contract":
            keys.add(value[1])
    require(keys == RUNTIME_KEYS, "M1227 direct contract keyset drift")
    return keys


def build_runtime_contract(m1313_contract: dict[str, Any],
                           contract_path: Path = FUTURE_RUNTIME_CONTRACT) -> dict[str, Any]:
    regular_exact(M1313_CONTRACT, M1313_CONTRACT_SHA256, "M1313 contract")
    require(m1313_contract == strict_json(M1313_CONTRACT), "only exact M1313 content allowed")
    require(contract_path.resolve() == FUTURE_RUNTIME_CONTRACT,
            "only future canonical M1325 runtime contract path allowed")
    runtime = {
        "contract_path": str(FUTURE_RUNTIME_CONTRACT.relative_to(ROOT)),
        "capture": {"attention_windows_per_call": 100},
        "cohort": copy.deepcopy(m1313_contract["cohort"]),
        "output": {"path": str(CANONICAL_RESULT.relative_to(ROOT))},
    }
    validate_runtime_contract(runtime, m1313_contract)
    return runtime


def validate_runtime_contract(runtime: Any, m1313_contract: dict[str, Any]) -> None:
    require(frozen_m1227_direct_contract_keys() == RUNTIME_KEYS,
            "frozen M1227 key audit failed")
    require(isinstance(runtime, dict) and set(runtime) == RUNTIME_KEYS,
            "runtime projection must contain exactly four keys")
    require(runtime["contract_path"] == str(FUTURE_RUNTIME_CONTRACT.relative_to(ROOT)),
            "runtime contract path mismatch")
    require(runtime["capture"] == {"attention_windows_per_call": 100},
            "runtime capture policy mismatch")
    require(runtime["cohort"] == m1313_contract["cohort"], "runtime cohort drift")
    require(runtime["output"] == {"path": str(CANONICAL_RESULT.relative_to(ROOT))},
            "runtime output mismatch")


def validate_identity_and_project() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only identity validation plus pure projection; no lease or attempt."""
    contract, binding = M1319.validate_exact_m1313_m1314(M1313_CONTRACT, M1314_ENTRY)
    return build_runtime_contract(contract), binding


def delegate_for_future_release(runtime: dict[str, Any], binding: dict[str, Any], substrate: Any):
    """Future-release hook; it owns neither attempt consumption nor authorization."""
    validate_runtime_contract(runtime, strict_json(M1313_CONTRACT))
    require(isinstance(binding, dict) and
            {"policy", "verified_samples", "identity", "selection",
             "checkpoint_path", "config_path"} <= set(binding),
            "M1319 binding incomplete")
    original = M1319.M1249.CANONICAL_RESULT
    try:
        M1319.M1249.CANONICAL_RESULT = CANONICAL_RESULT
        output = M1319.M1249.run_capture(runtime, binding, substrate=substrate)
    finally:
        M1319.M1249.CANONICAL_RESULT = original
    require(Path(output) == CANONICAL_RESULT, "capture chain returned non-M1325 output")
    return Path(output)


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1325 source policy mismatch")
    source = policy.get("source")
    test = policy.get("test")
    require(source == {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                       "sha256": sha256(Path(__file__).resolve())},
            "M1325 source identity mismatch")
    require(test == {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1325 test identity mismatch")
    require(policy.get("predecessor") == {
        "path": str(M1319_SOURCE.relative_to(ROOT)), "sha256": M1319_SOURCE_SHA256},
        "M1319 predecessor mismatch")
    require(policy.get("failure_forensic") == M1324_ENTRY, "M1324 policy entry mismatch")
    require(policy.get("production_authorized") is False, "source policy cannot authorize production")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1325 is source-only; production CLI is forbidden")
    validate_source_policy()
    verify_m1324_forensic()
    frozen_m1227_direct_contract_keys()
    build_runtime_contract(strict_json(M1313_CONTRACT))
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "M1325 namespace is not fresh")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
