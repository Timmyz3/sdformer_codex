#!/opt/conda/envs/sdformerflow/bin/python
"""M1327 source-only bridge for one permanently consumed M1249 attempt.

M1325 projected the correct four runtime keys, but its exact M1319 validation
still called M1249 ``ensure_fresh_namespaces``.  The old M1249 attempt is
sealed as permanently consumed by the failed M1320 run, so that check can
never pass on the real remote host.

This additive successor does not copy or weaken the identity validator.  Only
while the unchanged M1319 exact validation runs, it temporarily replaces that
one freshness callback with a fail-closed, read-only proof of the exact old
failure state: exact non-writable attempt token, absent old result/log, and the
sealed zero-byte temporary-log evidence.  ``finally`` restores the original
callback.  It then builds the same minimal runtime object in fresh M1327
namespaces.  This file has no attempt consumer and no production CLI.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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
M1325_SOURCE = Path(__file__).with_name(
    "capture_m1325_motion_ep34_runtime_projection_successor_r1.py")
M1325_SOURCE_SHA256 = "d3aba86b9003f1ee3cba2b1f81ff02ab8b43e7f5ca7bd56a18ba1c265ab76000"
M1326_ROOT = HW / (
    "reviews/m1326_m1325_motion_ep34_runtime_projection_blind_hammer_r1_20260831")
M1326_ENTRY = {
    "path": str(M1326_ROOT.relative_to(ROOT)),
    "manifest_sha256": "ecda13539273a91b9bb8dc9f677fbe501c0aab215750e4f12beca561c1a40c89",
    "outer_file_sha256": "351daabe6bd28005a06b6019c6e7a05d66a130b9c218241d709ccbb40c17067a",
    "review_sha256": "62650776c28eb43d050a4bcbed4d7d1058b4ca0bba2a022765c9c46b01705935",
}
M1326_SCHEMA = "m1326_m1325_motion_ep34_runtime_projection_blind_hammer_r1_v1"
M1326_STATUS = "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
M1313_CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")
M1313_CONTRACT_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
M1314_ENTRY = {
    "path": "hw_autoresearch_nts07/reviews/"
            "m1314_m1313_motion_ep34_final_unified_capture_production_launch_"
            "blind_hammer_r1_20260831",
    "manifest_sha256": "1fbd77896e91241df5b1ffa32efdbd76fdc145b5af3823ad79272fc9241db1d5",
    "outer_file_sha256": "44cf8e5f8babf96346878cfbe8efb83929f13fa4c81fe180fd38646b82d3cef2",
    "review_sha256": "26a01134f4089f67ae3c74ca4633939f26d0b3b0d29d5ebf7b31bdb96d0027b6",
}
FORENSIC_ROOT = HW / "results/m1320_remote_failed_attempt_forensic_r1_20260831"
FORENSIC_ATTEMPT = FORENSIC_ROOT / "attempt_consumed"
FORENSIC_ATTEMPT_SHA256 = "9be7c7f0db51d15310fcd43698b502e49fec5f5d7710b91ab0b345481fd6b737"
FORENSIC_TEMP_LOG = FORENSIC_ROOT / "temp.log"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SOURCE_CONTRACT = HW / (
    "contracts/m1327_motion_ep34_consumed_namespace_bridge_source_contract_r1_20260831.json")
TEST = HW / "tests/test_m1327_motion_ep34_consumed_namespace_bridge.py"
FUTURE_RUNTIME_CONTRACT = HW / (
    "contracts/m1327_motion_ep34_consumed_namespace_bridge_production_launch_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CANONICAL_RESULT = HW / (
    "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831")
CANONICAL_ATTEMPT = HW / (
    "results/.m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "production.log")

SOURCE_SCHEMA = "m1327_motion_ep34_consumed_namespace_bridge_source_r1_v1"
SOURCE_STATUS = (
    "SOURCE_ONLY__M1326_P0_CLOSED__DIFFERENT_AUTHOR_HAMMER_AND_RELEASE_REQUIRED__NO_GPU")
RUNTIME_KEYS = {"contract_path", "capture", "cohort", "output"}
PASS_TOKEN = "PASS_M1327_SOURCE_SELF_CHECK__NO_ATTEMPT_NO_GPU_NO_CAPTURE"


class M1327Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1327Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1327Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1327Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _load_m1325():
    regular_exact(M1325_SOURCE, M1325_SOURCE_SHA256, "sealed failed M1325 source")
    spec = importlib.util.spec_from_file_location("m1327_sealed_m1325", M1325_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load sealed M1325")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1325 = _load_m1325()
M1319 = M1325.M1319
M1249 = M1319.M1249


def verify_m1326_failure(entry: Any = M1326_ENTRY) -> dict[str, Any]:
    require(entry == M1326_ENTRY, "exact M1326 failure hammer required")
    rows = M1249.M1243.verify_double_seal(
        M1326_ROOT, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "M1326 review member mismatch")
    review = strict_json(M1326_ROOT / "review.json")
    require(review.get("schema") == M1326_SCHEMA and review.get("status") == M1326_STATUS and
            review.get("verdict") == "NO_GO_M1325_PRODUCTION_RELEASE",
            "M1326 failure verdict mismatch")
    require(review.get("p0", {}).get("sealed_forensic_old_attempt_consumed") is True and
            review.get("p0", {}).get("sealed_forensic_old_attempt_reuse") is False and
            review.get("authorization", {}).get("additive_successor_source_authoring") is True and
            review.get("authorization", {}).get("production_release") is False,
            "M1326 successor authorization mismatch")
    return review


def verify_old_consumed_failure_state() -> dict[str, Any]:
    """Replace only the impossible old freshness predicate with exact failure proof."""
    forensic = M1325.verify_m1324_forensic()
    expected_token = M1249.ATTEMPT_TOKEN.encode("ascii")
    attempt = M1249.CANONICAL_ATTEMPT
    try:
        mode = attempt.lstat().st_mode
    except FileNotFoundError as error:
        raise M1327Error("old M1249 consumed attempt is missing") from error
    require(stat.S_ISREG(mode) and not attempt.is_symlink(),
            "old M1249 consumed attempt must be regular non-symlink")
    require(mode & 0o222 == 0, "old M1249 consumed attempt must be read-only")
    payload = attempt.read_bytes()
    require(payload == expected_token, "old M1249 attempt token mismatch")
    require(hashlib.sha256(payload).hexdigest() ==
            forensic["failed_execution_evidence"]["attempt_sha256"],
            "old M1249 attempt differs from sealed failure evidence")
    require(not os.path.lexists(str(M1249.CANONICAL_RESULT)),
            "old M1249 result unexpectedly exists")
    require(not os.path.lexists(str(M1249.CANONICAL_LOG)),
            "old M1249 canonical log unexpectedly exists")
    regular_exact(FORENSIC_ATTEMPT, FORENSIC_ATTEMPT_SHA256,
                  "sealed forensic attempt copy")
    require(FORENSIC_ATTEMPT.read_bytes() == expected_token,
            "sealed forensic attempt token drift")
    regular_exact(FORENSIC_TEMP_LOG, EMPTY_SHA256, "sealed zero temporary log")
    require(forensic["failed_execution_evidence"]["empty_temp_log_sha256"] == EMPTY_SHA256,
            "sealed zero temporary-log identity drift")
    return {
        "status": "PASS_EXACT_OLD_CONSUMED_FAILURE_STATE",
        "old_attempt_sha256": hashlib.sha256(payload).hexdigest(),
        "old_attempt_read_only": True, "old_result_absent": True,
        "old_canonical_log_absent": True, "sealed_temp_log_zero": True,
    }


@contextlib.contextmanager
def old_consumed_freshness_bridge() -> Iterator[None]:
    """Narrow hook: patch one callback during exact validation, always restore."""
    original = M1249.ensure_fresh_namespaces
    require(original is not verify_old_consumed_failure_state,
            "M1249 freshness callback already patched")
    M1249.ensure_fresh_namespaces = verify_old_consumed_failure_state
    try:
        yield
    finally:
        M1249.ensure_fresh_namespaces = original


def build_runtime_contract(m1313_contract: dict[str, Any],
                           contract_path: Path = FUTURE_RUNTIME_CONTRACT) -> dict[str, Any]:
    regular_exact(M1313_CONTRACT, M1313_CONTRACT_SHA256, "M1313 contract")
    require(m1313_contract == strict_json(M1313_CONTRACT), "only exact M1313 content allowed")
    require(contract_path.resolve() == FUTURE_RUNTIME_CONTRACT,
            "only future canonical M1327 runtime contract path allowed")
    runtime = {
        "contract_path": str(FUTURE_RUNTIME_CONTRACT.relative_to(ROOT)),
        "capture": {"attention_windows_per_call": 100},
        "cohort": copy.deepcopy(m1313_contract["cohort"]),
        "output": {"path": str(CANONICAL_RESULT.relative_to(ROOT))},
    }
    validate_runtime_contract(runtime, m1313_contract)
    return runtime


def validate_runtime_contract(runtime: Any, m1313_contract: dict[str, Any]) -> None:
    require(M1325.frozen_m1227_direct_contract_keys() == RUNTIME_KEYS,
            "frozen M1227 direct runtime key audit failed")
    require(type(runtime) is dict and set(runtime) == RUNTIME_KEYS,
            "runtime projection must contain exactly four keys")
    require(runtime["contract_path"] == str(FUTURE_RUNTIME_CONTRACT.relative_to(ROOT)),
            "runtime contract path mismatch")
    require(runtime["capture"] == {"attention_windows_per_call": 100},
            "runtime capture100 policy mismatch")
    require(runtime["cohort"] == m1313_contract["cohort"], "runtime cohort drift")
    require(runtime["output"] == {"path": str(CANONICAL_RESULT.relative_to(ROOT))},
            "runtime output mismatch")


def validate_identity_and_project() -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact unchanged identity validation with only old freshness semantics bridged."""
    verify_m1326_failure()
    with old_consumed_freshness_bridge():
        contract, binding = M1319.validate_exact_m1313_m1314(M1313_CONTRACT, M1314_ENTRY)
    require(M1249.ensure_fresh_namespaces is not verify_old_consumed_failure_state,
            "old freshness callback was not restored")
    return build_runtime_contract(contract), binding


def delegate_for_future_release(runtime: dict[str, Any], binding: dict[str, Any], substrate: Any):
    """Future release hook; this source owns no lease or attempt consumption."""
    validate_runtime_contract(runtime, strict_json(M1313_CONTRACT))
    require(type(binding) is dict and
            {"policy", "verified_samples", "identity", "selection",
             "checkpoint_path", "config_path"} <= set(binding),
            "M1319 binding incomplete")
    original = M1249.CANONICAL_RESULT
    try:
        M1249.CANONICAL_RESULT = CANONICAL_RESULT
        output = M1249.run_capture(runtime, binding, substrate=substrate)
    finally:
        M1249.CANONICAL_RESULT = original
    require(Path(output) == CANONICAL_RESULT, "capture chain returned non-M1327 output")
    return Path(output)


def require_fresh_m1327_namespaces() -> None:
    require(len({CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG}) == 3,
            "M1327 namespaces are not pairwise distinct")
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "M1327 namespace is not fresh")


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1327 source policy mismatch")
    require(policy.get("source") == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve())}, "M1327 source identity mismatch")
    require(policy.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        "M1327 test identity mismatch")
    require(policy.get("predecessor") == {
        "path": str(M1325_SOURCE.relative_to(ROOT)), "sha256": M1325_SOURCE_SHA256},
        "M1325 predecessor identity mismatch")
    require(policy.get("m1326_failure_hammer") == M1326_ENTRY,
            "M1326 failure-hammer policy mismatch")
    require(policy.get("production_authorized") is False,
            "source policy cannot authorize production")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1327 is source-only; production CLI is forbidden")
    validate_source_policy()
    verify_m1326_failure()
    M1325.verify_m1324_forensic()
    M1325.frozen_m1227_direct_contract_keys()
    build_runtime_contract(strict_json(M1313_CONTRACT))
    require_fresh_m1327_namespaces()
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
