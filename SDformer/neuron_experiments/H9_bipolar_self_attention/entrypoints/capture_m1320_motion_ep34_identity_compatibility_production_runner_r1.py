#!/opt/conda/envs/sdformerflow/bin/python
"""M1320 exact-closed, one-shot production runner for sealed M1319.

The read-only preflight revalidates the exact M1319 source policy, the sealed
different-author M1320 hammer, exact M1313/M1314 and all M1249 canonical
namespaces.  Production is restricted to uid 0 and invokes M1319 exactly once.
There is no retry path.  Output is first written to a fresh private temporary
log and is published to the canonical log only after success by an atomic hard
link which cannot replace an existing pathname.
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
M1319_SOURCE = Path(__file__).with_name(
    "capture_m1319_motion_ep34_identity_compatibility_successor_r1.py")
M1319_SOURCE_SHA256 = "84a43559c408fcdb0f02a6cbbf76fc2d062d1749224b2302bffd79af609698f2"
M1319_TEST = HW / "tests/test_m1319_m1249_ep34_identity_compatibility_successor.py"
M1319_TEST_SHA256 = "aa5266f8159e0e9d1fe8c5964a7eabd5a0a911c7d349d614e0853a74f05c67d5"
M1319_CONTRACT = HW / (
    "contracts/m1319_m1249_ep34_identity_compatibility_successor_"
    "source_contract_r1_20260831.json")
M1319_CONTRACT_SHA256 = "a568ac6a6fb85adeffdcaf3422cfde4d88b6434195018b470a46925b558d0698"
M1320_HAMMER_ROOT = HW / (
    "reviews/m1320_m1319_identity_compatibility_independent_blind_hammer_"
    "r1_20260831")
M1320_HAMMER_ENTRY = {
    "path": str(M1320_HAMMER_ROOT.relative_to(ROOT)),
    "manifest_sha256": "dd6018af910ae7ef7ea92e327060ff445b7bb084992c38132ead4284ae546219",
    "outer_file_sha256": "6c1d8734fc100c1167350171b524e1e8b902484ba97fb7ec66c113c464b77219",
    "review_sha256": "24985a016018117e4adeee0a73123c67a0184706797906e4384d7c28367e6358",
}
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
SOURCE_CONTRACT = HW / (
    "contracts/m1320_motion_ep34_identity_compatibility_production_runner_"
    "source_contract_r1_20260831.json")
TEST = HW / (
    "tests/test_m1320_motion_ep34_identity_compatibility_production_runner.py")
PRODUCTION_RELEASE = HW / (
    "contracts/m1320_motion_ep34_identity_compatibility_production_launch_"
    "release_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SOURCE_SCHEMA = "m1320_motion_ep34_identity_compatibility_production_runner_source_r1_v1"
SOURCE_STATUS = "SOURCE_SEALED__EXACT_RELEASE_ONLY__NO_EXECUTION_BY_SELF_CHECK"
RELEASE_SCHEMA = "m1320_motion_ep34_identity_compatibility_production_launch_release_r1_v1"
RELEASE_STATUS = (
    "EXACT_M1319_M1320_M1313_M1314_BOUND__ROOT_AGENT_ONE_REMOTE_RUN__NO_RETRY")
HAMMER_SCHEMA = "m1320_m1319_identity_compatibility_independent_blind_hammer_r1_v1"
HAMMER_STATUS = (
    "PASS_M1320_M1319_DIFFERENT_AUTHOR_BLIND_HAMMER__"
    "MINIMUM_PRODUCTION_RELEASE_AUTHORING_ALLOWED")
PREFLIGHT_TOKEN = "PASS_M1320_READ_ONLY_PREFLIGHT__NO_GPU_NO_CAPTURE"
PASS_TOKEN = "PASS_M1320_ONE_SHOT_CAPTURE_AND_ATOMIC_LOG_PUBLISH"

CANONICAL_RESULT = HW / (
    "results/m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830")
CANONICAL_ATTEMPT = HW / (
    "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_"
    "20260830.attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_"
    "20260830.production.log")


class M1320Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1320Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise M1320Error("missing " + label) from exc
    require(stat.S_ISREG(observed.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M1320Error("cannot parse " + str(path)) from exc
    require(isinstance(value, dict), str(path) + " must contain a JSON object")
    return value


def _load_m1319():
    regular_exact(M1319_SOURCE, M1319_SOURCE_SHA256, "M1319 source")
    spec = importlib.util.spec_from_file_location("m1320_sealed_m1319", str(M1319_SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load sealed M1319")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1319 = _load_m1319()


def verify_m1320_hammer(entry: Any) -> dict[str, Any]:
    require(entry == M1320_HAMMER_ENTRY, "exact M1320 hammer entry required")
    try:
        rows = M1319.M1249.M1243.verify_double_seal(
            M1320_HAMMER_ROOT, entry["manifest_sha256"], entry["outer_file_sha256"])
    except Exception as exc:
        raise M1320Error(str(exc)) from exc
    require(rows.get("review.json") == entry["review_sha256"],
            "M1320 review member mismatch")
    review = strict_json(M1320_HAMMER_ROOT / "review.json")
    require(review.get("schema") == HAMMER_SCHEMA and review.get("status") == HAMMER_STATUS,
            "M1320 hammer schema/status mismatch")
    require(review.get("independence") == {"different_author": True},
            "M1320 hammer independence mismatch")
    require(review.get("authorization") == {
        "production_release_authoring": True,
        "authorized_actor": "root_agent",
        "remote_capture_runs": 1,
        "automatic_retry": False,
        "exact_M1313": True,
        "exact_M1314": True,
        "temporary_log_required": True,
        "atomic_no_replace_log_publish_required": True,
    }, "M1320 hammer authorization mismatch")
    require(all(review.get("hammer_execution", {}).get(key) is False for key in
                ("remote_write", "remote_python", "gpu", "capture", "eda", "production")),
            "M1320 hammer execution boundary mismatch")
    require(review.get("docs359_sha256") == DOCS359_SHA256,
            "M1320 docs359 pin mismatch")
    return review


def validate_source_contract() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1320 source contract schema/status mismatch")
    source = policy.get("source")
    test = policy.get("test")
    require(isinstance(source, dict) and set(source) == {"path", "sha256"} and
            source["path"] == str(Path(__file__).resolve().relative_to(ROOT)),
            "M1320 source identity mismatch")
    require(isinstance(test, dict) and set(test) == {"path", "sha256"} and
            test["path"] == str(TEST.relative_to(ROOT)), "M1320 test identity mismatch")
    regular_exact(Path(__file__).resolve(), source["sha256"], "M1320 runner")
    regular_exact(TEST, test["sha256"], "M1320 test")
    require(policy.get("predecessor") == {
        "path": str(M1319_SOURCE.relative_to(ROOT)), "sha256": M1319_SOURCE_SHA256},
        "M1319 source-contract predecessor mismatch")
    require(policy.get("blind_hammer") == M1320_HAMMER_ENTRY,
            "M1320 source-contract hammer mismatch")
    require(policy.get("production_authorized_by_source_contract") is False,
            "source contract must not independently authorize production")
    return policy


def _exact_identity(value: Any, path: Path, expected: str, label: str) -> None:
    require(value == {"path": str(path.relative_to(ROOT)), "sha256": expected},
            label + " identity mismatch")
    regular_exact(path, expected, label)


def validate_release_static(release_path: Path = PRODUCTION_RELEASE) -> dict[str, Any]:
    require(release_path.resolve() == PRODUCTION_RELEASE,
            "only the exact canonical M1320 production release is allowed")
    validate_source_contract()
    M1319.validate_source_policy()
    verify_m1320_hammer(M1320_HAMMER_ENTRY)
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs/359")
    release = strict_json(PRODUCTION_RELEASE)
    required = {"schema", "status", "contract_path", "authorized_actor",
                "release_identity", "inputs", "one_shot", "production_log",
                "claim_boundary"}
    require(set(release) == required, "M1320 release top-level keys mismatch")
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == RELEASE_STATUS,
            "M1320 release schema/status mismatch")
    require(release.get("contract_path") == str(PRODUCTION_RELEASE.relative_to(ROOT)),
            "M1320 release canonical path mismatch")
    require(release.get("authorized_actor") == "root_agent",
            "only root_agent is authorized")
    identity = release.get("release_identity")
    require(isinstance(identity, dict) and set(identity) == {
        "source_path", "source_sha256", "test_path", "test_sha256",
        "source_contract_path", "source_contract_sha256"},
        "M1320 release identity keyset mismatch")
    regular_exact(Path(__file__).resolve(), identity["source_sha256"], "M1320 runner")
    regular_exact(TEST, identity["test_sha256"], "M1320 test")
    regular_exact(SOURCE_CONTRACT, identity["source_contract_sha256"],
                  "M1320 source contract")
    require(identity["source_path"] == str(Path(__file__).resolve().relative_to(ROOT)) and
            identity["test_path"] == str(TEST.relative_to(ROOT)) and
            identity["source_contract_path"] == str(SOURCE_CONTRACT.relative_to(ROOT)),
            "M1320 release identity paths mismatch")
    inputs = release.get("inputs")
    require(isinstance(inputs, dict) and set(inputs) == {
        "m1319_source", "m1319_test", "m1319_source_contract",
        "m1320_blind_hammer", "m1313_contract", "m1314_blind_hammer"},
        "M1320 release inputs mismatch")
    _exact_identity(inputs["m1319_source"], M1319_SOURCE, M1319_SOURCE_SHA256,
                    "M1319 source")
    _exact_identity(inputs["m1319_test"], M1319_TEST, M1319_TEST_SHA256, "M1319 test")
    _exact_identity(inputs["m1319_source_contract"], M1319_CONTRACT,
                    M1319_CONTRACT_SHA256, "M1319 contract")
    require(inputs["m1320_blind_hammer"] == M1320_HAMMER_ENTRY,
            "M1320 hammer release entry mismatch")
    _exact_identity(inputs["m1313_contract"], M1313_CONTRACT, M1313_CONTRACT_SHA256,
                    "M1313 contract")
    require(inputs["m1314_blind_hammer"] == M1314_ENTRY,
            "M1314 release entry mismatch")
    require(release.get("one_shot") == {
        "attempt_marker": str(CANONICAL_ATTEMPT.relative_to(ROOT)),
        "remote_capture_runs": 1, "automatic_retry": False,
        "result_path": str(CANONICAL_RESULT.relative_to(ROOT)),
    }, "M1320 one-shot namespace/policy mismatch")
    require(release.get("production_log") == {
        "canonical_path": str(CANONICAL_LOG.relative_to(ROOT)),
        "temporary_log_required": True,
        "atomic_no_replace_publish": "hard_link_then_unlink_temp",
        "publish_only_after_capture_success": True,
    }, "M1320 production-log policy mismatch")
    require(release.get("claim_boundary") == {
        "capture_complete_only_after_double_seal": True,
        "paper_metric": False, "hardware_speedup": False,
        "system_speedup": False, "energy": False, "ppa": False,
    }, "M1320 release claim boundary mismatch")
    return release


def ensure_fresh_namespaces() -> None:
    for path, label in ((CANONICAL_RESULT, "result"), (CANONICAL_ATTEMPT, "attempt"),
                        (CANONICAL_LOG, "production log")):
        require(not os.path.lexists(str(path)), "M1249 canonical " + label + " is occupied")


def read_only_preflight(release_path: Path = PRODUCTION_RELEASE) -> dict[str, Any]:
    """Read/hash/verify only.  No lease, attempt, GPU, capture, log or publication."""
    release = validate_release_static(release_path)
    ensure_fresh_namespaces()
    _contract, binding = M1319.validate_exact_m1313_m1314(M1313_CONTRACT, M1314_ENTRY)
    ensure_fresh_namespaces()
    return {"release_status": release["status"],
            "binding_projection": binding["identity"]["m1319_projection"],
            "namespaces_fresh": True}


def _validate_temp_log(path: Path) -> None:
    require(path.is_absolute(), "temporary log path must be absolute")
    require(path.parent.resolve() == CANONICAL_LOG.parent.resolve(),
            "temporary log must share canonical log directory")
    require(path.name.startswith(CANONICAL_LOG.name + ".tmp."),
            "temporary log name must use the canonical .tmp. prefix")
    require(path != CANONICAL_LOG, "temporary log cannot be canonical log")


def publish_temp_log_no_replace(temp_path: Path) -> None:
    """Atomically create canonical log; never replace it; preserve temp on failure."""
    _validate_temp_log(temp_path)
    before = temp_path.lstat()
    require(stat.S_ISREG(before.st_mode) and not temp_path.is_symlink(),
            "temporary log must be a regular non-symlink")
    require(not os.path.lexists(str(CANONICAL_LOG)), "canonical log already exists")
    try:
        os.link(str(temp_path), str(CANONICAL_LOG), follow_symlinks=False)
    except FileExistsError as exc:
        raise M1320Error("canonical log already exists") from exc
    except OSError as exc:
        raise M1320Error("atomic no-replace log publication failed") from exc
    after_temp = temp_path.lstat()
    after_log = CANONICAL_LOG.lstat()
    require((before.st_dev, before.st_ino) == (after_temp.st_dev, after_temp.st_ino) ==
            (after_log.st_dev, after_log.st_ino),
            "published log inode mismatch")
    parent_fd = os.open(str(CANONICAL_LOG.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
        temp_path.unlink()
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def execute_production_once(temp_log: Path,
                            release_path: Path = PRODUCTION_RELEASE) -> Path:
    require(os.geteuid() == 0, "production execution requires root_agent uid 0")
    _validate_temp_log(temp_log)
    require(not os.path.lexists(str(temp_log)), "temporary log namespace is occupied")
    read_only_preflight(release_path)
    descriptor = os.open(str(temp_log), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    success = False
    output: Path | None = None
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        substrate = M1319.M1249.R1.load_substrate()
        output = Path(M1319.execute_once(M1313_CONTRACT, M1314_ENTRY, substrate))
        M1319.M1249.R1.verify_double_seal(output)
        print(PASS_TOKEN + " " + str(output), flush=True)
        os.fsync(descriptor)
        success = True
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(descriptor)
    require(success and output is not None, "one-shot capture did not complete")
    publish_temp_log_no_replace(temp_log)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--release", type=Path, default=PRODUCTION_RELEASE)
    parser.add_argument("--temporary-log", type=Path)
    args = parser.parse_args()
    if args.preflight:
        require(args.temporary_log is None, "preflight cannot create a temporary log")
        read_only_preflight(args.release.resolve())
        print(PREFLIGHT_TOKEN)
        return 0
    require(args.temporary_log is not None, "production run requires --temporary-log")
    output = execute_production_once(args.temporary_log.resolve(), args.release.resolve())
    print(PASS_TOKEN + " " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
