#!/usr/bin/env python3
"""M1329 minimal one-shot production runner for sealed M1327."""
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


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1327_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1327_motion_ep34_consumed_namespace_bridge_r1.py")
M1327_SOURCE_SHA256 = "2ab5024a11a81f7bb3ed75956114cc95e07dbe0782328414f2bd3c79342c3ac9"
M1327_TEST = HW / "tests/test_m1327_motion_ep34_consumed_namespace_bridge.py"
M1327_TEST_SHA256 = "3c37f5cd2dbf7611e2c984bcad61d65a2492004db44fcc85b20aa678c7fc1dcf"
M1327_CONTRACT = HW / (
    "contracts/m1327_motion_ep34_consumed_namespace_bridge_source_contract_"
    "r1_20260831.json")
M1327_CONTRACT_SHA256 = "03aca58a422bdfd080b82ea79429948bfb6a04ef4ba1b3b4d6e52e2f75214330"
M1327_AUTHOR_ROOT = HW / (
    "reviews/m1327_motion_ep34_consumed_namespace_bridge_source_author_r1_20260831")
M1327_AUTHOR_ENTRY = {
    "path": str(M1327_AUTHOR_ROOT.relative_to(ROOT)),
    "manifest_sha256": "55b8bd6ce0177637dfa1889b6d4291d32d50bf545a69af1c7a06f133f278fda0",
    "outer_file_sha256": "14a82a5ffd941bd1533906b2488d8202cd6caf93d903f7fbaf354dc9a5665000",
    "receipt_sha256": "e5d3d58653d5dac54e6d3e07d6a905dce70243dac29a38b7e2791318a6f031bc",
}
M1328_ROOT = HW / (
    "reviews/m1328_m1327_consumed_namespace_bridge_blind_hammer_r1_20260831")
M1328_ENTRY = {
    "path": str(M1328_ROOT.relative_to(ROOT)),
    "manifest_sha256": "92a9ad1b3893061b030b3505a61f61d247c879e86e6afda7d3fdb3b470623f58",
    "outer_file_sha256": "cc7b8a7e8c47e1b19c9079064d0bbd548d6688353b20f34328ab5fcdebbb0cd4",
    "review_sha256": "657639d89b19a73eeaf2e6d95abe14a82eed3bf77888668e12797d4515cde103",
}
RUNTIME_CONTRACT = HW / (
    "contracts/m1327_motion_ep34_consumed_namespace_bridge_production_launch_"
    "r1_20260831.json")
RUNTIME_CONTRACT_SHA256 = "10c1f9ef06976846ee39f88efbb5c5df1e8bcd6f1d9db4542ecc09b43aae72d7"
SOURCE_CONTRACT = HW / (
    "contracts/m1329_m1327_motion_ep34_production_runner_source_contract_"
    "r1_20260831.json")
RELEASE = HW / (
    "contracts/m1329_m1327_motion_ep34_production_release_r1_20260831.json")
TEST = HW / "tests/test_run_m1329_m1327_motion_ep34_production_one_shot.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CANONICAL_RESULT = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
CANONICAL_ATTEMPT = HW / (
    "results/.m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831."
    "production.log")
ATTEMPT_TOKEN = b"M1329_M1327_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
SOURCE_SCHEMA = "m1329_m1327_motion_ep34_production_runner_source_r1_v1"
SOURCE_STATUS = "SOURCE_BOUND__EXACT_M1328_AND_RELEASE_REQUIRED__NO_EXECUTION_BY_SELF_CHECK"
RELEASE_SCHEMA = "m1329_m1327_motion_ep34_production_release_r1_v1"
RELEASE_STATUS = "EXACT_M1327_M1328_BOUND__ROOT_ONE_REMOTE_RUN__NO_RETRY"
M1328_SCHEMA = "m1328_m1327_consumed_namespace_bridge_blind_hammer_review_r1_v1"
M1328_STATUS = "PASS_M1328_M1327_SOURCE_HAMMER__MINIMAL_RELEASE_AUTHORING_ALLOWED"


class M1329Error(RuntimeError): pass


def require(ok: bool, message: str) -> None:
    if not ok: raise M1329Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    observed = path.lstat()
    require(stat.S_ISREG(observed.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, "JSON root must be object")
    return value


def _load_m1327():
    regular_exact(M1327_SOURCE, M1327_SOURCE_SHA256, "M1327 source")
    spec = importlib.util.spec_from_file_location("m1329_sealed_m1327", M1327_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1327")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1327 = _load_m1327()


def verify_sealed_entry(root: Path, entry: dict[str, Any], member: str,
                        member_key: str) -> dict[str, Any]:
    rows = M1327.M1249.M1243.verify_double_seal(
        root, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get(member) == entry[member_key], member + " seal mismatch")
    return strict_json(root / member)


def verify_authorities() -> None:
    author = verify_sealed_entry(M1327_AUTHOR_ROOT, M1327_AUTHOR_ENTRY,
                                 "receipt.json", "receipt_sha256")
    require(author.get("status") ==
            "PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION",
            "M1327 author receipt mismatch")
    hammer = verify_sealed_entry(M1328_ROOT, M1328_ENTRY, "review.json", "review_sha256")
    require(hammer.get("schema") == M1328_SCHEMA and hammer.get("status") == M1328_STATUS,
            "M1328 schema/status mismatch")
    require(hammer.get("independence") == {"different_author": True} and
            hammer.get("authorization") == {
                "minimal_release_authoring": True, "production_execution": False,
                "remote": False, "gpu": False}, "M1328 authority mismatch")


def validate_runtime_file() -> dict[str, Any]:
    regular_exact(RUNTIME_CONTRACT, RUNTIME_CONTRACT_SHA256, "M1327 runtime contract")
    runtime = strict_json(RUNTIME_CONTRACT)
    expected = M1327.build_runtime_contract(M1327.strict_json(M1327.M1313_CONTRACT))
    require(runtime == expected and set(runtime) == {"contract_path", "capture", "cohort", "output"},
            "runtime contract is not exact M1327 projection")
    return runtime


def validate_source_contract() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "source contract mismatch")
    require(policy.get("source") == {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                                      "sha256": sha256(Path(__file__).resolve())},
            "runner identity mismatch")
    require(policy.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                    "sha256": sha256(TEST)}, "test identity mismatch")
    require(policy.get("production_authorized") is False, "source contract cannot authorize")
    return policy


def validate_release_static(path: Path = RELEASE) -> dict[str, Any]:
    require(path.resolve() == RELEASE, "only canonical M1329 release allowed")
    validate_source_contract(); verify_authorities(); validate_runtime_file()
    regular_exact(M1327_TEST, M1327_TEST_SHA256, "M1327 test")
    regular_exact(M1327_CONTRACT, M1327_CONTRACT_SHA256, "M1327 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    value = strict_json(RELEASE)
    require(value.get("schema") == RELEASE_SCHEMA and value.get("status") == RELEASE_STATUS,
            "release schema/status mismatch")
    require(value.get("contract_path") == str(RELEASE.relative_to(ROOT)) and
            value.get("authorized_actor") == "root_agent", "release authority mismatch")
    require(value.get("m1327_identity") == {
        "source_sha256": M1327_SOURCE_SHA256, "test_sha256": M1327_TEST_SHA256,
        "contract_sha256": M1327_CONTRACT_SHA256,
        "author": M1327_AUTHOR_ENTRY, "hammer": M1328_ENTRY}, "M1327 release identity mismatch")
    require(value.get("runtime_contract") == {
        "path": str(RUNTIME_CONTRACT.relative_to(ROOT)), "sha256": RUNTIME_CONTRACT_SHA256},
        "runtime release identity mismatch")
    require(value.get("one_shot") == {
        "result": str(CANONICAL_RESULT.relative_to(ROOT)),
        "attempt": str(CANONICAL_ATTEMPT.relative_to(ROOT)),
        "log": str(CANONICAL_LOG.relative_to(ROOT)),
        "runs": 1, "automatic_retry": False}, "one-shot mismatch")
    identity = value.get("release_identity", {})
    regular_exact(Path(__file__).resolve(), identity["source_sha256"], "M1329 runner")
    regular_exact(TEST, identity["test_sha256"], "M1329 test")
    regular_exact(SOURCE_CONTRACT, identity["source_contract_sha256"], "M1329 source contract")
    return value


def ensure_fresh() -> None:
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "M1327 production namespace is not fresh")


def read_only_preflight(path: Path = RELEASE):
    validate_release_static(path); ensure_fresh()
    runtime, _binding = M1327.validate_identity_and_project()
    require(runtime == validate_runtime_file(), "preflight runtime drift")
    ensure_fresh()
    return runtime


def consume_attempt() -> None:
    descriptor = os.open(str(CANONICAL_ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, ATTEMPT_TOKEN); os.fsync(descriptor)
    finally: os.close(descriptor)


def execute_under_lease(preflight_runtime: dict[str, Any], substrate: Any) -> Path:
    with substrate.exclusive_gpu_lease(M1327.M1249.R1.CANONICAL_LEASE):
        runtime, binding = M1327.validate_identity_and_project()
        require(runtime == preflight_runtime == validate_runtime_file(),
                "under-lease runtime projection drift")
        ensure_fresh(); consume_attempt()
        output = M1327.delegate_for_future_release(runtime, binding, substrate)
    M1327.M1249.R1.verify_double_seal(output)
    return Path(output)


def _validate_temp(path: Path) -> None:
    require(path.is_absolute() and path.parent.resolve() == CANONICAL_LOG.parent.resolve(),
            "temporary log directory mismatch")
    require(path.name.startswith(CANONICAL_LOG.name + ".tmp.") and path != CANONICAL_LOG,
            "temporary log name mismatch")


def publish_no_replace(temp: Path) -> None:
    _validate_temp(temp)
    before = temp.lstat()
    require(stat.S_ISREG(before.st_mode) and not temp.is_symlink(), "temp log type mismatch")
    require(not os.path.lexists(str(CANONICAL_LOG)), "canonical log occupied")
    os.link(str(temp), str(CANONICAL_LOG), follow_symlinks=False)
    after = CANONICAL_LOG.lstat()
    require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "log inode mismatch")
    temp.unlink()


def execute_once(temp: Path, release_path: Path = RELEASE) -> Path:
    require(os.geteuid() == 0, "root_agent uid 0 required")
    _validate_temp(temp); require(not os.path.lexists(str(temp)), "temp log occupied")
    runtime = read_only_preflight(release_path)
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    out, err = os.dup(1), os.dup(2)
    try:
        os.dup2(fd, 1); os.dup2(fd, 2)
        substrate = M1327.M1249.R1.load_substrate()
        output = execute_under_lease(runtime, substrate)
        print("PASS_M1329_M1327_ONE_SHOT " + str(output), flush=True); os.fsync(fd)
    finally:
        os.dup2(out, 1); os.dup2(err, 2)
        os.close(out); os.close(err); os.close(fd)
    publish_no_replace(temp)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--temporary-log", type=Path)
    args = parser.parse_args()
    if args.preflight:
        require(args.temporary_log is None, "preflight cannot create log")
        read_only_preflight(args.release.resolve()); print("PASS_M1329_READ_ONLY_PREFLIGHT")
        return 0
    require(args.temporary_log is not None, "run requires temporary log")
    execute_once(args.temporary_log.resolve(), args.release.resolve())
    return 0


if __name__ == "__main__": raise SystemExit(main())
