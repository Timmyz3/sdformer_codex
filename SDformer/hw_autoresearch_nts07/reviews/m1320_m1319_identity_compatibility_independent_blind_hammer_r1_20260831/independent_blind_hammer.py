#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1319_motion_ep34_identity_compatibility_successor_r1.py")
SOURCE_SHA = "84a43559c408fcdb0f02a6cbbf76fc2d062d1749224b2302bffd79af609698f2"
TEST = HW / "tests/test_m1319_m1249_ep34_identity_compatibility_successor.py"
TEST_SHA = "aa5266f8159e0e9d1fe8c5964a7eabd5a0a911c7d349d614e0853a74f05c67d5"
CONTRACT = HW / (
    "contracts/m1319_m1249_ep34_identity_compatibility_successor_source_contract_r1_20260831.json")
CONTRACT_SHA = "a568ac6a6fb85adeffdcaf3422cfde4d88b6434195018b470a46925b558d0698"
AUTHOR_ROOT = HW / (
    "reviews/m1319_m1249_ep34_identity_compatibility_successor_source_author_r1_20260831")
AUTHOR_ENTRY = {
    "manifest_sha256": "07f20665562724e18adabefcf5a64d1185bb0a0cf201ca946a310152ee35ce9a",
    "outer_file_sha256": "83c5f89fce4bb2ee36ffb7018cf8d3b4b76c856df779371f4c310f23976b365b",
    "review_sha256": "f5e9a9f9100547787e40bba050780da61a5f16ae2f185b139c246b2dcad29b8c",
}
OBSERVATION = Path(__file__).with_name("remote_read_only_observation.json")
OBSERVATION_SHA = "8898863c7595a20c07b907876aa9aa0f63e8fdc97e17fffaf09d1099b9ed7e99"
M1313 = HW / "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json"
M1313_SHA = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class HammerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " not regular")
    require(sha(path) == expected, label + " SHA drift")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    regular_exact(root / "SHA256SUMS", manifest_sha, "manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha, "outer seal")
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in rows and "/" not in name, "unsafe/duplicate manifest member")
        rows[name] = digest
        regular_exact(root / name, digest, name)
    population = {path.name for path in root.iterdir()
                  if path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(population == set(rows), "author seal population mismatch")
    return rows


def identity(path: Path) -> dict[str, Any]:
    row = path.lstat()
    return {
        "absolute_path": str(path.resolve()), "size_bytes": row.st_size,
        "mtime_ns": row.st_mtime_ns, "sha256": sha(path), "device": row.st_dev,
        "inode": row.st_ino, "mode": row.st_mode,
    }


def main() -> int:
    checks: list[str] = []
    regular_exact(SOURCE, SOURCE_SHA, "M1319 source")
    regular_exact(TEST, TEST_SHA, "M1319 test")
    regular_exact(CONTRACT, CONTRACT_SHA, "M1319 contract")
    regular_exact(M1313, M1313_SHA, "M1313 contract")
    regular_exact(DOCS359, DOCS359_SHA, "docs359")
    rows = verify_seal(AUTHOR_ROOT, AUTHOR_ENTRY["manifest_sha256"],
                       AUTHOR_ENTRY["outer_file_sha256"])
    require(rows.get("review.json") == AUTHOR_ENTRY["review_sha256"],
            "author review SHA mismatch")
    checks.append("source_test_contract_author_double_seal")

    author_review = json.loads((AUTHOR_ROOT / "review.json").read_text(encoding="utf-8"))
    require(author_review["source_identity"] == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA,
        "contract_path": str(CONTRACT.relative_to(ROOT)), "contract_sha256": CONTRACT_SHA,
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": TEST_SHA},
        "author source identity mismatch")
    require(author_review["execution"] == {
        "remote": False, "gpu": False, "capture": False, "eda": False,
        "production": False}, "author execution boundary mismatch")
    checks.append("author_review_shape_and_boundary")

    completed = subprocess.run(
        [sys.executable, "-m", "unittest",
         "hw_autoresearch_nts07.tests.test_m1319_m1249_ep34_identity_compatibility_successor"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and "Ran 11 tests" in completed.stderr and
            "OK" in completed.stderr, "author tests did not independently pass")
    checks.append("author_tests_11_of_11")

    M = load("m1320_blind_m1319", SOURCE)
    T = load("m1320_blind_m1319_fixture", TEST)
    require(M.LEGACY_KEYS == {"absolute_path", "size_bytes", "mtime_ns", "sha256"} and
            M.EXTENDED_KEYS == M.LEGACY_KEYS | {"device", "inode", "mode"},
            "identity keysets are not exact legacy4+entity3")
    checks.append("exact_legacy4_plus_device_inode_mode")

    with tempfile.TemporaryDirectory(prefix="m1320_entity_") as name:
        path = Path(name) / "entity.bin"
        path.write_bytes(b"m1320-real-stat-sha\n")
        exact = identity(path)
        M.exact_extended_identity(exact, "entity")
        for key, value in (("device", exact["device"] + 1),
                           ("inode", exact["inode"] + 1),
                           ("mode", exact["mode"] ^ 0o100),
                           ("sha256", "0" * 64)):
            attack = dict(exact)
            attack[key] = value
            try:
                M.exact_extended_identity(attack, "entity")
            except M.M1319Error:
                pass
            else:
                raise HammerError(key + " attack accepted")
        extra = dict(exact, extra=True)
        try:
            M.exact_extended_identity(extra, "entity")
        except M.M1319Error:
            pass
        else:
            raise HammerError("extra identity field accepted")
    checks.append("real_stat_sha_and_entity_mutations")

    fixture = T.SelectionFixture()
    original_keys = M.FROZEN_M1233.IDENTITY_KEYS
    original_validator = M.FROZEN_M1233.validate_final_selection
    try:
        def fail_validator(*_args, **_kwargs):
            raise RuntimeError("injected frozen-validator failure")
        M.FROZEN_M1233.validate_final_selection = fail_validator
        try:
            M.compat_validate_final_selection(fixture.selection_entry, fixture.hammer_entry)
        except M.M1319Error:
            pass
        else:
            raise HammerError("injected validator failure was accepted")
        require(M.FROZEN_M1233.IDENTITY_KEYS == original_keys,
                "IDENTITY_KEYS not restored after exception")
    finally:
        M.FROZEN_M1233.validate_final_selection = original_validator
        fixture.close()
    checks.append("identity_keys_finally_restored_on_exception")

    original_hook = M.M1249.M1243.validate_final_selection
    try:
        try:
            with M._m1249_validation_hook():
                require(M.M1249.M1243.validate_final_selection is
                        M.compat_validate_final_selection, "hook not installed")
                raise RuntimeError("injected hook-body failure")
        except RuntimeError:
            pass
        require(M.M1249.M1243.validate_final_selection is original_hook,
                "M1249 validation hook not restored")
    finally:
        M.M1249.M1243.validate_final_selection = original_hook
    checks.append("m1249_hook_finally_restored_on_exception")

    M.verify_m1314(M.M1314_ENTRY)
    checks.append("M1314_exact_recursive_seal")
    policy = M.validate_source_policy()
    require(policy["production_authorized"] is False and
            policy["execution"] == {"remote": False, "gpu": False,
                                     "capture": False, "eda": False},
            "source-only policy promoted production")
    checks.append("source_only_no_gpu_no_production")

    regular_exact(OBSERVATION, OBSERVATION_SHA, "remote observation")
    observation = json.loads(OBSERVATION.read_text(encoding="utf-8"))
    contract = json.loads(M1313.read_text(encoding="utf-8"))
    require(observation["cohort"]["samples"] == contract["cohort"]["samples"] and
            observation["cohort"]["remote_size_sha_exact"] == 40,
            "remote cohort observation mismatch")
    require(observation["canonical_namespaces"] == {
        "result": contract["output"]["path"],
        "attempt": contract["one_shot"]["attempt_marker"],
        "log": contract["production_log"]["path"],
        "result_fresh": True, "attempt_fresh": True, "log_fresh": True},
        "remote namespace observation mismatch")
    expected_artifacts = {
        "checkpoint": (25015239938, 33152, 225504447,
                       "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"),
        "configuration": (26561699333, 33152, 6481,
                          "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"),
        "profile": (28836708109, 33152, 29183,
                    "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"),
    }
    for row in observation["selected_artifacts"]:
        require((row["inode"], row["mode"], row["size_bytes"], row["sha256"]) ==
                expected_artifacts[row["role"]] and row["device"] == 194,
                "remote artifact stat/SHA mismatch")
    checks.append("remote_extended7_40_samples_and_fresh_namespaces")

    for path in (M.M1249.CANONICAL_RESULT, M.M1249.CANONICAL_ATTEMPT,
                 M.M1249.CANONICAL_LOG):
        require(not os.path.lexists(str(path)), "local M1249 namespace occupied")
    checks.append("local_canonical_namespaces_fresh")

    result = {
        "status": "PASS_M1320_M1319_DIFFERENT_AUTHOR_BLIND_HAMMER",
        "checks": checks,
        "checks_passed": len(checks),
        "author_tests": "11/11",
        "remote_samples": "40/40 size+SHA",
        "production_release_authoring": True,
        "production_execution": False,
        "remote_mutation": False,
        "gpu": False,
        "capture": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
