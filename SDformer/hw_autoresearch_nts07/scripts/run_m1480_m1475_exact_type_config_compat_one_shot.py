#!/opt/conda/envs/sdformerflow/bin/python
"""Exact-type successor for the narrow M1475 configuration wrapper."""
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
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_run_m1480_m1475_exact_type_config_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1480_m1475_exact_type_config_compat_source_contract_r1_20260831.json")
M1475_SOURCE = HW / "scripts/run_m1475_m1458_config_content_compat_one_shot.py"
M1475_SOURCE_SHA256 = "2a5104b79e0d6563d8145a1e4ba136c9a2a047963d66d08a5d6b0bde93c5ac06"
M1475_TEST = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
M1475_TEST_SHA256 = "25de303df2883dc450080d7f57c1f64047a349c32610f555cea990d5553ac10b"
M1475_CONTRACT = HW / (
    "contracts/m1475_m1458_config_content_compat_source_contract_r1_20260831.json")
M1475_CONTRACT_SHA256 = "9cb1fd126621f85f7ab6ba4e7c960687ea19c49c85c901e8335a12108d4ab7b2"
M1476_FAIL = HW / (
    "reviews/m1476_m1475_m1458_config_content_compat_source_blind_hammer_r1_20260831")
M1476_REVIEW_SHA256 = "013308a83ca8f9732f9c600562c49d5ff15cb3b35fc65a3ec58230b396d0bd70"
M1476_MANIFEST_SHA256 = "d3f541b213d2a5efe0b2ef9224d3ba9977f4245d7039a4b28ce7b1f3bfa12c1d"
M1476_OUTER_SHA256 = "4f2a7138200de5059ca43031b613a2dc8fb52801340e7497af21309b8cada727"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

BLIND = HW / (
    "reviews/m1481_m1480_m1475_exact_type_config_compat_source_blind_hammer_"
    "r1_20260831")
RELEASE = HW / (
    "contracts/m1482_m1480_m1475_exact_type_config_compat_launch_release_"
    "r1_20260831.json")
FINAL = HW / (
    "reviews/m1483_m1482_m1480_m1475_exact_type_config_compat_final_launch_"
    "hammer_r1_20260831")
ENV_BINDINGS = {
    "M1480_EXPECTED_RUNNER_SHA256": SOURCE,
    "M1480_EXPECTED_BLIND_REVIEW_SHA256": BLIND / "review.json",
    "M1480_EXPECTED_BLIND_MANIFEST_SHA256": BLIND / "SHA256SUMS",
    "M1480_EXPECTED_BLIND_OUTER_SHA256": BLIND / "SHA256SUMS.seal.sha256",
    "M1480_EXPECTED_RELEASE_SHA256": RELEASE,
    "M1480_EXPECTED_FINAL_REVIEW_SHA256": FINAL / "review.json",
    "M1480_EXPECTED_FINAL_MANIFEST_SHA256": FINAL / "SHA256SUMS",
    "M1480_EXPECTED_FINAL_OUTER_SHA256": FINAL / "SHA256SUMS.seal.sha256",
}


class M1480Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1480Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1480Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1480Error("nonfinite JSON " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1475():
    regular_exact(M1475_SOURCE, M1475_SOURCE_SHA256, "M1475 source")
    spec = importlib.util.spec_from_file_location("m1480_sealed_m1475", M1475_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1475")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1475_SOURCE, M1475_SOURCE_SHA256, "M1475 source after import")
    return module


M1475 = load_m1475()


def verify_predecessors() -> None:
    regular_exact(M1475_TEST, M1475_TEST_SHA256, "M1475 test")
    regular_exact(M1475_CONTRACT, M1475_CONTRACT_SHA256, "M1475 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    failed = M1475.M1458.verify_double_seal(
        M1476_FAIL, M1476_REVIEW_SHA256, M1476_MANIFEST_SHA256,
        M1476_OUTER_SHA256)
    require(failed.get("status") ==
            "FAIL_DO_NOT_CITE__M1475_FINAL_AUTHORITY_TYPE_CONFUSION" and
            failed.get("authorization", {}).get("launch") is False and
            failed.get("authorization", {}).get("remote_preflight") is False,
            "M1476 immutable failure mismatch")
    M1475.M1458.verify_prerequisites()


def validate_source_contract() -> None:
    verify_predecessors()
    value = strict_json(CONTRACT)
    require(value.get("status") ==
            "SOURCE_ONLY__M1475_EXACT_TYPE_SUCCESSOR__M1481_REQUIRED__NO_LAUNCH",
            "source contract status mismatch")
    require(value.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "source identity mismatch")
    require(value.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        "test identity mismatch")
    require(value.get("m1476_failure") == {
        "review_sha256": M1476_REVIEW_SHA256,
        "manifest_sha256": M1476_MANIFEST_SHA256,
        "outer_file_sha256": M1476_OUTER_SHA256},
        "M1476 failure contract mismatch")


def external_bindings(environment: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    values = {}
    for name, path in ENV_BINDINGS.items():
        value = environment.get(name, "")
        require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
                "missing/malformed external SHA " + name)
        regular_exact(path, value, name)
        values[name] = value
    require(values["M1480_EXPECTED_RUNNER_SHA256"] == sha256(SOURCE),
            "runner external SHA mismatch")
    return values


def exact_authorization(value: Any, launch: bool) -> None:
    require(type(value) is dict and set(value) == {
        "launch", "runs", "automatic_retry", "controller_restore"},
        "authorization shape mismatch")
    require(value["launch"] is launch and type(value["runs"]) is int and
            value["runs"] == (1 if launch else 0) and
            value["automatic_retry"] is False and
            value["controller_restore"] is False,
            "authorization exact-type mismatch")


def validate_future_authorities(values: dict[str, str]) -> None:
    blind = M1475.M1458.verify_double_seal(
        BLIND, values["M1480_EXPECTED_BLIND_REVIEW_SHA256"],
        values["M1480_EXPECTED_BLIND_MANIFEST_SHA256"],
        values["M1480_EXPECTED_BLIND_OUTER_SHA256"])
    require(blind.get("status") == "PASS_M1480_EXACT_TYPE_CONFIG_COMPAT_SOURCE" and
            blind.get("bindings", {}).get("runner_sha256") == sha256(SOURCE),
            "M1481 blind mismatch")
    exact_authorization(blind.get("authorization"), False)

    release = strict_json(RELEASE)
    require(release.get("status") ==
            "AUTHORIZE_ONE_M1480_EXACT_TYPE_CONFIG_COMPAT_M1458_ATTEMPT" and
            release.get("runner_sha256") == sha256(SOURCE) and
            release.get("m1475_runner_sha256") == M1475_SOURCE_SHA256 and
            release.get("result") == str(
                M1475.M1458.CANONICAL_RESULT.relative_to(ROOT)) and
            release.get("attempt") == str(
                M1475.M1458.CANONICAL_ATTEMPT.relative_to(ROOT)) and
            release.get("log") == str(M1475.M1458.CANONICAL_LOG.relative_to(ROOT)),
            "M1482 release mismatch")
    exact_authorization(release.get("authorization"), True)

    final = M1475.M1458.verify_double_seal(
        FINAL, values["M1480_EXPECTED_FINAL_REVIEW_SHA256"],
        values["M1480_EXPECTED_FINAL_MANIFEST_SHA256"],
        values["M1480_EXPECTED_FINAL_OUTER_SHA256"])
    require(final.get("status") ==
            "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH" and
            final.get("bindings", {}).get("release_sha256") == sha256(RELEASE),
            "M1483 final mismatch")
    exact_authorization(final.get("authorization"), True)


def source_self_check() -> None:
    validate_source_contract()
    require(all(not os.path.lexists(str(path)) for path in (BLIND, RELEASE, FINAL)),
            "future M1481/M1482/M1483 authority must be absent")


def remote_preflight() -> None:
    validate_source_contract()
    validate_future_authorities(external_bindings())
    with M1475.configuration_content_compatibility():
        M1475.M1458.remote_preflight()


def execute_once(temp_log: Path) -> Path:
    validate_source_contract()
    validate_future_authorities(external_bindings())
    with M1475.configuration_content_compatibility():
        return M1475.M1458.execute_once(temp_log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--remote-preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--temporary-log", type=Path)
    args = parser.parse_args()
    if args.source_self_check:
        require(args.temporary_log is None, "source check cannot name log")
        source_self_check()
        print("PASS_M1480_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT")
        return 0
    if args.remote_preflight:
        require(args.temporary_log is None, "preflight cannot name log")
        remote_preflight()
        print("PASS_M1480_REMOTE_READ_ONLY_PREFLIGHT__NO_ATTEMPT")
        return 0
    require(args.temporary_log is not None, "run requires temporary log")
    execute_once(args.temporary_log.resolve())
    print("PASS_M1480_M1458_EXACT_TYPE_CONFIG_COMPAT_ONE_SHOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
