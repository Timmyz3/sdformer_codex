#!/opt/conda/envs/sdformerflow/bin/python
"""Patch the second, frozen-M1233 configuration identity check only.

M1475 correctly relaxes the extended M1319 configuration identity to exact
path/type/size/SHA, but M1319 subsequently calls frozen M1233, which repeats a
legacy mtime identity check.  This wrapper keeps M1480/M1475/M1458 and their
one-shot namespace unchanged and applies the same content-exact rule only to
that nested ``selected configuration`` call.  Checkpoint and profile calls
continue through the original frozen-M1233 verifier.
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


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1485_m1480_nested_m1233_config_compat_source_contract_r1_"
    "20260831.json")
M1480_SOURCE = HW / "scripts/run_m1480_m1475_exact_type_config_compat_one_shot.py"
M1480_SOURCE_SHA256 = "3a0235f91d8d6acd4c94168b3b611cb53504f50e3843580c09bc1673042df4ce"
M1480_TEST = HW / "tests/test_run_m1480_m1475_exact_type_config_compat_one_shot.py"
M1480_TEST_SHA256 = "dea2bc2cb3851a40462f5200b423c623331aa20abc054debc8e2ea661fc99ea3"
M1480_CONTRACT = HW / (
    "contracts/m1480_m1475_exact_type_config_compat_source_contract_r1_20260831.json")
M1480_CONTRACT_SHA256 = "c4ec0a4792a7647c46614652147de6999d2dce0c6c55d5d46a88798e12ad90e4"
M1483 = HW / (
    "reviews/m1483_m1482_m1480_m1475_exact_type_config_compat_final_launch_"
    "hammer_r1_20260831")
M1483_REVIEW_SHA256 = "7df093c24f2826fe7ddd1127a429d1fdad4330deabf24f81245869074636caed"
M1483_MANIFEST_SHA256 = "0c5beeac1c3cfa1b1319506b96810e3b30b7b24c19e61fdf7589da3c87894f21"
M1483_OUTER_SHA256 = "e07a9ac4e8e057b1ec29fa67d7486bb3c8c0f1249383b0aa08f08349c023df5d"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

BLIND = HW / (
    "reviews/m1486_m1485_nested_m1233_config_compat_source_blind_hammer_"
    "r1_20260831")
RELEASE = HW / (
    "contracts/m1487_m1485_nested_m1233_config_compat_launch_release_"
    "r1_20260831.json")
FINAL = HW / (
    "reviews/m1488_m1487_m1485_nested_m1233_config_compat_final_launch_hammer_"
    "r1_20260831")
ENV_BINDINGS = {
    "M1485_EXPECTED_RUNNER_SHA256": SOURCE,
    "M1485_EXPECTED_BLIND_REVIEW_SHA256": BLIND / "review.json",
    "M1485_EXPECTED_BLIND_MANIFEST_SHA256": BLIND / "SHA256SUMS",
    "M1485_EXPECTED_BLIND_OUTER_SHA256": BLIND / "SHA256SUMS.seal.sha256",
    "M1485_EXPECTED_RELEASE_SHA256": RELEASE,
    "M1485_EXPECTED_FINAL_REVIEW_SHA256": FINAL / "review.json",
    "M1485_EXPECTED_FINAL_MANIFEST_SHA256": FINAL / "SHA256SUMS",
    "M1485_EXPECTED_FINAL_OUTER_SHA256": FINAL / "SHA256SUMS.seal.sha256",
}


class M1485Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1485Error(message)


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
        raise M1485Error("missing " + label) from exc
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
                           M1485Error("nonfinite JSON " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1480():
    regular_exact(M1480_SOURCE, M1480_SOURCE_SHA256, "M1480 source")
    spec = importlib.util.spec_from_file_location("m1485_sealed_m1480", M1480_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1480")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1480_SOURCE, M1480_SOURCE_SHA256, "M1480 source after import")
    return module


M1480 = load_m1480()
FROZEN_M1233 = M1480.M1475.M1319.FROZEN_M1233
M1319 = M1480.M1475.M1319
ORIGINAL_EXTENDED_EXACT_IDENTITY = M1480.M1475.ORIGINAL_EXACT_EXTENDED_IDENTITY
ORIGINAL_M1233_EXACT_IDENTITY = FROZEN_M1233.exact_identity


def verify_frozen_config_entity_exact_type(value: Any) -> None:
    expected = M1480.M1475.FROZEN_CONFIG_ENTITY
    require(type(value) is dict and set(value) == set(expected),
            "frozen configuration entity shape mismatch")
    for key, frozen in expected.items():
        require(type(value[key]) is type(frozen) and value[key] == frozen,
                "frozen configuration entity exact-type/value drift: " + key)


@contextlib.contextmanager
def dual_configuration_compatibility() -> Iterator[None]:
    require(M1319.exact_extended_identity is ORIGINAL_EXTENDED_EXACT_IDENTITY,
            "M1319 extended identity verifier already replaced")
    require(FROZEN_M1233.exact_identity is ORIGINAL_M1233_EXACT_IDENTITY,
            "frozen M1233 identity verifier already replaced")

    def extended_narrow(value: Any, label: str):
        if label == "selected configuration":
            verify_frozen_config_entity_exact_type(value)
            return M1480.M1475.verify_configuration_content_identity(value)
        return ORIGINAL_EXTENDED_EXACT_IDENTITY(value, label)

    def frozen_narrow(value: Any, label: str):
        if label == "selected configuration":
            verify_frozen_config_entity_exact_type(value)
            return M1480.M1475.verify_configuration_content_identity(value)
        return ORIGINAL_M1233_EXACT_IDENTITY(value, label)

    M1319.exact_extended_identity = extended_narrow
    FROZEN_M1233.exact_identity = frozen_narrow
    tampered = False
    try:
        yield
    finally:
        tampered = (M1319.exact_extended_identity is not extended_narrow or
                    FROZEN_M1233.exact_identity is not frozen_narrow)
        M1319.exact_extended_identity = ORIGINAL_EXTENDED_EXACT_IDENTITY
        FROZEN_M1233.exact_identity = ORIGINAL_M1233_EXACT_IDENTITY
        require(not tampered, "identity verifier changed inside compatibility scope")


def validate_source_contract() -> None:
    regular_exact(M1480_TEST, M1480_TEST_SHA256, "M1480 test")
    regular_exact(M1480_CONTRACT, M1480_CONTRACT_SHA256, "M1480 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    M1480.validate_source_contract()
    predecessor = M1480.M1475.M1458.verify_double_seal(
        M1483, M1483_REVIEW_SHA256, M1483_MANIFEST_SHA256, M1483_OUTER_SHA256)
    require(predecessor.get("status") ==
            "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH",
            "M1483 predecessor mismatch")
    value = strict_json(CONTRACT)
    require(value.get("status") ==
            "SOURCE_ONLY__NESTED_M1233_CONFIG_CONTENT_COMPAT__M1486_REQUIRED__NO_LAUNCH",
            "source contract status mismatch")
    require(value.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "source identity mismatch")
    require(value.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        "test identity mismatch")


def external_bindings(environment: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    values = {}
    for name, path in ENV_BINDINGS.items():
        value = environment.get(name, "")
        require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
                "missing/malformed external SHA " + name)
        regular_exact(path, value, name)
        values[name] = value
    require(values["M1485_EXPECTED_RUNNER_SHA256"] == sha256(SOURCE),
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
    verify = M1480.M1475.M1458.verify_double_seal
    blind = verify(BLIND, values["M1485_EXPECTED_BLIND_REVIEW_SHA256"],
                   values["M1485_EXPECTED_BLIND_MANIFEST_SHA256"],
                   values["M1485_EXPECTED_BLIND_OUTER_SHA256"])
    require(blind.get("status") ==
            "PASS_M1486_M1485_NESTED_M1233_CONFIG_COMPAT_SOURCE" and
            blind.get("bindings", {}).get("runner_sha256") == sha256(SOURCE),
            "M1486 blind mismatch")
    exact_authorization(blind.get("authorization"), False)
    release = strict_json(RELEASE)
    capture = M1480.M1475.M1458
    require(release.get("status") ==
            "AUTHORIZE_ONE_M1485_NESTED_M1233_CONFIG_COMPAT_M1458_ATTEMPT" and
            release.get("runner_sha256") == sha256(SOURCE) and
            release.get("result") == str(capture.CANONICAL_RESULT.relative_to(ROOT)) and
            release.get("attempt") == str(capture.CANONICAL_ATTEMPT.relative_to(ROOT)) and
            release.get("log") == str(capture.CANONICAL_LOG.relative_to(ROOT)),
            "M1487 release mismatch")
    exact_authorization(release.get("authorization"), True)
    final = verify(FINAL, values["M1485_EXPECTED_FINAL_REVIEW_SHA256"],
                   values["M1485_EXPECTED_FINAL_MANIFEST_SHA256"],
                   values["M1485_EXPECTED_FINAL_OUTER_SHA256"])
    require(final.get("status") ==
            "PASS_M1488_M1485_NESTED_M1233_CONFIG_COMPAT_FINAL_LAUNCH" and
            final.get("bindings", {}).get("release_sha256") == sha256(RELEASE),
            "M1488 final mismatch")
    exact_authorization(final.get("authorization"), True)


def source_self_check() -> None:
    validate_source_contract()
    require(all(not os.path.lexists(str(path)) for path in (BLIND, RELEASE, FINAL)),
            "future M1486/M1487/M1488 authority must be absent")


def remote_preflight() -> None:
    validate_source_contract()
    validate_future_authorities(external_bindings())
    M1480.validate_future_authorities(M1480.external_bindings())
    with dual_configuration_compatibility():
        M1480.M1475.M1458.remote_preflight()
    require(M1319.exact_extended_identity is ORIGINAL_EXTENDED_EXACT_IDENTITY and
            FROZEN_M1233.exact_identity is ORIGINAL_M1233_EXACT_IDENTITY,
            "identity verifiers not restored")


def execute_once(temp_log: Path) -> Path:
    validate_source_contract()
    validate_future_authorities(external_bindings())
    M1480.validate_future_authorities(M1480.external_bindings())
    with dual_configuration_compatibility():
        result = M1480.M1475.M1458.execute_once(temp_log)
    require(M1319.exact_extended_identity is ORIGINAL_EXTENDED_EXACT_IDENTITY and
            FROZEN_M1233.exact_identity is ORIGINAL_M1233_EXACT_IDENTITY,
            "identity verifiers not restored")
    return result


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
        print("PASS_M1485_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT")
        return 0
    if args.remote_preflight:
        require(args.temporary_log is None, "preflight cannot name log")
        remote_preflight()
        print("PASS_M1485_REMOTE_READ_ONLY_PREFLIGHT__NO_ATTEMPT")
        return 0
    require(args.temporary_log is not None, "run requires temporary log")
    execute_once(args.temporary_log.resolve())
    print("PASS_M1485_M1480_NESTED_M1233_CONFIG_COMPAT_ONE_SHOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
