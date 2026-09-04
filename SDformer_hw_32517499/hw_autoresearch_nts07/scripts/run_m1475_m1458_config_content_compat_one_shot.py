#!/opt/conda/envs/sdformerflow/bin/python
"""Narrow content-identity compatibility wrapper for the sealed M1458 run.

The frozen ep34 selection records inode/mode/mtime for the configuration file.
The remote server later recreated that file with byte-identical contents, so
M1458's read-only preflight stopped before consuming an attempt.  This wrapper
changes no capture logic or namespace.  While M1458 builds its runtime, and
only for the ``selected configuration`` identity, it accepts the pinned file
by stable path, regular-file type, exact size, and exact SHA-256.  Checkpoint
and profile identities continue through the original strict M1319 verifier.
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
TEST = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1475_m1458_config_content_compat_source_contract_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

M1458_SOURCE = HW / "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
M1458_SOURCE_SHA256 = "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154"

CONFIG_ABSOLUTE = (
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
CONFIG_PATH = Path(CONFIG_ABSOLUTE)
CONFIG_SIZE = 6481
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
FROZEN_CONFIG_ENTITY = {
    "absolute_path": CONFIG_ABSOLUTE,
    "size_bytes": CONFIG_SIZE,
    "mtime_ns": 1788081356000000000,
    "sha256": CONFIG_SHA256,
    "device": 194,
    "inode": 26561699333,
    "mode": 33152,
}

BLIND = HW / (
    "reviews/m1476_m1475_m1458_config_content_compat_source_blind_hammer_"
    "r1_20260831")
RELEASE = HW / (
    "contracts/m1477_m1475_m1458_config_content_compat_launch_release_"
    "r1_20260831.json")
FINAL = HW / (
    "reviews/m1478_m1477_m1475_m1458_config_content_compat_final_launch_hammer_"
    "r1_20260831")

ENV_BINDINGS = {
    "M1475_EXPECTED_RUNNER_SHA256": SOURCE,
    "M1475_EXPECTED_BLIND_REVIEW_SHA256": BLIND / "review.json",
    "M1475_EXPECTED_BLIND_MANIFEST_SHA256": BLIND / "SHA256SUMS",
    "M1475_EXPECTED_BLIND_OUTER_SHA256": BLIND / "SHA256SUMS.seal.sha256",
    "M1475_EXPECTED_RELEASE_SHA256": RELEASE,
    "M1475_EXPECTED_FINAL_REVIEW_SHA256": FINAL / "review.json",
    "M1475_EXPECTED_FINAL_MANIFEST_SHA256": FINAL / "SHA256SUMS",
    "M1475_EXPECTED_FINAL_OUTER_SHA256": FINAL / "SHA256SUMS.seal.sha256",
}


class M1475Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1475Error(message)


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
        raise M1475Error("missing " + label) from exc
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
                           M1475Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def load_m1458():
    regular_exact(M1458_SOURCE, M1458_SOURCE_SHA256, "M1458 source")
    spec = importlib.util.spec_from_file_location("m1475_sealed_m1458", M1458_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1458")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1458_SOURCE, M1458_SOURCE_SHA256, "M1458 source after import")
    return module


M1458 = load_m1458()
M1319 = M1458.M1434.M1349.M1327.M1319
ORIGINAL_EXACT_EXTENDED_IDENTITY = M1319.exact_extended_identity


def verify_configuration_content_identity(value: Any) -> dict[str, Any]:
    require(type(value) is dict and value == FROZEN_CONFIG_ENTITY,
            "frozen configuration selection identity drift")
    path = CONFIG_PATH
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise M1475Error("selected configuration missing") from exc
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            "selected configuration must be regular non-symlink")
    digest = sha256(path)
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
             before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size,
             after.st_mtime_ns), "selected configuration changed while hashing")
    require(after.st_size == CONFIG_SIZE and digest == CONFIG_SHA256 and
            str(path) == CONFIG_ABSOLUTE,
            "selected configuration path/size/SHA drift")
    return dict(value)


@contextlib.contextmanager
def configuration_content_compatibility() -> Iterator[None]:
    require(M1319.exact_extended_identity is ORIGINAL_EXACT_EXTENDED_IDENTITY,
            "M1319 identity verifier already replaced")

    def narrow(value: Any, label: str):
        if label == "selected configuration":
            return verify_configuration_content_identity(value)
        return ORIGINAL_EXACT_EXTENDED_IDENTITY(value, label)

    M1319.exact_extended_identity = narrow
    try:
        yield
    finally:
        require(M1319.exact_extended_identity is narrow,
                "M1319 identity verifier changed inside compatibility scope")
        M1319.exact_extended_identity = ORIGINAL_EXACT_EXTENDED_IDENTITY


def validate_source_contract() -> None:
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    value = strict_json(CONTRACT)
    require(value.get("status") ==
            "SOURCE_ONLY__CONFIG_CONTENT_IDENTITY_COMPAT__M1476_REQUIRED__NO_LAUNCH",
            "source contract status mismatch")
    require(value.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "source contract identity mismatch")
    require(value.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        "test contract identity mismatch")
    require(value.get("configuration") == FROZEN_CONFIG_ENTITY,
            "configuration contract mismatch")
    require(value.get("claim_boundary", {}).get("launch") is False and
            value.get("claim_boundary", {}).get("capture") is False,
            "source contract overclaim")


def external_bindings(environment: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    values = {}
    for name, path in ENV_BINDINGS.items():
        value = environment.get(name, "")
        require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
                "missing/malformed external SHA: " + name)
        regular_exact(path, value, name)
        values[name] = value
    require(values["M1475_EXPECTED_RUNNER_SHA256"] == sha256(SOURCE),
            "external runner SHA mismatch")
    return values


def validate_future_authorities(values: dict[str, str]) -> None:
    blind = M1458.verify_double_seal(
        BLIND, values["M1475_EXPECTED_BLIND_REVIEW_SHA256"],
        values["M1475_EXPECTED_BLIND_MANIFEST_SHA256"],
        values["M1475_EXPECTED_BLIND_OUTER_SHA256"])
    require(blind.get("status") == "PASS_M1475_CONFIG_CONTENT_COMPAT_SOURCE" and
            blind.get("authorization", {}).get("launch") is False and
            blind.get("bindings", {}).get("runner_sha256") == sha256(SOURCE),
            "M1476 blind authority mismatch")
    release = strict_json(RELEASE)
    require(release.get("status") ==
            "AUTHORIZE_ONE_M1475_CONFIG_CONTENT_COMPAT_M1458_ATTEMPT" and
            type(release.get("runs")) is int and release["runs"] == 1 and
            release.get("automatic_retry") is False and
            release.get("controller_restore") is False and
            release.get("runner_sha256") == sha256(SOURCE) and
            release.get("m1458_runner_sha256") == M1458_SOURCE_SHA256 and
            release.get("result") == str(M1458.CANONICAL_RESULT.relative_to(ROOT)) and
            release.get("attempt") == str(M1458.CANONICAL_ATTEMPT.relative_to(ROOT)) and
            release.get("log") == str(M1458.CANONICAL_LOG.relative_to(ROOT)),
            "M1477 release mismatch")
    final = M1458.verify_double_seal(
        FINAL, values["M1475_EXPECTED_FINAL_REVIEW_SHA256"],
        values["M1475_EXPECTED_FINAL_MANIFEST_SHA256"],
        values["M1475_EXPECTED_FINAL_OUTER_SHA256"])
    require(final.get("status") ==
            "PASS_M1478_M1475_CONFIG_CONTENT_COMPAT_FINAL_LAUNCH" and
            final.get("authorization") == {
                "launch": True, "runs": 1, "automatic_retry": False,
                "controller_restore": False} and
            final.get("bindings", {}).get("release_sha256") == sha256(RELEASE),
            "M1478 final authority mismatch")


def source_self_check() -> None:
    validate_source_contract()
    require(all(not os.path.lexists(str(path)) for path in (BLIND, RELEASE, FINAL)),
            "future M1476/M1477/M1478 authority must be absent")
    require(M1319.exact_extended_identity is ORIGINAL_EXACT_EXTENDED_IDENTITY,
            "source check observed patched verifier")


def remote_preflight() -> None:
    validate_source_contract()
    values = external_bindings()
    validate_future_authorities(values)
    with configuration_content_compatibility():
        M1458.remote_preflight()
    require(M1319.exact_extended_identity is ORIGINAL_EXACT_EXTENDED_IDENTITY,
            "M1319 verifier not restored after preflight")


def execute_once(temp_log: Path) -> Path:
    validate_source_contract()
    values = external_bindings()
    validate_future_authorities(values)
    with configuration_content_compatibility():
        result = M1458.execute_once(temp_log)
    require(M1319.exact_extended_identity is ORIGINAL_EXACT_EXTENDED_IDENTITY,
            "M1319 verifier not restored after execution")
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
        require(args.temporary_log is None, "source check cannot name a log")
        source_self_check()
        print("PASS_M1475_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT")
        return 0
    if args.remote_preflight:
        require(args.temporary_log is None, "preflight cannot name a log")
        remote_preflight()
        print("PASS_M1475_REMOTE_READ_ONLY_PREFLIGHT__NO_ATTEMPT")
        return 0
    require(args.temporary_log is not None, "run requires --temporary-log")
    execute_once(args.temporary_log.resolve())
    print("PASS_M1475_M1458_CONFIG_CONTENT_COMPAT_ONE_SHOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
