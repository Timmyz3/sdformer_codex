#!/opt/conda/envs/sdformerflow/bin/python
"""Export two omitted M1434 digest aliases around the authorized M1485 run.

M1458 validates the profile and ATLIF source through attributes that M1434
retains under its sealed M1349 predecessor but forgot to re-export.  This
bootstrap adds only those two exact aliases to the already imported M1434
module, calls the exact M1485 authority, then unconditionally removes them.
It neither owns nor changes the M1458 result/attempt/log namespace.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
M1485_SOURCE = HW / "scripts/run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
M1485_SOURCE_SHA256 = "d9779f52bd6342898b26f14b05f8052888fd81cb35d73d10168319ade6d8db9a"
M1485_TEST = HW / "tests/test_run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
M1485_TEST_SHA256 = "7ff297bfc5a16e3dc01b2bac089d216fb5a899a5acae889ba9f072734da4510c"
M1485_CONTRACT = HW / (
    "contracts/m1485_m1480_nested_m1233_config_compat_source_contract_r1_20260831.json")
M1485_CONTRACT_SHA256 = "44e8d98a5b3d997a16bdac158936e27e95eb4f66787602abc0c78edbd7aa7e2e"
M1488 = HW / (
    "reviews/m1488_m1487_m1485_nested_m1233_config_compat_final_launch_hammer_"
    "r1_20260831")
M1488_REVIEW_SHA256 = "119fe0db1453b5699d50a33e1f2fbdd8a66d2626873006c2f119bf51a4edcd4f"
M1488_MANIFEST_SHA256 = "ca796bb72ffdb5d02ff291918d6f94281711b3b89261b83b76cc7e9fa428afa4"
M1488_OUTER_SHA256 = "8894a538c85c1afb19a69322de157ef71ec299ab35e6661e582bba42b7e65b01"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PROFILE_SHA256 = "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684"
ATLIF_SHA256 = "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3"


class M1489Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1489Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA mismatch")


def load_m1485():
    regular_exact(M1485_SOURCE, M1485_SOURCE_SHA256, "M1485 source")
    spec = importlib.util.spec_from_file_location("m1489_sealed_m1485", M1485_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1485")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1485_SOURCE, M1485_SOURCE_SHA256, "M1485 source after import")
    return module


M1485 = load_m1485()
M1434 = M1485.M1480.M1475.M1458.M1434


def validate_bootstrap() -> None:
    regular_exact(M1485_TEST, M1485_TEST_SHA256, "M1485 test")
    regular_exact(M1485_CONTRACT, M1485_CONTRACT_SHA256, "M1485 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    final = M1485.M1480.M1475.M1458.verify_double_seal(
        M1488, M1488_REVIEW_SHA256, M1488_MANIFEST_SHA256, M1488_OUTER_SHA256)
    require(final.get("status") ==
            "PASS_M1488_M1485_NESTED_M1233_CONFIG_COMPAT_FINAL_LAUNCH",
            "M1488 final authority mismatch")
    M1485.exact_authorization(final.get("authorization"), True)
    require(not hasattr(M1434, "PROFILE_SOURCE_SHA256") and
            not hasattr(M1434, "ATLIF_OVERLAY_SOURCE_SHA256"),
            "M1434 digest alias preinstalled")
    require(type(M1434.M1349.PROFILE_SOURCE_SHA256) is str and
            M1434.M1349.PROFILE_SOURCE_SHA256 == PROFILE_SHA256 and
            type(M1434.M1349.ATLIF_OVERLAY_SOURCE_SHA256) is str and
            M1434.M1349.ATLIF_OVERLAY_SOURCE_SHA256 == ATLIF_SHA256,
            "sealed M1349 digest source mismatch")


@contextlib.contextmanager
def export_digest_aliases() -> Iterator[None]:
    validate_bootstrap()
    setattr(M1434, "PROFILE_SOURCE_SHA256", PROFILE_SHA256)
    setattr(M1434, "ATLIF_OVERLAY_SOURCE_SHA256", ATLIF_SHA256)
    tampered = False
    try:
        yield
    finally:
        tampered = (getattr(M1434, "PROFILE_SOURCE_SHA256", None) != PROFILE_SHA256 or
                    getattr(M1434, "ATLIF_OVERLAY_SOURCE_SHA256", None) != ATLIF_SHA256)
        for name in ("PROFILE_SOURCE_SHA256", "ATLIF_OVERLAY_SOURCE_SHA256"):
            if hasattr(M1434, name):
                delattr(M1434, name)
        require(not tampered, "M1434 digest alias changed inside bootstrap")


def source_self_check() -> None:
    validate_bootstrap()
    with export_digest_aliases():
        require(M1434.PROFILE_SOURCE_SHA256 == PROFILE_SHA256 and
                M1434.ATLIF_OVERLAY_SOURCE_SHA256 == ATLIF_SHA256,
                "M1434 digest alias export mismatch")
    require(not hasattr(M1434, "PROFILE_SOURCE_SHA256") and
            not hasattr(M1434, "ATLIF_OVERLAY_SOURCE_SHA256"),
            "M1434 digest aliases not restored")


def remote_preflight() -> None:
    with export_digest_aliases():
        M1485.remote_preflight()


def execute_once(temp_log: Path) -> Path:
    with export_digest_aliases():
        return M1485.execute_once(temp_log)


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
        print("PASS_M1489_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT")
        return 0
    if args.remote_preflight:
        require(args.temporary_log is None, "preflight cannot name log")
        remote_preflight()
        print("PASS_M1489_REMOTE_READ_ONLY_PREFLIGHT__NO_ATTEMPT")
        return 0
    require(args.temporary_log is not None, "run requires temporary log")
    execute_once(args.temporary_log.resolve())
    print("PASS_M1489_M1485_M1434_EXPORT_ALIAS_ONE_SHOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
