#!/usr/bin/env python3
"""Read-only independent hammer for the M1837 C3 recovery source.

This program does not import or invoke the M1808 runner, query licenses,
create an attempt, or create a result/release.  It reproduces the semantic
validation escapes that make M1837 fail closed at M1838.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HW = Path(__file__).resolve().parents[2]
AUTHOR_CHECKER = HW / "system_simulator/scripts/check_m1837_m1808_c3_preflight_recovery_source.py"
AUTHOR_CONTRACT = HW / "contracts/m1837_m1808_c3_preflight_recovery_source_contract_r1_20260902.json"
AUTHOR_SIDECAR = Path(str(AUTHOR_CONTRACT) + ".sha256")
AUTHOR_OUTER = Path(str(AUTHOR_CONTRACT) + ".sha256.seal.sha256")
AUTHOR_RECEIPT = HW / "reviews/m1837_m1808_c3_preflight_recovery_source_author_receipt_r1_20260902"
PREFLIGHT = HW / "results/m1808_c3_mapped_energy_r1_20260902.preflight_rejected_source_chain_governance_quarantine"
ORIGINAL_FAILURE = HW / "results/m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
ATTEMPT = HW / "results/.m1808_c3_mapped_energy_attempt_consumed"
RESULT = HW / "results/m1808_c3_mapped_energy_r1_20260902"
PRIVATE = HW / "results/m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"

FIXED = {
    "contract": "7257c39b9d68ecc92af36124b490d2f46b97ec7d961fc218abdf8880533382ab",
    "contract_sidecar": "28ad3eb39b903cff1634fd2d7650d58e4ba83e982eda220b31d9f35e87d20c48",
    "contract_outer": "cff64fc604ec10f5eebcc5484f429771a0254b17f2c479ad321fe7473f42b410",
    "checker": "3c80c06ab6d3feb96e216c3f7516d4aefd4e9168ee7e51445f6b281dda0882bb",
    "receipt_manifest": "1a3780999ac0d7847d65c13db5ab048d563895339509d359758390c73ece6ff4",
    "receipt_outer": "01467cc1d3b228e128c579c577fc5952a963dd5814fb0f749c5d0c50983658f7",
    "preflight_failure": "ea9d08303dd29196a761c1e9927e5aa148a5f8746e1d5b4a64f354d66c74eda8",
    "preflight_manifest": "e243c0f10d810b1b5d39523ad479a1df2d751a3f139d7eae944072d2788eb856",
    "preflight_outer": "d9824a782b5ee5f1ba116abe2c7719a24579815798ee5b1d48b342de38784124",
}


class HammerFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise HammerFailure("identity drift: " + str(path))


def verify_file_seal(path, sidecar, outer, file_sha, sidecar_sha, outer_sha):
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if Path(sidecar).read_text().split() != [file_sha, Path(path).name]:
        raise HammerFailure("sidecar content drift")
    if Path(outer).read_text().split() != [sidecar_sha, Path(sidecar).name]:
        raise HammerFailure("outer content drift")


def verify_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise HammerFailure("directory outer seal drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe/duplicate manifest member")
        exact(root / rel, digest)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HammerFailure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise HammerFailure("sealed population drift")
    return listed


def load_author_checker():
    exact(AUTHOR_CHECKER, FIXED["checker"])
    spec = importlib.util.spec_from_file_location("m1837_author_checker", str(AUTHOR_CHECKER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attack_values(base):
    values = []

    def add(name, mutate):
        value = json.loads(json.dumps(base))
        mutate(value)
        values.append((name, value))

    add("diagnosis_license_or_eda_reached_true",
        lambda v: v["diagnosis"].__setitem__("license_or_eda_reached", True))
    add("diagnosis_correct_m1815_manifest_wrong",
        lambda v: v["diagnosis"].__setitem__("correct_m1815_manifest_sha256", "0" * 64))
    add("diagnosis_attempt_consumed_true",
        lambda v: v["diagnosis"].__setitem__("attempt_consumed", True))
    add("milestone_changed",
        lambda v: v.__setitem__("milestone", "M9999"))
    add("purpose_changed_to_immediate_launch",
        lambda v: v.__setitem__("purpose", "authorize immediate launch"))
    add("unknown_top_level_launch_authorized_true",
        lambda v: v.__setitem__("launch_authorized_now", True))
    return values


def main():
    verify_file_seal(
        AUTHOR_CONTRACT, AUTHOR_SIDECAR, AUTHOR_OUTER,
        FIXED["contract"], FIXED["contract_sidecar"], FIXED["contract_outer"])
    verify_directory(AUTHOR_RECEIPT, FIXED["receipt_manifest"], FIXED["receipt_outer"])
    members = verify_directory(PREFLIGHT, FIXED["preflight_manifest"], FIXED["preflight_outer"])
    if members != {"failure.json"}:
        raise HammerFailure("preflight member set drift")
    exact(PREFLIGHT / "failure.json", FIXED["preflight_failure"])
    if os.path.lexists(str(ORIGINAL_FAILURE)):
        raise HammerFailure("original failure namespace still present")
    for path in (ATTEMPT, RESULT, PRIVATE):
        if os.path.lexists(str(path)):
            raise HammerFailure("pre-attempt namespace unexpectedly present: " + str(path))

    checker = load_author_checker()
    actual = checker.validate_sources()
    if actual.get("status") != "PASS_M1837_ONE_MANUAL_RECOVERY_SOURCE":
        raise HammerFailure("author checker does not accept frozen source")

    base = json.loads(AUTHOR_CONTRACT.read_text())
    escaped = []
    rejected = []
    for name, value in attack_values(base):
        try:
            checker.validate_sources(json.dumps(value, sort_keys=True))
        except Exception as error:
            rejected.append({"attack": name, "error": str(error)})
        else:
            escaped.append(name)
    expected = [name for name, _ in attack_values(base)]
    if escaped != expected or rejected:
        raise HammerFailure("semantic escape reproduction drift")
    print(json.dumps({
        "status": "FAIL_P1_M1837_SEMANTIC_VALIDATION_ESCAPES",
        "p0": 0,
        "p1": 1,
        "escaped_attacks": escaped,
        "escaped_count": len(escaped),
        "attempt_consumed": False,
        "launch_authorized": False,
        "release_permitted": False,
        "eda_or_license_run_by_hammer": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
