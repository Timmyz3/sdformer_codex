#!/usr/bin/env python3
"""Fail-closed checker for the M1837 one-manual-recovery source contract.

The checker is governance-only.  It never imports or launches the M1808
runner, never queries a license, and never creates an attempt or result.
"""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


HW = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1837_m1808_c3_preflight_recovery_source.py"
CONTRACT = HW / "contracts/m1837_m1808_c3_preflight_recovery_source_contract_r1_20260902.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")

RUNNER = HW / "dc_handoff/scripts/run_m1808_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
M1808_CHECKER = HW / "system_simulator/scripts/check_m1808_c3_m1454_fixed_t10_mapped_energy_source.py"
M1808_TEST = HW / "system_simulator/tests/test_m1808_c3_m1454_fixed_t10_mapped_energy_source.py"
M1808_CONTRACT = HW / "contracts/m1808_m1807_c3_m1454_fixed_t10_mapped_energy_reset_settling_source_contract_r1_20260902.json"
M1808_CONTRACT_SIDECAR = Path(str(M1808_CONTRACT) + ".sha256")
M1808_CONTRACT_OUTER = Path(str(M1808_CONTRACT) + ".sha256.seal.sha256")
M1815 = HW / "reviews/m1815_m1808_c3_m1454_fixed_t10_mapped_energy_source_hammer_r1_20260902"
M1816 = HW / "contracts/m1816_m1815_m1808_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_20260902.json"
M1816_SIDECAR = Path(str(M1816) + ".sha256")
M1816_OUTER = Path(str(M1816) + ".sha256.seal.sha256")

ORIGINAL_FAILURE = HW / "results/m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
PREFLIGHT_QUARANTINE = HW / "results/m1808_c3_mapped_energy_r1_20260902.preflight_rejected_source_chain_governance_quarantine"
ATTEMPT = HW / "results/.m1808_c3_mapped_energy_attempt_consumed"
RESULT = HW / "results/m1808_c3_mapped_energy_r1_20260902"
PRIVATE = HW / "results/m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"

FIXED_SHA = {
    "runner": "17262b329a130c027d3be4b0a912ac75a34d63bc29c568372433a5126d6d6e51",
    "m1808_checker": "cf36c026997e066871b9db68770e1dd6cf7a6ed3bf15ae1b858f91680206c498",
    "m1808_test": "78f273d6563bc2b7d3d324339cd1f7c4b7ca65308bd6d1c4ef86658f8ca60585",
    "m1808_contract": "cfba88c6866dbcd67a97680f0276dba53443b95bd44d00732aa134c67cb11c92",
    "m1808_contract_sidecar": "5f06f79589d9c44bbe849392537dfe6897d52e2641f5ad6611cce136cccdf488",
    "m1808_contract_outer": "6b05fbda8951e5dbbf78413d0b3c4badc8f15a727aca24b28f745a32656ffadb",
    "m1815_review": "5a5ecdd93d78033c842b5985028b243eea71361b360e27513d2e9361a6870092",
    "m1815_manifest": "5b124a54f4bfe9b64369990958a053175358d97783f080aff08b99c923233099",
    "m1815_outer": "d0841a30da88a4ca37cdf56ea263a0e52fa1121a2b1ba7d314f14c902f3b7777",
    "m1816_release": "c948d79fb6fd93a2d4f33b6c16c83c33b6a2985cdaef7d928e63fc292dc3549f",
    "m1816_sidecar": "33e2391ad4c7952b6e04371cc81fdaf378071bd735d49ac614ccfa601aecb1b2",
    "m1816_outer": "fa2e6415a991449b3f4329ea27b8e3f8f04ded4183a9635ff6035f860a63b38b",
    "preflight_failure": "ea9d08303dd29196a761c1e9927e5aa148a5f8746e1d5b4a64f354d66c74eda8",
    "preflight_manifest": "e243c0f10d810b1b5d39523ad479a1df2d751a3f139d7eae944072d2788eb856",
    "preflight_outer": "d9824a782b5ee5f1ba116abe2c7719a24579815798ee5b1d48b342de38784124",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_FAILURE = {
    "attempt_consumed": False,
    "automatic_retry": False,
    "canonical_result": False,
    "counts": {"ptpx_runs": 0, "saif_files": 0,
               "simv_runs": 0, "vcs_compiles": 0},
    "error": "Failure",
    "phase": "SOURCE_CHAIN",
    "status": "FAILED_OR_INCOMPLETE",
}

CLAIM_BOUNDARY = dict((name, False) for name in (
    "launch_authorized_now", "final_recovery_release_created",
    "license_queried", "attempt_consumed", "vcs_compile", "simv_run",
    "saif_generated", "ptpx_run", "mapped_vcs", "production_saif",
    "component_power", "component_energy", "energy_per_frame", "speedup",
    "system_speedup", "paper_ppa_ready", "paper_citable", "headline"))


class CheckFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if (re.fullmatch(r"[0-9a-f]{64}", digest or "") is None
            or not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
            or sha(path) != digest):
        raise CheckFailure("identity drift: " + str(path))


def strict_json_text(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise CheckFailure("duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise CheckFailure("JSON root is not object")
    return value


def strict_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CheckFailure("JSON absent/nonregular: " + str(path))
    return strict_json_text(path.read_text())


def verify_sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise CheckFailure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise CheckFailure("outer seal content drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise CheckFailure("manifest syntax: " + str(root))
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise CheckFailure("manifest unsafe/duplicate member")
        exact(root / rel, digest)
        listed[name] = digest
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CheckFailure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise CheckFailure("sealed population drift: " + str(root))
    return listed


def verify_file_double_seal(path, sidecar, outer, file_sha, sidecar_sha, outer_sha):
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if Path(sidecar).read_text().split() != [file_sha, Path(path).name]:
        raise CheckFailure("file sidecar content drift")
    if Path(outer).read_text().split() != [sidecar_sha, Path(sidecar).name]:
        raise CheckFailure("file outer seal content drift")


def validate_failure_value(value):
    if value != EXPECTED_FAILURE:
        raise CheckFailure("pre-attempt failure semantics drift")


def verify_preflight_quarantine():
    members = verify_sealed_directory(
        PREFLIGHT_QUARANTINE, FIXED_SHA["preflight_manifest"],
        FIXED_SHA["preflight_outer"])
    if members != {"failure.json": FIXED_SHA["preflight_failure"]}:
        raise CheckFailure("preflight quarantine member set drift")
    exact(PREFLIGHT_QUARANTINE / "failure.json", FIXED_SHA["preflight_failure"])
    validate_failure_value(strict_json(PREFLIGHT_QUARANTINE / "failure.json"))
    if os.path.lexists(str(ORIGINAL_FAILURE)):
        raise CheckFailure("original failure namespace was not governance-moved")
    for path in (ATTEMPT, RESULT, PRIVATE):
        if os.path.lexists(str(path)):
            raise CheckFailure("pre-attempt namespace unexpectedly present: " + str(path))
    for pattern in (".m1808_c3_mapped_energy_work.*",
                    ".m1808_c3_mapped_energy_stage.*",
                    ".m1808_c3_mapped_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise CheckFailure("private M1808 residue: " + pattern)


def verify_original_authority():
    for path, key in (
            (RUNNER, "runner"), (M1808_CHECKER, "m1808_checker"),
            (M1808_TEST, "m1808_test"), (M1808_CONTRACT, "m1808_contract"),
            (M1808_CONTRACT_SIDECAR, "m1808_contract_sidecar"),
            (M1808_CONTRACT_OUTER, "m1808_contract_outer"),
            (HW / "docs/359_DATE终局冻结_20260813.md", "docs359")):
        exact(path, FIXED_SHA[key])
    if M1808_CONTRACT_SIDECAR.read_text().split() != [
            FIXED_SHA["m1808_contract"], M1808_CONTRACT.name]:
        raise CheckFailure("M1808 contract sidecar drift")
    if M1808_CONTRACT_OUTER.read_text().split() != [
            FIXED_SHA["m1808_contract_sidecar"], M1808_CONTRACT_SIDECAR.name]:
        raise CheckFailure("M1808 contract outer drift")

    members = verify_sealed_directory(
        M1815, FIXED_SHA["m1815_manifest"], FIXED_SHA["m1815_outer"])
    if members.get("review.json") != FIXED_SHA["m1815_review"]:
        raise CheckFailure("M1815 review is not transitively sealed")
    review = strict_json(M1815 / "review.json")
    if (review.get("status") != "PASS_M1815_M1808_C3_MAPPED_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_M1808_CAMPAIGN"
            or review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}):
        raise CheckFailure("M1815 review semantics drift")

    verify_file_double_seal(
        M1816, M1816_SIDECAR, M1816_OUTER,
        FIXED_SHA["m1816_release"], FIXED_SHA["m1816_sidecar"],
        FIXED_SHA["m1816_outer"])
    release = strict_json(M1816)
    identity = release.get("identity", {})
    if (release.get("status") != "AUTHORIZE_ONE_FRESH_M1808_C3_MAPPED_ENERGY_CAMPAIGN"
            or identity.get("runner_sha256") != FIXED_SHA["runner"]
            or identity.get("source_contract_sha256") != FIXED_SHA["m1808_contract"]
            or identity.get("source_review_json_sha256") != FIXED_SHA["m1815_review"]
            or identity.get("source_review_manifest_sha256") != FIXED_SHA["m1815_manifest"]
            or identity.get("source_review_outer_file_sha256") != FIXED_SHA["m1815_outer"]
            or release.get("authorization") != {
                "launch_m1808_once": True,
                "automatic_retry": False,
                "publish_only_after_all_gates": True,
                "result_hammer_still_required": True}):
        raise CheckFailure("M1816 original authority drift")


def validate_contract_value(value, enforce_source_hashes=True):
    if (value.get("schema") != "m1837_m1808_c3_preflight_recovery_source_contract_r1_v1"
            or value.get("status") != "SOURCE_ONLY__ONE_MANUAL_RECOVERY_PROPOSED__INDEPENDENT_REVIEW_AND_FINAL_RELEASE_REQUIRED__NO_EDA"):
        raise CheckFailure("M1837 contract identity drift")
    evidence = value.get("preflight_rejection_evidence", {})
    if evidence != {
            "original_failure_namespace_absent_after_governance_move": True,
            "quarantine": PREFLIGHT_QUARANTINE.relative_to(HW).as_posix(),
            "failure_json_sha256": FIXED_SHA["preflight_failure"],
            "manifest_sha256": FIXED_SHA["preflight_manifest"],
            "outer_file_sha256": FIXED_SHA["preflight_outer"],
            "phase": "SOURCE_CHAIN",
            "attempt_consumed": False,
            "vcs_compiles": 0,
            "simv_runs": 0,
            "saif_files": 0,
            "ptpx_runs": 0,
            "automatic_retry": False,
            "preserved_not_deleted": True}:
        raise CheckFailure("M1837 preflight evidence drift")
    identity = value.get("frozen_identity", {})
    if identity != {
            "runner_sha256": FIXED_SHA["runner"],
            "m1808_checker_sha256": FIXED_SHA["m1808_checker"],
            "m1808_test_sha256": FIXED_SHA["m1808_test"],
            "m1808_source_contract_sha256": FIXED_SHA["m1808_contract"],
            "m1815_review_sha256": FIXED_SHA["m1815_review"],
            "m1815_correct_manifest_sha256": FIXED_SHA["m1815_manifest"],
            "m1815_outer_file_sha256": FIXED_SHA["m1815_outer"],
            "m1816_release_sha256": FIXED_SHA["m1816_release"],
            "m1816_sidecar_sha256": FIXED_SHA["m1816_sidecar"],
            "m1816_outer_file_sha256": FIXED_SHA["m1816_outer"],
            "docs359_sha256": FIXED_SHA["docs359"]}:
        raise CheckFailure("M1837 frozen identity drift")
    policy = value.get("manual_recovery_policy", {})
    if policy != {
            "same_runner_only": True,
            "same_runner_sha256": FIXED_SHA["runner"],
            "proposed_relaunches": 1,
            "automatic_retry": False,
            "original_m1816_bound": True,
            "original_m1816_alone_no_longer_sufficient": True,
            "independent_m1837_source_review_required": True,
            "separately_double_sealed_final_recovery_release_required": True,
            "final_recovery_release_created_now": False,
            "attempt_latch_must_be_absent_before_relaunch": ATTEMPT.relative_to(HW).as_posix(),
            "attempt_latch_must_be_consumed_exactly_once_after_relaunch": True,
            "second_relaunch_forbidden_even_if_recovery_fails": True,
            "caller_must_use_correct_m1815_manifest_sha256": FIXED_SHA["m1815_manifest"]}:
        raise CheckFailure("M1837 manual recovery policy drift")
    hammer = value.get("future_independent_result_hammer", {})
    if hammer != {
            "required": True,
            "must_audit_preflight_quarantine": PREFLIGHT_QUARANTINE.relative_to(HW).as_posix(),
            "must_audit_preflight_failure_sha256": FIXED_SHA["preflight_failure"],
            "must_audit_unique_consumed_attempt": ATTEMPT.relative_to(HW).as_posix(),
            "must_audit_attempt_json_and_both_seals": True,
            "must_prove_exactly_one_consumed_attempt": True,
            "must_audit_canonical_result_or_consumed_failure": True,
            "may_not_hide_or_replace_preflight_failure": True}:
        raise CheckFailure("M1837 future result-hammer obligation drift")
    if value.get("claim_boundary") != CLAIM_BOUNDARY:
        raise CheckFailure("M1837 claim boundary drift")
    execution = value.get("author_execution", {})
    if execution != {
            "source_only": True, "governance_move": 1,
            "failure_deleted": False, "license_queries": 0,
            "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
            "ptpx_runs": 0, "attempts_created": 0, "results_created": 0,
            "final_releases_created": 0}:
        raise CheckFailure("M1837 author execution drift")
    sources = value.get("source_files", [])
    mapping = {}
    for row in sources:
        if type(row) is not dict or set(row) != {"path", "sha256"} or row["path"] in mapping:
            raise CheckFailure("M1837 source inventory malformed")
        mapping[row["path"]] = row["sha256"]
    expected_paths = {CHECKER.relative_to(HW).as_posix(), TEST.relative_to(HW).as_posix()}
    if set(mapping) != expected_paths:
        raise CheckFailure("M1837 source inventory incomplete")
    if enforce_source_hashes:
        for name, digest in mapping.items():
            exact(HW / name, digest)


def verify_contract_double_seal():
    if not CONTRACT.is_file() or CONTRACT.is_symlink():
        raise CheckFailure("M1837 contract absent/nonregular")
    value = strict_json(CONTRACT)
    validate_contract_value(value, enforce_source_hashes=True)
    contract_sha = sha(CONTRACT)
    exact(CONTRACT_SIDECAR, sha(CONTRACT_SIDECAR))
    exact(CONTRACT_OUTER, sha(CONTRACT_OUTER))
    if CONTRACT_SIDECAR.read_text().split() != [contract_sha, CONTRACT.name]:
        raise CheckFailure("M1837 contract sidecar content drift")
    if CONTRACT_OUTER.read_text().split() != [sha(CONTRACT_SIDECAR), CONTRACT_SIDECAR.name]:
        raise CheckFailure("M1837 contract outer content drift")


def validate_sources(contract_text=None):
    verify_original_authority()
    verify_preflight_quarantine()
    if contract_text is None:
        verify_contract_double_seal()
    else:
        validate_contract_value(strict_json_text(contract_text),
                                enforce_source_hashes=False)
    return {
        "status": "PASS_M1837_ONE_MANUAL_RECOVERY_SOURCE",
        "attempt_consumed": False,
        "eda_or_license_run": False,
        "launch_authorized_now": False,
        "final_release_created": False,
    }


def main():
    try:
        result = validate_sources()
    except (CheckFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
