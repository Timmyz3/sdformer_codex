#!/usr/bin/env python3
"""Strict schema-closed checker for superseding M1839 recovery source."""
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
TEST = HW / "system_simulator/tests/test_m1839_m1808_c3_preflight_recovery_source.py"
CONTRACT = HW / "contracts/m1839_m1838_m1837_m1808_c3_preflight_recovery_source_contract_r1_20260902.json"
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
M1837_CONTRACT = HW / "contracts/m1837_m1808_c3_preflight_recovery_source_contract_r1_20260902.json"
M1837_CONTRACT_SIDECAR = Path(str(M1837_CONTRACT) + ".sha256")
M1837_CONTRACT_OUTER = Path(str(M1837_CONTRACT) + ".sha256.seal.sha256")
M1837_AUTHOR = HW / "reviews/m1837_m1808_c3_preflight_recovery_source_author_receipt_r1_20260902"
M1838 = HW / "reviews/m1838_m1837_c3_preflight_recovery_source_hammer_r1_20260902"

PREFLIGHT_QUARANTINE = HW / "results/m1808_c3_mapped_energy_r1_20260902.preflight_rejected_source_chain_governance_quarantine"
ORIGINAL_FAILURE = HW / "results/m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
ATTEMPT = HW / "results/.m1808_c3_mapped_energy_attempt_consumed"
RESULT = HW / "results/m1808_c3_mapped_energy_r1_20260902"
PRIVATE = HW / "results/m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"
M1840 = HW / "reviews/m1840_m1839_c3_preflight_recovery_source_hammer_r1_20260902"
M1841 = HW / "contracts/m1841_m1840_m1839_m1808_c3_preflight_recovery_launch_release_r1_20260902.json"

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
    "m1816": "c948d79fb6fd93a2d4f33b6c16c83c33b6a2985cdaef7d928e63fc292dc3549f",
    "m1816_sidecar": "33e2391ad4c7952b6e04371cc81fdaf378071bd735d49ac614ccfa601aecb1b2",
    "m1816_outer": "fa2e6415a991449b3f4329ea27b8e3f8f04ded4183a9635ff6035f860a63b38b",
    "m1837_contract": "7257c39b9d68ecc92af36124b490d2f46b97ec7d961fc218abdf8880533382ab",
    "m1837_sidecar": "28ad3eb39b903cff1634fd2d7650d58e4ba83e982eda220b31d9f35e87d20c48",
    "m1837_outer": "cff64fc604ec10f5eebcc5484f429771a0254b17f2c479ad321fe7473f42b410",
    "m1837_receipt": "1b5d39efa39d8762a55d4ac9e29c608086ef2d1e03a8e65ca2892e9f83371208",
    "m1837_manifest": "1a3780999ac0d7847d65c13db5ab048d563895339509d359758390c73ece6ff4",
    "m1837_author_outer": "01467cc1d3b228e128c579c577fc5952a963dd5814fb0f749c5d0c50983658f7",
    "m1838_review": "b7ab0f6d37843fb8ba839d77f58caf77e6e02aab189c0812d4e5f6642a8209dd",
    "m1838_manifest": "ffe6ac9a09b5d53ec135851da2502bd14df85b9f7c7efa0c84c349db71d9d323",
    "m1838_outer": "e89bcba894ccc565b8e4d369dafa4b7a76f29e4433a3295848cb2a7f44019b7f",
    "preflight_failure": "ea9d08303dd29196a761c1e9927e5aa148a5f8746e1d5b4a64f354d66c74eda8",
    "preflight_manifest": "e243c0f10d810b1b5d39523ad479a1df2d751a3f139d7eae944072d2788eb856",
    "preflight_outer": "d9824a782b5ee5f1ba116abe2c7719a24579815798ee5b1d48b342de38784124",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PURPOSE = (
    "Supersede the sealed-but-rejected M1837 recovery source by closing all "
    "M1838 schema escapes while preserving the zero-attempt M1808 preflight "
    "quarantine and proposing exactly one same-runner manual recovery only "
    "after independent M1840 review and a separate double-sealed M1841 release.")

DIAGNOSIS = {
    "machine_observed": "verify_authority failed during SOURCE_CHAIN before attempt consumption",
    "operator_reported": "the supplied M1815 manifest SHA had one extra trailing character",
    "correct_m1815_manifest_sha256": FIXED_SHA["m1815_manifest"],
    "attempt_consumed": False,
    "license_or_eda_reached": False,
    "diagnosis_is_not_a_tool_result": True,
}

M1838_ESCAPES = [
    "diagnosis.license_or_eda_reached=true",
    "diagnosis.correct_m1815_manifest_sha256=64_zeroes",
    "diagnosis.attempt_consumed=true",
    "milestone=M9999",
    "purpose=authorize immediate launch",
    "unknown top-level launch_authorized_now=true",
]

TOP_KEYS = {
    "schema", "milestone", "status", "purpose", "diagnosis",
    "supersession", "m1838_failed_review", "preflight_rejection_evidence",
    "frozen_identity", "manual_recovery_policy",
    "future_independent_result_hammer", "claim_boundary",
    "author_execution", "source_files",
}

CLAIM_BOUNDARY = dict((name, False) for name in (
    "launch_authorized_now", "m1837_authority_valid", "m1838_review_passed",
    "m1840_review_created", "m1841_release_created", "license_queried",
    "attempt_consumed", "vcs_compile", "simv_run", "saif_generated",
    "ptpx_run", "mapped_vcs", "production_saif", "component_power",
    "component_energy", "energy_per_frame", "speedup", "system_speedup",
    "paper_ppa_ready", "paper_citable", "headline"))


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
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise CheckFailure("identity drift: " + str(path))


def strict_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckFailure("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise CheckFailure("JSON root must be object")
    return value


def strict_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CheckFailure("JSON absent/nonregular: " + str(path))
    return strict_json_text(path.read_text())


def deep_exact(actual, expected, where):
    if type(actual) is not type(expected):
        raise CheckFailure(where + " type drift")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise CheckFailure(where + " exact-key set drift")
        for key in expected:
            deep_exact(actual[key], expected[key], where + "." + key)
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise CheckFailure(where + " list length drift")
        for index, item in enumerate(expected):
            deep_exact(actual[index], item, where + "[" + str(index) + "]")
    elif actual != expected:
        raise CheckFailure(where + " exact-value drift")


def verify_sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise CheckFailure("sealed directory invalid: " + str(root))
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha); exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise CheckFailure("outer seal drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise CheckFailure("manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise CheckFailure("manifest unsafe/duplicate member")
        exact(root / rel, digest); listed[name] = digest
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CheckFailure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise CheckFailure("sealed population drift")
    return listed


def verify_file_double_seal(path, sidecar, outer, file_sha, sidecar_sha, outer_sha):
    exact(path, file_sha); exact(sidecar, sidecar_sha); exact(outer, outer_sha)
    if Path(sidecar).read_text().split() != [file_sha, Path(path).name]:
        raise CheckFailure("file sidecar drift")
    if Path(outer).read_text().split() != [sidecar_sha, Path(sidecar).name]:
        raise CheckFailure("file outer drift")


def verify_frozen_evidence():
    for path, key in (
            (RUNNER, "runner"), (M1808_CHECKER, "m1808_checker"),
            (M1808_TEST, "m1808_test"), (M1808_CONTRACT, "m1808_contract"),
            (M1808_CONTRACT_SIDECAR, "m1808_contract_sidecar"),
            (M1808_CONTRACT_OUTER, "m1808_contract_outer"),
            (HW / "docs/359_DATE终局冻结_20260813.md", "docs359")):
        exact(path, FIXED_SHA[key])
    verify_sealed_directory(M1815, FIXED_SHA["m1815_manifest"], FIXED_SHA["m1815_outer"])
    exact(M1815 / "review.json", FIXED_SHA["m1815_review"])
    verify_file_double_seal(M1816, M1816_SIDECAR, M1816_OUTER,
                            FIXED_SHA["m1816"], FIXED_SHA["m1816_sidecar"],
                            FIXED_SHA["m1816_outer"])
    release = strict_json(M1816)
    if (release.get("status") != "AUTHORIZE_ONE_FRESH_M1808_C3_MAPPED_ENERGY_CAMPAIGN"
            or release.get("identity", {}).get("runner_sha256") != FIXED_SHA["runner"]
            or release.get("identity", {}).get("source_review_manifest_sha256") != FIXED_SHA["m1815_manifest"]):
        raise CheckFailure("M1816 semantic drift")
    verify_file_double_seal(M1837_CONTRACT, M1837_CONTRACT_SIDECAR,
                            M1837_CONTRACT_OUTER, FIXED_SHA["m1837_contract"],
                            FIXED_SHA["m1837_sidecar"], FIXED_SHA["m1837_outer"])
    members = verify_sealed_directory(M1837_AUTHOR, FIXED_SHA["m1837_manifest"],
                                      FIXED_SHA["m1837_author_outer"])
    if members.get("receipt.json") != FIXED_SHA["m1837_receipt"]:
        raise CheckFailure("M1837 author receipt seal drift")
    members = verify_sealed_directory(M1838, FIXED_SHA["m1838_manifest"],
                                      FIXED_SHA["m1838_outer"])
    if members.get("review.json") != FIXED_SHA["m1838_review"]:
        raise CheckFailure("M1838 review seal drift")
    review = strict_json(M1838 / "review.json")
    if (review.get("status") != "FAIL_M1838_M1837_C3_PREFLIGHT_RECOVERY_SOURCE_HAMMER__P1_SEMANTIC_VALIDATION_ESCAPES__NO_RELEASE"
            or review.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or review.get("finding", {}).get("reproduced_escapes") != M1838_ESCAPES):
        raise CheckFailure("M1838 FAIL semantics drift")
    members = verify_sealed_directory(PREFLIGHT_QUARANTINE,
                                      FIXED_SHA["preflight_manifest"],
                                      FIXED_SHA["preflight_outer"])
    if members != {"failure.json": FIXED_SHA["preflight_failure"]}:
        raise CheckFailure("preflight quarantine population drift")
    failure = strict_json(PREFLIGHT_QUARANTINE / "failure.json")
    expected_failure = {
        "attempt_consumed": False, "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure", "phase": "SOURCE_CHAIN",
        "status": "FAILED_OR_INCOMPLETE"}
    deep_exact(failure, expected_failure, "preflight.failure")
    for path in (ORIGINAL_FAILURE, ATTEMPT, RESULT, PRIVATE, M1840, M1841,
                 Path(str(M1841) + ".sha256"),
                 Path(str(M1841) + ".sha256.seal.sha256")):
        if os.path.lexists(str(path)):
            raise CheckFailure("forbidden pre-review/pre-release namespace: " + str(path))


def expected_static_contract():
    return {
        "schema": "m1839_m1838_m1837_m1808_c3_preflight_recovery_source_contract_r1_v1",
        "milestone": "M1839",
        "status": "SOURCE_ONLY__M1838_P1_REPAIRED__M1840_REVIEW_AND_M1841_RELEASE_REQUIRED__NO_EDA",
        "purpose": PURPOSE,
        "diagnosis": DIAGNOSIS,
        "supersession": {
            "supersedes_m1837_without_modifying_it": True,
            "m1837_contract_sha256": FIXED_SHA["m1837_contract"],
            "m1837_author_receipt_sha256": FIXED_SHA["m1837_receipt"],
            "m1838_formal_fail_bound": True,
            "m1837_or_m1838_authorizes_launch": False,
        },
        "m1838_failed_review": {
            "path": M1838.relative_to(HW).as_posix(),
            "review_sha256": FIXED_SHA["m1838_review"],
            "manifest_sha256": FIXED_SHA["m1838_manifest"],
            "outer_file_sha256": FIXED_SHA["m1838_outer"],
            "status": "FAIL_M1838_M1837_C3_PREFLIGHT_RECOVERY_SOURCE_HAMMER__P1_SEMANTIC_VALIDATION_ESCAPES__NO_RELEASE",
            "severity_counts": {"p0": 0, "p1": 1, "p2": 0},
            "escape_count": 6,
            "reproduced_escapes": M1838_ESCAPES,
        },
        "preflight_rejection_evidence": {
            "quarantine": PREFLIGHT_QUARANTINE.relative_to(HW).as_posix(),
            "failure_json_sha256": FIXED_SHA["preflight_failure"],
            "manifest_sha256": FIXED_SHA["preflight_manifest"],
            "outer_file_sha256": FIXED_SHA["preflight_outer"],
            "phase": "SOURCE_CHAIN", "attempt_consumed": False,
            "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
            "ptpx_runs": 0, "automatic_retry": False,
            "preserved_not_deleted": True,
            "attempt_result_private_absent": True,
        },
        "frozen_identity": {
            "runner_sha256": FIXED_SHA["runner"],
            "m1808_checker_sha256": FIXED_SHA["m1808_checker"],
            "m1808_test_sha256": FIXED_SHA["m1808_test"],
            "m1808_source_contract_sha256": FIXED_SHA["m1808_contract"],
            "m1815_review_sha256": FIXED_SHA["m1815_review"],
            "m1815_correct_manifest_sha256": FIXED_SHA["m1815_manifest"],
            "m1815_outer_file_sha256": FIXED_SHA["m1815_outer"],
            "m1816_release_sha256": FIXED_SHA["m1816"],
            "m1816_sidecar_sha256": FIXED_SHA["m1816_sidecar"],
            "m1816_outer_file_sha256": FIXED_SHA["m1816_outer"],
            "docs359_sha256": FIXED_SHA["docs359"],
        },
        "manual_recovery_policy": {
            "same_runner_only": True,
            "same_runner_sha256": FIXED_SHA["runner"],
            "proposed_relaunches": 1, "automatic_retry": False,
            "original_m1816_alone_insufficient": True,
            "m1840_review_required": M1840.relative_to(HW).as_posix(),
            "m1841_double_sealed_release_required": M1841.relative_to(HW).as_posix(),
            "m1840_or_m1841_created_now": False,
            "attempt_latch_must_be_absent_before_relaunch": ATTEMPT.relative_to(HW).as_posix(),
            "attempt_latch_must_be_consumed_exactly_once_after_relaunch": True,
            "second_relaunch_forbidden_even_if_recovery_fails": True,
            "caller_must_use_correct_m1815_manifest_sha256": FIXED_SHA["m1815_manifest"],
        },
        "future_independent_result_hammer": {
            "required": True,
            "must_audit_preflight_quarantine": PREFLIGHT_QUARANTINE.relative_to(HW).as_posix(),
            "must_audit_preflight_failure_sha256": FIXED_SHA["preflight_failure"],
            "must_audit_unique_consumed_attempt": ATTEMPT.relative_to(HW).as_posix(),
            "must_audit_attempt_json_and_both_seals": True,
            "must_prove_exactly_one_consumed_attempt": True,
            "must_audit_canonical_result_or_consumed_failure": True,
            "may_not_hide_or_replace_preflight_failure": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "author_execution": {
            "source_only": True, "governance_moves": 0,
            "failure_deleted": False, "license_queries": 0,
            "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
            "ptpx_runs": 0, "attempts_created": 0, "results_created": 0,
            "reviews_created": 0, "releases_created": 0,
        },
    }


def validate_contract_value(value, enforce_source_hashes=True):
    if type(value) is not dict or set(value) != TOP_KEYS:
        raise CheckFailure("contract top-level exact-key set drift")
    expected = expected_static_contract()
    for key in TOP_KEYS - {"source_files"}:
        deep_exact(value[key], expected[key], "contract." + key)
    sources = value["source_files"]
    if type(sources) is not list or len(sources) != 2:
        raise CheckFailure("contract.source_files exact list drift")
    mapping = {}
    for index, row in enumerate(sources):
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise CheckFailure("contract.source_files[" + str(index) + "] exact-key/type drift")
        if type(row["path"]) is not str or type(row["sha256"]) is not str:
            raise CheckFailure("contract.source_files member type drift")
        if row["path"] in mapping or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise CheckFailure("contract.source_files duplicate/hash drift")
        mapping[row["path"]] = row["sha256"]
    expected_paths = {CHECKER.relative_to(HW).as_posix(), TEST.relative_to(HW).as_posix()}
    if set(mapping) != expected_paths:
        raise CheckFailure("contract.source_files population drift")
    if enforce_source_hashes:
        for name, digest in mapping.items():
            exact(HW / name, digest)


def verify_contract_double_seal():
    value = strict_json(CONTRACT)
    validate_contract_value(value, enforce_source_hashes=True)
    contract_sha = sha(CONTRACT)
    if (not CONTRACT_SIDECAR.is_file() or CONTRACT_SIDECAR.is_symlink()
            or CONTRACT_SIDECAR.read_text().split() != [contract_sha, CONTRACT.name]
            or not CONTRACT_OUTER.is_file() or CONTRACT_OUTER.is_symlink()
            or CONTRACT_OUTER.read_text().split() != [sha(CONTRACT_SIDECAR), CONTRACT_SIDECAR.name]):
        raise CheckFailure("M1839 contract double seal drift")


def validate_sources(contract_text=None):
    verify_frozen_evidence()
    if contract_text is None:
        verify_contract_double_seal()
    else:
        validate_contract_value(strict_json_text(contract_text),
                                enforce_source_hashes=False)
    return {"status": "PASS_M1839_SCHEMA_CLOSED_RECOVERY_SOURCE",
            "m1838_six_escapes_closed": True,
            "attempt_consumed": False, "eda_or_license_run": False,
            "launch_authorized_now": False, "m1841_release_created": False}


def main():
    try:
        result = validate_sources()
    except (CheckFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    sys.exit(main())
