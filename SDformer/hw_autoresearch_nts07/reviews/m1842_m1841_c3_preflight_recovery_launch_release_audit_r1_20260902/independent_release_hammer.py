#!/usr/bin/env python3
"""Read-only independent semantic and identity audit of the M1841 release."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1841_m1840_m1839_m1808_c3_preflight_recovery_launch_release_r1_20260902.json"
SIDECAR = Path(str(RELEASE) + ".sha256")
OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
M1839_CHECKER = HW / "system_simulator/scripts/check_m1839_m1808_c3_preflight_recovery_source.py"
SPEC = importlib.util.spec_from_file_location("m1839_checker_for_m1842", str(M1839_CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

M1839_AUTHOR = HW / "reviews/m1839_m1838_m1837_m1808_c3_preflight_recovery_source_author_receipt_r1_20260902"
M1840 = HW / "reviews/m1840_m1839_c3_preflight_recovery_source_hammer_r1_20260902"

SHA = {
    "release": "68698e10cb2e625b949d98f157d70ca896546aef3149a96bae285ede2f09c6da",
    "release_sidecar": "3cb4c33b7daeabe28d3dda989925515d2af302880dc78425cfa64797786560d8",
    "release_outer": "7d31e71ab305ef0d37f2ab615bcac2676f64b24e7ea7a2f16af229dbabcbf53f",
    "m1839_contract": "69ceca601aca774028e4fef0324a7297bc1bf77af2ad748f6bb9998483a78a96",
    "m1839_sidecar": "0e7882364efbaf4dd98bdb2ebb4a3bf6730548222ac155acc2d0dc12ed0bbad8",
    "m1839_outer": "e1da10eaeada96662cdf7b94c682f9743680f3f3d2a6b1740a125728480c7767",
    "m1839_checker": "ed9f07716adcd54559b17a7cf720b2b0c661306f54c2aa24b50dd88d010e73b4",
    "m1839_test": "b585f717bc416bbc56befb9079f80f7b52ecb5a8521db62be2613c9b7f34fe06",
    "m1839_author_receipt": "fce778b752279478d2bd0004b3f6021f1bea0caf3998138b396365bac3c97378",
    "m1839_author_manifest": "9b3acfae0908faaced2bb439405e07365feb90b6de7e8e87e030158ebe0f36b3",
    "m1839_author_outer": "10f0d50675e5d66b8cf6d8922b4e150c272384f59e5d431e83c44db8aa4281aa",
    "m1840_review": "b833ea7fa77a11194bc720cb8a35070b7b886b7bd1640e70aacdad3b5d337497",
    "m1840_manifest": "d9cf595f92c91542581a99301f2474c5b2166d3924f8cc8ca560e5070bb3803d",
    "m1840_outer": "916a6907fcaaef4c0e53b07ff06d5c83d7b5d1f339590e097b88191a8724f3f9",
}


class AuditFailure(RuntimeError):
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
        raise AuditFailure("identity drift: " + str(path))


def strict_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AuditFailure("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AuditFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise AuditFailure("JSON root must be object")
    return value


def strict_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise AuditFailure("JSON absent/nonregular: " + str(path))
    return strict_json_text(path.read_text())


def deep_exact(actual, expected, where):
    if type(actual) is not type(expected):
        raise AuditFailure(where + " type drift")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise AuditFailure(where + " exact-key set drift")
        for key in expected:
            deep_exact(actual[key], expected[key], where + "." + key)
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise AuditFailure(where + " list length drift")
        for index, item in enumerate(expected):
            deep_exact(actual[index], item, where + "[" + str(index) + "]")
    elif actual != expected:
        raise AuditFailure(where + " exact-value drift")


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise AuditFailure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise AuditFailure("outer content drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise AuditFailure("manifest syntax: " + str(root))
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise AuditFailure("unsafe/duplicate member: " + name)
        exact(root / rel, digest)
        listed[name] = digest
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditFailure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise AuditFailure("sealed population drift: " + str(root))
    return listed


def expected_release():
    runner = C.FIXED_SHA["runner"]
    source = C.FIXED_SHA["m1808_contract"]
    review = C.FIXED_SHA["m1815_review"]
    manifest = C.FIXED_SHA["m1815_manifest"]
    m1815_outer = C.FIXED_SHA["m1815_outer"]
    m1816 = C.FIXED_SHA["m1816"]
    m1816_sidecar = C.FIXED_SHA["m1816_sidecar"]
    m1816_outer = C.FIXED_SHA["m1816_outer"]
    return {
        "schema": "m1841_m1840_m1839_m1808_c3_preflight_recovery_launch_release_r1_v1",
        "milestone": "M1841",
        "created_utc": "2026-09-01T22:15:55Z",
        "status": "AUTHORIZE_ONE_MANUAL_M1808_C3_MAPPED_ENERGY_RECOVERY_AFTER_ZERO_ATTEMPT_PREFLIGHT_REJECTION",
        "purpose": "Authorize exactly one same-runner M1808 production attempt after independently proving that the earlier caller-pin rejection consumed no attempt and invoked no EDA; preserve that rejection permanently and forbid any second relaunch.",
        "identity": {
            "m1808_runner_sha256": runner,
            "m1808_source_contract_sha256": source,
            "m1815_review_sha256": review,
            "m1815_correct_manifest_sha256": manifest,
            "m1815_outer_file_sha256": m1815_outer,
            "m1816_release_sha256": m1816,
            "m1816_sidecar_sha256": m1816_sidecar,
            "m1816_outer_file_sha256": m1816_outer,
            "m1839_contract_sha256": SHA["m1839_contract"],
            "m1839_contract_sidecar_sha256": SHA["m1839_sidecar"],
            "m1839_contract_outer_file_sha256": SHA["m1839_outer"],
            "m1839_checker_sha256": SHA["m1839_checker"],
            "m1839_test_sha256": SHA["m1839_test"],
            "m1839_author_receipt_sha256": SHA["m1839_author_receipt"],
            "m1839_author_manifest_sha256": SHA["m1839_author_manifest"],
            "m1839_author_outer_file_sha256": SHA["m1839_author_outer"],
            "m1840_review_sha256": SHA["m1840_review"],
            "m1840_manifest_sha256": SHA["m1840_manifest"],
            "m1840_outer_file_sha256": SHA["m1840_outer"],
            "docs359_sha256": C.FIXED_SHA["docs359"],
        },
        "preserved_preflight_rejection": {
            "path": C.PREFLIGHT_QUARANTINE.relative_to(HW).as_posix(),
            "failure_json_sha256": C.FIXED_SHA["preflight_failure"],
            "manifest_sha256": C.FIXED_SHA["preflight_manifest"],
            "outer_file_sha256": C.FIXED_SHA["preflight_outer"],
            "phase": "SOURCE_CHAIN", "attempt_consumed": False,
            "license_or_eda_reached": False, "vcs_compiles": 0,
            "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0,
            "preserved_not_deleted": True,
        },
        "superseded_governance": {
            "m1837_contract_sha256": C.FIXED_SHA["m1837_contract"],
            "m1838_failed_review_sha256": C.FIXED_SHA["m1838_review"],
            "m1838_manifest_sha256": C.FIXED_SHA["m1838_manifest"],
            "m1838_outer_file_sha256": C.FIXED_SHA["m1838_outer"],
            "m1837_or_m1838_authorizes_launch": False,
        },
        "prelaunch_namespaces": {
            "attempt_absent": C.ATTEMPT.relative_to(HW).as_posix(),
            "canonical_result_absent": C.RESULT.relative_to(HW).as_posix(),
            "ordinary_failure_absent": C.ORIGINAL_FAILURE.relative_to(HW).as_posix(),
            "private_build_absent": C.PRIVATE.relative_to(HW).as_posix(),
        },
        "correct_caller_pins": {
            "M1808_EXPECTED_RUNNER_SHA256": runner,
            "M1808_EXPECTED_SOURCE_CONTRACT_SHA256": source,
            "M1808_EXPECTED_M1815_MANIFEST_SHA256": manifest,
            "M1808_EXPECTED_M1815_OUTER_FILE_SHA256": m1815_outer,
            "M1808_EXPECTED_M1815_REVIEW_SHA256": review,
            "M1808_EXPECTED_M1816_RELEASE_SHA256": m1816,
            "M1808_EXPECTED_M1816_SIDECAR_SHA256": m1816_sidecar,
            "M1808_EXPECTED_M1816_OUTER_FILE_SHA256": m1816_outer,
        },
        "execution_budget": {
            "manual_relaunches": 1, "production_attempts": 1,
            "vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,
            "ptpx_runs": 1, "automatic_retry": False,
            "second_relaunch": False, "reuse_prior_simv_saif_ptpx": False,
        },
        "authorization": {
            "launch_exact_same_m1808_runner_once": True,
            "manual_recovery_only": True,
            "m1816_alone_is_insufficient": True,
            "m1839_and_m1840_bound": True,
            "publish_only_after_all_m1808_gates": True,
            "independent_result_hammer_required": True,
            "result_hammer_must_audit_preserved_preflight_and_unique_consumed_attempt": True,
            "automatic_retry": False,
            "second_relaunch_permitted": False,
        },
        "prelaunch_claim_boundary": dict((name, False) for name in (
            "mapped_vcs", "production_saif", "component_power",
            "component_energy", "energy_per_frame", "speedup",
            "system_speedup", "paper_ppa_ready", "paper_citable", "headline")),
    }


def validate_release(value):
    deep_exact(value, expected_release(), "release")


def verify_release_seals():
    exact(RELEASE, SHA["release"])
    exact(SIDECAR, SHA["release_sidecar"])
    exact(OUTER, SHA["release_outer"])
    if SIDECAR.read_text().split() != [SHA["release"], RELEASE.name]:
        raise AuditFailure("release sidecar content drift")
    if OUTER.read_text().split() != [SHA["release_sidecar"], SIDECAR.name]:
        raise AuditFailure("release outer content drift")


def verify_upstream_and_namespaces():
    # M1808/M1815/M1816 frozen authority.
    for path, key in ((C.RUNNER, "runner"), (C.M1808_CHECKER, "m1808_checker"),
                      (C.M1808_TEST, "m1808_test"), (C.M1808_CONTRACT, "m1808_contract"),
                      (C.M1808_CONTRACT_SIDECAR, "m1808_contract_sidecar"),
                      (C.M1808_CONTRACT_OUTER, "m1808_contract_outer"),
                      (HW / "docs/359_DATE终局冻结_20260813.md", "docs359")):
        exact(path, C.FIXED_SHA[key])
    C.verify_sealed_directory(C.M1815, C.FIXED_SHA["m1815_manifest"], C.FIXED_SHA["m1815_outer"])
    exact(C.M1815 / "review.json", C.FIXED_SHA["m1815_review"])
    C.verify_file_double_seal(C.M1816, C.M1816_SIDECAR, C.M1816_OUTER,
                              C.FIXED_SHA["m1816"], C.FIXED_SHA["m1816_sidecar"], C.FIXED_SHA["m1816_outer"])

    # M1837 FAIL source and M1838 formal FAIL remain immutable and bound.
    C.verify_file_double_seal(C.M1837_CONTRACT, C.M1837_CONTRACT_SIDECAR, C.M1837_CONTRACT_OUTER,
                              C.FIXED_SHA["m1837_contract"], C.FIXED_SHA["m1837_sidecar"], C.FIXED_SHA["m1837_outer"])
    m1837 = C.verify_sealed_directory(C.M1837_AUTHOR, C.FIXED_SHA["m1837_manifest"], C.FIXED_SHA["m1837_author_outer"])
    if m1837.get("receipt.json") != C.FIXED_SHA["m1837_receipt"]:
        raise AuditFailure("M1837 identity chain drift")
    m1838 = C.verify_sealed_directory(C.M1838, C.FIXED_SHA["m1838_manifest"], C.FIXED_SHA["m1838_outer"])
    if m1838.get("review.json") != C.FIXED_SHA["m1838_review"]:
        raise AuditFailure("M1838 identity chain drift")

    # M1839 repaired source and author receipt.
    exact(C.CONTRACT, SHA["m1839_contract"])
    exact(C.CONTRACT_SIDECAR, SHA["m1839_sidecar"])
    exact(C.CONTRACT_OUTER, SHA["m1839_outer"])
    exact(M1839_CHECKER, SHA["m1839_checker"])
    exact(C.TEST, SHA["m1839_test"])
    C.verify_contract_double_seal()
    m1839 = sealed_directory(M1839_AUTHOR, SHA["m1839_author_manifest"], SHA["m1839_author_outer"])
    if m1839.get("receipt.json") != SHA["m1839_author_receipt"]:
        raise AuditFailure("M1839 author identity chain drift")

    # M1840 independent PASS.
    m1840 = sealed_directory(M1840, SHA["m1840_manifest"], SHA["m1840_outer"])
    if m1840.get("review.json") != SHA["m1840_review"]:
        raise AuditFailure("M1840 review identity chain drift")
    review = strict_json(M1840 / "review.json")
    if (review.get("status") != "PASS_M1840_M1839_C3_PREFLIGHT_RECOVERY_SOURCE_HAMMER__M1841_FINAL_RELEASE_REQUIRED__NO_LAUNCH"
            or review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}
            or review.get("verdict", {}).get("m1841_final_double_sealed_release_required") is not True):
        raise AuditFailure("M1840 PASS semantics drift")

    # Preserved zero-attempt preflight failure.
    preflight = C.verify_sealed_directory(C.PREFLIGHT_QUARANTINE,
                                           C.FIXED_SHA["preflight_manifest"],
                                           C.FIXED_SHA["preflight_outer"])
    if preflight != {"failure.json": C.FIXED_SHA["preflight_failure"]}:
        raise AuditFailure("preflight identity/population drift")
    failure = strict_json(C.PREFLIGHT_QUARANTINE / "failure.json")
    deep_exact(failure, {
        "attempt_consumed": False, "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0, "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure", "phase": "SOURCE_CHAIN", "status": "FAILED_OR_INCOMPLETE"},
        "preflight.failure")
    for path in (C.ATTEMPT, C.RESULT, C.ORIGINAL_FAILURE, C.PRIVATE):
        if os.path.lexists(str(path)):
            raise AuditFailure("prelaunch namespace not absent: " + str(path))


def paths(value, path=()):
    yield path, value
    if type(value) is dict:
        for key in sorted(value):
            for item in paths(value[key], path + (key,)):
                yield item
    elif type(value) is list:
        for index, child in enumerate(value):
            for item in paths(child, path + (index,)):
                yield item


def get_at(root, path):
    value = root
    for step in path:
        value = value[step]
    return value


def set_at(root, path, value):
    parent = get_at(root, path[:-1])
    parent[path[-1]] = value


def run_attack(name, base, mutate, results):
    value = copy.deepcopy(base)
    mutate(value)
    try:
        validate_release(value)
    except AuditFailure:
        results.append({"name": name, "result": "REJECTED"})
        return
    raise AuditFailure("release mutation escaped: " + name)


def main():
    verify_release_seals()
    value = strict_json(RELEASE)
    validate_release(value)
    verify_upstream_and_namespaces()
    results = []

    explicit = [
        ("wrong_status", lambda v: v.update(status="AUTHORIZE_UNLIMITED")),
        ("wrong_caller_manifest", lambda v: v["correct_caller_pins"].update(M1808_EXPECTED_M1815_MANIFEST_SHA256="0" * 64)),
        ("preflight_attempt_true", lambda v: v["preserved_preflight_rejection"].update(attempt_consumed=True)),
        ("preflight_eda_true", lambda v: v["preserved_preflight_rejection"].update(license_or_eda_reached=True)),
        ("manual_relaunches_two", lambda v: v["execution_budget"].update(manual_relaunches=2)),
        ("production_attempts_two", lambda v: v["execution_budget"].update(production_attempts=2)),
        ("automatic_retry_true", lambda v: v["execution_budget"].update(automatic_retry=True)),
        ("second_relaunch_true", lambda v: v["execution_budget"].update(second_relaunch=True)),
        ("authorization_second_true", lambda v: v["authorization"].update(second_relaunch_permitted=True)),
        ("drop_result_hammer_obligation", lambda v: v["authorization"].update(result_hammer_must_audit_preserved_preflight_and_unique_consumed_attempt=False)),
    ]
    for name, mutate in explicit:
        run_attack("explicit_" + name, value, mutate, results)

    snapshot = list(paths(value))
    dict_paths = [path for path, item in snapshot if type(item) is dict]
    for index, path in enumerate(dict_paths):
        run_attack("dict_%02d_unknown" % index, value,
                   lambda v, p=path: get_at(v, p).update(__m1842_unknown__=False), results)
        key = sorted(get_at(value, path))[0]
        run_attack("dict_%02d_missing_%s" % (index, key), value,
                   lambda v, p=path, k=key: get_at(v, p).pop(k), results)

    for index, (path, item) in enumerate(snapshot):
        if not path or type(item) is dict:
            continue
        if type(item) is bool:
            replacement = 1 if item else 0
        elif type(item) is int:
            replacement = bool(item)
        elif type(item) is str:
            replacement = [item]
        elif type(item) is list:
            replacement = {"was_list": True}
        else:
            continue
        run_attack("value_%03d_type_%s" % (index, type(item).__name__), value,
                   lambda v, p=path, x=replacement: set_at(v, p, x), results)

    duplicate = RELEASE.read_text().replace('"milestone": "M1841",',
                                             '"milestone": "M1841",\n  "milestone": "M9999",', 1)
    try:
        strict_json_text(duplicate)
    except AuditFailure:
        results.append({"name": "duplicate_json_key", "result": "REJECTED"})
    else:
        raise AuditFailure("duplicate JSON key escaped")

    output = {
        "status": "PASS_M1842_INDEPENDENT_RELEASE_AUDIT",
        "attacks_total": len(results),
        "attacks_rejected": len(results),
        "attacks_escaped": 0,
        "dict_objects_attacked": len(dict_paths),
        "release_double_sealed": True,
        "prelaunch_namespaces_absent": True,
        "license_or_eda_run": False,
        "attempt_or_result_created": False,
        "release_modified": False,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditFailure, C.CheckFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL_M1842_INDEPENDENT_RELEASE_AUDIT", "error": str(error)}, sort_keys=True))
        sys.exit(1)
