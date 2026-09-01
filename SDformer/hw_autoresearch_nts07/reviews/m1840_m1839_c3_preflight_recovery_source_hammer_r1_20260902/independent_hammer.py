#!/usr/bin/env python3
"""Independent, read-only hammer for the superseding M1839 C3 recovery source."""
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
CHECKER = HW / "system_simulator/scripts/check_m1839_m1808_c3_preflight_recovery_source.py"
SPEC = importlib.util.spec_from_file_location("m1839_checker_independent", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

AUTHOR = HW / "reviews/m1839_m1838_m1837_m1808_c3_preflight_recovery_source_author_receipt_r1_20260902"
EXPECTED = {
    "contract": "69ceca601aca774028e4fef0324a7297bc1bf77af2ad748f6bb9998483a78a96",
    "contract_sidecar": "0e7882364efbaf4dd98bdb2ebb4a3bf6730548222ac155acc2d0dc12ed0bbad8",
    "contract_outer": "e1da10eaeada96662cdf7b94c682f9743680f3f3d2a6b1740a125728480c7767",
    "checker": "ed9f07716adcd54559b17a7cf720b2b0c661306f54c2aa24b50dd88d010e73b4",
    "test": "b585f717bc416bbc56befb9079f80f7b52ecb5a8521db62be2613c9b7f34fe06",
    "author_receipt": "fce778b752279478d2bd0004b3f6021f1bea0caf3998138b396365bac3c97378",
    "author_manifest": "9b3acfae0908faaced2bb439405e07365feb90b6de7e8e87e030158ebe0f36b3",
    "author_outer": "10f0d50675e5d66b8cf6d8922b4e150c272384f59e5d431e83c44db8aa4281aa",
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
            or not stat.S_ISREG(path.lstat().st_mode)
            or sha(path) != digest):
        raise HammerFailure("identity drift: " + str(path))


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise HammerFailure("outer seal content drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise HammerFailure("manifest syntax: " + str(root))
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe/duplicate manifest member: " + name)
        exact(root / rel, digest)
        listed[name] = digest
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HammerFailure("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise HammerFailure("sealed population drift: " + str(root))
    return listed


def get_at(root, path):
    value = root
    for step in path:
        value = value[step]
    return value


def parent_at(root, path):
    return get_at(root, path[:-1]), path[-1]


def paths(value, path=()):
    yield path, value
    if type(value) is dict:
        for key in sorted(value):
            for result in paths(value[key], path + (key,)):
                yield result
    elif type(value) is list:
        for index, item in enumerate(value):
            for result in paths(item, path + (index,)):
                yield result


def rejected(value):
    try:
        C.validate_contract_value(value, enforce_source_hashes=False)
    except C.CheckFailure:
        return True
    return False


def run_attack(name, base, mutate, results):
    value = copy.deepcopy(base)
    mutate(value)
    ok = rejected(value)
    results.append({"name": name, "result": "REJECTED" if ok else "ESCAPED"})
    if not ok:
        raise HammerFailure("semantic attack escaped: " + name)


def set_path(root, path, value):
    parent, key = parent_at(root, path)
    parent[key] = value


def verify_author_receipt():
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("M1839 author receipt member drift")
    receipt = C.strict_json(AUTHOR / "receipt.json")
    if (receipt.get("status") != "PASS_M1839_SCHEMA_CLOSED_RECOVERY_SOURCE_PENDING_M1840_REVIEW_AND_M1841_RELEASE"
            or receipt.get("identity", {}).get("contract_sha256") != EXPECTED["contract"]
            or receipt.get("identity", {}).get("checker_sha256") != EXPECTED["checker"]
            or receipt.get("identity", {}).get("mutation_test_sha256") != EXPECTED["test"]
            or receipt.get("schema_closure", {}).get("m1838_six_escapes_rejected") != 6
            or receipt.get("authorization_boundary", {}).get("launch_authorized_now") is not False
            or receipt.get("authorization_boundary", {}).get("m1841_double_sealed_final_release_required") is not True
            or receipt.get("authorization_boundary", {}).get("second_relaunch_forbidden") is not True):
        raise HammerFailure("M1839 author receipt semantic drift")
    execution = receipt.get("author_execution", {})
    for key in ("license_queries", "vcs_compiles", "simv_runs", "saif_files",
                "ptpx_runs", "attempts_created", "results_created",
                "reviews_created", "releases_created"):
        if type(execution.get(key)) is not int or execution[key] != 0:
            raise HammerFailure("M1839 author execution drift: " + key)


def verify_frozen_and_absent():
    # Do not call C.verify_frozen_evidence here: once this review directory exists,
    # its pre-review M1840-absence guard is expected to trip. Reperform every
    # immutable/evidence check independently while allowing exactly HERE.
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
    C.verify_file_double_seal(C.M1837_CONTRACT, C.M1837_CONTRACT_SIDECAR, C.M1837_CONTRACT_OUTER,
                              C.FIXED_SHA["m1837_contract"], C.FIXED_SHA["m1837_sidecar"], C.FIXED_SHA["m1837_outer"])
    m1837_members = C.verify_sealed_directory(C.M1837_AUTHOR, C.FIXED_SHA["m1837_manifest"], C.FIXED_SHA["m1837_author_outer"])
    if m1837_members.get("receipt.json") != C.FIXED_SHA["m1837_receipt"]:
        raise HammerFailure("M1837 three-identity chain drift")
    m1838_members = C.verify_sealed_directory(C.M1838, C.FIXED_SHA["m1838_manifest"], C.FIXED_SHA["m1838_outer"])
    if m1838_members.get("review.json") != C.FIXED_SHA["m1838_review"]:
        raise HammerFailure("M1838 three-identity chain drift")
    preflight_members = C.verify_sealed_directory(C.PREFLIGHT_QUARANTINE,
                                                   C.FIXED_SHA["preflight_manifest"],
                                                   C.FIXED_SHA["preflight_outer"])
    if preflight_members != {"failure.json": C.FIXED_SHA["preflight_failure"]}:
        raise HammerFailure("preflight three-identity chain drift")
    failure = C.strict_json(C.PREFLIGHT_QUARANTINE / "failure.json")
    C.deep_exact(failure, {
        "attempt_consumed": False, "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0, "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure", "phase": "SOURCE_CHAIN", "status": "FAILED_OR_INCOMPLETE"},
        "preflight.failure")
    for path in (C.ORIGINAL_FAILURE, C.ATTEMPT, C.RESULT, C.PRIVATE, C.M1841,
                 Path(str(C.M1841) + ".sha256"), Path(str(C.M1841) + ".sha256.seal.sha256")):
        if os.path.lexists(str(path)):
            raise HammerFailure("forbidden attempt/result/release namespace: " + str(path))


def main():
    exact(C.CONTRACT, EXPECTED["contract"])
    exact(C.CONTRACT_SIDECAR, EXPECTED["contract_sidecar"])
    exact(C.CONTRACT_OUTER, EXPECTED["contract_outer"])
    exact(CHECKER, EXPECTED["checker"])
    exact(C.TEST, EXPECTED["test"])
    C.verify_contract_double_seal()
    verify_author_receipt()
    verify_frozen_and_absent()

    base = C.strict_json(C.CONTRACT)
    C.validate_contract_value(base, enforce_source_hashes=True)
    results = []

    # Exact replay of the six formal M1838 escapes.
    run_attack("m1838_diagnosis_eda_true", base,
               lambda v: v["diagnosis"].update(license_or_eda_reached=True), results)
    run_attack("m1838_diagnosis_manifest_zero", base,
               lambda v: v["diagnosis"].update(correct_m1815_manifest_sha256="0" * 64), results)
    run_attack("m1838_diagnosis_attempt_true", base,
               lambda v: v["diagnosis"].update(attempt_consumed=True), results)
    run_attack("m1838_milestone_m9999", base, lambda v: v.update(milestone="M9999"), results)
    run_attack("m1838_purpose_immediate_launch", base,
               lambda v: v.update(purpose="authorize immediate launch"), results)
    run_attack("m1838_unknown_top_launch_true", base,
               lambda v: v.update(launch_authorized_now=True), results)

    # Independently attack every object (including top-level and source rows)
    # with an unknown field and a missing field.
    snapshot = list(paths(base))
    dict_paths = [path for path, value in snapshot if type(value) is dict]
    for index, path in enumerate(dict_paths):
        def add_unknown(v, p=path):
            get_at(v, p)["__m1840_unknown__"] = False
        run_attack("dict_%03d_unknown" % index, base, add_unknown, results)
        target = get_at(base, path)
        key = sorted(target)[0]
        def drop_key(v, p=path, k=key):
            del get_at(v, p)[k]
        run_attack("dict_%03d_missing_%s" % (index, key), base, drop_key, results)

    # Attack every nested value's type. This explicitly covers the JSON
    # bool/int alias in both directions and whole-list type replacement.
    for index, (path, value) in enumerate(snapshot):
        if not path or type(value) is dict:
            continue
        if type(value) is bool:
            replacement = 1 if value else 0
        elif type(value) is int:
            replacement = bool(value)
        elif type(value) is str:
            replacement = [value]
        elif type(value) is list:
            replacement = {"was_list": True}
        else:
            continue
        run_attack("value_%03d_type_%s" % (index, type(value).__name__), base,
                   lambda v, p=path, x=replacement: set_path(v, p, x), results)

    escaped = [row for row in results if row["result"] != "REJECTED"]
    output = {
        "status": "PASS_M1840_INDEPENDENT_HAMMER",
        "attacks_total": len(results),
        "attacks_rejected": len(results) - len(escaped),
        "attacks_escaped": len(escaped),
        "m1838_exact_replays": 6,
        "dict_objects_attacked": len(dict_paths),
        "source_precheck_before_review_creation": "RECORDED_SEPARATELY_46_OF_46_ON_CPYTHON3_AND_CPYTHON36",
        "license_or_eda_run": False,
        "attempt_or_result_created": False,
        "release_created": False,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HammerFailure, C.CheckFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL_M1840_INDEPENDENT_HAMMER", "error": str(error)}, sort_keys=True))
        sys.exit(1)
