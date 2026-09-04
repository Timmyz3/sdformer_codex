#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author final launch hammer for the inert M1508 release.

No VCS, simv, EDA, license query, SSH, GPU work, or canonical attempt is
performed.  The program validates identities and runs Python-only tests.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1506_m1497_c1_active_next_oracle_release_safe_successor_one_shot.py"
CHECKER = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/check_m1506_source.py"
TESTS = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/test_m1506_source.py"
CONTRACT = HW / "contracts/m1506_c1_active_next_oracle_release_safe_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1506_c1_active_next_oracle_release_safe_successor_source_author_r1_20260831"
HAMMER = HW / "reviews/m1507_m1506_c1_active_next_oracle_release_safe_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1508_m1507_m1506_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
M1498 = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
PRECHECK = HERE / "freshness_precheck.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    "runner": "9613922eb3aec2c7fe0efa69cafb4fb8337009b26686435f44cc139c774317cc",
    "checker": "1cb79e04fbbbcb76d914567d20eb5ad1d595a128a12db5d7da106a241fb0320f",
    "tests": "f4c1dd7211d84eef5b469e23d1ee58db0076f05b75e4f0424a55f6550392c58c",
    "contract": "fb5d5d4d8d5e7fcd427265f2770a544eb1de1ab01385262f63469a61ab524346",
    "author_review": "ea526e6b1988d4d96fd301f9de38c1d6faf0563057b1004d8b15ce7ac339bf92",
    "author_manifest": "19927c78074fc489c26d64b9707cf7d9a0a8499858bfb6d3f0771f10a36c5bfd",
    "author_outer": "1bc99841bfc2a01b81e11a30f72a02935d89a41bce862e8bdfb8b2c5a32a96ad",
    "hammer_review": "6ebfc8f3db944fecad0a5672d236a7dca09a9e8df5d0c09ec2b6141045c3a74b",
    "hammer_manifest": "a2d4f9a4f74a23077cc76b83e5ac2dde968abf089b57b293bfa36a636b816956",
    "hammer_outer": "47923e3ba2835be6c0d76b6eaa95e328e7f83e07da9452a6c2657e3f7f97bd2f",
    "release": "336706960c054e1181bc826c28916d89220c588df493ae9d44a6b916435075df",
    "release_sidecar": "805a736e3319ea1e03051aa6181a5310ebf2778c4f76202ef77f4d15f0d46eb4",
    "release_outer": "97bbc06cf203f2f73196fd4f1eb1d03e40258688548f129321dc0f68272b816b",
    "m1498_review": "806cd6f629d17076e7f8bc1df0a633fb6d0a9cd68cf762d8f167123d3c7913b8",
    "m1498_manifest": "df0b581860be722c7c2e49bde4878dee317f72a5097d2b6e6c4e5c1861ddd300",
    "m1498_outer": "0e1d91e0dd700390abf78df87ab5a53fc3187eea1e4d53a8310ae77961eac2d4",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


C = load("m1509_bound_m1506_checker", CHECKER)
R = load("m1509_bound_m1506_runner", RUNNER)
T = load("m1509_bound_m1506_tests", TESTS)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError("nonregular JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token:
                      (_ for _ in ()).throw(RuntimeError(token)))


def rel(path: Path) -> str:
    return path.relative_to(HW).as_posix()


def expected_release() -> dict:
    return {
        "schema": "m1508_m1507_m1506_c1_active_next_oracle_vcs_launch_release_r1_v1",
        "status": R.RELEASE_STATUS,
        "date": "2026-08-31",
        "objective": "Inert release for one future M1506 UNIT_DELAY VCS compile and one simulation after the M1507 different-author source hammer closed every M1498 release blocker. A fresh M1509 final hammer remains mandatory before launch.",
        "launch_now": False,
        "identity": {
            "runner_path": rel(RUNNER), "runner_sha256": PINS["runner"],
            "source_contract_path": rel(CONTRACT),
            "source_contract_sha256": PINS["contract"],
            "source_author_review_sha256": PINS["author_review"],
            "source_hammer_path": rel(HAMMER),
            "source_hammer_review_sha256": PINS["hammer_review"],
            "source_hammer_manifest_sha256": PINS["hammer_manifest"],
            "source_hammer_outer_file_sha256": PINS["hammer_outer"],
            "docs359_sha256": PINS["docs359"],
        },
        "authorization": copy.deepcopy(R.AUTHORIZATION),
        "one_shot": {
            "attempt": rel(R.ATTEMPT), "result": rel(R.RESULT),
            "failure_quarantine": rel(R.QUARANTINE),
            "all_absent_at_release_authoring": True,
            "attempt_before_first_vcs": True, "atomic_no_replace": True,
        },
        "final_hammer_gate": {
            "path": rel(R.FINAL), "present_at_release_authoring": False,
            "fresh_different_author_required": True,
            "required_status": R.FINAL_STATUS,
        },
        "claim_boundary": copy.deepcopy(R.CLAIMS),
    }


def validate_release(value) -> None:
    if type(value) is not dict or value != expected_release():
        raise RuntimeError("M1508 exact schema/set/value drift")


def walk_dicts(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from walk_dicts(item, path + (key,))


def walk_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_leaves(item, path + (key,))
    else:
        yield path, value


def parent_at(value, path):
    for key in path:
        value = value[key]
    return value


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "__M1509_MUTATION"
    raise TypeError(type(value).__name__)


def duplicate_dump(value, target_path, path=()) -> str:
    if not isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    parts = []
    for key in sorted(value):
        item = json.dumps(key) + ":" + duplicate_dump(
            value[key], target_path, path + (key,))
        parts.append(item)
        if path + (key,) == target_path:
            parts.append(item)
    return "{" + ",".join(parts) + "}"


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []
    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"name": name, "pass": bool(condition), "category": category})
    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"name": name, "rejected": caught,
                        "false_negative": not caught, "category": category})

    paths = (("runner", RUNNER), ("checker", CHECKER), ("tests", TESTS),
             ("contract", CONTRACT), ("release", RELEASE),
             ("release_sidecar", Path(str(RELEASE) + ".sha256")),
             ("release_outer", Path(str(RELEASE) + ".sha256.seal.sha256")),
             ("docs359", DOCS359))
    for label, path in paths:
        check("exact_" + label, sha(path) == PINS[label], "identity")
    actual = strict_json(RELEASE)
    validate_release(actual)
    check("release_exact_schema_set_value", True, "release")
    check("authorization_exact", actual["authorization"] == {
        "vcs_compiles": 1, "simv_runs": 1,
        "all_other_eda_runs": 0, "automatic_retry": False}, "authorization")
    check("claim_boundary_exact_runner", actual["claim_boundary"] == R.CLAIMS,
          "claim")
    check("launch_now_false", actual["launch_now"] is False, "authorization")

    source_author = R.P.P.verify_authority(
        AUTHOR, PINS["author_review"], PINS["author_manifest"], PINS["author_outer"])
    source_hammer = R.P.P.verify_authority(
        HAMMER, PINS["hammer_review"], PINS["hammer_manifest"], PINS["hammer_outer"])
    m1498 = R.P.P.verify_authority(
        M1498, PINS["m1498_review"], PINS["m1498_manifest"], PINS["m1498_outer"])
    check("m1506_author_status", source_author.get("status") == R.AUTHOR_STATUS,
          "authority")
    check("m1507_hammer_status", source_hammer.get("status") == R.HAMMER_STATUS,
          "authority")
    check("m1498_failure_status", m1498.get("status") == R.M1498_STATUS,
          "authority")
    check("m1498_release_still_forbidden",
          m1498.get("authorization", {}).get("m1499_release_authoring") is False,
          "authority")
    check("release_binds_m1507_review", actual["identity"][
        "source_hammer_review_sha256"] == sha(HAMMER / "review.json"), "authority")
    check("release_binds_m1507_manifest", actual["identity"][
        "source_hammer_manifest_sha256"] == sha(HAMMER / "SHA256SUMS"), "authority")
    check("release_binds_m1507_outer", actual["identity"][
        "source_hammer_outer_file_sha256"] == sha(
            HAMMER / "SHA256SUMS.seal.sha256"), "authority")

    contract = C.expected_contract()
    R.validate_frozen_inputs(contract)
    check("exact_frozen_inputs", True, "frozen_input")
    check("m1497_tb_exact", sha(R.P.TB) == R.M1497_PINS["tb"], "frozen_input")
    check("m1497_runner_exact", sha(R.M1497_RUNNER) == R.M1497_PINS["runner"],
          "frozen_input")
    source = C.check_source(False)
    check("source_checker", source.get("status") == C.AUTHOR_STATUS, "source")
    stream = io.StringIO()
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(T))
    check("source_tests_16", replay.testsRun == 16 and not replay.failures and
          not replay.errors, "source")

    precheck = json.loads(PRECHECK.read_text())
    check("m1509_fresh_before_creation",
          precheck.get("m1509_final_hammer_namespace_absent") is True, "freshness")
    for label, path in (("attempt", R.ATTEMPT), ("result", R.RESULT),
                        ("failure_quarantine", R.QUARANTINE)):
        check("fresh_" + label, not os.path.lexists(path), "freshness")
    R.namespace_gate()
    check("runner_namespace_gate", True, "freshness")

    canonical = expected_release()
    leaf_count = deletion_count = extra_count = duplicate_count = 0
    for path, value in walk_leaves(canonical):
        candidate = copy.deepcopy(canonical)
        parent_at(candidate, path[:-1])[path[-1]] = changed(value)
        category = ("authorization" if path and path[0] == "authorization" else
                    "claim" if path and path[0] == "claim_boundary" else
                    "path" if path and (path[-1].endswith("path") or
                                         path[-1] in {"attempt", "result", "failure_quarantine"}) else
                    "sha" if path and path[-1].endswith("sha256") else
                    "release_leaf")
        attack("release_leaf." + ".".join(path),
               lambda value=candidate: validate_release(value), category)
        leaf_count += 1
    for path, mapping in list(walk_dicts(canonical)):
        for key in tuple(mapping):
            candidate = copy.deepcopy(canonical)
            del parent_at(candidate, path)[key]
            attack("release_delete." + ".".join(path + (key,)),
                   lambda value=candidate: validate_release(value), "release_delete")
            deletion_count += 1
            with tempfile.TemporaryDirectory(prefix="m1509_dup_") as name:
                candidate_path = Path(name) / "duplicate.json"
                candidate_path.write_text(duplicate_dump(canonical, path + (key,)))
                attack("release_duplicate." + ".".join(path + (key,)),
                       lambda value=candidate_path: strict_json(value),
                       "release_duplicate")
            duplicate_count += 1
        candidate = copy.deepcopy(canonical)
        parent_at(candidate, path)["__M1509_EXTRA__"] = True
        attack("release_extra." + (".".join(path) or "root"),
               lambda value=candidate: validate_release(value), "release_extra")
        extra_count += 1

    p0 = sum(not row["rejected"] for row in attacks)
    p1 = sum(not row["pass"] for row in checks)
    output = {
        "schema": "m1509_m1508_m1506_c1_active_next_oracle_final_launch_hammer_output_r1_v1",
        "status": (R.FINAL_STATUS if p0 == 0 and p1 == 0
                   else "FAIL_DO_NOT_LAUNCH_M1506"),
        "passed_check_names": [row["name"] for row in checks if row["pass"]],
        "failed_check_names": [row["name"] for row in checks if not row["pass"]],
        "attack_category_counts": {
            category: sum(row["category"] == category for row in attacks)
            for category in sorted({row["category"] for row in attacks})},
        "false_negative_names": [row["name"] for row in attacks
                                 if not row["rejected"]],
        "summary": {
            "checks_passed": sum(row["pass"] for row in checks),
            "checks_total": len(checks),
            "mutations_rejected": sum(row["rejected"] for row in attacks),
            "mutations_total": len(attacks),
            "false_negatives": p0, "failed_checks": p1,
            "release_leaf_mutations": leaf_count,
            "release_key_deletions": deletion_count,
            "release_object_extras": extra_count,
            "release_duplicate_keys": duplicate_count,
            "source_tests_run": replay.testsRun,
            "source_test_failures": len(replay.failures) + len(replay.errors),
        },
        "authorization": copy.deepcopy(R.AUTHORIZATION),
        "claim_boundary": copy.deepcopy(R.CLAIMS),
        "execution": {"license_query": 0, "vcs": 0, "simv": 0,
                      "synthesis": 0, "sta": 0, "power": 0,
                      "ssh": 0, "gpu": 0, "attempts_consumed": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
