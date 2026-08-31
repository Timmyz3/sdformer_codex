#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author, no-EDA blind hammer for M1506.

The hammer imports the frozen source only.  It never launches VCS, simv,
synthesis, STA, power, SSH, GPU work, or a license query.  Runtime-path tests
replace every external action with local mocks under TemporaryDirectory.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1506_m1497_c1_active_next_oracle_release_safe_successor_one_shot.py"
CHECKER = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/check_m1506_source.py"
TESTS = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/test_m1506_source.py"
CONTRACT = HW / "contracts/m1506_c1_active_next_oracle_release_safe_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1506_c1_active_next_oracle_release_safe_successor_source_author_r1_20260831"
M1498 = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
M1508 = HW / "contracts/m1508_m1507_m1506_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
M1509 = HW / "reviews/m1509_m1508_m1506_c1_active_next_oracle_final_launch_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PRECHECK = HERE / "freshness_precheck.json"

PINS = {
    "runner": "9613922eb3aec2c7fe0efa69cafb4fb8337009b26686435f44cc139c774317cc",
    "checker": "1cb79e04fbbbcb76d914567d20eb5ad1d595a128a12db5d7da106a241fb0320f",
    "tests": "f4c1dd7211d84eef5b469e23d1ee58db0076f05b75e4f0424a55f6550392c58c",
    "contract": "fb5d5d4d8d5e7fcd427265f2770a544eb1de1ab01385262f63469a61ab524346",
    "contract_sidecar": "5674b8d7fe7748d30e2fa9445a131d43e0b845213d3a456b06b4fd7d60d8ab6d",
    "contract_outer": "2d779483655e371b8723bf1c33a0c672281f7c25ca0ba2969d0d75b6eaa8ee52",
    "author_review": "ea526e6b1988d4d96fd301f9de38c1d6faf0563057b1004d8b15ce7ac339bf92",
    "author_manifest": "19927c78074fc489c26d64b9707cf7d9a0a8499858bfb6d3f0771f10a36c5bfd",
    "author_outer": "1bc99841bfc2a01b81e11a30f72a02935d89a41bce862e8bdfb8b2c5a32a96ad",
    "m1498_review": "806cd6f629d17076e7f8bc1df0a633fb6d0a9cd68cf762d8f167123d3c7913b8",
    "m1498_manifest": "df0b581860be722c7c2e49bde4878dee317f72a5097d2b6e6c4e5c1861ddd300",
    "m1498_outer": "0e1d91e0dd700390abf78df87ab5a53fc3187eea1e4d53a8310ae77961eac2d4",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


C = load("m1507_bound_m1506_checker", CHECKER)
R = load("m1507_bound_m1506_runner", RUNNER)
T = load("m1507_bound_m1506_tests", TESTS)
M1497C = load("m1507_bound_m1497_checker", C.M1497_CHECKER)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_log() -> str:
    lines = [
        R.R13_ENTER,
        R.R13_COMPLETE,
        R.WITNESS_OPERANDS,
        "COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 "
        "join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 "
        "task_completions=1 response_cycle_gap=2 oracle_records=80 "
        "parent_issue_override=0 child_issue_override=0",
        '"sva.sv", 133: tb.u_protocol_sva.cp_nonfirst, 80 attempts, 1 match',
        '"sva.sv", 142: tb.u_protocol_sva.cp_ii2, 80 attempts, 1 match',
        R.BASE.R13_PASS,
        R.BASE.R15_PASS,
    ]
    lines.extend("ORACLE_M1270R13 site=x pass=1 index=%d" % index
                 for index in range(80))
    return "\n".join(lines) + "\n"


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
        return value + "__M1507_MUTATION"
    if type(value) is list:
        return list(value) + ["__M1507_MUTATION"]
    raise TypeError(type(value).__name__)


def duplicate_dump(value, target_path, path=()) -> str:
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    parts = []
    for key in sorted(value):
        child = duplicate_dump(value[key], target_path, path + (key,))
        item = json.dumps(key) + ":" + child
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


def run_quarantine_attack() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="m1507_quarantine_") as name:
        root = Path(name)
        raw = root / "raw"
        raw.mkdir()
        paths = {
            "ATTEMPT": root / "attempt", "RESULT": root / "result",
            "QUARANTINE": root / "quarantine", "RAW_BUILD": raw,
            "CLEAN_RESULT_STAGE": root / "clean",
            "ATTEMPT_STAGE": root / "attempt_stage",
            "FAILURE_STAGE": root / "failure_stage",
        }
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")
        caught = False
        with ExitStack() as stack:
            for attr, path in paths.items():
                stack.enter_context(mock.patch.object(R, attr, path))
            stack.enter_context(mock.patch.object(R, "validate_authority", return_value=None))
            stack.enter_context(mock.patch.object(R.subprocess, "run", return_value=completed))
            stack.enter_context(mock.patch.object(R, "namespace_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.BASE, "collision_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.BASE, "resource_gate", return_value=None))
            stack.enter_context(mock.patch.object(
                R, "publish_no_replace", side_effect=lambda source, destination:
                os.rename(source, destination)))
            try:
                R.main()
            except FileExistsError:
                caught = True
        R.P.P.verify_recursive_seal_generic(paths["QUARANTINE"])
        receipt = json.loads((paths["QUARANTINE"] /
            "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
        return {
            "collision_caught": caught,
            "attempt_consumed": paths["ATTEMPT"].is_dir(),
            "quarantine_double_sealed": paths["QUARANTINE"].is_dir(),
            "failure_status": receipt["status"],
            "functional_vcs": receipt["claim_boundary"]["functional_vcs"],
        }


def nonregular_raw_attack(kind: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="m1507_nonregular_") as name:
        root = Path(name)
        raw = root / "raw"; raw.mkdir()
        target = raw / "compile.log"
        if kind == "symlink":
            outside = root / "outside"; outside.write_text("must not be followed\n")
            target.symlink_to(outside)
        elif kind == "directory":
            target.mkdir()
        else:
            raise ValueError(kind)
        (raw / "sim.log").write_text("regular failure log\n")
        stage = root / "failure"
        with mock.patch.object(R, "RAW_BUILD", raw):
            R.make_clean_evidence(stage, "COMPILE", "RuntimeError: injected", 1, 0, None)
        R.P.P.verify_recursive_seal_generic(stage)
        payload = (stage / "compile.log").read_text()
        receipt = json.loads((stage /
            "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
        return "nonregular" in payload and receipt["status"] == "FAILED_OR_INCOMPLETE"


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"name": name, "category": category, "pass": bool(condition)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"name": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # Exact identities, double seals, predecessor failure, and byte identity.
    for name, path in (("runner", RUNNER), ("checker", CHECKER),
                       ("tests", TESTS), ("contract", CONTRACT),
                       ("contract_sidecar", Path(str(CONTRACT) + ".sha256")),
                       ("contract_outer", Path(str(CONTRACT) + ".sha256.seal.sha256")),
                       ("docs359", DOCS359)):
        check("exact_" + name, sha(path) == PINS[name], "identity")
    author = R.P.P.verify_authority(AUTHOR, PINS["author_review"],
                                    PINS["author_manifest"], PINS["author_outer"])
    check("author_status", author.get("status") == R.AUTHOR_STATUS, "authority")
    failure = R.P.P.verify_authority(M1498, PINS["m1498_review"],
                                     PINS["m1498_manifest"], PINS["m1498_outer"])
    check("m1498_failure_status", failure.get("status") == R.M1498_STATUS,
          "predecessor_failure")
    check("m1498_release_forbidden",
          failure.get("authorization", {}).get("m1499_release_authoring") is False,
          "predecessor_failure")
    old = C.R13.read_text(); new = C.M1497_TB.read_text()
    check("m1497_tb_byte_identity", old.count(M1497C.OLD) == 1 and
          new == old.replace(M1497C.OLD, M1497C.NEW), "oracle")
    check("m1497_tb_sha", sha(C.M1497_TB) == C.M1497_PINS["testbench_sha256"],
          "oracle")
    check("m1497_filelist_sha", sha(C.M1497_FILELIST) ==
          C.M1497_PINS["filelist_sha256"], "oracle")
    precheck = json.loads(PRECHECK.read_text())
    check("m1507_fresh_before_creation",
          precheck.get("m1507_source_hammer_namespace_absent") is True, "freshness")
    check("m1508_fresh", not os.path.lexists(M1508), "freshness")
    check("m1509_fresh", not os.path.lexists(M1509), "freshness")

    # Native author controls, independently rerun in-process.
    source = C.check_source(False)
    check("native_source_checker", source.get("status") == C.AUTHOR_STATUS, "baseline")
    stream = io.StringIO()
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(T))
    check("author_tests_16", replay.testsRun == 16 and
          not replay.failures and not replay.errors, "baseline")

    # Exhaustive contract set/value campaign: every leaf, key deletion,
    # object extra, and every key duplicated in its own containing object.
    canonical = C.expected_contract()
    leaf_count = deletion_count = extra_count = duplicate_count = 0
    for path, value in walk_leaves(canonical):
        candidate = copy.deepcopy(canonical)
        parent_at(candidate, path[:-1])[path[-1]] = changed(value)
        attack("contract_leaf." + ".".join(path),
               lambda value=candidate: C.validate_contract(value, canonical),
               "contract_leaf")
        leaf_count += 1
    for path, mapping in list(walk_dicts(canonical)):
        for key in tuple(mapping):
            candidate = copy.deepcopy(canonical)
            del parent_at(candidate, path)[key]
            attack("contract_delete." + ".".join(path + (key,)),
                   lambda value=candidate: C.validate_contract(value, canonical),
                   "contract_delete")
            deletion_count += 1
            with tempfile.TemporaryDirectory(prefix="m1507_dup_") as name:
                candidate_path = Path(name) / "duplicate.json"
                candidate_path.write_text(duplicate_dump(canonical, path + (key,)))
                attack("contract_duplicate." + ".".join(path + (key,)),
                       lambda path=candidate_path: C.strict_json(path),
                       "contract_duplicate")
            duplicate_count += 1
        candidate = copy.deepcopy(canonical)
        parent_at(candidate, path)["__M1507_EXTRA__"] = True
        attack("contract_extra." + (".".join(path) or "root"),
               lambda value=candidate: C.validate_contract(value, canonical),
               "contract_extra")
        extra_count += 1

    # Runtime exact-read corpus.  Each target is independently turned into an
    # injected identity failure; validate_frozen_inputs must reject every one.
    corpus = {
        "m1497_runner": R.M1497_RUNNER, "m1497_checker": R.P.CHECKER,
        "m1497_tests": R.P.TESTS, "m1497_tb": R.P.TB,
        "m1497_filelist": R.P.FILELIST, "m1497_r13": R.P.TB_R13,
        "m1497_contract": R.P.CONTRACT,
        "m1497_contract_sidecar": Path(str(R.P.CONTRACT) + ".sha256"),
        "m1497_contract_outer": Path(str(R.P.CONTRACT) + ".sha256.seal.sha256"),
        "m1506_checker": R.CHECKER, "m1506_tests": R.TESTS,
    }
    corpus.update({"base_" + path.name: path for path in R.BASE.EXACT})
    seen: set[Path] = set()
    real_exact = R.exact
    with mock.patch.object(R, "exact", side_effect=lambda path, digest:
                           (seen.add(Path(path)), real_exact(Path(path), digest))[1]):
        R.validate_frozen_inputs(canonical)
    for name, path in corpus.items():
        check("runtime_exact_read_" + name, Path(path) in seen, "runtime_identity")
        def injected(target=Path(path)):
            def exact(path, digest):
                if Path(path) == target:
                    raise RuntimeError("injected identity drift")
                return real_exact(Path(path), digest)
            with mock.patch.object(R, "exact", side_effect=exact):
                R.validate_frozen_inputs(canonical)
        attack("runtime_identity_mutation." + name, injected, "runtime_identity")

    # Log admission: every exact token, witness field, coverage field,
    # CP cover, oracle population/pass, forbidden diagnostic and fault class.
    good = positive_log()
    check("positive_log", R.validate_sim_log(good)["oracle_records"] == 80,
          "log_admission")
    for token_name, token in (("r13_pass", R.BASE.R13_PASS),
                              ("r15_pass", R.BASE.R15_PASS),
                              ("phase_enter", R.R13_ENTER),
                              ("phase_complete", R.R13_COMPLETE),
                              ("witness", R.WITNESS_OPERANDS)):
        attack("log_missing_" + token_name,
               lambda token=token: R.validate_sim_log(good.replace(token + "\n", "", 1)),
               "pass_cardinality")
        attack("log_duplicate_" + token_name,
               lambda token=token: R.validate_sim_log(good + token + "\n"),
               "pass_cardinality")
    witness_pairs = (("weight_req=2", "weight_req=1"),
                     ("psum_req=1", "psum_req=0"),
                     ("responses=2", "responses=1"),
                     ("core_accepts=2", "core_accepts=1"),
                     ("psum_commits=1", "psum_commits=0"),
                     ("rows=1", "rows=0"), ("tasks=1", "tasks=0"),
                     ("design_issue=2", "design_issue=1"),
                     ("design_commit=1", "design_commit=0"),
                     ("design_rows=1", "design_rows=0"),
                     ("masks=0", "masks=1"), ("faults=0", "faults=1"))
    for old_token, new_token in witness_pairs:
        attack("witness_" + old_token.split("=")[0],
               lambda old=old_token, new=new_token:
               R.validate_sim_log(good.replace(old, new, 1)), "witness")
    coverage_pairs = (("first_beats=1", "first_beats=0"),
                      ("nonfirst_beats=1", "nonfirst_beats=0"),
                      ("join_hold_cycles=2", "join_hold_cycles=1"),
                      ("issue_accepts=2", "issue_accepts=1"),
                      ("psum_reads=1", "psum_reads=0"),
                      ("row_completions=1", "row_completions=0"),
                      ("task_completions=1", "task_completions=0"),
                      ("response_cycle_gap=2", "response_cycle_gap=1"),
                      ("oracle_records=80", "oracle_records=79"),
                      ("parent_issue_override=0", "parent_issue_override=1"),
                      ("child_issue_override=0", "child_issue_override=1"))
    for old_token, new_token in coverage_pairs:
        attack("coverage_" + old_token.split("=")[0],
               lambda old=old_token, new=new_token:
               R.validate_sim_log(good.replace(old, new, 1)), "coverage")
    for label, needle in (("cp_nonfirst", "cp_nonfirst, 80 attempts, 1 match"),
                          ("cp_ii2", "cp_ii2, 80 attempts, 1 match")):
        attack("coverage_missing_" + label,
               lambda needle=needle: R.validate_sim_log(
                   "\n".join(line for line in good.splitlines() if needle not in line) + "\n"),
               "coverage")
        line = next(line for line in good.splitlines() if needle in line)
        attack("coverage_duplicate_" + label,
               lambda line=line: R.validate_sim_log(good + line + "\n"), "coverage")
    attack("oracle_79_records", lambda: R.validate_sim_log(
        good.replace("ORACLE_M1270R13 site=x pass=1 index=79\n", "", 1)), "oracle")
    attack("oracle_failed_record", lambda: R.validate_sim_log(
        good.replace(" pass=1 index=0", " pass=0 index=0", 1)), "oracle")
    for index, line in enumerate(("Error: injected", "Fatal: injected",
                                  "$error injected", "$fatal injected",
                                  "Assertion failure injected",
                                  "assertion produced an error")):
        attack("forbidden_diagnostic_%d" % index,
               lambda line=line: R.validate_sim_log(good + line + "\n"),
               "forbidden_diagnostic")
    for index, line in enumerate(("boundary_fault=1", "boundary_fault=x",
                                  "core_fault=X", "m935_fault=z", "faults=2")):
        attack("fault_unknown_or_nonzero_%d" % index,
               lambda line=line: R.validate_sim_log(good + line + "\n"), "fault")

    quarantine = run_quarantine_attack()
    check("post_attempt_collision_caught", quarantine["collision_caught"], "quarantine")
    check("post_attempt_consumed", quarantine["attempt_consumed"], "quarantine")
    check("post_attempt_double_sealed_quarantine",
          quarantine["quarantine_double_sealed"], "quarantine")
    check("post_attempt_failure_not_functional",
          quarantine["failure_status"] == "FAILED_OR_INCOMPLETE" and
          quarantine["functional_vcs"] is False, "quarantine")
    check("nonregular_raw_symlink_not_followed", nonregular_raw_attack("symlink"),
          "result_hygiene")
    check("nonregular_raw_directory_not_followed", nonregular_raw_attack("directory"),
          "result_hygiene")
    with tempfile.TemporaryDirectory(prefix="m1507_clean_") as name:
        root = Path(name) / "clean"; root.mkdir()
        for member in R.CLEAN_PAYLOAD:
            (root / member).write_text("regular\n")
        (root / "compile.log").unlink()
        (root / "compile.log").symlink_to(root / "sim.log")
        attack("clean_payload_symlink", lambda: R.seal_clean_result(root),
               "result_hygiene")

    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1507_m1506_c1_active_next_oracle_release_safe_source_blind_hammer_output_r1_v1",
        "status": ("PASS_ZERO_FALSE_NEGATIVES__M1508_RELEASE_AUTHORING_ONLY"
                   if p0 == 0 and p1 == 0 else "FAIL_DO_NOT_CITE__NO_M1508"),
        "passed_check_names": [item["name"] for item in checks if item["pass"]],
        "failed_check_names": [item["name"] for item in checks if not item["pass"]],
        "attack_category_counts": {
            category: sum(item["category"] == category for item in attacks)
            for category in sorted({item["category"] for item in attacks})
        },
        "false_negative_names": [item["name"] for item in attacks
                                 if not item["rejected"]],
        "summary": {
            "checks_passed": sum(item["pass"] for item in checks),
            "checks_total": len(checks),
            "mutations_rejected": sum(item["rejected"] for item in attacks),
            "mutations_total": len(attacks),
            "false_negatives": p0,
            "failed_checks": p1,
            "author_tests_run": replay.testsRun,
            "author_test_failures": len(replay.failures) + len(replay.errors),
            "contract_leaf_mutations": leaf_count,
            "contract_key_deletions": deletion_count,
            "contract_object_extras": extra_count,
            "contract_duplicate_keys": duplicate_count,
            "runtime_exact_read_targets": len(corpus),
        },
        "authorization": {
            "m1508_release_authoring": p0 == 0 and p1 == 0,
            "m1509_final_launch_hammer_authoring": False,
            "vcs_launch": False,
            "automatic_retry": False,
        },
        "claim_boundary": copy.deepcopy(R.CLAIMS),
        "execution": {"license_query": 0, "vcs": 0, "simv": 0,
                      "synthesis": 0, "sta": 0, "power": 0,
                      "ssh": 0, "gpu": 0, "attempts_consumed": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
