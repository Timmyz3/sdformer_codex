#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent source-only blind hammer for M1459.

This program never invokes VCS, simv, a license query, or any EDA tool.  It
loads the inert M1459 runner as a Python module, replays its source-only suite,
and attacks the generic-seal/authority-seal split in temporary directories.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1459_m1433_c1_runtime_split_generic_seal_successor.py"
CHECKER = HW / "verif_m1459_c1_generic_seal_successor/check_m1459_c1_generic_seal_successor_source.py"
TESTS = HW / "verif_m1459_c1_generic_seal_successor/test_m1459_c1_generic_seal_successor_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
CONTRACT = HW / "contracts/m1459_m1433_c1_generic_seal_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1459_m1433_c1_generic_seal_successor_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE\u7ec8\u5c40\u51bb\u7ed3_20260813.md"
OLD_RUNNER = HW / "dc_handoff/scripts/run_vcs_m1433_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_runtime_split_exact.py"
M1433_FAILURE = HW / "results/.m1433_c1_r16_runtime_split_vcs_failure_stage.381989"
M1459_ATTEMPT = HW / "results/.m1459_c1_generic_seal_vcs_attempt_consumed"
M1459_RESULT = HW / "results/m1459_c1_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
M1459_QUARANTINE = Path(str(M1459_RESULT) + ".failed_or_incomplete.quarantine")
FINAL = HW / "reviews/m1464_m1459_c1_generic_seal_successor_source_blind_hammer_r1_20260831"

PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
OLD_RUNNER_SHA = "443ef3f2a2bc777095a5574da6b91aa2c97786505f86bff607fbc537adbae07a"
RUNTIME_TESTS_SHA = "b3b9d130749eb4a8a79148072350b76aeeb59520f85718e0663df62f40731ad4"
EXPECTED_RUNNER_SHA = "3c0028c41fbbd8f6d1ede4b284aece877dd926a2b82a67de26d71f5322a9e891"
EXPECTED_CHECKER_SHA = "efdf56d8b22ef6205c9f7059648bbb62c6c0cbc81606571b3473864cb613bbd9"
EXPECTED_TESTS_SHA = "d47c951e3e8dc75be733438e2504fa6b920d893704df890cd6c2761f553bdbb4"
EXPECTED_CONTRACT_SHA = "cd4e2d6075a644f365f1c6c7b097afbae0e287101e563d92c9b52241c60fb910"
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False,
          "energy": False, "system_speedup": False, "headline": False}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(RuntimeError(token)))


def load_runner():
    spec = importlib.util.spec_from_file_location("m1464_target", RUNNER)
    need(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(RUNNER)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def expect_reject(name, action, attacks):
    rejected = False
    try:
        action()
    except BaseException:
        rejected = True
    attacks.append({"attack": name, "rejected": rejected})
    need(rejected, "false negative: " + name)


def reseal(module, root: Path) -> None:
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        path = root / name
        if path.exists() or path.is_symlink():
            path.unlink()
    module.seal_dir_generic(root)


def main() -> int:
    need(not FINAL.exists(), "final M1464 namespace already exists")
    need(sha(DOCS359) == DOCS359_SHA, "docs359 drift")
    need(sha(RUNNER) == EXPECTED_RUNNER_SHA, "runner drift")
    need(sha(CHECKER) == EXPECTED_CHECKER_SHA, "checker drift")
    need(sha(TESTS) == EXPECTED_TESTS_SHA, "tests drift")
    need(sha(CONTRACT) == EXPECTED_CONTRACT_SHA, "contract drift")
    need(sha(OLD_RUNNER) == OLD_RUNNER_SHA, "M1433 runner drift")
    need(sha(RUNTIME_TESTS) == RUNTIME_TESTS_SHA, "runtime tests drift")
    need(all(not os.path.lexists(path) for path in
             (M1459_ATTEMPT, M1459_RESULT, M1459_QUARANTINE)),
         "M1459 namespace not fresh")

    runner = load_runner()
    text = RUNNER.read_text()
    generic = text[text.index("def verify_recursive_seal_generic("):
                   text.index("def verify_authority(")]
    authority = text[text.index("def verify_authority("):
                     text.index("def seal_dir_generic(")]
    need("review.json" not in generic, "generic verifier still assumes review")
    need("verify_recursive_seal_generic(root" in authority and
         'review = root / "review.json"' in authority and
         "strict_json(review)" in authority,
         "authority verifier is not strict generic-plus-review")
    need(text.count('run_python_gate(BASE.SOURCE_CHECKER, "runtime_present")') == 1,
         "source checker runtime gate cardinality")
    need(text.count('run_python_gate(BASE.RUNTIME_TESTS, "runtime_present")') == 1,
         "runtime suite cardinality")
    need("BASE.SOURCE_TESTS" not in text, "source-only suite reachable at launch")
    need(text.count("BASE.collision_gate()") == 2, "double collision gate drift")
    need(text.count("compile_count = 1") == 1 and text.count("sim_count = 1") == 1,
         "one compile/one sim bound drift")
    need(text.index("publish_no_replace(ATTEMPT_STAGE, ATTEMPT)") <
         text.index("run_tool(COMPILE_COMMAND"), "attempt not before tool")
    need("seal_dir_generic(ATTEMPT_STAGE)" in text and
         "seal_dir_generic(FAILURE_STAGE)" in text and
         "seal_dir_generic(WORK)" in text, "generic stage seal missing")
    need("publish_no_replace(FAILURE_STAGE, QUARANTINE)" in text and
         "publish_no_replace(WORK, RESULT)" in text,
         "atomic quarantine/result publication missing")
    need(all(token in text for token in
             ("M1433_CHAIN_PINS", "M1464", "M1465", "M1466")),
         "authority chain namespace/pin drift")

    contract = strict_json(CONTRACT)
    author_review = strict_json(AUTHOR / "review.json")
    runner.verify_authority(AUTHOR)
    runner.verify_recursive_seal_generic(M1433_FAILURE)
    need(contract["claim_boundary"] == CLAIMS and
         author_review["claim_boundary"] == CLAIMS,
         "claim boundary drift")
    need(contract["repair_invariant"] == {
        "generic_recursive_verifier_requires_review_json": False,
        "authority_verifier_requires_review_json": True,
        "attempt_stage_uses_generic_verifier": True,
        "failure_stage_uses_generic_verifier": True,
        "success_stage_uses_generic_verifier": True,
        "authority_chain_uses_authority_verifier": True,
        "recursive_membership_and_outer_seal_preserved": True,
        "atomic_noreplace_publish_preserved": True,
        "m1433_exact_workload_preserved": True,
        "m1433_runtime_present_suite_preserved": True,
        "m1433_source_only_suite_unreachable_at_launch": True,
    }, "repair invariant not exact")
    need(contract["future_execution"]["automatic_retry"] is False and
         contract["future_execution"]["maximum_vcs_compiles"] == 1 and
         contract["future_execution"]["maximum_simv_runs"] == 1 and
         contract["future_execution"]["all_other_eda_runs"] == 0,
         "tool authorization drift")

    source_check = subprocess.run(
        [str(PYTHON), "-I", str(CHECKER), "--mode", "source_absent"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=120)
    need(source_check.returncode == 0, "source checker failed: " + source_check.stderr)
    source_payload = json.loads(source_check.stdout)
    need(source_payload["status"] ==
         "PASS_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE__NO_VCS_NO_EDA",
         "source checker status")
    test_run = subprocess.run(
        [str(PYTHON), "-I", str(TESTS)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False, timeout=120)
    need(test_run.returncode == 0, "source suite failed: " + test_run.stdout)
    (HERE / "source_test_output.txt").write_text(test_run.stdout)
    match = re.search(r"Ran (\d+) tests", test_run.stdout)
    need(match is not None and int(match.group(1)) == 18, "source test count")

    attacks = []
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)

        # The repaired positive case: a generic attempt is sealed without a review.
        attempt = temp / "attempt"
        attempt.mkdir()
        (attempt / "attempt.json").write_text('{"status":"CONSUMED"}\n')
        runner.seal_dir_generic(attempt)
        runner.verify_recursive_seal_generic(attempt)
        need(not (attempt / "review.json").exists(), "generic seal injected review")

        expect_reject("authority_missing_review",
                      lambda: runner.verify_authority(attempt), attacks)

        malformed = temp / "authority_duplicate_json"
        malformed.mkdir()
        (malformed / "review.json").write_text('{"status":"PASS","status":"EVIL"}\n')
        runner.seal_dir_generic(malformed)
        expect_reject("authority_duplicate_review_key",
                      lambda: runner.verify_authority(malformed), attacks)

        nonfinite = temp / "authority_nonfinite_json"
        nonfinite.mkdir()
        (nonfinite / "review.json").write_text('{"score":NaN}\n')
        runner.seal_dir_generic(nonfinite)
        expect_reject("authority_nonfinite_review",
                      lambda: runner.verify_authority(nonfinite), attacks)

        valid = temp / "authority_valid"
        valid.mkdir()
        (valid / "review.json").write_text('{"status":"PASS"}\n')
        runner.seal_dir_generic(valid)
        need(runner.verify_authority(valid)["status"] == "PASS",
             "valid authority rejected")

        payload_mut = temp / "payload_mutation"
        shutil.copytree(attempt, payload_mut)
        (payload_mut / "attempt.json").write_text('{"status":"MUTATED"}\n')
        expect_reject("generic_payload_mutation",
                      lambda: runner.verify_recursive_seal_generic(payload_mut), attacks)

        manifest_extra = temp / "manifest_extra"
        shutil.copytree(attempt, manifest_extra)
        (manifest_extra / "unlisted.txt").write_text("attack\n")
        expect_reject("manifest_unlisted_member",
                      lambda: runner.verify_recursive_seal_generic(manifest_extra), attacks)

        outer_mut = temp / "outer_mutation"
        shutil.copytree(attempt, outer_mut)
        (outer_mut / "SHA256SUMS.seal.sha256").write_text("0" * 64 + "  SHA256SUMS\n")
        expect_reject("outer_seal_mutation",
                      lambda: runner.verify_recursive_seal_generic(outer_mut), attacks)

        traversal = temp / "manifest_traversal"
        shutil.copytree(attempt, traversal)
        manifest = traversal / "SHA256SUMS"
        manifest.write_text(manifest.read_text() + "0" * 64 + "  ../escape\n")
        (traversal / "SHA256SUMS.seal.sha256").write_text(
            sha(manifest) + "  SHA256SUMS\n")
        expect_reject("manifest_path_traversal",
                      lambda: runner.verify_recursive_seal_generic(traversal), attacks)

        absolute = temp / "manifest_absolute"
        shutil.copytree(attempt, absolute)
        manifest = absolute / "SHA256SUMS"
        manifest.write_text(manifest.read_text() + "0" * 64 + "  /etc/passwd\n")
        (absolute / "SHA256SUMS.seal.sha256").write_text(
            sha(manifest) + "  SHA256SUMS\n")
        expect_reject("manifest_absolute_path",
                      lambda: runner.verify_recursive_seal_generic(absolute), attacks)

        duplicate = temp / "manifest_duplicate"
        shutil.copytree(attempt, duplicate)
        manifest = duplicate / "SHA256SUMS"
        manifest.write_text(manifest.read_text() + manifest.read_text())
        (duplicate / "SHA256SUMS.seal.sha256").write_text(
            sha(manifest) + "  SHA256SUMS\n")
        expect_reject("manifest_duplicate_member",
                      lambda: runner.verify_recursive_seal_generic(duplicate), attacks)

        symlink = temp / "seal_symlink"
        symlink.mkdir()
        (symlink / "payload").write_text("safe\n")
        (symlink / "alias").symlink_to(symlink / "payload")
        expect_reject("generic_seal_symlink",
                      lambda: runner.seal_dir_generic(symlink), attacks)

        replacement = temp / "member_path_replacement"
        shutil.copytree(attempt, replacement)
        member = replacement / "attempt.json"
        member.unlink()
        member.symlink_to("/etc/passwd")
        expect_reject("manifest_member_path_replacement",
                      lambda: runner.verify_recursive_seal_generic(replacement), attacks)

        # A pre-existing canonical attempt must block a repeat before any tool.
        saved = (runner.ATTEMPT, runner.RESULT, runner.QUARANTINE,
                 runner.WORK, runner.ATTEMPT_STAGE, runner.FAILURE_STAGE)
        try:
            runner.ATTEMPT = temp / "already_consumed"
            runner.RESULT = temp / "fresh_result"
            runner.QUARANTINE = temp / "fresh_quarantine"
            runner.WORK = temp / "fresh_work"
            runner.ATTEMPT_STAGE = temp / "fresh_attempt_stage"
            runner.FAILURE_STAGE = temp / "fresh_failure_stage"
            runner.ATTEMPT.mkdir()
            expect_reject("duplicate_attempt_namespace",
                          runner.namespace_gate, attacks)
        finally:
            (runner.ATTEMPT, runner.RESULT, runner.QUARANTINE,
             runner.WORK, runner.ATTEMPT_STAGE, runner.FAILURE_STAGE) = saved

    need(all(item["rejected"] for item in attacks), "mutation FN")
    mechanical = {
        "schema": "m1464_m1459_c1_generic_seal_successor_mechanical_checks_r1_v1",
        "status": "PASS_SOURCE_ONLY_NO_TOOL",
        "source_tests_run": 18,
        "source_tests_passed": 18,
        "attacks_run": len(attacks),
        "attacks_rejected": sum(item["rejected"] for item in attacks),
        "false_negatives": 0,
        "attacks": attacks,
        "generic_attempt_without_review_accepted": True,
        "authority_review_required_and_strict": True,
        "attempt_before_tool": True,
        "double_collision_gate": True,
        "one_compile_one_sim_bound": True,
        "atomic_quarantine_no_retry": True,
        "m1433_exact_runner_and_runtime_suite_pinned": True,
    }
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True) + "\n")
    (HERE / "hammer_output.json").write_text(json.dumps({
        "status": "PASS_M1464_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE_BLIND_HAMMER",
        "p0_count": 0, "p1_count": 0, "score": 100,
        "false_negatives": 0, "launch_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    (HERE / "NO_LICENSE_NO_VCS_NO_SIMV_NO_EDA.txt").write_text(
        "M1464 performed source-only Python checks. No license query, VCS, simv, or EDA was invoked.\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1464_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE_BLIND_HAMMER\n")

    bindings = {
        "runner_sha256": sha(RUNNER),
        "source_checker_sha256": sha(CHECKER),
        "source_tests_sha256": sha(TESTS),
        "runtime_tests_sha256": sha(RUNTIME_TESTS),
        "source_contract_sha256": sha(CONTRACT),
        "author_review_sha256": sha(AUTHOR / "review.json"),
        "author_manifest_sha256": sha(AUTHOR / "SHA256SUMS"),
        "author_outer_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
    }
    review = {
        "schema": "m1464_m1459_c1_generic_seal_successor_source_blind_hammer_r1_v1",
        "status": "PASS_M1464_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE__RELEASE_NOT_AUTHORED",
        "date": "2026-08-31",
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "bindings": bindings,
        "validation": mechanical,
        "authorization": {
            "launch_release": False, "vcs": False, "simv": False,
            "license_query": False, "dc": False, "pt": False,
            "ptpx": False, "eda": False, "gpu": False, "remote": False,
            "automatic_retry": False,
        },
        "claim_boundary": CLAIMS,
        "docs359_sha256": sha(DOCS359),
        "verdict": (
            "M1459 fixes only the M1433 generic-stage sealing defect: an attempt "
            "without review.json now seals and verifies, while all authorities "
            "still require a recursively sealed, strict-JSON review. Independent "
            "source replay passes 18/18 and rejects all 12 mutation families with "
            "zero false negatives. The exact M1433 runtime suite, one-compile/one-"
            "simulation bound, double collision gate, attempt-before-tool ordering, "
            "atomic quarantine, and no-retry contract remain intact. M1464 does not "
            "authorize launch; M1465 release and M1466 final hammer remain mandatory."
        ),
    }
    (HERE / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (HERE / "review.md").write_text(
        "# M1464 independent blind hammer\n\n"
        "PASS 100/100; P0=0, P1=0. The generic recursive seal accepts a sealed "
        "attempt without `review.json`; authority verification still requires a "
        "sealed, strict-JSON `review.json`. Source replay passed 18/18, and 12/12 "
        "mutations were rejected with zero false negatives. No VCS, simv, license "
        "query, or EDA was run. This review does not authorize launch.\n")

    rows = []
    for path in sorted(HERE.iterdir(), key=lambda item: item.name):
        if path.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        need(path.is_file() and not path.is_symlink(), "review member nonregular")
        rows.append((path.name, sha(path)))
    manifest = HERE / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows))
    (HERE / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")
    runner.verify_authority(HERE)
    print(json.dumps({"status": review["status"], "score": 100,
                      "source_tests": 18, "attacks": len(attacks),
                      "false_negatives": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
