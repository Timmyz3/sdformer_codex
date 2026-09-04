#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only authority checker for the additive M1459 C1 successor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1459_m1433_c1_runtime_split_generic_seal_successor.py"
OLD_RUNNER = HW / "dc_handoff/scripts/run_vcs_m1433_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_runtime_split_exact.py"
TESTS = HERE / "test_m1459_c1_generic_seal_successor_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
CONTRACT = HW / "contracts/m1459_m1433_c1_generic_seal_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1459_m1433_c1_generic_seal_successor_source_author_r1_20260831"
FAILURE = HW / "results/.m1433_c1_r16_runtime_split_vcs_failure_stage.381989"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1433_ATTEMPT = HW / "results/.m1433_c1_r16_runtime_split_vcs_attempt_consumed"
M1433_RESULT = HW / "results/m1433_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
M1433_QUARANTINE = Path(str(M1433_RESULT) + ".failed_or_incomplete.quarantine")
M1459_ATTEMPT = HW / "results/.m1459_c1_generic_seal_vcs_attempt_consumed"
M1459_RESULT = HW / "results/m1459_c1_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
M1459_QUARANTINE = Path(str(M1459_RESULT) + ".failed_or_incomplete.quarantine")
M1464 = HW / "reviews/m1464_m1459_c1_generic_seal_successor_source_blind_hammer_r1_20260831"
M1465 = HW / "contracts/m1465_m1464_m1459_c1_generic_seal_successor_vcs_launch_release_r1_20260831.json"
M1466 = HW / "reviews/m1466_m1465_m1459_c1_generic_seal_successor_final_launch_hammer_r1_20260831"

OLD_RUNNER_SHA = "443ef3f2a2bc777095a5574da6b91aa2c97786505f86bff607fbc537adbae07a"
RUNTIME_TESTS_SHA = "b3b9d130749eb4a8a79148072350b76aeeb59520f85718e0663df62f40731ad4"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
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


def verify_recursive_seal_generic(root: Path) -> set[str]:
    need(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest row invalid")
        digest, name = fields
        name = name.lstrip("*")
        rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "manifest digest invalid")
        need(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "manifest member invalid")
        member = root / rel
        need(member.is_file() and not member.is_symlink()
             and stat.S_ISREG(member.lstat().st_mode), "manifest member nonregular")
        need(sha(member) == digest, "manifest digest drift")
        listed.add(name)
    actual = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        need(not any((base_path / name).is_symlink() for name in dirs + files),
             "sealed symlink")
        for name in files:
            rel = (base_path / name).relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                actual.add(rel)
    need(listed == actual, "sealed membership drift")
    return listed


def check_failure_evidence() -> dict:
    members = verify_recursive_seal_generic(FAILURE)
    need(members == {"RUN_FAILED_OR_INCOMPLETE.json",
                     "private_attempt_stage/SHA256SUMS",
                     "private_attempt_stage/SHA256SUMS.seal.sha256",
                     "private_attempt_stage/attempt.json"},
         "M1433 failure members drift")
    verify_recursive_seal_generic(FAILURE / "private_attempt_stage")
    failure = strict_json(FAILURE / "RUN_FAILED_OR_INCOMPLETE.json")
    need(failure == {
        "automatic_retry": False,
        "compile_count": 0,
        "cycles_measured": False,
        "energy": False,
        "exception": "FileNotFoundError: [Errno 2] No such file or directory: '/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/.m1433_c1_r16_runtime_split_vcs_attempt_stage.381989/review.json'",
        "functional_vcs": False,
        "headline": False,
        "phase": "ATTEMPT_CONSUME",
        "power": False,
        "ppa": False,
        "sim_count": 0,
        "speedup": False,
        "status": "FAILED_OR_INCOMPLETE",
        "system_speedup": False,
        "timing_verified": False,
    }, "M1433 failure payload drift")
    need(sha(FAILURE / "RUN_FAILED_OR_INCOMPLETE.json") ==
         "c792b37b60ccbfb0070a18dad83a2ab4dba819e14dd95d79fa000275eb88b73b",
         "M1433 failure JSON SHA drift")
    need(sha(FAILURE / "SHA256SUMS") ==
         "29c2dd1ef9d1fe631834f88e3b360cfdec37f3b2b678c5a809cd2f64df68dbee",
         "M1433 failure manifest SHA drift")
    need(sha(FAILURE / "SHA256SUMS.seal.sha256") ==
         "3eff7fcae81096be9c345d85bfde1c64580e521b04d4b6f088a2dee20a0bd18e",
         "M1433 failure outer SHA drift")
    need(not any(os.path.lexists(path) for path in
                 (M1433_ATTEMPT, M1433_RESULT, M1433_QUARANTINE)),
         "M1433 canonical namespace unexpectedly exists")
    return failure


def check_runner_static() -> dict:
    text = RUNNER.read_text()
    old = OLD_RUNNER.read_text()
    need(sha(OLD_RUNNER) == OLD_RUNNER_SHA, "M1433 runner drift")
    need(sha(RUNTIME_TESTS) == RUNTIME_TESTS_SHA, "M1433 runtime suite drift")
    need('return strict_json(root / "review.json")' in old
         and "verify_recursive_seal(root)" in old,
         "M1433 root cause no longer exact")
    need("def verify_recursive_seal_generic(" in text
         and "def verify_authority(" in text
         and "def seal_dir_generic(" in text,
         "M1459 verifier split absent")
    need("verify_recursive_seal_generic(root)" in text,
         "M1459 generic seal does not self-verify")
    authority_body = text[text.index("def verify_authority("):
                          text.index("def seal_dir_generic(")]
    generic_body = text[text.index("def verify_recursive_seal_generic("):
                        text.index("def verify_authority(")]
    need('strict_json(review)' in authority_body and "review.json" in authority_body,
         "M1459 authority review absent")
    need("review.json" not in generic_body,
         "M1459 generic verifier still assumes authority payload")
    need(text.count("run_python_gate(BASE.SOURCE_CHECKER, \"runtime_present\")") == 1
         and text.count("run_python_gate(BASE.RUNTIME_TESTS, \"runtime_present\")") == 1,
         "M1433 runtime gates not exact one each")
    need("BASE.SOURCE_TESTS" not in text, "source-only suite reachable at launch")
    compile_at = text.index("run_tool(COMPILE_COMMAND")
    attempt_at = text.index("publish_no_replace(ATTEMPT_STAGE, ATTEMPT)")
    need(attempt_at < compile_at, "attempt is not consumed before tool")
    need(text.count("BASE.collision_gate()") == 2,
         "collision gate cardinality drift")
    need(text.count("compile_count = 1") == 1 and text.count("sim_count = 1") == 1,
         "one-shot tool count drift")
    need("seal_dir_generic(FAILURE_STAGE)" in text
         and "publish_no_replace(FAILURE_STAGE, QUARANTINE)" in text,
         "failure quarantine not generic-sealed/atomic")
    need("seal_dir_generic(WORK)" in text
         and "publish_no_replace(WORK, RESULT)" in text,
         "success result not generic-sealed/atomic")
    need(all(token in text for token in ("M1464", "M1465", "M1466")),
         "C1 successor authority namespace drift")
    return {"attempt_before_tool": True, "collision_gates": 2,
            "generic_authority_split": True, "one_compile": True,
            "one_sim": True, "runtime_suite_preserved": True}


def check_contract() -> dict:
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1459_m1433_c1_generic_seal_successor_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE_READY__FRESH_M1464_REQUIRED__NO_LAUNCH",
         "contract status")
    identity = contract.get("identity", {})
    need(identity == {
        "runner_path": RUNNER.relative_to(HW).as_posix(),
        "runner_sha256": sha(RUNNER),
        "source_checker_path": HERE.joinpath(Path(__file__).name).relative_to(HW).as_posix(),
        "source_checker_sha256": sha(Path(__file__).resolve()),
        "source_tests_path": TESTS.relative_to(HW).as_posix(),
        "source_tests_sha256": sha(TESTS),
        "runtime_tests_path": RUNTIME_TESTS.relative_to(HW).as_posix(),
        "runtime_tests_sha256": sha(RUNTIME_TESTS),
        "m1433_runner_sha256": sha(OLD_RUNNER),
    }, "contract identity drift")
    evidence = contract.get("predecessor_failure_evidence", {})
    need(evidence.get("compile_count") == 0 and evidence.get("sim_count") == 0
         and evidence.get("phase") == "ATTEMPT_CONSUME"
         and evidence.get("root_cause") ==
         "generic_recursive_seal_verifier_unconditionally_required_review_json",
         "contract failure evidence drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim boundary drift")
    need(contract.get("future_execution", {}).get("automatic_retry") is False,
         "retry drift")
    need(contract.get("future_execution", {}).get("maximum_vcs_compiles") == 1
         and contract.get("future_execution", {}).get("maximum_simv_runs") == 1,
         "future tool cardinality drift")
    need(contract.get("author_execution", {}).get("vcs") is False
         and contract.get("author_execution", {}).get("simv") is False
         and contract.get("author_execution", {}).get("eda") is False,
         "author execution drift")
    return contract


def check_source(require_future_absent: bool = True) -> dict:
    need(sha(DOCS359) == DOCS359_SHA, "docs359 drift")
    failure = check_failure_evidence()
    runner = check_runner_static()
    contract = check_contract()
    need(not any(os.path.lexists(path) for path in
                 (M1459_ATTEMPT, M1459_RESULT, M1459_QUARANTINE)),
         "M1459 canonical namespace is not fresh")
    if require_future_absent:
        need(not any(os.path.lexists(path) for path in (M1464, M1465, M1466)),
             "future authority already exists")
    return {"schema": "m1459_c1_generic_seal_successor_source_check_r1_v1",
            "status": "PASS_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE__NO_VCS_NO_EDA",
            "runner": runner, "failure_phase": failure["phase"],
            "contract_status": contract["status"], "claim_boundary": CLAIMS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    args = parser.parse_args()
    del args
    print(json.dumps(check_source(require_future_absent=True), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
