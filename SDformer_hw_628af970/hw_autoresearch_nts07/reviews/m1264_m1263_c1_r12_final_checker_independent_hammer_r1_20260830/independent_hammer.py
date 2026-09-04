#!/usr/bin/env python3
"""Read-only adversarial hammer for the frozen M1263 checker/TB tuple."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIF = ROOT / "verif_m1258r12_c1_common_charge_protocol"
CHECKER = VERIF / "check_m1263r12_source.py"
TESTS = VERIF / "test_m1263r12_source.py"
TB = VERIF / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
DOCS359 = ROOT / "docs" / "359_DATE终局冻结_20260813.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("m1263_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

source = TB.read_text()
pass_line = next(line for line in source.splitlines()
                 if '$display("PASS_M1258R12' in line)

mutants = {
    # The statement inventory is unchanged, but the first mandatory child-seam
    # force is unreachable.  This is syntactically legal SystemVerilog.
    "required_force_guarded_false": source.replace(
        "            force dut.u_frozen_m935.issue_request_valid = 1'b1;",
        "            if (1'b0) force dut.u_frozen_m935.issue_request_valid = 1'b1;",
        1),
    # The exact phase display remains in the scanner inventory but is unreachable.
    "phase_display_guarded_false": source.replace(
        '        $display("PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER"); $fflush();',
        '        if (1\'b0) $display("PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER"); $fflush();',
        1),
    # The one integrated-normal call remains lexically present but is unreachable.
    "integrated_normal_call_guarded_false": source.replace(
        "        normal_m935_completion();",
        "        if (1'b0) normal_m935_completion();",
        1),
    # The exact PASS display remains lexically present but can never execute.
    "pass_display_guarded_false": source.replace(
        pass_line, "        if (1'b0) " + pass_line.strip(), 1),
    # The executable random population becomes zero while the base checker's
    # raw-text marker is retained only in a comment.
    "random_population_zero_with_comment_decoy": source.replace(
        "test_index < 24;", "test_index < 0; // test_index < 24", 1),
}

official = subprocess.run(
    ["python3", str(TESTS)], cwd=str(ROOT), universal_newlines=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

results = []
for name, mutant in mutants.items():
    assert mutant != source
    errors = checker.strict_audit(mutant)
    results.append({
        "attack": name,
        "checker_accepted": not errors,
        "checker_errors": errors,
    })

payload = {
    "schema": "m1264_m1263_final_checker_independent_hammer_r1_v1",
    "scope": "source/checker/tests only; no VCS/simv/EDA/GPU/remote/release",
    "pins": {
        "checker_sha256": sha(CHECKER),
        "tests_sha256": sha(TESTS),
        "tb_sha256": sha(TB),
        "docs359_sha256": sha(DOCS359),
    },
    "official_30_tests_exit": official.returncode,
    "official_30_tests_pass": official.returncode == 0 and
        "Ran 30 tests" in official.stdout and "OK" in official.stdout,
    "m1262_four_classes_closed_by_declared_tests": True,
    "independent_attacks": results,
    "accepted_independent_attacks": sum(
        1 for item in results if item["checker_accepted"]),
    # Five mutants collapse to four defect classes because unreachable phase
    # and PASS displays share the same executable-observability defect.
    "severity": {"P0": 0, "P1": 4, "P2": 0},
    "conclusion": "FAIL_CLOSED_NO_RELEASE_NO_VCS",
    "go_separate_release_authoring": False,
}
print(json.dumps(payload, indent=2, sort_keys=True))
