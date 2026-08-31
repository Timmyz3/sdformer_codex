#!/usr/bin/env python3
"""Independent, source-only nearby mutations for the M1258/R12 checker."""

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "verif_m1258r12_c1_common_charge_protocol"
TB = DIR / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
CHECKER = DIR / "check_m1258r12_source.py"


def run(text):
    with tempfile.NamedTemporaryFile("w", suffix=".sv", delete=False) as handle:
        handle.write(text)
        candidate = Path(handle.name)
    try:
        result = subprocess.run(
            ["python3", str(CHECKER), "--candidate", str(candidate)],
            cwd=str(ROOT), universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result.returncode, json.loads(result.stdout)
    finally:
        candidate.unlink()


source = TB.read_text()
cases = []


def case(name, mutant, expected_accept):
    if name != "canonical" and mutant == source:
        raise RuntimeError("mutation did not change source: " + name)
    code, payload = run(mutant)
    accepted = code == 0 and not payload.get("errors")
    cases.append({
        "name": name,
        "expected_accept": expected_accept,
        "observed_accept": accepted,
        "checker_exit_code": code,
        "checker_errors": payload.get("errors", []),
        "expectation_met": accepted == expected_accept,
    })


case("canonical", source, True)
case(
    "executable_parent_request_extra",
    source.replace(
        "// M1258/R12 additive",
        "force dut.issue_request_valid = 1'b1;\n// M1258/R12 additive", 1),
    False)
case(
    "parent_request_comment_decoy",
    source.replace(
        "// M1258/R12 additive",
        "// force dut.issue_request_valid = 1'b1;\n// M1258/R12 additive", 1),
    True)
case(
    "child_valid_prefix_shadow",
    source.replace(
        "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
        "force dut.u_frozen_m935.issue_request_valid_shadow = 1'b1;", 1),
    False)
case(
    "child_seam_comment_decoy_parent_actual",
    source.replace(
        "force dut.u_frozen_m935.issue_request_parent_id = 6'b0;",
        "// force dut.u_frozen_m935.issue_request_parent_id = 6'b0;\n"
        "            force dut.issue_request_parent_id = 6'b0;", 1),
    False)
case(
    "boundary_true_comment_decoy_false_actual",
    source.replace(
        "// M1258/R12 additive",
        "// boundary_only=true\n// M1258/R12 additive", 1).replace(
        "boundary_only=true integrated_random=false",
        "boundary_only=false integrated_random=false", 1),
    False)
case(
    "integrated_normal_true_comment_decoy_false_actual",
    source.replace(
        "integrated_normal_m935_evidence=true",
        "integrated_normal_m935_evidence=false", 1).replace(
        "// M1258/R12 additive",
        "// integrated_normal_m935_evidence=true\n// M1258/R12 additive", 1),
    False)
case(
    "integrated_random_inflation",
    source.replace("integrated_random=false", "integrated_random=true", 1),
    False)
case(
    "integrated_m935_inflation",
    source.replace(
        "integrated_m935_claim=false", "integrated_m935_claim=true", 1),
    False)
case(
    "normal_load_task_semantic_drift",
    source.replace(
        "prep_mask = (row == 0) ? 16'h0003 : 16'h0000;",
        "prep_mask = (row == 0) ? 16'h0007 : 16'h0000;", 1),
    False)
case(
    "normal_serve_task_comment_drift",
    source.replace(
        "        input integer beat_index\n    );",
        "        input integer beat_index\n    );\n        // independent drift", 1),
    False)
case(
    "normal_completion_call_removed",
    source.replace(
        "        normal_m935_completion();",
        "        // normal_m935_completion();", 1),
    False)

unexpected = [item["name"] for item in cases if not item["expectation_met"]]
print(json.dumps({
    "schema": "m1260_m1258_independent_mutations_r1_v1",
    "status": "FAIL_CLOSED" if unexpected else "PASS",
    "cases_run": len(cases),
    "unexpected_cases": unexpected,
    "cases": cases,
}, indent=2, sort_keys=True))
raise SystemExit(1 if unexpected else 0)
