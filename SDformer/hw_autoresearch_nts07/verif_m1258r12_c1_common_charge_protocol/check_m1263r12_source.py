#!/usr/bin/env python3
"""Final additive checker-only hardening for frozen M1258/R12 TB.

No compiler, simulator, EDA, GPU, or remote command is invoked here.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR_PATH = HERE / "check_m1261r12_source.py"
TB = HERE / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"

spec = importlib.util.spec_from_file_location("m1261_prior", str(PRIOR_PATH))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
base = prior.base


def norm(value):
    return " ".join(value.split())


EXPECTED_FORCE_RELEASE = {
    "force_boundary_core_output_request": [
        ("force", "dut.u_frozen_m935.issue_request_valid = 1'b1"),
        ("force", "dut.u_frozen_m935.issue_request_epoch = force_stage_epoch_q"),
        ("force", "dut.u_frozen_m935.issue_request_row_id = force_stage_row_q"),
        ("force", "dut.u_frozen_m935.issue_request_first = force_stage_first_q"),
        ("force", "dut.u_frozen_m935.issue_request_last = force_stage_last_q"),
        ("force", "dut.u_frozen_m935.issue_request_source_valid = 1'b1"),
        ("force", "dut.u_frozen_m935.issue_request_source_index = force_stage_source_q"),
        ("force", "dut.u_frozen_m935.issue_request_parent_valid = 1'b0"),
        ("force", "dut.u_frozen_m935.issue_request_parent_id = 6'b0"),
        ("force", "dut.u_frozen_m935.issue_data_ready = 1'b1"),
    ],
    "force_boundary_core_output_request_no_ready": [
        ("force", "dut.u_frozen_m935.issue_request_valid = 1'b1"),
        ("force", "dut.u_frozen_m935.issue_request_epoch = force_stage_epoch_q"),
        ("force", "dut.u_frozen_m935.issue_request_row_id = force_stage_row_q"),
        ("force", "dut.u_frozen_m935.issue_request_first = force_stage_first_q"),
        ("force", "dut.u_frozen_m935.issue_request_last = force_stage_last_q"),
        ("force", "dut.u_frozen_m935.issue_request_source_valid = 1'b1"),
        ("force", "dut.u_frozen_m935.issue_request_source_index = force_stage_source_q"),
        ("force", "dut.u_frozen_m935.issue_request_parent_valid = 1'b0"),
        ("force", "dut.u_frozen_m935.issue_request_parent_id = 6'b0"),
    ],
    "release_boundary_core_output_request": [
        ("release", "dut.u_frozen_m935.issue_request_valid"),
        ("release", "dut.u_frozen_m935.issue_request_epoch"),
        ("release", "dut.u_frozen_m935.issue_request_row_id"),
        ("release", "dut.u_frozen_m935.issue_request_first"),
        ("release", "dut.u_frozen_m935.issue_request_last"),
        ("release", "dut.u_frozen_m935.issue_request_source_valid"),
        ("release", "dut.u_frozen_m935.issue_request_source_index"),
        ("release", "dut.u_frozen_m935.issue_request_parent_valid"),
        ("release", "dut.u_frozen_m935.issue_request_parent_id"),
        ("release", "dut.u_frozen_m935.issue_data_ready"),
    ],
    "retire_random_boundary_core_output_tuple": [
        ("release", "dut.u_frozen_m935.issue_request_valid"),
        ("release", "dut.u_frozen_m935.issue_request_epoch"),
        ("release", "dut.u_frozen_m935.issue_request_row_id"),
        ("release", "dut.u_frozen_m935.issue_request_first"),
        ("release", "dut.u_frozen_m935.issue_request_last"),
        ("release", "dut.u_frozen_m935.issue_request_source_valid"),
        ("release", "dut.u_frozen_m935.issue_request_source_index"),
        ("release", "dut.u_frozen_m935.issue_request_parent_valid"),
        ("release", "dut.u_frozen_m935.issue_request_parent_id"),
    ],
    "directed_psum_first_and_backpressure": [
        ("force", "dut.u_frozen_m935.issue_data_ready = 1'b0"),
        ("force", "dut.u_frozen_m935.issue_data_ready = 1'b1"),
    ],
    "directed_ii2": [
        ("force", "dut.u_frozen_m935.issue_request_epoch = 16'h4405"),
        ("force", "dut.u_frozen_m935.issue_request_row_id = 6'd13"),
        ("force", "dut.u_frozen_m935.issue_request_source_index = 4'd2"),
    ],
    "sticky_fault_attacks": [
        ("force", "dut.u_frozen_m935.issue_request_valid = 1'b0"),
        ("force", "dut.u_frozen_m935.issue_request_epoch = 16'hdead"),
    ],
    "random_boundary_transaction": [
        ("force", "dut.u_frozen_m935.issue_data_ready = 1'b0"),
        ("force", "dut.u_frozen_m935.issue_data_ready = 1'b1"),
        ("release", "dut.u_frozen_m935.issue_data_ready"),
    ],
}

EXPECTED_PHASE_DISPLAYS = [
    "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER",
    "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_COMPLETE",
    "PHASE_M1258R12_BOUNDARY_ONLY_RESET_PENDING_ENTER",
    "PHASE_M1258R12_BOUNDARY_ONLY_RESET_PENDING_COMPLETE",
    "PHASE_M1258R12_BOUNDARY_ONLY_STICKY_ATTACKS_ENTER",
    "PHASE_M1258R12_BOUNDARY_ONLY_STICKY_ATTACKS_COMPLETE",
    "PHASE_M1258R12_BOUNDARY_ONLY_SERVICE_ATTACKS_ENTER",
    "PHASE_M1258R12_BOUNDARY_ONLY_SERVICE_ATTACKS_COMPLETE",
    "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_ENTER count=24",
    "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_TRANSACTION_ENTER index=%0d",
    "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_TRANSACTION_COMPLETE index=%0d",
    "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_COMPLETE count=24",
    "PHASE_M1258R12_INTEGRATED_NORMAL_M935_ENTER",
    "PHASE_M1258R12_INTEGRATED_NORMAL_M935_COMPLETE",
]

PASS_TOKEN = "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE"
REQUIRED_PASS_FIELDS = {
    "boundary_only": "true", "integrated_random": "false",
    "parent_connection_force": "0", "child_core_output_seam_force": "1",
    "zero_sva_failures_required": "true", "boundary_fault": "0",
    "core_fault": "0", "functional_vcs_only": "false",
    "timing_verified": "false", "cycles_measured": "false",
    "speedup": "false", "ppa": "false", "energy": "false",
    "system_speedup": "false", "headline": "false",
    "integrated_m935_claim": "false",
    "integrated_normal_m935_evidence": "true",
}


def system_string_calls(text, name):
    """Lex executable calls whose first argument is one SV string literal."""
    calls = []
    index = 0
    state = "code"
    needle = "$" + name
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "line":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state == "string":
            if char == "\\" and nxt:
                index += 2
            elif char == '"':
                state = "code"
                index += 1
            else:
                index += 1
            continue
        if char == "/" and nxt == "/":
            state = "line"
            index += 2
            continue
        if char == "/" and nxt == "*":
            state = "block"
            index += 2
            continue
        if char == '"':
            state = "string"
            index += 1
            continue
        if text.startswith(needle, index):
            after = index + len(needle)
            if after < len(text) and (text[after].isalnum() or text[after] == "_"):
                index += 1
                continue
            cursor = after
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "(":
                index += 1
                continue
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != '"':
                index += 1
                continue
            cursor += 1
            value = []
            while cursor < len(text):
                if text[cursor] == "\\" and cursor + 1 < len(text):
                    value.extend((text[cursor], text[cursor + 1]))
                    cursor += 2
                    continue
                if text[cursor] == '"':
                    calls.append("".join(value))
                    cursor += 1
                    break
                value.append(text[cursor])
                cursor += 1
            index = cursor
            continue
        index += 1
    return calls


def task_force_release_inventory(executable):
    tasks = {}
    covered_spans = []
    pattern = re.compile(
        r"^    task automatic\s+(\w+)\b.*?^    endtask$",
        flags=re.MULTILINE | re.DOTALL)
    statement_pattern = re.compile(r"\b(force|release)\s+([^;]+);")
    for match in pattern.finditer(executable):
        name = match.group(1)
        statements = [(kind, norm(body)) for kind, body in
                      statement_pattern.findall(match.group(0))]
        if statements:
            tasks[name] = statements
        covered_spans.append((match.start(), match.end()))
    outside = []
    for match in statement_pattern.finditer(executable):
        if not any(start <= match.start() < end for start, end in covered_spans):
            outside.append((match.group(1), norm(match.group(2))))
    return tasks, outside


def strict_audit(text):
    errors = list(prior.hard_audit(text))
    executable = base.strip_comments_strings(text)

    observed, outside = task_force_release_inventory(executable)
    if outside:
        errors.append("force/release outside automatic task: %r" % (outside,))
    if observed != EXPECTED_FORCE_RELEASE:
        errors.append("exact ordered force/release task inventory mismatch")

    display_strings = system_string_calls(text, "display")
    phase_displays = [value for value in display_strings
                      if value.split(None, 1)[0].startswith("PHASE_M1258R12")]
    if phase_displays != EXPECTED_PHASE_DISPLAYS:
        errors.append("exact executable phase $display inventory mismatch")

    pass_neighbours = [value for value in display_strings
                       if value.split(None, 1)[0].startswith(PASS_TOKEN)]
    pass_displays = [value for value in pass_neighbours
                     if value.split(None, 1)[0] == PASS_TOKEN]
    if len(pass_neighbours) != 1 or len(pass_displays) != 1:
        errors.append("exact executable PASS $display count/token mismatch")
    else:
        fields = {}
        duplicates = set()
        for item in pass_displays[0].split()[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            if key in fields:
                duplicates.add(key)
            fields[key] = value
        if duplicates:
            errors.append("duplicate exact PASS fields: " +
                          ",".join(sorted(duplicates)))
        for key, value in REQUIRED_PASS_FIELDS.items():
            if fields.get(key) != value:
                errors.append("exact PASS field mismatch: " + key)

    normal_calls = len(re.findall(
        r"\bnormal_m935_completion\s*\(\s*\)\s*;", executable))
    if normal_calls != 1:
        errors.append("exact executable integrated-normal call count != 1")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=TB)
    args = parser.parse_args()
    errors = strict_audit(args.candidate.read_text())
    payload = {
        "schema": "m1263r12_final_checker_source_static_r1_v1",
        "status": "PASS_M1263_R12_FINAL_CHECKER_SOURCE_ONLY" if not errors
                  else "FAIL_CLOSED",
        "candidate": str(args.candidate),
        "candidate_sha256": base.sha(args.candidate),
        "prior_checker_sha256": base.sha(PRIOR_PATH),
        "checks_source_only": True,
        "vcs_invoked": False,
        "release_published": False,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
