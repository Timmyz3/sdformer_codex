#!/usr/bin/env python3
"""M1261 additive hardening for the frozen M1258/R12 source checker.

Source only: this checker never invokes VCS, simv, EDA, GPU, or remote work.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "check_m1258r12_source.py"
TB = HERE / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"

spec = importlib.util.spec_from_file_location("m1258_base", str(BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


ALLOWED_SEAM = {
    "dut.u_frozen_m935.issue_request_" + suffix
    for suffix in (
        "valid", "epoch", "row_id", "first", "last", "source_valid",
        "source_index", "parent_valid", "parent_id")
}
ALLOWED_SEAM.add("dut.u_frozen_m935.issue_data_ready")


def sv_strings(text):
    """Return SystemVerilog string literal contents while ignoring comments."""
    strings = []
    state = "code"
    value = []
    index = 0
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
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
                value = []
                index += 1
                continue
            index += 1
            continue
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
        if char == "\\" and nxt:
            value.extend((char, nxt))
            index += 2
            continue
        if char == '"':
            strings.append("".join(value))
            state = "code"
            index += 1
            continue
        value.append(char)
        index += 1
    return strings


def hard_audit(text):
    errors = list(base.audit(text))
    executable = base.strip_comments_strings(text)
    strings = sv_strings(text)

    statements = re.findall(r"\b(force|release)\s+([^;]+);", executable)
    for kind, body in statements:
        target = body.split("=", 1)[0].strip() if kind == "force" else body.strip()
        if target not in ALLOWED_SEAM:
            errors.append("non-exact child seam %s target: %s" % (kind, target))

    helpers = {
        "force_boundary_core_output_request": {
            "force": ALLOWED_SEAM,
            "release": set(),
        },
        "force_boundary_core_output_request_no_ready": {
            "force": ALLOWED_SEAM - {"dut.u_frozen_m935.issue_data_ready"},
            "release": set(),
        },
        "release_boundary_core_output_request": {
            "force": set(),
            "release": ALLOWED_SEAM,
        },
        "retire_random_boundary_core_output_tuple": {
            "force": set(),
            "release": ALLOWED_SEAM - {"dut.u_frozen_m935.issue_data_ready"},
        },
    }
    for name, expected in helpers.items():
        body = base.task(executable, name) or ""
        observed = {"force": set(), "release": set()}
        for kind, statement in re.findall(r"\b(force|release)\s+([^;]+);", body):
            target = (statement.split("=", 1)[0].strip()
                      if kind == "force" else statement.strip())
            observed[kind].add(target)
        for kind in ("force", "release"):
            if observed[kind] != expected[kind]:
                errors.append("exact %s inventory mismatch in %s" % (kind, name))

    phase_tokens = []
    for phase in (
            "DIRECTED", "RESET_PENDING", "STICKY_ATTACKS",
            "SERVICE_ATTACKS", "RANDOM"):
        phase_tokens.extend([
            "PHASE_M1258R12_BOUNDARY_ONLY_%s_ENTER" % phase,
            "PHASE_M1258R12_BOUNDARY_ONLY_%s_COMPLETE" % phase,
        ])
    phase_tokens.extend([
        "PHASE_M1258R12_INTEGRATED_NORMAL_M935_ENTER",
        "PHASE_M1258R12_INTEGRATED_NORMAL_M935_COMPLETE",
    ])
    for token in phase_tokens:
        count = sum(item.startswith(token) for item in strings)
        if count != 1:
            errors.append("executable display phase count != 1: " + token)

    pass_prefix = "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE"
    pass_strings = [item for item in strings if item.startswith(pass_prefix)]
    if len(pass_strings) != 1:
        errors.append("executable PASS display count != 1")
    else:
        fields = {}
        duplicates = set()
        for token in pass_strings[0].split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in fields:
                duplicates.add(key)
            fields[key] = value
        required = {
            "boundary_only": "true",
            "integrated_random": "false",
            "parent_connection_force": "0",
            "child_core_output_seam_force": "1",
            "zero_sva_failures_required": "true",
            "boundary_fault": "0",
            "core_fault": "0",
            "functional_vcs_only": "false",
            "timing_verified": "false",
            "cycles_measured": "false",
            "speedup": "false",
            "ppa": "false",
            "energy": "false",
            "system_speedup": "false",
            "headline": "false",
            "integrated_m935_claim": "false",
            "integrated_normal_m935_evidence": "true",
        }
        if duplicates:
            errors.append("duplicate PASS fields: " + ",".join(sorted(duplicates)))
        for key, expected in required.items():
            if fields.get(key) != expected:
                errors.append("PASS field mismatch: %s" % key)

    normal_calls = len(re.findall(
        r"\bnormal_m935_completion\s*\(\s*\)\s*;", executable))
    if normal_calls != 1:
        errors.append("executable normal_m935_completion call count != 1")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=TB)
    args = parser.parse_args()
    text = args.candidate.read_text()
    errors = hard_audit(text)
    payload = {
        "schema": "m1261r12_source_static_check_r1_v1",
        "status": "PASS_M1261_R12_HARDENED_SOURCE_ONLY" if not errors else "FAIL_CLOSED",
        "candidate": str(args.candidate),
        "candidate_sha256": base.sha(args.candidate),
        "base_checker_sha256": base.sha(BASE_PATH),
        "checks_source_only": True,
        "vcs_invoked": False,
        "release_published": False,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
