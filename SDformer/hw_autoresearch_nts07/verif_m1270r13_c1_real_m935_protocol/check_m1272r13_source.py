#!/usr/bin/env python3
"""M1272 checker-only repair for the frozen M1270/R13 real-M935 TB."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parent
TB = HERE / "tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
CONTRACT = HW / "contracts/m1270_c1_r13_real_m935_integrated_protocol_source_contract_r1_20260830.json"
M528 = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1162 = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    TB: "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    CONTRACT: "f17a02226b4d8a391d6cbb5830e16f7e0716b7a9f1e342457add79e0438e15ee",
    M528: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1162: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
ENTER = "PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER"
DONE = "PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE"
PASS = "PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lexical_view(text, keep_strings):
    """Remove comments; optionally retain complete string literals."""
    out = []
    i = 0
    state = "normal"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "normal":
            if ch == "/" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "line"
                continue
            if ch == "/" and nxt == "*":
                out.extend("  ")
                i += 2
                state = "block"
                continue
            if ch == '"':
                out.append(ch if keep_strings else " ")
                i += 1
                state = "string"
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line":
            out.append("\n" if ch == "\n" else " ")
            i += 1
            if ch == "\n":
                state = "normal"
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "normal"
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if state == "string":
            if ch == "\\" and i + 1 < len(text):
                if keep_strings:
                    out.extend(text[i:i + 2])
                else:
                    out.extend("  ")
                i += 2
                continue
            out.append(ch if keep_strings else " ")
            i += 1
            if ch == '"':
                state = "normal"
            continue
    if state in ("block", "string"):
        raise AssertionError("unterminated block comment or string")
    return "".join(out)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def task_body(view, name):
    match = re.search(r"\btask\s+automatic\s+" + re.escape(name)
                      + r"\b(.*?)\bendtask\b", view, re.S)
    require(match is not None, "missing task " + name)
    require(len(re.findall(r"\btask\s+automatic\s+" + re.escape(name)
                           + r"\b", view)) == 1,
            "duplicate task " + name)
    return match.group(1)


def display_tokens(view):
    return re.findall(r'\$display\s*\(\s*"([A-Za-z0-9_]+)', view)


def check_text(text):
    executable = lexical_view(text, True)
    structural = lexical_view(text, False)
    request_object = (r"(?:dut(?:\.u_frozen_m935)?\.)?"
                      r"issue_request_[A-Za-z0-9_]+")

    require(len(re.findall(r"\binitial\s+begin\b", structural)) == 1,
            "must have exactly one executable initial block")
    require(not re.search(r"\b(?:force|release)\s+" + request_object,
                          structural),
            "bare or hierarchical issue_request force/release present")
    require(not re.search(r"\b" + request_object
                          + r"\s*(?:<=|(?<![=!<>])=(?!=))", structural),
            "bare or hierarchical issue_request assignment present")

    # M1271 P1: exact runtime tokens must be executable statements in the sole
    # authoritative initial flow, not comments, strings, or dormant tasks.
    initial = re.search(r"\binitial\s+begin\b(.*?)\bend\s*\nendmodule\b",
                        executable, re.S)
    require(initial is not None, "authoritative initial flow not found")
    initial_body = initial.group(1)
    tokens = display_tokens(initial_body)
    require(tokens.count(ENTER) == 1 and tokens.count(DONE) == 1
            and tokens.count(PASS) == 1,
            "exact executable PHASE/PASS tokens missing or duplicated")
    require(tokens.index(ENTER) < tokens.index(DONE) < tokens.index(PASS),
            "executable PHASE/PASS order invalid")
    require(len(re.findall(r"\breal_m935_completion\s*\(\s*\)\s*;",
                           initial_body)) == 1,
            "real completion call not exactly once in initial flow")

    # M1271 P1: the authoritative completion task must execute both exact beat
    # calls and no statically false guard/control escape may disable them.
    completion = task_body(executable, "real_m935_completion")
    require(len(re.findall(r"\bserve_real_m935_beat\s*\(\s*1'b1\s*,\s*0\s*\)\s*;",
                           completion)) == 1,
            "real first-beat call missing/duplicated")
    require(len(re.findall(r"\bserve_real_m935_beat\s*\(\s*1'b0\s*,\s*1\s*\)\s*;",
                           completion)) == 1,
            "real non-first call missing/duplicated")
    require(not re.search(r"\bif\s*\(\s*(?:0|1'b0|1'h0)\s*\)",
                          structural), "statically false guard present")
    require(not re.search(r"\b(?:disable|return|break|continue)\b",
                          structural), "control escape present")
    require(not re.search(r"`(?:ifdef|ifndef|elsif|else|endif)\b",
                          structural), "conditional compilation present")

    # M1271 P1: the executable operand print and flush must precede the sole
    # fatal in the oracle body. Raw/commented text cannot satisfy this.
    oracle_exec = task_body(executable, "oracle")
    oracle_struct = task_body(structural, "oracle")
    displays = list(re.finditer(r'\$display\s*\(\s*"ORACLE_M1270R13\b',
                                oracle_exec))
    flushes = list(re.finditer(r"\$fflush\s*\(\s*\)\s*;", oracle_struct))
    fatals = list(re.finditer(r"\$fatal\s*\(", oracle_struct))
    require(len(displays) == 1 and len(flushes) == 1 and len(fatals) == 1,
            "oracle executable display/flush/fatal cardinality invalid")
    require(displays[0].start() < flushes[0].start() < fatals[0].start(),
            "oracle operand display/flush does not dominate fatal")
    # Textual order alone is not dominance: reject a runtime guard placed in
    # front of the operand print, and require the only branch between the
    # flush and fatal to be the authoritative X-safe oracle predicate.
    before_display = oracle_struct[:displays[0].start()]
    require(not re.search(
        r"\b(?:if|case|casex|casez|while|for|repeat|forever|fork|disable|return)\b",
        before_display), "oracle operand display is conditionally reachable")
    flush_to_fatal = oracle_struct[flushes[0].end():fatals[0].start()]
    require(re.fullmatch(r"\s*if\s*\(\s*condition\s*!==\s*1'b1\s*\)\s*",
                         flush_to_fatal) is not None,
            "oracle print/flush does not directly dominate X-safe fatal")
    require("condition !== 1'b1" in oracle_struct,
            "oracle does not reject X condition")
    require(len(re.findall(r"\$fatal\s*\(", structural)) == 1,
            "fatal exists outside sole operand oracle")

    # Preserve the remaining M1270 semantic and claim-boundary gates.
    require("prep_mask = (row == 0) ? 16'h0003 : 16'h0000;" in structural,
            "normal mask changed")
    require('oracle("first_weight_only_join_hold"' in executable
            and "repeat (2)" in structural, "join hold missing")
    require("psum_fire_count == p0 + expect_first" in structural,
            "first-only psum count missing")
    require("second_response_cycle - first_response_cycle >= 2" in structural,
            "II>=2 missing")
    require("count_issue_accepts == issue0 + 2" in structural
            and "row_complete_count == row0 + 1" in structural
            and "task_done_count == done0 + 1" in structural,
            "completion oracle missing")
    require("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva"
            in structural, "R3 SVA instance missing")
    require("m1168r3_service_assumption_checker u_service_checker"
            in structural, "service checker instance missing")
    pass_match = re.search(r'\$display\s*\(\s*"(' + re.escape(PASS)
                           + r'[^"\\]*)"\s*\)\s*;', initial_body)
    require(pass_match is not None, "exact executable PASS statement missing")
    pass_payload = pass_match.group(1)
    for literal in (
        "real_m935=true", "parent_issue_override=0",
        "child_issue_override=0", "first_beats=1", "nonfirst_beats=1",
        "weight_requests=2", "psum_requests=1", "ii_ge_2=true",
        "every_oracle_operands=true", "zero_sva_failures_required=true",
        "functional_vcs=false", "timing_verified=false",
        "cycles_measured=false", "speedup=false", "ppa=false",
        "energy=false", "system_speedup=false", "headline=false",
    ):
        require(literal in pass_payload, "PASS boundary missing " + literal)

    return {
        "schema": "m1272r13_real_m935_checker_only_static_v1",
        "status": "PASS_CHECKER_TESTS_ONLY__NO_VCS_NO_EDA",
        "exact_executable_tokens": 3,
        "real_completion_calls_in_initial": 1,
        "real_beat_calls_in_completion": 2,
        "issue_request_assignments": 0,
        "oracle_executable_displays": 1,
        "fatal_sites_outside_oracle": 0,
    }


def main():
    for path, expected in EXPECTED_SHA.items():
        require(path.is_file(), "missing frozen input " + str(path))
        require(sha(path) == expected, "frozen SHA drift " + str(path))
    result = check_text(TB.read_text())
    result["frozen_sha256"] = {str(p.relative_to(HW)): v
                                for p, v in EXPECTED_SHA.items()}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": "m1272r13_real_m935_checker_only_static_v1",
                          "status": "FAIL_CLOSED", "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
