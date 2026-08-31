#!/usr/bin/env python3
"""Fail-closed source-only checker for M1270/R13 real-M935 harness."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parent
TB = HERE / "tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
M528 = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1162 = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    M528: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1162: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PHASE_ENTER = "PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER"
PHASE_DONE = "PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE"
PASS_TOKEN = "PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_without_comments_or_strings(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return text


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_text(text):
    code = code_without_comments_or_strings(text)
    require(text.count("module tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13;") == 1,
            "exact top missing or duplicated")
    require(len(re.findall(r"\binitial\s+begin\b", code)) == 1,
            "must have exactly one initial block")
    require(not re.search(r"(?m)^\s*(?:force|release)\b", code),
            "procedural hierarchical override keyword present")
    require(not re.search(r"\bdut(?:\.u_frozen_m935)?\.issue_request_[A-Za-z0-9_]+\s*(?:<=|(?<![=!<>])=(?!=))", code),
            "issue-request hierarchy is assigned")
    require(len(re.findall(r"\$fatal\s*\(", code)) == 1,
            "all failures must route through the sole operand-printing oracle")
    oracle_match = re.search(r"task\s+automatic\s+oracle\b(.*?)endtask", text, re.S)
    require(oracle_match is not None, "oracle task missing")
    oracle_body = oracle_match.group(1)
    require("ORACLE_M1270R13" in oracle_body and "$fflush" in oracle_body
            and "$fatal" in oracle_body, "oracle must print/flush before fatal")
    require("condition !== 1'b1" in oracle_body,
            "oracle must reject X as well as false")

    for task in ("load_normal_task", "serve_real_m935_beat",
                 "real_m935_completion"):
        require(len(re.findall(rf"task\s+automatic\s+{task}\b", code)) == 1,
                f"{task} missing or duplicated")
    require("prep_mask = (row == 0) ? 16'h0003 : 16'h0000;" in text,
            "authoritative two-source normal mask changed")
    require("serve_real_m935_beat(1'b1, 0);" in text
            and "serve_real_m935_beat(1'b0, 1);" in text,
            "real first/non-first sequence missing")
    require('oracle("first_weight_only_join_hold"' in text
            and "repeat (2)" in text,
            "first-beat response join hold missing")
    require("psum_fire_count == p0 + expect_first" in text,
            "first-only psum exact count missing")
    require("second_response_cycle - first_response_cycle >= 2" in text,
            "II>=2 oracle missing")
    require("count_issue_accepts == issue0 + 2" in text
            and "row_complete_count == row0 + 1" in text
            and "task_done_count == done0 + 1" in text,
            "architectural completion oracle missing")
    require("!protocol_error && !dut.boundary_fault_q" in text
            and text.count("!dut.core_protocol_error") >= 2
            and text.count("!dut.u_frozen_m935.fault_q") >= 2,
            "zero-fault oracle missing")
    require("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva" in text,
            "frozen R3 SVA not instantiated")
    require("m1168r3_service_assumption_checker u_service_checker" in text,
            "service assumption checker not instantiated")

    display_tokens = re.findall(r'\$display\("([A-Za-z0-9_]+)', text)
    require(display_tokens.count(PHASE_ENTER) == 1
            and display_tokens.count(PHASE_DONE) == 1
            and display_tokens.count(PASS_TOKEN) == 1,
            "phase/PASS display tokens missing or duplicated")
    require(display_tokens.index(PHASE_ENTER) < display_tokens.index(PHASE_DONE)
            < display_tokens.index(PASS_TOKEN), "phase/PASS order invalid")
    pass_line = next(line for line in text.splitlines() if PASS_TOKEN in line)
    for literal in (
        "real_m935=true", "parent_issue_override=0",
        "child_issue_override=0", "first_beats=1", "nonfirst_beats=1",
        "weight_requests=2", "psum_requests=1", "ii_ge_2=true",
        "every_oracle_operands=true", "zero_sva_failures_required=true",
        "functional_vcs=false", "timing_verified=false",
        "cycles_measured=false", "speedup=false", "ppa=false",
        "energy=false", "system_speedup=false", "headline=false",
    ):
        require(literal in pass_line, f"PASS claim boundary missing {literal}")

    return {
        "schema": "m1270r13_real_m935_source_static_check_v1",
        "status": "PASS_SOURCE_ONLY_NO_VCS_NO_EDA",
        "top": "tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13",
        "single_initial": True,
        "issue_request_overrides": 0,
        "fatal_sites_outside_operand_oracle": 0,
        "real_beats": 2,
        "first_beats": 1,
        "nonfirst_beats": 1,
        "expected_psum_requests": 1,
    }


def main():
    for path, expected in EXPECTED_SHA.items():
        require(path.is_file(), f"missing frozen input {path}")
        require(sha(path) == expected, f"frozen SHA drift {path}")
    require(TB.is_file(), "R13 TB missing")
    result = check_text(TB.read_text())
    result["tb_sha256"] = sha(TB)
    result["frozen_sha256"] = {str(p.relative_to(HW)): v
                                for p, v in EXPECTED_SHA.items()}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": "m1270r13_real_m935_source_static_check_v1",
                          "status": "FAIL_CLOSED", "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
