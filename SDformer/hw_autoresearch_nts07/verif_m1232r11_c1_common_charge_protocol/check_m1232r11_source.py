#!/usr/bin/env python3
"""Fail-closed static source audit for M1232/R11.

This checker reads local text and hashes only.  It does not compile or launch
VCS, simv, EDA, GPU, or remote work.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


EXPECTED = {
    "r11_tb": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "m1229_review": "3726ff3b3f43ce963d1f22182fe74a86d52261f97661d61ca5de7e8543250aad",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def extract_task(text: str, name: str) -> Optional[str]:
    match = re.search(
        r"    task automatic " + re.escape(name) + r"\b.*?^    endtask$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return None if match is None else match.group(0)


def strip_sv_comments_and_strings(text: str) -> str:
    """Blank SV comments and string bodies while preserving offsets/newlines.

    The checker intentionally does not treat lexical tokens in ``//``,
    ``/*...*/``, or string literals as executable evidence.  Keeping every
    non-newline byte position stable lets the ordered checks below compare
    statement offsets without constructing a full SystemVerilog parser.
    """
    out: List[str] = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                out.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                out.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                out.append(" ")
                index += 1
                state = "string"
                continue
            out.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                out.append("\n")
                state = "code"
            else:
                out.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                out.extend((" ", " "))
                index += 2
                state = "code"
                continue
            out.append("\n" if char == "\n" else " ")
            index += 1
            continue
        # SystemVerilog strings use backslash escapes.  Blank both bytes so an
        # escaped quote cannot terminate the string early.
        if char == "\\" and next_char:
            out.append(" ")
            out.append("\n" if next_char == "\n" else " ")
            index += 2
            continue
        if char == '"':
            out.append(" ")
            index += 1
            state = "code"
            continue
        out.append("\n" if char == "\n" else " ")
        index += 1
    return "".join(out)


def executable_task(text: str, name: str) -> Optional[str]:
    return extract_task(strip_sv_comments_and_strings(text), name)


def audit_text(text: str, r10_text: str,
               enforce_identity: bool = False,
               actual_sha: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    executable = strip_sv_comments_and_strings(text)
    require(
        "module tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11;"
        in executable,
        "wrong R11 module identity", errors)
    require("module tb_m1226r10_" not in executable,
            "R10 module identity reused", errors)
    require(not re.search(r"\bwait\s*\(", executable),
            "unbounded wait statement remains", errors)
    loops = re.findall(r"while\s*\((.*?)\)\s*begin", executable, re.S)
    require(len(loops) == 8,
            "expected 8 bounded while loops, found %d" % len(loops), errors)
    for number, header in enumerate(loops):
        require("watchdog <" in header,
                "while loop %d lacks watchdog bound" % number, errors)

    random_task_raw = extract_task(text, "random_legal_transaction")
    random_task = executable_task(text, "random_legal_transaction")
    require(random_task is not None and random_task_raw is not None,
            "random task missing", errors)
    random_task = random_task or ""
    random_task_raw = random_task_raw or ""
    normal_task = extract_task(text, "serve_normal_beat")
    r10_normal_task = extract_task(r10_text, "serve_normal_beat")
    require(normal_task is not None and normal_task == r10_normal_task,
            "R10 normal exact-retirement task changed", errors)

    request_exact = (
        "while ((weight_fire_count != w0 + 1\n"
        "                    || (first && psum_fire_count != p0 + 1))\n"
        "                    && watchdog < R9_RANDOM_WAIT_LIMIT) begin")
    require(request_exact in random_task,
            "random exact request-fire loop missing", errors)
    require("r11_random_request_overshoot" in random_task_raw and
            "r11_random_request_accept" in random_task_raw,
            "random request watchdog/overshoot dump missing", errors)

    # The exact-one request window must have one executable enable and one
    # executable retirement.  No immediate overwrite, conditional decoy, or
    # comment/string token may satisfy this lifetime proof.
    window_writes = list(re.finditer(
        r"\brandom_request_window_active\s*(?:<=|=)\s*[^;]+;",
        random_task))
    window_enable = list(re.finditer(
        r"\brandom_request_window_active\s*=\s*1'b1\s*;",
        random_task))
    window_disable = list(re.finditer(
        r"\brandom_request_window_active\s*=\s*1'b0\s*;",
        random_task))
    force_request_pos = random_task.find("force_request(first")
    request_loop_pos = random_task.find(request_exact)
    request_retire_pos = random_task.find(
        "weight_req_ready = 1'b0;\n"
        "            psum_req_ready = 1'b0;\n"
        "            random_request_window_active = 1'b0;")
    backpressure_pos = random_task.find(
        "force dut.core_issue_data_ready = 1'b0;")
    window_order_ok = (
        len(window_writes) == 2 and len(window_enable) == 1
        and len(window_disable) == 1 and force_request_pos >= 0
        and request_loop_pos >= 0 and request_retire_pos >= 0
        and backpressure_pos >= 0
        and window_enable[0].start() < force_request_pos < request_loop_pos
        < request_retire_pos < window_disable[0].start() < backpressure_pos
    )
    require(window_order_ok,
            "random exact-one request window lifetime/order is not unique",
            errors)

    ready_retire = (
        "@(negedge clk_core);\n"
        "            weight_req_ready = 1'b0;\n"
        "            psum_req_ready = 1'b0;\n"
        "            random_request_window_active = 1'b0;\n"
        "            #1ps;\n"
        "            if (weight_req_ready || psum_req_ready")
    require(ready_retire in random_task,
            "random request-ready exact retirement missing", errors)
    require("r11_random_request_retire" in random_task_raw and
            "random_weight_request_handshakes != 1" in random_task and
            "random_psum_request_handshakes != first" in random_task,
            "random request retirement oracle incomplete", errors)

    # Ordering anchors: backpressure before response drive, ready transition
    # only at a negedge, and stable payload through the exact accept.
    backpressure = random_task.find(
        "force dut.core_issue_data_ready = 1'b0;")
    response_drive = random_task.find("if (index[0]) begin", backpressure)
    require(backpressure >= 0 and response_drive > backpressure,
            "random response is driven before backpressure", errors)
    ready_negedge = (
        "@(negedge clk_core);\n"
        "            force dut.core_issue_data_ready = 1'b1;")
    require(ready_negedge in random_task,
            "random core-ready does not enter at a negedge", errors)
    hold_writes = list(re.finditer(
        r"(?:\bhold_cycles\s*(?:=|<=|\+=|-=|\*=|/=|%=|\+\+|--)"
        r"|(?:\+\+|--)\s*hold_cycles\b)", random_task))
    hold_assign = list(re.finditer(
        r"\bhold_cycles\s*=\s*1\s*\+\s*prng_q\[9:7\]\s*;",
        random_task))
    hold_repeat = list(re.finditer(
        r"\brepeat\s*\(\s*hold_cycles\s*\)\s*begin",
        random_task))
    hold_order_ok = (
        len(hold_writes) == 1 and len(hold_assign) == 1
        and len(hold_repeat) == 1
        and hold_assign[0].start() < backpressure < response_drive
        < hold_repeat[0].start()
    )
    require("r11_random_response_hold" in random_task_raw and
            "r11_random_response_unstable" in random_task_raw and
            hold_order_ok and
            random_task.count("weight_data !== '0") == 2 and
            random_task.count("psum_rsp_valid !== first") == 2 and
            "response_accept_count != response0 + 1" in random_task,
            "random response stability/exact-accept oracle incomplete",
            errors)

    tuple_call = random_task.find("retire_random_forced_issue_tuple();")
    exact_accept = random_task.find(
        "if (response_accept_count != response0 + 1) begin")
    response_retire = random_task.find(
        "@(negedge clk_core);\n"
        "            weight_rsp_valid = 1'b0;\n"
        "            psum_rsp_valid = 1'b0;\n"
        "            release dut.core_issue_data_ready;")
    tuple_to_response_retire = (
        random_task[tuple_call:response_retire]
        if tuple_call >= 0 and response_retire > tuple_call else "")
    require(exact_accept >= 0 and tuple_call > exact_accept and
            response_retire > tuple_call and
            "@(posedge clk_core)" not in tuple_to_response_retire and
            "@(negedge clk_core)" not in tuple_to_response_retire,
            "random tuple/response retirement ordering broken", errors)
    require("r11_random_tuple_retire" in random_task_raw and
            "dut.request_active_q || weight_req_valid || psum_req_valid"
            in random_task,
            "random forced-tuple retirement proof missing", errors)
    exact_response_retire = (
        "@(negedge clk_core);\n"
        "            weight_rsp_valid = 1'b0;\n"
        "            psum_rsp_valid = 1'b0;\n"
        "            release dut.core_issue_data_ready;")
    require(exact_response_retire in random_task,
            "random immediate-negedge response retirement missing", errors)
    require("r11_random_response_retire" in random_task_raw and
            "r11_random_post_retire" in random_task_raw and
            "@(posedge clk_core); #1ps;\n"
            "            if (weight_fire_count != w0 + 1\n"
            "                    || psum_fire_count != p0 + first\n"
            "                    || response_accept_count != response0 + 1"
            in random_task,
            "random post-retirement sampled-edge proof missing", errors)

    retire_task = executable_task(
        text, "retire_random_forced_issue_tuple") or ""
    for signal in (
        "issue_request_valid", "issue_request_epoch", "issue_request_row_id",
        "issue_request_first", "issue_request_last",
        "issue_request_source_valid", "issue_request_source_index",
        "issue_request_parent_valid", "issue_request_parent_id",
    ):
        require("release dut.%s;" % signal in retire_task,
                "random tuple retire misses %s" % signal, errors)
    require("core_issue_data_ready" not in retire_task,
            "tuple-retire helper releases core ready early", errors)

    # No legal-random assertion mask is permitted.
    require("request_hold_attack_mode =" not in random_task and
            "weight_service_attack_mode =" not in random_task and
            "psum_service_attack_mode =" not in random_task,
            "random legal task changes an SVA attack mask", errors)

    for marker in (
        "test_index < 24", "cov_random_transactions != 24",
        "cov_normal_issue != 2 || cov_normal_row != 1",
        "|| cov_normal_task != 1 || cov_legal_masks_clear != 29",
        "cov_request_attack_windows != 2",
        "cov_weight_service_attack_windows != 1",
        "cov_psum_service_attack_windows != 1",
        "directed_ii2();", "cov_ii2 != 1",
        "prep_mask = (row == 0) ? 16'h0003 : 16'h0000",
        "serve_normal_beat(1'b1, 0);",
        "serve_normal_beat(1'b0, 1);",
    ):
        require(marker in executable,
                "frozen workload marker missing: %s" % marker,
                errors)
    require(
        "PASS_M1232R11_M1162_COMMON_CHARGE_PROTOCOL_SOURCE_CANDIDATE" in text,
        "R11 source-only PASS token missing", errors)
    require("random_request_single_fire=1" in text and
            "random_response_exact_accept=1" in text and
            "random_tuple_retire=1" in text and
            "random_post_retire_edge=1" in text,
            "R11 random source claims incomplete", errors)
    require("zero_sva_failures_required=true" in text,
            "zero-SVA-failure execution gate missing", errors)
    require("functional_vcs_only=false timing_verified=false" in text and
            "system_speedup=false headline=false" in text,
            "source-only claim boundary changed", errors)

    if enforce_identity:
        require(actual_sha == EXPECTED["r11_tb"],
                "R11 SHA mismatch %s" % actual_sha, errors)
    return errors


def canonical_paths(root: Path) -> Dict[str, Path]:
    return {
        "r11_tb": root / "verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv",
        "r10_tb": root / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
        "m1229_review": root / "reviews/m1229_m1226_c1_r10_tb_source_independent_hammer_r1_20260830/review.json",
        "m528": root / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
        "m935": root / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
        "m1162": root / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
        "sva_r3": root / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
        "docs359": root / "docs/359_DATE终局冻结_20260813.md",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paths = canonical_paths(args.root.resolve())
    errors: List[str] = []
    hashes: Dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            errors.append("missing canonical file %s: %s" % (key, path))
            continue
        hashes[key] = sha256(path)
        if hashes[key] != EXPECTED[key]:
            errors.append("%s SHA mismatch %s" % (key, hashes[key]))
    if "r11_tb" in hashes and "r10_tb" in hashes:
        errors.extend(audit_text(
            paths["r11_tb"].read_text(), paths["r10_tb"].read_text(),
            True, hashes["r11_tb"]))
    result = {
        "milestone": "M1246",
        "status": (
            "PASS_M1246_R11_CHECKER_HARDENED_SOURCE_ONLY"
            if not errors else "FAIL_M1246_R11_CHECKER_HARDENING"),
        "checker_tests_only": True,
        "source_only": True,
        "vcs_invoked": False,
        "release_published": False,
        "rtl_mutated": False,
        "fresh_independent_hammer_required": True,
        "hashes": hashes,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
