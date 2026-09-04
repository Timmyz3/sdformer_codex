#!/usr/bin/env python3
"""Fail-closed, source-only audit for the M1258/R12 boundary seam TB.

This script reads local source and hashes.  It never compiles or invokes VCS,
simv, EDA, GPU, or remote work.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "verif_m1258r12_c1_common_charge_protocol" / (
    "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv")
R11 = ROOT / "verif_m1232r11_c1_common_charge_protocol" / (
    "tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv")
PATHS = {
    "m528": ROOT / "rtl_m528_dw1rw" /
        "m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": ROOT / "rtl_m935_c1_match_pipeline" /
        "m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": ROOT / "rtl_m1162_c1_common_charge_protocol" /
        "m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": ROOT / "verif_m1168r3_c1_common_charge_protocol" /
        "m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "m1256_review": ROOT / "reviews" /
        "m1256_m1250_c1_r11_vcs_failure_forensic_r1_20260830" / "review.json",
    "docs359": ROOT / "docs" / "359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "r11": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "m1256_review": "80cfef8664978128e1c67dc5546e2669ffedb37812ce8031bd3526a549e751c1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str, errors: List[str]) -> None:
    if not ok:
        errors.append(message)


def strip_comments_strings(text: str) -> str:
    out: List[str] = []
    state = "code"
    index = 0
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                out.extend((" ", " "))
                index += 2
                state = "line"
                continue
            if char == "/" and nxt == "*":
                out.extend((" ", " "))
                index += 2
                state = "block"
                continue
            if char == '"':
                out.append(" ")
                index += 1
                state = "string"
                continue
            out.append(char)
            index += 1
            continue
        if state == "line":
            if char == "\n":
                out.append("\n")
                state = "code"
            else:
                out.append(" ")
            index += 1
            continue
        if state == "block":
            if char == "*" and nxt == "/":
                out.extend((" ", " "))
                index += 2
                state = "code"
                continue
            out.append("\n" if char == "\n" else " ")
            index += 1
            continue
        if char == "\\" and nxt:
            out.extend((" ", "\n" if nxt == "\n" else " "))
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


def task(text: str, name: str) -> Optional[str]:
    match = re.search(
        r"^    task automatic " + re.escape(name) + r"\b.*?^    endtask$",
        text, flags=re.MULTILINE | re.DOTALL)
    return None if match is None else match.group(0)


def audit(text: str) -> List[str]:
    errors: List[str] = []
    executable = strip_comments_strings(text)
    r11_text = R11.read_text()

    require(
        "module tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12;"
        in executable, "wrong R12 module identity", errors)
    require("random_legal_transaction" not in executable,
            "random path still called legal/integrated", errors)
    require(task(executable, "random_boundary_transaction") is not None,
            "boundary-only random task missing", errors)

    for name, path in PATHS.items():
        require(sha(path) == EXPECTED[name], "frozen hash changed: " + name,
                errors)
    require(sha(R11) == EXPECTED["r11"], "R11 identity changed", errors)

    force_statements = re.findall(
        r"\b(?:force|release)\s+([^;]+);", executable)
    require(bool(force_statements), "no executable seam force inventory", errors)
    for target in force_statements:
        target = target.strip()
        require(not target.startswith("dut.issue_request"),
                "parent issue_request force/release remains: " + target,
                errors)
        require(not target.startswith("dut.core_issue_data_ready"),
                "parent core-ready force/release remains: " + target, errors)
        require(target.startswith("dut.u_frozen_m935.issue_request") or
                target.startswith("dut.u_frozen_m935.issue_data_ready"),
                "force/release is outside child core-output seam: " + target,
                errors)

    request_helper = task(executable,
                          "force_boundary_core_output_request") or ""
    no_ready_helper = task(
        executable, "force_boundary_core_output_request_no_ready") or ""
    release_helper = task(executable,
                          "release_boundary_core_output_request") or ""
    retire_helper = task(
        executable, "retire_random_boundary_core_output_tuple") or ""
    for signal in (
        "valid", "epoch", "row_id", "first", "last", "source_valid",
        "source_index", "parent_valid", "parent_id"):
        require("force dut.u_frozen_m935.issue_request_" + signal
                in request_helper, "request helper misses child seam " + signal,
                errors)
        require("force dut.u_frozen_m935.issue_request_" + signal
                in no_ready_helper,
                "no-ready helper misses child seam " + signal, errors)
        require("release dut.u_frozen_m935.issue_request_" + signal
                in release_helper,
                "release helper misses child seam " + signal, errors)
        require("release dut.u_frozen_m935.issue_request_" + signal
                in retire_helper,
                "random retire misses child seam " + signal, errors)
    require("force dut.u_frozen_m935.issue_data_ready = 1'b1;"
            in request_helper, "request helper misses child-ready seam", errors)
    require("issue_data_ready" not in no_ready_helper,
            "service no-ready helper reaches child-ready seam", errors)
    require("release dut.u_frozen_m935.issue_data_ready;" in release_helper,
            "release helper misses child-ready seam", errors)
    require("issue_data_ready" not in retire_helper,
            "random tuple retire releases ready early", errors)

    random_task = task(executable, "random_boundary_transaction") or ""
    require("reset_dut();" in random_task,
            "random boundary case lacks isolated reset", errors)
    require("force_boundary_core_output_request(first" in random_task,
            "random does not enter through child output seam", errors)
    require("force dut.u_frozen_m935.issue_data_ready = 1'b0;"
            in random_task and
            "force dut.u_frozen_m935.issue_data_ready = 1'b1;"
            in random_task and
            "release dut.u_frozen_m935.issue_data_ready;" in random_task,
            "random child-ready seam choreography incomplete", errors)

    for name in (
        "directed_weight_first", "directed_psum_first_and_backpressure",
        "directed_nonfirst", "directed_ii2", "reset_pending_cases",
        "sticky_fault_attacks", "service_assumption_attacks"):
        body = task(executable, name) or ""
        require("reset_dut();" in body,
                "boundary-only task lacks reset isolation: " + name, errors)

    for name in ("load_normal_task", "serve_normal_beat",
                 "normal_m935_completion"):
        require(task(text, name) == task(r11_text, name),
                "frozen R11 integrated-normal task drift: " + name, errors)

    boundary_tokens = (
        "DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
        "RANDOM")
    for phase in boundary_tokens:
        for edge in ("ENTER", "COMPLETE"):
            token = "PHASE_M1258R12_BOUNDARY_ONLY_%s_%s" % (phase, edge)
            require(text.count(token) == 1,
                    "boundary phase token missing/duplicate: " + token, errors)
    require(text.count(
        "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_TRANSACTION_ENTER") == 1 and
        text.count(
        "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_TRANSACTION_COMPLETE") == 1,
        "boundary random transaction tokens missing/duplicate", errors)
    require(text.count("PHASE_M1258R12_INTEGRATED_NORMAL_M935_ENTER") == 1
            and text.count(
                "PHASE_M1258R12_INTEGRATED_NORMAL_M935_COMPLETE") == 1,
            "integrated normal phase tokens missing/duplicate", errors)

    for marker in (
        "boundary_only=true", "integrated_random=false",
        "parent_connection_force=0", "child_core_output_seam_force=1",
        "integrated_m935_claim=false",
        "integrated_normal_m935_evidence=true",
        "zero_sva_failures_required=true", "test_index < 24",
        "cov_random_transactions != 24", "normal_m935_completion();"):
        require(marker in text, "required boundary/claim marker missing: " + marker,
                errors)
    require("integrated_random=true" not in text,
            "random boundary path inflated to integrated", errors)
    require("integrated_m935_claim=true" not in text,
            "boundary-only traffic inflated to M935 claim", errors)
    require("PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE" in text,
            "R12 source-only PASS token missing", errors)
    require("functional_vcs_only=false timing_verified=false" in text and
            "cycles_measured=false speedup=false ppa=false energy=false"
            in text and "system_speedup=false headline=false" in text,
            "claim boundary inflated", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANONICAL)
    args = parser.parse_args()
    errors = audit(args.candidate.read_text())
    payload = {
        "schema": "m1258r12_source_static_check_r1_v1",
        "status": "PASS_M1258_R12_SOURCE_ONLY" if not errors else "FAIL_CLOSED",
        "candidate": str(args.candidate),
        "candidate_sha256": sha(args.candidate),
        "checks_source_only": True,
        "vcs_invoked": False,
        "release_published": False,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
