#!/usr/bin/env python3
"""Fail-closed source audit for the additive M1226/R10 TB repair.

This checker reads text and hashes only.  It never compiles or launches VCS,
simv, EDA, GPU, or remote work.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


EXPECTED = {
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "r9_tb": "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
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


def audit_text(text: str, enforce_identity: bool = False,
               actual_sha: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    require(
        "module tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10;"
        in text,
        "wrong R10 module identity", errors)
    require("module tb_m1219r9_" not in text,
            "R9 module identity reused", errors)
    require(not re.search(r"\bwait\s*\(", text),
            "unbounded wait statement remains", errors)
    loops = re.findall(r"while\s*\((.*?)\)\s*begin", text, re.S)
    require(len(loops) == 9,
            f"expected 9 bounded while loops, found {len(loops)}", errors)
    for number, header in enumerate(loops):
        require("watchdog <" in header,
                f"while loop {number} lacks watchdog bound", errors)

    # Exact normal-service source gates.  These anchors deliberately bind both
    # action and ordering so a cosmetic comment cannot satisfy the audit.
    request_retire = (
        "@(negedge clk_core);\n"
        "            weight_req_ready = 1'b0;\n"
        "            psum_req_ready = 1'b0;\n"
        "            #1ps;\n"
        "            if (weight_req_ready || psum_req_ready")
    require(request_retire in text,
            "race-free request-ready retirement missing", errors)
    normal_onefire = (
        "if (weight_fire_count > w0 + 1\n"
        "                        || psum_fire_count > p0 + expect_first) begin\n"
        "                    dump_r9_liveness_state(\"normal_request_overshoot\"")
    require(normal_onefire in text,
            "normal one-fire overshoot proof missing", errors)
    require("normal_request_accept" in text and
            "normal_request_retire" in text,
            "normal request timeout/retire dump missing", errors)

    stable_gate = (
        "if (!weight_rsp_valid || weight_data !== '0\n"
        "                        || psum_rsp_valid !== expect_first\n"
        "                        || psum_data !== '0)")
    require(stable_gate in text,
            "response valid/payload stability proof missing", errors)
    exact_retire = (
        "// Exact-accept retirement: no extra posedge is permitted here.\n"
        "            @(negedge clk_core);\n"
        "            weight_rsp_valid = 1'b0;\n"
        "            psum_rsp_valid = 1'b0;")
    require(exact_retire in text,
            "exact-accept immediate-negedge response retirement missing",
            errors)
    require("normal_response_accept" in text and
            "normal_response_unstable" in text and
            "normal_response_retire" in text,
            "normal response timeout/stability/retire dump missing", errors)

    require("normal_retired_beats != beat_index" in text and
            "response_accept_count\n"
            "                        != normal_response_base + beat_index"
            in text and
            "if (beat_index > 0 && dut.request_active_q\n"
            "                    && (!issue_request_valid\n"
            "                        || dut.weight_request_accepted_q\n"
            "                        || dut.psum_request_accepted_q" in text and
            "dut.request_source_index_q\n"
            "                            != issue_request_source_index" in text,
            "beat-two count/valid/retired admission gate missing", errors)
    require("normal_wrapper_retire" in text and
            "dut.request_source_index_q == served_source" in text,
            "post-response wrapper retirement dump missing", errors)
    require('dump_r9_liveness_state("normal_issue_request"' in text and
            'dump_r9_liveness_state("normal_response_accept"' in text and
            'dump_r9_liveness_state("normal_task_completion"' in text,
            "normal watchdog state dump missing", errors)

    # Frozen workload and claim boundary.
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
        require(marker in text, f"frozen workload marker missing: {marker}",
                errors)
    require(
        "PASS_M1226R10_M1162_COMMON_CHARGE_PROTOCOL_SOURCE_CANDIDATE" in text,
        "R10 source-only PASS token missing", errors)
    require("zero_sva_failures_required=true" in text,
            "zero-SVA-failure execution gate missing", errors)
    require("functional_vcs_only=false timing_verified=false" in text and
            "system_speedup=false headline=false" in text,
            "source-only claim boundary changed", errors)

    if enforce_identity:
        require(actual_sha == EXPECTED["r10_tb"],
                f"R10 SHA mismatch {actual_sha}", errors)
    return errors


def canonical_paths(root: Path) -> Dict[str, Path]:
    return {
        "r10_tb": root / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
        "r9_tb": root / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv",
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
            errors.append(f"missing canonical file {key}: {path}")
            continue
        hashes[key] = sha256(path)
        if hashes[key] != EXPECTED[key]:
            errors.append(f"{key} SHA mismatch {hashes[key]}")
    if "r10_tb" in hashes:
        errors.extend(audit_text(paths["r10_tb"].read_text(), True,
                                 hashes["r10_tb"]))
    result = {
        "milestone": "M1226",
        "status": "PASS" if not errors else "FAIL",
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
