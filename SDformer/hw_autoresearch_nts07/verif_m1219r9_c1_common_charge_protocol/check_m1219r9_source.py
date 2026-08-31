#!/usr/bin/env python3
"""Fail-closed static audit for the M1219/R9 source-only TB revision.

This checker performs no compilation, simulation, VCS, or EDA action.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


EXPECTED = {
    "r9_tb": "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    "r8_tb": "060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b",
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
    errors = []  # type: List[str]
    require("module tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9;" in text,
            "wrong R9 module identity", errors)
    require("module tb_m1210r8_" not in text, "R8 module name leaked", errors)
    require(not re.search(r"\bwait\s*\(", text),
            "unbounded wait statement remains", errors)
    while_headers = re.findall(r"while\s*\((.*?)\)\s*begin", text, re.S)
    require(len(while_headers) == 8,
            f"expected 8 bounded while loops, found {len(while_headers)}", errors)
    for number, header in enumerate(while_headers):
        require("watchdog <" in header,
                f"while loop {number} lacks watchdog bound", errors)

    for marker in (
        '"random_weight_request"', '"random_psum_request"',
        '"random_response_accept"', '"normal_prep_ready"',
        '"clean_reset_prep_ready"', "TIMEOUT_M1219R9",
        "R9_RANDOM_WAIT_LIMIT", "R9_PREP_WAIT_LIMIT",
        "R9_CLEAN_RESET_PREP_LIMIT",
    ):
        require(marker in text, f"missing liveness marker {marker}", errors)

    phase_pairs = (
        "DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
        "RANDOM", "NORMAL_M935", "CLEAN_RESET_PREP",
    )
    for phase in phase_pairs:
        require(f"PHASE_M1219R9_{phase}_ENTER" in text,
                f"missing {phase} enter token", errors)
        require(f"PHASE_M1219R9_{phase}_COMPLETE" in text,
                f"missing {phase} complete token", errors)
    require("PHASE_M1219R9_RANDOM_TRANSACTION_ENTER index=%0d" in text,
            "missing per-random enter token", errors)
    require("PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE index=%0d" in text,
            "missing per-random complete token", errors)
    require(text.count("$fflush();") >= 16, "phase tokens are not flushed", errors)

    # Frozen functional workload and R8 quiesce semantics.
    require("test_index < 24" in text, "24 random transactions changed", errors)
    require("cov_random_transactions != 24" in text,
            "random coverage minimum changed", errors)
    require("cov_normal_issue != 2 || cov_normal_row != 1" in text,
            "normal M935 issue/row minima changed", errors)
    require("|| cov_normal_task != 1 || cov_legal_masks_clear != 29" in text,
            "normal task/legal minima changed", errors)
    require("cov_request_attack_windows != 2" in text,
            "protocol attack minimum changed", errors)
    require("cov_weight_service_attack_windows != 1" in text and
            "cov_psum_service_attack_windows != 1" in text,
            "service attack minima changed", errors)
    require("directed_ii2();" in text and "cov_ii2 != 1" in text,
            "II=2 test changed", errors)
    q_anchor = "R8_RANDOM_REQUEST_READY_QUIESCE_BOUNDARY"
    q_pos = text.find(q_anchor)
    q_window = text[q_pos:q_pos + 900] if q_pos >= 0 else ""
    require(q_pos >= 0 and "weight_req_ready = 1'b0;" in q_window and
            "psum_req_ready = 1'b0;" in q_window and
            "random_request_window_active = 1'b0;" in q_window,
            "R8 ready-quiesce semantics changed", errors)
    require("require_clean_reset_prep_ready();" in text,
            "clean reset prep gate not called", errors)
    require("PASS_M1219R9_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE" in text,
            "R9 fail-closed PASS token missing", errors)
    require("bounded_waits=4 clean_reset_prep_bounded=1 phase_observability=1" in text,
            "R9 PASS qualifiers missing", errors)
    require("functional_vcs_only=true timing_verified=false cycles_measured=false" in text,
            "claim boundary changed", errors)
    if enforce_identity:
        require(actual_sha == EXPECTED["r9_tb"],
                f"R9 SHA mismatch {actual_sha}", errors)
    return errors


def canonical_paths(root: Path) -> Dict[str, Path]:
    return {
        "r9_tb": root / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv",
        "r8_tb": root / "verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv",
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
    errors = []  # type: List[str]
    hashes = {}  # type: Dict[str, str]
    for key, path in paths.items():
        if not path.is_file():
            errors.append(f"missing canonical file {key}: {path}")
            continue
        hashes[key] = sha256(path)
        if hashes[key] != EXPECTED[key]:
            errors.append(f"{key} SHA mismatch {hashes[key]}")
    if "r9_tb" in hashes:
        errors.extend(audit_text(paths["r9_tb"].read_text(), True,
                                 hashes["r9_tb"]))
    result = {
        "milestone": "M1219",
        "status": "PASS" if not errors else "FAIL",
        "source_only": True,
        "vcs_invoked": False,
        "eda_invoked": False,
        "fresh_hammer_required": True,
        "hashes": hashes,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
