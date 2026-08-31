#!/usr/bin/env python3
"""Source-only checks for M1168.  This script never invokes VCS or EDA."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv"
SVA = HW / "verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1168_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1166 = HW / "reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830"

EXPECTED = {
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1166 / "review.json": "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c",
    M1166 / "SHA256SUMS": "da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363",
    M1166 / "SHA256SUMS.seal.sha256": "afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular {path}")
        require(sha(path) == digest, f"frozen SHA mismatch {path}")

    review = json.loads((M1166 / "review.json").read_text())
    require(review["status"] ==
            "PASS_M1166_M1162_PROTOCOL_REPAIR_SOURCE_HAMMER__AUTHORIZE_ONE_ADDITIVE_VCS_SOURCE_LAUNCH_PACKAGE__NO_VCS_NO_EDA",
            "M1166 status mismatch")
    require(review["issue_counts"] == {"P0": 0, "P1": 1, "P2": 0},
            "M1166 issue boundary changed")
    require(review["authorization"]["one_additive_vcs_tb_sva_filelist_launcher_source_package_next"] is True,
            "M1166 did not authorize this source package")
    require(review["authorization"]["fresh_hammer_of_vcs_package_before_run"] is True,
            "fresh hammer requirement absent")

    tb = TB.read_text()
    sva = SVA.read_text()
    flines = [line.strip() for line in FILELIST.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    require(len(flines) == 6 and len(set(flines)) == 6,
            "filelist must contain six unique exact sources")
    for item in flines:
        require(Path(item).is_file(), f"filelist source missing {item}")

    required_tb_tokens = [
        "directed_weight_first", "directed_psum_first_and_backpressure",
        "directed_nonfirst", "directed_ii2", "reset_pending_cases",
        "sticky_fault_attacks", "service_assumption_attacks",
        "random_legal_transaction", "normal_m935_completion",
        "cov_reset_partial", "cov_reset_complete", "cov_reset_skew",
        "cov_unsolicited_weight", "cov_unsolicited_psum",
        "cov_duplicate_response", "cov_weight_payload_mutation",
        "cov_psum_valid_drop", "cov_no_duplicate_request",
        "PASS_M1168_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE",
        "functional_vcs_only=true", "timing_verified=false",
        "cycles_measured=false", "speedup=false", "ppa=false",
        "energy=false", "system_speedup=false", "headline=false",
    ]
    for token in required_tb_tokens:
        require(token in tb, f"TB token absent: {token}")
    require(re.search(r"for \(integer test_index = 0; test_index < 24;", tb),
            "24-transaction deterministic random suite absent")
    require(tb.count("m1162_m935_c1_common_charge_protocol_boundary dut") == 1,
            "wrapper instance count mismatch")
    require(tb.count("m1168_m1162_common_charge_protocol_assertions_r1 u_protocol_sva") == 1,
            "SVA instance count mismatch")

    required_sva = [
        "ap_weight_request_hold", "ap_psum_request_hold",
        "ap_weight_no_reissue", "ap_psum_no_reissue",
        "ap_nonfirst_never_requests_psum",
        "ap_core_valid_requires_requests", "ap_weight_ready_is_atomic",
        "ap_psum_ready_is_first_atomic", "ap_no_lone_weight_consume",
        "ap_no_lone_psum_consume", "ap_core_backpressure_atomic",
        "ap_weight_response_hold", "ap_psum_response_hold",
        "ap_boundary_fault_sticky", "ap_reset_clears_transaction",
        "ap_no_consecutive_response_accept", "cp_ii2",
    ]
    for token in required_sva:
        require(token in sva, f"SVA property absent: {token}")
    require("service_attack_mode" in sva and "independent TB" in sva,
            "service assumption attack boundary not explicit")

    # Exact frozen hierarchy and no accidental replacement of the compute RTL.
    wrapper = WRAPPER.read_text()
    require(wrapper.count(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935") == 1,
        "frozen M935 hierarchy changed")
    require("response_accept_w = core_issue_data_valid && core_issue_data_ready" in wrapper,
            "M1162 response acceptance changed")
    require("request_tuple_mutated_w" in wrapper and "boundary_fault_q" in wrapper,
            "M1162 sticky checks absent")

    result = {
        "schema": "m1168_m1162_common_charge_protocol_vcs_source_static_check_v1",
        "status": "PASS_SOURCE_ONLY__NO_VCS_NO_EDA__FRESH_HAMMER_REQUIRED",
        "source_sha256": {str(p.relative_to(HW)): sha(p)
                           for p in (TB, SVA, FILELIST)},
        "directed_protocol_cases": 18,
        "deterministic_random_transactions": 24,
        "normal_frozen_m935_rows": 1,
        "normal_frozen_m935_tasks": 1,
        "claim_boundary": {
            "vcs_run": False, "functional_vcs_verified": False,
            "timing_verified": False, "cycles_measured": False,
            "speedup": False, "ppa": False, "energy": False,
            "system_speedup": False, "paper_citable": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
