#!/usr/bin/env python3
"""Source-only checks for the M1168R2 compile repair; no VCS or EDA."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv"
SVA = HW / "verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1166 = HW / "reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830"
R1_QUARANTINE = HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine"
R1_ATTEMPT_ID = HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed/identity.txt"

EXPECTED = {
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1166 / "review.json": "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c",
    M1166 / "SHA256SUMS": "da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363",
    M1166 / "SHA256SUMS.seal.sha256": "afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef",
    R1_QUARANTINE / "compile.log": "39765d45f5e53de02a4c9139915253b0d0d8190f042027b70344dea08b0037ff",
    R1_QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt": "a93d50e1ee3170f2e688c250fe7f75861f79176ff4ca60b407a0fb07515e185b",
    R1_QUARANTINE / "SHA256SUMS": "6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c",
    R1_QUARANTINE / "SHA256SUMS.seal.sha256": "72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7",
    R1_ATTEMPT_ID: "7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_force_staging(tb: str) -> None:
    match = re.search(r"task automatic force_request\((.*?)endtask", tb, re.S)
    require(match is not None, "force_request task absent")
    body = match.group(1)
    declarations = (
        "logic force_stage_first_q, force_stage_last_q;",
        "logic [15:0] force_stage_epoch_q;",
        "logic [5:0] force_stage_row_q;",
        "logic [3:0] force_stage_source_q;",
    )
    for token in declarations:
        require(token in tb[:match.start()], "module-scope force stage absent: " + token)
    mappings = (
        ("force_stage_first_q = first;", "force dut.issue_request_first = force_stage_first_q;"),
        ("force_stage_last_q = last;", "force dut.issue_request_last = force_stage_last_q;"),
        ("force_stage_epoch_q = epoch;", "force dut.issue_request_epoch = force_stage_epoch_q;"),
        ("force_stage_row_q = row;", "force dut.issue_request_row_id = force_stage_row_q;"),
        ("force_stage_source_q = source;", "force dut.issue_request_source_index = force_stage_source_q;"),
    )
    for assignment, forced in mappings:
        require(body.count(assignment) == 1 and body.count(forced) == 1,
                "staging mapping absent/duplicated")
        require(body.index(assignment) < body.index(forced), "force precedes stage assignment")
    for formal in ("epoch", "row", "first", "last", "source"):
        require(re.search(r"force\s+dut\.[^;]+?=\s*" + formal + r"\s*;", body) is None,
                "automatic task formal leaked to force RHS: " + formal)
    require(body.count("force dut.") == 10, "hierarchical DUT force cardinality changed")


def force_staging_mutation_test(tb: str) -> int:
    mutations = (
        ("force_stage_epoch_q", "epoch"),
        ("force_stage_row_q", "row"),
        ("force_stage_first_q", "first"),
        ("force_stage_last_q", "last"),
        ("force_stage_source_q", "source"),
    )
    rejected = 0
    for stage, formal in mutations:
        mutated = re.sub(r"(force\s+dut\.[^;]+?=\s*)" + stage + r"\s*;",
                         r"\1" + formal + ";", tb, count=1)
        require(mutated != tb, "mutation anchor absent: " + stage)
        try:
            validate_force_staging(mutated)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError("automatic-force mutation accepted: " + formal)
    return rejected


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
    compile_log = (R1_QUARANTINE / "compile.log").read_text()
    require(compile_log.count("Error-[DTINPCIL]") == 5 and
            compile_log.count("Error-[IRFPCA-AUTOVAR]") == 5 and
            "Automatic variable may not be used in non-procedural context" in compile_log,
            "r1 compile-failure diagnosis drift")
    listed = {}
    for line in (R1_QUARANTINE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.lstrip("*")] = digest
    require(listed == {
        "RUN_FAILED_OR_INCOMPLETE.txt": EXPECTED[R1_QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt"],
        "compile.log": EXPECTED[R1_QUARANTINE / "compile.log"],
    }, "r1 quarantine recursive member drift")
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
        "PASS_M1168R2_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE",
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
    require(tb.count("m1168r2_m1162_common_charge_protocol_assertions_r2 u_protocol_sva") == 1,
            "SVA instance count mismatch")
    validate_force_staging(tb)
    force_mutations_rejected = force_staging_mutation_test(tb)

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
    require(sva.count("assert property") == 16, "must preserve all 16 assertions")
    require(sva.count("cover property") == 6, "must preserve all six covers")
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
        "schema": "m1168r2_m1162_common_charge_protocol_vcs_source_static_check_r2_v1",
        "status": "PASS_SOURCE_ONLY__NO_VCS_NO_EDA__FRESH_HAMMER_REQUIRED",
        "source_sha256": {str(p.relative_to(HW)): sha(p)
                           for p in (TB, SVA, FILELIST)},
        "directed_protocol_cases": 18,
        "deterministic_random_transactions": 24,
        "normal_frozen_m935_rows": 1,
        "normal_frozen_m935_tasks": 1,
        "r1_failure_forensics": {
            "dtinpcil": 5,
            "irfpca_autovar": 5,
            "quarantine_recursively_sealed": True,
            "old_attempt_reuse": False,
        },
        "r2_force_repair": {
            "module_scope_static_stage_fields": 5,
            "hierarchical_dut_force_statements": 10,
            "automatic_force_rhs": 0,
            "automatic_rhs_mutations_rejected": force_mutations_rejected,
        },
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
