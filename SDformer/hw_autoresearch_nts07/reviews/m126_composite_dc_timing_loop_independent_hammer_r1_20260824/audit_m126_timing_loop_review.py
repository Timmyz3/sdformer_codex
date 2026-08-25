#!/usr/bin/env python3
import hashlib
import json
import pathlib
import re


REVIEW = pathlib.Path(__file__).resolve().parent
HW = REVIEW.parent.parent
SD = HW.parent
FAILED = HW / "dc_handoff/runs/m126_logic_only_dc_3p000ns_exploratory_r1_20260824"
SEALED = HW / "dc_handoff/runs/m126_block_phased_k4_forwarding_accumulator_vcs_r1_sealed_20260824"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL " + message)


frozen = {
    "m125_rtl": (HW / "rtl_m125/m125_block_phased_k4_row_fold.sv",
                 "cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"),
    "m123_core_rtl": (HW / "rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv",
                      "7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"),
    "m123_adapter_rtl": (HW / "rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv",
                         "a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"),
    "m126_rtl": (HW / "rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv",
                 "b75c64cfa0803461bef4690025a723df9e039e8d2eef6a0da918fc3b9c063e01"),
    "docs_359": (HW / "docs/359_DATE终局冻结_20260813.md",
                 "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}
for name, (path, expected) in frozen.items():
    require(path.is_file(), name + " missing")
    require(sha256(path) == expected, name + " SHA drift")

failed_expected = {
    "dc.log": "e0f160b2a6b9ecb12f8bb8ab2e039bb44b188598f4aff63885623d66fa15a5c2",
    "RUN_FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt": "ee31e49d271499035ab83248da5bbb9d0975e744811a82a5a3a7cb4f72cf192c",
    "reports/check_timing_precompile.rpt": "a677ed25d86c253731d1e7293c812218c68d14ea38d7744cb095ad001713af7e",
    "reports/check_design_precompile.rpt": "1762206deec9f3b7d469f64d2d81af1e179cf0d8b10ad32af0236343bc6a7f12",
    "reports/resources_precompile.rpt": "91485aac95123e3d52884b42fe622a4d3a96c244303d3840df1a73ba80393917",
}
for relative, expected in failed_expected.items():
    path = FAILED / relative
    require(path.is_file() and sha256(path) == expected,
            "failed DC evidence drift: " + relative)
require(not (FAILED / "netlist/m126_block_phased_k4_forwarding_accumulator_island_mapped.v").exists(),
        "failed run unexpectedly gained mapped Verilog")
require(not (FAILED / "netlist/m126_block_phased_k4_forwarding_accumulator_island.ddc").exists(),
        "failed run unexpectedly gained DDC")
for absent_report in ("area.rpt", "timing_setup.rpt", "timing_hold.rpt",
                      "qor.rpt", "check_timing_postcompile.rpt"):
    require(not (FAILED / "reports" / absent_report).exists(),
            "failed run unexpectedly gained " + absent_report)

marker = text(FAILED / "RUN_FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt")
failed_log = text(FAILED / "dc.log")
failed_timing = text(FAILED / "reports/check_timing_precompile.rpt")
require("status=FAILED_DO_NOT_CITE" in marker, "failure marker not fail-closed")
require("Warning: timing loops detected. (TIM-209)" in failed_timing,
        "production precompile TIM-209 absent")
require("Timing loop detected. (OPT-150)" in failed_log,
        "production mapped OPT-150 absent")
require("Error id=263749" in failed_log and
        "Fatal: Internal system error, cannot recover." in failed_log,
        "production fatal signature absent")

original_timing = text(REVIEW / "original_dc/check_timing.rpt")
delta_timing = text(REVIEW / "delta_dc/check_timing.rpt")
delta_dc_log = text(REVIEW / "delta_dc/dc.raw.log")
require(text(REVIEW / "original_dc/dc.rc").strip() == "0",
        "original check_timing dc rc")
require(text(REVIEW / "delta_dc/dc.rc").strip() == "0",
        "delta check_timing dc rc")
require("Warning: timing loops detected. (TIM-209)" in original_timing,
        "independent original TIM-209 reproduction absent")
require("Warning: timing loops detected. (TIM-209)" not in delta_timing,
        "review delta still has TIM-209")
require("Timing loop detected. (OPT-150)" not in delta_dc_log,
        "review delta has OPT-150")

fold_delta = text(REVIEW / "m125_registered_state_busy_delta.sv")
top_delta = text(REVIEW / "m126_registered_fault_barrier_delta.sv")
require("assign busy = fill_active_q || row_active_q;" in fold_delta,
        "M125 registered-state busy delta absent")
require("assign busy = fill_active_q || row_active_q || update_valid;" not in fold_delta,
        "M125 redundant busy term remains")
require("else if (wrapper_illegal_request || fold_protocol_error" in top_delta,
        "registered fault capture absent")

truth_table = []
for fill_active in (0, 1):
    for row_active in (0, 1):
        for selected_nonzero in (0, 1):
            for protocol_error in (0, 1):
                update_valid = row_active and selected_nonzero and not protocol_error
                before = bool(fill_active or row_active or update_valid)
                after = bool(fill_active or row_active)
                truth_table.append({
                    "fill_active": fill_active,
                    "row_active": row_active,
                    "selected_nonzero": selected_nonzero,
                    "protocol_error": protocol_error,
                    "update_valid": int(update_valid),
                    "before_busy": int(before),
                    "after_busy": int(after),
                    "equal": before == after,
                })
require(all(row["equal"] for row in truth_table),
        "M125 busy Boolean equivalence failed")

require(text(REVIEW / "delta_vcs/compile.rc").strip() == "0",
        "delta VCS compile rc")
require(text(REVIEW / "delta_vcs/sim.rc").strip() == "0",
        "delta VCS sim rc")
delta_sim = text(REVIEW / "delta_vcs/sim.raw.log")
delta_assert = text(REVIEW / "delta_vcs/assert.report")
pass_match = re.search(r"^PASS M126 K4 fold plus forwarding accumulator VCS .*$",
                       delta_sim, re.MULTILINE)
require(pass_match is not None, "delta VCS PASS line absent")
for name, expected in {
    "cp_four_consecutive_same_row_folds": 160,
    "cp_full_k4_to_write": 5115,
    "cp_tail_to_write": 2211,
    "cp_commit_stall_release": 384,
    "cp_reset_with_prior_update": 1,
}.items():
    match = re.search(r"%s, \d+ attempts, (\d+) match" % re.escape(name),
                      delta_assert)
    require(match is not None and int(match.group(1)) == expected,
            "delta VCS cover mismatch: " + name)

sealed_pass = re.search(
    r"^PASS M126 K4 fold plus forwarding accumulator VCS .*$",
    text(SEALED / "sim.raw.log"), re.MULTILINE)
require(sealed_pass is not None, "sealed production VCS PASS absent")
require(pass_match.group(0) == sealed_pass.group(0),
        "delta and sealed production PASS lines differ")

failed_breakpoints = len(re.findall(r"#\s*$", failed_timing, re.MULTILINE))
original_breakpoints = len(re.findall(r"#\s*$", original_timing, re.MULTILINE))
delta_breakpoints = len(re.findall(r"#\s*$", delta_timing, re.MULTILINE))

audit = {
    "schema": "m126_composite_dc_timing_loop_independent_hammer_v1",
    "status": "FAIL_PRODUCTION_PHYSICAL_ADMISSION_REVIEW_DELTA_ONLY",
    "date": "2026-08-24",
    "score": {
        "scope": "M126 physical/DC admission only; does not replace the separate 92/100 directed functional review",
        "total": 42,
        "out_of": 100,
        "p0": 1,
        "p1": 2,
        "p2": 2,
    },
    "frozen_identity": {name + "_sha256": expected
                        for name, (_, expected) in frozen.items()},
    "failed_dc": {
        "citable": False,
        "paper_ppa_ready": False,
        "physical_speedup": False,
        "system_speedup": False,
        "backend_complete": False,
        "precompile_tim209": True,
        "precompile_loop_breakpoints": failed_breakpoints,
        "mapped_opt150_occurrences": failed_log.count("Timing loop detected. (OPT-150)"),
        "dc_auto_disabled_arcs": failed_log.count("to break a timing loop. (OPT-314)"),
        "hold_only_timing_update_failed_for_loops":
            "Timing update failed because design has loops." in failed_log,
        "fatal_error_id": 263749,
        "mapped_netlist_present": False,
        "mapped_area_report_present": False,
        "mapped_setup_hold_reports_present": False,
        "power_report_present": False,
        "evidence_sha256": failed_expected,
    },
    "independent_reproduction": {
        "tool": "Synopsys Design Compiler V-2023.12-SP3",
        "stage": "analyze/elaborate/link/uniquify/SDC/check_timing; no compile",
        "production_exact_sha_tim209": True,
        "production_loop_breakpoints": original_breakpoints,
        "review_delta_tim209": False,
        "review_delta_loop_breakpoints": delta_breakpoints,
        "tool_only_anomaly": False,
    },
    "static_cones": [
        {
            "name": "cross_child_error_valid_error_cycle",
            "path": [
                "M125 fold_protocol_error",
                "M126 raw error gate on accumulator update/start/end valid",
                "M123 illegal_request/protocol_error",
                "M126 raw accumulator error gate on fold row/fill valid or update ready",
                "M125 illegal_request/protocol_error",
            ],
            "cut": "capture child errors into sticky wrapper_fault_q; do not feed raw child errors into sibling valid/ready",
        },
        {
            "name": "fold_busy_wrapper_audit_cycle",
            "path": [
                "M125 protocol_error",
                "M125 update_valid",
                "M125 busy redundant update_valid term",
                "M126 wrapper_illegal_request fold_busy audit",
                "M126 fold row/fill valid gate",
                "M125 illegal_request/protocol_error",
            ],
            "cut": "Boolean-equivalent M125 busy = fill_active_q || row_active_q; update_valid implies row_active_q",
        },
    ],
    "review_only_delta": {
        "production_modified": False,
        "admitted": False,
        "m125_busy_truth_table_cases": len(truth_table),
        "m125_busy_truth_table_mismatches": sum(not row["equal"] for row in truth_table),
        "precompile_loop_free": True,
        "production_directed_vcs_pass_line_exact_match": True,
        "fault_contract_reverified": False,
        "compile_ultra_complete": False,
        "formality_complete": False,
        "pt_sta_complete": False,
        "ptpx_complete": False,
        "paper_ppa_ready": False,
    },
    "functional_physical_boundary": {
        "sealed_production_directed_vcs_remains_valid": True,
        "functional_vcs_proves_legal_directed_data_conservation": True,
        "functional_vcs_does_not_prove_combinational_acyclicity": True,
        "functional_vcs_does_not_prove_synthesizability_or_ppa": True,
        "production_m126_physical_admission": False,
        "citable_area_um2": None,
        "citable_fmax_mhz": None,
        "citable_power_mw": None,
        "citable_energy": None,
        "citable_physical_speedup": None,
    },
    "priorities": {
        "p0": [
            "Production M126 contains real combinational timing loops; physical admission is blocked and the failed DC run is non-citable.",
        ],
        "p1": [
            "No mapped netlist, clean STA, area or power was produced; do not recover any intermediate number from the failed run.",
            "The review-only delta needs dedicated child-fault/illegal-request VCS plus production review before any merge or admission.",
        ],
        "p2": [
            "After RTL repair, rerun compile_ultra, Formality, PT STA and PTPX; precompile check_timing is necessary but not sufficient.",
            "The SP3 error-263749 crash is downstream of the RTL loop; file a tool issue only after supplying a loop-free reproducer.",
        ],
    },
    "safe_statement": (
        "Commercial VCS functional evidence for the frozen M126 legal directed scope remains valid, "
        "but independent precompile DC reproduces real combinational timing loops in production M126. "
        "The 3 ns exploratory DC run is failed and supplies no citable PPA or physical speedup. "
        "A review-only registered fault barrier plus a Boolean-equivalent M125 busy simplification removes "
        "TIM-209 and preserves the existing directed VCS PASS line; it is not production or physical admission."
    ),
}

out = REVIEW / "m126_composite_dc_timing_loop_independent_audit.json"
out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
               encoding="utf-8")
(REVIEW / "m125_busy_boolean_equivalence_exhaustive.json").write_text(
    json.dumps({
        "schema": "m125_busy_boolean_equivalence_exhaustive_v1",
        "status": "PASS",
        "identity": "fill || row || (row && selected && !error) == fill || row",
        "cases": truth_table,
        "mismatches": 0,
    }, indent=2) + "\n", encoding="utf-8")
print("PASS M126 timing-loop independent audit " + sha256(out))
