#!/usr/bin/env python3
"""Source-only closure tests for the M872/M803 three-axis DC identity.

No EDA executable, simulator, license query, remote command, or workload is
invoked.  The positive full-path runner replay is performed separately under
its explicit NO_EDA_FULL_PATH_SELF_TEST boundary.
"""

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[1]
RUNNER = HW_ROOT / "dc_handoff/scripts/run_dc_m872_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
CONTRACT = HW_ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
CANDIDATE = HW_ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
FILELIST = HW_ROOT / "dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f"
TCL = HW_ROOT / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)


def strict_load(path):
    return strict_load_bytes(path.read_bytes())


def expect_duplicate_rejected(payload, label):
    try:
        strict_load_bytes(payload)
    except ValueError as error:
        require("duplicate JSON key" in str(error), "%s rejected for wrong reason" % label)
        return
    raise RuntimeError("%s duplicate key was accepted" % label)


def validate_semantics(contract, candidate):
    required_identity = {
        "dc_actual_exec_path", "dc_actual_exec_sha256", "dc_filelist_path",
        "dc_filelist_sha256", "dc_runner_path", "dc_runner_sha256",
        "dc_shell_path", "dc_shell_sha256", "dc_tcl_path", "dc_tcl_sha256",
        "dc_wrapper_path", "dc_wrapper_sha256", "docs359_path",
        "docs359_sha256", "fast_lib_path", "fast_lib_sha256", "lmutil_path",
        "lmutil_sha256", "r5_final_failure_review_outer_seal_file_sha256",
        "r5_final_failure_review_path", "r5_quarantine_outer_seal_file_sha256",
        "r5_quarantine_path", "r5_static_review_outer_seal_file_sha256",
        "r5_static_review_path", "r5_vcs_result_outer_seal_file_sha256",
        "r5_vcs_result_path", "r5_vcs_review_outer_seal_file_sha256",
        "r5_vcs_review_path", "r6_static_review_outer_seal_file_sha256",
        "r6_static_review_path", "r7_disqualified_review_outer_seal_file_sha256",
        "r7_disqualified_review_path", "recovery_contract_path",
        "recovery_contract_sha256", "sdc_path", "sdc_sha256", "slow_lib_path",
        "slow_lib_sha256", "snps_license_file_path", "snps_license_file_sha256",
    }
    require(set(candidate["identity"]) == required_identity, "candidate identity key closure failed")
    require(candidate["launch_now"] is False, "source candidate launch_now must be false")
    require(candidate["authorization"] == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_vcs": False,
    }, "source candidate typed authorization drift")
    plan = candidate["three_axis_pre_attempt_plan"]
    require(plan == contract["three_axis_pre_attempt_plan"], "plan contract/candidate mismatch")
    require(plan["point_order"] == ["k1", "k8", "k1x8"], "axis order drift")
    require(plan["tim209_required_each_axis"] == 0, "TIM-209 gate drift")
    require(plan["opt150_required_each_axis"] == 0, "OPT-150 gate drift")
    require(plan["all_three_axes_same_attempt_required"] is True, "same-attempt gate drift")
    require(plan["partial_axis_or_cross_attempt_reuse_citable"] is False,
            "partial-axis citation gate drift")
    require(candidate["m803_vcs_authority"] == contract["m803_vcs_authority"],
            "M803 VCS authority mismatch")
    require(candidate["m800_failure_authority"] == contract["m800_failure_authority"],
            "M800 failure authority mismatch")


def main():
    runner_text = RUNNER.read_text(encoding="utf-8")
    contract = strict_load(CONTRACT)
    candidate = strict_load(CANDIDATE)
    require("PENDING_FINAL_SOURCE_CLOSE" not in runner_text, "runner placeholder remains")
    require("PENDING_FINAL_SOURCE_CLOSE" not in CONTRACT.read_text(encoding="utf-8"),
            "contract placeholder remains")
    require("PENDING_FINAL_SOURCE_CLOSE" not in CANDIDATE.read_text(encoding="utf-8"),
            "candidate placeholder remains")
    require(subprocess.call(["bash", "-n", str(RUNNER)]) == 0, "bash -n failed")

    definitions = set(re.findall(r"^(m872_m803_dc_[a-z0-9_]+)\(\)", runner_text, re.M))
    command_tokens = set(re.findall(r"\b(m872_m803_dc_[a-z0-9_]+)(?=[ \t])", runner_text))
    non_function_loop_or_counter = {
        "m872_m803_dc_artifact_test_negative_count",
        "m872_m803_dc_axis",
    }
    missing = sorted(command_tokens - definitions - non_function_loop_or_counter)
    require(not missing, "undefined M872 runner functions: %s" % missing)

    require(contract["setup_area_flow"]["runner_sha256"] == sha(RUNNER), "runner SHA drift")
    require(candidate["identity"]["dc_runner_sha256"] == sha(RUNNER), "candidate runner SHA drift")
    require(candidate["identity"]["recovery_contract_sha256"] == sha(CONTRACT),
            "candidate contract SHA drift")
    for relative, expected in contract["exact_files"].items():
        path = HW_ROOT / relative
        require(path.is_file() and not path.is_symlink(), "exact file absent/symlink: %s" % relative)
        require(sha(path) == expected, "exact file SHA drift: %s" % relative)

    expected_rtl = [
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
        "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
        "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
        "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
        "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
        "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
        "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    ]
    actual_rtl = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    require(actual_rtl == expected_rtl, "three-axis filelist membership/order drift")
    require(len(actual_rtl) == len(set(actual_rtl)), "duplicate filelist member")

    tcl = TCL.read_text(encoding="utf-8")
    order = [tcl.index(token) for token in [
        "analyze -format sverilog", "elaborate $design_name",
        'check_design > "$output_dir/reports/check_design_precompile.rpt"',
        "redirect $precompile_timing_report {check_timing}",
        "if {$precompile_tim209_count != 0 || $precompile_opt150_count != 0}",
        "\n    compile_ultra\n",
    ]]
    require(order == sorted(order) and len(set(order)) == len(order),
            "analyze/elaborate/timing-gate/compile ordering drift")
    for token in ["TIM-209=0", "OPT-150=0", "artifact_count=7",
                  "mapped_verilog mapped_sdc ddc svf area_report qor_report setup_timing_report"]:
        require(token in runner_text, "runner hard gate missing: %s" % token)
    validate_semantics(contract, candidate)

    expect_duplicate_rejected(b'{"status":"A","status":"B"}', "top-status")
    expect_duplicate_rejected(b'{"authorization":{"run_dc":true,"run_dc":false}}',
                              "authorization-run_dc")
    expect_duplicate_rejected(
        b'{"identity":{"dc_runner_sha256":"a","dc_runner_sha256":"b"}}',
        "identity-runner-sha")

    missing_identity = copy.deepcopy(candidate)
    del missing_identity["identity"]["dc_runner_sha256"]
    try:
        validate_semantics(contract, missing_identity)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("missing identity key mutation was accepted")
    bad_plan = copy.deepcopy(candidate)
    bad_plan["three_axis_pre_attempt_plan"]["tim209_required_each_axis"] = 1
    try:
        validate_semantics(contract, bad_plan)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("TIM-209 mutation was accepted")

    print("PASS_M872_M803_THREE_AXIS_DC_SOURCE_CLOSURE "
          "strict_json=3 duplicate_negatives=3 semantic_negatives=2 "
          "functions_closed=true exact_files=%d no_eda=true" % len(contract["exact_files"]))


if __name__ == "__main__":
    main()
