#!/usr/bin/env python3
"""Safe source/JSON mutation checks for the M704 M519 R10 release.

No runner or EDA tool is executed.  Mutations exist only in memory and prove
that the static identity/admission predicates fail closed.
"""

import copy
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_dc_m519_r10_setup_area_three_axis_exact_sha_r3.sh"
CONTRACT = HW / "contracts/m519_r10_setup_area_three_axis_recovery_contract_r3_20260828.json"
ADMISSION = HW / "contracts/m519_r10_setup_area_three_axis_dc_launch_admission_r3_20260828.json"
M694 = HW / "reviews/m694_m519_r9_three_axis_dc_release_fresh_hammer_r1_20260828/review.json"
M701 = HW / "reviews/m701_m519_r9_pre_eda_shell_failure_receipt_r1_20260828/review.json"
RUNS = HW / "dc_handoff/runs"

EXPECTED_RUNNER_SHA = "7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f"
EXPECTED_CONTRACT_SHA = "2ba563ed4c3ddb2c89d0a13855bb4b11be7522aef505cfe1ef374a33b5501a4e"
EXPECTED_ADMISSION_SHA = "f4bccc501dea216396d2755ef6b1f627209efe18346701cd5d448367cf4a3424"
CANONICAL = RUNS / "m519_r10_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r3_20260828"
ATTEMPT = RUNS / ".m519_r10_channel_local_fault_dc_attempt_consumed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_keys(value, expected):
    return isinstance(value, dict) and set(value) == set(expected)


def main():
    text = RUNNER.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    m694 = json.loads(M694.read_text(encoding="utf-8"))
    m701 = json.loads(M701.read_text(encoding="utf-8"))
    checks = {}

    checks["exact_sha_identity"] = (
        sha(RUNNER) == EXPECTED_RUNNER_SHA and
        sha(CONTRACT) == EXPECTED_CONTRACT_SHA and
        sha(ADMISSION) == EXPECTED_ADMISSION_SHA)

    # Both historical set -u same-command dependencies must remain absent.
    payload_bad = 'local payload=$1 sidecar=${payload}.sha256'
    id_bad = 'local id=$1 mode=$2 point="${m519_r10_work}/${id}"'
    checks["two_set_u_regressions_absent"] = (
        payload_bad not in text and id_bad not in text and
        "local payload\n    local sidecar" in text and
        "local id\n    local mode\n    local point" in text)
    mutated_source = text.replace(
        "local payload\n    local sidecar", payload_bad + "\n    # injected")
    checks["payload_sidecar_mutation_rejected"] = payload_bad in mutated_source
    mutated_source = text.replace(
        "local id\n    local mode\n    local point", id_bad + "\n    # injected")
    checks["id_point_mutation_rejected"] = id_bad in mutated_source

    # The first occurrence of each unique boundary marker proves the static
    # order.  The dc-shell launch marker includes output redirection and '&'.
    positions = {
        "bash_n": text.index('bash -n "${m519_r10_runner}"'),
        "selftest": text.index('if [[ -n "${M519_R10_NO_EDA_SELF_TEST:-}" ]]'),
        "admission": text.index('m519_r10_expect "${m519_r10_admission}"'),
        "preflight": text.index('if ! m519_r10_axis_preflight k1'),
        "attempt": text.index('mv -T "${m519_r10_work}/.attempt_staging" "${m519_r10_attempt}"'),
        "dc_launch": text.index('"${m519_r10_dc}" -f "${m519_r10_hw_root}/${m519_r10_tcl}"'),
    }
    checks["boundary_order"] = (
        positions["bash_n"] < positions["selftest"] < positions["admission"] <
        positions["preflight"] < positions["attempt"] < positions["dc_launch"])

    expected_authorization = {
        "max_attempts", "run_dc", "run_formality", "run_pt", "run_ptpx",
        "run_remote", "run_vcs",
    }
    checks["authorization_keys_closed"] = exact_keys(
        admission["authorization"], expected_authorization)
    mutated = copy.deepcopy(admission)
    mutated["authorization"]["run_gpu"] = False
    checks["unknown_authorization_key_rejected"] = not exact_keys(
        mutated["authorization"], expected_authorization)
    mutated = copy.deepcopy(admission)
    mutated["authorization"]["run_vcs"] = True
    checks["scope_expansion_rejected"] = not (
        mutated["authorization"]["run_dc"] is True and
        mutated["authorization"]["max_attempts"] == 1 and
        all(mutated["authorization"][key] is False for key in
            ("run_vcs", "run_formality", "run_pt", "run_ptpx", "run_remote")))

    expected_provenance = {
        "m694_manifest_file_sha256", "m694_outer_seal_file_sha256",
        "m694_review_path", "m694_review_sha256", "m694_status",
        "m701_manifest_file_sha256", "m701_no_eda_started",
        "m701_outer_seal_file_sha256", "m701_review_path",
        "m701_review_sha256", "m701_status", "r10_is_additive",
        "r9_attempt_remains_absent", "r9_result_remains_absent",
    }
    checks["provenance_keys_closed"] = exact_keys(
        admission["r10_repair_provenance"], expected_provenance)
    mutated = copy.deepcopy(admission)
    mutated["r10_repair_provenance"]["m701_status"] += "_MUTATED"
    checks["m701_status_mutation_rejected"] = (
        mutated["r10_repair_provenance"] != contract["r10_repair_provenance"] or
        mutated["r10_repair_provenance"]["m701_status"] != m701["status"])
    mutated = copy.deepcopy(admission)
    mutated["r10_repair_provenance"]["m694_review_sha256"] = "0" * 64
    checks["m694_sha_mutation_rejected"] = (
        mutated["r10_repair_provenance"] != contract["r10_repair_provenance"] or
        mutated["r10_repair_provenance"]["m694_review_sha256"] != sha(M694))

    checks["sealed_statuses_exact"] = (
        admission["r10_repair_provenance"]["m694_status"] == m694["status"] ==
        "GO_ONE_M519_R9_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED" and
        admission["r10_repair_provenance"]["m701_status"] == m701["status"] ==
        "PRE_EDA_SHELL_FAILURE__NO_DC_STARTED__M519_R9_NOT_CITABLE__ADDITIVE_R10_REQUIRED")
    checks["contract_admission_provenance_equal"] = (
        admission["r10_repair_provenance"] == contract["r10_repair_provenance"])

    expected_paths = {
        "dc_handoff/scripts/run_dc_m519_r10_setup_area_three_axis_exact_sha_r3.sh",
        "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl",
        "dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f",
        "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc",
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
        "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
        "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
        "rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv",
        "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
        "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
        "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
        "rtl_m519/m519_fc2_registered_release_matched_8bank_raw4_acc24.sv",
        "docs/359_DATE终局冻结_20260813.md",
    }
    checks["exact_file_path_set_closed"] = set(contract["exact_files"]) == expected_paths
    mutated = copy.deepcopy(contract)
    mutated["exact_files"]["rtl_m519/unregistered.sv"] = "0" * 64
    checks["unknown_exact_file_rejected"] = set(mutated["exact_files"]) != expected_paths
    mutated = copy.deepcopy(admission)
    mutated["identity"]["dc_runner_sha256"] = "0" * 64
    checks["runner_identity_mutation_rejected"] = (
        mutated["identity"]["dc_runner_sha256"] !=
        contract["setup_area_flow"]["runner_sha256"])

    checks["resource_and_collision_gate_not_weakened"] = all(token in text for token in (
        "m519_r10_preflight_commit_kib=67108864",
        "m519_r10_runtime_commit_kib=33554432",
        "m519_r10_mem_available_kib=134217728",
        "m519_r10_swap_free_kib=33554432",
        '"${M519_R10_PROC_UID}" == "${m519_r10_uid}"',
        "new_external_same_uid_eda_collision",
        "PASS_FINAL_GATE_ACK",
    ))
    checks["canonical_absent"] = not CANONICAL.exists()
    checks["attempt_absent"] = not ATTEMPT.exists()
    checks["all_pass"] = all(checks.values())

    result = {
        "schema": "m708.m704.m519_r10.safe_static_mutation_review.v1",
        "checks": checks,
        "boundary_byte_offsets": positions,
        "eda_launched": False,
        "runner_executed": False,
        "live_process_state_used_as_launch_evidence": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
