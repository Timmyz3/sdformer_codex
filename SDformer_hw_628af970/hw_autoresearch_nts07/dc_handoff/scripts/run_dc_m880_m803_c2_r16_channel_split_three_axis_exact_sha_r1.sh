#!/usr/bin/env bash
set -euo pipefail

m872_m803_dc_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m872_m803_dc_hw_root="$(cd "${m872_m803_dc_dc_root}/.." && pwd)"
m872_m803_dc_runner="$(realpath "${BASH_SOURCE[0]}")"
m872_m803_dc_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m872_m803_dc_dc_wrapper=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m872_m803_dc_dc_actual_exe=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m872_m803_dc_dc_install_root=/opt/synopsys/syn/V-2023.12-SP3
m872_m803_dc_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m872_m803_dc_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m872_m803_dc_filelist=dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f
m872_m803_dc_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m872_m803_dc_tcl=dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl
m872_m803_dc_contract=contracts/m880_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json
m872_m803_dc_final_admission=contracts/m882_m880_m803_c2_r16_channel_split_three_axis_dc_launch_admission_r1_20260829.json
m872_m803_dc_candidate=contracts/m880_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json
m872_m803_dc_admission=${m872_m803_dc_final_admission}
m872_m803_dc_expected_admission_status=AUTHORIZED_ONE_M880_M803_C2_R16_CHANNEL_SPLIT_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT_R1
m872_m803_dc_expected_launch_now=true
m872_m803_dc_snpslmd_license_file=27030@ic.ismd-nemo
m872_m803_dc_lm_license_file=/opt/synopsys/Synopsys.dat
m872_m803_dc_license_file=/opt/synopsys/Synopsys.dat
m872_m803_dc_license_file_sha256=fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490
m872_m803_dc_lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
m872_m803_dc_lmutil_sha256=e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07
m872_m803_dc_r5_static=reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827
m872_m803_dc_r5_vcs=results/m519_r5_channel_local_fault_vcs_r1_20260827
m872_m803_dc_r5_vcs_review=reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827
m872_m803_dc_r5_failure=reviews/m519_r5_final_failure_receipt_hammer_r1_20260827
m872_m803_dc_r5_quarantine=dc_handoff/runs/m519_r5_channel_local_fault_three_axis_logic_only_dc_3p000ns_r1_20260827.failed_or_incomplete.4165439.quarantine
m872_m803_dc_r6_failed_review=reviews/m538_m519_r6_setup_area_flow_static_hammer_r1_20260827
m872_m803_dc_r7_disqualified_review=reviews/m540_m519_r7_setup_area_flow_static_hammer_r1_20260827
m872_m803_dc_m694=reviews/m694_m519_r9_three_axis_dc_release_fresh_hammer_r1_20260828
m872_m803_dc_m701=reviews/m701_m519_r9_pre_eda_shell_failure_receipt_r1_20260828
m872_m803_dc_r10_failure=dc_handoff/runs/m519_r10_pre_attempt_shell_failure.693765.receipt
m872_m803_dc_m740=reviews/m740_m519_r10_pre_eda_shell_failure_fresh_hammer_r1_20260828
m872_m803_dc_r11_quarantine=dc_handoff/runs/m519_r11_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260828.failed_or_incomplete.974009.quarantine
m872_m803_dc_r11_attempt=dc_handoff/runs/.m519_r11_channel_local_fault_dc_attempt_consumed
m872_m803_dc_m752=reviews/m752_m519_r11_license_env_failure_fresh_hammer_r1_20260828
m872_m803_dc_r12_quarantine=dc_handoff/runs/m519_r12_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260828.failed_or_incomplete.1800161.quarantine
m872_m803_dc_r12_attempt=dc_handoff/runs/.m519_r12_channel_local_fault_dc_attempt_consumed
m872_m803_dc_m769=reviews/m769_m519_r12_postdc_log_gate_failure_fresh_hammer_r1_20260828
m872_m803_dc_m774=reviews/m774_m519_r13_bootstrap_whitelist_three_axis_dc_source_fresh_hammer_r1_20260828
m872_m803_dc_m780=reviews/m780_m519_r14_artifact_complete_three_axis_dc_source_fresh_hammer_r1_20260828
m872_m803_dc_m800=reviews/m800_m519_r15_k8_tim209_failure_hammer_r1_20260828
m872_m803_dc_m803_handoff=reviews/m803_c2_r16_channel_split_author_handoff_r1_20260828
m872_m803_dc_r25_release=contracts/m861_m859_c2_r25_shared_whitelist_vcs_launch_admission_r1_20260829.json
m872_m803_dc_r25_result=results/m859_c2_r25_shared_whitelist_vcs_r1_20260829
m872_m803_dc_m867=reviews/m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_r1_20260829
m872_m803_dc_bootstrap_block_sha256=3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1
m872_m803_dc_r12_dc_log_sha256=03f153c07bfec23e45e0cee940a13c7c3f3dd24c4b826b2ab491d577a4bdb5ba
m872_m803_dc_canonical="${m872_m803_dc_dc_root}/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
m872_m803_dc_work="${m872_m803_dc_dc_root}/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_work.$$"
m872_m803_dc_attempt="${m872_m803_dc_dc_root}/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_attempt_consumed"
m872_m803_dc_quarantine="${m872_m803_dc_canonical}.failed_or_incomplete.$$.quarantine"
m872_m803_dc_preflight_staging="${m872_m803_dc_dc_root}/runs/.m872_m803_dc_preflight.$$.staging"
m872_m803_dc_preflight_reject="${m872_m803_dc_canonical}.preflight_rejected.$$.quarantine"
m872_m803_dc_license_preflight_staging="${m872_m803_dc_dc_root}/runs/.m872_m803_dc_license_preflight.$$.staging"
m872_m803_dc_license_preflight_reject="${m872_m803_dc_canonical}.license_preflight_rejected.$$.quarantine"
m872_m803_dc_uid="$(id -u)"
m872_m803_dc_attempt_consumed=0
if [[ -n "${M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST:-}" && \
      -n "${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT:-}" ]]; then
    m872_m803_dc_pre_attempt_receipt_root="${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT:-}"
elif [[ -n "${M872_M803_DC_NO_EDA_SELF_TEST:-}" && \
      -n "${M872_M803_DC_SELF_TEST_ROOT:-}" ]]; then
    m872_m803_dc_pre_attempt_receipt_root="${M872_M803_DC_SELF_TEST_ROOT:-}"
elif [[ -n "${M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST:-}" && \
        -n "${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT:-}" ]]; then
    m872_m803_dc_pre_attempt_receipt_root="${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT:-}"
else
    m872_m803_dc_pre_attempt_receipt_root="${m872_m803_dc_dc_root}/runs"
fi
m872_m803_dc_pre_attempt_receipt="${m872_m803_dc_pre_attempt_receipt_root}/m872_m803_dc_pre_attempt_shell_failure.$$.receipt"

# This minimal trap is installed before any admission/helper call.  It uses no
# compound local declaration and guarantees a fresh, noncanonical, double-
# sealed receipt for every runtime shell failure before the attempt is consumed
# or the full post-work cleanup trap takes ownership.
m872_m803_dc_pre_attempt_seal_dir() {
    local seal_dir
    seal_dir=$1
    (
        cd "${seal_dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' -print0 | sort -z | \
            xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m872_m803_dc_pre_attempt_failure_cleanup() {
    m872_m803_dc_pre_attempt_saved_rc=$?
    local rc
    rc=${m872_m803_dc_pre_attempt_saved_rc}
    set +e
    if [[ "${rc}" -ne 0 && "${m872_m803_dc_attempt_consumed}" -eq 0 && \
          -n "${m872_m803_dc_pre_attempt_receipt_root}" && \
          ! -e "${m872_m803_dc_pre_attempt_receipt}" ]]; then
        mkdir -p "${m872_m803_dc_pre_attempt_receipt}"
        printf 'status=PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED\nexit_code=%s\nattempt_consumed=false\nrunner=%s\n' \
            "${rc}" "${m872_m803_dc_runner}" \
            >"${m872_m803_dc_pre_attempt_receipt}/FAILURE.txt"
        m872_m803_dc_pre_attempt_seal_dir "${m872_m803_dc_pre_attempt_receipt}"
    fi
    return "${rc}"
}
trap m872_m803_dc_pre_attempt_failure_cleanup EXIT

# All memory units are KiB, matching /proc/meminfo.
m872_m803_dc_preflight_commit_kib=67108864
m872_m803_dc_runtime_commit_kib=33554432
m872_m803_dc_mem_available_kib=134217728
m872_m803_dc_swap_free_kib=33554432

m872_m803_dc_sha() { sha256sum "$1" | awk '{print $1}'; }
m872_m803_dc_strict_json() {
    /usr/libexec/platform-python3.6 - "$1" <<'PY'
import json
import sys

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result

def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)

with open(sys.argv[1], "rb") as handle:
    json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique_object,
               parse_constant=reject_nonfinite)
PY
}
m872_m803_dc_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m872_m803_dc_sha "${path}")" == "${expected}" ]] || {
        echo "M872 M803 DC identity mismatch: ${path}" >&2
        exit 3
    }
}
m872_m803_dc_closed_keys() {
    local file=$1 expression=$2 expected=$3 actual
    actual="$(jq -er "${expression} | keys[]" "${file}" | LC_ALL=C sort | paste -sd, -)"
    [[ "${actual}" == "${expected}" ]] || {
        echo "M872 M803 DC unknown or missing JSON key at ${expression}: ${actual}" >&2
        exit 3
    }
}
m872_m803_dc_json_equal() {
    local left_file=$1 left_expr=$2 right_file=$3 right_expr=$4
    [[ "$(jq -er "${left_expr}" "${left_file}")" == \
       "$(jq -er "${right_expr}" "${right_file}")" ]] || {
        echo "M872 M803 DC admission/contract identity disagreement: ${left_expr}" >&2
        exit 3
    }
}
m872_m803_dc_verify_double_seal_file() {
    local payload
    local sidecar
    local outer
    local dir
    local base
    payload=$1
    sidecar="${payload}.sha256"
    outer="${payload}.sha256.seal.sha256"
    [[ -f "${sidecar}" && -f "${outer}" ]] || exit 3
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}

# Reject a symlink at any component, not only at the leaf.  Requiring the
# lexical absolute path to equal realpath -m also rejects dot-dot escape and a
# symlink component even when that symlink happens to resolve beneath point.
m872_m803_dc_path_is_absolute_nosymlink() {
    local path=$1 kind=$2 normalized cursor part
    local -a parts=()
    [[ "${path}" == /* && "${path}" != *$'\n'* && "${path}" != *$'\t'* ]] || return 1
    normalized="$(realpath -m -- "${path}")" || return 1
    [[ "${normalized}" == "${path}" ]] || return 1
    cursor=/
    IFS=/ read -r -a parts <<<"${path#/}"
    for part in "${parts[@]}"; do
        [[ -n "${part}" && "${part}" != . && "${part}" != .. ]] || return 1
        if [[ "${cursor}" == / ]]; then
            cursor="/${part}"
        else
            cursor="${cursor}/${part}"
        fi
        [[ ! -L "${cursor}" ]] || return 1
    done
    case "${kind}" in
        dir) [[ -d "${path}" && ! -L "${path}" ]] ;;
        file) [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] ;;
        *) return 1 ;;
    esac
}

m872_m803_dc_artifact_paths_are_closed() {
    local point=$1 netlist path
    local design=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    netlist="${point}/netlist"
    m872_m803_dc_path_is_absolute_nosymlink "${point}" dir || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${netlist}" dir || return 1
    [[ "${netlist}" == "${point}/netlist" ]] || return 1
    for path in \
            "${netlist}/${design}_mapped.v" \
            "${netlist}/${design}_mapped.sdc" \
            "${netlist}/${design}.ddc" \
            "${netlist}/${design}.svf" \
            "${point}/reports/area.rpt" \
            "${point}/reports/qor.rpt" \
            "${point}/reports/timing_setup.rpt"; do
        [[ "${path}" == "${point}/netlist/"* || \
           "${path}" == "${point}/reports/"* ]] || return 1
        m872_m803_dc_path_is_absolute_nosymlink "${path}" file || return 1
        [[ "$(realpath -e -- "${path}")" == "${path}" ]] || return 1
    done
}

# Recompute the exact canonical bytes of both receipts from the seven live
# artifacts.  This is used immediately after the single-directory publication,
# after RUN_COMPLETE, and again after the final enclosing manifest is sealed.
m872_m803_dc_verify_live_artifact_receipts() {
    local point=$1 receipt_dir inventory terminal design netlist
    local label path size sha line inventory_expected terminal_expected index
    local -a labels=(mapped_verilog mapped_sdc ddc svf area_report qor_report setup_timing_report)
    local -a paths=()
    design=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    netlist="${point}/netlist"
    receipt_dir="${point}/artifact_receipts"
    inventory="${receipt_dir}/artifact_inventory.tsv"
    terminal="${receipt_dir}/artifact_terminal_receipt.txt"
    m872_m803_dc_artifact_paths_are_closed "${point}" || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${receipt_dir}" dir || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${inventory}" file || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${terminal}" file || return 1
    paths=(
        "${netlist}/${design}_mapped.v"
        "${netlist}/${design}_mapped.sdc"
        "${netlist}/${design}.ddc"
        "${netlist}/${design}.svf"
        "${point}/reports/area.rpt"
        "${point}/reports/qor.rpt"
        "${point}/reports/timing_setup.rpt"
    )
    inventory_expected=$'artifact\tpath\tsize_bytes\tsha256\n'
    terminal_expected=$'artifact_count=7\n'
    for index in 0 1 2 3 4 5 6; do
        label=${labels[${index}]}
        path=${paths[${index}]}
        size="$(stat -Lc %s -- "${path}")" || return 1
        sha="$(m872_m803_dc_sha "${path}")" || return 1
        [[ "${size}" =~ ^[1-9][0-9]*$ && "${sha}" =~ ^[0-9a-f]{64}$ ]] || return 1
        printf -v line '%s\t%s\t%s\t%s\n' "${label}" \
            "${path#${point}/}" "${size}" "${sha}" || return 1
        inventory_expected+="${line}"
        printf -v line '%s_path=%s\n%s_size_bytes=%s\n%s_sha256=%s\n' \
            "${label}" "${path#${point}/}" "${label}" "${size}" \
            "${label}" "${sha}" || return 1
        terminal_expected+="${line}"
    done
    terminal_expected+=$'status=PASS_M872_M803_DC_ATOMIC_RECEIPT_AND_LIVE_SEVEN_ARTIFACT_TUPLE\n'
    printf '%s' "${inventory_expected}" | cmp -s - "${inventory}" || return 1
    printf '%s' "${terminal_expected}" | cmp -s - "${terminal}" || return 1
}

# The final root manifest is authoritative only if it actually includes each
# artifact and both receipts once, with the same live SHA already frozen in the
# two receipts.  No string claim can substitute for these nine manifest rows.
m872_m803_dc_verify_axis_artifact_manifest() {
    local root=$1 point=$2 manifest rel path expected actual count
    local design=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    local -a paths=()
    m872_m803_dc_verify_live_artifact_receipts "${point}" || return 1
    manifest="${root}/SHA256SUMS"
    m872_m803_dc_path_is_absolute_nosymlink "${root}" dir || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${manifest}" file || return 1
    [[ "${point}" == "${root}/"* ]] || return 1
    rel=${point#${root}/}
    [[ -n "${rel}" && "${rel}" != "${point}" && "${rel}" != */../* ]] || return 1
    paths=(
        "${point}/netlist/${design}_mapped.v"
        "${point}/netlist/${design}_mapped.sdc"
        "${point}/netlist/${design}.ddc"
        "${point}/netlist/${design}.svf"
        "${point}/reports/area.rpt"
        "${point}/reports/qor.rpt"
        "${point}/reports/timing_setup.rpt"
        "${point}/artifact_receipts/artifact_inventory.tsv"
        "${point}/artifact_receipts/artifact_terminal_receipt.txt"
    )
    for path in "${paths[@]}"; do
        expected="$(m872_m803_dc_sha "${path}")" || return 1
        rel="./${path#${root}/}"
        count="$(awk -v p="${rel}" '$2 == p {n++} END {print n+0}' "${manifest}")" || return 1
        [[ "${count}" -eq 1 ]] || return 1
        actual="$(awk -v p="${rel}" '$2 == p {print $1}' "${manifest}")" || return 1
        [[ "${actual}" == "${expected}" ]] || return 1
    done
}

# Publish the inventory and terminal receipt as one directory rename.  Every
# operation has an explicit checked result because production calls this helper
# from an OR-list, where Bash errexit is deliberately not relied upon.
m872_m803_dc_record_output_artifacts() {
    local point=$1
    local design=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    local netlist="${point}/netlist"
    local receipt_dir="${point}/artifact_receipts"
    local staging="${point}/.artifact_receipts.$$.staging"
    local inventory="${staging}/artifact_inventory.tsv"
    local terminal="${staging}/artifact_terminal_receipt.txt"
    local label path size sha
    local -a labels=(mapped_verilog mapped_sdc ddc svf area_report qor_report setup_timing_report)
    local -a paths=(
        "${netlist}/${design}_mapped.v"
        "${netlist}/${design}_mapped.sdc"
        "${netlist}/${design}.ddc"
        "${netlist}/${design}.svf"
        "${point}/reports/area.rpt"
        "${point}/reports/qor.rpt"
        "${point}/reports/timing_setup.rpt"
    )
    local -a sizes=()
    local -a shas=()

    m872_m803_dc_artifact_paths_are_closed "${point}" || return 1
    [[ ! -e "${receipt_dir}" && ! -L "${receipt_dir}" && \
       ! -e "${staging}" && ! -L "${staging}" ]] || return 1
    mkdir -- "${staging}" || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${staging}" dir || return 1

    printf 'artifact\tpath\tsize_bytes\tsha256\n' >"${inventory}" || return 1
    printf 'artifact_count=7\n' >"${terminal}" || return 1
    for index in 0 1 2 3 4 5 6; do
        label=${labels[${index}]}
        path=${paths[${index}]}
        size="$(stat -Lc %s "${path}")" || return 1
        sha="$(m872_m803_dc_sha "${path}")" || return 1
        [[ "${size}" =~ ^[1-9][0-9]*$ && "${sha}" =~ ^[0-9a-f]{64}$ ]] || return 1
        sizes[${index}]=${size}
        shas[${index}]=${sha}
        printf '%s\t%s\t%s\t%s\n' "${label}" \
            "${path#${point}/}" "${size}" "${sha}" >>"${inventory}" || return 1
        printf '%s_path=%s\n%s_size_bytes=%s\n%s_sha256=%s\n' \
            "${label}" "${path#${point}/}" "${label}" "${size}" \
            "${label}" "${sha}" >>"${terminal}" || return 1
    done
    # Close the check/use window before publishing either success receipt.
    for index in 0 1 2 3 4 5 6; do
        path=${paths[${index}]}
        [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] || return 1
        [[ "$(stat -Lc %s "${path}")" == "${sizes[${index}]}" && \
           "$(m872_m803_dc_sha "${path}")" == "${shas[${index}]}" ]] || return 1
    done
    printf 'status=PASS_M872_M803_DC_ATOMIC_RECEIPT_AND_LIVE_SEVEN_ARTIFACT_TUPLE\n' \
        >>"${terminal}" || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${inventory}" file || return 1
    m872_m803_dc_path_is_absolute_nosymlink "${terminal}" file || return 1
    if [[ "${M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST:-}" == 1 && \
          "${M872_M803_DC_ARTIFACT_TEST_FAULT:-}" == partial_mv ]]; then
        mkdir -- "${receipt_dir}" || return 1
        printf 'block-atomic-rename\n' >"${receipt_dir}/injected_blocker" || return 1
    fi
    mv -T -- "${staging}" "${receipt_dir}" || return 1
    m872_m803_dc_verify_live_artifact_receipts "${point}" || return 1
}

# The current full runner is syntax-checked before any admission, resource,
# attempt or tool path.  The injectable self-test exercises the exact double-
# seal helper and optionally the early failure trap, then exits before EDA.
bash -n "${m872_m803_dc_runner}"
if [[ -n "${M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST:-}" ]]; then
    [[ "${M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST}" == 1 && \
       -n "${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT:-}" && \
       "${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT}" == /* && \
       -d "${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT}" ]] || exit 88
    m872_m803_dc_artifact_test_root=${M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT}
    m872_m803_dc_artifact_test_point="${m872_m803_dc_artifact_test_root}/point"
    m872_m803_dc_artifact_test_design=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    m872_m803_dc_artifact_test_reset() {
        rm -rf -- "${m872_m803_dc_artifact_test_point}" || exit 89
        rm -f -- "${m872_m803_dc_artifact_test_root}/SHA256SUMS" \
            "${m872_m803_dc_artifact_test_root}/SHA256SUMS.seal.sha256" || exit 89
        mkdir -p -- "${m872_m803_dc_artifact_test_point}/netlist" \
            "${m872_m803_dc_artifact_test_point}/reports" || exit 89
        printf 'mapped-verilog\n' >"${m872_m803_dc_artifact_test_point}/netlist/${m872_m803_dc_artifact_test_design}_mapped.v" || exit 89
        printf 'mapped-sdc\n' >"${m872_m803_dc_artifact_test_point}/netlist/${m872_m803_dc_artifact_test_design}_mapped.sdc" || exit 89
        printf 'ddc\n' >"${m872_m803_dc_artifact_test_point}/netlist/${m872_m803_dc_artifact_test_design}.ddc" || exit 89
        printf 'svf\n' >"${m872_m803_dc_artifact_test_point}/netlist/${m872_m803_dc_artifact_test_design}.svf" || exit 89
        printf 'area\n' >"${m872_m803_dc_artifact_test_point}/reports/area.rpt" || exit 89
        printf 'qor\n' >"${m872_m803_dc_artifact_test_point}/reports/qor.rpt" || exit 89
        printf 'setup\n' >"${m872_m803_dc_artifact_test_point}/reports/timing_setup.rpt" || exit 89
    }
    m872_m803_dc_artifact_test_seal_root() {
        (
            cd "${m872_m803_dc_artifact_test_root}" || exit 1
            find ./point -type f -print0 | sort -z | \
                xargs -0 sha256sum >SHA256SUMS || exit 1
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 || exit 1
            sha256sum -c SHA256SUMS >/dev/null || exit 1
            sha256sum -c SHA256SUMS.seal.sha256 >/dev/null || exit 1
        )
    }
    m872_m803_dc_artifact_test_reset
    m872_m803_dc_record_output_artifacts "${m872_m803_dc_artifact_test_point}"
    m872_m803_dc_verify_live_artifact_receipts "${m872_m803_dc_artifact_test_point}"
    [[ "$(awk -F= '/^artifact_count=/ {print $2}' \
        "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_terminal_receipt.txt")" == 7 ]] || exit 89
    printf 'status=SELFTEST_RUN_COMPLETE_AFTER_ATOMIC_RECEIPTS\n' \
        >"${m872_m803_dc_artifact_test_point}/RUN_COMPLETE.txt" || exit 89
    m872_m803_dc_verify_live_artifact_receipts "${m872_m803_dc_artifact_test_point}"
    m872_m803_dc_artifact_test_seal_root
    m872_m803_dc_verify_axis_artifact_manifest \
        "${m872_m803_dc_artifact_test_root}" "${m872_m803_dc_artifact_test_point}" || exit 89
    m872_m803_dc_artifact_test_negative_count=0
    for artifact in \
            "netlist/${m872_m803_dc_artifact_test_design}_mapped.v" \
            "netlist/${m872_m803_dc_artifact_test_design}_mapped.sdc" \
            "netlist/${m872_m803_dc_artifact_test_design}.ddc" \
            "netlist/${m872_m803_dc_artifact_test_design}.svf" \
            "reports/area.rpt" "reports/qor.rpt" \
            "reports/timing_setup.rpt"; do
        for fault in deleted zero symlink; do
            m872_m803_dc_artifact_test_reset
            m872_m803_dc_artifact_test_path="${m872_m803_dc_artifact_test_point}/${artifact}"
            case "${fault}" in
                deleted) rm -f -- "${m872_m803_dc_artifact_test_path}" || exit 90 ;;
                zero) : >"${m872_m803_dc_artifact_test_path}" || exit 90 ;;
                symlink)
                    mv -- "${m872_m803_dc_artifact_test_path}" \
                        "${m872_m803_dc_artifact_test_path}.target" || exit 90
                    ln -s "$(basename "${m872_m803_dc_artifact_test_path}.target")" \
                        "${m872_m803_dc_artifact_test_path}" || exit 90
                    ;;
            esac
            if m872_m803_dc_record_output_artifacts "${m872_m803_dc_artifact_test_point}"; then
                exit 90
            fi
            [[ ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_inventory.tsv" && \
               ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_terminal_receipt.txt" && \
               ! -e "${m872_m803_dc_artifact_test_point}/RUN_COMPLETE.txt" ]] || exit 90
            m872_m803_dc_artifact_test_negative_count=$((m872_m803_dc_artifact_test_negative_count + 1))
        done
    done
    [[ "${m872_m803_dc_artifact_test_negative_count}" -eq 21 ]] || exit 90

    # A destination collision makes the single directory rename fail.  Neither
    # receipt leaf and no RUN_COMPLETE may become visible.
    m872_m803_dc_artifact_test_reset
    M872_M803_DC_ARTIFACT_TEST_FAULT=partial_mv
    export M872_M803_DC_ARTIFACT_TEST_FAULT
    if m872_m803_dc_record_output_artifacts "${m872_m803_dc_artifact_test_point}"; then
        exit 90
    fi
    unset M872_M803_DC_ARTIFACT_TEST_FAULT || exit 90
    [[ ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_inventory.tsv" && \
       ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_terminal_receipt.txt" && \
       ! -e "${m872_m803_dc_artifact_test_point}/RUN_COMPLETE.txt" ]] || exit 90

    # An artifact directory symlink is an ancestor escape even though each leaf
    # reached through it is a regular file.
    m872_m803_dc_artifact_test_reset
    mv -- "${m872_m803_dc_artifact_test_point}/netlist" \
        "${m872_m803_dc_artifact_test_root}/external_netlist" || exit 90
    ln -s -- "${m872_m803_dc_artifact_test_root}/external_netlist" \
        "${m872_m803_dc_artifact_test_point}/netlist" || exit 90
    if m872_m803_dc_record_output_artifacts "${m872_m803_dc_artifact_test_point}"; then
        exit 90
    fi
    [[ ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_inventory.tsv" && \
       ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts/artifact_terminal_receipt.txt" ]] || exit 90
    rm -rf -- "${m872_m803_dc_artifact_test_root}/external_netlist" || exit 90

    # A lexical path escape is rejected before any receipt publication.
    m872_m803_dc_artifact_test_reset
    if m872_m803_dc_record_output_artifacts \
            "${m872_m803_dc_artifact_test_point}/../point"; then
        exit 90
    fi
    [[ ! -e "${m872_m803_dc_artifact_test_point}/artifact_receipts" ]] || exit 90

    # Bytes changed after atomic receipt publication must fail both the live
    # close and the final-manifest close, even if a RUN_COMPLETE file is forged.
    m872_m803_dc_artifact_test_reset
    m872_m803_dc_record_output_artifacts "${m872_m803_dc_artifact_test_point}"
    printf 'mutated-after-receipt\n' \
        >"${m872_m803_dc_artifact_test_point}/netlist/${m872_m803_dc_artifact_test_design}.ddc" || exit 90
    if m872_m803_dc_verify_live_artifact_receipts "${m872_m803_dc_artifact_test_point}"; then
        exit 90
    fi
    printf 'status=FORGED_RUN_COMPLETE_MUST_NOT_PASS\n' \
        >"${m872_m803_dc_artifact_test_point}/RUN_COMPLETE.txt" || exit 90
    m872_m803_dc_artifact_test_seal_root
    if m872_m803_dc_verify_axis_artifact_manifest \
            "${m872_m803_dc_artifact_test_root}" "${m872_m803_dc_artifact_test_point}"; then
        exit 90
    fi

    printf 'status=PASS_M872_M803_DC_ATOMIC_ARTIFACT_GATE_NO_EDA_SELF_TEST\npositive_cases=1\nnegative_cases=25\ndeleted_cases=7\nzero_byte_cases=7\nleaf_symlink_cases=7\npartial_publish_cases=1\nancestor_symlink_cases=1\npath_escape_cases=1\npostreceipt_mutation_cases=1\nfinal_manifest_positive_cases=1\n' \
        >"${m872_m803_dc_artifact_test_root}/ARTIFACT_GATE_SELF_TEST_PASS.txt" || exit 90
    trap - EXIT
    exit 0
fi
if [[ -n "${M872_M803_DC_NO_EDA_SELF_TEST:-}" ]]; then
    [[ "${M872_M803_DC_NO_EDA_SELF_TEST}" == 1 && \
       -n "${M872_M803_DC_SELF_TEST_ROOT:-}" && \
       "${M872_M803_DC_SELF_TEST_ROOT}" == /* && \
       -d "${M872_M803_DC_SELF_TEST_ROOT}" ]] || exit 83
    m872_m803_dc_self_payload="${M872_M803_DC_SELF_TEST_ROOT}/payload.txt"
    printf 'm872-m803-c2-r16-no-eda-self-test\n' >"${m872_m803_dc_self_payload}"
    (
        cd "${M872_M803_DC_SELF_TEST_ROOT}"
        sha256sum payload.txt >payload.txt.sha256
        sha256sum payload.txt.sha256 >payload.txt.sha256.seal.sha256
    )
    m872_m803_dc_verify_double_seal_file "${m872_m803_dc_self_payload}"
    [[ -z "${M872_M803_DC_SELF_TEST_INJECT_PRE_ATTEMPT_FAILURE:-}" ]] || exit 86
    trap - EXIT
    exit 0
fi

# Unlike the early helper test above, this mode traverses the complete sealed
# candidate-admission and recovery-contract validation path.  It switches only
# the admission identity and expected launch bit, then exits at the explicit
# marker below before resource preflight, attempt publication, or any tool.
if [[ -n "${M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST:-}" ]]; then
    [[ "${M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST}" == 1 && \
       -z "${M872_M803_DC_NO_EDA_SELF_TEST:-}" && \
       -n "${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT:-}" && \
       "${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT}" == /* && \
       -d "${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT}" ]] || exit 87
    m872_m803_dc_admission=${m872_m803_dc_candidate}
    m872_m803_dc_expected_admission_status=READY_FOR_FRESH_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED
    m872_m803_dc_expected_launch_now=false
fi

[[ -n "${M872_M803_DC_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m872_m803_dc_sha "${m872_m803_dc_runner}")" == \
   "${M872_M803_DC_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M872 M803 DC caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M872 M803 DC source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ ! -e "${m872_m803_dc_canonical}" && ! -e "${m872_m803_dc_work}" && \
   ! -e "${m872_m803_dc_attempt}" && ! -e "${m872_m803_dc_quarantine}" && \
   ! -e "${m872_m803_dc_preflight_staging}" && \
   ! -e "${m872_m803_dc_license_preflight_staging}" ]] || {
    echo "M872 M803 DC refuses consumed or colliding result identity" >&2
    exit 5
}
[[ -z "${M872_M803_DC_DC_RUN:-}" ]] || {
    echo "M872 M803 DC canonical path override is forbidden" >&2
    exit 5
}

cd "${m872_m803_dc_hw_root}"
m872_m803_dc_expect "${m872_m803_dc_admission}" \
    "${M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256}"
m872_m803_dc_verify_double_seal_file "${m872_m803_dc_admission}"
m872_m803_dc_strict_json "${m872_m803_dc_admission}" || exit 3
jq -e --arg expected_status "${m872_m803_dc_expected_admission_status}" \
       --argjson expected_launch_now "${m872_m803_dc_expected_launch_now}" \
       '.status == $expected_status
       and .launch_now == $expected_launch_now
       and .authorization.run_dc == true
       and .authorization.max_attempts == 1
       and .authorization.run_vcs == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false
       and .authorization.run_formality == false
       and .authorization.run_remote == false' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.authorization' \
    'max_attempts,run_dc,run_formality,run_pt,run_ptpx,run_remote,run_vcs'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r10_repair_provenance' \
    'm694_manifest_file_sha256,m694_outer_seal_file_sha256,m694_review_path,m694_review_sha256,m694_status,m701_manifest_file_sha256,m701_no_eda_started,m701_outer_seal_file_sha256,m701_review_path,m701_review_sha256,m701_status,r10_is_additive,r9_attempt_remains_absent,r9_result_remains_absent'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r11_repair_provenance' \
    'm740_manifest_file_sha256,m740_outer_seal_file_sha256,m740_review_path,m740_review_sha256,m740_status,r10_attempt_consumed,r10_canonical_absent,r10_failure_manifest_file_sha256,r10_failure_outer_seal_file_sha256,r10_failure_path,r10_failure_payload_sha256,r11_is_additive'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r12_license_recovery_provenance' \
    'm752_manifest_file_sha256,m752_outer_seal_file_sha256,m752_review_path,m752_review_sha256,m752_status,r11_attempt_manifest_file_sha256,r11_attempt_outer_seal_file_sha256,r11_attempt_path,r11_attempt_payload_sha256,r11_canonical_absent,r11_quarantine_manifest_file_sha256,r11_quarantine_outer_seal_file_sha256,r11_quarantine_path,r12_is_additive'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r13_bootstrap_log_recovery_provenance' \
    'bootstrap_block_end_offset,bootstrap_block_sha256,bootstrap_block_start_max_line,bootstrap_error_line,m769_manifest_file_sha256,m769_outer_seal_file_sha256,m769_review_path,m769_review_sha256,m769_status,r12_attempt_manifest_file_sha256,r12_attempt_outer_seal_file_sha256,r12_attempt_path,r12_attempt_payload_sha256,r12_canonical_absent,r12_dc_log_sha256,r12_failure_payload_sha256,r12_quarantine_manifest_file_sha256,r12_quarantine_outer_seal_file_sha256,r12_quarantine_path,r13_all_three_axes_rerun,r13_is_additive,r13_reuses_r12_k1'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r14_artifact_completeness_repair_provenance' \
    'artifact_gate_scope,m774_manifest_file_sha256,m774_outer_seal_file_sha256,m774_review_path,m774_review_sha256,m774_status,r13_attempt_absent,r13_canonical_absent,r14_all_three_axes_rerun,r14_is_additive,r14_reuses_r13_outputs'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.r15_atomic_artifact_gate_repair_provenance' \
    'artifact_manifest_scope,artifact_path_scope,atomic_publication_scope,m780_manifest_file_sha256,m780_outer_seal_file_sha256,m780_p1_ids,m780_review_path,m780_review_sha256,m780_status,r14_attempt_absent,r14_canonical_absent,r15_all_three_axes_rerun,r15_is_additive,r15_reuses_r14_outputs'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.license_environment' \
    'dc_ultra_feature,design_compiler_feature,lm_license_file,lmutil_path,lmutil_sha256,snps_license_file_path,snps_license_file_sha256,snpslmd_license_file'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.m800_failure_authority' \
    'outer_seal_file_sha256,r15_attempt_consumed,r15_cross_attempt_k1_reuse_forbidden,required_successor_behavior,review_path,review_sha256,status'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.m803_vcs_authority' \
    'dc_or_physical_ppa_from_vcs,exact_cycles_k1x8,exact_cycles_k8,m867_manifest_sha256,m867_outer_seal_file_sha256,m867_review_path,m867_review_sha256,m867_status,r25_release_outer_seal_file_sha256,r25_release_path,r25_release_sha256,r25_result_outer_seal_file_sha256,r25_result_path'
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.three_axis_pre_attempt_plan' \
    'all_three_axes_same_attempt_required,check_design_and_check_timing_before_compile,k1_binding,k1x8_binding,k8_binding,opt150_required_each_axis,partial_axis_or_cross_attempt_reuse_citable,point_order,same_filelist_tcl_sdc_libraries_clock,source_analyze_then_elaborate_before_compile,tim209_required_each_axis'
jq -e '.m800_failure_authority.r15_attempt_consumed == true
       and .m800_failure_authority.r15_cross_attempt_k1_reuse_forbidden == true
       and .m803_vcs_authority.m867_status == "PASS100_M859_R25_DIRECTED_COMPONENT_VCS_E3_RESULT_ADMITTED"
       and .m803_vcs_authority.exact_cycles_k8 == [51,131,486,1231,14]
       and .m803_vcs_authority.exact_cycles_k1x8 == [53,133,499,1246,14]
       and .m803_vcs_authority.dc_or_physical_ppa_from_vcs == false
       and .three_axis_pre_attempt_plan.point_order == ["k1","k8","k1x8"]
       and .three_axis_pre_attempt_plan.tim209_required_each_axis == 0
       and .three_axis_pre_attempt_plan.opt150_required_each_axis == 0
       and .three_axis_pre_attempt_plan.all_three_axes_same_attempt_required == true
       and .three_axis_pre_attempt_plan.partial_axis_or_cross_attempt_reuse_citable == false' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.r10_repair_provenance.r10_is_additive == true
       and .r10_repair_provenance.m701_no_eda_started == true
       and .r10_repair_provenance.r9_result_remains_absent == true
       and .r10_repair_provenance.r9_attempt_remains_absent == true' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
[[ "$(jq -er '.r10_repair_provenance.m694_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m694}/review.json" ]] || exit 3
[[ "$(jq -er '.r10_repair_provenance.m701_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m701}/review.json" ]] || exit 3

# R10 exists only because the independently sealed R9 GO was consumed by the
# independently sealed pre-EDA shell failure.  Bind both exact receipts and
# their exact statuses before any resource preflight or attempt consumption.
for sealed in "${m872_m803_dc_m694}" "${m872_m803_dc_m701}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
for item in m694 m701; do
    if [[ "${item}" == m694 ]]; then
        basis=${m872_m803_dc_m694}
    else
        basis=${m872_m803_dc_m701}
    fi
    m872_m803_dc_expect "${basis}/review.json" \
        "$(jq -er ".r10_repair_provenance.${item}_review_sha256" "${m872_m803_dc_admission}")"
    m872_m803_dc_expect "${basis}/SHA256SUMS" \
        "$(jq -er ".r10_repair_provenance.${item}_manifest_file_sha256" "${m872_m803_dc_admission}")"
    m872_m803_dc_expect "${basis}/SHA256SUMS.seal.sha256" \
        "$(jq -er ".r10_repair_provenance.${item}_outer_seal_file_sha256" "${m872_m803_dc_admission}")"
done
[[ "$(jq -er '.status' "${m872_m803_dc_m694}/review.json")" == \
   "$(jq -er '.r10_repair_provenance.m694_status' "${m872_m803_dc_admission}")" ]] || exit 3
[[ "$(jq -er '.status' "${m872_m803_dc_m701}/review.json")" == \
   "$(jq -er '.r10_repair_provenance.m701_status' "${m872_m803_dc_admission}")" ]] || exit 3
jq -e '.status == "GO_ONE_M519_R9_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED"
       and .severity_counts.p0 == 0 and .severity_counts.p1 == 0
       and .authorization.max_attempts == 1
       and .authorization.run_dc == true
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false' \
    "${m872_m803_dc_m694}/review.json" >/dev/null || exit 3
jq -e '.status == "PRE_EDA_SHELL_FAILURE__NO_DC_STARTED__M519_R9_NOT_CITABLE__ADDITIVE_R10_REQUIRED"
       and .failure.exit_code == 1
       and .failure.failure_stage == "shell function definition before admission verification, preflight, attempt consumption, or EDA launch"
       and .observed_absence_after_failure.m519_r9_canonical_result_absent == true
       and .observed_absence_after_failure.m519_r9_attempt_sentinel_absent == true
       and .claim_boundary.dc_started == false
       and (.required_next_step | contains("additive M519 R10"))' \
    "${m872_m803_dc_m701}/review.json" >/dev/null || exit 3

# R11 is an additive successor to the double-sealed R10 pre-EDA failure.  Bind
# both that receipt and M740's independent causal audit before evaluating the
# remaining inherited admission and contract predicates.
for sealed in "${m872_m803_dc_r10_failure}" "${m872_m803_dc_m740}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m872_m803_dc_expect "${m872_m803_dc_r10_failure}/FAILURE.txt" \
    "$(jq -er '.r11_repair_provenance.r10_failure_payload_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r10_failure}/SHA256SUMS" \
    "$(jq -er '.r11_repair_provenance.r10_failure_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r10_failure}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r11_repair_provenance.r10_failure_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m740}/review.json" \
    "$(jq -er '.r11_repair_provenance.m740_review_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m740}/SHA256SUMS" \
    "$(jq -er '.r11_repair_provenance.m740_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m740}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r11_repair_provenance.m740_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.r11_repair_provenance.r10_failure_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_r10_failure}" ]] || exit 3
[[ "$(jq -er '.r11_repair_provenance.m740_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m740}/review.json" ]] || exit 3
jq -e '.r11_repair_provenance.r11_is_additive == true
       and .r11_repair_provenance.r10_attempt_consumed == false
       and .r11_repair_provenance.r10_canonical_absent == true' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.status == "PASS_FAILURE_AUDIT__R10_BLOCKED__PRE_EDA_JQ_ESCAPE__ADDITIVE_R11_REQUIRED"
       and .verdict == "PASS"
       and .score_out_of_100 == 100
       and .finding.exact_invalid_program_replay_rc == 3
       and .finding.same_predicate_without_literal_backslash_rc == 0
       and .authorization.run_r11_now == false
       and .authorization.run_dc == false' \
    "${m872_m803_dc_m740}/review.json" >/dev/null || exit 3
grep -Fxq 'status=PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED' \
    "${m872_m803_dc_r10_failure}/FAILURE.txt"
grep -Fxq 'exit_code=3' "${m872_m803_dc_r10_failure}/FAILURE.txt"
grep -Fxq 'attempt_consumed=false' "${m872_m803_dc_r10_failure}/FAILURE.txt"

# R12 is a license-discovery-only additive successor.  R11 remains a consumed,
# double-sealed failure and can never be reinterpreted as DC or PPA evidence.
for sealed in "${m872_m803_dc_r11_quarantine}" "${m872_m803_dc_r11_attempt}" \
        "${m872_m803_dc_m752}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m872_m803_dc_expect "${m872_m803_dc_r11_quarantine}/SHA256SUMS" \
    "$(jq -er '.r12_license_recovery_provenance.r11_quarantine_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r11_quarantine}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r12_license_recovery_provenance.r11_quarantine_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r11_attempt}/ATTEMPT_CONSUMED.txt" \
    "$(jq -er '.r12_license_recovery_provenance.r11_attempt_payload_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r11_attempt}/SHA256SUMS" \
    "$(jq -er '.r12_license_recovery_provenance.r11_attempt_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r11_attempt}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r12_license_recovery_provenance.r11_attempt_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m752}/review.json" \
    "$(jq -er '.r12_license_recovery_provenance.m752_review_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m752}/SHA256SUMS" \
    "$(jq -er '.r12_license_recovery_provenance.m752_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m752}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r12_license_recovery_provenance.m752_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.r12_license_recovery_provenance.r11_quarantine_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_r11_quarantine}" && \
   "$(jq -er '.r12_license_recovery_provenance.r11_attempt_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_r11_attempt}" && \
   "$(jq -er '.r12_license_recovery_provenance.m752_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m752}/review.json" ]] || exit 3
jq -e '.r12_license_recovery_provenance.r12_is_additive == true
       and .r12_license_recovery_provenance.r11_canonical_absent == true' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.status == "PASS_FAILURE_AUDIT__R11_BLOCKED__LICENSE_DISCOVERY_ENV_OMITTED__ADDITIVE_R12_REQUIRED"
       and .verdict == "PASS" and .score_out_of_100 == 100
       and .severity_counts == {"p0":0,"p1":0,"p2":0}
       and .r11_quarantine.status == "FAILED_OR_INCOMPLETE_DO_NOT_CITE"
       and .r11_attempt.status == "CONSUMED_AT_FIRST_DC_LAUNCH"
       and .minimal_additive_r12.license_preflight_required == true
       and .minimal_additive_r12.license_preflight_before_attempt_consumption == true
       and .minimal_additive_r12.license_preflight_double_sealed == true
       and .authorization.author_additive_r12_source_package == true
       and .authorization.run_r12_now == false
       and .authorization.run_dc == false' \
    "${m872_m803_dc_m752}/review.json" >/dev/null || exit 3

# R13 is an additive successor to R12's consumed, double-sealed post-DC
# classifier failure.  Bind the exact R12 quarantine, unique attempt, and
# M769 independent audit before any license/resource preflight or attempt
# publication.  R12 K1 remains noncitable and is never reused: all three axes
# are rerun under this fresh identity.
for sealed in "${m872_m803_dc_r12_quarantine}" "${m872_m803_dc_r12_attempt}" \
        "${m872_m803_dc_m769}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m872_m803_dc_expect "${m872_m803_dc_r12_quarantine}/RUN_FAILED_OR_INCOMPLETE.txt" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_failure_payload_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_quarantine}/k1/dc.log" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_dc_log_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_quarantine}/SHA256SUMS" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_quarantine_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_quarantine}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_quarantine_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_attempt}/ATTEMPT_CONSUMED.txt" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_attempt_payload_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_attempt}/SHA256SUMS" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_attempt_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_r12_attempt}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_attempt_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m769}/review.json" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.m769_review_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m769}/SHA256SUMS" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.m769_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m769}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r13_bootstrap_log_recovery_provenance.m769_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_quarantine_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_r12_quarantine}" && \
   "$(jq -er '.r13_bootstrap_log_recovery_provenance.r12_attempt_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_r12_attempt}" && \
   "$(jq -er '.r13_bootstrap_log_recovery_provenance.m769_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m769}/review.json" ]] || exit 3
jq -e --arg block_sha "${m872_m803_dc_bootstrap_block_sha256}" \
       --arg log_sha "${m872_m803_dc_r12_dc_log_sha256}" \
       '.r13_bootstrap_log_recovery_provenance.r13_is_additive == true
       and .r13_bootstrap_log_recovery_provenance.r13_all_three_axes_rerun == true
       and .r13_bootstrap_log_recovery_provenance.r13_reuses_r12_k1 == false
       and .r13_bootstrap_log_recovery_provenance.r12_canonical_absent == true
       and .r13_bootstrap_log_recovery_provenance.bootstrap_block_sha256 == $block_sha
       and .r13_bootstrap_log_recovery_provenance.r12_dc_log_sha256 == $log_sha
       and .r13_bootstrap_log_recovery_provenance.bootstrap_block_start_max_line == 64
       and .r13_bootstrap_log_recovery_provenance.bootstrap_block_end_offset == 15
       and .r13_bootstrap_log_recovery_provenance.bootstrap_error_line ==
          "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.status == "PASS_FAILURE_AUDIT__R12_BLOCKED__SOLE_FIXED_BOOTSTRAP_ERROR_FALSE_POSITIVE__ADDITIVE_R13_ALL_AXIS_SOURCE_ONLY_AUTHORIZED"
       and .verdict == "PASS" and .score_out_of_100 == 100
       and .severity_counts == {"p0":0,"p1":0,"p2":0}
       and .r12_quarantine.status == "FAILED_OR_INCOMPLETE_DO_NOT_CITE"
       and .r12_attempt.status == "CONSUMED_AT_FIRST_DC_LAUNCH"
       and .recovery_option_a.verdict == "PREFERRED_AND_SOURCE_ONLY_AUTHORIZED"
       and .recovery_option_a.only_functional_repair ==
          "exact whitelist of one fixed bootstrap block before project Tcl"
       and .recovery_option_a.all_three_axes_must_rerun == true
       and .authorization.author_additive_r13_all_axis_source_package == true
       and .authorization.run_r13_now == false
       and .authorization.run_dc == false' \
    "${m872_m803_dc_m769}/review.json" >/dev/null || exit 3

# R14 fixes only M774's artifact-completeness P1.  Bind that independent FAIL
# exactly and retain every R13 functional, resource, tool, log and claim gate.
(cd "${m872_m803_dc_m774}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
m872_m803_dc_expect "${m872_m803_dc_m774}/review.json" \
    "$(jq -er '.r14_artifact_completeness_repair_provenance.m774_review_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m774}/SHA256SUMS" \
    "$(jq -er '.r14_artifact_completeness_repair_provenance.m774_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m774}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r14_artifact_completeness_repair_provenance.m774_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.r14_artifact_completeness_repair_provenance.m774_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m774}/review.json" ]] || exit 3
jq -e '.r14_artifact_completeness_repair_provenance.r14_is_additive == true
       and .r14_artifact_completeness_repair_provenance.r13_canonical_absent == true
       and .r14_artifact_completeness_repair_provenance.r13_attempt_absent == true
       and .r14_artifact_completeness_repair_provenance.r14_all_three_axes_rerun == true
       and .r14_artifact_completeness_repair_provenance.r14_reuses_r13_outputs == false
       and .r14_artifact_completeness_repair_provenance.artifact_gate_scope ==
          "per-axis mapped Verilog mapped SDC DDC SVF area QoR and setup timing must each be regular non-symlink nonempty before receipts RUN_COMPLETE or sealing"' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.status == "FAIL_STATIC_HAMMER__MISSING_DDC_COMPLETENESS_GATE__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION"
       and .verdict == "FAIL" and .score_out_of_100 == 96
       and .severity_counts == {"p0":0,"p1":1,"p2":0}
       and (.p1_findings | length) == 1
       and .p1_findings[0].id == "P1_DDC_COMPLETENESS_NOT_FAIL_CLOSED"
       and .authorization.author_may_create_additive_launch_release_now == false
       and .authorization.author_may_run_dc == false
       and .authorization.author_may_create_fresh_additive_source_repair == true
       and .authorization.fresh_source_hammer_required_after_repair == true' \
    "${m872_m803_dc_m774}/review.json" >/dev/null || exit 3

# R15 fixes exactly M780's two artifact-closure P1 findings.  It retains every
# R14 gate and reruns all axes under a new identity; no R14 result is reused.
(cd "${m872_m803_dc_m780}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
m872_m803_dc_expect "${m872_m803_dc_m780}/review.json" \
    "$(jq -er '.r15_atomic_artifact_gate_repair_provenance.m780_review_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m780}/SHA256SUMS" \
    "$(jq -er '.r15_atomic_artifact_gate_repair_provenance.m780_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m780}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r15_atomic_artifact_gate_repair_provenance.m780_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.r15_atomic_artifact_gate_repair_provenance.m780_review_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_m780}/review.json" ]] || exit 3
jq -e '.r15_atomic_artifact_gate_repair_provenance.r15_is_additive == true
       and .r15_atomic_artifact_gate_repair_provenance.r14_canonical_absent == true
       and .r15_atomic_artifact_gate_repair_provenance.r14_attempt_absent == true
       and .r15_atomic_artifact_gate_repair_provenance.r15_all_three_axes_rerun == true
       and .r15_atomic_artifact_gate_repair_provenance.r15_reuses_r14_outputs == false
       and .r15_atomic_artifact_gate_repair_provenance.atomic_publication_scope ==
          "inventory and terminal receipt publish through one checked staging-directory rename; any failure forbids RUN_COMPLETE"
       and .r15_atomic_artifact_gate_repair_provenance.artifact_path_scope ==
          "artifact leaves and every ancestor must be nonsymlink and resolve beneath the exact axis directory"
       and .r15_atomic_artifact_gate_repair_provenance.artifact_manifest_scope ==
          "after receipt publication and again after final sealing, seven live artifacts must match both receipts and occur exactly once with the same SHA in the enclosing manifest"
       and .r15_atomic_artifact_gate_repair_provenance.m780_p1_ids == [
          "P1_NONATOMIC_DUAL_RECEIPT_PUBLICATION_CAN_FALSE_PASS",
          "P1_ANCESTOR_SYMLINK_AND_POST_RECEIPT_TOCTOU_BYTES_ESCAPE_SEAL_CLOSURE"
       ]' "${m872_m803_dc_admission}" >/dev/null || exit 3
jq -e '.status == "FAIL_STATIC_HAMMER__NONATOMIC_ARTIFACT_PUBLICATION_AND_UNSEALED_ANCESTOR_PATH__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION"
       and .verdict == "FAIL" and .score_out_of_100 == 90
       and .severity_counts == {"p0":0,"p1":2,"p2":0}
       and (.p1_findings | length) == 2
       and .p1_findings[0].id == "P1_NONATOMIC_DUAL_RECEIPT_PUBLICATION_CAN_FALSE_PASS"
       and .p1_findings[1].id == "P1_ANCESTOR_SYMLINK_AND_POST_RECEIPT_TOCTOU_BYTES_ESCAPE_SEAL_CLOSURE"
       and .authorization.author_may_create_launch_release_now == false
       and .authorization.author_may_run_dc == false
       and .authorization.author_may_create_fresh_additive_source_repair == true
       and .authorization.fresh_source_hammer_required_after_repair == true' \
    "${m872_m803_dc_m780}/review.json" >/dev/null || exit 3

# The exact clean environment is a closed contract, not an implicit shell
# inheritance.  Byte-check both the local license file and lmutil without
# contacting the server; live server/feature queries happen later, only after
# resource preflight and still before attempt consumption.
[[ "${SNPSLMD_LICENSE_FILE:-}" == "${m872_m803_dc_snpslmd_license_file}" && \
   "${LM_LICENSE_FILE:-}" == "${m872_m803_dc_lm_license_file}" ]] || {
    echo "M872 M803 DC exact license environment is required" >&2
    exit 3
}
[[ ! -v HOME ]] || {
    echo "M872 M803 DC requires HOME to remain absent; synthesizing or inheriting HOME is forbidden" >&2
    exit 3
}
m872_m803_dc_expect "${m872_m803_dc_license_file}" "${m872_m803_dc_license_file_sha256}"
m872_m803_dc_expect "${m872_m803_dc_lmutil}" "${m872_m803_dc_lmutil_sha256}"
jq -e --arg snps "${m872_m803_dc_snpslmd_license_file}" \
       --arg lm "${m872_m803_dc_lm_license_file}" \
       --arg file "${m872_m803_dc_license_file}" \
       --arg file_sha "${m872_m803_dc_license_file_sha256}" \
       --arg lmutil "${m872_m803_dc_lmutil}" \
       --arg lmutil_sha "${m872_m803_dc_lmutil_sha256}" \
       '.license_environment.snpslmd_license_file == $snps
       and .license_environment.lm_license_file == $lm
       and .license_environment.snps_license_file_path == $file
       and .license_environment.snps_license_file_sha256 == $file_sha
       and .license_environment.lmutil_path == $lmutil
       and .license_environment.lmutil_sha256 == $lmutil_sha
       and .license_environment.design_compiler_feature == "Design-Compiler"
       and .license_environment.dc_ultra_feature == "DC-Ultra"' \
    "${m872_m803_dc_admission}" >/dev/null || exit 3
export SNPSLMD_LICENSE_FILE="${m872_m803_dc_snpslmd_license_file}"
export LM_LICENSE_FILE="${m872_m803_dc_lm_license_file}"
# Bind the exact independently sealed M576 status; a paraphrase is not admissible.
m872_m803_dc_m576=reviews/m576_m519_r8_dc_launch_admission_candidate_hammer_r1_20260828
(cd "${m872_m803_dc_m576}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
m872_m803_dc_expect "${m872_m803_dc_m576}/review.json" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m576}/SHA256SUMS" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_manifest_file_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_expect "${m872_m803_dc_m576}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_outer_seal_file_sha256' "${m872_m803_dc_admission}")"
[[ "$(jq -er '.fresh_successor_provenance.candidate_hammer_status' "${m872_m803_dc_admission}")" == \
   "$(jq -er '.status' "${m872_m803_dc_m576}/review.json")" ]] || exit 3
jq -e '.verdict == "PASS" and .score_out_of_100 == 100
       and .severity_counts == {"p0":0,"p1":0,"p2":0}' \
    "${m872_m803_dc_m576}/review.json" >/dev/null || exit 3
m872_m803_dc_closed_keys "${m872_m803_dc_admission}" '.identity' \
    'dc_actual_exec_path,dc_actual_exec_sha256,dc_filelist_path,dc_filelist_sha256,dc_runner_path,dc_runner_sha256,dc_shell_path,dc_shell_sha256,dc_tcl_path,dc_tcl_sha256,dc_wrapper_path,dc_wrapper_sha256,docs359_path,docs359_sha256,fast_lib_path,fast_lib_sha256,lmutil_path,lmutil_sha256,r5_final_failure_review_outer_seal_file_sha256,r5_final_failure_review_path,r5_quarantine_outer_seal_file_sha256,r5_quarantine_path,r5_static_review_outer_seal_file_sha256,r5_static_review_path,r5_vcs_result_outer_seal_file_sha256,r5_vcs_result_path,r5_vcs_review_outer_seal_file_sha256,r5_vcs_review_path,r6_static_review_outer_seal_file_sha256,r6_static_review_path,r7_disqualified_review_outer_seal_file_sha256,r7_disqualified_review_path,recovery_contract_path,recovery_contract_sha256,sdc_path,sdc_sha256,slow_lib_path,slow_lib_sha256,snps_license_file_path,snps_license_file_sha256'
for key in $(jq -r '.identity | keys[]' "${m872_m803_dc_admission}"); do
    value="$(jq -er ".identity.${key}" "${m872_m803_dc_admission}")"
    case "${key}" in
        *_sha256) [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3 ;;
        *_path) [[ -n "${value}" && "${value}" != *$'\n'* && \
                    "${value}" != *$'\t'* ]] || exit 3 ;;
        *) exit 3 ;;
    esac
done
[[ "$(jq -er '.identity.recovery_contract_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_contract}" ]] || exit 3
m872_m803_dc_expect "${m872_m803_dc_contract}" \
    "$(jq -er '.identity.recovery_contract_sha256' "${m872_m803_dc_admission}")"
m872_m803_dc_verify_double_seal_file "${m872_m803_dc_contract}"
m872_m803_dc_strict_json "${m872_m803_dc_contract}" || exit 3
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r10_repair_provenance' \
    "${m872_m803_dc_contract}" '.r10_repair_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r11_repair_provenance' \
    "${m872_m803_dc_contract}" '.r11_repair_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r12_license_recovery_provenance' \
    "${m872_m803_dc_contract}" '.r12_license_recovery_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r13_bootstrap_log_recovery_provenance' \
    "${m872_m803_dc_contract}" '.r13_bootstrap_log_recovery_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r14_artifact_completeness_repair_provenance' \
    "${m872_m803_dc_contract}" '.r14_artifact_completeness_repair_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.r15_atomic_artifact_gate_repair_provenance' \
    "${m872_m803_dc_contract}" '.r15_atomic_artifact_gate_repair_provenance'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.license_environment' \
    "${m872_m803_dc_contract}" '.license_environment'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.m800_failure_authority' \
    "${m872_m803_dc_contract}" '.m800_failure_authority'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.m803_vcs_authority' \
    "${m872_m803_dc_contract}" '.m803_vcs_authority'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.three_axis_pre_attempt_plan' \
    "${m872_m803_dc_contract}" '.three_axis_pre_attempt_plan'

jq -e '.status == "AUTHOR_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_ONLY_COMPLETE__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"
       and .authorization.author_ran_eda == false
       and .authorization.run_dc_now == false
       and .authorization.run_vcs_now == false
       and .authorization.run_pt_now == false
       and .authorization.run_ptpx_now == false
       and .authorization.run_formality_now == false
       and .authorization.run_remote_now == false' \
    "${m872_m803_dc_contract}" >/dev/null || exit 3

m872_m803_dc_expected_exact_paths=(
    dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh
    dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl
    dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f
    dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
    rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv
    rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv
    rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv
    rtl_m218/m218_fc2_tagged_slice_service_island.sv
    rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv
    rtl_m519/m519_fc2_k1_registered_release_service_island.sv
    rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv
    rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv
    rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv
    rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv
    rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv
    rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv
    docs/359_DATE终局冻结_20260813.md
)
m872_m803_dc_actual_exact_paths="$(jq -r '.exact_files | keys[]' \
    "${m872_m803_dc_contract}" | LC_ALL=C sort | paste -sd, -)"
m872_m803_dc_expected_exact_csv="$(printf '%s\n' "${m872_m803_dc_expected_exact_paths[@]}" | \
    LC_ALL=C sort | paste -sd, -)"
[[ "${m872_m803_dc_actual_exact_paths}" == "${m872_m803_dc_expected_exact_csv}" ]] || {
    echo "M872 M803 DC contract exact_files has unknown or missing path" >&2
    exit 3
}
: > /tmp/m872_m803_dc_exact_verified.$$.tsv
while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m872_m803_dc_expect "${path}" "${expected}"
    printf '%s\t%s\n' "${path}" "${expected}" \
        >>/tmp/m872_m803_dc_exact_verified.$$.tsv
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' \
    "${m872_m803_dc_contract}")

# Cross-check every future admission path and SHA against the frozen contract.
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_runner_path' \
    "${m872_m803_dc_contract}" '.setup_area_flow.runner'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_runner_sha256' \
    "${m872_m803_dc_contract}" '.setup_area_flow.runner_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_tcl_path' \
    "${m872_m803_dc_contract}" '.setup_area_flow.tcl'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_tcl_sha256' \
    "${m872_m803_dc_contract}" '.setup_area_flow.tcl_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_filelist_path' \
    "${m872_m803_dc_contract}" '.setup_area_flow.filelist'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_filelist_sha256' \
    "${m872_m803_dc_contract}" '.setup_area_flow.filelist_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.sdc_path' \
    "${m872_m803_dc_contract}" '.setup_area_flow.sdc'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.sdc_sha256' \
    "${m872_m803_dc_contract}" '.setup_area_flow.sdc_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_shell_path' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_shell_sha256' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_wrapper_path' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell_wrapper'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_wrapper_sha256' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell_wrapper_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_actual_exec_path' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell_actual_executable'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.dc_actual_exec_sha256' \
    "${m872_m803_dc_contract}" '.tool_identity.dc_shell_actual_executable_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.slow_lib_path' \
    "${m872_m803_dc_contract}" '.tool_identity.slow_library'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.slow_lib_sha256' \
    "${m872_m803_dc_contract}" '.tool_identity.slow_library_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.fast_lib_path' \
    "${m872_m803_dc_contract}" '.tool_identity.fast_library'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.fast_lib_sha256' \
    "${m872_m803_dc_contract}" '.tool_identity.fast_library_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.snps_license_file_path' \
    "${m872_m803_dc_contract}" '.license_environment.snps_license_file_path'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.snps_license_file_sha256' \
    "${m872_m803_dc_contract}" '.license_environment.snps_license_file_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.lmutil_path' \
    "${m872_m803_dc_contract}" '.license_environment.lmutil_path'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.lmutil_sha256' \
    "${m872_m803_dc_contract}" '.license_environment.lmutil_sha256'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.docs359_path' \
    "${m872_m803_dc_contract}" '.frozen_docs.path'
m872_m803_dc_json_equal "${m872_m803_dc_admission}" '.identity.docs359_sha256' \
    "${m872_m803_dc_contract}" '.docs359_sha256'
for stem in r5_static_review r5_vcs_result r5_vcs_review \
        r5_final_failure_review r5_quarantine; do
    m872_m803_dc_json_equal "${m872_m803_dc_admission}" ".identity.${stem}_path" \
        "${m872_m803_dc_contract}" ".sealed_basis.${stem}"
    m872_m803_dc_json_equal "${m872_m803_dc_admission}" \
        ".identity.${stem}_outer_seal_file_sha256" \
        "${m872_m803_dc_contract}" \
        ".sealed_basis.${stem}_outer_seal_file_sha256"
done
for stem in r6_static_review r7_disqualified_review; do
    m872_m803_dc_json_equal "${m872_m803_dc_admission}" ".identity.${stem}_path" \
        "${m872_m803_dc_contract}" ".sealed_basis.${stem}"
    m872_m803_dc_json_equal "${m872_m803_dc_admission}" \
        ".identity.${stem}_outer_seal_file_sha256" \
        "${m872_m803_dc_contract}" \
        ".sealed_basis.${stem}_outer_seal_file_sha256"
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m872_m803_dc_admission}")" == \
   "${M872_M803_DC_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.identity.dc_runner_path' "${m872_m803_dc_admission}")" == \
   dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh ]] || exit 3
[[ "$(jq -er '.identity.dc_tcl_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_tcl}" && \
   "$(jq -er '.identity.dc_filelist_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_filelist}" && \
   "$(jq -er '.identity.sdc_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_sdc}" && \
   "$(jq -er '.identity.dc_shell_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_dc}" && \
   "$(jq -er '.identity.dc_wrapper_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_dc_wrapper}" && \
   "$(jq -er '.identity.dc_actual_exec_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_dc_actual_exe}" && \
   "$(jq -er '.identity.slow_lib_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_slow}" && \
   "$(jq -er '.identity.fast_lib_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_fast}" && \
   "$(jq -er '.identity.snps_license_file_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_license_file}" && \
   "$(jq -er '.identity.lmutil_path' "${m872_m803_dc_admission}")" == \
   "${m872_m803_dc_lmutil}" ]] || exit 3

# Launch-time byte closure covers the symlinked entry, its wrapper, the actual
# long-lived common_shell executable, both timing libraries, and all workspace
# exact_files checked above.  Contract/admission string equality is never used
# as a substitute for checking the current bytes.
[[ "$(realpath "${m872_m803_dc_dc}")" == "${m872_m803_dc_dc_wrapper}" ]] || exit 3
m872_m803_dc_expect "${m872_m803_dc_dc}" \
    "$(jq -er '.tool_identity.dc_shell_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_dc_wrapper}" \
    "$(jq -er '.tool_identity.dc_shell_wrapper_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_dc_actual_exe}" \
    "$(jq -er '.tool_identity.dc_shell_actual_executable_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_slow}" \
    "$(jq -er '.tool_identity.slow_library_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_fast}" \
    "$(jq -er '.tool_identity.fast_library_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_license_file}" \
    "$(jq -er '.license_environment.snps_license_file_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_lmutil}" \
    "$(jq -er '.license_environment.lmutil_sha256' "${m872_m803_dc_contract}")"

# Bind the exact M800 failure cause and the admitted M803/R25 directed VCS E3
# result.  These authorities establish why all axes must rerun from zero and
# why ARCH_MODE=1 is the channel-split candidate; neither authority supplies
# DC area, timing, power, energy, PPA, or system evidence.
for sealed in "${m872_m803_dc_m800}" "${m872_m803_dc_r25_result}" \
        "${m872_m803_dc_m867}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m872_m803_dc_verify_double_seal_file "${m872_m803_dc_r25_release}"
for json_file in \
        "${m872_m803_dc_m800}/review.json" \
        "${m872_m803_dc_r25_release}" \
        "${m872_m803_dc_r25_result}/m859_c2_r25_shared_whitelist_vcs_receipt_r1.json" \
        "${m872_m803_dc_m867}/review.json"; do
    m872_m803_dc_strict_json "${json_file}" || exit 3
done
m872_m803_dc_expect "${m872_m803_dc_m800}/review.json" \
    "$(jq -er '.m800_failure_authority.review_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_m800}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.m800_failure_authority.outer_seal_file_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_r25_release}" \
    "$(jq -er '.m803_vcs_authority.r25_release_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_r25_release}.sha256.seal.sha256" \
    "$(jq -er '.m803_vcs_authority.r25_release_outer_seal_file_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_r25_result}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.m803_vcs_authority.r25_result_outer_seal_file_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_m867}/review.json" \
    "$(jq -er '.m803_vcs_authority.m867_review_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_m867}/SHA256SUMS" \
    "$(jq -er '.m803_vcs_authority.m867_manifest_sha256' "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_m867}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.m803_vcs_authority.m867_outer_seal_file_sha256' "${m872_m803_dc_contract}")"
jq -e '.status == "PASS_FAILURE_AUDIT__M519_R15_K8_TIM209__THREE_AXIS_CAMPAIGN_NONCITABLE__ADDITIVE_R16_SOURCE_ONLY_AUTHORIZED"
       and .campaign_outcome.three_axis_campaign_citable == false
       and .k8_failure.tim209 == 1
       and .k8_failure.opt150 == 0
       and .decision.r15_attempt_status == "CONSUMED_NO_RETRY"' \
    "${m872_m803_dc_m800}/review.json" >/dev/null || exit 3
jq -e '.status == "PASS100_M859_R25_DIRECTED_COMPONENT_VCS_E3_RESULT_ADMITTED"
       and .p0_count == 0 and .p1_count == 0 and .p2_count == 0
       and .vcs_evidence.equal_bandwidth.exact_cycles.k8 == [51,131,486,1231,14]
       and .vcs_evidence.equal_bandwidth.exact_cycles.k1x8 == [53,133,499,1246,14]
       and .claim_boundary.dc_or_physical_ppa == false
       and .claim_boundary.system_speedup == false' \
    "${m872_m803_dc_m867}/review.json" >/dev/null || exit 3

for sealed in "${m872_m803_dc_r5_static}" "${m872_m803_dc_r5_vcs}" \
        "${m872_m803_dc_r5_vcs_review}" "${m872_m803_dc_r5_failure}" \
        "${m872_m803_dc_r5_quarantine}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done

# Each sealed basis must both self-verify and equal the outer-seal file SHA
# independently frozen in the contract and future admission.
for stem in r5_static_review r5_vcs_result r5_vcs_review \
        r5_final_failure_review r5_quarantine; do
    basis_path="$(jq -er ".sealed_basis.${stem}" "${m872_m803_dc_contract}")"
    basis_sha="$(jq -er ".sealed_basis.${stem}_outer_seal_file_sha256" \
        "${m872_m803_dc_contract}")"
    m872_m803_dc_expect "${basis_path}/SHA256SUMS.seal.sha256" "${basis_sha}"
done
for sealed in "${m872_m803_dc_r6_failed_review}" \
        "${m872_m803_dc_r7_disqualified_review}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m872_m803_dc_expect "${m872_m803_dc_r6_failed_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.sealed_basis.r6_static_review_outer_seal_file_sha256' \
        "${m872_m803_dc_contract}")"
m872_m803_dc_expect "${m872_m803_dc_r7_disqualified_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.sealed_basis.r7_disqualified_review_outer_seal_file_sha256' \
        "${m872_m803_dc_contract}")"
jq -e '.status == "FAIL_STATIC_HAMMER__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION"
       and .severity_counts.p0 == 0 and .severity_counts.p1 == 3' \
    "${m872_m803_dc_r6_failed_review}/m538_m519_r6_setup_area_flow_static_hammer_verdict_r1.json" \
    >/dev/null || exit 3
jq -e '.status == "DISQUALIFIED_REVIEWER_TOOL_INVOCATION__R7_SOURCE_BLOCKED__NO_LAUNCH_ADMISSION"
       and .severity_counts.p0 == 2 and .severity_counts.p1 == 2
       and .review_protocol.reviewer_eligible_for_launch_admission == false
       and .review_protocol.accidental_dc_executable_invocations == 1' \
    "${m872_m803_dc_r7_disqualified_review}/review.json" >/dev/null || exit 3

# Before any resource, license, attempt, or tool action, prove that the frozen
# three-axis source plan is the one reviewed here and that the Tcl performs
# analyze/elaborate/check_timing/TIM-209+OPT-150 gating before compile_ultra.
m872_m803_dc_verify_three_axis_source_plan() {
    local analyze_line elaborate_line design_line timing_line gate_line compile_line
    local -a expected_rtl=()
    local -a actual_rtl=()
    expected_rtl=(
        rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv
        rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv
        rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv
        rtl_m218/m218_fc2_tagged_slice_service_island.sv
        rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv
        rtl_m519/m519_fc2_k1_registered_release_service_island.sv
        rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv
        rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv
        rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv
        rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv
        rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv
        rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv
    )
    mapfile -t actual_rtl < <(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' \
        "${m872_m803_dc_filelist}")
    [[ "${#actual_rtl[@]}" -eq "${#expected_rtl[@]}" ]] || return 1
    [[ "$(printf '%s\n' "${actual_rtl[@]}")" == \
       "$(printf '%s\n' "${expected_rtl[@]}")" ]] || return 1
    [[ "$(printf '%s\n' "${actual_rtl[@]}" | LC_ALL=C sort -u | wc -l)" -eq \
       "${#expected_rtl[@]}" ]] || return 1
    for path in "${actual_rtl[@]}"; do
        [[ -f "${path}" && ! -L "${path}" ]] || return 1
    done
    analyze_line="$(grep -n '^analyze -format sverilog' "${m872_m803_dc_tcl}" | cut -d: -f1)"
    elaborate_line="$(grep -n '^[[:space:]]*elaborate \$design_name' "${m872_m803_dc_tcl}" | head -1 | cut -d: -f1)"
    design_line="$(grep -n '^check_design > .*check_design_precompile' "${m872_m803_dc_tcl}" | cut -d: -f1)"
    timing_line="$(grep -n '^redirect \$precompile_timing_report' "${m872_m803_dc_tcl}" | cut -d: -f1)"
    gate_line="$(grep -n '^if {\$precompile_tim209_count != 0 || \$precompile_opt150_count != 0}' "${m872_m803_dc_tcl}" | cut -d: -f1)"
    compile_line="$(grep -n '^[[:space:]]*compile_ultra$' "${m872_m803_dc_tcl}" | cut -d: -f1)"
    for value in "${analyze_line}" "${elaborate_line}" "${design_line}" \
            "${timing_line}" "${gate_line}" "${compile_line}"; do
        [[ "${value}" =~ ^[0-9]+$ ]] || return 1
    done
    [[ "${analyze_line}" -lt "${elaborate_line}" && \
       "${elaborate_line}" -lt "${design_line}" && \
       "${design_line}" -lt "${timing_line}" && \
       "${timing_line}" -lt "${gate_line}" && \
       "${gate_line}" -lt "${compile_line}" ]] || return 1
    jq -e '.setup_area_flow.point_order ==
              [{"id":"k1","arch_mode":0},{"id":"k8","arch_mode":1},{"id":"k1x8","arch_mode":2}]
           and .three_axis_pre_attempt_plan.tim209_required_each_axis == 0
           and .three_axis_pre_attempt_plan.opt150_required_each_axis == 0
           and .three_axis_pre_attempt_plan.all_three_axes_same_attempt_required == true' \
        "${m872_m803_dc_contract}" >/dev/null || return 1
}
m872_m803_dc_verify_three_axis_source_plan || exit 3

if [[ -n "${M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST:-}" ]]; then
    printf '%s\n' \
        'status=PASS_M872_M803_DC_FULL_ADMISSION_CONTRACT_PATH_NO_EDA' \
        'admission_launch_now=false' \
        'preflight_started=false' \
        'attempt_consumed=false' \
        'dc_shell_started=false' \
        'three_axis_source_plan=PASS_K1_M803K8_K1X8_TIM209_OPT150_PRECOMPILE_GATE' \
        >"${M872_M803_DC_FULL_PATH_SELF_TEST_ROOT}/FULL_PATH_PASS.txt"
    rm -f /tmp/m872_m803_dc_exact_verified.$$.tsv
    trap - EXIT
    exit 0
fi

m872_m803_dc_proc_identity() {
    local pid=$1 raw rest uid exe
    local -a fields
    [[ "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/stat" ]] || return 1
    IFS= read -r raw <"/proc/${pid}/stat" || return 1
    [[ "${raw}" == *") "* ]] || return 1
    rest="${raw##*) }"
    read -r -a fields <<<"${rest}"
    [[ "${#fields[@]}" -ge 20 ]] || return 1
    uid="$(stat -Lc %u "/proc/${pid}" 2>/dev/null)" || return 1
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null)" || exe=UNREADABLE
    M872_M803_DC_PROC_PID=${pid}
    M872_M803_DC_PROC_STATE=${fields[0]}
    M872_M803_DC_PROC_PPID=${fields[1]}
    M872_M803_DC_PROC_STARTTIME=${fields[19]}
    M872_M803_DC_PROC_UID=${uid}
    M872_M803_DC_PROC_EXE=${exe}
    M872_M803_DC_PROC_COMM_HEX="$(od -An -tx1 -v "/proc/${pid}/comm" \
        2>/dev/null | tr -d ' \n')"
    M872_M803_DC_PROC_EXE_HEX="$(printf '%s' "${exe}" | od -An -tx1 -v | tr -d ' \n')"
    M872_M803_DC_PROC_CMDLINE_NUL_HEX="$(od -An -tx1 -v "/proc/${pid}/cmdline" \
        2>/dev/null | tr -d ' \n')"
    return 0
}

# Return 0 only for the exact live tuple, 1 if absent/completed zombie, and 2
# for PID reuse or any birth identity mismatch.  Optional parent and complete
# NUL-safe cmdline pins extend the tuple.  Callers never signal return-2.
m872_m803_dc_root_state() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=${5:-} cmdline_hex=${6:-}
    [[ -e "/proc/${pid}" ]] || return 1
    m872_m803_dc_proc_identity "${pid}" || return 2
    [[ "${M872_M803_DC_PROC_STARTTIME}" == "${start}" && \
       "${M872_M803_DC_PROC_UID}" == "${uid}" ]] || return 2
    [[ -z "${parent}" || "${M872_M803_DC_PROC_PPID}" == "${parent}" ]] || return 2
    [[ "${M872_M803_DC_PROC_STATE}" != Z ]] || return 1
    [[ "${M872_M803_DC_PROC_EXE}" == "${exe}" ]] || return 2
    [[ -z "${cmdline_hex}" || \
       "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" == "${cmdline_hex}" ]] || return 2
    return 0
}

# Every ancestor is represented by a (pid,starttime) pair, then reread before
# accepting the chain.  This closes intermediate as well as root PID reuse.
m872_m803_dc_pid_is_descendant() {
    local pid=$1 candidate_start=$2 root=$3 root_start=$4
    local guard=0 index current_start parent
    local -a chain_pid=() chain_start=()
    while [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 && \
             "${guard}" -lt 64 ]]; do
        m872_m803_dc_proc_identity "${pid}" || return 2
        current_start=${M872_M803_DC_PROC_STARTTIME}
        [[ "${guard}" -ne 0 || "${current_start}" == "${candidate_start}" ]] \
            || return 2
        chain_pid+=("${pid}"); chain_start+=("${current_start}")
        if [[ "${pid}" -eq "${root}" ]]; then
            [[ "${current_start}" == "${root_start}" ]] || return 2
            for index in "${!chain_pid[@]}"; do
                m872_m803_dc_proc_identity "${chain_pid[${index}]}" || return 2
                [[ "${M872_M803_DC_PROC_STARTTIME}" == \
                   "${chain_start[${index}]}" ]] || return 2
            done
            return 0
        fi
        parent=${M872_M803_DC_PROC_PPID}
        [[ "${parent}" =~ ^[0-9]+$ && "${parent}" -ne "${pid}" ]] || return 2
        pid=${parent}; guard=$((guard + 1))
    done
    return 1
}

m872_m803_dc_external_eda_pids() {
    local root=${1:-} root_start=${2:-} root_uid=${3:-} root_exe=${4:-}
    local root_parent=${5:-} root_cmdline=${6:-} collision_log=$7 label=$8
    local proc pid comm exe_base first=1 state=1 candidate_start rc kind
    local saved_ppid saved_uid saved_start saved_state saved_comm_hex
    local saved_exe_hex saved_cmdline_hex
    if [[ ! -e "${collision_log}" ]]; then
        printf 'timestamp\tlabel\tkind\tpid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
            >"${collision_log}"
    fi
    if [[ -n "${root}" ]]; then
        set +e; m872_m803_dc_root_state "${root}" "${root_start}" \
            "${root_uid}" "${root_exe}" "${root_parent}" \
            "${root_cmdline}"; state=$?; set -e
        if [[ "${state}" -eq 2 ]]; then
            printf 'campaign_root_identity_mismatch:%s' "${root}"
            first=0
            if m872_m803_dc_proc_identity "${root}"; then
                printf '%s\t%s\tcampaign_root_identity_mismatch\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date --iso-8601=seconds)" "${label}" \
                    "${M872_M803_DC_PROC_PID}" "${M872_M803_DC_PROC_PPID}" \
                    "${M872_M803_DC_PROC_UID}" "${M872_M803_DC_PROC_STARTTIME}" \
                    "${M872_M803_DC_PROC_STATE}" "${M872_M803_DC_PROC_COMM_HEX}" \
                    "${M872_M803_DC_PROC_EXE_HEX}" \
                    "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" >>"${collision_log}"
            fi
        fi
    fi
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m872_m803_dc_proc_identity "${pid}" || continue
        [[ "${M872_M803_DC_PROC_UID}" == "${m872_m803_dc_uid}" && \
           "${M872_M803_DC_PROC_STATE}" != Z ]] || continue
        IFS= read -r comm <"/proc/${pid}/comm" 2>/dev/null || continue
        exe_base=${M872_M803_DC_PROC_EXE##*/}
        case "${comm}:${exe_base}" in
            dc_shell:*|dc_shell-t:*|fm_shell:*|pt_shell:*|vcs:*|vcs1:*|vlogan:*|simv:*|common_shell_ex*:common_shell_exec)
                ;;
            *) continue ;;
        esac
        candidate_start=${M872_M803_DC_PROC_STARTTIME}
        saved_ppid=${M872_M803_DC_PROC_PPID}
        saved_uid=${M872_M803_DC_PROC_UID}
        saved_start=${M872_M803_DC_PROC_STARTTIME}
        saved_state=${M872_M803_DC_PROC_STATE}
        saved_comm_hex=${M872_M803_DC_PROC_COMM_HEX}
        saved_exe_hex=${M872_M803_DC_PROC_EXE_HEX}
        saved_cmdline_hex=${M872_M803_DC_PROC_CMDLINE_NUL_HEX}
        kind=external_eda_collision
        if [[ "${state}" -eq 0 ]]; then
            set +e
            m872_m803_dc_pid_is_descendant "${pid}" "${candidate_start}" \
                "${root}" "${root_start}"
            rc=$?
            set -e
            [[ "${rc}" -ne 0 ]] || continue
            [[ "${rc}" -ne 2 ]] || kind=ancestry_identity_mismatch
        fi
        # Reread immediately before emitting the independently reconstructable
        # collision tuple; PID reuse becomes explicit mismatch evidence.
        if ! m872_m803_dc_proc_identity "${pid}" || \
                [[ "${M872_M803_DC_PROC_STARTTIME}" != "${candidate_start}" ]]; then
            kind=collision_identity_changed_before_record
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date --iso-8601=seconds)" "${label}" "${kind}" "${pid}" \
                "${saved_ppid}" "${saved_uid}" "${saved_start}" \
                "${saved_state}" "${saved_comm_hex}" "${saved_exe_hex}" \
                "${saved_cmdline_hex}" >>"${collision_log}"
            [[ "${first}" -eq 1 ]] || printf ','
            printf '%s:%s' "${kind}" "${pid}"
            first=0
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${kind}" \
            "${M872_M803_DC_PROC_PID}" "${M872_M803_DC_PROC_PPID}" \
            "${M872_M803_DC_PROC_UID}" "${M872_M803_DC_PROC_STARTTIME}" \
            "${M872_M803_DC_PROC_STATE}" "${M872_M803_DC_PROC_COMM_HEX}" \
            "${M872_M803_DC_PROC_EXE_HEX}" "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" \
            >>"${collision_log}"
        [[ "${first}" -eq 1 ]] || printf ','
        printf '%s:%s:%s' "${kind}" "${pid}" "${candidate_start}"
        first=0
    done
}

m872_m803_dc_read_cgroup() {
    M872_M803_DC_CGROUP_FAILCNT="$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)"
    M872_M803_DC_CGROUP_UNDER_OOM="$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    M872_M803_DC_CGROUP_OOM_KILL="$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
}

m872_m803_dc_resource_snapshot() {
    local label=$1 log=$2 h0=${3:-NA} root=${4:-}
    local root_start=${5:-} root_uid=${6:-} root_exe=${7:-}
    local root_parent=${8:-} root_cmdline=${9:-}
    local limit committed delta
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    M872_M803_DC_HEADROOM_KIB=$((limit - committed))
    M872_M803_DC_MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    M872_M803_DC_SWAP_FREE_KIB="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    m872_m803_dc_read_cgroup
    M872_M803_DC_COLLISION="$(m872_m803_dc_external_eda_pids "${root}" \
        "${root_start}" "${root_uid}" "${root_exe}" "${root_parent}" \
        "${root_cmdline}" "${log%.log}_external_collisions.tsv" "${label}")"
    M872_M803_DC_IDENTITY_MISMATCH=0
    [[ "${M872_M803_DC_COLLISION}" != *identity_mismatch* ]] || \
        M872_M803_DC_IDENTITY_MISMATCH=1
    if [[ "${h0}" =~ ^[0-9]+$ ]]; then
        delta=$((h0 - M872_M803_DC_HEADROOM_KIB))
    else
        delta=NA
    fi
    printf 'timestamp=%s label=%s h0_commit_headroom_kib=%s commit_headroom_kib=%s h0_minus_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s external_eda_collision=%s campaign_identity_mismatch=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${h0}" \
        "${M872_M803_DC_HEADROOM_KIB}" "${delta}" \
        "${M872_M803_DC_MEM_AVAILABLE_KIB}" "${M872_M803_DC_SWAP_FREE_KIB}" \
        "${M872_M803_DC_CGROUP_FAILCNT}" "${M872_M803_DC_CGROUP_UNDER_OOM}" \
        "${M872_M803_DC_CGROUP_OOM_KILL}" "${M872_M803_DC_COLLISION:-none}" \
        "${M872_M803_DC_IDENTITY_MISMATCH}" >>"${log}"
}

m872_m803_dc_pid_tree_snapshot() {
    local label=$1 log=$2 proc pid
    printf 'timestamp=%s label=%s\n' "$(date --iso-8601=seconds)" \
        "${label}" >>"${log}"
    printf 'pid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
        >>"${log}"
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m872_m803_dc_proc_identity "${pid}" || continue
        [[ "${M872_M803_DC_PROC_UID}" == "${m872_m803_dc_uid}" ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${M872_M803_DC_PROC_PID}" "${M872_M803_DC_PROC_PPID}" \
            "${M872_M803_DC_PROC_UID}" "${M872_M803_DC_PROC_STARTTIME}" \
            "${M872_M803_DC_PROC_STATE}" "${M872_M803_DC_PROC_COMM_HEX}" \
            "${M872_M803_DC_PROC_EXE_HEX}" \
            "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" >>"${log}"
    done
}

m872_m803_dc_seal_dir() {
    local dir=$1
    (
        cd "${dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' -print0 | sort -z | \
            xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}

m872_m803_dc_axis_preflight() {
    local axis=$1 dir=$2 sample pass=1 h0=0
    mkdir -p "${dir}"
    : >"${dir}/resource_preflight.log"
    : >"${dir}/pid_tree_preflight.log"
    for sample in 1 2 3; do
        m872_m803_dc_resource_snapshot "${axis}_preflight_${sample}" \
            "${dir}/resource_preflight.log" NA
        m872_m803_dc_pid_tree_snapshot "${axis}_preflight_${sample}" \
            "${dir}/pid_tree_preflight.log"
        if [[ "${sample}" -eq 1 || "${M872_M803_DC_HEADROOM_KIB}" -lt "${h0}" ]]; then
            h0=${M872_M803_DC_HEADROOM_KIB}
        fi
        if [[ "${M872_M803_DC_HEADROOM_KIB}" -lt "${m872_m803_dc_preflight_commit_kib}" || \
              "${M872_M803_DC_MEM_AVAILABLE_KIB}" -lt "${m872_m803_dc_mem_available_kib}" || \
              "${M872_M803_DC_SWAP_FREE_KIB}" -lt "${m872_m803_dc_swap_free_kib}" || \
              "${M872_M803_DC_CGROUP_FAILCNT}" -ne 0 || \
              "${M872_M803_DC_CGROUP_UNDER_OOM}" -ne 0 || \
              "${M872_M803_DC_CGROUP_OOM_KILL}" -ne 0 || \
              -n "${M872_M803_DC_COLLISION}" ]]; then
            pass=0
        fi
        [[ "${sample}" -eq 3 ]] || sleep 10
    done
    printf 'axis=%s\nh0_commit_headroom_kib=%s\nsamples=3\nsample_interval_seconds=10\ncommit_headroom_gate_kib=%s\nmem_available_gate_kib=%s\nswap_free_gate_kib=%s\ncgroup_required_zero=true\nsame_uid_external_eda_required_none=true\nstatus=%s\n' \
        "${axis}" "${h0}" "${m872_m803_dc_preflight_commit_kib}" \
        "${m872_m803_dc_mem_available_kib}" "${m872_m803_dc_swap_free_kib}" \
        "$([[ "${pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
        >"${dir}/preflight_receipt.txt"
    m872_m803_dc_seal_dir "${dir}"
    [[ "${pass}" -eq 1 ]]
}

# This is a status-only FlexNet query.  It never invokes dc_shell and never
# checks out or reserves a feature.  A successful, parseable service response
# plus at least one currently free seat for both required features is mandatory
# before the unique R12 attempt sentinel may be published.  All raw streams,
# return codes and the parsed receipt are sealed together.
m872_m803_dc_license_feature_parse() {
    local feature=$1 stdout_file=$2 receipt_file=$3
    local summary issued in_use free
    summary="$(grep -F "Users of ${feature}:" "${stdout_file}" | head -n 1 || true)"
    issued="$(printf '%s\n' "${summary}" | sed -n \
        's/.*Total of \([0-9][0-9]*\) licenses issued.*/\1/p')"
    in_use="$(printf '%s\n' "${summary}" | sed -n \
        's/.*Total of \([0-9][0-9]*\) licenses in use.*/\1/p')"
    if [[ ! "${issued}" =~ ^[0-9]+$ || ! "${in_use}" =~ ^[0-9]+$ || \
          "${issued}" -lt "${in_use}" ]]; then
        printf 'feature=%s\nparse_status=FAIL_UNCERTAIN\n' "${feature}" \
            >>"${receipt_file}"
        return 1
    fi
    free=$((issued - in_use))
    printf 'feature=%s\nissued=%s\nin_use=%s\nfree=%s\nparse_status=%s\n' \
        "${feature}" "${issued}" "${in_use}" "${free}" \
        "$([[ "${issued}" -gt 0 && "${free}" -gt 0 ]] && \
            echo PASS_FREE_SEAT_OBSERVED || echo FAIL_NO_FREE_SEAT)" \
        >>"${receipt_file}"
    [[ "${issued}" -gt 0 && "${free}" -gt 0 ]]
}

m872_m803_dc_license_preflight() {
    local dir=$1 pass=1 feature rc
    local -a features=(Design-Compiler DC-Ultra)
    mkdir -p "${dir}"
    printf 'SNPSLMD_LICENSE_FILE=%s\nLM_LICENSE_FILE=%s\nlicense_file=%s\nlicense_file_sha256=%s\nlmutil=%s\nlmutil_sha256=%s\nquery_is_status_only=true\nquery_is_reservation=false\n' \
        "${SNPSLMD_LICENSE_FILE}" "${LM_LICENSE_FILE}" \
        "${m872_m803_dc_license_file}" "$(m872_m803_dc_sha "${m872_m803_dc_license_file}")" \
        "${m872_m803_dc_lmutil}" "$(m872_m803_dc_sha "${m872_m803_dc_lmutil}")" \
        >"${dir}/environment.txt"
    set +e
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
        SNPSLMD_LICENSE_FILE="${m872_m803_dc_snpslmd_license_file}" \
        LM_LICENSE_FILE="${m872_m803_dc_lm_license_file}" \
        "${m872_m803_dc_lmutil}" lmstat -a \
        -c "${m872_m803_dc_snpslmd_license_file}" \
        >"${dir}/server.stdout" 2>"${dir}/server.stderr"
    rc=$?
    set -e
    printf '%s\n' "${rc}" >"${dir}/server.rc"
    if [[ "${rc}" -ne 0 ]] || \
       ! grep -Fq 'License server status:' "${dir}/server.stdout" || \
       grep -Eqi 'cannot connect|connection refused|license server machine is down|no such feature|error' \
           "${dir}/server.stdout" "${dir}/server.stderr"; then
        pass=0
    fi
    : >"${dir}/parsed_receipt.txt"
    printf 'server_query_rc=%s\nserver_status_parse=%s\n' "${rc}" \
        "$([[ "${pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
        >>"${dir}/parsed_receipt.txt"
    for feature in "${features[@]}"; do
        set +e
        env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
            SNPSLMD_LICENSE_FILE="${m872_m803_dc_snpslmd_license_file}" \
            LM_LICENSE_FILE="${m872_m803_dc_lm_license_file}" \
            "${m872_m803_dc_lmutil}" lmstat -f "${feature}" \
            -c "${m872_m803_dc_snpslmd_license_file}" \
            >"${dir}/${feature}.stdout" 2>"${dir}/${feature}.stderr"
        rc=$?
        set -e
        printf '%s\n' "${rc}" >"${dir}/${feature}.rc"
        printf 'feature_query=%s\nquery_rc=%s\n' "${feature}" "${rc}" \
            >>"${dir}/parsed_receipt.txt"
        if [[ "${rc}" -ne 0 ]] || \
           grep -Eqi 'cannot connect|connection refused|license server machine is down|no such feature|error' \
               "${dir}/${feature}.stdout" "${dir}/${feature}.stderr" || \
           ! m872_m803_dc_license_feature_parse "${feature}" \
               "${dir}/${feature}.stdout" "${dir}/parsed_receipt.txt"; then
            pass=0
        fi
    done
    printf 'attempt_consumed=false\nstatus=%s\n' \
        "$([[ "${pass}" -eq 1 ]] && \
            echo PASS_LICENSE_STATUS_OBSERVED_NOT_RESERVED || \
            echo FAIL_LICENSE_PREFLIGHT_NO_ATTEMPT_CONSUMED)" \
        >>"${dir}/parsed_receipt.txt"
    m872_m803_dc_seal_dir "${dir}"
    [[ "${pass}" -eq 1 ]]
}

if ! m872_m803_dc_axis_preflight k1 "${m872_m803_dc_preflight_staging}"; then
    printf 'status=PREFLIGHT_REJECTED_NO_DC_ATTEMPT_CONSUMED\n' \
        >"${m872_m803_dc_preflight_staging}/PREFLIGHT_REJECTED.txt"
    m872_m803_dc_seal_dir "${m872_m803_dc_preflight_staging}"
    mv -T "${m872_m803_dc_preflight_staging}" "${m872_m803_dc_preflight_reject}"
    rm -f /tmp/m872_m803_dc_exact_verified.$$.tsv
    exit 40
fi

if ! m872_m803_dc_license_preflight "${m872_m803_dc_license_preflight_staging}"; then
    mv -T "${m872_m803_dc_license_preflight_staging}" \
        "${m872_m803_dc_license_preflight_reject}"
    rm -f /tmp/m872_m803_dc_exact_verified.$$.tsv
    exit 41
fi

mkdir "${m872_m803_dc_work}"
mkdir "${m872_m803_dc_work}/preflight"
mv -T "${m872_m803_dc_preflight_staging}" "${m872_m803_dc_work}/preflight/k1"
mv -T "${m872_m803_dc_license_preflight_staging}" \
    "${m872_m803_dc_work}/preflight/license"
mv -T /tmp/m872_m803_dc_exact_verified.$$.tsv \
    "${m872_m803_dc_work}/contract_exact_files_verified.tsv"
m872_m803_dc_run_created=1
m872_m803_dc_complete=0
m872_m803_dc_child_pid=""
m872_m803_dc_child_start=""
m872_m803_dc_child_uid=""
m872_m803_dc_child_exe=""
m872_m803_dc_child_parent=""
m872_m803_dc_child_cmdline=""
m872_m803_dc_monitor_pid=""
m872_m803_dc_monitor_start=""
m872_m803_dc_child_rc=not_started
m872_m803_dc_monitor_rc=not_started
m872_m803_dc_signal=none
m872_m803_dc_runtime_latch=0
m872_m803_dc_runtime_latch_reason=none

m872_m803_dc_term_exact() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=$5 cmdline=$6 signal_name=$7 state
    set +e
    m872_m803_dc_root_state "${pid}" "${start}" "${uid}" "${exe}" \
        "${parent}" "${cmdline}"
    state=$?
    set -e
    if [[ "${state}" -eq 0 ]]; then
        kill -s "${signal_name}" "${pid}" 2>/dev/null || return 1
        return 0
    elif [[ "${state}" -eq 1 ]]; then
        return 0
    fi
    return 2
}

m872_m803_dc_signal_handler() {
    local signal_name=$1 term_rc=0
    m872_m803_dc_signal="${signal_name}"
    if [[ -n "${m872_m803_dc_child_pid}" ]]; then
        set +e
        m872_m803_dc_term_exact "${m872_m803_dc_child_pid}" "${m872_m803_dc_child_start}" \
            "${m872_m803_dc_child_uid}" "${m872_m803_dc_child_exe}" \
            "${m872_m803_dc_child_parent}" "${m872_m803_dc_child_cmdline}" \
            "${signal_name}"
        term_rc=$?
        set -e
    fi
    printf 'timestamp=%s signal=%s child_pid=%s child_starttime=%s exact_term_rc=%s monitor_pid=%s monitor_starttime=%s\n' \
        "$(date --iso-8601=seconds)" "${signal_name}" \
        "${m872_m803_dc_child_pid:-none}" "${m872_m803_dc_child_start:-none}" \
        "${term_rc}" "${m872_m803_dc_monitor_pid:-none}" \
        "${m872_m803_dc_monitor_start:-none}" \
        >>"${m872_m803_dc_work}/signal_provenance.txt"
}
trap 'm872_m803_dc_signal_handler INT' INT
trap 'm872_m803_dc_signal_handler TERM' TERM

m872_m803_dc_failure_cleanup() {
    local rc=$? state term_rc=0
    set +e
    if [[ -n "${m872_m803_dc_child_pid}" ]]; then
        m872_m803_dc_root_state "${m872_m803_dc_child_pid}" "${m872_m803_dc_child_start}" \
            "${m872_m803_dc_child_uid}" "${m872_m803_dc_child_exe}" \
            "${m872_m803_dc_child_parent}" "${m872_m803_dc_child_cmdline}"
        state=$?
        if [[ "${state}" -eq 0 ]]; then
            m872_m803_dc_pid_tree_snapshot failure_before_term \
                "${m872_m803_dc_work}/failure_pid_tree.log"
            m872_m803_dc_term_exact "${m872_m803_dc_child_pid}" "${m872_m803_dc_child_start}" \
                "${m872_m803_dc_child_uid}" "${m872_m803_dc_child_exe}" \
                "${m872_m803_dc_child_parent}" "${m872_m803_dc_child_cmdline}" TERM
            term_rc=$?
            wait "${m872_m803_dc_child_pid}"
            m872_m803_dc_child_rc=$?
        elif [[ "${state}" -eq 2 ]]; then
            term_rc=2
        fi
    fi
    if [[ -n "${m872_m803_dc_monitor_pid}" ]]; then
        wait "${m872_m803_dc_monitor_pid}" 2>/dev/null
        m872_m803_dc_monitor_rc=$?
    fi
    if [[ "${m872_m803_dc_run_created}" -eq 1 && \
          "${m872_m803_dc_complete}" -ne 1 && -d "${m872_m803_dc_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nruntime_resource_latch=%s\nruntime_latch_reason=%s\nexact_term_rc=%s\n' \
            "${rc}" "${m872_m803_dc_child_rc}" "${m872_m803_dc_monitor_rc}" \
            "${m872_m803_dc_signal}" "${m872_m803_dc_runtime_latch}" \
            "${m872_m803_dc_runtime_latch_reason}" "${term_rc}" \
            >"${m872_m803_dc_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m872_m803_dc_seal_dir "${m872_m803_dc_work}"
        mv -T "${m872_m803_dc_work}" "${m872_m803_dc_quarantine}"
        m872_m803_dc_run_created=0
    fi
    return "${rc}"
}
trap m872_m803_dc_failure_cleanup EXIT

mkdir "${m872_m803_dc_work}/.attempt_staging"
printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\npreflight_k1_outer_seal_sha256=%s\nlicense_preflight_outer_seal_sha256=%s\nlicense_query_is_reservation=false\n' \
    "$(date --iso-8601=seconds)" "${m872_m803_dc_canonical}" \
    "$(m872_m803_dc_sha "${m872_m803_dc_work}/preflight/k1/SHA256SUMS.seal.sha256")" \
    "$(m872_m803_dc_sha "${m872_m803_dc_work}/preflight/license/SHA256SUMS.seal.sha256")" \
    >"${m872_m803_dc_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m872_m803_dc_runner}" "${m872_m803_dc_contract}" \
    "${m872_m803_dc_admission}" >"${m872_m803_dc_work}/.attempt_staging/identity.sha256"
m872_m803_dc_seal_dir "${m872_m803_dc_work}/.attempt_staging"
mv -T "${m872_m803_dc_work}/.attempt_staging" "${m872_m803_dc_attempt}"
m872_m803_dc_attempt_consumed=1

sha256sum "${m872_m803_dc_runner}" "${m872_m803_dc_contract}" \
    "${m872_m803_dc_admission}" "${m872_m803_dc_tcl}" "${m872_m803_dc_filelist}" \
    "${m872_m803_dc_sdc}" "${m872_m803_dc_dc}" "${m872_m803_dc_dc_wrapper}" \
    "${m872_m803_dc_dc_actual_exe}" "${m872_m803_dc_slow}" "${m872_m803_dc_fast}" \
    "${m872_m803_dc_license_file}" "${m872_m803_dc_lmutil}" \
    "${m872_m803_dc_r5_static}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r5_vcs}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r5_vcs_review}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r5_failure}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r5_quarantine}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r6_failed_review}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r7_disqualified_review}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m694}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m701}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r11_quarantine}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r11_attempt}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m752}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r12_quarantine}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r12_attempt}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m769}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m774}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m780}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m800}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_r25_release}.sha256.seal.sha256" \
    "${m872_m803_dc_r25_result}/SHA256SUMS.seal.sha256" \
    "${m872_m803_dc_m867}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md >"${m872_m803_dc_work}/input_sha256.txt"
cp "${m872_m803_dc_contract}" "${m872_m803_dc_work}/contract.json"

export HW_ROOT="${m872_m803_dc_hw_root}"
export LIB_DB="${m872_m803_dc_slow}"
export MIN_LIB_DB="${m872_m803_dc_fast}"
export SDC_FILE="${m872_m803_dc_hw_root}/${m872_m803_dc_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m872_m803_dc_gate_current_snapshot() {
    local label=$1 point=$2 sample=$3 current_reason=none
    if [[ "${M872_M803_DC_HEADROOM_KIB}" -lt "${m872_m803_dc_runtime_commit_kib}" ]]; then
        M872_M803_DC_COMMIT_BAD_COUNT=$((M872_M803_DC_COMMIT_BAD_COUNT + 1))
    else
        M872_M803_DC_COMMIT_BAD_COUNT=0
    fi
    if [[ "${M872_M803_DC_IDENTITY_MISMATCH}" -ne 0 ]]; then
        current_reason=campaign_pid_identity_mismatch
    elif [[ "${M872_M803_DC_COMMIT_BAD_COUNT}" -ge 3 ]]; then
        current_reason=commit_headroom_below_32gib_for_three_consecutive_samples
    elif [[ "${M872_M803_DC_MEM_AVAILABLE_KIB}" -lt "${m872_m803_dc_mem_available_kib}" ]]; then
        current_reason=mem_available_below_128gib
    elif [[ "${M872_M803_DC_SWAP_FREE_KIB}" -lt "${m872_m803_dc_swap_free_kib}" ]]; then
        current_reason=swap_free_below_32gib
    elif [[ "${M872_M803_DC_CGROUP_FAILCNT}" -ne 0 || \
            "${M872_M803_DC_CGROUP_UNDER_OOM}" -ne 0 || \
            "${M872_M803_DC_CGROUP_OOM_KILL}" -ne 0 ]]; then
        current_reason=cgroup_or_oom_counter_nonzero
    elif [[ -n "${M872_M803_DC_COLLISION}" ]]; then
        current_reason=new_external_same_uid_eda_collision
    fi
    printf 'timestamp=%s label=%s sample=%s commit_bad_consecutive=%s gate_reason=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${sample}" \
        "${M872_M803_DC_COMMIT_BAD_COUNT}" "${current_reason}" \
        >>"${point}/runtime_gate_every_snapshot.log"
    if [[ "${current_reason}" != none ]]; then
        M872_M803_DC_RUNTIME_FAILED=1
        [[ "${M872_M803_DC_RUNTIME_REASON}" != none ]] || \
            M872_M803_DC_RUNTIME_REASON=${current_reason}
        printf 'timestamp=%s status=RUNTIME_RESOURCE_LATCH reason=%s label=%s sample=%s commit_bad_consecutive=%s\n' \
            "$(date --iso-8601=seconds)" "${current_reason}" "${label}" \
            "${sample}" "${M872_M803_DC_COMMIT_BAD_COUNT}" \
            >>"${point}/runtime_latch.txt"
        return 1
    fi
    return 0
}

m872_m803_dc_record_descendants() {
    local child=$1 child_start=$2 sample=$3 point=$4 proc pid rc key candidate_start
    local vmpeak vmsize vmrss vmswap
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m872_m803_dc_proc_identity "${pid}" || continue
        candidate_start=${M872_M803_DC_PROC_STARTTIME}
        set +e
        m872_m803_dc_pid_is_descendant "${pid}" "${candidate_start}" \
            "${child}" "${child_start}"
        rc=$?
        set -e
        [[ "${rc}" -eq 0 ]] || {
            if [[ "${rc}" -eq 2 ]]; then
                printf 'timestamp=%s sample=%s pid=%s status=ANCESTRY_IDENTITY_MISMATCH\n' \
                    "$(date --iso-8601=seconds)" "${sample}" "${pid}" \
                    >>"${point}/descendant_identity_faults.log"
                M872_M803_DC_DESCENDANT_IDENTITY_FAULT=1
            fi
            continue
        }
        # The ancestry checker deliberately overwrites its scratch identity
        # while walking the chain.  Reread and revalidate the candidate tuple
        # before recording its own provenance and memory counters.
        if ! m872_m803_dc_proc_identity "${pid}" || \
                [[ "${M872_M803_DC_PROC_STARTTIME}" != "${candidate_start}" ]]; then
            printf 'timestamp=%s sample=%s pid=%s status=CANDIDATE_IDENTITY_CHANGED_BEFORE_RECORD\n' \
                "$(date --iso-8601=seconds)" "${sample}" "${pid}" \
                >>"${point}/descendant_identity_faults.log"
            M872_M803_DC_DESCENDANT_IDENTITY_FAULT=1
            continue
        fi
        vmpeak="$(awk '/^VmPeak:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmpeak=${vmpeak:-0}
        vmsize="$(awk '/^VmSize:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmsize=${vmsize:-0}
        vmrss="$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmrss=${vmrss:-0}
        vmswap="$(awk '/^VmSwap:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmswap=${vmswap:-0}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${sample}" "${M872_M803_DC_PROC_PID}" \
            "${M872_M803_DC_PROC_PPID}" "${M872_M803_DC_PROC_UID}" \
            "${M872_M803_DC_PROC_STARTTIME}" "${M872_M803_DC_PROC_COMM_HEX}" \
            "${M872_M803_DC_PROC_EXE_HEX}" "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" \
            "${vmpeak}" "${vmsize}" "${vmrss}" "${vmswap}" \
            >>"${point}/descendant_memory_runtime.tsv"
        key="${pid}_${M872_M803_DC_PROC_STARTTIME}"
        M872_M803_DC_HIGH_PID[${key}]=${pid}
        M872_M803_DC_HIGH_START[${key}]=${M872_M803_DC_PROC_STARTTIME}
        M872_M803_DC_HIGH_COMM[${key}]=${M872_M803_DC_PROC_COMM_HEX}
        M872_M803_DC_HIGH_EXE[${key}]=${M872_M803_DC_PROC_EXE_HEX}
        M872_M803_DC_HIGH_CMD[${key}]=${M872_M803_DC_PROC_CMDLINE_NUL_HEX}
        [[ "${vmpeak}" -le "${M872_M803_DC_HIGH_PEAK[${key}]:-0}" ]] || M872_M803_DC_HIGH_PEAK[${key}]=${vmpeak}
        [[ "${vmsize}" -le "${M872_M803_DC_HIGH_SIZE[${key}]:-0}" ]] || M872_M803_DC_HIGH_SIZE[${key}]=${vmsize}
        [[ "${vmrss}" -le "${M872_M803_DC_HIGH_RSS[${key}]:-0}" ]] || M872_M803_DC_HIGH_RSS[${key}]=${vmrss}
        [[ "${vmswap}" -le "${M872_M803_DC_HIGH_SWAP[${key}]:-0}" ]] || M872_M803_DC_HIGH_SWAP[${key}]=${vmswap}
    done
}

m872_m803_dc_runtime_monitor() {
    local child=$1 child_start=$2 child_uid=$3 child_exe=$4
    local child_parent=$5 child_cmdline=$6 h0=$7 point=$8
    local state sample=0 gate_rc=0 key
    M872_M803_DC_RUNTIME_FAILED=0
    M872_M803_DC_RUNTIME_REASON=none
    M872_M803_DC_COMMIT_BAD_COUNT=0
    M872_M803_DC_DESCENDANT_IDENTITY_FAULT=0
    declare -Ag M872_M803_DC_HIGH_PID=() M872_M803_DC_HIGH_START=()
    declare -Ag M872_M803_DC_HIGH_COMM=() M872_M803_DC_HIGH_EXE=() M872_M803_DC_HIGH_CMD=()
    declare -Ag M872_M803_DC_HIGH_PEAK=() M872_M803_DC_HIGH_SIZE=()
    declare -Ag M872_M803_DC_HIGH_RSS=() M872_M803_DC_HIGH_SWAP=()
    : >"${point}/resource_runtime.log"
    : >"${point}/resource_runtime_external_collisions.tsv"
    printf 'timestamp\tlabel\tkind\tpid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
        >"${point}/resource_runtime_external_collisions.tsv"
    : >"${point}/runtime_gate_every_snapshot.log"
    : >"${point}/runtime_latch.txt"
    : >"${point}/descendant_identity_faults.log"
    printf 'timestamp\tsample\tpid\tppid\tuid\tstarttime\tcomm_hex\texe_hex\tcmdline_nul_hex\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_runtime.tsv"

    while true; do
        set +e; m872_m803_dc_root_state "${child}" "${child_start}" \
            "${child_uid}" "${child_exe}" "${child_parent}" \
            "${child_cmdline}"; state=$?; set -e
        [[ "${state}" -eq 0 ]] || break
        sample=$((sample + 1))
        m872_m803_dc_resource_snapshot "runtime_${sample}" \
            "${point}/resource_runtime.log" "${h0}" "${child}" \
            "${child_start}" "${child_uid}" "${child_exe}" \
            "${child_parent}" "${child_cmdline}"
        m872_m803_dc_record_descendants "${child}" "${child_start}" \
            "${sample}" "${point}"
        [[ "${M872_M803_DC_DESCENDANT_IDENTITY_FAULT}" -eq 0 ]] || \
            M872_M803_DC_IDENTITY_MISMATCH=1
        set +e
        m872_m803_dc_gate_current_snapshot "runtime_${sample}" "${point}" "${sample}"
        gate_rc=$?
        set -e
        if [[ "${gate_rc}" -ne 0 ]]; then
            set +e
            m872_m803_dc_term_exact "${child}" "${child_start}" "${child_uid}" \
                "${child_exe}" "${child_parent}" "${child_cmdline}" TERM
            set -e
            break
        fi
        sleep 10
    done

    if [[ "${state}" -eq 2 ]]; then
        M872_M803_DC_RUNTIME_FAILED=1
        M872_M803_DC_RUNTIME_REASON=campaign_pid_identity_mismatch
    fi
    # A latched child must be gone before the synchronous final sample.  The
    # exact tuple is polled; a reused PID is never signalled and is a failure.
    while [[ "${state}" -eq 0 && "${M872_M803_DC_RUNTIME_FAILED}" -ne 0 ]]; do
        sleep 1
        set +e; m872_m803_dc_root_state "${child}" "${child_start}" \
            "${child_uid}" "${child_exe}" "${child_parent}" \
            "${child_cmdline}"; state=$?; set -e
    done
    [[ "${state}" -ne 2 ]] || {
        M872_M803_DC_RUNTIME_FAILED=1
        M872_M803_DC_RUNTIME_REASON=campaign_pid_identity_mismatch
    }

    sample=$((sample + 1))
    m872_m803_dc_resource_snapshot runtime_final \
        "${point}/resource_runtime.log" "${h0}"
    [[ "${state}" -ne 2 ]] || M872_M803_DC_IDENTITY_MISMATCH=1
    set +e
    m872_m803_dc_gate_current_snapshot runtime_final "${point}" "${sample}"
    gate_rc=$?
    set -e

    printf 'pid\tstarttime\tcomm_hex\texe_hex\tcmdline_nul_hex\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_highwater.tsv"
    for key in "${!M872_M803_DC_HIGH_PID[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${M872_M803_DC_HIGH_PID[${key}]}" "${M872_M803_DC_HIGH_START[${key}]}" \
            "${M872_M803_DC_HIGH_COMM[${key}]}" "${M872_M803_DC_HIGH_EXE[${key}]}" \
            "${M872_M803_DC_HIGH_CMD[${key}]}" "${M872_M803_DC_HIGH_PEAK[${key}]:-0}" \
            "${M872_M803_DC_HIGH_SIZE[${key}]:-0}" "${M872_M803_DC_HIGH_RSS[${key}]:-0}" \
            "${M872_M803_DC_HIGH_SWAP[${key}]:-0}"
    done | sort -n >>"${point}/descendant_memory_highwater.tsv"
    printf 'timestamp=%s final_sample_label=runtime_final final_gate_applied=true child_exact_state=%s commit_below_32gib_consecutive_final=%s runtime_resource_latch=%s reason=%s status=%s\n' \
        "$(date --iso-8601=seconds)" "${state}" \
        "${M872_M803_DC_COMMIT_BAD_COUNT}" "${M872_M803_DC_RUNTIME_FAILED}" \
        "${M872_M803_DC_RUNTIME_REASON}" \
        "$([[ "${M872_M803_DC_RUNTIME_FAILED}" -eq 0 && "${gate_rc}" -eq 0 ]] && \
            echo PASS_FINAL_GATE_ACK || echo FAIL_FINAL_GATE_ACK)" \
        >"${point}/runtime_final_gate_ack.txt"
    printf 'runtime_resource_latch=%s\nreason=%s\ncommit_below_32gib_consecutive_final=%s\nfinal_gate_ack_present=true\n' \
        "${M872_M803_DC_RUNTIME_FAILED}" "${M872_M803_DC_RUNTIME_REASON}" \
        "${M872_M803_DC_COMMIT_BAD_COUNT}" >>"${point}/resource_runtime.log"
    [[ "${M872_M803_DC_RUNTIME_FAILED}" -eq 0 && "${gate_rc}" -eq 0 ]]
}

m872_m803_dc_dc_cmdline_matches() {
    local pid
    local exact_tcl
    local -a argv=()
    pid=$1
    exact_tcl="${m872_m803_dc_hw_root}/${m872_m803_dc_tcl}"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 1
    [[ "${#argv[@]}" -eq 7 && \
       "${argv[0]}" == "${m872_m803_dc_dc_actual_exe}" && \
       "${argv[1]}" == -shell && "${argv[2]}" == dc_shell && \
       "${argv[3]}" == -r && "${argv[4]}" == "${m872_m803_dc_dc_install_root}" && \
       "${argv[5]}" == -f && "${argv[6]}" == "${exact_tcl}" ]]
}

# Capture succeeds only after the stable wrapper PID has exec'd into the
# frozen common_shell executable and exposes the exact dc_shell selector,
# install root and R8 Tcl argv.  PID birth, UID and parent must remain unchanged
# throughout the wrapper-to-exec transition.
m872_m803_dc_capture_dc_identity() {
    local pid=$1 tries birth_start= birth_uid= birth_parent=
    for tries in $(seq 1 200); do
        m872_m803_dc_proc_identity "${pid}" || return 1
        if [[ -z "${birth_start}" ]]; then
            birth_start=${M872_M803_DC_PROC_STARTTIME}
            birth_uid=${M872_M803_DC_PROC_UID}
            birth_parent=${M872_M803_DC_PROC_PPID}
            m872_m803_dc_child_start=${birth_start}
            m872_m803_dc_child_uid=${birth_uid}
            m872_m803_dc_child_parent=${birth_parent}
        fi
        [[ "${M872_M803_DC_PROC_STARTTIME}" == "${birth_start}" && \
           "${M872_M803_DC_PROC_UID}" == "${birth_uid}" && \
           "${M872_M803_DC_PROC_PPID}" == "${birth_parent}" && \
           "${birth_uid}" == "${m872_m803_dc_uid}" && \
           "${birth_parent}" == "$$" && \
           "${M872_M803_DC_PROC_STATE}" != Z ]] || return 1
        m872_m803_dc_child_exe=${M872_M803_DC_PROC_EXE}
        m872_m803_dc_child_cmdline=${M872_M803_DC_PROC_CMDLINE_NUL_HEX}
        if [[ "${M872_M803_DC_PROC_EXE}" == "${m872_m803_dc_dc_actual_exe}" ]]; then
            m872_m803_dc_dc_cmdline_matches "${pid}" || return 1
            # Reread after argv parsing to close a transition/reuse race.
            m872_m803_dc_proc_identity "${pid}" || return 1
            [[ "${M872_M803_DC_PROC_STARTTIME}" == "${birth_start}" && \
               "${M872_M803_DC_PROC_UID}" == "${birth_uid}" && \
               "${M872_M803_DC_PROC_PPID}" == "${birth_parent}" && \
               "${M872_M803_DC_PROC_EXE}" == "${m872_m803_dc_dc_actual_exe}" ]] || return 1
            m872_m803_dc_dc_cmdline_matches "${pid}" || return 1
            m872_m803_dc_child_exe=${M872_M803_DC_PROC_EXE}
            m872_m803_dc_child_cmdline=${M872_M803_DC_PROC_CMDLINE_NUL_HEX}
            return 0
        fi
        sleep 0.01
    done
    return 1
}

# If stable common_shell capture fails, no runtime monitor is allowed to be
# skipped.  TERM is issued immediately only to the exact fork birth tuple;
# after a bounded grace period KILL is permitted only for that same tuple.
m872_m803_dc_fail_closed_capture() {
    local pid=$1 point=$2 state=1 tries signal_sent=none
    printf 'timestamp=%s status=FAIL_DC_IDENTITY_CAPTURE child_pid=%s frozen_starttime=%s frozen_uid=%s frozen_parent=%s last_exe=%s last_cmdline_nul_hex=%s\n' \
        "$(date --iso-8601=seconds)" "${pid}" \
        "${m872_m803_dc_child_start:-unavailable}" \
        "${m872_m803_dc_child_uid:-unavailable}" \
        "${m872_m803_dc_child_parent:-unavailable}" \
        "${m872_m803_dc_child_exe:-unavailable}" \
        "${m872_m803_dc_child_cmdline:-unavailable}" \
        >"${point}/dc_identity_capture_failure.txt"
    if [[ -n "${m872_m803_dc_child_start}" && -n "${m872_m803_dc_child_uid}" && \
          -n "${m872_m803_dc_child_parent}" ]] && m872_m803_dc_proc_identity "${pid}" && \
            [[ "${M872_M803_DC_PROC_STARTTIME}" == "${m872_m803_dc_child_start}" && \
               "${M872_M803_DC_PROC_UID}" == "${m872_m803_dc_child_uid}" && \
               "${M872_M803_DC_PROC_PPID}" == "${m872_m803_dc_child_parent}" ]]; then
        printf 'term_tuple_exe=%s\nterm_tuple_cmdline_nul_hex=%s\n' \
            "${M872_M803_DC_PROC_EXE}" "${M872_M803_DC_PROC_CMDLINE_NUL_HEX}" \
            >>"${point}/dc_identity_capture_failure.txt"
        kill -TERM "${pid}" 2>/dev/null || true
        signal_sent=TERM
        for tries in $(seq 1 50); do
            [[ -e "/proc/${pid}" ]] || break
            m872_m803_dc_proc_identity "${pid}" || break
            [[ "${M872_M803_DC_PROC_STARTTIME}" == "${m872_m803_dc_child_start}" && \
               "${M872_M803_DC_PROC_UID}" == "${m872_m803_dc_child_uid}" && \
               "${M872_M803_DC_PROC_PPID}" == "${m872_m803_dc_child_parent}" && \
               "${M872_M803_DC_PROC_STATE}" != Z ]] || break
            sleep 0.1
        done
        if m872_m803_dc_proc_identity "${pid}" && \
                [[ "${M872_M803_DC_PROC_STARTTIME}" == "${m872_m803_dc_child_start}" && \
                   "${M872_M803_DC_PROC_UID}" == "${m872_m803_dc_child_uid}" && \
                   "${M872_M803_DC_PROC_PPID}" == "${m872_m803_dc_child_parent}" && \
                   "${M872_M803_DC_PROC_STATE}" != Z ]]; then
            kill -KILL "${pid}" 2>/dev/null || true
            signal_sent=TERM_THEN_KILL
        fi
    fi
    set +e
    wait "${pid}"
    m872_m803_dc_child_rc=$?
    set -e
    printf 'signal_sent=%s\nwait_exit_code=%s\nstatus=QUARANTINE_REQUIRED_NO_MONITOR_BYPASS\n' \
        "${signal_sent}" "${m872_m803_dc_child_rc}" \
        >>"${point}/dc_identity_capture_failure.txt"
}

# Accept only the one fixed 16-line Design Vision bootstrap block that M769
# independently sealed.  HOME remains unset; this function never invents or
# mutates it.  The block must occur once in startup (line <=64), be bracketed
# by the exact startup context, and match the frozen SHA byte-for-byte.  After
# removing only those 16 lines, every other anchored Error/Fatal and emitted
# TIM-209/OPT-150 diagnostic remains fatal.
m872_m803_dc_validate_dc_log() {
    local log receipt error_line start end block_sha error_count filtered
    local -a error_lines=()
    log=$1
    receipt=$2
    error_line='Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl'
    [[ -s "${log}" ]] || return 1
    mapfile -t error_lines < <(grep -nE '^Error:|^Fatal:' "${log}" || true)
    [[ "${#error_lines[@]}" -eq 1 ]] || return 1
    [[ "${error_lines[0]#*:}" == "${error_line}" ]] || return 1
    start=${error_lines[0]%%:*}
    [[ "${start}" =~ ^[0-9]+$ && "${start}" -ge 2 && "${start}" -le 64 ]] || return 1
    end=$((start + 15))
    [[ "$(sed -n "$((start - 1))p" "${log}")" == Initializing... ]] || return 1
    [[ "$(sed -n "$((end + 1))p" "${log}")" == "Current time:"* ]] || return 1
    block_sha="$(sed -n "${start},${end}p" "${log}" | sha256sum | awk '{print $1}')"
    [[ "${block_sha}" == "${m872_m803_dc_bootstrap_block_sha256}" ]] || return 1
    error_count="$(grep -iEc 'error:|fatal:' "${log}" || true)"
    [[ "${error_count}" -eq 1 ]] || return 1
    filtered="${receipt}.filtered.tmp"
    awk -v start="${start}" -v end="${end}" \
        'NR < start || NR > end { print }' "${log}" >"${filtered}"
    if grep -Eq '^Error:|^Fatal:|^(Warning|Information):.*\((TIM-209|OPT-150)\)' \
            "${filtered}"; then
        rm -f "${filtered}"
        return 1
    fi
    rm -f "${filtered}"
    printf 'status=PASS_EXACT_SINGLE_M769_BOOTSTRAP_BLOCK_WHITELIST\nblock_start_line=%s\nblock_end_line=%s\nblock_sha256=%s\nanchored_error_or_fatal_count=1\ncase_insensitive_error_or_fatal_count=1\nother_error_fatal_tim209_opt150_count=0\nhome_was_not_set_or_synthesized_by_runner=true\n' \
        "${start}" "${end}" "${block_sha}" >"${receipt}"
}

m872_m803_dc_run_point() {
    local id
    local mode
    local point
    local h0
    local state
    id=$1
    mode=$2
    point="${m872_m803_dc_work}/${id}"
    h0="$(awk -F= '/^h0_commit_headroom_kib=/ {print $2}' \
        "${m872_m803_dc_work}/preflight/${id}/preflight_receipt.txt")"
    mkdir "${point}"
    export DESIGN_NAME=m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24
    export RTL_FILELIST="${m872_m803_dc_hw_root}/${m872_m803_dc_filelist}"
    export OUTPUT_DIR="${point}"
    export ELAB_PARAMETERS="ARCH_MODE=${mode}"
    m872_m803_dc_child_pid=""; m872_m803_dc_child_start=""; m872_m803_dc_child_uid=""
    m872_m803_dc_child_exe=""; m872_m803_dc_child_parent=""; m872_m803_dc_child_cmdline=""
    m872_m803_dc_monitor_pid=""; m872_m803_dc_monitor_start=""
    m872_m803_dc_child_rc=running; m872_m803_dc_monitor_rc=running
    set +e
    "${m872_m803_dc_dc}" -f "${m872_m803_dc_hw_root}/${m872_m803_dc_tcl}" \
        >"${point}/dc.log" 2>&1 &
    m872_m803_dc_child_pid=$!
    m872_m803_dc_capture_dc_identity "${m872_m803_dc_child_pid}"
    state=$?
    if [[ "${state}" -ne 0 ]]; then
        m872_m803_dc_fail_closed_capture "${m872_m803_dc_child_pid}" "${point}"
        return 47
    fi
    printf 'timestamp=%s axis=%s child_pid=%s child_starttime=%s child_uid=%s child_parent=%s child_exe=%s child_cmdline_nul_hex=%s runner_pid=%s h0_commit_headroom_kib=%s\n' \
        "$(date --iso-8601=seconds)" "${id}" "${m872_m803_dc_child_pid}" \
        "${m872_m803_dc_child_start}" "${m872_m803_dc_child_uid}" \
        "${m872_m803_dc_child_parent}" "${m872_m803_dc_child_exe}" \
        "${m872_m803_dc_child_cmdline}" "$$" "${h0}" \
        >"${point}/launch_pid_tree_root.txt"
    m872_m803_dc_runtime_monitor "${m872_m803_dc_child_pid}" "${m872_m803_dc_child_start}" \
        "${m872_m803_dc_child_uid}" "${m872_m803_dc_child_exe}" \
        "${m872_m803_dc_child_parent}" "${m872_m803_dc_child_cmdline}" \
        "${h0}" "${point}" &
    m872_m803_dc_monitor_pid=$!
    if m872_m803_dc_proc_identity "${m872_m803_dc_monitor_pid}"; then
        m872_m803_dc_monitor_start=${M872_M803_DC_PROC_STARTTIME}
    else
        m872_m803_dc_monitor_start=unavailable
    fi
    printf 'monitor_pid=%s\nmonitor_starttime=%s\nmonitor_launch_liveness=%s\n' \
        "${m872_m803_dc_monitor_pid}" "${m872_m803_dc_monitor_start}" \
        "$([[ -e "/proc/${m872_m803_dc_monitor_pid}" ]] && echo ALIVE || echo EXITED_EARLY)" \
        >>"${point}/launch_pid_tree_root.txt"
    wait "${m872_m803_dc_child_pid}"
    m872_m803_dc_child_rc=$?
    wait "${m872_m803_dc_monitor_pid}"
    m872_m803_dc_monitor_rc=$?
    set -e
    printf '%s\n' "${m872_m803_dc_child_rc}" >"${point}/dc.rc"
    printf '%s\n' "${m872_m803_dc_monitor_rc}" >"${point}/runtime_monitor.rc"
    m872_m803_dc_child_pid=""; m872_m803_dc_child_start=""; m872_m803_dc_child_uid=""
    m872_m803_dc_child_exe=""; m872_m803_dc_child_parent=""; m872_m803_dc_child_cmdline=""
    m872_m803_dc_monitor_pid=""; m872_m803_dc_monitor_start=""

    [[ "${m872_m803_dc_signal}" == none ]] || return 130
    [[ -s "${point}/runtime_final_gate_ack.txt" ]] || return 42
    grep -Fq 'final_gate_applied=true' "${point}/runtime_final_gate_ack.txt" || return 42
    grep -Fq 'status=PASS_FINAL_GATE_ACK' "${point}/runtime_final_gate_ack.txt" || return 42
    [[ "${m872_m803_dc_monitor_rc}" -eq 0 ]] || {
        m872_m803_dc_runtime_latch=1
        m872_m803_dc_runtime_latch_reason="$(awk -F= '/^reason=/ {print $2}' \
            "${point}/resource_runtime.log" | tail -1)"
        return 42
    }
    [[ "${m872_m803_dc_child_rc}" -eq 0 ]] || return "${m872_m803_dc_child_rc}"
    [[ -s "${point}/TCL_PASS_TERMINAL.txt" ]] || return 43
    grep -Fxq 'status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL' \
        "${point}/TCL_PASS_TERMINAL.txt" || return 43
    grep -Fxq 'compile_ultra_count=1' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'incremental_compile_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_optimization_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_not_closed_at_dc=true' "${point}/TCL_PASS_TERMINAL.txt"
    [[ ! -e "${point}/TCL_EXPLICIT_FAILURE.txt" ]] || return 43
    grep -Fxq 'TIM-209=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'OPT-150=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'flow=m519_r8_setup_area_only' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'compile_ultra_count=1' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'incremental_compile_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_fix_command_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_only_optimization_count=0' "${point}/reports/flow_contract.rpt"
    m872_m803_dc_validate_dc_log "${point}/dc.log" \
        "${point}/bootstrap_log_whitelist_receipt.txt" || return 44
    for report in area.rpt qor.rpt timing_setup.rpt \
            timing_hold_diagnostic.rpt constraint_setup.rpt \
            constraint_hold_diagnostic.rpt constraint_max_capacitance.rpt \
            constraint_max_transition.rpt constraint_max_fanout.rpt \
            check_design_postcompile.rpt check_timing_postcompile.rpt \
            flow_contract.rpt compile_receipt.rpt; do
        [[ -s "${point}/reports/${report}" ]] || return 45
    done
    m872_m803_dc_record_output_artifacts "${point}" || return 45
    ! grep -Fq 'slack (VIOLATED)' "${point}/reports/timing_setup.rpt" || return 46
    for report in constraint_setup.rpt constraint_max_capacitance.rpt \
            constraint_max_transition.rpt constraint_max_fanout.rpt; do
        grep -Fq 'This design has no violated constraints.' \
            "${point}/reports/${report}" || return 46
    done
    printf 'status=PASS_M872_M803_DC_%s_SETUP_AREA_LOGIC_ONLY_DC_3NS_PENDING_RECEIPT_REVIEW\nmacro_count=0\nlogic_only_pre_macro=true\nhold_not_closed_at_dc=true\nhold_diagnostic_only=true\npower=false\nenergy=false\nppa=false\npaper_ppa_ready=false\nsystem=false\nsystem_speedup=false\nheadline=false\n' \
        "${id^^}" >"${point}/RUN_COMPLETE.txt" || return 47
    m872_m803_dc_verify_live_artifact_receipts "${point}" || return 47
}

m872_m803_dc_run_point k1 0
m872_m803_dc_axis_preflight k8 "${m872_m803_dc_work}/preflight/k8" || exit 40
m872_m803_dc_run_point k8 1
m872_m803_dc_axis_preflight k1x8 "${m872_m803_dc_work}/preflight/k1x8" || exit 40
m872_m803_dc_run_point k1x8 2
m872_m803_dc_axis_preflight post_k1x8_recovery \
    "${m872_m803_dc_work}/preflight/post_k1x8_recovery" || exit 40

for m872_m803_dc_axis in k1 k8 k1x8; do
    m872_m803_dc_verify_live_artifact_receipts \
        "${m872_m803_dc_work}/${m872_m803_dc_axis}" || exit 48
done
printf 'status=PASS_M872_M803_DC_THREE_AXIS_SETUP_AREA_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RECEIPT_REVIEW\nlogic_only_pre_macro=true\nhold_not_closed_at_dc=true\nhold_diagnostic_only=true\npower=false\nenergy=false\nppa=false\npaper_ppa_ready=false\nsystem=false\nsystem_speedup=false\nheadline=false\n' \
    >"${m872_m803_dc_work}/RUN_COMPLETE.txt" || exit 48
# Recheck after the enclosing RUN_COMPLETE publication and immediately before
# sealing.  A post-receipt byte change therefore cannot inherit a success mark.
for m872_m803_dc_axis in k1 k8 k1x8; do
    m872_m803_dc_verify_live_artifact_receipts \
        "${m872_m803_dc_work}/${m872_m803_dc_axis}" || exit 48
done
m872_m803_dc_seal_dir "${m872_m803_dc_work}" || exit 48
for m872_m803_dc_axis in k1 k8 k1x8; do
    m872_m803_dc_verify_axis_artifact_manifest "${m872_m803_dc_work}" \
        "${m872_m803_dc_work}/${m872_m803_dc_axis}" || exit 48
done
(cd "${m872_m803_dc_work}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 48
mv -T -- "${m872_m803_dc_work}" "${m872_m803_dc_canonical}" || exit 48
m872_m803_dc_run_created=0
m872_m803_dc_complete=1
trap - EXIT INT TERM
echo "PASS M872 M803 DC raw setup/area DC result sealed at ${m872_m803_dc_canonical}"
