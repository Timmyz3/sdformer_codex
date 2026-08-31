#!/usr/bin/env bash
set -euo pipefail

m485_runner_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m485_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m485_hw="$(cd "${m485_dc_root}/.." && pwd)"
m485_runner="${m485_runner_abs}"
m485_canonical_run="${m485_dc_root}/runs/m519_fc2_registered_release_three_axis_logic_only_dc_3p000ns_r1_20260827"
m485_run="${m485_dc_root}/runs/.m519_fc2_registered_release_three_axis_dc_r1_work.$$"
m485_attempt="${m485_dc_root}/runs/.m519_fc2_registered_release_three_axis_dc_r1_attempt_consumed"
m485_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m485_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m485_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m485_f493=dc_handoff/filelists/date_m519_fc2_registered_release_three_axis_logic_only_dc.f
m485_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m485_tcl=dc_handoff/scripts/run_dc_m519_fc2_registered_release_three_axis.tcl
m485_contract=contracts/m519_fc2_registered_release_three_axis_recovery_contract_r3_20260827.json
m485_vcs=results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827/m519_fc2_registered_release_vcs_receipt_r2.json
m485_vcs_seal=results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827/SHA256SUMS.seal.sha256
m485_static_review=reviews/m519_registered_release_static_hammer_r3_20260827/SHA256SUMS.seal.sha256
m485_vcs_review=reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827/SHA256SUMS.seal.sha256
m485_launch_admission=contracts/m519_fc2_registered_release_dc_launch_admission_r2_20260827.json
m485_failure_review=reviews/m496_r3_internal_loop_failure_hammer_r1_20260827/SHA256SUMS.seal.sha256

[[ -n "${M519_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m485_runner_abs}" | awk '{print $1}')" == \
   "${M519_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M519 caller did not pin the independently reviewed runner SHA" >&2
    exit 3
}

m485_sha() { sha256sum "$1" | awk '{print $1}'; }
m485_expect() {
    local m485_path=$1 m485_expected=$2
    [[ -f "${m485_path}" ]] || { echo "missing ${m485_path}" >&2; exit 3; }
    [[ "$(m485_sha "${m485_path}")" == "${m485_expected}" ]] || {
        echo "M519 SHA mismatch ${m485_path}" >&2
        exit 3
    }
}

[[ -z "${M519_DC_RUN:-}" ]] || {
    echo "M519 r1 canonical run path is locked; M519_DC_RUN is forbidden" >&2
    exit 5
}
[[ ! -e "${m485_canonical_run}" && ! -e "${m485_run}" && \
   ! -e "${m485_attempt}" ]] || {
    echo "M519 refuses to overwrite ${m485_canonical_run}" >&2
    exit 5
}
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '/common_shell_exec -shell dc_shell ' >/dev/null; then
    echo "M519 refuses to collide with another Design Compiler run" >&2
    exit 4
fi

cd "${m485_hw}"
[[ -n "${M519_EXPECTED_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M519 launch admission is not pinned; DC remains locked" >&2
    exit 3
}
m485_expect "${m485_launch_admission}" \
    "${M519_EXPECTED_LAUNCH_ADMISSION_SHA256}"
[[ "$(jq -er '.status' "${m485_launch_admission}")" == \
   "AUTHORIZED_ONE_M519_DC_ATTEMPT" ]] || {
    echo "M519 launch admission does not authorize the one DC attempt" >&2
    exit 3
}
m485_expected_contract="$(jq -er '.identity.recovery_contract_sha256' \
    "${m485_launch_admission}")"
m485_expected_vcs="$(jq -er '.identity.vcs_receipt_sha256' \
    "${m485_launch_admission}")"
m485_expected_vcs_seal="$(jq -er '.identity.vcs_outer_seal_file_sha256' \
    "${m485_launch_admission}")"
m485_expected_static_review="$(jq -er '.identity.static_review_outer_seal_file_sha256' \
    "${m485_launch_admission}")"
m485_expected_vcs_review="$(jq -er '.identity.vcs_review_outer_seal_file_sha256' \
    "${m485_launch_admission}")"
m485_expected_tcl="$(jq -er '.identity.dc_tcl_sha256' \
    "${m485_launch_admission}")"
for m485_identity_sha in \
        "${m485_expected_contract}" \
        "${m485_expected_vcs}" \
        "${m485_expected_vcs_seal}" \
        "${m485_expected_static_review}" \
        "${m485_expected_vcs_review}" \
        "${m485_expected_tcl}"; do
    [[ "${m485_identity_sha}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "M519 launch admission contains a non-canonical SHA256 identity" >&2
        exit 3
    }
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m485_launch_admission}")" == \
   "${M519_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.docs359_sha256' "${m485_launch_admission}")" == \
   "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]] || {
    echo "M519 launch admission does not bind the frozen docs/359 identity" >&2
    exit 3
}

m485_verify_all_inputs() {
m485_expect "${m485_runner}" "${M519_EXPECTED_RUNNER_SHA256}"
m485_expect "${m485_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m485_expect "${m485_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m485_expect "${m485_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m485_expect rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5
m485_expect rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv 8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0
m485_expect rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv 529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267
m485_expect rtl_m218/m218_fc2_tagged_slice_service_island.sv f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1
m485_expect rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv 597e4d9e9a606afa58111d01be8e8304e4fb5d4656cabdd4da9fca4b8393f43b
m485_expect rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv e5f3022e23736216f61482e1e33638d84c9a39dfb807c1c2fc53a14c90696456
m485_expect rtl_m519/m519_fc2_k1_registered_release_service_island.sv 3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871
m485_expect rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv 010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b
m485_expect rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv 6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815
m485_expect rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv 5a4b05af5dcecd9c104aef00b4e0f818bc26e48e7c061424699a5ab00cefc96b
m485_expect rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv 11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff
m485_expect rtl_m519/m519_fc2_registered_release_matched_8bank_raw4_acc24.sv f07dc54820721cbded4d26b5b9ca7a756ba4940906ddf8b618595f260c5f86df
m485_expect "${m485_f493}" 954a20cab1f944d9e618043e640571d7fac6361095624adf6c10164f31377132
m485_expect "${m485_sdc}" 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5
m485_expect "${m485_tcl}" "${m485_expected_tcl}"
m485_expect "${m485_contract}" "${m485_expected_contract}"
m485_expect "${m485_vcs}" "${m485_expected_vcs}"
m485_expect "${m485_vcs_seal}" "${m485_expected_vcs_seal}"
m485_expect "${m485_static_review}" "${m485_expected_static_review}"
m485_expect "${m485_vcs_review}" "${m485_expected_vcs_review}"
m485_expect "${m485_failure_review}" c8e49b3aeb1406c103604d6fec23e48ff27682f58eaed0e9abdd5b2cae6b3b79
m485_expect "${m485_launch_admission}" "${M519_EXPECTED_LAUNCH_ADMISSION_SHA256}"
m485_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
(cd results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd reviews/m519_registered_release_static_hammer_r3_20260827 && \
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827 && \
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd reviews/m496_r3_internal_loop_failure_hammer_r1_20260827 && \
    sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

m485_verify_all_inputs

export HW_ROOT="${m485_hw}"
export LIB_DB="${m485_slow}"
export MIN_LIB_DB="${m485_fast}"
export SDC_FILE="${m485_hw}/${m485_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m485_resource_snapshot() {
    local m485_label=$1 m485_log=$2
    local m485_limit m485_committed m485_available m485_swap
    local m485_headroom m485_failcnt m485_under_oom m485_oom_kill
    m485_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    m485_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    m485_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    m485_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    m485_headroom=$((m485_limit - m485_committed))
    m485_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    m485_under_oom=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    m485_oom_kill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'timestamp=%s label=%s commit_limit_kib=%s committed_as_kib=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "$(date --iso-8601=seconds)" "${m485_label}" "${m485_limit}" \
        "${m485_committed}" "${m485_headroom}" "${m485_available}" \
        "${m485_swap}" "${m485_failcnt}" "${m485_under_oom}" \
        "${m485_oom_kill}" >>"${m485_log}"
    {
        printf 'timestamp=%s label=%s\n' "$(date --iso-8601=seconds)" "${m485_label}"
        cat /proc/meminfo
    } >>"${m485_log%.log}.meminfo.log"
    [[ "${m485_headroom}" -ge 67108864 && \
       "${m485_available}" -ge 134217728 && \
       "${m485_swap}" -ge 33554432 && \
       "${m485_failcnt}" -eq 0 && "${m485_under_oom}" -eq 0 && \
       "${m485_oom_kill}" -eq 0 ]]
}

m485_forbidden_process_gate() {
    local m485_log=$1 m485_hits=0
    if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
            pgrep -f '/common_shell_exec -shell dc_shell ' >/dev/null; then
        printf 'forbidden_process=dc\n' >>"${m485_log}"
        pgrep -a -x dc_shell >>"${m485_log}" 2>/dev/null || true
        pgrep -a -x dc_shell-t >>"${m485_log}" 2>/dev/null || true
        pgrep -a -f '/common_shell_exec -shell dc_shell ' >>"${m485_log}" 2>/dev/null || true
        m485_hits=1
    fi
    if pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null || \
            pgrep -f '/common_shell_exec -shell (fm_shell|pt_shell) ' >/dev/null; then
        printf 'forbidden_process=fm_or_pt\n' >>"${m485_log}"
        m485_hits=1
    fi
    if pgrep -f '(^|/)(vcs)( |$)' >/dev/null || \
            pgrep -x vcs1 >/dev/null || pgrep -x vlogan >/dev/null || \
            pgrep -x vhdlan >/dev/null; then
        printf 'forbidden_process=vcs_family\n' >>"${m485_log}"
        m485_hits=1
    fi
    local m485_simv_pid m485_simv_owner m485_simv_state
    local m485_simv_cpu m485_simv_rss m485_current_user
    m485_current_user=$(id -un)
    while read -r m485_simv_pid; do
        [[ -n "${m485_simv_pid}" ]] || continue
        m485_simv_owner=$(ps -o user= -p "${m485_simv_pid}" | xargs)
        m485_simv_state=$(ps -o stat= -p "${m485_simv_pid}" | xargs)
        m485_simv_cpu=$(ps -o pcpu= -p "${m485_simv_pid}" | xargs)
        m485_simv_rss=$(ps -o rss= -p "${m485_simv_pid}" | xargs)
        if [[ "${m485_simv_owner}" == "${m485_current_user}" ||
              ! "${m485_simv_state}" =~ ^[SI] ]] ||
                ! awk -v cpu="${m485_simv_cpu}" -v rss="${m485_simv_rss}" \
                    'BEGIN {exit !(cpu <= 0.5 && rss <= 262144)}'; then
            printf 'forbidden_process=active_or_same_user_simv pid=%s owner=%s state=%s pcpu=%s rss_kib=%s\n' \
                "${m485_simv_pid}" "${m485_simv_owner}" \
                "${m485_simv_state}" "${m485_simv_cpu}" \
                "${m485_simv_rss}" >>"${m485_log}"
            m485_hits=1
        else
            printf 'allowed_foreign_idle_simv pid=%s owner=%s state=%s pcpu=%s rss_kib=%s policy=foreign_sleeping_cpu_le_0p5_rss_le_256mib\n' \
                "${m485_simv_pid}" "${m485_simv_owner}" \
                "${m485_simv_state}" "${m485_simv_cpu}" \
                "${m485_simv_rss}" >>"${m485_log}"
        fi
    done < <(pgrep -x simv || true)
    if pgrep -f '(^|[ /])[^ ]*(analyze|independent|sweep|dse|simulate)_m[0-9][^ ]*\.py( |$)' \
            >/dev/null; then
        printf 'forbidden_process=project_cpu_dse\n' >>"${m485_log}"
        m485_hits=1
    fi
    [[ "${m485_hits}" -eq 0 ]]
}

m485_resource_gate() {
    local m485_id=$1 m485_log=$2 m485_failures=0
    : >"${m485_log}"
    for m485_sample in 1 2 3; do
        if ! m485_resource_snapshot "preflight_${m485_id}_${m485_sample}" \
                "${m485_log}"; then
            m485_failures=$((m485_failures + 1))
        fi
        if ! m485_forbidden_process_gate "${m485_log}"; then
            m485_failures=$((m485_failures + 1))
        fi
        ps -eo pid,ppid,etime,stat,pcpu,rss,vsz,args >>"${m485_log}"
        if [[ "${m485_sample}" -ne 3 ]]; then sleep 10; fi
    done
    [[ "${m485_failures}" -eq 0 ]] || return 40
    m485_forbidden_process_gate "${m485_log}" || return 41
}

m485_cgroup_oom_clear() {
    local m485_failcnt m485_under_oom m485_oom_kill
    m485_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    m485_under_oom=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    m485_oom_kill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    [[ "${m485_failcnt}" -eq 0 && "${m485_under_oom}" -eq 0 && \
       "${m485_oom_kill}" -eq 0 ]]
}

m485_verify_attempt_receipt() {
    local m485_actual
    [[ -d "${m485_attempt}" && ! -L "${m485_attempt}" ]]
    m485_actual=$(find "${m485_attempt}" -mindepth 1 -maxdepth 1 \
        -printf '%f\n' | LC_ALL=C sort)
    [[ "${m485_actual}" == $'ATTEMPT_CONSUMED.txt\nSHA256SUMS\nSHA256SUMS.seal.sha256\nidentity.sha256' ]]
    grep -Fxq 'status=CONSUMED_AT_FIRST_DC_LAUNCH' \
        "${m485_attempt}/ATTEMPT_CONSUMED.txt"
    grep -Fxq "canonical_run=${m485_canonical_run}" \
        "${m485_attempt}/ATTEMPT_CONSUMED.txt"
    (cd "${m485_attempt}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
    sha256sum -c "${m485_attempt}/identity.sha256" >/dev/null
}

m485_runtime_monitor() {
    local m485_pid=$1 m485_log=$2
    local m485_oom_latched=0
    while kill -0 "${m485_pid}" 2>/dev/null; do
        m485_resource_snapshot runtime "${m485_log}" || true
        m485_cgroup_oom_clear || m485_oom_latched=1
        sleep 10
    done
    m485_resource_snapshot runtime_final "${m485_log}" || true
    m485_cgroup_oom_clear || m485_oom_latched=1
    printf 'runtime_cgroup_oom_latched=%s\n' "${m485_oom_latched}" \
        >>"${m485_log}"
    [[ "${m485_oom_latched}" -eq 0 ]]
}

m485_complete=0
m485_run_created=0
m485_preflight=""
m485_prelaunch_quarantine="${m485_canonical_run}.prelaunch_failed.$$.quarantine"
m485_failed_quarantine="${m485_canonical_run}.failed_or_incomplete.$$.quarantine"
m485_cleanup_and_receipt() {
    local m485_rc=$?
    if [[ -n "${m485_preflight}" && -d "${m485_preflight}" ]]; then
        rm -f "${m485_preflight}/resource_preflight.log" \
            "${m485_preflight}/resource_preflight.meminfo.log"
        rmdir "${m485_preflight}" 2>/dev/null || true
    fi
    if [[ "${m485_run_created}" -eq 1 && ! -e "${m485_attempt}" ]]; then
        if [[ ! -e "${m485_prelaunch_quarantine}" ]]; then
            mv -T "${m485_run}" "${m485_prelaunch_quarantine}"
            [[ ! -e "${m485_run}" && -d "${m485_prelaunch_quarantine}" ]]
            m485_run_created=0
        fi
    elif [[ "${m485_run_created}" -eq 1 && \
            "${m485_complete}" -ne 1 ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${m485_rc}" >"${m485_run}/RUN_FAILED_OR_INCOMPLETE.txt"
        if [[ ! -e "${m485_failed_quarantine}" ]]; then
            mv -T "${m485_run}" "${m485_failed_quarantine}"
            [[ ! -e "${m485_run}" && -d "${m485_failed_quarantine}" ]]
            m485_run_created=0
        fi
    fi
    return "${m485_rc}"
}
trap m485_cleanup_and_receipt EXIT

m485_run_point() {
    local m485_id=$1 m485_top=$2 m485_filelist=$3 m485_parameters=$4
    local m485_dir="${m485_run}/${m485_id}"
    export DESIGN_NAME="${m485_top}"
    export RTL_FILELIST="${m485_hw}/${m485_filelist}"
    export OUTPUT_DIR="${m485_dir}"
    export ELAB_PARAMETERS="${m485_parameters}"
    if [[ "${m485_id}" == k1 ]]; then
        m485_preflight=$(mktemp -d \
            "${m485_dc_root}/runs/.m519_r1_preflight.XXXXXXXX")
        m485_resource_gate "${m485_id}" \
            "${m485_preflight}/resource_preflight.log"
        m485_verify_all_inputs
        m485_forbidden_process_gate \
            "${m485_preflight}/resource_preflight.log" || return 41
        [[ ! -e "${m485_canonical_run}" && ! -e "${m485_run}" && \
           ! -e "${m485_attempt}" ]] || return 42
        mkdir "${m485_run}"
        m485_run_created=1
        mkdir "${m485_dir}"
        mv "${m485_preflight}/resource_preflight.log" \
            "${m485_dir}/resource_preflight.log"
        mv "${m485_preflight}/resource_preflight.meminfo.log" \
            "${m485_dir}/resource_preflight.meminfo.log"
        rmdir "${m485_preflight}"
        m485_preflight=""
        sha256sum rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv \
            rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv \
            rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv \
            rtl_m218/m218_fc2_tagged_slice_service_island.sv \
            rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv \
            rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv \
            rtl_m519/m519_fc2_k1_registered_release_service_island.sv \
            rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv \
            rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv \
            rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv \
            rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv \
            rtl_m519/m519_fc2_registered_release_matched_8bank_raw4_acc24.sv \
            "${m485_f493}" "${m485_sdc}" "${m485_tcl}" \
            "${m485_contract}" "${m485_vcs}" "${m485_vcs_seal}" \
            "${m485_static_review}" "${m485_vcs_review}" \
            "${m485_failure_review}" \
            "${m485_launch_admission}" \
            docs/359_DATE终局冻结_20260813.md \
            "${m485_runner}" "${m485_dc}" "${m485_slow}" "${m485_fast}" \
            >"${m485_run}/input_sha256.txt"
        cp "${m485_contract}" "${m485_run}/contract.json"
        sha256sum "${m485_runner}" >"${m485_run}/runner_sha256.txt"
    else
        mkdir "${m485_dir}"
        m485_resource_gate "${m485_id}" "${m485_dir}/resource_preflight.log"
    fi
    m485_verify_all_inputs
    m485_forbidden_process_gate "${m485_dir}/resource_preflight.log" || return 41
    if [[ "${m485_id}" == k1 ]]; then
        mkdir "${m485_run}/.attempt_staging" || return 44
        printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical_run=%s\n' \
            "$(date --iso-8601=seconds)" "${m485_canonical_run}" \
            >"${m485_run}/.attempt_staging/ATTEMPT_CONSUMED.txt"
        sha256sum "${m485_runner}" "${m485_contract}" \
            >"${m485_run}/.attempt_staging/identity.sha256"
        (cd "${m485_run}/.attempt_staging" && \
            sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && \
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
            sha256sum -c SHA256SUMS >/dev/null && \
            sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
        mv -T "${m485_run}/.attempt_staging" "${m485_attempt}" || return 44
        m485_verify_attempt_receipt
    fi
    set +e
    "${m485_dc}" -f "${m485_hw}/${m485_tcl}" \
        >"${m485_dir}/dc.log" 2>&1 &
    local m485_dc_pid=$!
    m485_runtime_monitor "${m485_dc_pid}" \
        "${m485_dir}/resource_runtime.log" &
    local m485_monitor_pid=$!
    wait "${m485_dc_pid}"
    local m485_rc=$?
    wait "${m485_monitor_pid}"
    local m485_monitor_rc=$?
    set -e
    echo "${m485_rc}" >"${m485_dir}/dc.rc"
    echo "${m485_monitor_rc}" >"${m485_dir}/runtime_monitor.rc"
    [[ "${m485_rc}" -eq 0 ]] || return 20
    [[ "${m485_monitor_rc}" -eq 0 ]] || return 23
    m485_cgroup_oom_clear || return 24
    m485_verify_all_inputs
    m485_verify_attempt_receipt
    ! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
        "${m485_dir}/dc.log" || return 21
    grep -Fq 'Thank you...' "${m485_dir}/dc.log" || return 22
    for m485_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt hierarchy_postcompile.rpt \
            resources_postcompile.rpt references_postcompile.rpt ports.rpt \
            port_count.txt check_timing_precompile.rpt \
            precompile_loop_gate.rpt; do
        [[ -s "${m485_dir}/reports/${m485_report}" ]] || return 30
    done
    grep -Fxq 'TIM-209=0' \
        "${m485_dir}/reports/precompile_loop_gate.rpt" || return 36
    grep -Fxq 'OPT-150=0' \
        "${m485_dir}/reports/precompile_loop_gate.rpt" || return 36
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${m485_dir}/reports/precompile_loop_gate.rpt" || return 36
    [[ -s "${m485_dir}/netlist/${m485_top}_mapped.v" ]] || return 31
    ! grep -Fq 'slack (VIOLATED)' "${m485_dir}/reports/timing_setup.rpt" \
        "${m485_dir}/reports/timing_hold.rpt" || return 32
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "${m485_dir}/reports/constraint_violators.rpt")" -eq 5 ]] \
        || return 33
    ! grep -Eqi 'unresolved reference|multiply driven|latch inferred' \
        "${m485_dir}/dc.log" "${m485_dir}/reports/check_design_postcompile.rpt" \
        || return 34

    local m485_area m485_cells m485_seq m485_combo m485_levels
    local m485_path m485_setup m485_hold m485_ports
    m485_area=$(awk '/Total cell area:/ {print $4; exit}' "${m485_dir}/reports/area.rpt")
    m485_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m485_dir}/reports/area.rpt")
    m485_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m485_dir}/reports/area.rpt")
    m485_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m485_dir}/reports/area.rpt")
    m485_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m485_dir}/reports/qor.rpt")
    m485_path=$(awk '/Critical Path Length:/ {print $4; exit}' "${m485_dir}/reports/qor.rpt")
    m485_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m485_dir}/reports/timing_setup.rpt")
    m485_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m485_dir}/reports/timing_hold.rpt")
    m485_ports=$(tr -d '[:space:]' <"${m485_dir}/reports/port_count.txt")
    for m485_value in "${m485_area}" "${m485_cells}" "${m485_seq}" \
            "${m485_combo}" "${m485_levels}" "${m485_path}" \
            "${m485_setup}" "${m485_hold}" "${m485_ports}"; do
        [[ -n "${m485_value}" ]] || return 35
    done
    awk -v x="${m485_setup}" 'BEGIN {exit !(x >= 0)}'
    awk -v x="${m485_hold}" 'BEGIN {exit !(x >= 0)}'
    printf '%s\n' \
        "status=PASS_M519_${m485_id^^}_LOGIC_ONLY_DC_3NS_CLEAN" \
        "design=${m485_top}" \
        "elaboration_parameters=${m485_parameters:-none}" \
        "cell_area_um2=${m485_area}" \
        "cell_count=${m485_cells}" \
        "sequential_cells=${m485_seq}" \
        "combinational_cells=${m485_combo}" \
        "logic_levels=${m485_levels}" \
        "critical_path_length_ns=${m485_path}" \
        "setup_worst_slack_ns=${m485_setup}" \
        "hold_worst_slack_ns=${m485_hold}" \
        "reported_port_count=${m485_ports:-unknown}" \
        "macro_count=0" \
        "paper_ppa_ready=false" \
        "system_speedup=false" \
        >"${m485_dir}/RUN_COMPLETE.txt"
    sha256sum "${m485_dir}/dc.log" "${m485_dir}/reports/"*.rpt \
        "${m485_dir}/netlist/"* "${m485_dir}/RUN_COMPLETE.txt" \
        >"${m485_dir}/evidence_manifest.sha256"
}

m485_run_point k1 m519_fc2_registered_release_matched_8bank_raw4_acc24 \
    "${m485_f493}" ARCH_MODE=0
m485_run_point k8 m519_fc2_registered_release_matched_8bank_raw4_acc24 \
    "${m485_f493}" ARCH_MODE=1
m485_run_point k1x8 m519_fc2_registered_release_matched_8bank_raw4_acc24 \
    "${m485_f493}" ARCH_MODE=2

: >"${m485_run}/final_identity_resource_process.log"
m485_resource_snapshot final_before_receipt \
    "${m485_run}/final_identity_resource_process.log" || true
m485_cgroup_oom_clear || exit 24
m485_forbidden_process_gate \
    "${m485_run}/final_identity_resource_process.log" || exit 41
m485_verify_all_inputs
m485_verify_attempt_receipt
sha256sum -c "${m485_run}/input_sha256.txt" >/dev/null

python3 - "${m485_run}" "${m485_hw}/${m485_vcs}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
vcs_receipt_path = Path(sys.argv[2])
vcs = json.loads(vcs_receipt_path.read_text())

def read_point(name):
    values = {}
    for line in (run/name/'RUN_COMPLETE.txt').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key] = value
    return {
        'cell_area_um2': float(values['cell_area_um2']),
        'cell_count': int(values['cell_count']),
        'sequential_cells': int(values['sequential_cells']),
        'combinational_cells': int(values['combinational_cells']),
        'logic_levels': float(values['logic_levels']),
        'critical_path_length_ns': float(values['critical_path_length_ns']),
        'setup_worst_slack_ns': float(values['setup_worst_slack_ns']),
        'hold_worst_slack_ns': float(values['hold_worst_slack_ns']),
        'reported_port_count': values['reported_port_count'],
    }

k1, k8, k1x8 = (read_point(x) for x in ('k1', 'k8', 'k1x8'))
k8_k1_area = k8['cell_area_um2'] / k1['cell_area_um2']
k8_k1x8_area = k8['cell_area_um2'] / k1x8['cell_area_um2']
k8_k1_seq = k8['sequential_cells'] / k1['sequential_cells']
k8_k1x8_seq = k8['sequential_cells'] / k1x8['sequential_cells']
ports_identical = len({p['reported_port_count'] for p in (k1, k8, k1x8)}) == 1
k1_rows = vcs['k1_vs_k1x8_cycle_rows'][:4]
k8_rows = vcs['k8_vs_k1x8_cycle_rows'][:4]
if len(k1_rows) != 4 or len(k8_rows) != 4:
    raise SystemExit('M519 VCS receipt lacks four nonzero three-axis rows')
for left, right in zip(k1_rows, k8_rows):
    if (left['output_blocks'], left['events'],
            left['registered_release_k1x8_cycles']) != (
            right['output_blocks'], right['events'],
            right['registered_release_k1x8_cycles']):
        raise SystemExit('M519 VCS repeated K1x8 identity mismatch')
k1_cycles = sum(row['registered_release_k1_cycles'] for row in k1_rows)
k8_cycles = sum(row['registered_release_k8_cycles'] for row in k8_rows)
k1x8_cycles = sum(row['registered_release_k1x8_cycles'] for row in k8_rows)
m519_k8_over_k1 = k1_cycles / k8_cycles
m519_k8_over_k1x8 = k1x8_cycles / k8_cycles
m519_k1x8_over_k1 = k1_cycles / k1x8_cycles
gates = {
    'all_three_dc_constraint_clean': all(
        p['setup_worst_slack_ns'] >= 0 and p['hold_worst_slack_ns'] >= 0
        for p in (k1, k8, k1x8)),
    'reported_port_count_identical': ports_identical,
    'm519_directed_k8_over_k1_ge_3': m519_k8_over_k1 >= 3.0,
    'k8_over_k1_area_lte_1p25': k8_k1_area <= 1.25,
    'k8_over_k1_throughput_per_area_ge_2p4': (
        m519_k8_over_k1 / k8_k1_area) >= 2.4,
    'm519_k8_over_k1x8_equal_bandwidth_ge_0p98': (
        m519_k8_over_k1x8 >= 0.98),
    'k8_over_k1x8_area_lte_0p50': k8_k1x8_area <= 0.5,
    'k8_over_k1x8_sequential_cells_lte_0p50': k8_k1x8_seq <= 0.5,
    'throughput_per_area_improvement_gte_2': (
        m519_k8_over_k1x8 / k8_k1x8_area) >= 2.0,
}
logic_gate = all(gates.values())
receipt = {
    'schema': 'm519_fc2_registered_release_three_axis_logic_only_dc_receipt_v1',
    'status': ('PASS_M519_THREE_AXIS_MATCHED_LOGIC_PARETO_GATE'
               if logic_gate else 'PASS_M519_DC_BUT_THREE_AXIS_LOGIC_PARETO_NO_GO'),
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'clock_network': 'ideal',
    'wireload': 'ZeroWireload',
    'macro_count_each': 0,
    'measured': {'k1': k1, 'k8': k8, 'k1x8': k1x8},
    'measured_area_ratios': {
        'k8_over_k1': k8_k1_area,
        'k8_over_k1x8': k8_k1x8_area,
        'k1x8_over_k8': 1.0 / k8_k1x8_area,
    },
    'measured_sequential_cell_ratios': {
        'k8_over_k1': k8_k1_seq,
        'k8_over_k1x8': k8_k1x8_seq,
        'k1x8_over_k8': 1.0 / k8_k1x8_seq,
    },
    'throughput_per_area_improvement': {
        'k8_over_k1_using_m519_directed_cycles': (
            m519_k8_over_k1 / k8_k1_area),
        'k8_over_k1x8_using_m519_directed_equal_bandwidth_cycles': (
            m519_k8_over_k1x8 / k8_k1x8_area),
    },
    'm519_vcs_cycle_ratios_remeasured_same_identity': {
        'k8_over_k1_directed_aggregate': m519_k8_over_k1,
        'k8_over_k1x8_equal_peak_bandwidth_aggregate': m519_k8_over_k1x8,
        'k1x8_over_k1_directed_aggregate': m519_k1x8_over_k1,
        'k1x8_over_k1_is_eightfold_bandwidth_scaling': True,
        'old_m216_m492_m497_cycles_reused': False,
    },
    'hard_gate_results': gates,
    'logic_pareto_gate': logic_gate,
    'fairness_limitations': [
        'All three elaborations use the same M519 top and eight scalar bank endpoints.',
        'Debug counters are not observable at M519 and may be optimized away.',
        'Weight SRAM macros and explicit paper context macros are excluded.'
    ],
    'admission': {
        'matched_compile_and_constraints': True,
        'canonical_bank_wrapper': True,
        'formality': False,
        'power': False,
        'energy': False,
        'paper_ppa_ready': False,
        'full_ffn': False,
        'full_network': False,
        'system_speedup': False,
        'date_headline': False,
    },
    'required_next_gate': (
        'Independent receipt-blind hammer, then Formality and matched SAIF/PTPX energy per completed work only if logic_pareto_gate is true.'
    )
}
(run/'m519_fc2_registered_release_three_axis_logic_only_dc_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
(run/'RUN_COMPLETE.txt').write_text(
    receipt['status'] + '\n'
    + f"k8_over_k1_area_ratio={k8_k1_area:.12f}\n"
    + f"k8_over_k1x8_area_ratio={k8_k1x8_area:.12f}\n"
    + f"k8_over_k1x8_sequential_cell_ratio={k8_k1x8_seq:.12f}\n"
    + f"k8_over_k1_throughput_per_area={m519_k8_over_k1 / k8_k1_area:.12f}\n"
    + f"k8_over_k1x8_throughput_per_area={m519_k8_over_k1x8 / k8_k1x8_area:.12f}\n"
    + 'paper_ppa_ready=false\nsystem_speedup=false\n')
files = [p for p in sorted(run.rglob('*')) if p.is_file()
         and p.name not in {'evidence_manifest.sha256',
                            'evidence_manifest.seal.sha256'}]
(run/'evidence_manifest.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n'
    for p in files))
(run/'evidence_manifest.seal.sha256').write_text(
    hashlib.sha256((run/'evidence_manifest.sha256').read_bytes()).hexdigest()
    + '  evidence_manifest.sha256\n')
PY
m485_verify_all_inputs
m485_verify_attempt_receipt
sha256sum -c "${m485_run}/input_sha256.txt" >/dev/null
m485_cgroup_oom_clear || exit 24
(cd "${m485_run}" && sha256sum -c evidence_manifest.sha256 >/dev/null \
    && sha256sum -c evidence_manifest.seal.sha256 >/dev/null)
[[ ! -e "${m485_canonical_run}" ]]
m485_complete=1
mv -T "${m485_run}" "${m485_canonical_run}"
m485_run="${m485_canonical_run}"
rm -f "${m485_run}/RUN_FAILED_OR_INCOMPLETE.txt"
echo "PASS M519 three-axis matched logic-only DC sealed at ${m485_run}"
