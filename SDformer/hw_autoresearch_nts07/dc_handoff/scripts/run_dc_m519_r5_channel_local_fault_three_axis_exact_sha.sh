#!/usr/bin/env bash
set -euo pipefail

m519_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m519_hw_root="$(cd "${m519_dc_root}/.." && pwd)"
m519_runner="$(realpath "${BASH_SOURCE[0]}")"
m519_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m519_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m519_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m519_filelist=dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f
m519_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m519_tcl=dc_handoff/scripts/run_dc_m519_r5_channel_local_fault_three_axis.tcl
m519_contract=contracts/m519_r5_channel_local_fault_recovery_contract_r1_20260827.json
m519_static_dir=reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827
m519_vcs_result=results/m519_r5_channel_local_fault_vcs_r1_20260827
m519_vcs_review=reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827
m519_admission=contracts/m519_r5_channel_local_fault_dc_launch_admission_r1_20260827.json
m519_canonical="${m519_dc_root}/runs/m519_r5_channel_local_fault_three_axis_logic_only_dc_3p000ns_r1_20260827"
m519_work="${m519_dc_root}/runs/.m519_r5_channel_local_fault_dc_work.$$"
m519_attempt="${m519_dc_root}/runs/.m519_r5_channel_local_fault_dc_attempt_consumed"
m519_quarantine="${m519_canonical}.failed_or_incomplete.$$.quarantine"

m519_sha() { sha256sum "$1" | awk '{print $1}'; }
m519_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m519_sha "${path}")" == "${expected}" ]] || {
        echo "M519 R5 identity mismatch: ${path}" >&2
        exit 3
    }
}

[[ -n "${M519_R5_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m519_sha "${m519_runner}")" == \
   "${M519_R5_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M519 R5 caller must pin independently reviewed DC runner SHA" >&2
    exit 3
}
[[ -n "${M519_R5_EXPECTED_DC_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M519 R5 DC launch admission is not caller-pinned" >&2
    exit 3
}
[[ ! -e "${m519_canonical}" && ! -e "${m519_work}" && \
   ! -e "${m519_attempt}" && ! -e "${m519_quarantine}" ]] || {
    echo "M519 R5 refuses consumed or colliding result identity" >&2
    exit 5
}
[[ -z "${M519_R5_DC_RUN:-}" ]] || {
    echo "M519 R5 canonical path override is forbidden" >&2
    exit 5
}
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null || \
        pgrep -u "$(id -u)" -x vcs >/dev/null || \
        pgrep -u "$(id -u)" -x vcs1 >/dev/null || \
        pgrep -u "$(id -u)" -x vlogan >/dev/null || \
        pgrep -u "$(id -u)" -x simv >/dev/null; then
    echo "M519 R5 refuses DC/VCS/FM/PT collision" >&2
    exit 4
fi

cd "${m519_hw_root}"
m519_expect "${m519_admission}" \
    "${M519_R5_EXPECTED_DC_LAUNCH_ADMISSION_SHA256}"
jq -e '.status == "AUTHORIZED_ONE_M519_R5_THREE_AXIS_DC_ATTEMPT"
       and .authorization.run_dc == true
       and .authorization.max_attempts == 1' "${m519_admission}" >/dev/null \
    || exit 3
for key_path in \
    recovery_contract_sha256 \
    static_review_outer_seal_file_sha256 \
    vcs_result_outer_seal_file_sha256 \
    vcs_review_outer_seal_file_sha256 \
    dc_tcl_sha256 dc_filelist_sha256 sdc_sha256 \
    slow_lib_sha256 fast_lib_sha256; do
    value="$(jq -er ".identity.${key_path}" "${m519_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m519_admission}")" == \
   "${M519_R5_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.docs359_sha256' "${m519_admission}")" == \
   dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 ]] \
    || exit 3
m519_expect "${m519_contract}" \
    "$(jq -er '.identity.recovery_contract_sha256' "${m519_admission}")"
m519_expect "${m519_static_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.static_review_outer_seal_file_sha256' "${m519_admission}")"
m519_expect "${m519_vcs_result}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.vcs_result_outer_seal_file_sha256' "${m519_admission}")"
m519_expect "${m519_vcs_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.vcs_review_outer_seal_file_sha256' "${m519_admission}")"
m519_expect "${m519_tcl}" \
    "$(jq -er '.identity.dc_tcl_sha256' "${m519_admission}")"
m519_expect "${m519_filelist}" \
    "$(jq -er '.identity.dc_filelist_sha256' "${m519_admission}")"
m519_expect "${m519_sdc}" \
    "$(jq -er '.identity.sdc_sha256' "${m519_admission}")"
m519_expect "${m519_slow}" \
    "$(jq -er '.identity.slow_lib_sha256' "${m519_admission}")"
m519_expect "${m519_fast}" \
    "$(jq -er '.identity.fast_lib_sha256' "${m519_admission}")"
m519_expect "${m519_dc}" \
    23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m519_expect docs/359_DATE终局冻结_20260813.md \
    dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
for sealed in "${m519_static_dir}" "${m519_vcs_result}" \
        "${m519_vcs_review}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done

m519_resource_snapshot() {
    local label=$1 log=$2
    local limit committed available swap headroom failcnt under oomkill
    limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    headroom=$((limit - committed))
    failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    under=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    oomkill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'timestamp=%s label=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${headroom}" \
        "${available}" "${swap}" "${failcnt}" "${under}" "${oomkill}" \
        >>"${log}"
    [[ "${headroom}" -ge 67108864 && "${available}" -ge 134217728 \
       && "${swap}" -ge 33554432 && "${failcnt}" -eq 0 \
       && "${under}" -eq 0 && "${oomkill}" -eq 0 ]]
}

for sample in 1 2 3; do
    m519_resource_snapshot "preflight_${sample}" \
        "${m519_dc_root}/runs/.m519_r5_resource_preflight.$$.log" || exit 40
done

mkdir "${m519_work}"
m519_run_created=1
m519_complete=0
m519_child_pid=""
m519_monitor_pid=""
m519_child_rc="not_started"
m519_monitor_rc="not_started"
m519_signal="none"
m519_runtime_latch=0

m519_signal_handler() {
    local signal_name=$1
    m519_signal="${signal_name}"
    printf 'timestamp=%s signal=%s child_pid=%s monitor_pid=%s\n' \
        "$(date --iso-8601=seconds)" "${signal_name}" \
        "${m519_child_pid:-none}" "${m519_monitor_pid:-none}" \
        >>"${m519_work}/signal_provenance.txt"
    if [[ -n "${m519_child_pid}" ]] && kill -0 "${m519_child_pid}" 2>/dev/null; then
        kill -s "${signal_name}" "${m519_child_pid}" 2>/dev/null || true
    fi
    if [[ -n "${m519_monitor_pid}" ]] && kill -0 "${m519_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m519_monitor_pid}" 2>/dev/null || true
    fi
}
trap 'm519_signal_handler INT' INT
trap 'm519_signal_handler TERM' TERM

m519_failure_cleanup() {
    local rc=$?
    set +e
    if [[ -n "${m519_child_pid}" ]] && kill -0 "${m519_child_pid}" 2>/dev/null; then
        kill -TERM "${m519_child_pid}" 2>/dev/null
        wait "${m519_child_pid}"
        m519_child_rc=$?
    fi
    if [[ -n "${m519_monitor_pid}" ]] && kill -0 "${m519_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m519_monitor_pid}" 2>/dev/null
        wait "${m519_monitor_pid}"
        m519_monitor_rc=$?
    fi
    if [[ "${m519_run_created}" -eq 1 && "${m519_complete}" -ne 1 \
          && -d "${m519_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nruntime_resource_latch=%s\n' \
            "${rc}" "${m519_child_rc}" "${m519_monitor_rc}" \
            "${m519_signal}" "${m519_runtime_latch}" \
            >"${m519_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        (cd "${m519_work}" && \
            find . -type f ! -name SHA256SUMS \
                ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | \
                xargs -0 sha256sum >SHA256SUMS && \
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
            sha256sum -c SHA256SUMS >/dev/null && \
            sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
        mv -T "${m519_work}" "${m519_quarantine}"
        m519_run_created=0
    fi
    return "${rc}"
}
trap m519_failure_cleanup EXIT

mkdir "${m519_work}/.attempt_staging"
printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m519_canonical}" \
    >"${m519_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m519_runner}" "${m519_contract}" "${m519_admission}" \
    >"${m519_work}/.attempt_staging/identity.sha256"
(cd "${m519_work}/.attempt_staging" && \
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
mv -T "${m519_work}/.attempt_staging" "${m519_attempt}"

sha256sum "${m519_runner}" "${m519_contract}" "${m519_admission}" \
    "${m519_tcl}" "${m519_filelist}" "${m519_sdc}" "${m519_dc}" \
    "${m519_slow}" "${m519_fast}" docs/359_DATE终局冻结_20260813.md \
    >"${m519_work}/input_sha256.txt"
cp "${m519_contract}" "${m519_work}/contract.json"

export HW_ROOT="${m519_hw_root}"
export LIB_DB="${m519_slow}"
export MIN_LIB_DB="${m519_fast}"
export SDC_FILE="${m519_hw_root}/${m519_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m519_monitor() {
    local child=$1 log=$2
    local failed=0
    while kill -0 "${child}" 2>/dev/null; do
        m519_resource_snapshot runtime "${log}" || failed=1
        sleep 10
    done
    m519_resource_snapshot runtime_final "${log}" || failed=1
    printf 'runtime_resource_latch=%s\n' "${failed}" >>"${log}"
    return "${failed}"
}

m519_run_point() {
    local id=$1 mode=$2
    local point="${m519_work}/${id}"
    mkdir "${point}"
    export DESIGN_NAME=m519_fc2_registered_release_matched_8bank_raw4_acc24
    export RTL_FILELIST="${m519_hw_root}/${m519_filelist}"
    export OUTPUT_DIR="${point}"
    export ELAB_PARAMETERS="ARCH_MODE=${mode}"
    m519_child_pid=""
    m519_monitor_pid=""
    m519_child_rc="running"
    m519_monitor_rc="running"
    set +e
    "${m519_dc}" -f "${m519_hw_root}/${m519_tcl}" \
        >"${point}/dc.log" 2>&1 &
    m519_child_pid=$!
    m519_monitor "${m519_child_pid}" "${point}/resource_runtime.log" &
    m519_monitor_pid=$!
    wait "${m519_child_pid}"
    m519_child_rc=$?
    wait "${m519_monitor_pid}"
    m519_monitor_rc=$?
    set -e
    echo "${m519_child_rc}" >"${point}/dc.rc"
    echo "${m519_monitor_rc}" >"${point}/runtime_monitor.rc"
    m519_child_pid=""
    m519_monitor_pid=""
    [[ "${m519_signal}" == none ]] || return 130
    [[ "${m519_child_rc}" -eq 0 ]] || return "${m519_child_rc}"
    [[ "${m519_monitor_rc}" -eq 0 ]] || {
        m519_runtime_latch=1
        return 42
    }
    [[ -s "${point}/TCL_PASS_TERMINAL.txt" ]] || return 43
    grep -Fxq 'status=PASS_M519_R5_DC_TCL_TERMINAL' \
        "${point}/TCL_PASS_TERMINAL.txt" || return 43
    [[ ! -e "${point}/TCL_EXPLICIT_FAILURE.txt" ]] || return 43
    grep -Fxq 'TIM-209=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'OPT-150=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${point}/reports/precompile_loop_gate.rpt"
    # dc_shell -f echoes Tcl source lines, including the literal diagnostic
    # tokens used by the fail-closed precompile counter.  Reject emitted tool
    # diagnostics and Tcl failures, while leaving the source echo to the
    # authoritative precompile_loop_gate.rpt checks above.
    ! grep -Eq '^(Warning|Information):.*\((TIM-209|OPT-150)\)|^Error:|^Fatal:' \
        "${point}/dc.log" \
        || return 44
    for report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt; do
        [[ -s "${point}/reports/${report}" ]] || return 45
    done
    [[ -s "${point}/netlist/m519_fc2_registered_release_matched_8bank_raw4_acc24_mapped.v" ]] \
        || return 45
    ! grep -Fq 'slack (VIOLATED)' "${point}/reports/timing_setup.rpt" \
        "${point}/reports/timing_hold.rpt" || return 46
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "${point}/reports/constraint_violators.rpt")" -eq 5 ]] || return 46
    printf 'status=PASS_M519_R5_%s_LOGIC_ONLY_DC_3NS_CLEAN\nmacro_count=0\npaper_ppa_ready=false\nsystem_speedup=false\n' \
        "${id^^}" >"${point}/RUN_COMPLETE.txt"
}

m519_run_point k1 0
m519_run_point k8 1
m519_run_point k1x8 2
printf 'PASS_M519_R5_THREE_AXIS_LOGIC_ONLY_DC_PENDING_RECEIPT_REVIEW\n' \
    >"${m519_work}/RUN_COMPLETE.txt"
(
    cd "${m519_work}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
mv -T "${m519_work}" "${m519_canonical}"
m519_run_created=0
m519_complete=1
trap - EXIT INT TERM
echo "PASS M519 R5 DC raw result sealed at ${m519_canonical}"
