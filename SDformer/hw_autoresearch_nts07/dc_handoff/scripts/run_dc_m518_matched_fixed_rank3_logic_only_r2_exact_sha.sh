#!/usr/bin/env bash
set -euo pipefail

m518_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m518_hw_root="$(cd "${m518_dc_root}/.." && pwd)"
m518_runner="$(realpath "${BASH_SOURCE[0]}")"
m518_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m518_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m518_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m518_filelist=dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f
m518_sdc=dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc
m518_tcl=dc_handoff/scripts/run_dc_m518_matched_fixed_rank3_logic_only_r2.tcl
m518_contract=contracts/m518_matched_fixed_rank3_logic_only_dc_contract_r2_20260827.json
m518_admission=contracts/m518_matched_fixed_rank3_logic_only_dc_launch_admission_r2_20260827.json
m518_fixed_result=results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827
m518_fixed_review=reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827
m518_rank3_result=results/m285_m273r2_glitch_clean_zero_tile_vcs_r1_exact_20260825
m518_rank3_review=results/m286_m285_m273r2_independent_review_r1_20260825
m518_r1_static_review=reviews/m518_matched_fixed_rank3_dc_static_hammer_r1_20260827
m518_canonical="${m518_dc_root}/runs/m518_matched_fixed_rank3_logic_only_dc_3p000ns_r2_20260827"
m518_work="${m518_dc_root}/runs/.m518_matched_fixed_rank3_dc_r2_work.$$"
m518_attempt="${m518_dc_root}/runs/.m518_matched_fixed_rank3_logic_only_dc_r2_attempt_consumed"
m518_quarantine="${m518_canonical}.failed_or_incomplete.$$.quarantine"
m518_preflight_log="${m518_dc_root}/runs/.m518_matched_fixed_rank3_resource_preflight.$$.log"

m518_sha() { sha256sum "$1" | awk '{print $1}'; }
m518_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m518_sha "${path}")" == "${expected}" ]] || {
        echo "M518 matched DC identity mismatch: ${path}" >&2
        exit 3
    }
}

# Live self identity, launch admission, canonical collision and tool collision
# gates all precede resource sampling, work-directory creation and attempt use.
[[ -n "${M518_MATCHED_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m518_sha "${m518_runner}")" == \
   "${M518_MATCHED_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M518 matched DC caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M518_MATCHED_EXPECTED_DC_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M518 matched DC launch admission is not caller-pinned" >&2
    exit 3
}
[[ -z "${M518_MATCHED_DC_RUN_DIR:-}" ]] || {
    echo "M518 matched DC canonical path override is forbidden" >&2
    exit 5
}
[[ ! -e "${m518_canonical}" && ! -e "${m518_work}" && \
   ! -e "${m518_attempt}" && ! -e "${m518_quarantine}" ]] || {
    echo "M518 matched DC refuses consumed or colliding result identity" >&2
    exit 5
}
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null || \
        pgrep -u "$(id -u)" -x vcs >/dev/null || \
        pgrep -u "$(id -u)" -x vcs1 >/dev/null || \
        pgrep -u "$(id -u)" -x vlogan >/dev/null || \
        pgrep -u "$(id -u)" -x simv >/dev/null; then
    echo "M518 matched DC refuses DC/VCS/FM/PT collision" >&2
    exit 4
fi

cd "${m518_hw_root}"
m518_expect "${m518_contract}" \
    18ae1c4fc48e421720ea41ffeb76528c2efe56264d3d3eaf5affda4ba364860d
m518_expect "${m518_admission}" \
    "${M518_MATCHED_EXPECTED_DC_LAUNCH_ADMISSION_SHA256}"
jq -e '.status == "AUTHORIZED_ONE_M518_MATCHED_FIXED_RANK3_R2_DC_ATTEMPT"
       and .authorization.run_dc == true
       and .authorization.max_attempts == 1
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false' \
    "${m518_admission}" >/dev/null || exit 3
for m518_key in author_contract_sha256 dc_runner_sha256 dc_tcl_sha256 \
        dc_filelist_sha256 sdc_sha256 slow_db_sha256 fast_db_sha256 \
        fixed_vcs_result_outer_seal_file_sha256 \
        fixed_vcs_review_outer_seal_file_sha256 \
        rank3_vcs_result_outer_seal_file_sha256 \
        rank3_vcs_review_outer_seal_file_sha256 \
        r1_static_review_verdict_sha256 \
        r1_static_review_outer_seal_file_sha256; do
    m518_value="$(jq -er ".identity.${m518_key}" "${m518_admission}")"
    [[ "${m518_value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m518_admission}")" == \
   "${M518_MATCHED_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.docs359_sha256' "${m518_admission}")" == \
   dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 ]] \
    || exit 3

m518_expect "${m518_contract}" \
    "$(jq -er '.identity.author_contract_sha256' "${m518_admission}")"
m518_expect "${m518_tcl}" \
    "$(jq -er '.identity.dc_tcl_sha256' "${m518_admission}")"
m518_expect "${m518_filelist}" \
    "$(jq -er '.identity.dc_filelist_sha256' "${m518_admission}")"
m518_expect "${m518_sdc}" \
    "$(jq -er '.identity.sdc_sha256' "${m518_admission}")"
m518_expect "${m518_slow}" \
    "$(jq -er '.identity.slow_db_sha256' "${m518_admission}")"
m518_expect "${m518_fast}" \
    "$(jq -er '.identity.fast_db_sha256' "${m518_admission}")"
m518_expect "${m518_dc}" \
    23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m518_expect rtl_m518/m518_matched_fixed_t10_atlif.sv \
    8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6
m518_expect rtl_m273/m273_integrated_rank3_atlif.sv \
    11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d
m518_expect docs/359_DATE终局冻结_20260813.md \
    dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
m518_expect "${m518_fixed_result}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.fixed_vcs_result_outer_seal_file_sha256' "${m518_admission}")"
m518_expect "${m518_fixed_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.fixed_vcs_review_outer_seal_file_sha256' "${m518_admission}")"
m518_expect "${m518_rank3_result}/RUN_MANIFEST.seal.sha256" \
    "$(jq -er '.identity.rank3_vcs_result_outer_seal_file_sha256' "${m518_admission}")"
m518_expect "${m518_rank3_review}/RUN_MANIFEST.seal.sha256" \
    "$(jq -er '.identity.rank3_vcs_review_outer_seal_file_sha256' "${m518_admission}")"
m518_expect "${m518_r1_static_review}/m518_matched_fixed_rank3_dc_static_hammer_verdict_r1.json" \
    "$(jq -er '.identity.r1_static_review_verdict_sha256' "${m518_admission}")"
m518_expect "${m518_r1_static_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.r1_static_review_outer_seal_file_sha256' "${m518_admission}")"
jq -e '.status == "NEEDS_REVISION__R1_LAUNCH_NOT_AUTHORIZED"
       and .authorization.root_may_sign_r1_launch_admission == false
       and .authorization.run_dc == false' \
    "${m518_r1_static_review}/m518_matched_fixed_rank3_dc_static_hammer_verdict_r1.json" \
    >/dev/null || exit 3
(cd "${m518_fixed_result}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
(cd "${m518_fixed_review}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
(cd "${m518_rank3_result}" && sha256sum -c RUN_MANIFEST.sha256 >/dev/null && \
    sha256sum -c RUN_MANIFEST.seal.sha256 >/dev/null) || exit 3
(cd "${m518_rank3_review}" && sha256sum -c RUN_MANIFEST.sha256 >/dev/null && \
    sha256sum -c RUN_MANIFEST.seal.sha256 >/dev/null) || exit 3
(cd "${m518_r1_static_review}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3

# Compare the synthesis-visible top-level port signatures from source before
# consuming an attempt.  The M518 VCS-only conditional harness is excluded.
python3 - <<'PY'
import re
from pathlib import Path

def signature(path, module):
    text = Path(path).read_text()
    start = text.index("module " + module)
    body = text[text.index(") (", start) + 3:text.index(");", start)]
    result = []
    skip = False
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("`ifdef"):
            skip = True
            continue
        if line.startswith("`endif"):
            skip = False
            continue
        if skip:
            continue
        line = line.split("//", 1)[0].strip().lstrip(",").rstrip(",").strip()
        if not line:
            continue
        match = re.fullmatch(
            r"(input|output)\s+logic\s*(\[[^]]+\])?\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)", line)
        if not match:
            raise SystemExit("unparsed port declaration: " + line)
        result.append((match.group(1),
                       re.sub(r"\s+", "", match.group(2) or ""),
                       match.group(3)))
    return result

fixed = signature("rtl_m518/m518_matched_fixed_t10_atlif.sv",
                  "m518_matched_fixed_t10_atlif")
rank3 = signature("rtl_m273/m273_integrated_rank3_atlif.sv",
                  "m273_integrated_rank3_atlif")
if fixed != rank3 or len(fixed) != 50:
    raise SystemExit("M518 Fixed/rank3 synthesis-visible port mismatch")
PY

m518_resource_snapshot() {
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

trap 'rm -f "${m518_preflight_log}"' EXIT
for m518_sample in 1 2 3; do
    m518_resource_snapshot "preflight_${m518_sample}" \
        "${m518_preflight_log}" || exit 40
done

# Only this point consumes the one allowed attempt.
mkdir "${m518_work}"
cp "${m518_preflight_log}" "${m518_work}/resource_preflight.log"
rm -f "${m518_preflight_log}"
trap - EXIT
m518_run_created=1
m518_complete=0
m518_child_pid=""
m518_monitor_pid=""
m518_child_rc="not_started"
m518_monitor_rc="not_started"
m518_signal="none"
m518_runtime_latch=0

m518_signal_handler() {
    local signal_name=$1
    m518_signal="${signal_name}"
    printf 'timestamp=%s signal=%s child_pid=%s monitor_pid=%s\n' \
        "$(date --iso-8601=seconds)" "${signal_name}" \
        "${m518_child_pid:-none}" "${m518_monitor_pid:-none}" \
        >>"${m518_work}/signal_provenance.txt"
    if [[ -n "${m518_child_pid}" ]] && kill -0 "${m518_child_pid}" 2>/dev/null; then
        kill -s "${signal_name}" "${m518_child_pid}" 2>/dev/null || true
    fi
    if [[ -n "${m518_monitor_pid}" ]] && kill -0 "${m518_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m518_monitor_pid}" 2>/dev/null || true
    fi
}
trap 'm518_signal_handler INT' INT
trap 'm518_signal_handler TERM' TERM

m518_failure_cleanup() {
    local rc=$?
    set +e
    if [[ -n "${m518_child_pid}" ]] && kill -0 "${m518_child_pid}" 2>/dev/null; then
        kill -TERM "${m518_child_pid}" 2>/dev/null
        wait "${m518_child_pid}"
        m518_child_rc=$?
    fi
    if [[ -n "${m518_monitor_pid}" ]] && kill -0 "${m518_monitor_pid}" 2>/dev/null; then
        kill -TERM "${m518_monitor_pid}" 2>/dev/null
        wait "${m518_monitor_pid}"
        m518_monitor_rc=$?
    fi
    if [[ "${m518_run_created}" -eq 1 && "${m518_complete}" -ne 1 \
          && -d "${m518_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nruntime_resource_latch=%s\n' \
            "${rc}" "${m518_child_rc}" "${m518_monitor_rc}" \
            "${m518_signal}" "${m518_runtime_latch}" \
            >"${m518_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        (cd "${m518_work}" && \
            find . -type f ! -name SHA256SUMS \
                ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | \
                xargs -0 sha256sum >SHA256SUMS && \
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
            sha256sum -c SHA256SUMS >/dev/null && \
            sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
        mv -T "${m518_work}" "${m518_quarantine}"
        m518_run_created=0
    fi
    return "${rc}"
}
trap m518_failure_cleanup EXIT

mkdir "${m518_work}/.attempt_staging"
printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m518_canonical}" \
    >"${m518_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m518_runner}" "${m518_contract}" "${m518_admission}" \
    >"${m518_work}/.attempt_staging/identity.sha256"
(cd "${m518_work}/.attempt_staging" && \
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
mv -T "${m518_work}/.attempt_staging" "${m518_attempt}"

sha256sum "${m518_runner}" "${m518_contract}" "${m518_admission}" \
    "${m518_tcl}" "${m518_filelist}" "${m518_sdc}" "${m518_dc}" \
    "${m518_slow}" "${m518_fast}" \
    rtl_m518/m518_matched_fixed_t10_atlif.sv \
    rtl_m273/m273_integrated_rank3_atlif.sv \
    docs/359_DATE终局冻结_20260813.md \
    >"${m518_work}/input_sha256.txt"
cp "${m518_contract}" "${m518_work}/contract.json"
cp "${m518_admission}" "${m518_work}/launch_admission.json"

export HW_ROOT="${m518_hw_root}"
export LIB_DB="${m518_slow}"
export MIN_LIB_DB="${m518_fast}"
export RTL_FILELIST="${m518_hw_root}/${m518_filelist}"
export SDC_FILE="${m518_hw_root}/${m518_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m518_monitor() {
    local child=$1 log=$2 failed=0
    while kill -0 "${child}" 2>/dev/null; do
        m518_resource_snapshot runtime "${log}" || failed=1
        sleep 10
    done
    m518_resource_snapshot runtime_final "${log}" || failed=1
    printf 'runtime_resource_latch=%s\n' "${failed}" >>"${log}"
    return "${failed}"
}

m518_run_point() {
    local id=$1 top=$2
    local point="${m518_work}/${id}"
    mkdir "${point}"
    export DESIGN_NAME="${top}"
    export OUTPUT_DIR="${point}"
    m518_child_pid=""
    m518_monitor_pid=""
    m518_child_rc="running"
    m518_monitor_rc="running"
    set +e
    "${m518_dc}" -f "${m518_hw_root}/${m518_tcl}" \
        >"${point}/dc.log" 2>&1 &
    m518_child_pid=$!
    m518_monitor "${m518_child_pid}" "${point}/resource_runtime.log" &
    m518_monitor_pid=$!
    wait "${m518_child_pid}"
    m518_child_rc=$?
    wait "${m518_monitor_pid}"
    m518_monitor_rc=$?
    set -e
    echo "${m518_child_rc}" >"${point}/dc.rc"
    echo "${m518_monitor_rc}" >"${point}/runtime_monitor.rc"
    m518_child_pid=""
    m518_monitor_pid=""
    [[ "${m518_signal}" == none ]] || return 130
    [[ "${m518_child_rc}" -eq 0 ]] || return "${m518_child_rc}"
    [[ "${m518_monitor_rc}" -eq 0 ]] || {
        m518_runtime_latch=1
        return 42
    }
    [[ -s "${point}/TCL_PASS_TERMINAL.txt" ]] || return 43
    grep -Fxq 'status=PASS_M518_MATCHED_R2_DC_TCL_TERMINAL' \
        "${point}/TCL_PASS_TERMINAL.txt" || return 43
    grep -Fxq "design=${top}" "${point}/TCL_PASS_TERMINAL.txt" || return 43
    [[ ! -e "${point}/TCL_EXPLICIT_FAILURE.txt" ]] || return 43
    grep -Fxq 'TIM-209=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'OPT-150=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'sources=precompile_build.rpt,check_design_precompile.rpt,check_timing_precompile.rpt' \
        "${point}/reports/precompile_loop_gate.rpt"
    for precompile_report in precompile_build.rpt check_design_precompile.rpt \
            check_timing_precompile.rpt; do
        ! grep -Eq 'TIM-209|OPT-150' \
            "${point}/reports/${precompile_report}" || return 44
    done
    ! grep -Eq '^(Warning|Information):.*\((TIM-209|OPT-150)\)|^Error:|^Fatal:' \
        "${point}/dc.log" || return 44
    for report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
            constraint_violators.rpt check_design_postcompile.rpt \
            check_timing_postcompile.rpt hierarchy_postcompile.rpt \
            resources_postcompile.rpt references_postcompile.rpt ports.rpt \
            port_count.txt precompile_build.rpt check_design_precompile.rpt \
            check_timing_precompile.rpt precompile_loop_gate.rpt \
            unconstrained_audit_route.rpt; do
        [[ -s "${point}/reports/${report}" ]] || return 45
    done
    [[ -s "${point}/netlist/${top}_mapped.v" \
       && -s "${point}/netlist/${top}_mapped.sdc" \
       && -s "${point}/netlist/${top}.ddc" \
       && -s "${point}/netlist/${top}.svf" ]] || return 45
    grep -Fq 'slack (MET)' "${point}/reports/timing_setup.rpt" || return 46
    grep -Fq 'slack (MET)' "${point}/reports/timing_hold.rpt" || return 46
    ! grep -Fq 'slack (VIOLATED)' "${point}/reports/timing_setup.rpt" \
        "${point}/reports/timing_hold.rpt" || return 46
    [[ "$(grep -Fc 'This design has no violated constraints.' \
        "${point}/reports/constraint_violators.rpt")" -eq 5 ]] || return 46
    [[ "$(awk 'NF{last=$0} END{gsub(/[[:space:]]/,"",last);print last}' \
        "${point}/reports/check_design_postcompile.rpt")" == 1 ]] || return 47
    [[ "$(awk 'NF{last=$0} END{gsub(/[[:space:]]/,"",last);print last}' \
        "${point}/reports/check_timing_postcompile.rpt")" == 1 ]] || return 47
    grep -Eq 'Checking .*unconstrained_endpoints' \
        "${point}/reports/check_timing_postcompile.rpt" || return 47
    ! grep -Eqi 'unconstrained endpoint.*(found|exist|[1-9])' \
        "${point}/reports/check_timing_postcompile.rpt" || return 47
    ! grep -Eiq 'inferred latch|latch inferred' \
        "${point}/dc.log" "${point}/reports/check_design_postcompile.rpt" \
        || return 48
    ! grep -Eiq 'multiply driven|multiple driver|multiply-driven' \
        "${point}/dc.log" "${point}/reports/check_design_postcompile.rpt" \
        || return 48
    ! grep -Eiq 'unresolved reference|black[ -]?box' \
        "${point}/dc.log" "${point}/reports/check_design_postcompile.rpt" \
        || return 48
    grep -Fq 'Number of macros/black boxes:               0' \
        "${point}/reports/area.rpt" || return 48
    printf 'unconstrained_endpoints=0\ninferred_latches=0\nmultiply_or_multiple_drivers=0\nunresolved_references=0\nblack_boxes=0\n' \
        >"${point}/reports/structural_cleanliness_audit.txt"

    local area comb_area seq_area cells seq_cells levels path setup hold ports
    area=$(awk '/Total cell area:/ {print $4; exit}' "${point}/reports/area.rpt")
    comb_area=$(awk '/Combinational area:/ {print $3; exit}' "${point}/reports/area.rpt")
    seq_area=$(awk '/Noncombinational area:/ {print $3; exit}' "${point}/reports/area.rpt")
    cells=$(awk '/Number of cells:/ {print $4; exit}' "${point}/reports/area.rpt")
    seq_cells=$(awk '/Number of sequential cells:/ {print $5; exit}' "${point}/reports/area.rpt")
    levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${point}/reports/qor.rpt")
    path=$(awk '/Critical Path Length:/ {print $4; exit}' "${point}/reports/qor.rpt")
    setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${point}/reports/timing_setup.rpt")
    hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${point}/reports/timing_hold.rpt")
    ports=$(tr -d '[:space:]' <"${point}/reports/port_count.txt")
    for value in "${area}" "${comb_area}" "${seq_area}" "${cells}" \
            "${seq_cells}" "${levels}" "${path}" "${setup}" "${hold}" \
            "${ports}"; do
        [[ -n "${value}" ]] || return 49
    done
    awk -v x="${area}" 'BEGIN{exit !(x>0 && x<500000)}' || return 49
    awk -v x="${setup}" 'BEGIN{exit !(x>=0)}' || return 49
    awk -v x="${hold}" 'BEGIN{exit !(x>=0)}' || return 49
    [[ "${ports}" -eq 50 ]] || return 49
    printf 'status=PASS_M518_MATCHED_%s_LOGIC_ONLY_DC_3NS_CLEAN\ndesign=%s\ncell_area_um2=%s\ncombinational_area_um2=%s\nsequential_area_um2=%s\ncell_count=%s\nsequential_cells=%s\nlogic_levels=%s\ncritical_path_length_ns=%s\nsetup_worst_slack_ns=%s\nhold_worst_slack_ns=%s\nreported_port_count=%s\nmacro_count=0\nclock_period_ns=3.000\nclock_network=ideal\nwireload=ZeroWireload\npaper_ppa_ready=false\nsystem_speedup=false\nheadline=false\n' \
        "${id^^}" "${top}" "${area}" "${comb_area}" "${seq_area}" \
        "${cells}" "${seq_cells}" "${levels}" "${path}" "${setup}" \
        "${hold}" "${ports}" >"${point}/RUN_COMPLETE.txt"
}

m518_run_point fixed m518_matched_fixed_t10_atlif
m518_run_point rank3 m273_integrated_rank3_atlif

[[ "$(awk -F= '$1=="reported_port_count"{print $2}' \
    "${m518_work}/fixed/RUN_COMPLETE.txt")" == \
   "$(awk -F= '$1=="reported_port_count"{print $2}' \
    "${m518_work}/rank3/RUN_COMPLETE.txt")" ]] || exit 50

python3 - "${m518_work}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
def receipt(point):
    out = {}
    for line in (root / point / "RUN_COMPLETE.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out

fixed = receipt("fixed")
rank3 = receipt("rank3")
fa = float(fixed["cell_area_um2"])
ra = float(rank3["cell_area_um2"])
result = {
    "schema": "m518_matched_fixed_rank3_logic_only_dc_author_receipt_v2",
    "status": "PASS_RAW_MATCHED_LOGIC_ONLY_DC__AWAITING_INDEPENDENT_RECEIPT_REVIEW",
    "scope": "standalone complete ATLIF protocol tops; flattened standard-cell logic only; zero macros",
    "clock_period_ns": 3.0,
    "cycle_denominators": {
        "fixed_N1": 29,
        "fixed_N4": 80,
        "rank3_N1": 24,
        "rank3_N4": 39,
        "source": "separately sealed Synopsys VCS receipts"
    },
    "dc": {
        "fixed": fixed,
        "rank3": rank3
    },
    "comparisons": {
        "rank3_over_fixed_N1_throughput": 29 / 24,
        "rank3_over_fixed_N4_throughput": 80 / 39,
        "rank3_area_over_fixed_area": ra / fa,
        "rank3_over_fixed_N4_throughput_per_area": (80 * fa) / (39 * ra),
        "fixed_N4_tiles_per_second": (4 / 80) / 3e-9,
        "rank3_N4_tiles_per_second": (4 / 39) / 3e-9,
        "fixed_N4_tiles_per_second_per_um2": ((4 / 80) / 3e-9) / fa,
        "rank3_N4_tiles_per_second_per_um2": ((4 / 39) / 3e-9) / ra
    },
    "claim_boundary": {
        "author_raw_dc": True,
        "independent_receipt_review": False,
        "logic_only": True,
        "macro_inclusive_ppa": False,
        "power": False,
        "energy": False,
        "trained_rank3_accuracy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False
    }
}
def finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        raise SystemExit("nonfinite receipt value")
    if isinstance(x, dict):
        for value in x.values(): finite(value)
    elif isinstance(x, list):
        for value in x: finite(value)
finite(result)
(root / "m518_matched_fixed_rank3_logic_only_dc_author_receipt_r2.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

printf 'PASS_M518_MATCHED_FIXED_RANK3_R2_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RECEIPT_REVIEW\n' \
    >"${m518_work}/RUN_COMPLETE.txt"
(
    cd "${m518_work}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
mv -T "${m518_work}" "${m518_canonical}"
m518_run_created=0
m518_complete=1
trap - EXIT INT TERM
echo "PASS M518 matched r2 DC raw result sealed at ${m518_canonical}"
