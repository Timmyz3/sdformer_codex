#!/usr/bin/env bash
set -euo pipefail

m514_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m514_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m514_hw="$(cd "${m514_dc_root}/.." && pwd)"
m514_canonical="${m514_hw}/results/m514_c2d_directed_vcs_r1_20260827"
m514_attempt="${m514_hw}/results/.m514_c2d_directed_vcs_r1_attempt_consumed"
m514_work="${m514_hw}/results/.m514_c2d_directed_vcs_r1_work.$$"
m514_vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
m514_filelist=dc_handoff/filelists/date_m514_c2d_directed_vcs.f
m514_rtl=rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv
m514_tb=dc_handoff/tb/tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv
m514_contract=contracts/m514_c2d_directed_vcs_contract_r1_20260827.json
m514_review=reviews/m514_c2d_static_hammer_r3_20260827/SHA256SUMS

[[ -n "${M514_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m514_runner}" | awk '{print $1}')" == \
   "${M514_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M514 caller must pin the independently reviewed runner SHA" >&2
    exit 3
}
[[ ! -e "${m514_canonical}" && ! -e "${m514_attempt}" && \
   ! -e "${m514_work}" ]] || {
    echo "M514 refuses to overwrite canonical/attempt/work" >&2
    exit 4
}

mkdir "${m514_work}"
m514_complete=0
m514_attempt_live=0
m514_preflight_quarantine="${m514_canonical}.preflight_failed.$$.quarantine"
m514_failed_quarantine="${m514_canonical}.failed_or_incomplete.$$.quarantine"
m514_cleanup() {
    local m514_rc=$?
    if [[ "${m514_complete}" -ne 1 && -d "${m514_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${m514_rc}" >"${m514_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        if [[ "${m514_attempt_live}" -eq 0 ]]; then
            [[ ! -e "${m514_preflight_quarantine}" ]]
            mv -T "${m514_work}" "${m514_preflight_quarantine}"
        else
            [[ ! -e "${m514_failed_quarantine}" ]]
            mv -T "${m514_work}" "${m514_failed_quarantine}"
        fi
    fi
    return "${m514_rc}"
}
trap m514_cleanup EXIT

cd "${m514_hw}"
m514_sha() { sha256sum "$1" | awk '{print $1}'; }
m514_expect() {
    local m514_path=$1 m514_expected=$2
    [[ -f "${m514_path}" && ! -L "${m514_path}" && \
       "$(m514_sha "${m514_path}")" == "${m514_expected}" ]]
}
m514_require_no_match() {
    local m514_pattern=$1
    shift
    local m514_grep_rc
    set +e
    grep -Eiq -- "${m514_pattern}" "$@"
    m514_grep_rc=$?
    set -e
    [[ "${m514_grep_rc}" -eq 1 ]]
}
m514_verify_inputs() {
    m514_expect "${m514_runner}" "${M514_EXPECTED_RUNNER_SHA256}"
    m514_expect "${m514_vcs}" \
        0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
    m514_expect "${m514_rtl}" \
        90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a
    m514_expect "${m514_tb}" \
        6c283bf94d6933e6aa866428f63d6a8b9a2066da2deb39220301f781ec3df47a
    m514_expect "${m514_filelist}" \
        0a0dbfb33d429566e695afbdbcf48b5081e25fac30d925956a5e96804658adbc
    m514_expect "${m514_contract}" \
        60e4fe5921a374f399bef82fd1902718428bb8f9d6f3d86dc5d03bda7953ab5b
    m514_expect "${m514_review}" \
        20eb76fa32976d4789581c921fae6247c7cee254c090665b922e09609751177e
    m514_expect docs/359_DATE终局冻结_20260813.md \
        dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
    (cd reviews/m514_c2d_static_hammer_r3_20260827 && \
        sha256sum -c SHA256SUMS >/dev/null)
}

m514_process_resource_gate() {
    local m514_log=$1 m514_failures=0
    : >"${m514_log}"
    for m514_sample in 1 2 3; do
        local m514_limit m514_committed m514_available m514_swap
        local m514_headroom m514_failcnt m514_under_oom m514_oom_kill
        m514_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
        m514_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
        m514_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        m514_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
        m514_headroom=$((m514_limit - m514_committed))
        m514_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
        m514_under_oom=$(awk '/^under_oom / {print $2}' \
            /sys/fs/cgroup/memory/user.slice/memory.oom_control)
        m514_oom_kill=$(awk '/^oom_kill / {print $2}' \
            /sys/fs/cgroup/memory/user.slice/memory.oom_control)
        printf 'sample=%s timestamp=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s failcnt=%s under_oom=%s oom_kill=%s\n' \
            "${m514_sample}" "$(date --iso-8601=seconds)" \
            "${m514_headroom}" "${m514_available}" "${m514_swap}" \
            "${m514_failcnt}" "${m514_under_oom}" "${m514_oom_kill}" \
            >>"${m514_log}"
        if [[ "${m514_headroom}" -lt 33554432 || \
              "${m514_available}" -lt 134217728 || \
              "${m514_swap}" -lt 33554432 || \
              "${m514_failcnt}" -ne 0 || "${m514_under_oom}" -ne 0 || \
              "${m514_oom_kill}" -ne 0 ]]; then
            m514_failures=$((m514_failures + 1))
        fi
        if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
                pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null || \
                pgrep -x vcs >/dev/null || pgrep -x vcs1 >/dev/null || \
                pgrep -x vlogan >/dev/null || pgrep -x vhdlan >/dev/null || \
                pgrep -f '/common_shell_exec -shell (dc_shell|fm_shell|pt_shell) ' \
                    >/dev/null; then
            printf 'sample=%s forbidden_process=synopsys_eda\n' \
                "${m514_sample}" >>"${m514_log}"
            m514_failures=$((m514_failures + 1))
        fi
        local m514_pid m514_owner m514_state m514_cpu m514_rss
        local m514_user
        m514_user=$(id -un)
        while read -r m514_pid; do
            [[ -n "${m514_pid}" ]] || continue
            m514_owner=$(ps -o user= -p "${m514_pid}" | xargs)
            m514_state=$(ps -o stat= -p "${m514_pid}" | xargs)
            m514_cpu=$(ps -o pcpu= -p "${m514_pid}" | xargs)
            m514_rss=$(ps -o rss= -p "${m514_pid}" | xargs)
            if [[ "${m514_owner}" == "${m514_user}" || \
                  ! "${m514_state}" =~ ^[SI] ]] || \
                    ! awk -v cpu="${m514_cpu}" -v rss="${m514_rss}" \
                        'BEGIN {exit !(cpu <= 0.5 && rss <= 262144)}'; then
                printf 'sample=%s forbidden_simv pid=%s owner=%s stat=%s pcpu=%s rss_kib=%s\n' \
                    "${m514_sample}" "${m514_pid}" "${m514_owner}" \
                    "${m514_state}" "${m514_cpu}" "${m514_rss}" \
                    >>"${m514_log}"
                m514_failures=$((m514_failures + 1))
            else
                printf 'sample=%s allowed_foreign_idle_simv pid=%s owner=%s stat=%s pcpu=%s rss_kib=%s\n' \
                    "${m514_sample}" "${m514_pid}" "${m514_owner}" \
                    "${m514_state}" "${m514_cpu}" "${m514_rss}" \
                    >>"${m514_log}"
            fi
        done < <(pgrep -x simv || true)
        if pgrep -f '(^|[ /])[^ ]*(analyze|independent|sweep|dse|simulate)_m[0-9][^ ]*\.py( |$)' \
                >/dev/null; then
            printf 'sample=%s forbidden_process=project_cpu_dse\n' \
                "${m514_sample}" >>"${m514_log}"
            m514_failures=$((m514_failures + 1))
        fi
        [[ "${m514_sample}" -eq 3 ]] || sleep 5
    done
    [[ "${m514_failures}" -eq 0 ]]
}

m514_verify_inputs
m514_process_resource_gate "${m514_work}/resource_preflight.log"
m514_verify_inputs

sha256sum "${m514_runner}" "${m514_vcs}" "${m514_rtl}" "${m514_tb}" \
    "${m514_filelist}" "${m514_contract}" "${m514_review}" \
    docs/359_DATE终局冻结_20260813.md >"${m514_work}/input_sha256.txt"
cp "${m514_contract}" "${m514_work}/contract.json"
mkdir "${m514_work}/.attempt_staging"
printf 'status=CONSUMED_BEFORE_EXACT_VCS_COMPILE\ncanonical=%s\n' \
    "${m514_canonical}" \
    >"${m514_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m514_runner}" "${m514_vcs}" "${m514_rtl}" "${m514_tb}" \
    "${m514_filelist}" "${m514_contract}" "${m514_review}" \
    docs/359_DATE终局冻结_20260813.md \
    >"${m514_work}/.attempt_staging/identity.sha256"
(cd "${m514_work}/.attempt_staging" && \
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv -T "${m514_work}/.attempt_staging" "${m514_attempt}"
m514_attempt_live=1
[[ "$(find -P "${m514_attempt}" -mindepth 1 -maxdepth 1 -printf '%f\n' | \
    LC_ALL=C sort)" == $'ATTEMPT_CONSUMED.txt\nSHA256SUMS\nSHA256SUMS.seal.sha256\nidentity.sha256' ]]
while IFS= read -r -d '' m514_attempt_member; do
    [[ -f "${m514_attempt_member}" && ! -L "${m514_attempt_member}" ]]
done < <(find -P "${m514_attempt}" -mindepth 1 -maxdepth 1 -print0)

export VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
export VCS_ARCH_OVERRIDE=linux
set +e
"${m514_vcs}" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m514_work}/csrc" -f "${m514_filelist}" \
    -top tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper \
    -o "${m514_work}/simv" >"${m514_work}/compile.log" 2>&1
m514_rc=$?
set -e
printf '%s\n' "${m514_rc}" >"${m514_work}/compile.rc"
[[ "${m514_rc}" -eq 0 && -x "${m514_work}/simv" ]]
m514_require_no_match \
    'Warning-\[|^[[:space:]]*Warning:|Error-\[|^[[:space:]]*Error|^[[:space:]]*Fatal|Fatal:' \
    "${m514_work}/compile.log"

set +e
"${m514_work}/simv" +ntb_random_seed=514027 -no_save -cm assert \
    -assert report="${m514_work}/assert.report" \
    >"${m514_work}/sim.log" 2>&1
m514_rc=$?
set -e
printf '%s\n' "${m514_rc}" >"${m514_work}/sim.rc"
[[ "${m514_rc}" -eq 0 ]]
[[ -f "${m514_work}/sim.log" && ! -L "${m514_work}/sim.log" ]]
[[ -f "${m514_work}/assert.report" && \
   ! -L "${m514_work}/assert.report" ]]
m514_require_no_match \
    'failed at|Offending|^[[:space:]]*Error|^[[:space:]]*Fatal|Fatal:|watchdog|timeout' \
    "${m514_work}/sim.log" "${m514_work}/assert.report"
m514_pass=$(grep -E '^PASS M514 exact_taps=43 stalls=[1-9][0-9]* replacements=[1-9][0-9]* phases=6/10/10/17 protocol_attack=1$' \
    "${m514_work}/sim.log")
[[ "$(printf '%s\n' "${m514_pass}" | grep -c '^PASS M514')" -eq 1 ]]

python3 - "${m514_work}" "${m514_pass}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
line = sys.argv[2]
match = re.fullmatch(
    r"PASS M514 exact_taps=(\d+) stalls=(\d+) replacements=(\d+) "
    r"phases=(\d+)/(\d+)/(\d+)/(\d+) protocol_attack=(\d+)", line)
if match is None:
    raise SystemExit("M514 pass-line parse failure")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m514_c2d_directed_vcs_receipt_v1",
    "status": "PASS_M514_C2D_DIRECTED_FUNCTIONAL_COMPLETENESS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 514027,
    "measured": {
        "exact_taps": values[0],
        "stall_cycles": values[1],
        "same_edge_replacements": values[2],
        "phase_counts_00_01_10_11": values[3:7],
        "protocol_attacks": values[7],
    },
    "coverage": {
        "fanout_4_6_6_9": True,
        "maximum_legal_size32_source31_destination63": True,
        "stalled_tap_illegal_successor_drain": True,
        "same_edge_successor_replacement": True,
    },
    "claim_boundary": {
        "directed_functional_completeness": True,
        "full_decoder_trace": False,
        "cycle_speedup": False,
        "area": False,
        "timing": False,
        "formality": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(root / "m514_c2d_directed_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(root / "RUN_COMPLETE.txt").write_text(
    "PASS_M514_C2D_DIRECTED_FUNCTIONAL_COMPLETENESS\n")
PY

m514_verify_inputs
sha256sum -c "${m514_work}/input_sha256.txt" >/dev/null
(cd "${m514_attempt}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
sha256sum -c "${m514_attempt}/identity.sha256" >/dev/null
(
    cd "${m514_work}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
m514_verify_inputs
sha256sum -c "${m514_work}/input_sha256.txt" >/dev/null
[[ ! -e "${m514_canonical}" ]]
mv -T "${m514_work}" "${m514_canonical}"
m514_complete=1
echo "PASS M514 exact VCS sealed at ${m514_canonical}"
