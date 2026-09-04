#!/usr/bin/env bash
set -euo pipefail

# M803/C2 R16 future one-attempt VCS runner.  This file is source-only until a
# fresh independent source hammer and a separately sealed launch admission are
# present.  The source dry-run terminates before every VCS/license probe and
# before creation of either the attempt sentinel or result identity.

dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "${dc_root}/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
python36=/usr/libexec/platform-python3.6
vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
license_file=/opt/synopsys/Synopsys.dat
source_contract_rel=contracts/m803_c2_r16_channel_split_source_only_contract_r1_20260828.json
source_contract="${hw_root}/${source_contract_rel}"
source_hammer_rel=reviews/m803_c2_r16_channel_split_source_fresh_hammer_r1_20260828
candidate_hammer_rel=reviews/m803_c2_r16_channel_split_vcs_launch_candidate_hammer_r1_20260828
launch_rel=contracts/m803_c2_r16_channel_split_vcs_launch_admission_r1_20260828.json
final_hammer_rel=reviews/m803_c2_r16_channel_split_vcs_launch_final_hammer_r1_20260828
result="${hw_root}/results/m803_c2_r16_channel_split_vcs_r1_20260828"
attempt="${hw_root}/results/.m803_c2_r16_channel_split_vcs_attempt_consumed"
work="${hw_root}/results/.m803_c2_r16_channel_split_vcs_work.$$"
quarantine="${hw_root}/results/m803_c2_r16_channel_split_vcs_r1_20260828.failed_or_incomplete.$$.quarantine"
complete=0
attempt_consumed=0

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M803 R16 gate failure: %s\n' "$*" >&2; exit 3; }
expect_file_sha() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && "$(sha "${path}")" == "${expected}" ]] \
        || fail "SHA mismatch or non-regular file: ${path}"
}
verify_double_seal_dir() {
    local rel=$1 dir="${hw_root}/$1"
    [[ -d "${dir}" && ! -L "${dir}" ]] || fail "missing sealed directory ${rel}"
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
        && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) \
        || fail "double-seal failure ${rel}"
}
verify_double_seal_file() {
    local rel=$1 base="${hw_root}/$1"
    [[ -f "${base}" && ! -L "${base}" ]] || fail "missing sealed file ${rel}"
    (cd "$(dirname "${base}")" \
        && sha256sum -c "$(basename "${base}").sha256" >/dev/null \
        && sha256sum -c "$(basename "${base}").sha256.seal.sha256" >/dev/null) \
        || fail "double-seal file failure ${rel}"
}
json_gate() {
    local path=$1 status=$2 launch=$3
    "${python36}" - "${path}" "${status}" "${launch}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or p.is_symlink():
    raise SystemExit(2)
d = json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != sys.argv[2]:
    raise SystemExit(3)
want = sys.argv[3]
if want != "NA" and d.get("authorization", {}).get("launch_now") is not (want == "true"):
    raise SystemExit(4)
PY
}
verify_source_contract() {
    verify_double_seal_file "${source_contract_rel}"
    "${python36}" - "${hw_root}" "${source_contract}" "$(sha "${runner}")" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
contract = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if contract.get("status") != "SOURCE_ONLY__NO_VCS_AUTHORIZATION":
    raise SystemExit(10)
if contract.get("authorization", {}).get("launch_now") is not False:
    raise SystemExit(11)
if contract.get("runner_sha256") != sys.argv[3]:
    raise SystemExit(12)
for rel, expected in contract.get("source_sha256", {}).items():
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise SystemExit(13)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(14)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(15)
if contract.get("claim_boundary", {}).get("vcs_validated") is not False:
    raise SystemExit(16)
PY
}
verify_m800_authority() {
    local rel=reviews/m800_m519_r15_k8_tim209_failure_hammer_r1_20260828
    verify_double_seal_dir "${rel}"
    json_gate "${hw_root}/${rel}/review.json" \
        PASS_FAILURE_AUDIT__M519_R15_K8_TIM209__THREE_AXIS_CAMPAIGN_NONCITABLE__ADDITIVE_R16_SOURCE_ONLY_AUTHORIZED NA
}
verify_future_launch_chain() {
    verify_double_seal_dir "${source_hammer_rel}"
    json_gate "${hw_root}/${source_hammer_rel}/review.json" \
        PASS_M803_R16_SOURCE__ONE_VCS_CANDIDATE_MAY_BE_AUTHORED false
    verify_double_seal_dir "${candidate_hammer_rel}"
    json_gate "${hw_root}/${candidate_hammer_rel}/review.json" \
        PASS_M803_R16_LAUNCH_CANDIDATE__FINAL_RELEASE_MAY_BE_AUTHORED false
    verify_double_seal_file "${launch_rel}"
    json_gate "${hw_root}/${launch_rel}" \
        AUTHORIZED_ONE_M803_R16_CHANNEL_SPLIT_VCS_ATTEMPT true
    verify_double_seal_dir "${final_hammer_rel}"
    json_gate "${hw_root}/${final_hammer_rel}/review.json" \
        PASS_M803_R16_FINAL_LAUNCH_RELEASE__ONE_VCS_ATTEMPT_AUTHORIZED true
    [[ -n "${M803_R16_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256:-}" \
       && "$(sha "${hw_root}/${final_hammer_rel}/SHA256SUMS.seal.sha256")" \
          == "${M803_R16_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256}" ]] \
        || fail "caller did not pin final hammer outer-seal SHA"
}
seal_dir() {
    local dir=$1
    (cd "${dir}" \
        && find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
            -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS \
        && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 \
        && sha256sum -c SHA256SUMS >/dev/null \
        && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
failure_cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${attempt_consumed}" -eq 1 && "${complete}" -ne 1 ]]; then
        if [[ ! -d "${work}" ]]; then mkdir "${work}"; fi
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nattempt_consumed=1\n' \
            "${rc}" >"${work}/RUN_FAILED_OR_INCOMPLETE.txt"
        if seal_dir "${work}"; then
            if [[ ! -e "${quarantine}" ]]; then mv "${work}" "${quarantine}" || true; fi
        elif [[ ! -e "${quarantine}.UNSEALED_DO_NOT_CITE" ]]; then
            mv "${work}" "${quarantine}.UNSEALED_DO_NOT_CITE" || true
        fi
    fi
    exit "${rc}"
}
signal_exit() {
    trap - INT TERM HUP
    exit 130
}
trace_event() {
    local event=$1
    printf '{"event":"%s","totals":{"vcs_identity_probe_runs":0,"license_server_queries":0,"vcs_compile_runs":0,"simv_runs":0,"result_directories_created":0}}\n' \
        "${event}" >>"${M803_R16_SOURCE_DRY_RUN_TRACE}"
}
source_dry_run() {
    local root=${M803_R16_SOURCE_DRY_RUN_ROOT:-}
    local nonce=${M803_R16_SOURCE_DRY_RUN_NONCE:-}
    local trace=${M803_R16_SOURCE_DRY_RUN_TRACE:-}
    [[ -n "${root}" && -d "${root}" && ! -L "${root}" \
       && -n "${nonce}" && -f "${nonce}" && ! -L "${nonce}" \
       && -n "${trace}" && ! -e "${trace}" \
       && "$(<"${nonce}")" == M803_R16_SOURCE_HAMMER_ONLY ]] \
        || fail "invalid source dry-run nonce/root/trace"
    case "${trace}" in "${root}"/*) ;; *) fail "dry-run trace escapes root" ;; esac
    : >"${trace}"
    trace_event stub_collision_initial
    trace_event stub_cgroup
    trace_event stub_resource
    trace_event stub_collision_final
    trace_event live_probe_boundary_stop
    printf '%s\n' M803_R16_STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY__NO_LIVE_PROBE__NO_RESULT_MKDIR
    exit 86
}
collision_gate() {
    if pgrep -x vcs1 >/dev/null || pgrep -x vlogan >/dev/null \
       || pgrep -f '(^|/)(vcs)( |$)' >/dev/null \
       || pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
       || pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null; then
        fail "VCS/DC/FM/PT collision"
    fi
}
resource_gate() {
    local available_kib commit_limit_kib committed_kib
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    commit_limit_kib=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    committed_kib=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    [[ "${available_kib}" -ge 8388608 \
       && $((commit_limit_kib - committed_kib)) -ge 8388608 ]] \
        || fail "less than 8 GiB memory/commit headroom"
}
license_gate() {
    [[ "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo \
       && "${LM_LICENSE_FILE:-}" == "${license_file}" ]] \
        || fail "license environment mismatch"
    [[ -f "${license_file}" && ! -L "${license_file}" ]] \
        || fail "license file missing/non-regular"
    "${lmutil}" lmstat -a -c "${SNPSLMD_LICENSE_FILE}" >/dev/null 2>&1 \
        || fail "license server status failed"
}
compile_and_run() {
    local phase=$1 filelist=$2 top=$3 seed=$4
    local phase_dir="${work}/${phase}"
    mkdir "${phase_dir}"
    set +e
    "${vcs}" -full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED \
        -timescale=1ns/1ps -cm assert -Mdir="${phase_dir}/csrc" \
        -f "${filelist}" -top "${top}" -o "${phase_dir}/simv" \
        >"${phase_dir}/compile.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "${rc}" >"${phase_dir}/compile.rc"
    [[ "${rc}" -eq 0 && -x "${phase_dir}/simv" ]] || return 20
    ! grep -Eiq 'Error-\[|^Error|^Fatal|Fatal:' "${phase_dir}/compile.log" || return 21
    set +e
    "${phase_dir}/simv" "+ntb_random_seed=${seed}" -no_save \
        -assert report="${phase_dir}/assert.report" -cm assert \
        >"${phase_dir}/sim.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "${rc}" >"${phase_dir}/sim.rc"
    [[ "${rc}" -eq 0 ]] || return 22
    ! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "${phase_dir}/sim.log" "${phase_dir}/assert.report" || return 23
}

# The runner identity is the first gate and therefore a wrong-SHA call has no
# result, attempt, trace, license, or tool side effect.
[[ ! -v HOME ]] || fail "HOME must be absent; runner never synthesizes HOME"
[[ -n "${M803_R16_EXPECTED_VCS_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M803_R16_EXPECTED_VCS_RUNNER_SHA256}" ]] \
    || fail "caller must pin the exact independently reviewed runner SHA"
expect_file_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f
verify_source_contract
verify_m800_authority

if [[ "${M803_R16_SOURCE_DRY_RUN:-0}" == 1 ]]; then source_dry_run; fi

verify_future_launch_chain
expect_file_sha "${vcs}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
expect_file_sha "${lmutil}" e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07
[[ "${VCS_HOME:-}" == /opt/synopsys/vcs/V-2023.12-SP1 \
   && "${VCS_ARCH_OVERRIDE:-}" == linux ]] || fail "VCS environment mismatch"
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
    || fail "R16 result/attempt/work identity already exists"
collision_gate
resource_gate
license_gate
collision_gate

mkdir "${attempt}"
attempt_consumed=1
mkdir "${work}"
trap failure_cleanup EXIT
trap signal_exit INT TERM HUP
printf 'status=ONE_M803_R16_VCS_ATTEMPT_CONSUMED\nrunner_sha256=%s\n' \
    "$(sha "${runner}")" >"${attempt}/ATTEMPT.txt"

cd "${hw_root}"
compile_and_run attack \
    dc_handoff/filelists/date_m803_c2_r16_channel_split_adapter_attacks_vcs.f \
    tb_m803_c2_r16_channel_split_adapter_attacks 803016
grep -Eq '^PASS M803 R16 channel-split cutthrough adapter VCS attack_classes=[1-9][0-9]* reset_cases=[1-9][0-9]* legal_response_on_request_fault=2 same_cycle_reuse_cases=1 sticky_quiescent_checks=[1-9][0-9]* normal_requests=[1-9][0-9]* normal_responses=[1-9][0-9]* request_side_effect_violations=0 response_side_effect_violations=0$' \
    "${work}/attack/sim.log"
for cover in cp_legal_response_illegal_request_same_cycle \
        cp_illegal_response_legal_request_same_cycle \
        cp_pending_drain_request_attack cp_response_backpressure_then_attack \
        cp_held_response_request_attack_retire \
        cp_cutthrough_request_attack_retire cp_same_cycle_slot_reuse \
        cp_sticky_fault_quiescent; do
    grep -Eq "m803_sva\\.${cover}, .* [1-9][0-9]* match" \
        "${work}/attack/assert.report"
done

compile_and_run equalbw \
    dc_handoff/filelists/date_m803_c2_r16_channel_split_k8_vs_k1x8_vcs.f \
    tb_m803_c2_r16_channel_split_k8_vs_k1x8_raw4_acc24 803017
grep -Eq '^PASS M803EQ channel-split cutthrough-8bank equal-bandwidth FC2 VCS clean_cases=10 exact_cycle_cases=5 cycles=51/53,131/133,486/499,1231/1246,14/14 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 service_sva_bound=true adapter_sva_bound=true racefree_cycle_monitor=true request_stalls=[1-9][0-9]* result_stalls=[1-9][0-9]* raw_stalls=[1-9][0-9]* full8_requests=[1-9][0-9]* k1x8_full_issue=[1-9][0-9]* candidate_younger_before_older=[1-9][0-9]* baseline_younger_before_older=[1-9][0-9]*$' \
    "${work}/equalbw/sim.log"
for row in \
    'B=1 events=20 k8_cycles=51 k1x8_cycles=53' \
    'B=2 events=41 k8_cycles=131 k1x8_cycles=133' \
    'B=4 events=90 k8_cycles=486 k1x8_cycles=499' \
    'B=8 events=110 k8_cycles=1231 k1x8_cycles=1246' \
    'B=1 events=0 k8_cycles=14 k1x8_cycles=14'; do
    grep -Fq "M803EQ cutthrough equalbw ${row}" "${work}/equalbw/sim.log"
done

"${python36}" - "${work}" "$(sha "${runner}")" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
receipt = {
  "schema": "m803_c2_r16_channel_split_vcs_receipt_v1",
  "status": "PASS_M803_R16_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER",
  "runner_sha256": sys.argv[2],
  "tool": "Synopsys VCS V-2023.12-SP1",
  "attack_contract": {"same_cycle_slot_reuse": 1, "ledger_conservation": True,
    "illegal_response_closes_both": True, "legal_response_survives_request_fault": True},
  "exact_cycles": {"k8": [51,131,486,1231,14],
    "k1x8": [53,133,499,1246,14]},
  "frozen_k1_vs_k1x8": "SOURCE_SHA_BOUND_ONLY__NOT_RERUN_OR_CHANGED",
  "claim_boundary": {"vcs_validated": True, "dc": False, "ppa": False,
    "system_speedup": False, "headline": False}
}
(root / "m803_c2_r16_channel_split_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
printf '%s\n' PASS_M803_R16_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER \
    >"${work}/RUN_COMPLETE.txt"
seal_dir "${work}"
mv "${work}" "${result}"
complete=1
trap - EXIT INT TERM HUP
