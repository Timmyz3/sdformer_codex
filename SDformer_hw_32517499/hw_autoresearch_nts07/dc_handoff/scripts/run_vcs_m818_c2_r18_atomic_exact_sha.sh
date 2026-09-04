#!/usr/bin/env bash
set -euo pipefail

# M818/C2 R18 future one-attempt VCS runner. Source-only until the separately
# sealed fresh source hammer, true release, and final release hammer exist.
# Source dry-run exits before license/tool probes and before formal identities.

dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "${dc_root}/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
python36=/usr/libexec/platform-python3.6
guard="${hw_root}/verif_m818/m818_c2_r18_atomic_guard.py"
contract="${hw_root}/contracts/m818_c2_r18_atomic_source_only_contract_r1_20260829.json"
candidate="${hw_root}/contracts/m818_c2_r18_vcs_launch_candidate_source_only_r1_20260829.json"
source_hammer="${hw_root}/reviews/m819_m818_c2_r18_atomic_source_fresh_hammer_r1_20260829"
release="${hw_root}/contracts/m818_c2_r18_atomic_vcs_launch_admission_r1_20260829.json"
final_hammer="${hw_root}/reviews/m821_m818_c2_r18_atomic_final_launch_hammer_r1_20260829"
vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
license_file=/opt/synopsys/Synopsys.dat
result="${hw_root}/results/m818_c2_r18_atomic_channel_split_vcs_r1_20260829"
attempt="${hw_root}/results/.m818_c2_r18_atomic_channel_split_vcs_attempt_consumed"
attempt_stage="${hw_root}/results/.m818_c2_r18_atomic_channel_split_vcs_attempt_stage.$$"
work="${hw_root}/results/.m818_c2_r18_atomic_channel_split_vcs_work.$$"
runner_log="${hw_root}/results/.m818_c2_r18_atomic_channel_split_vcs_runner.$$.log"
failure_primary="m818_c2_r18_atomic_channel_split_vcs_r1_20260829.failed_or_incomplete.$$.quarantine"
complete=0
attempt_consumed=0
failure_armed=0
phase=SOURCE_PREFLIGHT
contract_sha256=none
candidate_sha256=none
release_sha256=none
final_outer_sha256=none

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M818 R18 gate failure: %s\n' "$*" >&2; exit 3; }
expect_file_sha() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && "$(sha "${path}")" == "${expected}" ]] \
        || fail "SHA mismatch or nonregular file: ${path}"
}
log_phase() {
    phase=$1
    if [[ "${failure_armed}" -eq 1 ]]; then
        printf 'phase=%s\n' "${phase}" >>"${runner_log}"
    fi
}
trace_event() {
    local event=$1
    printf '{"event":"%s","totals":{"vcs_identity_probe_runs":0,"license_server_queries":0,"vcs_compile_runs":0,"simv_runs":0,"formal_attempts_created":0,"formal_results_created":0,"failure_quarantines_created":0}}\n' \
        "${event}" >>"${M818_R18_SOURCE_DRY_RUN_TRACE}"
}
source_dry_run() {
    local root=${M818_R18_SOURCE_DRY_RUN_ROOT:-}
    local nonce=${M818_R18_SOURCE_DRY_RUN_NONCE:-}
    local trace=${M818_R18_SOURCE_DRY_RUN_TRACE:-}
    [[ -n "${root}" && -d "${root}" && ! -L "${root}" \
       && -n "${nonce}" && -f "${nonce}" && ! -L "${nonce}" \
       && -n "${trace}" && ! -e "${trace}" \
       && "$(<"${nonce}")" == M818_R18_SOURCE_HAMMER_ONLY ]] \
        || fail "invalid source dry-run nonce/root/trace"
    case "${trace}" in "${root}"/*) ;; *) fail "dry-run trace escapes root" ;; esac
    : >"${trace}"
    trace_event source_contract_verified
    trace_event m814_repair_authority_verified
    "${python36}" "${guard}" self-test >/dev/null
    trace_event atomic_guard_selftest
    trace_event live_vcs_license_boundary_stop
    printf '%s\n' M818_R18_STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY__NO_LIVE_PROBE__NO_FORMAL_IDENTITY
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
        || fail "license file missing/nonregular"
    "${lmutil}" lmstat -a -c "${SNPSLMD_LICENSE_FILE}" >/dev/null 2>&1 \
        || fail "license server status failed"
}
compile_and_run() {
    local run_phase=$1 filelist=$2 top=$3 seed=$4
    local phase_dir="${work}/${run_phase}"
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
publish_failure_receipt() {
    local rc=$1 published=false
    [[ "${attempt_consumed}" -eq 1 ]] && published=true
    set +e
    "${python36}" "${guard}" write-failure-quarantine \
        --parent "${hw_root}/results" --primary-name "${failure_primary}" \
        --phase "${phase}" --return-code "${rc}" \
        --shell-published "${published}" \
        --attempt-path "${attempt}" --attempt-stage "${attempt_stage}" \
        --runner-sha256 "$(sha "${runner}")" \
        --contract-sha256 "${contract_sha256}" \
        --candidate-sha256 "${candidate_sha256}" \
        --release-sha256 "${release_sha256}" \
        --final-hammer-outer-seal-sha256 "${final_outer_sha256}" \
        --runner-log "${runner_log}" >&2
    local receipt_rc=$?
    set -e
    [[ "${receipt_rc}" -eq 0 ]] || \
        printf 'M818 R18 failure receipt publication itself failed rc=%s\n' \
            "${receipt_rc}" >&2
}
failure_cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${failure_armed}" -eq 1 && "${complete}" -ne 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        publish_failure_receipt "${rc}"
    fi
    exit "${rc}"
}
signal_exit() {
    trap - INT TERM HUP
    exit 130
}

# Wrong runner SHA and source dry-run are before failure arming: they create no
# formal attempt/result/quarantine and do not query a license or VCS identity.
[[ ! -v HOME ]] || fail "HOME must be absent; runner never synthesizes HOME"
[[ -n "${M818_R18_EXPECTED_VCS_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M818_R18_EXPECTED_VCS_RUNNER_SHA256}" ]] \
    || fail "caller must pin exact independently reviewed runner SHA"
expect_file_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f
"${python36}" "${guard}" validate-source --hw-root "${hw_root}" \
    --contract "${contract}" --candidate "${candidate}" --runner "${runner}" \
    >/dev/null

if [[ "${M818_R18_SOURCE_DRY_RUN:-0}" == 1 ]]; then source_dry_run; fi

# Arm a sealed non-paper failure boundary before the future launch-chain gate.
# Wrong-SHA and intentional source dry-run remain the only zero-side-effect
# exits. Missing/malformed source-hammer, release, or final-hammer identities
# are therefore auditable PRE_STAGE failures without consuming an attempt.
contract_sha256=$(sha "${contract}")
candidate_sha256=$(sha "${candidate}")
release_sha256=$(printf '0%.0s' {1..64})
if [[ -f "${release}" && ! -L "${release}" ]]; then
    release_sha256=$(sha "${release}")
fi
final_outer_sha256=${M818_R18_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256:-$(printf '0%.0s' {1..64})}
failure_armed=1
trap failure_cleanup EXIT
trap signal_exit INT TERM HUP
log_phase PRE_STAGE_LAUNCH_CHAIN

[[ -n "${M818_R18_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256:-}" ]] \
    || fail "caller must pin final hammer outer seal"
final_outer_sha256=${M818_R18_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256}
"${python36}" "${guard}" validate-launch-chain --hw-root "${hw_root}" \
    --contract "${contract}" --candidate "${candidate}" --runner "${runner}" \
    --source-hammer "${source_hammer}" --release "${release}" \
    --final-hammer "${final_hammer}" \
    --expected-final-outer "${final_outer_sha256}" >/dev/null
contract_sha256=$(sha "${contract}")
candidate_sha256=$(sha "${candidate}")
release_sha256=$(sha "${release}")

# Every failure from the launch-chain gate onward is a sealed non-paper failure
# boundary, including failures before or after formal attempt publication.
log_phase PRE_ATTEMPT_PREFLIGHT

expect_file_sha "${vcs}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
expect_file_sha "${lmutil}" e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07
[[ "${VCS_HOME:-}" == /opt/synopsys/vcs/V-2023.12-SP1 \
   && "${VCS_ARCH_OVERRIDE:-}" == linux ]] || fail "VCS environment mismatch"
[[ ! -e "${result}" && ! -L "${result}" \
   && ! -e "${attempt}" && ! -L "${attempt}" \
   && ! -e "${attempt_stage}" && ! -L "${attempt_stage}" \
   && ! -e "${work}" && ! -L "${work}" ]] \
    || fail "result/attempt/stage/work identity already exists"
collision_gate
resource_gate
license_gate
collision_gate

log_phase ATTEMPT_STAGE_BUILD
"${python36}" "${guard}" create-attempt-stage --stage "${attempt_stage}" \
    --runner-sha256 "$(sha "${runner}")" \
    --contract-sha256 "${contract_sha256}" \
    --candidate-sha256 "${candidate_sha256}" \
    --release-sha256 "${release_sha256}" \
    --final-hammer-outer-seal-sha256 "${final_outer_sha256}" >/dev/null
"${python36}" "${guard}" verify-sealed-directory --path "${attempt_stage}" \
    --exact-root-members attempt.json,SHA256SUMS,SHA256SUMS.seal.sha256 >/dev/null

log_phase ATTEMPT_ATOMIC_PUBLISH
"${python36}" "${guard}" publish-attempt-no-replace \
    --source "${attempt_stage}" \
    --destination "${attempt}" \
    --runner-sha256 "$(sha "${runner}")" \
    --contract-sha256 "${contract_sha256}" \
    --candidate-sha256 "${candidate_sha256}" \
    --release-sha256 "${release_sha256}" \
    --final-hammer-outer-seal-sha256 "${final_outer_sha256}" >/dev/null
attempt_consumed=1
log_phase ATTEMPT_POST_PUBLISH_VERIFY
"${python36}" "${guard}" verify-attempt --path "${attempt}" \
    --runner-sha256 "$(sha "${runner}")" \
    --contract-sha256 "${contract_sha256}" \
    --candidate-sha256 "${candidate_sha256}" \
    --release-sha256 "${release_sha256}" \
    --final-hammer-outer-seal-sha256 "${final_outer_sha256}" >/dev/null

log_phase WORK_STAGE_CREATE
mkdir "${work}"
printf 'runner_sha256=%s\ncontract_sha256=%s\ncandidate_sha256=%s\nrelease_sha256=%s\n' \
    "$(sha "${runner}")" "${contract_sha256}" "${candidate_sha256}" \
    "${release_sha256}" >"${work}/launch_identity.txt"

cd "${hw_root}"
log_phase ATTACK_VCS
compile_and_run attack \
    dc_handoff/filelists/date_m803_c2_r16_channel_split_adapter_attacks_vcs.f \
    tb_m803_c2_r16_channel_split_adapter_attacks 813017
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

log_phase EQUAL_BANDWIDTH_VCS
compile_and_run equalbw \
    dc_handoff/filelists/date_m803_c2_r16_channel_split_k8_vs_k1x8_vcs.f \
    tb_m803_c2_r16_channel_split_k8_vs_k1x8_raw4_acc24 813018
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

log_phase RESULT_STAGE_SEAL
"${python36}" - "${work}" "$(sha "${runner}")" \
    "${contract_sha256}" "${candidate_sha256}" "${release_sha256}" \
    "${final_outer_sha256}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
receipt = {
  "schema": "m818_c2_r18_atomic_channel_split_vcs_receipt_v1",
  "status": "PASS_M818_R18_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER",
  "runner_sha256": sys.argv[2], "contract_sha256": sys.argv[3],
  "candidate_sha256": sys.argv[4], "release_sha256": sys.argv[5],
  "final_hammer_outer_seal_sha256": sys.argv[6],
  "tool": "Synopsys VCS V-2023.12-SP1",
  "attack_contract": {"same_cycle_slot_reuse": 1,
    "ledger_conservation": True, "illegal_response_closes_both": True,
    "legal_response_survives_request_fault": True},
  "exact_cycles": {"k8": [51,131,486,1231,14],
    "k1x8": [53,133,499,1246,14]},
  "frozen_k1_vs_k1x8": "SOURCE_SHA_BOUND_ONLY__NOT_RERUN_OR_CHANGED",
  "claim_boundary": {"vcs_validated": True, "dc": False, "ppa": False,
    "system_speedup": False, "headline": False, "paper_citable": False}
}
(root / "m818_c2_r18_atomic_channel_split_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
printf '%s\n' PASS_M818_R18_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER \
    >"${work}/RUN_COMPLETE.txt"
[[ ! -e "${result}" && ! -L "${result}" ]] \
    || fail "canonical result appeared after precheck"
"${python36}" "${guard}" seal-directory --path "${work}" >/dev/null
"${python36}" "${guard}" verify-sealed-directory --path "${work}" >/dev/null

log_phase RESULT_ATOMIC_PUBLISH
"${python36}" "${guard}" publish-no-replace --source "${work}" \
    --destination "${result}" >/dev/null
"${python36}" "${guard}" verify-sealed-directory --path "${result}" >/dev/null
for root_file in RUN_COMPLETE.txt \
        m818_c2_r18_atomic_channel_split_vcs_receipt_r1.json \
        SHA256SUMS SHA256SUMS.seal.sha256; do
    [[ -f "${result}/${root_file}" && ! -L "${result}/${root_file}" ]] \
        || fail "canonical root member missing: ${root_file}"
done
[[ ! -e "${result}/$(basename "${work}")" ]] \
    || fail "nested work directory appeared in canonical result"

complete=1
trap - EXIT INT TERM HUP
printf 'M818 R18 canonical result atomically published; receipt hammer still required\n'
