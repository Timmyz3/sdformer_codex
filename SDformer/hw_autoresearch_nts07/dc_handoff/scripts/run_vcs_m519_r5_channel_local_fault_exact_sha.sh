#!/usr/bin/env bash
set -euo pipefail

m519_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m519_hw_root="$(cd "${m519_dc_root}/.." && pwd)"
m519_runner="$(realpath "${BASH_SOURCE[0]}")"
m519_vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
m519_contract_rel=contracts/m519_r5_channel_local_fault_recovery_contract_r1_20260827.json
m519_contract="${m519_hw_root}/${m519_contract_rel}"
m519_static_dir="${m519_hw_root}/reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827"
m519_static_seal="${m519_static_dir}/SHA256SUMS.seal.sha256"
m519_static_identity="${m519_static_dir}/evidence_identity.sha256"
m519_static_verdict="${m519_static_dir}/m519_r5_channel_local_fault_static_hammer_verdict_r1.json"
m519_run="${m519_hw_root}/results/m519_r5_channel_local_fault_vcs_r1_20260827"

[[ -n "${M519_R5_EXPECTED_VCS_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m519_runner}" | awk '{print $1}')" == \
   "${M519_R5_EXPECTED_VCS_RUNNER_SHA256}" ]] || {
    echo "M519 R5 caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M519_R5_EXPECTED_STATIC_OUTER_SEAL_FILE_SHA256:-}" && \
   -f "${m519_static_seal}" && \
   "$(sha256sum "${m519_static_seal}" | awk '{print $1}')" == \
   "${M519_R5_EXPECTED_STATIC_OUTER_SEAL_FILE_SHA256}" ]] || {
    echo "M519 R5 static review seal is absent or not caller-pinned" >&2
    exit 3
}
(cd "${m519_static_dir}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
jq -e '.status == "PASS_STATIC__R5_READY_FOR_ONE_VCS__DC_FORBIDDEN"
       and .severity.p0 == 0
       and .authorization.run_one_vcs == true
       and .authorization.run_dc == false' \
    "${m519_static_verdict}" >/dev/null || exit 3
cd "${m519_hw_root}"
sha256sum -c "${m519_static_identity}" >/dev/null || {
    echo "M519 R5 static-review input identity drift" >&2
    exit 3
}
grep -Fqx \
    "${M519_R5_EXPECTED_VCS_RUNNER_SHA256}  dc_handoff/scripts/run_vcs_m519_r5_channel_local_fault_exact_sha.sh" \
    "${m519_static_identity}" || exit 3
m519_contract_sha="$(awk -v path="${m519_contract_rel}" \
    '$2 == path {print $1}' "${m519_static_identity}")"
[[ "${m519_contract_sha}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "${m519_contract}" | awk '{print $1}')" == \
   "${m519_contract_sha}" ]] || exit 3
[[ "$(sha256sum "${m519_vcs}" | awk '{print $1}')" == \
   0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 ]] \
    || exit 3
[[ ! -e "${m519_run}" ]] || {
    echo "M519 R5 refuses to overwrite ${m519_run}" >&2
    exit 2
}
if pgrep -x vcs1 >/dev/null || pgrep -x vlogan >/dev/null || \
        pgrep -f '(^|/)(vcs)( |$)' >/dev/null || \
        pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null; then
    echo "M519 R5 refuses VCS/DC/FM/PT collision" >&2
    exit 4
fi

mkdir "${m519_run}"
m519_complete=0
m519_cleanup() {
    local rc=$?
    if [[ "${m519_complete}" -ne 1 ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${rc}" >"${m519_run}/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
    return "${rc}"
}
trap m519_cleanup EXIT
sha256sum -c "${m519_static_identity}" \
    >"${m519_run}/preflight_identity_check.txt"
sha256sum "${m519_runner}" "${m519_contract}" "${m519_vcs}" \
    "${m519_static_seal}" "${m519_static_identity}" \
    "${m519_static_verdict}" >"${m519_run}/launch_identity.sha256"

export VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
export VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

m519_compile_and_run() {
    local phase=$1 filelist=$2 top=$3 seed=$4
    local phase_dir="${m519_run}/${phase}"
    mkdir "${phase_dir}"
    set +e
    "${m519_vcs}" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="${phase_dir}/csrc" -f "${filelist}" -top "${top}" \
        -o "${phase_dir}/simv" >"${phase_dir}/compile.log" 2>&1
    local rc=$?
    set -e
    echo "${rc}" >"${phase_dir}/compile.rc"
    [[ "${rc}" -eq 0 && -x "${phase_dir}/simv" ]] || return 20
    ! grep -Eiq 'Error-\[|^Error|^Fatal|Fatal:' \
        "${phase_dir}/compile.log" || return 21
    set +e
    "${phase_dir}/simv" "+ntb_random_seed=${seed}" -no_save \
        -assert report="${phase_dir}/assert.report" -cm assert \
        >"${phase_dir}/sim.log" 2>&1
    rc=$?
    set -e
    echo "${rc}" >"${phase_dir}/sim.rc"
    [[ "${rc}" -eq 0 ]] || return 22
    ! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "${phase_dir}/sim.log" "${phase_dir}/assert.report" || return 23
}

m519_compile_and_run attack \
    dc_handoff/filelists/date_m519_r5_channel_local_fault_adapter_attacks_vcs.f \
    tb_m519_r5_channel_local_fault_adapter_attacks 519501
grep -Eq '^PASS M519 R5 channel-local fault adapter VCS attack_classes=12 reset_cases=[1-9][0-9]* legal_response_on_request_fault=2 sticky_quiescent_checks=10 normal_requests=[1-9][0-9]* normal_responses=[1-9][0-9]* request_side_effect_violations=0 response_side_effect_violations=0$' \
    "${m519_run}/attack/sim.log" || exit 30
for cover in cp_legal_response_illegal_request_same_cycle \
        cp_source_count_mismatch_attack cp_zero_mask_attack \
        cp_channel_bank_mismatch_attack \
        cp_illegal_response_legal_request_same_cycle \
        cp_pending_drain_request_attack \
        cp_response_backpressure_then_attack \
        cp_held_response_request_attack_retire \
        cp_cutthrough_request_attack_retire \
        cp_sticky_fault_quiescent; do
    grep -Eq "dut\.m499_sva\.${cover}, .* [1-9][0-9]* match" \
        "${m519_run}/attack/assert.report" || exit 31
done

# Preserve every r2 full-workload numeric, ordering, stall, and cycle scenario.
m519_compile_and_run primary \
    dc_handoff/filelists/date_m519_r5_channel_local_fault_k1_vs_k1x8_vcs.f \
    tb_m519_fc2_registered_release_k1_vs_k1x8_raw4_acc24 519027
grep -Eq '^PASS M519 registered-release K1 versus K1x8 FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 .*numeric_mismatches=0 .*tuple_mismatches=0 .*weight_mismatches=0 .*same_edge_release_violations=0 .*request_stalls=[1-9][0-9]* .*response_injection_stalls=[1-9][0-9]* .*result_stalls=[1-9][0-9]* .*raw_stalls=[1-9][0-9]* .*distinct_same_edge_req_rsp=[1-9][0-9]* .*next_cycle_slot_reuse=[1-9][0-9]* .*next_cycle_context_reuse=[1-9][0-9]*' \
    "${m519_run}/primary/sim.log" || exit 32
for row in \
    'B=1 events=20 k1_cycles=259 k1x8_cycles=53' \
    'B=2 events=41 k1_cycles=737 k1x8_cycles=133' \
    'B=4 events=90 k1_cycles=3153 k1x8_cycles=499' \
    'B=8 events=110 k1_cycles=7569 k1x8_cycles=1246' \
    'B=1 events=0 k1_cycles=14 k1x8_cycles=14'; do
    grep -Fq "M519 canonical K1 versus K1x8 ${row}" \
        "${m519_run}/primary/sim.log" || exit 33
done

m519_compile_and_run equalbw \
    dc_handoff/filelists/date_m519_r5_channel_local_fault_k8_vs_k1x8_vcs.f \
    tb_m519_fc2_registered_release_k8_vs_k1x8_raw4_acc24 519028
grep -Eq '^PASS M519EQ cutthrough-8bank equal-bandwidth FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 .*numeric_mismatches=0 .*tuple_mismatches=0 .*weight_mismatches=0 .*request_stalls=[1-9][0-9]* .*result_stalls=[1-9][0-9]* .*raw_stalls=[1-9][0-9]*' \
    "${m519_run}/equalbw/sim.log" || exit 34
for row in \
    'B=1 events=20 k8_cycles=51 k1x8_cycles=53' \
    'B=2 events=41 k8_cycles=131 k1x8_cycles=133' \
    'B=4 events=90 k8_cycles=486 k1x8_cycles=499' \
    'B=8 events=110 k8_cycles=1231 k1x8_cycles=1246' \
    'B=1 events=0 k8_cycles=14 k1x8_cycles=14'; do
    grep -Fq "M519EQ cutthrough equalbw ${row}" \
        "${m519_run}/equalbw/sim.log" || exit 35
done

python3 - "${m519_run}" "${M519_R5_EXPECTED_VCS_RUNNER_SHA256}" \
    "${M519_R5_EXPECTED_STATIC_OUTER_SEAL_FILE_SHA256}" \
    "${m519_contract_sha}" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
attack = (root / "attack/sim.log").read_text()
primary = (root / "primary/sim.log").read_text()
equalbw = (root / "equalbw/sim.log").read_text()
attack_match = re.search(
    r"PASS M519 R5 channel-local fault adapter VCS attack_classes=(\d+) "
    r"reset_cases=(\d+) legal_response_on_request_fault=(\d+) "
    r"sticky_quiescent_checks=(\d+)", attack)
if not attack_match:
    raise SystemExit(1)
def rows(text, pattern):
    out = []
    for match in re.finditer(pattern, text):
        out.append({key: int(value) for key, value in match.groupdict().items()})
    return out
k1 = rows(primary, r"M519 canonical K1 versus K1x8 B=(?P<blocks>\d+) "
    r"events=(?P<events>\d+) k1_cycles=(?P<k1>\d+) "
    r"k1x8_cycles=(?P<k1x8>\d+)")
k8 = rows(equalbw, r"M519EQ cutthrough equalbw B=(?P<blocks>\d+) "
    r"events=(?P<events>\d+) k8_cycles=(?P<k8>\d+) "
    r"k1x8_cycles=(?P<k1x8>\d+)")
expected_k1 = [(1,20,259,53),(2,41,737,133),(4,90,3153,499),
               (8,110,7569,1246),(1,0,14,14)]
expected_k8 = [(1,20,51,53),(2,41,131,133),(4,90,486,499),
               (8,110,1231,1246),(1,0,14,14)]
if [(r['blocks'],r['events'],r['k1'],r['k1x8']) for r in k1] != expected_k1:
    raise SystemExit(2)
if [(r['blocks'],r['events'],r['k8'],r['k1x8']) for r in k8] != expected_k8:
    raise SystemExit(3)
receipt = {
    "schema": "m519_r5_channel_local_fault_vcs_receipt_v1",
    "status": "PASS_M519_R5_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_REVIEW",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "runner_sha256": sys.argv[2],
    "static_review_outer_seal_file_sha256": sys.argv[3],
    "contract_sha256": sys.argv[4],
    "attack_classes": int(attack_match.group(1)),
    "reset_cases": int(attack_match.group(2)),
    "legal_response_on_request_fault": int(attack_match.group(3)),
    "sticky_quiescent_checks": int(attack_match.group(4)),
    "r2_cycles_preserved_exactly": True,
    "k1_vs_k1x8_rows": k1,
    "k8_vs_k1x8_rows": k8,
    "claim_boundary": {
        "channel_local_functional": True,
        "normal_cycle_regression": True,
        "combinational_loop_free": False,
        "dc": False,
        "ppa": False,
        "system_speedup": False,
        "headline": False,
    },
}
(root / "m519_r5_channel_local_fault_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
printf 'PASS_M519_R5_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_REVIEW\n' \
    >"${m519_run}/RUN_COMPLETE.txt"
(
    cd "${m519_run}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
m519_complete=1
trap - EXIT
echo "PASS M519 R5 exact VCS sealed at ${m519_run}"
