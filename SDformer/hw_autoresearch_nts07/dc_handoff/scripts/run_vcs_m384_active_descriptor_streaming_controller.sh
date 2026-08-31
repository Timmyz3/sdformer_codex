#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M384_RUN_DIR:-${task_hw_root}/results/m384_active_descriptor_streaming_controller_vcs_r1b_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m384/m384_active_descriptor_streaming_controller.sv"]="15f0e1d8aebfcb66ed58cefed988bde855a8b2a351e32c86beb2381a8c4e6b38"
    ["verif_m384/m384_active_descriptor_streaming_controller_assertions.sv"]="b7cc2a25e32caa9583c02b67a279f25af7a74bff3024627d7b200544cbda3470"
    ["tb_m384/tb_m384_active_descriptor_streaming_controller.sv"]="8aa9f3b27b39d90aa08b5b186abaa9a3a03eb94ec25efc1af1f83a61290db91b"
    ["dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_vcs.f"]="8414f0b0a854ab83a394a322a661d89ad14b73cb563597f875e002085bb2dd82"
    ["dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_rtl.f"]="c3db231e355357c138247c0c76a0352d80d5574a863988fb9af2746be9c37467"
    ["contracts/m384_active_descriptor_streaming_controller_directed_vcs_contract_r1_20260826.json"]="7dc11d0ddb090768f89bcef397dd8d2520a23eac38d22804067249251bf1bec9"
    ["results/m380_m377_active_descriptor_controller_prertl_hammer_r1_20260825/SHA256SUMS.seal.sha256"]="962fe76088392ec93e77752ecb2c055c3ab4547b4ae1bfefa9c5dc69ea65be1e"
    ["results/m381_q32_o4_burst_streaming_sensitivity_r1_20260825/SHA256SUMS.seal.sha256"]="1409999441bb6cec375cbb82d542f2d8e99a4dc7ebd4b2531821fe66d8afec29"
    ["results/m382_m381_burst_streaming_independent_hammer_r1_20260826/m382_m381_independent_hammer_review_r1.json"]="edf2028c05d0d0041b87b739202d315da9f7c7631223a38f7f88cdb809b39954"
    ["results/m382_m381_burst_streaming_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="fb260b75543f64b68564c950fd3fffb6dc1356a86da7e92534438d8d4b004c42"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "${task_path}" "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m384_active_descriptor_streaming_controller_directed_vcs_contract_r1_20260826.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_vcs.f \
    -top tb_m384_active_descriptor_streaming_controller \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=38420260826 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout|mismatches=[1-9]' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true

grep -Eq 'PASS M384 active descriptor streaming controller phases=4 replays=8 bundles=10804 pwp_runs=[4-9][0-9]*|PASS M384 active descriptor streaming controller phases=4 replays=8 bundles=10804 pwp_runs=[1-9][0-9]+' \
    "${task_run}/sim.log" || exit 30
grep -Eq 'prefetch_starts=[1-9][0-9]* prefetch_dones=[1-9][0-9]* zero=[1-9][0-9]* active=[1-9][0-9]* pop1=[1-9][0-9]* pwp=[1-9][0-9]* fallback=[1-9][0-9]* write_stalls=[1-9][0-9]* request_stalls=[1-9][0-9]* response_stalls=[0-9]+ backend_stalls=[1-9][0-9]* protocol_attacks=10 sticky_cycles=[1-9][0-9][0-9]+ max_fifo=8 max_outstanding=8 max_credit=8 latency_mask=116 reset_mask=87ff mismatches=0 exact_compaction=true direct_address_runs=true tile1_overlap=true dual_replay=true ii1_credit=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 31

for task_cover in cp_reload cp_zero_phase cp_active_one cp_active_2400 \
        cp_active_3000 cp_pop1_fallback cp_mixed_residual \
        cp_single_pwp_run cp_full_pwp_run cp_multi_pwp_run \
        cp_tile1_prefetch_overlap_start cp_tile1_prefetch_done \
        cp_fifo_full cp_outstanding_full cp_simultaneous_push_pop \
        cp_tile0_done cp_tile1_done cp_protocol_attack; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 32
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m384_active_descriptor_streaming_controller_vcs_receipt_r1b.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
pattern = re.compile(
    r"PASS M384 active descriptor streaming controller phases=(\d+) "
    r"replays=(\d+) bundles=(\d+) pwp_runs=(\d+) "
    r"prefetch_starts=(\d+) prefetch_dones=(\d+) zero=(\d+) active=(\d+) "
    r"pop1=(\d+) pwp=(\d+) fallback=(\d+) write_stalls=(\d+) "
    r"request_stalls=(\d+) response_stalls=(\d+) backend_stalls=(\d+) "
    r"protocol_attacks=(\d+) sticky_cycles=(\d+) max_fifo=(\d+) "
    r"max_outstanding=(\d+) max_credit=(\d+) latency_mask=([0-9a-f]+) "
    r"reset_mask=([0-9a-f]+) mismatches=(\d+)")
match = pattern.search(text)
if not match:
    raise SystemExit("missing M384 PASS payload")
values = match.groups()
receipt = {
    "schema": "m384_active_descriptor_streaming_controller_vcs_receipt_r1b",
    "status": "PASS_M384_R1B_EXACT_SHA_SYNOPSYS_VCS_BOUNDED_STREAMING_CONTROLLER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "normal_phases": int(values[0]),
    "normal_replays": int(values[1]),
    "checked_bundles": int(values[2]),
    "pwp_run_commands": int(values[3]),
    "tile1_prefetch_starts": int(values[4]),
    "tile1_prefetch_completions": int(values[5]),
    "directed_zero_rows": int(values[6]),
    "directed_active_rows": int(values[7]),
    "popcount_one_fallback_rows": int(values[8]),
    "pwp_rows": int(values[9]),
    "fallback_rows": int(values[10]),
    "descriptor_write_stall_cycles": int(values[11]),
    "descriptor_request_stall_cycles": int(values[12]),
    "descriptor_response_stall_cycles": int(values[13]),
    "backend_stall_cycles": int(values[14]),
    "protocol_attacks": int(values[15]),
    "sticky_error_cycles": int(values[16]),
    "maximum_fifo_occupancy": int(values[17]),
    "maximum_outstanding_reads": int(values[18]),
    "maximum_credit_used": int(values[19]),
    "read_latency_cover_mask_hex": values[20],
    "reset_state_cover_mask_hex": values[21],
    "numeric_or_order_mismatches": int(values[22]),
    "verified_properties": {
        "exact_zero_only_compaction": True,
        "canonical_48bit_descriptor": True,
        "direct_address_maximal_pwp_runs": True,
        "tile1_prefetch_overlaps_tile0_replay": True,
        "in_order_l1_to_l8_d8_credit": True,
        "exactly_two_replays_per_nonempty_phase": True,
        "sticky_fail_closed_protocol": True,
    },
    "claim_boundary": {
        "functional_controller_rtl": True,
        "complete_q32_matcher_integrated": False,
        "physical_sram": False,
        "frozen_17280_phase_cycle_match": False,
        "synopsys_area_or_timing": False,
        "energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M384_BOUNDED_ACTIVE_DESCRIPTOR_STREAMING_CONTROLLER_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M384 exact VCS sealed at ${task_run}"
