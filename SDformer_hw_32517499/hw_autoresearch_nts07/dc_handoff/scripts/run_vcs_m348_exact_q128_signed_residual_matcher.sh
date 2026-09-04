#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M348_RUN_DIR:-${task_hw_root}/results/m348_exact_q128_signed_residual_matcher_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m348/m348_exact_q128_signed_residual_matcher.sv"]="960b72268d526baad7e7d74cd159f2a9fbb286abac025678f239af3f5147eb1f"
    ["verif_m348/m348_exact_q128_signed_residual_matcher_assertions.sv"]="34a476ed43bbfac61a5cb9edcd4d09da27668985072854e9551de06498bd906d"
    ["tb_m348/tb_m348_exact_q128_signed_residual_matcher.sv"]="1978bf0d77d2f57611e85629eb647969cc16dd6dbf839bf93b82b713f871b52a"
    ["dc_handoff/filelists/date_m348_exact_q128_signed_residual_matcher_vcs.f"]="ccbcf9b807efbac61b143318959aec21479778db25a3e182dcf2f3ea748865cb"
    ["contracts/m348_exact_q128_signed_residual_matcher_directed_vcs_contract_r1_20260825.json"]="fd28608423375c31ab96abacb2d9afa37ebce1ee7b513df38a59a16f6cef0d14"
    ["results/m344_output_block_tiled_q128_kfirst_r1_20260825/m344_output_block_tiled_q128_kfirst_r1.json"]="a1adf022e656a9362f7a5ffb063f0024d7e0b1015836343aef3c630a727d7517"
    ["results/m343_m339_q128_selective_pwp_kfirst_independent_hammer_r1_20260825/m343_m339_q128_selective_pwp_kfirst_independent_hammer_r1.json"]="beb5ac8ee926db7a5c0b46591f2360c45f6052dbd5172e58281e840f9effd394"
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
cp contracts/m348_exact_q128_signed_residual_matcher_directed_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m348_exact_q128_signed_residual_matcher_vcs.f \
    -top tb_m348_exact_q128_signed_residual_matcher \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=34820260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M348 q128 exact signed matcher transactions=3000 use=[1-9][0-9]* fallback=[1-9][0-9]* mixed=[1-9][0-9]* exact=[1-9][0-9]* ties=[1-9][0-9]* stalls=[1-9][0-9]* protocol_attacks=1 max_accept_run=[2-9][0-9][0-9]+ max_retire_run=[1-9][0-9][0-9]+ latency_min=128 latency_max=[1-9][0-9]* mismatches=0 ii1=true center_id=true signed_residual=true exact_fallback=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_config cp_use_pwp cp_fallback \
        cp_positive_signed_residual cp_output_stall; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m348_exact_q128_signed_residual_matcher_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"PASS M348 q128 exact signed matcher transactions=(\d+) use=(\d+) "
    r"fallback=(\d+) mixed=(\d+) exact=(\d+) ties=(\d+) stalls=(\d+) "
    r"protocol_attacks=(\d+) max_accept_run=(\d+) max_retire_run=(\d+) "
    r"latency_min=(\d+) latency_max=(\d+) mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M348 PASS payload")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m348_exact_q128_signed_residual_matcher_vcs_receipt_v1",
    "status": "PASS_M348_EXACT_SHA_VCS_Q128_CENTER_ID_SIGNED_RESIDUAL",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "transactions": values[0],
    "use_pwp": values[1],
    "fallback_bit_sparse": values[2],
    "mixed_plus_minus_residual": values[3],
    "exact_pattern_use": values[4],
    "distance_tie_observations": values[5],
    "stalled_output_cycles": values[6],
    "protocol_attacks": values[7],
    "maximum_consecutive_input_accepts": values[8],
    "maximum_consecutive_output_retires": values[9],
    "minimum_observed_latency_cycles": values[10],
    "maximum_observed_latency_cycles": values[11],
    "numeric_or_order_mismatches": values[12],
    "architecture": {
        "patterns": 128,
        "configuration_beats": 8,
        "pipeline_stages": 128,
        "ii1_no_stall": True,
        "tie_break": "lowest center ID",
        "center_id_output": True,
        "plus_minus_residual_masks": True,
        "exact_bit_sparse_fallback": True,
    },
    "claim_boundary": {
        "functional_matcher_rtl": True,
        "complete_pwp_conv": False,
        "finite_queue_cycle_match": False,
        "dc_area_fmax": False,
        "physical_sram": False,
        "energy": False,
        "system_speedup": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M348_EXACT_Q128_SIGNED_RESIDUAL_MATCHER_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M348 exact VCS sealed at ${task_run}"
