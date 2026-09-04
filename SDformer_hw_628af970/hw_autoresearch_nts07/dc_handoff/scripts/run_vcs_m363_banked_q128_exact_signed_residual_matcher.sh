#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M363_RUN_DIR:-${task_hw_root}/results/m363_banked_q128_exact_signed_residual_matcher_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m363/m363_banked_q128_exact_signed_residual_matcher.sv"]="257084916c312c9db4e2d6ad59a4fe20fb604fa6c9d0a0573039339e9614879d"
    ["verif_m363/m363_banked_q128_exact_signed_residual_matcher_assertions.sv"]="8cb774078a5251d9403732cfa40ccf55c6a7db836eeb1a9f5beaa20c34c7e8f6"
    ["tb_m363/tb_m363_banked_q128_exact_signed_residual_matcher.sv"]="94c072d0f0d5fd70900528c23f5b5bb0aa5ef5d920c47308070e298014e4bea9"
    ["dc_handoff/filelists/date_m363_banked_q128_exact_signed_residual_matcher_vcs.f"]="2cde218dd3ba0404aa34fef8688a76dc4b907217c1f69cb8c75d068b14c4aeab"
    ["dc_handoff/filelists/date_m363_banked_q128_exact_signed_residual_matcher_rtl.f"]="1c3409aa73a632842be88a7a52dd29c2263c5eaaafc8744f17e44b5c16fa74ce"
    ["contracts/m363_banked_q128_exact_signed_residual_matcher_directed_vcs_contract_r1_20260825.json"]="8eada82433b15ee6b33ebf087eaaf9c4a3d18f73cdfb8dfbcb05153f62ae9704"
    ["results/m356_failclosed_q128_signed_residual_matcher_vcs_r1_20260825/m356_failclosed_q128_signed_residual_matcher_vcs_receipt_r1.json"]="299da387473af4f70d5823baa7774864d3f4012da96a0452df921b19bfa6381b"
    ["results/m357_m356_failclosed_matcher_independent_hammer_r1_20260825/m357_m356_independent_hammer_review_r1.json"]="20330f43387240806a4995a0a8832b375fed81cec31e1e4429aea7a88bab48ab"
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
cp contracts/m363_banked_q128_exact_signed_residual_matcher_directed_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m363_banked_q128_exact_signed_residual_matcher_vcs.f \
    -top tb_m363_banked_q128_exact_signed_residual_matcher \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=36320260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M363 banked q128 exact signed matcher transactions=3000 use=[1-9][0-9]* fallback=[1-9][0-9]* mixed=[1-9][0-9]* exact=[1-9][0-9]* transient_ties=[1-9][0-9]* stalls=[1-9][0-9]* protocol_attacks=5 sticky_reconfiguration_attempts=40 max_accept_run=[2-9][0-9][0-9]+ max_retire_run=[1-9][0-9][0-9]+ latency_min=4 latency_max=[1-9][0-9]* mismatches=0 ii1=true banked_tree=true center_id=true signed_residual=true exact_fallback=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_config cp_use_pwp cp_fallback \
        cp_positive_signed_residual cp_output_stall; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m363_banked_q128_exact_signed_residual_matcher_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"PASS M363 banked q128 exact signed matcher transactions=(\d+) use=(\d+) "
    r"fallback=(\d+) mixed=(\d+) exact=(\d+) transient_ties=(\d+) stalls=(\d+) "
    r"protocol_attacks=(\d+) sticky_reconfiguration_attempts=(\d+) "
    r"max_accept_run=(\d+) max_retire_run=(\d+) "
    r"latency_min=(\d+) latency_max=(\d+) mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M363 PASS payload")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m363_banked_q128_exact_signed_residual_matcher_vcs_receipt_v1",
    "status": "PASS_M363_EXACT_SHA_VCS_BALANCED_BANKED_Q128_MATCHER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "transactions": values[0],
    "use_pwp": values[1],
    "fallback_bit_sparse": values[2],
    "mixed_plus_minus_residual": values[3],
    "exact_pattern_use": values[4],
    "transient_best_distance_tie_observations": values[5],
    "stalled_output_cycles": values[6],
    "protocol_attacks": values[7],
    "sticky_reconfiguration_attempts": values[8],
    "sticky_reconfiguration_handshakes": 0,
    "maximum_consecutive_input_accepts": values[9],
    "maximum_consecutive_output_retires": values[10],
    "minimum_observed_latency_cycles": values[11],
    "maximum_observed_latency_cycles": values[12],
    "numeric_or_order_mismatches": values[13],
    "architecture": {
        "patterns": 128,
        "configuration_beats": 8,
        "pipeline_stages": 4,
        "balanced_four_way_reduction_stages": 3,
        "ii1_no_stall": True,
        "tie_break": "lowest center ID",
        "center_id_output": True,
        "plus_minus_residual_masks": True,
        "exact_bit_sparse_fallback": True,
        "sticky_protocol_error_blocks_configuration_until_reset": True,
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
printf '%s\n' "PASS_M363_BANKED_Q128_EXACT_SIGNED_RESIDUAL_MATCHER_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M363 exact VCS sealed at ${task_run}"
