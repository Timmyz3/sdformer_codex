#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M321_RUN_DIR:-${task_hw_root}/results/m321_near_match16_tau01_tournament2_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m321/m321_near_match16_tau01_tournament2.sv"]="e54fd31a0beca33c18938e8241e55a07939dc1ff924dce8b81ccbb3a57e242e2"
    ["verif_m321/m321_near_match16_tau01_tournament2_assertions.sv"]="eab99ca27528b63f528dd1d92c94dc65259b7967867e339a12af39a587b8b9b5"
    ["tb_m321/tb_m321_near_match16_tau01_tournament2.sv"]="7b62cbe39409678a2c3e77a41b8c7c3dba4a4fcc875c0e5fe45039cd140a680e"
    ["dc_handoff/filelists/date_m321_near_match16_tau01_tournament2_vcs.f"]="b1eb9008fdc234573135faaffa07291bc36f0d1912295444a54946cb66a5cd35"
    ["contracts/m321_near_match16_tau01_tournament2_directed_vcs_contract_r1_20260825.json"]="f84e6d44a3b80160ccb22d058f0a313f8c67239493087c95ac4afcde3092c3eb"
    ["results/m320_m311_timing_architecture_predesign_r1_20260825/m320_matcher_timing_architecture_predesign_r1.json"]="3739ddaa8c0b4adced0bd90b2785a13614f89924c2928441e02310accf1d5497"
    ["results/m320_m311_timing_architecture_predesign_r1_20260825/evidence_manifest.seal.sha256"]="e9c28a481a41c4fcc20f1648694d7e7514e0b9a3bd816a59e57c67c414ae3f96"
    ["results/m315_m311r4_m314_independent_hammer_r1_20260825/m315_independent_hammer_review_r1.json"]="4af54e316c0e8682ef6207f678e3dcd65d704076e662ed2f1d8697ccb1badd5f"
    ["contracts/m309_m306_valid825_selection_freeze_contract_r1_20260825.json"]="0d4a15124e039b62c54428e2ffe0b9a5b8293a849e1007ce536ea891ef8e6122"
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
cp contracts/m321_near_match16_tau01_tournament2_directed_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m321_near_match16_tau01_tournament2_vcs.f \
    -top tb_m321_near_match16_tau01_tournament2 \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=32120260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M321 tournament2 VCS transactions=3000 exact=[1-9][0-9]* positive=[1-9][0-9]* rejected=[1-9][0-9]* guard=[1-9][0-9]* stalls=[1-9][0-9]* ties=1 max_accept_run=[1-9][0-9][0-9]+ max_retire_run=[1-9][0-9][0-9]+ mismatches=0 latency=2 ii1=true tau0_subset=true executable_sram=false system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_stage0_full cp_stall cp_exact cp_positive cp_tau0 \
        cp_guard cp_distance_reject; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m321_near_match16_tau01_tournament2_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"PASS M321 tournament2 VCS transactions=(\d+) exact=(\d+) "
    r"positive=(\d+) rejected=(\d+) guard=(\d+) stalls=(\d+) ties=(\d+) "
    r"max_accept_run=(\d+) max_retire_run=(\d+) mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M321 PASS payload")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m321_near_match16_tau01_tournament2_vcs_receipt_v1",
    "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_AWAITING_DC_AND_INDEPENDENT_HAMMER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "transactions": values[0],
    "exact_hits": values[1],
    "positive_distance_hits": values[2],
    "distance_or_tau_rejections": values[3],
    "population_guard_rejections": values[4],
    "stalled_output_cycles": values[5],
    "directed_tie_cases": values[6],
    "maximum_consecutive_input_accepts": values[7],
    "maximum_consecutive_output_retires": values[8],
    "numeric_or_order_mismatches": values[9],
    "latency_cycles_no_stall": 2,
    "ii1_interface": True,
    "tau0_exact_hardware_subset": True,
    "open_source_rtl_tools_invoked": False,
    "claim_boundary": {
        "matcher_rtl": True,
        "complete_pwp_conv_rtl": False,
        "physical_metadata_sram": False,
        "dc": False,
        "accuracy": False,
        "system_speedup": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M321_TOURNAMENT2_EXACT_SHA_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M321 exact VCS sealed at ${task_run}"
