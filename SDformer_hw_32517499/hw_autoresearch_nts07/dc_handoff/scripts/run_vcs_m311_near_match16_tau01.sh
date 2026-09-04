#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M311_RUN_DIR:-${task_hw_root}/results/m311r4_near_match16_tau01_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m311/m311_near_match16_tau01.sv"]="c5e7a6dd7d522c1f7cd98b6e5fafda4deedefb70e36dd1ca1a72b41b2cd53170"
    ["verif_m311/m311_near_match16_tau01_assertions.sv"]="bf93984d2eafa26de004bfa613c203b25a1361581c07adcb1fa3504e95b7d9a0"
    ["tb_m311/tb_m311_near_match16_tau01.sv"]="cd4c568bfc690fbe023b3a0a37dee40d37663fcd1778afe313401f53246a3c73"
    ["dc_handoff/filelists/date_m311_near_match16_tau01_vcs.f"]="8648ba752df64ed03dc65acd56fa662978fdf1a67f3601817871963d9c97d7c9"
    ["contracts/m311r4_near_match16_tau01_directed_vcs_contract_r1_20260825.json"]="2b36e2c34458ddfa54f145bc1ec68745afc8cd69eec071d8be3cb2d63e084af8"
    ["results/m311_near_match16_tau01_vcs_r1_20260825/RUN_FAILED_OR_INCOMPLETE.txt"]="0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466"
    ["results/m311r2_near_match16_tau01_vcs_r1_20260825/RUN_FAILED_OR_INCOMPLETE.txt"]="0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466"
    ["results/m311r3_near_match16_tau01_vcs_r1_20260825/RUN_FAILED_OR_INCOMPLETE.txt"]="0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466"
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
cp contracts/m311r4_near_match16_tau01_directed_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m311_near_match16_tau01_vcs.f \
    -top tb_m311_near_match16_tau01 -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=31120260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M311 near-match16 tau01 VCS transactions=2200 exact=[1-9][0-9]* positive=[1-9][0-9]* rejected=[1-9][0-9]* guard=[1-9][0-9]* stalls=[1-9][0-9]* ties=1 mismatches=0 ii1=true tau0_subset=true executable_sram=false system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_stall cp_exact cp_positive cp_tau0 cp_guard \
        cp_distance_reject; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m311_near_match16_tau01_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"PASS M311 near-match16 tau01 VCS transactions=(\d+) exact=(\d+) "
    r"positive=(\d+) rejected=(\d+) guard=(\d+) stalls=(\d+) ties=(\d+) "
    r"mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M311 PASS payload")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m311_near_match16_tau01_vcs_receipt_v1",
    "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_AWAITING_INDEPENDENT_HAMMER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "transactions": values[0],
    "exact_hits": values[1],
    "positive_distance_hits": values[2],
    "distance_or_tau_rejections": values[3],
    "population_guard_rejections": values[4],
    "stalled_output_cycles": values[5],
    "directed_tie_cases": values[6],
    "numeric_or_order_mismatches": values[7],
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
printf '%s\n' "PASS_M311_NEAR_MATCH16_TAU01_EXACT_SHA_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M311 exact VCS sealed at ${task_run}"
