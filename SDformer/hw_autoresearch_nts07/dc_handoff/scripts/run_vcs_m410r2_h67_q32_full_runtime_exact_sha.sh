#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M410R2_VCS_RUN_DIR:-${task_hw_root}/results/m410r2_h67_q32_full_runtime_vcs_r2_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_config="${task_hw_root}/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_phase_config_768.memh"
task_rows="${task_hw_root}/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}/csrc"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m405/m405_q32_serial16_zero_stop_controller.sv"]="f412ab817eb29ab303da9ec011379a853efc567fa5a6a29a73fced52cf503b42"
    ["verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv"]="1e82a8e5ffbe80d0ae7a19a9d78899a2646d73f1994dc73796048dcfe2a47f11"
    ["tb_m405/tb_m410_h67_q32_full_runtime_vcs.sv"]="3f0c13b3bd8c03797b805cb01db216f48c81840e1413c8b037b277e47334d751"
    ["dc_handoff/filelists/date_m410_h67_q32_full_runtime_vcs.f"]="9a1e0754ee5c866fbe771afae76697097003472eb98a4b200299180fd684ba12"
    ["contracts/m410r2_h67_q32_full_runtime_vcs_contract_r2_20260826.json"]="5ca79a381fef4bd80fb90b5ca2dd6989d9f545ff05351c1f90d2f8add693cd7e"
    ["results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/m409_m408_static_codec_vcs_independent_hammer_review_r1.json"]="076fdb4e4a2bd7464f01618a389535b4b404acde57cc37bc7aae2b39a0f9adc4"
    ["results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="7fbb0caaa935451edcbf08a965b6bd99fda33ee01d626a53eb4d5a2559b2d8ec"
    ["results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_full_runtime_vcs_stimulus_r2.json"]="bb53a221efe2a555247e58411d0d3de0a9a45dd5c551b59ceaa875319e7d1619"
    ["results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_phase_config_768.memh"]="b35178f06529940403fea28b3d04dbf56eb8686a90f11e20d5678e6cfb348c04"
    ["results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"]="6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
    ["results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS"]="c0db8a02abe47bd43c8131febb3b6968cb2cc36e911b450c17f5b6bd847056bc"
    ["results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256"]="31abafb9e39e2a9fa39b348b0ab9954805ec94e58f1006a6f2d57e5d24946efc"
    ["results/m410_h67_q32_full_runtime_vcs_r1_20260826/sim.log"]="bcff561148560aa48150e752fba006c2a0a26e1ac65f3ede23fd2dbe7856d7d0"
    ["results/m410_h67_q32_full_runtime_vcs_r1_20260826/RUN_FAILED_OR_INCOMPLETE.txt"]="0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "${task_path}" \
        "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
sha256sum -c \
    results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS \
    >"${task_run}/m409_seal_check.log" 2>&1
sha256sum -c \
    results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    >>"${task_run}/m409_seal_check.log" 2>&1
(
    cd results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_run}/stimulus_seal_check.log" 2>&1
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m410r2_h67_q32_full_runtime_vcs_contract_r2_20260826.json \
    "${task_run}/contract.json"
export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m410_h67_q32_full_runtime_vcs.f \
    -top tb_m410_h67_q32_full_runtime_vcs \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

task_start="$(date +%s)"
set +e
"${task_run}/simv" +ntb_random_seed=41020260826 \
    "+M410_CONFIG=${task_config}" "+M410_ROWS=${task_rows}" \
    -no_save -cm assert -assert "report=${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
task_end="$(date +%s)"
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
printf '%s\n' "$((task_end-task_start))" >"${task_run}/sim_wall_seconds.txt"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M410 full ordered q32 runtime phases=17280 configs=17280 rows=51840000 pass0=51840000 pass1=16037540 early=3751608 pwp=16971357 results=51840000 metadata_mismatches=0 arithmetic_mismatches=0 config_mismatches=0 task_flag_mismatches=0 tie_lowest_id=true exact_runtime_order=true system_speedup=false headline=false cycles=[0-9]+' \
    "${task_run}/sim.log" || exit 24

python3 - "${task_run}" \
    "${task_run}/m410r2_h67_q32_full_runtime_vcs_receipt_r2.json" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "sim.log").read_text(errors="replace")
match = re.search(
    r"PASS M410 full ordered q32 runtime phases=(\d+) configs=(\d+) "
    r"rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) pwp=(\d+) "
    r"results=(\d+) metadata_mismatches=(\d+) arithmetic_mismatches=(\d+) "
    r"config_mismatches=(\d+) task_flag_mismatches=(\d+).*cycles=(\d+)",
    text)
if not match:
    raise SystemExit("missing M410 PASS ledger")
values = [int(value) for value in match.groups()]
expected = [17280, 17280, 51840000, 51840000, 16037540,
            3751608, 16971357, 51840000, 0, 0, 0, 0]
if values[:12] != expected:
    raise SystemExit("M410 PASS ledger drift")
receipt = {
    "schema": "m410r2_h67_q32_full_runtime_vcs_receipt_v2",
    "status": "PASS_M410R2_H67_Q32_CONTRACT_VISIBLE_FULL_ORDERED_RUNTIME_SYNOPSYS_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "population": {
        "phases": values[0],
        "configs": values[1],
        "source_rows": values[2],
        "pass0_tasks": values[3],
        "pass1_tasks": values[4],
        "early_stops": values[5],
        "pwp_rows": values[6],
        "results": values[7],
        "matcher_task_cycles": values[3] + values[4],
        "m401_matcher_cycles_with_two_cycle_phase_overhead":
            values[3] + values[4] + 2 * values[0],
    },
    "mismatches": {
        "metadata": values[8],
        "arithmetic": values[9],
        "config": values[10],
        "task_flags": values[11],
        "protocol": 0,
        "assertion": 0,
    },
    "testbench_raw_cycles_not_speed_metric": values[12],
    "sim_wall_seconds": int((root / "sim_wall_seconds.txt").read_text()),
    "claim_boundary": {
        "full_ordered_runtime_vcs": True,
        "rtl_realtrace_matcher_task_ledger": True,
        "full_selected_slice_realtrace_vcs": False,
        "rtl_measured_speedup": False,
        "system_speedup": False,
        "energy": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2,
                                       sort_keys=True) + "\n")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' PASS_M410R2_H67_Q32_CONTRACT_VISIBLE_FULL_ORDERED_RUNTIME_SYNOPSYS_VCS \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M410R2 full runtime VCS sealed at ${task_run}"
