#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_dc="${SYNOPSYS_DC_HOME:-/opt/synopsys/syn/V-2023.12-SP3}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
task_sdc="$task_hw_root/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"

cd "$task_hw_root"
rm -f "$task_review/RUN_COMPLETE.txt"
sha256sum \
    rtl_m125/m125_block_phased_k4_row_fold.sv \
    rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv \
    rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv \
    rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv \
    > "$task_review/preflight_sha_checks.txt"
python3 "$task_review/build_m126_registered_fault_barrier_delta.py" \
    > "$task_review/delta_generation.log"

rm -rf "$task_review/original_dc" "$task_review/delta_dc" \
       "$task_review/delta_vcs"
mkdir -p "$task_review/original_dc" "$task_review/delta_dc" \
         "$task_review/delta_vcs"

run_dc_check() {
    local task_name="$1"
    local task_filelist="$2"
    local task_output="$task_review/$task_name"
    set +e
    HW_ROOT="$task_hw_root" RTL_FILELIST="$task_filelist" \
    LIB_DB="$task_lib" MIN_LIB_DB="$task_min_lib" SDC_FILE="$task_sdc" \
    OUTPUT_DIR="$task_output" \
    "$task_dc/bin/dc_shell" -f "$task_review/check_timing_only.tcl" \
        > "$task_output/dc.raw.log" 2>&1
    local task_rc="$?"
    set -e
    printf '%s\n' "$task_rc" > "$task_output/dc.rc"
    [[ "$task_rc" -eq 0 ]]
}

run_dc_check original_dc "$task_review/original_dc.f"
grep -q 'Warning: timing loops detected. (TIM-209)' \
    "$task_review/original_dc/check_timing.rpt"

run_dc_check delta_dc "$task_review/delta_dc.f"
if grep -q 'Warning: timing loops detected. (TIM-209)' \
        "$task_review/delta_dc/check_timing.rpt"; then
    echo 'FAIL review delta still has TIM-209' >&2
    exit 20
fi
if grep -q 'Timing loop detected. (OPT-150)' \
        "$task_review/delta_dc/dc.raw.log"; then
    echo 'FAIL review delta still has OPT-150' >&2
    exit 21
fi

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$task_review/delta_vcs/csrc" \
    -f "$task_review/delta_vcs.f" \
    -top tb_m126_block_phased_k4_forwarding_accumulator_island \
    -o "$task_review/delta_vcs/simv" \
    > "$task_review/delta_vcs/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_review/delta_vcs/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_review/delta_vcs/simv" ]]
if grep -Eiq 'Warning-\[|Error-\[|^Error' \
        "$task_review/delta_vcs/compile.raw.log"; then
    echo 'FAIL VCS compile warning/error' >&2
    exit 22
fi
set +e
"$task_review/delta_vcs/simv" -no_save \
    -assert report="$task_review/delta_vcs/assert.report" \
    > "$task_review/delta_vcs/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_review/delta_vcs/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -q '^PASS M126 K4 fold plus forwarding accumulator VCS ' \
    "$task_review/delta_vcs/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_review/delta_vcs/sim.raw.log" \
        "$task_review/delta_vcs/assert.report"; then
    echo 'FAIL VCS runtime/assertion failure' >&2
    exit 23
fi

sha256sum \
    "$task_review/original_dc"/{dc.raw.log,dc.rc,check_timing.rpt} \
    "$task_review/delta_dc"/{dc.raw.log,dc.rc,check_timing.rpt} \
    "$task_review/delta_vcs"/{compile.raw.log,compile.rc,sim.raw.log,sim.rc,assert.report} \
    "$task_review/m125_registered_state_busy_delta.sv" \
    "$task_review/m126_registered_fault_barrier_delta.sv" \
    > "$task_review/reproduction_outputs.sha256"
{
    echo 'status=PASS_M126_LOOP_REPRODUCTION_AND_REVIEW_ONLY_BREAK'
    echo 'production_precompile_timing_loop=true'
    echo 'tool_only_anomaly=false'
    echo 'review_delta_precompile_timing_loop=false'
    echo 'review_delta_functional_vcs=true'
    echo 'production_dc_citable=false'
    echo 'paper_ppa_ready=false'
} > "$task_review/RUN_COMPLETE.txt"
echo "PASS M126 loop reproduction at $task_review"
