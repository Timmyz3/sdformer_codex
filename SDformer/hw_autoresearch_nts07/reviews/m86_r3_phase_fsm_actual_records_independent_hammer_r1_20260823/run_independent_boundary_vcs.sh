#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$REVIEW_DIR/../.." && pwd)"
RUN_DIR="$REVIEW_DIR/independent_boundary_vcs"
if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86-R3 independent boundary run" >&2
    exit 2
fi
mkdir "$RUN_DIR"
complete=0
on_exit() {
    local rc="$?"
    if [[ "$complete" -ne 1 ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "$rc" > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"
sha256sum \
    rtl_m82/zero_bubble_elastic_pwp_stream.sv \
    rtl_m85/guarded_wordpacked_pwp_stream.sv \
    rtl_m86/sync_banked_guarded_pwp_frontend.sv \
    rtl_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend.sv \
    reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/tb_m86_r3_boundary_hammer.sv \
    reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/independent_boundary_vcs.f \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f \
    reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/independent_boundary_vcs.f \
    -top tb_m86_r3_boundary_hammer -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    exit 30
fi
grep -Eq '^PASS M86-R3 independent boundary triple_states=3 rows_459_460_461=3 descriptors_127_128_129=3 early_commit_wait=3 late_commit_wait=3 drain_stall=6 held_loader_wait=[1-9][0-9]* fault_classes=3 reset_classes=5 repeated_descriptor_accepts=128 onehot_checks=[1-9][0-9]* issue=[1-9][0-9]* response=[0-9]+$' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$RUN_DIR/sim.raw.log"; then
    exit 31
fi
printf '%s\n' \
    'status=PASS_M86_R3_INDEPENDENT_BOUNDARY_ERROR_RESET_HAMMER' \
    'triple_states=3' \
    'row_boundaries_459_460_461=3' \
    'descriptor_boundaries_127_128_129=3' \
    'early_commit_wait_cycles=3' \
    'late_commit_wait_cycles=3' \
    'drain_stall_cycles=6' \
    'held_next_loader_eventual_accept=true' \
    'fault_classes=3' \
    'reset_classes=5' \
    'repeated_descriptor_accepts=128' \
    'actual_record_replay=false' \
    'rtl_cycle_speedup=false' \
    'system_speedup=false' \
    > "$RUN_DIR/RUN_COMPLETE.txt"
complete=1
echo "PASS M86-R3 independent boundary VCS"
