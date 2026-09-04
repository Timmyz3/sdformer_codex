#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$REVIEW_DIR/../.." && pwd)"
RUN_DIR="$REVIEW_DIR/independent_negative_vcs"
if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86 independent hammer run: $RUN_DIR" >&2
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
    verif_m86/sync_banked_guarded_pwp_frontend_assertions.sv \
    reviews/m86_sync_banked_guarded_pwp_independent_hammer_r1_20260823/tb_m86_hammer_negative.sv \
    reviews/m86_sync_banked_guarded_pwp_independent_hammer_r1_20260823/independent_negative_vcs.f \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f \
    reviews/m86_sync_banked_guarded_pwp_independent_hammer_r1_20260823/independent_negative_vcs.f \
    -top tb_m86_hammer_negative -o "$RUN_DIR/simv" \
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
grep -Eq '^PASS M86 independent hammer one_cycle_response=1 missing_row=1 oob_row=1 duplicate_row=1 fifo_full_hold=6 simultaneous_push_pop=[1-9][0-9]* bit_exact_outputs=12 simultaneous_valid_deadlock_cycles=4$' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$RUN_DIR/sim.raw.log"; then
    exit 31
fi
printf '%s\n' \
    'status=PASS_M86_INDEPENDENT_FLOW_AND_LOADER_ATTACKS' \
    'one_cycle_registered_response=true' \
    'missing_row_blocked=1' \
    'oob_row_fail_closed=1' \
    'duplicate_row_fail_closed=1' \
    'fifo_full_hold_cycles=6' \
    'simultaneous_push_pop_observed=true' \
    'bit_exact_outputs=12' \
    'simultaneous_valid_silent_deadlock_cycles=4' \
    'compiled_sram_macro=false' \
    'rtl_cycle_speedup=false' \
    'system_speedup=false' \
    > "$RUN_DIR/RUN_COMPLETE.txt"
complete=1
echo "PASS M86 independent hammer VCS"
