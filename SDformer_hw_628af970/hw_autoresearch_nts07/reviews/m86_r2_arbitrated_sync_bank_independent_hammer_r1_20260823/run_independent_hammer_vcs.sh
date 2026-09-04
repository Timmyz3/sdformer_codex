#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$REVIEW_DIR/../.." && pwd)"
RUN_DIR="$REVIEW_DIR/independent_hammer_vcs"
if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86-R2 independent run: $RUN_DIR" >&2
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
    rtl_m86_r2/arbitrated_sync_banked_guarded_pwp_frontend.sv \
    reviews/m86_r2_arbitrated_sync_bank_independent_hammer_r1_20260823/tb_m86_r2_hammer.sv \
    reviews/m86_r2_arbitrated_sync_bank_independent_hammer_r1_20260823/independent_hammer_vcs.f \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f \
    reviews/m86_r2_arbitrated_sync_bank_independent_hammer_r1_20260823/independent_hammer_vcs.f \
    -top tb_m86_r2_hammer -o "$RUN_DIR/simv" \
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
grep -Eq '^PASS M86-R2 independent hammer exact_r1_trigger_closed=2 phase_payload_silent_deadlock_cycles=4 bounded_descriptor_priority_accepts=8 losing_payload_recovery=1 error_propagation_classes=4 onehot_checks=[1-9][0-9]* bank_issues=[1-9][0-9]* bank_responses=[1-9][0-9]* outputs=8$' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$RUN_DIR/sim.raw.log"; then
    exit 31
fi
printf '%s\n' \
    'status=PASS_M86_R2_INDEPENDENT_CONTENTION_PHASE_ERROR_HAMMER' \
    'exact_r1_payload_descriptor_trigger_closed=2' \
    'phase_payload_silent_deadlock_cycles=4' \
    'bounded_descriptor_priority_accepts=8' \
    'losing_payload_eventual_recovery_after_winner_stops=1' \
    'unbounded_starvation_freedom=false' \
    'error_propagation_classes=4' \
    'actual_record_replay=false' \
    'rtl_cycle_speedup=false' \
    'system_speedup=false' \
    > "$RUN_DIR/RUN_COMPLETE.txt"
complete=1
echo "PASS M86-R2 independent hammer VCS"
