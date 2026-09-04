#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$REVIEW_DIR/../.." && pwd)"
RUN_DIR="$REVIEW_DIR/independent_negative_vcs"
if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M85 independent negative VCS: $RUN_DIR" >&2
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
    verif_m85/guarded_wordpacked_pwp_stream_assertions.sv \
    reviews/m85_guarded_wordpacked_pwp_stream_vcs_independent_hammer_r2_20260823/tb_m85_hammer_negative.sv \
    reviews/m85_guarded_wordpacked_pwp_stream_vcs_independent_hammer_r2_20260823/independent_negative_vcs.f \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f \
    reviews/m85_guarded_wordpacked_pwp_stream_vcs_independent_hammer_r2_20260823/independent_negative_vcs.f \
    -top tb_m85_hammer_negative -o "$RUN_DIR/simv" \
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
grep -qx 'PASS M85 independent negative poison=6 invalid_lookup=3 cross_row_address=8 held_output_cycles=4 elastic_overlap=1' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$RUN_DIR/sim.raw.log"; then
    exit 31
fi
printf '%s\n' \
    'status=PASS_M85_INDEPENDENT_NEGATIVE_AND_DIRECTED_BACKPRESSURE' \
    'poison_classes=6' \
    'invalid_lookup_classes=3' \
    'cross_row_addresses_checked=8' \
    'held_output_cycles=4' \
    'elastic_retire_and_next_input_overlap=1' \
    'random_backpressure=false' \
    'synchronous_sram=false' \
    'system_speedup=false' \
    > "$RUN_DIR/RUN_COMPLETE.txt"
complete=1
echo "PASS M85 independent negative VCS"
