#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$HW_ROOT/reviews/m84_hierarchical_pwp_bank_mapper_vcs_independent_hammer_r1_20260823"
RUN_DIR="${RUN_DIR:-$HERE/independent_directed_random_run}"
RTL="$HW_ROOT/rtl_m84/hierarchical_pwp_bank_mapper.sv"
TB="$HERE/tb_m84_independent_directed_random.sv"
FILELIST="$HERE/independent_directed_random.f"

[[ ! -e "$RUN_DIR" ]] || { echo "refusing overwrite: $RUN_DIR" >&2; exit 2; }
[[ "$(sha256sum "$RTL" | awk '{print $1}')" == \
   "8dafcf1e049dfee1a06999c93010a6e8c2458cc17c9a2de712b26d4fc40a2067" ]]
mkdir "$RUN_DIR"
sha256sum "$RTL" "$TB" "$FILELIST" > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
cd "$HW_ROOT"
"$VCS_HOME/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_m84_independent_directed_random -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
[[ "$compile_rc" -eq 0 && -x "$RUN_DIR/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
[[ "$sim_rc" -eq 0 ]]
grep -qx 'PASS M84 independent directed_random directed=24 random=4096 start_mod8=8 pattern15_block7=5 escape_neighbor=3 selected_reserved_blocked=3 prior_reserved_failopen=3 overflow_wrap=2' "$RUN_DIR/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal' "$RUN_DIR/sim.raw.log"
{
    echo 'status=PASS_M84_INDEPENDENT_DIRECTED_RANDOM_VCS'
    echo 'directed_checks=24'
    echo 'random_checks=4096'
    echo 'selected_reserved_codes_blocked=3'
    echo 'prior_reserved_failopen_observed=3'
    echo 'overflow_wrap_observed=2'
} > "$RUN_DIR/RUN_COMPLETE.txt"
echo "PASS M84 independent directed/random VCS at $RUN_DIR"
