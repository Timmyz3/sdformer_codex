#!/usr/bin/env bash
set -euo pipefail

task_review_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_root/../.." && pwd)"
task_out="$task_review_root/vcs_m106_controller_serialization_r1"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_tmp="$(mktemp -d)"
trap 'rm -rf "$task_tmp"' EXIT

if [[ -e "$task_out" ]]; then
    echo "refusing to overwrite independent VCS trace: $task_out" >&2
    exit 2
fi
mkdir "$task_out"
task_rtl="$task_hw_root/rtl_m106/m106_bounded_bitmap_transpose_scheduler.sv"
task_tb="$task_review_root/tb_m108_r2_m106_controller_serialization.sv"
test "$(sha256sum "$task_rtl" | awk '{print $1}')" = \
    "a6937765aea87269c3d38123b656c72b7ee400e36b0d634f21ab9c7dbdefc0b7"
sha256sum "$task_rtl" "$task_tb" > "$task_out/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_tmp/csrc" "$task_rtl" "$task_tb" \
    -top tb_m108_r2_m106_controller_serialization -o "$task_tmp/simv" \
    > "$task_out/compile.raw.log" 2>&1
"$task_tmp/simv" -no_save > "$task_out/sim.raw.log" 2>&1
grep -qx 'PASS M108 r2 independent M106 serialization VCS first_close=3 empty_close=4 first_service=5 last_service=8 empty_release=9 close_to_release=5 prior_tokens=4 dispatch_edges=2 pwp_tokens=0 system_speedup=false headline=false physical=false' \
    "$task_out/sim.raw.log"
if grep -Eiq 'Warning-\[|Error-\[|^Error|^Fatal|watchdog' \
        "$task_out/compile.raw.log" "$task_out/sim.raw.log"; then
    exit 3
fi
{
    echo 'status=PASS_M108_R2_INDEPENDENT_M106_CONTROLLER_SERIALIZATION_VCS'
    echo 'actual_frozen_m106_rtl=true'
    echo 'empty_descriptor_pwp_tokens=0'
    echo 'prior_descriptor_service_tokens=4'
    echo 'fill_only_close_to_release_prediction=1'
    echo 'observed_close_to_release_cycles=5'
    echo 'prior_drain_serialization_and_dispatch_edge_observed=true'
    echo 'full_combined_controller_pwp_accumulator_miter=false'
    echo 'system_speedup=false'
    echo 'headline=false'
    echo 'physical_speedup=false'
} > "$task_out/RUN_COMPLETE.txt"
sha256sum "$task_out/compile.raw.log" "$task_out/sim.raw.log" \
    "$task_out/RUN_COMPLETE.txt" > "$task_out/output_sha256.txt"
echo "PASS independent M108 r2 M106 serialization VCS: $task_out"
