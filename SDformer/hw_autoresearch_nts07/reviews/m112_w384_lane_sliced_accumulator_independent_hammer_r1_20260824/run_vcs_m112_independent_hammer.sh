#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_sealed"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite independent M112 sealed run: $task_run" >&2
    exit 2
fi
mkdir "$task_run"
task_complete=0
on_exit() {
    local task_rc="$?"
    if [[ "$task_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$task_rc"
        } > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$task_review_dir"
sha256sum -c input_manifest.sha256 > "$task_run/preflight_sha_checks.txt"

cd "$task_hw_root"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "reviews/m112_w384_lane_sliced_accumulator_independent_hammer_r1_20260824/m112_independent.f" \
    -top tb_m112_independent_hammer -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" -no_save \
    -assert "report=$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert -cm_name independent \
    > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M112 INDEPENDENT HAMMER commercial_vcs=true wrapper_cycle_equivalent_m111=true windows=2 reverse_updates=256 writes=256 ii1_pairs=255 read_write_overlap=255 commits=6144 lane_result_checks=589824 exact_flat_read_checks=518 exact_flat_write_checks=261 lane_write_slice_checks=25056 flat_zero=true flat_3071=true lazy_stale_zero=1 commit_stalls=2242 same_address_preserve=1 row384_range=1 positive_overflow=1 negative_overflow=1 collision=1 lane_macros=96 macro_depth=3072 macro_width=24 logical_accumulator_bytes=884736 valid_bits=3072 behavioral_sync_1r1w=true foundry_macro=false m109_r2_2p535_is_projection=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' \
    "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|Fatal:' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi
for task_cover in \
        'cp_update_ii1, .* 259 match' \
        'cp_dual_port_overlap, .* 259 match' \
        'cp_flat_zero_read, .* 2 match' \
        'cp_flat_last_read, .* 4 match' \
        'cp_flat_zero_write, .* 1 match' \
        'cp_flat_last_write, .* 3 match' \
        'cp_commit_stall_release, .* 1943 match' \
        'cp_complete, .* 2 match' \
        'cp_fault, .* 5 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M112_INDEPENDENT_WRAPPER_MAPPING_NUMERIC_PROTOCOL_DIRECTED"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "exact_sha=true"
    echo "compile_return_code=0"
    echo "simulation_return_code=0"
    echo "assertion_failures=0"
    echo "wrapper_cycle_equivalent_to_frozen_m111=true"
    echo "flatten_address_exact_block_times_384_plus_row=true"
    echo "flatten_address_min=0"
    echo "flatten_address_max=3071"
    echo "lane_write_slice_mapping_checks=25056"
    echo "windows=2"
    echo "reverse_updates=256"
    echo "nonconflicting_update_ii1_pairs=255"
    echo "read_write_overlap_cycles=255"
    echo "commit_vectors=6144"
    echo "signed_lane_result_checks=589824"
    echo "commit_backpressure_stalls=2242"
    echo "lazy_stale_memory_commits_zero=true"
    echo "same_address_older_write_preserved_fail_closed=true"
    echo "row384_fail_closed=true"
    echo "positive_overflow_suppressed=true"
    echo "negative_overflow_suppressed=true"
    echo "collision_fail_closed=true"
    echo "lane_macro_count=96"
    echo "lane_macro_depth=3072"
    echo "lane_macro_width_bits=24"
    echo "behavioral_sync_1r1w=true"
    echo "foundry_macro=false"
    echo "logical_accumulator_bytes=884736"
    echo "lazy_valid_bits=3072"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "m109_r2_ratio_is_projection=true"
    echo "actual_heldout_integrated_replay=false"
    echo "scheduled_cycle_ratio=false"
    echo "macro_inclusive_ppa=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"

sha256sum "$task_run"/compile.raw.log "$task_run"/compile.rc \
    "$task_run"/sim.raw.log "$task_run"/sim.rc \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"

task_complete=1
echo "PASS independent M112 lane-sliced accumulator hammer completed"
