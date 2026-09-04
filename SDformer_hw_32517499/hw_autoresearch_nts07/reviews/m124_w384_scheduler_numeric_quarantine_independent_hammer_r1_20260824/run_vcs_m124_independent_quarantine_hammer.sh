#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_quarantine_hammer_r1"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M124 independent run: $task_run" >&2
    exit 2
fi
mkdir "$task_run"
cd "$task_hw_root"

sha256sum \
    rtl_m117/m117_w384_prefetch_transpose_scheduler.sv \
    rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv \
    rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv \
    rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv \
    rtl_m124/m124_pwp_tail_mapper_signed19_forwarding_accumulator_island.sv \
    rtl_m124/m124_w384_scheduler_numeric_quarantine_island.sv \
    reviews/m124_w384_scheduler_numeric_quarantine_independent_hammer_r1_20260824/m124_independent.f \
    reviews/m124_w384_scheduler_numeric_quarantine_independent_hammer_r1_20260824/tb_m124_independent_quarantine_hammer.sv \
    > "$task_run/input_sha256.txt"

"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" \
    -f reviews/m124_w384_scheduler_numeric_quarantine_independent_hammer_r1_20260824/m124_independent.f \
    -top tb_m124_independent_quarantine_hammer -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
printf '0\n' > "$task_run/compile.rc"
"$task_run/simv" -no_save > "$task_run/sim.raw.log" 2>&1
printf '0\n' > "$task_run/sim.rc"

grep -q '^CLOSURE m121_scheduler_fault_end_commit_p0 ' "$task_run/sim.raw.log"
grep -q '^CLOSURE same_cycle_scheduler_fault_end_event_close_and_continuous_valid ' "$task_run/sim.raw.log"
grep -q '^CLOSURE same_cycle_scheduler_fault_start ' "$task_run/sim.raw.log"
grep -q '^CLOSURE reset_only_recovery ' "$task_run/sim.raw.log"
grep -q '^CLOSURE same_cycle_prefetch_fault ' "$task_run/sim.raw.log"
grep -q '^CLOSURE same_cycle_service_weight_fault ' "$task_run/sim.raw.log"
grep -q '^CLOSURE same_cycle_commit_fault ' "$task_run/sim.raw.log"
grep -q '^CLOSURE older_accepted_update_lane_write_drain ' "$task_run/sim.raw.log"
grep -q '^CLOSURE numeric_fault_same_cycle_and_sticky ' "$task_run/sim.raw.log"
grep -q '^CLOSURE accept_observation_consistency ' "$task_run/sim.raw.log"
grep -q '^PASS M124 independent quarantine hammer ' "$task_run/sim.raw.log"

set +e
/usr/bin/timeout --signal=TERM --kill-after=1s 3s \
    "$task_run/simv" +CROSS_FAULT_LOOP -no_save \
    > "$task_run/comb_loop_sim.raw.log" 2>&1
task_loop_rc=$?
set -e
printf '%s\n' "$task_loop_rc" > "$task_run/comb_loop_sim.rc"
if [[ "$task_loop_rc" != 124 && "$task_loop_rc" != 137 ]]; then
    echo "expected zero-time M124 cross-fault timeout, got rc=$task_loop_rc" >&2
    exit 3
fi
grep -q '^ARMED M124 cross_fault_comb_loop ' "$task_run/comb_loop_sim.raw.log"
if grep -q '^UNEXPECTED_ADVANCE M124 cross_fault_comb_loop' \
        "$task_run/comb_loop_sim.raw.log"; then
    echo "M124 cross-fault reproducer advanced past #0.1" >&2
    exit 4
fi

sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/compile.rc "$task_run"/sim.rc \
    "$task_run"/comb_loop_sim.raw.log \
    "$task_run"/comb_loop_sim.rc \
    > "$task_run/output_sha256.txt"
printf '%s\n' \
    'status=REVIEW_COMPLETE_M124_P0_CROSS_FAULT_COMBINATIONAL_LOOP' \
    'commercial_vcs=true' \
    'm121_scheduler_fault_end_commit_p0_closed=true' \
    'single_domain_same_cycle_public_quarantine=true' \
    'post_fault_continuous_valid_accepts=0' \
    'older_accepted_lane_write_drain_observed=true' \
    'post_fault_commit_outputs=0' \
    'numeric_fault_quarantine=true' \
    'reset_only_recovery_observed=true' \
    'scheduler_numeric_accept_mismatches=0' \
    'cross_domain_simultaneous_fault_combinational_loop=true' \
    "cross_domain_comb_loop_timeout_rc=$task_loop_rc" \
    'm123_instantiated=true' \
    'weight_response_valid=false' \
    'whole_descriptor_retry_deduplication=false' \
    'production_modified=false' > "$task_run/RUN_COMPLETE.txt"
echo "REVIEW COMPLETE M124 independent commercial VCS: P0 cross-fault combinational loop reproduced"
