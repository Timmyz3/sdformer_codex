#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_counterexamples_r1"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M121 independent run: $task_run" >&2
    exit 2
fi
mkdir "$task_run"
cd "$task_hw_root"

sha256sum \
    rtl_m117/m117_w384_prefetch_transpose_scheduler.sv \
    rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv \
    rtl_m118/m118_w384_signed19_accumulator_frontend.sv \
    rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv \
    rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv \
    rtl_m121/m121_w384_scheduler_numeric_island.sv \
    reviews/m121_w384_scheduler_numeric_island_independent_hammer_r1_20260824/m121_independent.f \
    reviews/m121_w384_scheduler_numeric_island_independent_hammer_r1_20260824/tb_m121_independent_counterexamples.sv \
    > "$task_run/input_sha256.txt"

"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" \
    -f reviews/m121_w384_scheduler_numeric_island_independent_hammer_r1_20260824/m121_independent.f \
    -top tb_m121_independent_counterexamples -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
printf '0\n' > "$task_run/compile.rc"
"$task_run/simv" -no_save > "$task_run/sim.raw.log" 2>&1
printf '0\n' > "$task_run/sim.rc"
grep -q '^COUNTEREXAMPLE descriptor_replay_accepted ' "$task_run/sim.raw.log"
grep -q '^COUNTEREXAMPLE scheduler_fault_commit_escape ' "$task_run/sim.raw.log"
grep -q '^COUNTEREXAMPLE delayed_weight_response ' "$task_run/sim.raw.log"
grep -q '^PASS M121 independent counterexamples ' "$task_run/sim.raw.log"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/compile.rc "$task_run"/sim.rc > "$task_run/output_sha256.txt"
printf '%s\n' 'status=PASS_M121_INDEPENDENT_COUNTEREXAMPLES' \
    'commercial_vcs=true' \
    'held_valid_exact_grace=true' \
    'whole_descriptor_replay_accepted=true' \
    'scheduler_fault_commit_escape=true' \
    'delayed_weight_response_data_corruption_with_counts_intact=true' \
    'production_modified=false' > "$task_run/RUN_COMPLETE.txt"
echo "PASS M121 independent VCS counterexamples"
