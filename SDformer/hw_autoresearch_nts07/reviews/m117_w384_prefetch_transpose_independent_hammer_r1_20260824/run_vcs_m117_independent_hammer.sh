#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_review_rel="reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824"
task_run_rel="$task_review_rel/vcs_sealed"
task_run="$task_hw_root/$task_run_rel"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M117 independent sealed VCS run: $task_run" >&2
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

cd "$task_hw_root"
sha256sum -c "$task_review_rel/input_manifest.sha256" \
    > "$task_run/preflight_sha_checks.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_review_rel/m117_independent.f" \
    -top tb_m117_independent_hammer -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M117 INDEPENDENT HAMMER commercial_vcs=true sparse_seeded_windows=2 full_key_windows=2 ingress_events=770 weight_prefetches=256 zero_bubble_scoreboard=254 zero_bubble_expected=254 service_events=768 load_tokens=768 service_stall_cycles=402 max_repeated_stall=14 stall_releases=63 pingpong_overlap=388 empty_descriptors=2 nonempty_descriptors=3 consecutive_empty_done=1 exact_event_grace=2 exact_close_grace=2 manual_prefetch_accepts=2 initial_prefetch_stalls=7 final_event_prefetch_stalls=6 post_prefetch_event_stalls=4 duplicate_prefetches=0 first_key_no_skip=true next_key_no_skip=true stall_identity_stable=true weight_payload_memory=false lane_sram_768b=false shared_arbiter=false numeric_mapper=false m109_2p535_is_projection=true one_bubble_per_group_ratio=2.4886483878017676 scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi
for task_cover in \
        'cp_simultaneous_zero_bubble_subset, .* 246 match' \
        'cp_early_prefetch_final_stall, .* 9 match' \
        'cp_repeated_service_stall_release, .* 63 match' \
        'cp_prefetch_repeated_stall_release, .* 2 match' \
        'cp_empty_done, .* 2 match' \
        'cp_nonempty_done, .* 3 match' \
        'cp_back_to_back_done_identities, .* 1 match' \
        'cp_pingpong_overlap, .* 388 match' \
        'cp_last_event_exact_grace_then_close, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M117_INDEPENDENT_COMMERCIAL_VCS_HAMMER"
    echo "exact_sha=true"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "sparse_seeded_windows=2"
    echo "full_key_windows=2"
    echo "weight_prefetches=256"
    echo "zero_bubble_scoreboard=254"
    echo "zero_bubble_sva_simultaneous_subset=246"
    echo "initial_prefetch_stalls=7"
    echo "final_event_prefetch_stalls=6"
    echo "post_prefetch_event_stalls=4"
    echo "duplicate_prefetches=0"
    echo "empty_descriptors=2"
    echo "nonempty_descriptors=3"
    echo "consecutive_empty_done_cycles=1"
    echo "weight_payload_memory=false"
    echo "lane_sram_768b=false"
    echo "shared_arbiter=false"
    echo "numeric_mapper=false"
    echo "m109_projection=2.53546204172554"
    echo "one_bubble_per_group_projection=2.4886483878017676"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run_rel/compile.raw.log" "$task_run_rel/compile.rc" \
    "$task_run_rel/sim.raw.log" "$task_run_rel/sim.rc" \
    "$task_run_rel/assert.report" "$task_run_rel/RUN_COMPLETE.txt" \
    > "$task_run/output_sha256.txt"
sha256sum "$task_review_rel/run_vcs_m117_independent_hammer.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M117 independent VCS sealed at $task_run"
