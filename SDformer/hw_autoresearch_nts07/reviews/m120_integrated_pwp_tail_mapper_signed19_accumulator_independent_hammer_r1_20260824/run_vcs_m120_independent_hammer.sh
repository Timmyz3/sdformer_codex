#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_review_rel="reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824"
task_run_rel="$task_review_rel/vcs_run_r1"
task_run="$task_hw_root/$task_run_rel"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M120 independent VCS run: $task_run" >&2
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
    -f "$task_review_rel/m120_independent.f" \
    -top tb_m120_independent_hammer -o "$task_run/simv" \
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
grep -Eq '^PASS M120 independent hammer commercial_vcs=true exact_source_sha=true positive_loads=768 positive_weight_reads=768 positive_events=1024 positive_updates=1024 positive_writes=1024 positive_ii1_pairs=768 positive_rw_overlap=768 mapper_lane_checks=98304 tail_bypass_hits=256 negate_events=512 commits=6144 commit_lane_checks=589824 .*same_address_events_accepted=2 same_address_updates_written=1 same_address_accept_then_loss_p0=true retry_events_accepted=3 retry_updates_written=3 retry_dedup_absent=true reset_events_accepted=1 reset_updates_written=0 reset_exact_once_undefined=true accumulator_bytes=700416 combined_bytes=725416 directed_legal_distinct_address_exact_once=true heldout_duplicate_retry_reset_exact_once=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false$' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi

{
    echo "status=PASS_M120_INDEPENDENT_HAMMER_P0_ACCEPTED_THEN_LOST_CONFIRMED"
    echo "exact_sha=true"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "positive_loads=768"
    echo "positive_weight_reads=768"
    echo "positive_events=1024"
    echo "positive_mapped_updates=1024"
    echo "positive_accumulator_writes=1024"
    echo "positive_ii1_pairs=768"
    echo "positive_read_write_overlap=768"
    echo "mapper_lane_checks=98304"
    echo "commit_vectors=6144"
    echo "commit_lane_checks=589824"
    echo "same_address_events_accepted=2"
    echo "same_address_mapped_updates=1"
    echo "same_address_accumulator_writes=1"
    echo "same_address_accept_then_loss_p0=true"
    echo "retry_events_accepted=3"
    echo "retry_updates_written=3"
    echo "retry_dedup_absent=true"
    echo "reset_events_accepted=1"
    echo "reset_updates_written=0"
    echo "reset_exact_once_undefined=true"
    echo "older_accepted_update_fault_drains=true"
    echo "directed_legal_distinct_address_exact_once=true"
    echo "full_integrated_exact_once_p0_closed=false"
    echo "foundry_sram_macro=false"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "m109_2p535_headline_admitted=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run_rel/compile.raw.log" "$task_run_rel/compile.rc" \
    "$task_run_rel/sim.raw.log" "$task_run_rel/sim.rc" \
    "$task_run_rel/assert.report" "$task_run_rel/RUN_COMPLETE.txt" \
    > "$task_run/output_sha256.txt"
sha256sum "$task_review_rel/run_vcs_m120_independent_hammer.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M120 independent VCS at $task_run"
