#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_review_rel="reviews/m119_pwp_weight_tail_bypass_mapper_independent_hammer_r1_20260824"
task_run_rel="$task_review_rel/vcs_run_r1"
task_run="$task_hw_root/$task_run_rel"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M119 independent VCS run: $task_run" >&2
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
    -f "$task_review_rel/m119_independent.f" \
    -top tb_m119_independent_hammer -o "$task_run/simv" \
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
grep -qx 'PASS M119 INDEPENDENT HAMMER commercial_vcs=true groups=129 reverse_keys=64 permuted_keys=64 repeated_key_groups=1 weight_loads=387 weight_reads=387 fixed_1cycle_responses=387 events=513 updates=513 lane_checks=49248 tail_bypass_first_events=129 negate_events=257 int8_minus128_checks=513 int8_plus127_checks=513 negate_minus128_to_plus128=257 output_stall_cycles=3 event_backpressure_cycles=5 simultaneous_retire_accept=1 malformed_attack_classes=7 beat_retry_fail_closed=true beat_skip_fail_closed=true wrong_key_fail_closed=true wrong_type_fail_closed=true exact_event_retry_detected=false duplicate_event_accepts=2 duplicate_updates=2 older_accepted_update_fault_drains=1 behavioral_sync256=true foundry_sram=false m117_payload_p0_standalone_closed=true m117_integrated_p0_closed=false m118_exact_once_p0_closed=false m109_2p535_is_projection=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi
for task_cover in \
        'cp_fixed_latency_tail, .* 132 match' \
        'cp_event_input_backpressure, .* 2 match' \
        'cp_output_backpressure, .* 4 match' \
        'cp_fault_with_older_update, .* 1 match' \
        'cp_signed_boundary_update, .* 261 match' \
        'cp_three_loads_then_tail_event, .* 132 match' \
        'cp_event_ii1, .* 384 match' \
        'cp_fault, .* 7 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M119_INDEPENDENT_COMMERCIAL_VCS_HAMMER"
    echo "exact_sha=true"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "groups=129"
    echo "weight_loads=387"
    echo "weight_reads=387"
    echo "fixed_1cycle_responses=387"
    echo "events=513"
    echo "updates=513"
    echo "lane_checks=49248"
    echo "tail_bypass_first_events=129"
    echo "int8_minus128_checks=513"
    echo "int8_plus127_checks=513"
    echo "negate_minus128_to_plus128=257"
    echo "malformed_attack_classes=7"
    echo "exact_event_retry_detected=false"
    echo "duplicate_event_accepts=2"
    echo "duplicate_updates=2"
    echo "older_accepted_update_fault_drains=1"
    echo "behavioral_sync256=true"
    echo "foundry_sram=false"
    echo "m117_payload_p0_standalone_closed=true"
    echo "m117_integrated_p0_closed=false"
    echo "m118_exact_once_p0_closed=false"
    echo "m109_projection=2.53546204172554"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run_rel/compile.raw.log" "$task_run_rel/compile.rc" \
    "$task_run_rel/sim.raw.log" "$task_run_rel/sim.rc" \
    "$task_run_rel/assert.report" "$task_run_rel/RUN_COMPLETE.txt" \
    > "$task_run/output_sha256.txt"
sha256sum "$task_review_rel/run_vcs_m119_independent_hammer.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M119 independent VCS at $task_run"
