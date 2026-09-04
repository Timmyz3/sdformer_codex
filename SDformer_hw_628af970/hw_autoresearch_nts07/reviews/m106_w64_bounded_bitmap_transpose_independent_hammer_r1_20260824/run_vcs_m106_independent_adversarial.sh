#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_sealed"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite independent M106 sealed run: $task_run" >&2
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
    -f "reviews/m106_w64_bounded_bitmap_transpose_independent_hammer_r1_20260824/m106_independent_adversarial.f" \
    -top tb_m106_independent_adversarial -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

run_case() {
    local task_case="$1"
    local task_stem="$2"
    set +e
    "$task_run/simv" "+CASE=$task_case" -no_save \
        -assert "report=$task_run/$task_stem.assert.report" \
        -cm line+cond+tgl+fsm+assert -cm_name "$task_stem" \
        > "$task_run/$task_stem.sim.raw.log" 2>&1
    local task_case_rc="$?"
    set -e
    printf '%s\n' "$task_case_rc" > "$task_run/$task_stem.sim.rc"
}

run_case POSITIVE positive
grep -qx 'PASS M106 independent positive windows=4 empty=1 full_keys=128 full_rows=64 ingress_events=8198 active_keys=133 load_tokens=399 event_tokens=8198 service_tokens=8597 stalls=7 overlaps=6 event_grace_cycles=2 close_grace_cycles=2 protocol_attacks=4 reset_recoveries=6 presence_bits=16384 direction_bits=16384 bitmap_payload_bits=32768 bank_metadata_bits_min=314 accumulator_implemented=false actual_record_replay=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' \
    "$task_run/positive.sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|Fatal:' \
        "$task_run/positive.sim.raw.log" "$task_run/positive.assert.report"; then
    exit 30
fi
for task_cover in \
        'cp_ping_pong_overlap, .* 6 match' \
        'cp_service_stall, .* 7 match' \
        'cp_event_grace, .* 2 match' \
        'cp_close_grace, .* 2 match' \
        'cp_fault, .* 16 match'; do
    grep -Eq "$task_cover" "$task_run/positive.assert.report"
done

run_case CLOSE_EXACT_HOLD close_exact_hold
grep -Eq 'ap_exact_held_close_is_not_reaccepted: .* failed at' \
    "$task_run/close_exact_hold.assert.report"
grep -Eq 'P0 M106 exact held close reaccepted across bank switch accepts=2' \
    "$task_run/close_exact_hold.sim.raw.log"

run_case EVENT_MUTATION event_mutation
grep -Eq 'ap_held_event_mutation_fails_closed: .* failed at' \
    "$task_run/event_mutation.assert.report"
grep -Eq 'P0 M106 held-event identity mutation accepted without valid-low accepts=2' \
    "$task_run/event_mutation.sim.raw.log"

run_case CLOSE_MUTATION close_mutation
grep -Eq 'ap_held_close_mutation_fails_closed: .* failed at' \
    "$task_run/close_mutation.assert.report"
grep -Eq 'P0 M106 held-close identity mutation accepted across bank switch accepts=2' \
    "$task_run/close_mutation.sim.raw.log"

{
    echo "status=FAIL_M106_ACCEPTED_VALID_CONTRACT_P0_INDEPENDENT_VCS_WITNESS"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "positive_capacity_and_bank_stress_pass=true"
    echo "positive_assertion_failures=0"
    echo "full_keys=128"
    echo "full_rows=64"
    echo "ingress_events=8198"
    echo "service_tokens=8597"
    echo "bitmap_payload_bits=32768"
    echo "bank_metadata_bits_min=314"
    echo "exact_close_hold_reaccept_witness=true"
    echo "held_event_mutation_accept_witness=true"
    echo "held_close_mutation_accept_witness=true"
    echo "accumulator_implemented=false"
    echo "next_accumulator_miter_admitted=false"
    echo "conditional_token_ratio=2.143907497115123"
    echo "conditional_control_charged_ratio=2.1422339037663227"
    echo "actual_record_replay=false"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"

sha256sum "$task_run"/compile.raw.log \
    "$task_run"/positive.sim.raw.log \
    "$task_run"/positive.assert.report \
    "$task_run"/close_exact_hold.sim.raw.log \
    "$task_run"/close_exact_hold.assert.report \
    "$task_run"/event_mutation.sim.raw.log \
    "$task_run"/event_mutation.assert.report \
    "$task_run"/close_mutation.sim.raw.log \
    "$task_run"/close_mutation.assert.report \
    "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"

task_complete=1
echo "PASS independent M106 hammer completed with expected P0 witnesses"
