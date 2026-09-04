#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_sealed"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite independent M110 sealed run: $task_run" >&2
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
    -f "reviews/m110_w384_full_capacity_transpose_independent_hammer_r1_20260824/m110_w384_independent.f" \
    -top tb_m110_w384_independent -o "$task_run/simv" \
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
grep -qx 'PASS M110 independent W384 full-capacity reverse windows=2 ingress_events=98304 active_keys=256 rows_per_key=384 load_tokens=768 event_tokens=98304 service_tokens=99072 event_ii1_pairs=98302 stalls=17708 overlap_cycles=49153 exact_event_grace=2 exact_close_grace=2 cross_bank_close_grace=1 changed_legal_close_run=2 protocol_attacks=5 reset_recoveries=8 presence_bits=98304 direction_bits=98304 raw_bitmap_bits=196608 metadata_bits_min=314 accumulator_implemented=false m109_r2_ratio_2p535_is_projection=true actual_record_replay=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' \
    "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|Fatal:' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi
for task_cover in \
        'cp_event_stream_ii1, .* 98302 match' \
        'cp_changed_legal_close, .* 1 match' \
        'cp_exact_close_cross_bank_grace, .* 1 match' \
        'cp_service_stall, .* 17712 match' \
        'cp_last_row_full_key, .* 2 match' \
        'cp_out_of_range, .* 1 match' \
        'cp_fault, .* 20 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M110_INDEPENDENT_W384_FULL_CAPACITY_REVERSE_STREAMING"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "exact_sha=true"
    echo "compile_return_code=0"
    echo "simulation_return_code=0"
    echo "assertion_failures=0"
    echo "window_rows=384"
    echo "full_keys=128"
    echo "full_windows=2"
    echo "continuous_changed_event_accepts_per_window=49152"
    echo "event_ii1_pairs=98302"
    echo "service_tokens=99072"
    echo "reverse_ingress_sorted_drain_scoreboard=true"
    echo "exact_event_grace_no_reaccept=true"
    echo "exact_cross_bank_close_grace_no_reaccept=true"
    echo "changed_legal_close_ii1=true"
    echo "service_stall_stability=true"
    echo "row384_out_of_range_fail_closed=true"
    echo "duplicate_context_collision_unavailable_fail_closed=true"
    echo "fault_sticky_reset_only=true"
    echo "presence_bits=98304"
    echo "direction_bits=98304"
    echo "raw_bitmap_payload_bits=196608"
    echo "minimum_bank_metadata_bits=314"
    echo "w384_controller_geometry_vcs=true"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "m109_r2_ratio_is_projection=true"
    echo "accumulator_implemented=false"
    echo "actual_heldout_record_replay=false"
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
echo "PASS independent M110 W384 full-capacity hammer completed"
