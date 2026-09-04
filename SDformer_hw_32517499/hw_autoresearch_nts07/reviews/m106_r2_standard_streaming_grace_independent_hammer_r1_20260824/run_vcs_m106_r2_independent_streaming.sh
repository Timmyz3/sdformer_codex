#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_run="$task_review_dir/vcs_sealed"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite independent M106 r2 sealed run: $task_run" >&2
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
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/main_csrc" \
    -f "reviews/m106_r2_standard_streaming_grace_independent_hammer_r1_20260824/m106_r2_independent_streaming.f" \
    -top tb_m106_r2_independent_streaming -o "$task_run/main_simv" \
    > "$task_run/main_compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/main_compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/main_simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/main_compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/main_simv" -no_save \
    -assert "report=$task_run/main.assert.report" \
    -cm line+cond+tgl+fsm+assert -cm_name main \
    > "$task_run/main.sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/main.sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M106 r2 independent standard-streaming full_keys=128 full_rows=64 ingress_stream_run=8192 service_tokens=8576 load_tokens=384 event_tokens=8192 exact_event_grace=1 exact_cross_bank_close_grace=1 changed_legal_close_run=2 illegal_main_attacks=4 stalls=3 presence_bits=16384 direction_bits=16384 bitmap_payload_bits=32768 metadata_bits_min=314 m107_r1_cycle_exact=false accumulator_implemented=false dc_admitted=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' \
    "$task_run/main.sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|Fatal:' \
        "$task_run/main.sim.raw.log" "$task_run/main.assert.report"; then
    exit 31
fi
for task_cover in \
        'cp_event_stream_ii1, .* 8191 match' \
        'cp_changed_legal_close, .* 1 match' \
        'cp_exact_close_cross_bank_grace, .* 1 match' \
        'cp_stall, .* 7 match' \
        'cp_fault, .* 16 match'; do
    grep -Eq "$task_cover" "$task_run/main.assert.report"
done

mkdir "$task_run/range"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SYNTHESIS -timescale=1ns/1ps \
    -Mdir="$task_run/range/csrc" \
    -f "reviews/m106_r2_standard_streaming_grace_independent_hammer_r1_20260824/m106_r2_independent_streaming.f" \
    -top tb_m106_r2_independent_range_probe -o "$task_run/range/simv" \
    > "$task_run/range/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/range/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/range/simv" ]]; then exit 40; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/range/compile.raw.log"; then
    exit 41
fi

set +e
"$task_run/range/simv" -no_save > "$task_run/range/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/range/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 50; fi
grep -qx 'PASS M106 r2 independent generic-range-probe win_rows=63 row_w=6 attacked_row=63 fail_closed=true sticky=true reset_only=true production_range_code_unrepresentable=true' \
    "$task_run/range/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|Fatal:' "$task_run/range/sim.raw.log"; then
    exit 51
fi

{
    echo "status=PASS_M106_R2_INDEPENDENT_STANDARD_STREAMING_GRACE_FULL_CAPACITY"
    echo "commercial_tool=Synopsys_VCS_V-2023.12-SP1"
    echo "exact_sha=true"
    echo "main_compile_return_code=0"
    echo "main_sim_return_code=0"
    echo "main_assertion_failures=0"
    echo "full_keys=128"
    echo "full_rows=64"
    echo "continuous_changed_event_accepts=8192"
    echo "continuous_event_ii1_cover_matches=8191"
    echo "service_tokens=8576"
    echo "exact_event_grace_no_reaccept=true"
    echo "exact_cross_bank_close_grace_no_reaccept=true"
    echo "changed_legal_close_ii1=true"
    echo "duplicate_range_context_collision_unavailable_fail_closed=true"
    echo "fault_sticky_reset_only=true"
    echo "production_range_attack_vacuous=true"
    echo "generic_range_probe_pass=true"
    echo "bitmap_payload_bits=32768"
    echo "minimum_bank_metadata_bits=314"
    echo "r1_exact_close_p0_closed=true"
    echo "r1_changed_payload_p0_superseded_by_r2_standard_streaming=true"
    echo "accumulator_implemented=false"
    echo "next_accumulator_miter_admitted=true"
    echo "logic_only_dc_launch_admitted=true"
    echo "dc_ppa_admitted=false"
    echo "m107_r1_cycle_exact=false"
    echo "actual_record_replay=false"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "macro_inclusive_ppa=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"

sha256sum \
    "$task_run/main_compile.raw.log" \
    "$task_run/main_compile.rc" \
    "$task_run/main.sim.raw.log" \
    "$task_run/main.sim.rc" \
    "$task_run/main.assert.report" \
    "$task_run/range/compile.raw.log" \
    "$task_run/range/compile.rc" \
    "$task_run/range/sim.raw.log" \
    "$task_run/range/sim.rc" \
    "$task_run/RUN_COMPLETE.txt" > "$task_run/output_sha256.txt"

task_complete=1
echo "PASS independent M106 r2 standard-streaming grace hammer completed"
