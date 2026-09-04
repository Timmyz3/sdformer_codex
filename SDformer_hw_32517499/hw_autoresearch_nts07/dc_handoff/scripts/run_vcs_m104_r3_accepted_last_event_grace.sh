#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m104_r3_accepted_last_event_grace_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M104 r3 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
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
task_rtl="rtl_m104/m104_held_weight_correction_broadcaster.sv"
task_sva="verif_m104/m104_held_weight_correction_broadcaster_assertions.sv"
task_tb="tb_m104/tb_m104_held_weight_correction_broadcaster.sv"
task_files="dc_handoff/filelists/date_m104_held_weight_correction_broadcaster_directed_vcs.f"
task_contract="contracts/m104_r3_accepted_last_event_grace_vcs_contract_r1_20260824.json"
task_r2_contract="contracts/m104_r2_literal_serial_token_correction_contract_r1_20260824.json"
task_r2_result="results/m104_r2_literal_serial_token_correction_r1_20260824/m104_r2_literal_serial_token_correction.json"
task_r2_review="reviews/m104_r2_literal_serial_token_correction_independent_hammer_r1_20260824/m104_r2_literal_serial_token_correction_independent_hammer_review.json"
task_r2_manifest="reviews/m104_r2_literal_serial_token_correction_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="7ea7978f431e917ee1a7835b8474af59e8f294587b1f115441388de8fb9c1ec5"
    ["$task_sva"]="d72b2e735f78f533f6da6f26d3ac3ad3a528b05520a2ed5a8b63271d62014c60"
    ["$task_tb"]="e7c09c3af9ae339db3b2b513f6644faedc5ee2c6980b6882618e36a7edf6b8cc"
    ["$task_files"]="a04e09b3029ee030f53e2cac6146ae13ed6c22bd96e57d86cbfae0adafbe6cbe"
    ["$task_contract"]="ff1aab0d45c6c57e304503940e7f29ffdabd63ca4a299485aa5310246c6a9b5a"
    ["$task_r2_contract"]="b88ec871b84342a39257497c4803db240f6898b0d5f748bb31d51966deb836c8"
    ["$task_r2_result"]="2c59c7c8836a5f7bf802f6b5eff1ccb8e2d1e3fecc074e307458cd8c08d3538e"
    ["$task_r2_review"]="129cd5598b8fb52e73c4e3df327ef833af889a7032283fb77c1ce205058773bb"
    ["$task_r2_manifest"]="3e8156bce441fb2a4a2ef8a1bc95719aaf7f180737908605c340adb0ef71c64f"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "$task_rtl" "$task_sva" "$task_tb" "$task_files" \
        "$task_contract" "$task_r2_contract" "$task_r2_result" \
        "$task_r2_review" "$task_r2_manifest"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M104 r3 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "$task_rtl" "$task_sva" "$task_tb" "$task_files" \
    "$task_contract" "$task_r2_contract" "$task_r2_result" \
    "$task_r2_review" "$task_r2_manifest" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m104_held_weight_correction_broadcaster \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M104 r3 symmetric event grace groups=7 load_beats=24 events=10 ii1_pairs=5 stalls=3 protocol_attacks=10 continuation_attacks=3 buffered_fault_attacks=1 accepted_event_grace_holds=1 lanes=96 macros=0' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_load_beats, .* 7 match' \
        'cp_positive_event, .* 5 match' \
        'cp_negative_event, .* 5 match' \
        'cp_consecutive_events, .* 5 match' \
        'cp_output_stall, .* 5 match' \
        'cp_last_releases_key, .* 3 match' \
        'cp_protocol_fault, .* 30 match' \
        'cp_fault_quarantines_buffered_output, .* 4 match' \
        'cp_accepted_event_grace, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M104_R3_ACCEPTED_LAST_EVENT_GRACE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "accepted_last_event_grace=true"
    echo "accepted_request_double_accept=false"
    echo "same_cycle_fault_quarantine=true"
    echo "conditional_same_clock_service_token_ratio=2.6679769126038075"
    echo "scheduled_cycle_speedup=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "macro_inclusive_ppa=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/*.raw.log "$task_run"/*.report \
    "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m104_r3_accepted_last_event_grace.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M104 r3 accepted-last-event grace sealed at $task_run"
