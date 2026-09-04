#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m104_held_weight_correction_broadcaster_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M104 sealed VCS run: $task_run" >&2
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
task_contract="contracts/m104_held_weight_correction_broadcaster_vcs_contract_r1_20260824.json"
task_m103_audit="reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/m103_correction_reuse_preflight_audit.json"
task_m103_review="reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/m103_correction_service_reuse_preflight_independent_hammer_review.json"
task_m103_manifest="reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/manifest.sha256"
task_m102_ledger="results/m102_r2_fail_closed_matched_vector_service_islands_vcs_cycle_ledger_r1_20260824/m102_r2_fail_closed_matched_vector_service_islands.json"

declare -A task_expected=(
    ["$task_rtl"]="37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a"
    ["$task_sva"]="ad63c0317b64b5e53aecd037d401669c42f5b4b40409563ed216e4eb776e2f98"
    ["$task_tb"]="7ed7fcf389c49dcc152a002416f6af9198fdb7c770373b6d711c828984529916"
    ["$task_files"]="a04e09b3029ee030f53e2cac6146ae13ed6c22bd96e57d86cbfae0adafbe6cbe"
    ["$task_contract"]="bbd086a36719f3682216d39450dfc86db46c9373fc508f65657cfac2277dbdd5"
    ["$task_m103_audit"]="935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe"
    ["$task_m103_review"]="402535d836d0715547a60ad07be59cfbe7572b84f4e853066ae29fa6ff7e5d26"
    ["$task_m103_manifest"]="cb964b9a43a4711c1cc4f93d7a8bfe425ceb7a2b7c647eca8f51df6b3e1c7996"
    ["$task_m102_ledger"]="a5d465b7d3361ed2ff176b4230d9051c29137aee86211cec9c3eb9ee8131aad5"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "$task_rtl" "$task_sva" "$task_tb" "$task_files" \
        "$task_contract" "$task_m103_audit" "$task_m103_review" \
        "$task_m103_manifest" "$task_m102_ledger"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M104 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "$task_rtl" "$task_sva" "$task_tb" "$task_files" \
    "$task_contract" "$task_m103_audit" "$task_m103_review" \
    "$task_m103_manifest" "$task_m102_ledger" \
    > "$task_run/input_sha256.txt"

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
grep -qx 'PASS M104 held-weight correction broadcaster groups=6 load_beats=21 events=9 ii1_pairs=5 stalls=3 protocol_attacks=10 continuation_attacks=3 buffered_fault_attacks=1 lanes=96 macros=0' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_load_beats, .* 6 match' \
        'cp_positive_event, .* 5 match' \
        'cp_negative_event, .* 4 match' \
        'cp_consecutive_events, .* 5 match' \
        'cp_output_stall, .* 4 match' \
        'cp_last_releases_key, .* 2 match' \
        'cp_protocol_fault, .* 30 match' \
        'cp_fault_quarantines_buffered_output, .* 4 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M104_HELD_WEIGHT_CORRECTION_BROADCASTER_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "load_slots_per_weight_key=3"
    echo "sustained_destination_descriptor_ii=1"
    echo "same_cycle_fault_quarantine=true"
    echo "conditional_same_clock_service_slot_ratio=2.6750597075487446"
    echo "ordered_transpose_schedule=false"
    echo "actual_record_replay=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "macro_inclusive_ppa=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/*.raw.log "$task_run"/*.report \
    "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m104_held_weight_correction_broadcaster.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M104 held-weight correction broadcaster sealed at $task_run"
