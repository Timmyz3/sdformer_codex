#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m106_r2_standard_streaming_grace_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M106 r2 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m106/m106_bounded_bitmap_transpose_scheduler.sv"
task_sva="verif_m106/m106_bounded_bitmap_transpose_scheduler_assertions.sv"
task_tb="tb_m106/tb_m106_bounded_bitmap_transpose_scheduler.sv"
task_vcs_files="dc_handoff/filelists/date_m106_bounded_bitmap_transpose_directed_vcs.f"
task_dc_files="dc_handoff/filelists/date_m106_bounded_bitmap_transpose_logic_only_dc.f"
task_contract="contracts/m106_r2_standard_streaming_grace_vcs_contract_r1_20260824.json"
task_r1_contract="contracts/m106_w64_bounded_bitmap_transpose_vcs_contract_r1_20260824.json"
task_r1_review="reviews/m106_w64_bounded_bitmap_transpose_independent_hammer_r1_20260824/m106_independent_hammer_review.json"
task_r1_manifest="reviews/m106_w64_bounded_bitmap_transpose_independent_hammer_r1_20260824/manifest.sha256"
task_m105="reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/m105_bounded_row_transpose_preflight.json"
task_m41="results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json"

declare -A task_expected=(
    ["$task_rtl"]="a6937765aea87269c3d38123b656c72b7ee400e36b0d634f21ab9c7dbdefc0b7"
    ["$task_sva"]="db98dc72b18aa789088bdbea40ab1b5a6cd7399b2bd8d373b37d17a5bcfba227"
    ["$task_tb"]="cb581e8b02fdf86a68fc95197c96c91673a6b6c5499829c2441a168cc6c544bb"
    ["$task_vcs_files"]="889163237de5394c39cfdf5edcfecd9670b69179c756dd9e5eaa9424f19692fc"
    ["$task_dc_files"]="c7aeab860833b025c33c497b1e7c8ac9d2d0fefaa7b325dd278521bab4580bb3"
    ["$task_contract"]="984ca6558ebbf3a58135e60b4aa889b7726532b8a4fc872acf7156f50d7d8196"
    ["$task_r1_contract"]="881491f58543f2c6b0b5b3c1d07d7b170cdbfb4190153a18929bdddd83a39999"
    ["$task_r1_review"]="62b7ba3977f88a4c0f67a81e4399f9ae95d407cb6877afada53ee1b992aa01d5"
    ["$task_r1_manifest"]="192e12dbaa6203156a04a9449ea7aead61428759e5f44f5cfed399584518c2e7"
    ["$task_m105"]="3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b"
    ["$task_m41"]="20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M106 r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_vcs_files" -top tb_m106_bounded_bitmap_transpose_scheduler \
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
grep -qx 'PASS M106 r2 standard-streaming grace windows=2 ingress_events=7 keys=5 load_tokens=15 event_tokens=7 service_tokens=22 stalls=3 event_grace=1 close_grace=2 cross_bank_close_grace=1 protocol_attacks=3 win_rows=64 bitmap_payload_bits=32768 metadata_payload_bits_min=314 accumulator_contract_bits=24 accumulator_port_cut=true macros=0' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_ping_pong_overlap, .* 2 match' \
        'cp_event_ii1, .* 2 match' \
        'cp_key_turnover_without_idle, .* 3 match' \
        'cp_event_grace, .* 1 match' \
        'cp_close_grace, .* 2 match' \
        'cp_fault, .* 8 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M106_R2_STANDARD_STREAMING_GRACE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "cross_bank_exact_close_reaccept=false"
    echo "standard_ready_valid_changed_payload=true"
    echo "event_initiation_interval=1"
    echo "bitmap_payload_bits=32768"
    echo "minimum_bank_metadata_bits=314"
    echo "conditional_token_ratio=2.143907497115123"
    echo "m107_r1_cycle_exact=false"
    echo "actual_record_replay=false"
    echo "accumulator_port_cut=true"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "macro_inclusive_ppa=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m106_r2_standard_streaming_grace.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M106 r2 standard-streaming grace sealed at $task_run"
