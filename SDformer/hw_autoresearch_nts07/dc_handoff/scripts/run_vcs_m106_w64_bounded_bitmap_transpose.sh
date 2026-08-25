#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m106_w64_bounded_bitmap_transpose_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M106 sealed VCS run: $task_run" >&2
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
task_contract="contracts/m106_w64_bounded_bitmap_transpose_vcs_contract_r1_20260824.json"
task_m105="reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/m105_bounded_row_transpose_preflight.json"
task_m105_review="reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/m105_bounded_row_transpose_independent_hammer_review.json"
task_m105_manifest="reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/manifest.sha256"
task_m41="results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json"

declare -A task_expected=(
    ["$task_rtl"]="0abc1adf612788bbfdd2f26ff847234ee7efaaa2addcc7f28f03ddac22cd68e7"
    ["$task_sva"]="db98dc72b18aa789088bdbea40ab1b5a6cd7399b2bd8d373b37d17a5bcfba227"
    ["$task_tb"]="cc0bbca17b61be28453e0ba79999c947347570490731760fec7a560c026b861c"
    ["$task_vcs_files"]="889163237de5394c39cfdf5edcfecd9670b69179c756dd9e5eaa9424f19692fc"
    ["$task_dc_files"]="c7aeab860833b025c33c497b1e7c8ac9d2d0fefaa7b325dd278521bab4580bb3"
    ["$task_contract"]="881491f58543f2c6b0b5b3c1d07d7b170cdbfb4190153a18929bdddd83a39999"
    ["$task_m105"]="3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b"
    ["$task_m105_review"]="467ae8bbc6fe14712cd9059e8c341ba0691b24d873c5f39cbab1e96efb9dc051"
    ["$task_m105_manifest"]="7123b60c43199811d907a71890ca3f171ffb86b04230a96eb409259427b9186b"
    ["$task_m41"]="20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M106 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_vcs_files" \
    -top tb_m106_bounded_bitmap_transpose_scheduler \
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
grep -qx 'PASS M106 bounded bitmap transpose windows=2 ingress_events=7 keys=5 load_tokens=15 event_tokens=7 service_tokens=22 stalls=3 event_grace=1 close_grace=1 protocol_attacks=3 win_rows=64 bitmap_payload_bits=32768 accumulator_contract_bits=24 accumulator_port_cut=true macros=0' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_ping_pong_overlap, .* 2 match' \
        'cp_event_ii1, .* 2 match' \
        'cp_key_turnover_without_idle, .* 3 match' \
        'cp_event_grace, .* 1 match' \
        'cp_close_grace, .* 1 match' \
        'cp_fault, .* 8 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M106_W64_BOUNDED_BITMAP_TRANSPOSE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "window_rows=64"
    echo "key_count=128"
    echo "event_bitmap_payload_bits=32768"
    echo "conditional_token_ratio=2.143907497115123"
    echo "conditional_control_charged_ratio=2.1422339037663227"
    echo "actual_record_replay=false"
    echo "accumulator_port_cut=true"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "equal_area=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m106_w64_bounded_bitmap_transpose.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M106 w64 bounded bitmap transpose sealed at $task_run"
