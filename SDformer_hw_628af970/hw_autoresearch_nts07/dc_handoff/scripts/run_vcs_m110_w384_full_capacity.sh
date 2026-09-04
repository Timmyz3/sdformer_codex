#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m110_w384_full_capacity_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M110 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m110/m110_w384_bounded_bitmap_transpose_scheduler.sv"
task_sva="verif_m110/m110_w384_bounded_bitmap_transpose_assertions.sv"
task_tb="tb_m110/tb_m110_w384_bounded_bitmap_transpose.sv"
task_files="dc_handoff/filelists/date_m110_w384_bounded_bitmap_transpose_directed_vcs.f"
task_contract="contracts/m110_w384_full_capacity_transpose_vcs_contract_r1_20260824.json"
task_m109_contract="contracts/m109_r2_window_storage_dual_timeline_frontier_contract_r1_20260824.json"
task_m109_manifest="results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/manifest.sha256"
task_m106_review="reviews/m106_r2_standard_streaming_grace_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="61a2c18f3b0a350bfc57193b9573f3d0ed5ea68f68ae4fc982ec1908054dbd6c"
    ["$task_sva"]="daf98af5808c58d90b7428eeb42061a956bbe6b4889a52dadf5b47d4f83bc8cf"
    ["$task_tb"]="1a59afc90e2a3e6c4b6edb233951e9811c89765ed610b4af4a80f5a85d7f70d4"
    ["$task_files"]="8f632f6976e9da38ec67a46cad75634092a06d0da63fa77b9da2a4dd34e2a741"
    ["$task_contract"]="4f2b5c329ea552742c55a362739f032272fb510cc3c659b0c73f52eced9f5253"
    ["$task_m109_contract"]="d80efd387bb6b5b01371ca7ed5d07d8e2ec97f3efa93aa1d385cc80281f63b44"
    ["$task_m109_manifest"]="3da3f081f867edadf4767ddd78413cfec4a6187055ab6ec3d2fb5b365be3b1ba"
    ["$task_m106_review"]="d0c297b3a2158e8ce55c12ed344667f29db547dc7c4fda47b119a713fe505eb5"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M110 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m110_w384_bounded_bitmap_transpose \
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
grep -qx 'PASS M110 W384 full-capacity VCS windows=2 ingress_events=98304 active_keys=256 rows_per_key=384 load_tokens=768 event_tokens=98304 service_tokens=99072 ii1_pairs=98302 stalls=9952 overlap_cycles=49152 close_grace=2 protocol_attacks=1 win_rows=384 bitmap_payload_bits=196608 accumulator_contract_bits=24 accumulator_implemented=false macros=0 scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_pingpong_overlap, .* 49152 match' \
        'cp_last_row, .* 256 match' \
        'cp_full_key_identity, .* 2 match' \
        'cp_stall, .* 9510 match' \
        'cp_fault, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M110_W384_FULL_CAPACITY_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "window_rows=384"
    echo "full_keys=128"
    echo "full_windows=2"
    echo "ingress_events=98304"
    echo "service_tokens=99072"
    echo "event_initiation_interval=1"
    echo "bitmap_payload_bits=196608"
    echo "w384_controller_geometry_vcs=true"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "accumulator_implemented=false"
    echo "actual_heldout_record_replay=false"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m110_w384_full_capacity.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M110 W384 full-capacity VCS sealed at $task_run"
