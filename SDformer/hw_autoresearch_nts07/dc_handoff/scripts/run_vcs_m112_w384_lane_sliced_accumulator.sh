#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M112 sealed VCS run: $task_run" >&2
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
task_adapter="rtl_m112/m112_w384_lane_sliced_accumulator_adapter.sv"
task_core="rtl_m111/m111_w384_signed24_accumulator_frontend.sv"
task_sva="verif_m112/m112_w384_lane_sliced_accumulator_assertions.sv"
task_tb="tb_m112/tb_m112_w384_lane_sliced_accumulator.sv"
task_files="dc_handoff/filelists/date_m112_w384_lane_sliced_accumulator_directed_vcs.f"
task_contract="contracts/m112_w384_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_m111="contracts/m111_w384_signed24_accumulator_vcs_contract_r1_20260824.json"
task_m111_run="dc_handoff/runs/m111_w384_signed24_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_adapter"]="ee5a2a84c8c28e113340c73195fc08eec4c975eed27622ea8eee654b3f25226e"
    ["$task_core"]="354e0de95ee4380098c09fac67af3e137b3ab8bb9f88ac706d62fe201179b43a"
    ["$task_sva"]="938373f712ef925d08fdad9aeeac4040e66b01c541f7f41416cafd76c1f4d874"
    ["$task_tb"]="7cbfa75bbe408fa080580dbe1037b04ef7c93db87e58efa68a349d154cfbee5e"
    ["$task_files"]="81e5d9570c3048461ca83563dbabc7861b604f563cfe5df4518c1ef1e8c16ea8"
    ["$task_contract"]="8eb2d82c329bd1612d2808a1edfb13345eddaa770156adf7da172a008f981f44"
    ["$task_m111"]="672dbdf2d8eea1c1ef58036a58bf2d3ca14dabb8f5feb5aed8dcbe0e036d22ef"
    ["$task_m111_run"]="9a10f6e25b4451d17ce6849624bdf205d64548e7085986db74b4e75694088bcc"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M112 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m112_w384_lane_sliced_accumulator \
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
grep -qx 'PASS M112 W384 lane-sliced accumulator VCS windows=2 updates=1056 vector_lane_checks=589824 commits=6144 lazy_valid_clears=2 positive_memory_writes=1056 ii1_pairs=1054 read_write_overlap=1054 commit_stalls=734 same_address_attacks=1 overflow_attacks=1 lanes=96 vector_bits=2304 accumulator_bytes=884736 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=24 behavioral_macro=true overflow_guard=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_update_ii1, .* 1056 match' \
        'cp_read_write_overlap, .* 1056 match' \
        'cp_commit_stall, .* 699 match' \
        'cp_full_commit, .* 2 match' \
        'cp_fault, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M112_W384_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "window_rows=384"
    echo "lane_macro_count=96"
    echo "lane_macro_depth=3072"
    echo "lane_macro_width_bits=24"
    echo "lanes=96"
    echo "signed_bits_per_lane=24"
    echo "vector_bits=2304"
    echo "logical_accumulator_bytes=884736"
    echo "lazy_valid_bits=3072"
    echo "update_initiation_interval=1_nonconflicting_addresses"
    echo "behavioral_sync_lane_sliced_1r1w_macro=true"
    echo "foundry_sram_macro=false"
    echo "full_lane_numeric_directed_miter=true"
    echo "exact_heldout_integrated_replay=false"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m112_w384_lane_sliced_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M112 W384 lane-sliced accumulator VCS sealed at $task_run"
