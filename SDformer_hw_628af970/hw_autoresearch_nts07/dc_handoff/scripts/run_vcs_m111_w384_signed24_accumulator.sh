#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m111_w384_signed24_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M111 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m111/m111_w384_signed24_accumulator_frontend.sv"
task_sva="verif_m111/m111_w384_signed24_accumulator_assertions.sv"
task_tb="tb_m111/tb_m111_w384_signed24_accumulator_frontend.sv"
task_files="dc_handoff/filelists/date_m111_w384_signed24_accumulator_directed_vcs.f"
task_contract="contracts/m111_w384_signed24_accumulator_vcs_contract_r1_20260824.json"
task_m109="contracts/m109_r2_window_storage_dual_timeline_frontier_contract_r1_20260824.json"
task_m110="contracts/m110_w384_full_capacity_transpose_vcs_contract_r1_20260824.json"
task_m110_run="dc_handoff/runs/m110_w384_full_capacity_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_rtl"]="354e0de95ee4380098c09fac67af3e137b3ab8bb9f88ac706d62fe201179b43a"
    ["$task_sva"]="5b8ff1446d109339ec9d4eb97a610c4575e5514b5fd6be222704ac3a4205b7d5"
    ["$task_tb"]="a423033e2d491efcf37f8f554f5952528112143fe8262e120de2aad46603d61e"
    ["$task_files"]="154245121831dd758f2179ac1b9f1c04c34ef5e06981eacf160c41705c0efca0"
    ["$task_contract"]="672dbdf2d8eea1c1ef58036a58bf2d3ca14dabb8f5feb5aed8dcbe0e036d22ef"
    ["$task_m109"]="d80efd387bb6b5b01371ca7ed5d07d8e2ec97f3efa93aa1d385cc80281f63b44"
    ["$task_m110"]="4f2b5c329ea552742c55a362739f032272fb510cc3c659b0c73f52eced9f5253"
    ["$task_m110_run"]="2b73e6e29fcd176ab17d479fa33c0d0d785d3e2b90719ec7047b9513f5acfef7"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M111 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m111_w384_signed24_accumulator_frontend \
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
grep -qx 'PASS M111 W384 signed24 accumulator VCS windows=2 updates=1056 vector_lane_checks=589824 commits=6144 lazy_valid_clears=2 positive_memory_writes=1056 ii1_pairs=1054 read_write_overlap=1054 commit_stalls=734 same_address_attacks=1 overflow_attacks=1 lanes=96 vector_bits=2304 accumulator_bytes=884736 valid_bits=3072 macro_ports=8x1R1W behavioral_macro=true overflow_guard=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
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
    echo "status=PASS_M111_W384_SIGNED24_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "window_rows=384"
    echo "banks=8"
    echo "lanes=96"
    echo "signed_bits_per_lane=24"
    echo "vector_bits=2304"
    echo "logical_accumulator_bytes=884736"
    echo "lazy_valid_bits=3072"
    echo "update_initiation_interval=1_nonconflicting_addresses"
    echo "behavioral_sync_1r1w_macro=true"
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
sha256sum "dc_handoff/scripts/run_vcs_m111_w384_signed24_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M111 W384 signed24 accumulator VCS sealed at $task_run"
