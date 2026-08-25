#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M116 sealed VCS run: $task_run" >&2
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
task_core="rtl_m116/m116_w384_signed20_accumulator_frontend.sv"
task_adapter="rtl_m116/m116_w384_signed20_lane_sliced_accumulator_adapter.sv"
task_sva="verif_m116/m116_w384_signed20_lane_sliced_accumulator_assertions.sv"
task_tb="tb_m116/tb_m116_w384_signed20_lane_sliced_accumulator.sv"
task_files="dc_handoff/filelists/date_m116_w384_signed20_lane_sliced_accumulator_directed_vcs.f"
task_contract="contracts/m116_w384_signed20_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_m115_analyzer="system_simulator/scripts/analyze_m115_pwp_transient_accumulator_width.py"
task_m115_result="results/m115_pwp_transient_accumulator_width_r1_20260824/m115_pwp_transient_accumulator_width.json"
task_m115_contract="contracts/m115_pwp_transient_accumulator_width_contract_r1_20260824.json"
task_m115_manifest="results/m115_pwp_transient_accumulator_width_r1_20260824/SHA256SUMS.txt"

declare -A task_expected=(
    ["$task_core"]="dd7e52e9ab3739972ca160283406c17f5d1a2947a3dd2456608a782b640c47b0"
    ["$task_adapter"]="074735e1f583d3dbef8e6dbee28f1ffb5a82bcda7a7328c8b520c5efc3c53a16"
    ["$task_sva"]="e7e36fbc3f695a71cc7b7c6e0393146131071152f1e7a6ad5df8f4d70732eecd"
    ["$task_tb"]="845c09847df7b65db4d787fce93283cc95b161e8f931112ae12d1609e7eec6d5"
    ["$task_files"]="4ed59a697bb688f1e53b90b313eea5cf6877c46af62022cd57b2bd30ef51f208"
    ["$task_contract"]="bb245aa111d9646ff6b772c65a3362ae266d2f492691dac83d1782789912b721"
    ["$task_m115_analyzer"]="bafadcf53e5221d70ab86da0fb17dcbae8da661b0148007dbd537f4fa519aa27"
    ["$task_m115_result"]="9f62d9cb3e56c293cc117bd92c21844e8bd10515ea418a51cbfae0ebab62b94b"
    ["$task_m115_contract"]="ba730fcb6612fd8aa5c8e8c7d1aba976b759de54cbab05779ca409dadf9af9c8"
    ["$task_m115_manifest"]="bb12196b1ed7e0c10cb6b41db85271db24bfefab62bf0058b194666353afc951"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M116 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m116_w384_signed20_lane_sliced_accumulator \
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
grep -qx 'PASS M116 W384 lane-sliced accumulator VCS windows=2 updates=1056 vector_lane_checks=589824 commits=6144 lazy_valid_clears=2 positive_memory_writes=1056 ii1_pairs=1054 read_write_overlap=1054 commit_stalls=734 same_address_attacks=1 overflow_attacks=1 lanes=96 vector_bits=1920 accumulator_bytes=737280 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=20 behavioral_macro=true overflow_guard=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
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
    echo "status=PASS_M116_W384_SIGNED20_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "checkpoint_transient_bound=436676"
    echo "window_rows=384"
    echo "lane_macro_count=96"
    echo "lane_macro_depth=3072"
    echo "lane_macro_width_bits=20"
    echo "lanes=96"
    echo "signed_bits_per_lane=20"
    echo "vector_bits=1920"
    echo "logical_accumulator_bytes=737280"
    echo "logical_saving_bytes_vs_signed24=147456"
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
sha256sum "dc_handoff/scripts/run_vcs_m116_w384_signed20_lane_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M116 W384 signed20 lane accumulator VCS sealed at $task_run"
