#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m118_w384_signed19_lane_accumulator_vcs_r2_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M118 sealed VCS run: $task_run" >&2
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
task_core="rtl_m118/m118_w384_signed19_accumulator_frontend.sv"
task_adapter="rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv"
task_sva="verif_m118/m118_w384_signed19_lane_sliced_accumulator_assertions.sv"
task_tb="tb_m118/tb_m118_w384_signed19_lane_sliced_accumulator.sv"
task_files="dc_handoff/filelists/date_m118_w384_signed19_lane_sliced_accumulator_directed_vcs.f"
task_contract="contracts/m118_w384_signed19_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_m115_analyzer="system_simulator/scripts/analyze_m115r2_pwp_prefix_coefficient_width.py"
task_m115_result="results/m115r2_pwp_prefix_coefficient_width_r1_20260824/m115r2_pwp_prefix_coefficient_width.json"
task_m115_contract="contracts/m115r2_pwp_prefix_coefficient_width_contract_r1_20260824.json"
task_m115_manifest="results/m115r2_pwp_prefix_coefficient_width_r1_20260824/SHA256SUMS.complete_r1.txt"

declare -A task_expected=(
    ["$task_core"]="0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482"
    ["$task_adapter"]="cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af"
    ["$task_sva"]="ccea5ca611265c4970ceda9dee7d714ba154730102940931c3549473d186b07c"
    ["$task_tb"]="3f084d0c3a406dbdb36d82f0230c3e6f4e2e194fe6d43224f982288d6ab3d66c"
    ["$task_files"]="a5042955a8dc9eae93b61aa1ba14bb2a93a79b6791504dc3e04bbc53bf811af0"
    ["$task_contract"]="c79f55a15e03bbf26c22e9da2f0eb35d53b1a9795ab02b24a6b3c951c729903e"
    ["$task_m115_analyzer"]="2f3512f2c664daea6430c1360838c7496228b49ae2dd5a648db9af361fbf0f31"
    ["$task_m115_result"]="b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708"
    ["$task_m115_contract"]="9edd6aac10186e24f21fffa5ce1b5a28da292258ad30df1d6934a7b1d1927eec"
    ["$task_m115_manifest"]="6b9af5e9e7de61edc770e1d4d738d6c0b0070e7947f6aec12633da7181f96326"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M118 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m118_w384_signed19_lane_sliced_accumulator \
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
grep -qx 'PASS M118 W384 lane-sliced accumulator VCS windows=2 updates=1056 vector_lane_checks=589824 commits=6144 lazy_valid_clears=2 positive_memory_writes=1056 ii1_pairs=1054 read_write_overlap=1054 commit_stalls=734 same_address_attacks=1 overflow_attacks=2 lanes=96 vector_bits=1824 accumulator_bytes=700416 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=19 behavioral_macro=true overflow_guard=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_update_ii1, .* 1058 match' \
        'cp_read_write_overlap, .* 1058 match' \
        'cp_commit_stall, .* 699 match' \
        'cp_full_commit, .* 2 match' \
        'cp_fault, .* 3 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M118_W384_SIGNED19_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "checkpoint_transient_bound=218338"
    echo "window_rows=384"
    echo "lane_macro_count=96"
    echo "lane_macro_depth=3072"
    echo "lane_macro_width_bits=19"
    echo "lanes=96"
    echo "signed_bits_per_lane=19"
    echo "vector_bits=1824"
    echo "logical_accumulator_bytes=700416"
    echo "logical_saving_bytes_vs_signed24=184320"
    echo "lazy_valid_bits=3072"
    echo "update_initiation_interval=1_nonconflicting_addresses"
    echo "behavioral_sync_lane_sliced_1r1w_macro=true"
    echo "foundry_sram_macro=false"
    echo "full_lane_numeric_directed_miter=true"
    echo "integrated_accepted_transaction_exact_once_miter=false"
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
sha256sum "dc_handoff/scripts/run_vcs_m118_w384_signed19_lane_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M118 W384 signed19 lane accumulator VCS sealed at $task_run"
