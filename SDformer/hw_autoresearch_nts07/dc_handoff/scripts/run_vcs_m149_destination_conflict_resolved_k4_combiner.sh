#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m149_destination_conflict_resolved_k4_combiner_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M149 sealed VCS run: $task_run" >&2
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
declare -A task_expected=(
    ["rtl_m149/m149_destination_conflict_resolved_k4_combiner.sv"]="8359cc2679eab7bd071bd44129d7e77f0bef41458ce0ee6e60a85f8962f897d6"
    ["verif_m149/m149_destination_conflict_resolved_k4_combiner_assertions.sv"]="df0634f2f4159d62cfd3dd66a961498cbe27cce9191d5c9208be0adde8fd6bdd"
    ["tb_m149/tb_m149_destination_conflict_resolved_k4_combiner.sv"]="3744b401898667bd7d303c6891b91b39da2434f7ed7b0d26fbcee41dfe3507ac"
    ["dc_handoff/filelists/date_m149_destination_conflict_resolved_k4_combiner_directed_vcs.f"]="d352119793b627ba7a99c4340b573bd95978908040376d2666d57658a37f732e"
    ["contracts/m149_destination_conflict_resolved_k4_combiner_vcs_contract_r1_20260824.json"]="c98f770dcc5671c45542f3585e3c82b1a1eb89a5989cf930e2c4c7db96c8d463"
    ["contracts/m147_independent_review_correction_overlay_r1_20260824.json"]="8a6ba2e9dce906378708a9ecc1cbc71b86153d7d011e1cb9cf8ff5718fa4c9af"
    ["dc_handoff/runs/m148_destination_tagged_mosaic_k4_packer_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="157de9fbd87fb4257b3ab87271585c717fa51fd41276e4a8f7e3a799f3ea92e9"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M149 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m149_destination_conflict_resolved_k4_combiner_directed_vcs.f \
    -top tb_m149_destination_conflict_resolved_k4_combiner \
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
task_pass='PASS M149 destination-conflict-resolved K4 combiner VCS descriptors=72 results=72 input_tuples=246 output_groups=156 combined_tuples=90 result_stalls=22 ii1_pairs=60 protocol_attacks=2 lanes=96 signed_input_bits=8 negate_bits=9 pair_sum_bits=10 result_bits=11 numeric_range=-512_to_512 stable_first_occurrence=true all_vectors_preavailable_assumption=true weight_storage=false sram_macro=false accumulator_commit=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in cp_all_same_destination cp_two_plus_two \
        cp_two_plus_one_plus_one cp_all_distinct cp_result_stall \
        cp_back_to_back_accept; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"
done

{
    echo "status=PASS_M149_DESTINATION_CONFLICT_RESOLVED_K4_COMBINER_VCS_SVA"
    echo "exact_sha=true"
    echo "descriptors=72"
    echo "results=72"
    echo "input_tuples=246"
    echo "output_destination_groups=156"
    echo "combined_tuples=90"
    echo "result_stalls=22"
    echo "consecutive_ii1_pairs=60"
    echo "protocol_attacks=2"
    echo "lanes=96"
    echo "signed_input_bits=8"
    echo "signed_negated_operand_bits=9"
    echo "signed_pair_sum_bits=10"
    echo "signed_result_bits=11"
    echo "numeric_range=-512_to_512"
    echo "stable_first_occurrence=true"
    echo "all_four_contribution_vectors_preavailable_assumption=true"
    echo "weight_or_pwp_storage=false"
    echo "sram_ports=false"
    echo "accumulator_commit=false"
    echo "m148_integrated_rtl=false"
    echo "m147_cycle_ratio_admitted=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m149_destination_conflict_resolved_k4_combiner.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M149 destination-conflict-resolved K4 combiner VCS sealed at $task_run"
