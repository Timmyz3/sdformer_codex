#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m185_fc2_k8_fixed_bank_accumulator_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M185 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m185/m185_fc2_k8_fixed_bank_accumulator.sv"]="60c836e6d1cef03279dd3fa4b68e9d18926ae86e06ca43cbeb1a9eae0335e00e"
    ["verif_m185/m185_fc2_k8_fixed_bank_accumulator_assertions.sv"]="2bc174ff8e62d703f01a99f1157e7bf5c1694e3be14c7f04948cf67ccc81e70f"
    ["tb_m185/tb_m185_fc2_k8_fixed_bank_accumulator.sv"]="037f432884c47367079fe4703d996472b671b7a31ba097bb7567bd93adb5a634"
    ["dc_handoff/filelists/date_m185_fc2_k8_fixed_bank_accumulator_directed_vcs.f"]="4c0f7639965a955e83e3998998e456eeb3d432ada3525305c561c056de43afbc"
    ["contracts/m185_fc2_k8_fixed_bank_accumulator_vcs_contract_r1_20260825.json"]="e1557381a80b3f700487e4031e8572e8bcbaab84f6b0de4e103e3be73aa7dbf7"
    ["dc_handoff/runs/m183_fc2_k8_unique_bank_accumulator_logic_only_dc_3p000ns_r1_sealed_20260825/RUN_COMPLETE.txt"]="7c5b61e56619c40f70727786599953403ce868c7d0601583348a7f3cd1b31190"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="178ff9b41b53779ebc66891defde36e19dae59677929ac57ceb2acad45bba139"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m185_fc2_k8_fixed_bank_accumulator_directed_vcs.f \
    -top tb_m185_fc2_k8_fixed_bank_accumulator \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=185025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M185 FC2 K8 fixed-bank accumulator VCS issues=481 results=481 one_source=60 two_source=60 three_source=60 four_source=60 five_source=60 six_source=60 seven_source=60 eight_source=61 accepted_weight_terms=2168 output_lanes=96 accumulator_bits=24 weight_bits=8 fixed_weight_banks=8 max_sources_per_issue=8 consecutive_issue_ii1_hits=159 same_cycle_result_replace=478 output_stall_cycles=101 overflow_attacks=1 empty_mask_attacks=1 arbitrary_nonprefix_masks=true bank_id_payload=false pairwise_bank_comparators=0 prefix_packing=false multipliers=0 weight_payload_bits_per_full_issue=6144 m182_bounded_exact_payload_k1_over_k8=4.344533568 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_one_source cp_two_source cp_three_source cp_four_source \
        cp_five_source cp_six_source cp_seven_source cp_full_eight_source \
        cp_nonprefix_sparse_mask cp_same_cycle_result_replace \
        cp_stall_then_accept cp_overflow_preserves_pending_result \
        cp_protocol_fault_preserves_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M185_FC2_K8_FIXED_BANK_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=185025"
    echo "accepted_issues=481"
    echo "accepted_results=481"
    echo "source_count_histogram_1_to_8=60,60,60,60,60,60,60,61"
    echo "accepted_weight_terms=2168"
    echo "output_lanes=96"
    echo "accumulator_bits_signed=24"
    echo "weight_bits_signed=8"
    echo "fixed_weight_banks=8"
    echo "maximum_sources_per_issue=8"
    echo "consecutive_issue_ii1_hits=159"
    echo "same_cycle_result_replace=478"
    echo "output_stall_cycles=101"
    echo "overflow_attacks=1"
    echo "empty_mask_attacks=1"
    echo "arbitrary_nonprefix_masks=true"
    echo "bank_id_payload_bits=0"
    echo "pairwise_bank_comparators=0"
    echo "prefix_packing=false"
    echo "multipliers_in_source=0"
    echo "weight_payload_bits_per_full_issue=6144"
    echo "sva_coverpoints_nonzero=13"
    echo "sn2_threshold_frozen_one_required=true"
    echo "external_accumulator_context=true"
    echo "weight_sram_response=false"
    echo "complete_fc2=false"
    echo "bn2=false"
    echo "residual=false"
    echo "paft_valid825=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m185_fc2_k8_fixed_bank_accumulator.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M185 FC2 K8 fixed-bank accumulator VCS sealed at $task_run"
