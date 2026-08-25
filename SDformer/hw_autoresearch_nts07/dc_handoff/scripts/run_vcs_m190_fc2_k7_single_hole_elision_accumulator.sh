#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m190_fc2_k7_single_hole_elision_accumulator_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M190 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m190/m190_fc2_k7_single_hole_elision_accumulator.sv"]="d607cb9f1a7c1bf7ed5917bcb42d87a2b13f9d414475113001c27b9e61e5bcd9"
    ["verif_m190/m190_fc2_k7_single_hole_elision_accumulator_assertions.sv"]="087315784f6c7047f726d4a3df7c91cca782f60f206610ac4f904c1151929eeb"
    ["tb_m190/tb_m190_fc2_k7_single_hole_elision_accumulator.sv"]="6fcdb8be549ccda20adf1672913141d2e2b200fb0b6a686328de8499fae4fe0e"
    ["dc_handoff/filelists/date_m190_fc2_k7_single_hole_elision_accumulator_directed_vcs.f"]="6dccb40b5e5c93e83f3bbfe667ac9120a3c4c943e4bdfa78967e035e79d090fd"
    ["contracts/m190_fc2_k7_single_hole_elision_accumulator_vcs_contract_r1_20260825.json"]="1c153e449475e30649cf5e1d6eeca82708cec4056f357d87c92e152e313c5d7f"
    ["results/m188_independent_hammer_review_r1_20260825/SHA256SUMS"]="ca0b4040c7ee06eb2bd6da6de3617ee6722fe14db34c9a525958c8b5e4e01df6"
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
    -f dc_handoff/filelists/date_m190_fc2_k7_single_hole_elision_accumulator_directed_vcs.f \
    -top tb_m190_fc2_k7_single_hole_elision_accumulator \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=190025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M190 FC2 K7 single-hole-elision accumulator VCS legal_masks_exhausted=254 numeric_lane_checks=24768 lowest_hole_positions=8 stall_hold_checks=2 same_cycle_replace_checks=1 overflow_attacks=1 full_mask_attacks=1 empty_mask_attacks=1 output_lanes=96 structural_weight_banks=8 elided_weight_lanes=7 adjacent_choices_per_lane=2 stable_prefix_compaction=false multipliers=0 sn2_threshold_frozen_one_required=true weight_sram_response=false full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_one_source cp_two_source cp_three_source cp_four_source \
        cp_five_source cp_six_source cp_seven_source cp_hole_0 cp_hole_1 \
        cp_hole_2 cp_hole_3 cp_hole_4 cp_hole_5 cp_hole_6 cp_hole_7 \
        cp_nonprefix_sparse_mask cp_same_cycle_result_replace \
        cp_stall_then_accept cp_overflow_preserves_pending_result \
        cp_empty_mask_attack cp_full_mask_attack; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M190_FC2_K7_SINGLE_HOLE_ELISION_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=190025"
    echo "legal_bank_masks_exhausted=254"
    echo "numeric_lane_checks=24768"
    echo "lowest_hole_positions_covered=8"
    echo "stall_hold_checks=2"
    echo "same_cycle_result_replace_checks=1"
    echo "overflow_attacks=1"
    echo "full_mask_attacks=1"
    echo "empty_mask_attacks=1"
    echo "output_lanes=96"
    echo "structural_weight_banks=8"
    echo "elided_weight_lanes=7"
    echo "adjacent_choices_per_lane=2"
    echo "stable_prefix_compaction=false"
    echo "multipliers_in_source=0"
    echo "sva_coverpoints_nonzero=21"
    echo "sn2_threshold_frozen_one_required=true"
    echo "weight_sram_response=false"
    echo "complete_fc2=false"
    echo "bn2=false"
    echo "residual=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m190_fc2_k7_single_hole_elision_accumulator.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M190 FC2 K7 single-hole-elision accumulator VCS sealed at $task_run"
