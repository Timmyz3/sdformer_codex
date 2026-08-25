#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m189_fc2_k7_bank_compacting_accumulator_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M189 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m189/m189_fc2_k7_bank_compacting_accumulator.sv"]="a9e419e38cf4b6cc00ead4fb73b86d4a21b3edda100f3a2c78b4f7c7bbce88ce"
    ["verif_m189/m189_fc2_k7_bank_compacting_accumulator_assertions.sv"]="ad508b69189e96c2fe42d87596c90d3dbc06dd19e7a7356a15e4dd585c13c4f4"
    ["tb_m189/tb_m189_fc2_k7_bank_compacting_accumulator.sv"]="4d200f79deec01c958b0c4be3864ea38b4205774b2a607bff11a7fa150c215e4"
    ["dc_handoff/filelists/date_m189_fc2_k7_bank_compacting_accumulator_directed_vcs.f"]="ebcb4a92b90583723bc31e7a530aaa2a71b5467c78b58b77c15ec0927fb2f5eb"
    ["contracts/m189_fc2_k7_bank_compacting_accumulator_vcs_contract_r1_20260825.json"]="89724ba3f6ec053c447482dbaea8b3fbaa908a543290e88ee043b52dd5d29172"
    ["dc_handoff/runs/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="ee5d41c6e5b6701a553769d36d948ca9ada1033a1a7badee64fae62c1c2deb99"
    ["contracts/m186_m187_independent_review_admission_overlay_r1_20260825.json"]="1080d4b915659c496344c3548289b4e474aa70eabaaa4f87bccd77cef64a77d2"
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
    -f dc_handoff/filelists/date_m189_fc2_k7_bank_compacting_accumulator_directed_vcs.f \
    -top tb_m189_fc2_k7_bank_compacting_accumulator \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=189025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M189 FC2 K7 bank-compacting accumulator VCS issues=575 results=575 one_source=23 two_source=73 three_source=132 four_source=155 five_source=118 six_source=57 seven_source=17 accepted_weight_terms=2236 legal_masks_exhausted=254 output_lanes=96 accumulator_bits=24 weight_bits=8 structural_weight_banks=8 compacted_weight_lanes=7 max_sources_per_issue=7 consecutive_issue_ii1_hits=253 same_cycle_result_replace=572 output_stall_cycles=124 overflow_attacks=1 empty_mask_attacks=1 full_mask_attacks=1 arbitrary_nonprefix_masks=true increasing_bank_order_compaction=true multipliers=0 structural_input_bits=6144 compacted_internal_bits=5376 nominal_lane_reduction_percent=12.5 m187_k7_over_k8_cycle_penalty_percent=0.088857646 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_one_source cp_two_source cp_three_source cp_four_source \
        cp_five_source cp_six_source cp_seven_source cp_hole_at_low_bank \
        cp_hole_at_high_bank cp_nonprefix_sparse_mask \
        cp_same_cycle_result_replace cp_stall_then_accept \
        cp_overflow_preserves_pending_result cp_empty_mask_attack \
        cp_full_mask_attack; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M189_FC2_K7_BANK_COMPACTING_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=189025"
    echo "accepted_issues=575"
    echo "accepted_results=575"
    echo "legal_bank_masks_exhausted=254"
    echo "source_count_histogram_1_to_7=23,73,132,155,118,57,17"
    echo "accepted_weight_terms=2236"
    echo "output_lanes=96"
    echo "accumulator_bits_signed=24"
    echo "weight_bits_signed=8"
    echo "structural_weight_banks=8"
    echo "compacted_weight_lanes=7"
    echo "maximum_sources_per_issue=7"
    echo "consecutive_issue_ii1_hits=253"
    echo "same_cycle_result_replace=572"
    echo "output_stall_cycles=124"
    echo "overflow_attacks=1"
    echo "empty_mask_attacks=1"
    echo "full_mask_attacks=1"
    echo "arbitrary_nonprefix_masks=true"
    echo "increasing_bank_order_compaction=true"
    echo "multipliers_in_source=0"
    echo "structural_input_bits=6144"
    echo "compacted_internal_bits=5376"
    echo "nominal_lane_reduction_percent=12.5"
    echo "m187_k7_over_k8_cycle_penalty_percent=0.088857646"
    echo "sva_coverpoints_nonzero=15"
    echo "sn2_threshold_frozen_one_required=true"
    echo "external_accumulator_context=true"
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
sha256sum "dc_handoff/scripts/run_vcs_m189_fc2_k7_bank_compacting_accumulator.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M189 FC2 K7 bank-compacting accumulator VCS sealed at $task_run"
