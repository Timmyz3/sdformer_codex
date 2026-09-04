#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m183_fc2_k8_unique_bank_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M183 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m183/m183_fc2_k8_unique_bank_accumulator.sv"]="9b5a57d79806da38590b0e316300446b90990f1828374bf5094bbe3dded81bec"
    ["verif_m183/m183_fc2_k8_unique_bank_accumulator_assertions.sv"]="332eead4a3b46c7ebcf867fdf796e3f9ddbb896ce75e906542f5698cd3a79ab7"
    ["tb_m183/tb_m183_fc2_k8_unique_bank_accumulator.sv"]="b4191d8fef41fdfaca6e4b01ab440cc5640038d3143e9a5569a5d1ee12cb9274"
    ["dc_handoff/filelists/date_m183_fc2_k8_unique_bank_accumulator_directed_vcs.f"]="2d0a417b347fe484641a749078ef4b9d9d1e1c5b983e157757fe3152bfc67733"
    ["contracts/m183_fc2_k8_unique_bank_accumulator_vcs_contract_r1_20260824.json"]="60c766ff11a0f88df294595467fa82f1290ffd5b562c7ec8ae92bbd783b5190c"
    ["contracts/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse_contract_r1_20260824.json"]="4dea36a1ebcb544ea597a84c34fdf7759962adaaf6d6ca2f2ae3a7f511be642a"
    ["results/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse_r1_20260824/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse.json"]="1ae1feb8ba274f2df7a0b5749596fae786a2d56b91650d8e15383a7bdb428b5b"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m183_fc2_k8_unique_bank_accumulator_directed_vcs.f \
    -top tb_m183_fc2_k8_unique_bank_accumulator \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=183024 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M183 FC2 K8 unique-bank accumulator VCS issues=481 results=481 one_source=60 two_source=60 three_source=60 four_source=60 five_source=60 six_source=60 seven_source=60 eight_source=61 accepted_weight_terms=2168 output_lanes=96 accumulator_bits=24 weight_bits=8 unique_weight_banks=8 max_sources_per_issue=8 consecutive_issue_ii1_hits=159 same_cycle_result_replace=[1-9][0-9]* output_stall_cycles=[1-9][0-9]* overflow_attacks=1 combined_nonprefix_duplicate_bank_attacks=1 multipliers=0 weight_payload_bits_per_full_issue=6144 m182_bounded_exact_payload_k1_over_k8=4.344533568 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_one_source cp_two_source cp_three_source \
        cp_four_source cp_five_source cp_six_source cp_seven_source \
        cp_full_eight_source cp_same_cycle_result_replace \
        cp_stall_then_accept cp_overflow_preserves_pending_result \
        cp_protocol_fault_preserves_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_replace="$(sed -n 's/.* same_cycle_result_replace=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_stalls="$(sed -n 's/.* output_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M183_FC2_K8_UNIQUE_BANK_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=183024"
    echo "accepted_issues=481"
    echo "accepted_results=481"
    echo "one_source_issues=60"
    echo "two_source_issues=60"
    echo "three_source_issues=60"
    echo "four_source_issues=60"
    echo "five_source_issues=60"
    echo "six_source_issues=60"
    echo "seven_source_issues=60"
    echo "eight_source_issues=61"
    echo "accepted_weight_terms=2168"
    echo "output_lanes=96"
    echo "accumulator_bits_signed=24"
    echo "weight_bits_signed=8"
    echo "weight_banks=8"
    echo "maximum_sources_per_issue=8"
    echo "multipliers_in_source=0"
    echo "weight_payload_bits_per_full_issue=6144"
    echo "consecutive_issue_ii1_hits=159"
    echo "same_cycle_result_replace=$task_replace"
    echo "output_stall_cycles=$task_stalls"
    echo "overflow_attacks=1"
    echo "combined_nonprefix_duplicate_bank_attacks=1"
    echo "m182_bounded_exact_payload_k1_over_k8=4.344533568"
    echo "sn2_threshold_frozen_one_required=true"
    echo "external_accumulator_context=true"
    echo "event_scheduler=false"
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
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m183_fc2_k8_unique_bank_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M183 FC2 K8 unique-bank accumulator VCS sealed at $task_run"
