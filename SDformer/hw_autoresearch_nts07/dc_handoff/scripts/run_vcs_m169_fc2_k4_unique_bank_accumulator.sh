#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m169_fc2_k4_unique_bank_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M169 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m169/m169_fc2_k4_unique_bank_accumulator.sv"]="10e766f96a2c635e3c64e036ee3d43c61f4471aefc3ef444d39f1f780597293f"
    ["verif_m169/m169_fc2_k4_unique_bank_accumulator_assertions.sv"]="e8e5a4e68e714b4c12e59b934d5b6af4ee1cc94e185e7b170abac45779af2e03"
    ["tb_m169/tb_m169_fc2_k4_unique_bank_accumulator.sv"]="31fdc8bd4445c8f56b3c9ef03951ae524a8bd54ba6db477a153e9ae02958d3b3"
    ["dc_handoff/filelists/date_m169_fc2_k4_unique_bank_accumulator_directed_vcs.f"]="09ab698c06d0cd444b330068a678ef6fd0dca8d15d562159fc9e1d2639abafc8"
    ["contracts/m169_fc2_k4_unique_bank_accumulator_vcs_contract_r1_20260824.json"]="417ff6db8f0f00c99b443c6821f361c4b21bb1cf445cf18c52aaff832c21ca3b"
    ["contracts/m168_h67_fc2_kbank_multisource_dse_contract_r1_20260824.json"]="93abf9e5ba4d11bd35821e01a62719600847b96c1362fde95bb0a9c1e26d3a3d"
    ["results/m168_h67_fc2_kbank_multisource_dse_r1_20260824/m168_h67_fc2_kbank_multisource_dse.json"]="d203ca6bb5a59e23c8b39cd8dff116d2134efb2280ba7889781021df1f96b137"
    ["results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/m160_h67_ffn_bn_atlif_fusion.json"]="7581ccfdfc6bffc198b4e4dabfad04269a0fc58031d743704a487c21e8aeb96d"
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
    -f dc_handoff/filelists/date_m169_fc2_k4_unique_bank_accumulator_directed_vcs.f \
    -top tb_m169_fc2_k4_unique_bank_accumulator \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=1 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M169 FC2 K4 unique-bank accumulator VCS issues=361 results=361 one_source=90 two_source=90 three_source=90 four_source=91 accepted_weight_terms=904 output_lanes=96 accumulator_bits=24 weight_bits=8 unique_weight_banks=8 max_sources_per_issue=4 consecutive_issue_ii1_hits=89 same_cycle_result_replace=[1-9][0-9]* output_stall_cycles=[1-9][0-9]* overflow_attacks=1 duplicate_bank_attacks=1 multipliers=0 weight_payload_bits_per_full_issue=3072 m168_exact_payload_k1_over_k4_boundary=3.8756597004323474 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_one_source cp_full_four_source \
        cp_same_cycle_result_replace cp_stall_then_accept \
        cp_overflow_preserves_pending_result \
        cp_protocol_fault_preserves_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_replace="$(sed -n 's/.* same_cycle_result_replace=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_stalls="$(sed -n 's/.* output_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M169_FC2_K4_UNIQUE_BANK_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "accepted_issues=361"
    echo "accepted_results=361"
    echo "one_source_issues=90"
    echo "two_source_issues=90"
    echo "three_source_issues=90"
    echo "four_source_issues=91"
    echo "accepted_weight_terms=904"
    echo "output_lanes=96"
    echo "accumulator_bits_signed=24"
    echo "weight_bits_signed=8"
    echo "weight_banks=8"
    echo "maximum_sources_per_issue=4"
    echo "multipliers_in_source=0"
    echo "weight_payload_bits_per_full_issue=3072"
    echo "consecutive_issue_ii1_hits=89"
    echo "same_cycle_result_replace=$task_replace"
    echo "output_stall_cycles=$task_stalls"
    echo "overflow_attacks=1"
    echo "duplicate_bank_attacks=1"
    echo "m168_exact_payload_k1_over_k4_boundary=3.8756597004323474"
    echo "sn2_threshold_frozen_one_required=true"
    echo "external_accumulator_context=true"
    echo "event_compactor=false"
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
sha256sum "dc_handoff/scripts/run_vcs_m169_fc2_k4_unique_bank_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M169 FC2 K4 unique-bank accumulator VCS sealed at $task_run"
