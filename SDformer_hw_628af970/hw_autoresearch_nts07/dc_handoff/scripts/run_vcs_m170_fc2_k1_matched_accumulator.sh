#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m170_fc2_k1_matched_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M170 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m170/m170_fc2_k1_single_bank_accumulator.sv"]="861928bd168b19f6a6c939bd6c9a862df22757d8f871ad84fd5e3dfc43f18295"
    ["verif_m170/m170_fc2_k1_single_bank_accumulator_assertions.sv"]="6b072a954429cd6dbc60871547d8ecf4f78d75a13dcce3eeb15d4c2d8c9b9893"
    ["tb_m170/tb_m170_fc2_k1_single_bank_accumulator.sv"]="270373640cf7141dcfdc707cda7ae88d07ebacdefca6e95e90b713a2d20226ca"
    ["dc_handoff/filelists/date_m170_fc2_k1_single_bank_accumulator_directed_vcs.f"]="adada991327dbacec7b11ed24c59957dce5c35a3d547b136736c319dd6b1bd27"
    ["contracts/m170_fc2_k1_matched_accumulator_vcs_contract_r1_20260824.json"]="e6570c4a25cb407daf72388f63cf7ebc7c8e4f2fddbe19b7a794aed4956721dd"
    ["contracts/m169_fc2_k4_unique_bank_accumulator_vcs_contract_r1_20260824.json"]="417ff6db8f0f00c99b443c6821f361c4b21bb1cf445cf18c52aaff832c21ca3b"
    ["dc_handoff/runs/m169_fc2_k4_unique_bank_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="d384d07a72e47ab1a9daff555dbea93c45a6b6a98a891eab4feb0535e29319f9"
    ["results/m168_h67_fc2_kbank_multisource_dse_r1_20260824/m168_h67_fc2_kbank_multisource_dse.json"]="d203ca6bb5a59e23c8b39cd8dff116d2134efb2280ba7889781021df1f96b137"
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
    -f dc_handoff/filelists/date_m170_fc2_k1_single_bank_accumulator_directed_vcs.f \
    -top tb_m170_fc2_k1_single_bank_accumulator \
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
task_pass_regex='^PASS M170 FC2 K1 single-bank accumulator VCS issues=361 results=361 accepted_weight_terms=361 output_lanes=96 accumulator_bits=24 weight_bits=8 weight_banks=8 max_sources_per_issue=1 consecutive_issue_ii1_hits=89 same_cycle_result_replace=[1-9][0-9]* output_stall_cycles=[1-9][0-9]* overflow_attacks=1 empty_issue_attacks=1 multipliers=0 weight_payload_bits_per_full_issue=768 matched_m169_interface_state=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_accept cp_same_cycle_result_replace \
        cp_stall_then_accept cp_overflow_preserves_pending_result \
        cp_protocol_fault_preserves_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_replace="$(sed -n 's/.* same_cycle_result_replace=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_stalls="$(sed -n 's/.* output_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M170_FC2_K1_MATCHED_ACCUMULATOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "accepted_issues=361"
    echo "accepted_results=361"
    echo "accepted_weight_terms=361"
    echo "output_lanes=96"
    echo "accumulator_bits_signed=24"
    echo "weight_bits_signed=8"
    echo "weight_banks=8"
    echo "maximum_sources_per_issue=1"
    echo "multipliers_in_source=0"
    echo "weight_payload_bits_per_full_issue=768"
    echo "consecutive_issue_ii1_hits=89"
    echo "same_cycle_result_replace=$task_replace"
    echo "output_stall_cycles=$task_stalls"
    echo "overflow_attacks=1"
    echo "empty_issue_attacks=1"
    echo "matched_m169_interface_state=true"
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
sha256sum "dc_handoff/scripts/run_vcs_m170_fc2_k1_matched_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M170 FC2 K1 matched accumulator VCS sealed at $task_run"
