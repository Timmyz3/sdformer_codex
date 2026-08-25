#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m186_fc2_k8_fixed_bank_issue_island_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M186 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m184/m184_fc2_dual_window_k8_fixed_bank_frontend.sv"]="c6212049305faf42cda13f7f3408d5fa478c79a7c76c142501ec01d9f1e01cd6"
    ["rtl_m185/m185_fc2_k8_fixed_bank_accumulator.sv"]="60c836e6d1cef03279dd3fa4b68e9d18926ae86e06ca43cbeb1a9eae0335e00e"
    ["rtl_m186/m186_fc2_k8_fixed_bank_issue_island.sv"]="8925b78a93aaae7813363cd61d838f7cbf2ca74b2451be39df5facb6a4e5f3cf"
    ["verif_m186/m186_fc2_k8_fixed_bank_issue_island_assertions.sv"]="9b0014ffa32d2da24002120639bcb0d1102cbd5090fdb772bb3c589d141d92f3"
    ["tb_m186/tb_m186_fc2_k8_fixed_bank_issue_island.sv"]="a9c21585b759a574307f716ac26d83ea95da7a01c283969aab920c21fe206862"
    ["dc_handoff/filelists/date_m186_fc2_k8_fixed_bank_issue_island_directed_vcs.f"]="979e0a18ba4b80843d142072cb29bf4d54e83790c090a993967db6afc9894a70"
    ["contracts/m186_fc2_k8_fixed_bank_issue_island_vcs_contract_r1_20260825.json"]="a8768d7f1ad2435e7785902085127bd5fc06efe0fd53bc2174a78ba7e90f0f11"
    ["dc_handoff/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="178ff9b41b53779ebc66891defde36e19dae59677929ac57ceb2acad45bba139"
    ["dc_handoff/runs/m185_fc2_k8_fixed_bank_accumulator_vcs_r1_sealed_20260825/RUN_COMPLETE.txt"]="1c5c22d3b42d5b705f99e66df0b22d0b3b3ca31e68d5d600d66a6c21e7a1e04d"
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
    -f dc_handoff/filelists/date_m186_fc2_k8_fixed_bank_issue_island_directed_vcs.f \
    -top tb_m186_fc2_k8_fixed_bank_issue_island \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=186025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M186 FC2 K8 fixed-bank issue island VCS headers=5 descriptors=34 tokens=5 requests=190 responses=190 results=190 bitmap_events=219 replayed_source_terms_expected=997 replayed_source_terms_observed=997 request_stall_cycles=46 response_stall_cycles=13 result_stall_cycles=66 done_wait_cycles=12 same_cycle_response_request_replace=147 nonprefix_requests=22 outstanding_slots=1 in_order_response=true direct_fixed_bank_mask=true bank_id_payload=false prefix_packing=false weight_response_payload_bits=6144 overflow_attacks=1 unsolicited_response_attacks=1 accumulator_context_external=true descriptor_producer=false weight_sram_macro=false bn2=false residual=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_nonprefix_request cp_same_cycle_response_request_replace \
        cp_zero_token_done cp_nonzero_token_done cp_protocol_fault \
        cp_numeric_overflow cp_request_stall_then_accept \
        cp_response_stall_then_accept cp_result_stall_then_accept \
        cp_done_waits_for_arithmetic; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M186_FC2_K8_FIXED_BANK_ISSUE_ISLAND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=186025"
    echo "headers=5"
    echo "descriptors=34"
    echo "tokens=5"
    echo "accepted_requests=190"
    echo "accepted_responses=190"
    echo "accepted_results=190"
    echo "bitmap_events=219"
    echo "expected_replayed_source_terms=997"
    echo "observed_replayed_source_terms=997"
    echo "request_stall_cycles=46"
    echo "response_stall_cycles=13"
    echo "result_stall_cycles=66"
    echo "done_wait_cycles=12"
    echo "same_cycle_response_request_replacements=147"
    echo "nonprefix_requests=22"
    echo "outstanding_weight_requests=1"
    echo "in_order_response=true"
    echo "direct_fixed_bank_mask=true"
    echo "bank_id_payload_bits=0"
    echo "prefix_packing=false"
    echo "weight_response_payload_bits=6144"
    echo "numeric_overflow_attacks=1"
    echo "unsolicited_response_attacks=1"
    echo "sva_coverpoints_nonzero=10"
    echo "accumulator_context_external=true"
    echo "descriptor_producer=false"
    echo "weight_sram_macro=false"
    echo "bn2=false"
    echo "residual=false"
    echo "complete_fc2=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m186_fc2_k8_fixed_bank_issue_island.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M186 FC2 K8 fixed-bank issue island VCS sealed at $task_run"
