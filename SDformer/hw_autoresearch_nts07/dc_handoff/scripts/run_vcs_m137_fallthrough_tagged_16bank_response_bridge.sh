#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m137_fallthrough_tagged_16bank_response_bridge_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M137 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m137/m137_fallthrough_tagged_16bank_response_bridge.sv"
task_sva="verif_m137/m137_fallthrough_tagged_16bank_response_bridge_assertions.sv"
task_tb="tb_m137/tb_m137_fallthrough_tagged_16bank_response_bridge.sv"
task_files="dc_handoff/filelists/date_m137_fallthrough_tagged_16bank_response_bridge_directed_vcs.f"
task_contract="contracts/m137_fallthrough_tagged_16bank_response_bridge_vcs_contract_r1_20260824.json"
task_m134_review="reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/manifest.sha256"
task_m134_receipt="dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m135_vcs_receipt="dc_handoff/runs/m135r2_conflict_free_16bank_pwp_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m135_dc_receipt="dc_handoff/runs/m135r3_flattened_conflict_free_16bank_pwp_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m135_review_overlay="contracts/m135r3_independent_review_and_r2_failure_identity_overlay_r1_20260824.json"
task_m136_vcs_receipt="dc_handoff/runs/m136_latency_tagged_16bank_response_bridge_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m136_dc_receipt="dc_handoff/runs/m136_latency_tagged_16bank_response_bridge_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_rtl"]="e2b0a271728dc8c0f79ba3361f76df554ad61e6d6efaf11ae09ff89be9384af2"
    ["$task_sva"]="f63332f442e3ac6c6ac6b5a258a436b9f53924bf637e06b567b14f55db2d3ec7"
    ["$task_tb"]="11478948707dbe91c40481c43f04b60015cc47e6c6f338cfbd3791faf3e4dc9a"
    ["$task_files"]="ced90f89b9c9001b9846ff63cc403752d21118dc0cc149bf93526dd56cfd6fc8"
    ["$task_contract"]="4bff3424bb8b8d921facbbe0987798447b165a89d84351181c7f325e2bcbbc8b"
    ["$task_m134_review"]="bf6ce236e2ad96c3d27621cd2add52c9e682a6ab0933074c7b96b2188f1ebec2"
    ["$task_m134_receipt"]="047dec485d9c5e748d2a98cb10cc65a946d6c39b4b7085e9363a78cb6958f17d"
    ["$task_m135_vcs_receipt"]="2048bdd5a1e8756a3760af2af32208fdbe7374a7cc7353a94c2de878b6c58510"
    ["$task_m135_dc_receipt"]="0856f5dabfebfedbc821b3f50c4c12a8898f52cadb3df0a6f1b71b3ed9482653"
    ["$task_m135_review_overlay"]="2ad920d745871b11b5b2336ec9a93231cda5a8bc2bbb41a8b61562b1754642da"
    ["$task_m136_vcs_receipt"]="c99c75dd322bc67419ffb7111ef895a3d1b0b074c23967c2535d49a4c4a8023d"
    ["$task_m136_dc_receipt"]="f7d0f35b05477e9e7f8db758392ec40fd9f1662cc362d9a25b910971a4ee9185"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M137 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m137_fallthrough_tagged_16bank_response_bridge \
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
task_pass='PASS M137 fallthrough tagged 16-bank response bridge VCS requests=128 outputs=128 words=2048 ii1=120 stalls=12 skid_buffer=12 row_crossings=120 wrong_token=1 missing=1 unsolicited=1 illegal_base=1 reset_recoveries=4 skid_depth=1 macro_latency=1 delivery_latency=1 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_contiguous_eight_requests, .* [1-9][0-9]* match' \
        'cp_fallthrough_and_next_request, .* [1-9][0-9]* match' \
        'cp_skid_capture_under_stall, .* [1-9][0-9]* match' \
        'cp_skid_release_and_request, .* [1-9][0-9]* match' \
        'cp_cross_row_request, .* [1-9][0-9]* match' \
        'cp_wrong_token_quarantine, .* [1-9][0-9]* match' \
        'cp_unsolicited_quarantine, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M137_FALLTHROUGH_TAGGED_16BANK_RESPONSE_BRIDGE_VCS_SVA"
    echo "exact_sha=true"
    echo "accepted_requests=128"
    echo "accepted_outputs=128"
    echo "logical_word_checks=2048"
    echo "interval_one_checks=120"
    echo "stall_cycles=12"
    echo "skid_buffer_cycles=12"
    echo "row_crossing_requests=120"
    echo "wrong_token_attacks=1"
    echo "missing_response_attacks=1"
    echo "unsolicited_response_attacks=1"
    echo "illegal_base_attacks=1"
    echo "reset_recoveries=4"
    echo "return_skid_depth=1"
    echo "macro_latency_cycles=1"
    echo "consumer_delivery_latency_cycles=1"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m137_fallthrough_tagged_16bank_response_bridge.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M137 fallthrough tagged 16-bank response bridge VCS sealed at $task_run"
