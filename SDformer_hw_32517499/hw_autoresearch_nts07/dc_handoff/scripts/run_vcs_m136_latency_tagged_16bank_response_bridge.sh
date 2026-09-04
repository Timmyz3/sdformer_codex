#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m136_latency_tagged_16bank_response_bridge_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M136 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m136/m136_latency_tagged_16bank_response_bridge.sv"
task_sva="verif_m136/m136_latency_tagged_16bank_response_bridge_assertions.sv"
task_tb="tb_m136/tb_m136_latency_tagged_16bank_response_bridge.sv"
task_files="dc_handoff/filelists/date_m136_latency_tagged_16bank_response_bridge_directed_vcs.f"
task_contract="contracts/m136_latency_tagged_16bank_response_bridge_vcs_contract_r1_20260824.json"
task_m134_review="reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/manifest.sha256"
task_m134_receipt="dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m135_vcs_receipt="dc_handoff/runs/m135r2_conflict_free_16bank_pwp_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m135_dc_receipt="dc_handoff/runs/m135r3_flattened_conflict_free_16bank_pwp_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_rtl"]="61c5008e0fe570973d94b748c7fb1f85947b289bdac96cc4294fb5d87750b870"
    ["$task_sva"]="3a46f5b106a85a5f1f6b0d2e035d80dda4079c656c174941c522688aab7157ce"
    ["$task_tb"]="63c1c1d0ac31d1647e8f6444cd23e12bc792284977d3a241bb8f0dc31c642a20"
    ["$task_files"]="f5d5b6ffdc60303c1e47c75090c41c2ef331785bd8e2eecc98ed28eeb85bc329"
    ["$task_contract"]="8205b01b47f60bdbbf931b76bab9cc2890bff49d94ff80aeb19880071582ad13"
    ["$task_m134_review"]="bf6ce236e2ad96c3d27621cd2add52c9e682a6ab0933074c7b96b2188f1ebec2"
    ["$task_m134_receipt"]="047dec485d9c5e748d2a98cb10cc65a946d6c39b4b7085e9363a78cb6958f17d"
    ["$task_m135_vcs_receipt"]="2048bdd5a1e8756a3760af2af32208fdbe7374a7cc7353a94c2de878b6c58510"
    ["$task_m135_dc_receipt"]="0856f5dabfebfedbc821b3f50c4c12a8898f52cadb3df0a6f1b71b3ed9482653"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M136 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m136_latency_tagged_16bank_response_bridge \
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
task_pass='PASS M136 one-cycle tagged 16-bank response bridge VCS requests=128 outputs=128 words=2048 ii1=120 stalls=12 fifo_full=12 row_crossings=120 wrong_token=1 missing=1 unsolicited=1 illegal_base=1 reset_recoveries=4 fifo_depth=2 latency=1 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_contiguous_eight_requests, .* [1-9][0-9]* match' \
        'cp_two_buffered_under_stall, .* [1-9][0-9]* match' \
        'cp_cross_row_request, .* [1-9][0-9]* match' \
        'cp_wrong_token_quarantine, .* [1-9][0-9]* match' \
        'cp_unsolicited_quarantine, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M136_ONE_CYCLE_TAGGED_16BANK_RESPONSE_BRIDGE_VCS_SVA"
    echo "exact_sha=true"
    echo "accepted_requests=128"
    echo "accepted_outputs=128"
    echo "logical_word_checks=2048"
    echo "interval_one_checks=120"
    echo "stall_cycles=12"
    echo "fifo_full_cycles=12"
    echo "row_crossing_requests=120"
    echo "wrong_token_attacks=1"
    echo "missing_response_attacks=1"
    echo "unsolicited_response_attacks=1"
    echo "illegal_base_attacks=1"
    echo "reset_recoveries=4"
    echo "return_fifo_depth=2"
    echo "fixed_latency_cycles=1"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m136_latency_tagged_16bank_response_bridge.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M136 one-cycle tagged 16-bank response bridge VCS sealed at $task_run"
