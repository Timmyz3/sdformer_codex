#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m194_fc2_same_token_pair_bank_head_selector_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M194 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m194/m194_fc2_same_token_pair_bank_head_selector.sv"]="014a2187e5b8ae6a1402f19f1b58abb9ef66ea95183983d2fd3de6a23ac5ae91"
    ["verif_m194/m194_fc2_same_token_pair_bank_head_selector_assertions.sv"]="36f1e6b54ce3c88a2cadefb821e806382060c832b3271c42e48a0bffef112f3f"
    ["tb_m194/tb_m194_fc2_same_token_pair_bank_head_selector.sv"]="a7d37f3322c6cfa7f051da79d0fb87978b7d99503081b2e3363ff16602852e3a"
    ["dc_handoff/filelists/date_m194_fc2_same_token_pair_bank_head_selector_directed_vcs.f"]="7f3db9bce8ba10d578b044890d72a8cb9aca95515be1eb1fc4ad0c710421ac02"
    ["contracts/m194_fc2_same_token_pair_bank_head_selector_vcs_contract_r1_20260825.json"]="ca52c3a2c3d91c134768ff604cf1f27335d413f14a10a17ff2592d2502d18941"
    ["results/m195_h67_fc2_token_flush_pair_fusion_dse_r1_20260825/manifest.sha256"]="be3b8e42f7ca64c50f5742bc1a7647ecc2ec77252eb119a411b171cbfe46541e"
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
    -f dc_handoff/filelists/date_m194_fc2_same_token_pair_bank_head_selector_directed_vcs.f \
    -top tb_m194_fc2_same_token_pair_bank_head_selector \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=194025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M194 FC2 same-token pair bank-head selector VCS legal_pairs=5004 bank_selection_checks=40032 stalls=3 same_cycle_replace=45 cross_token_attacks=1 empty_pair_attacks=1 invalid_window_attacks=1 bad_channel_attacks=1 physical_banks=8 windows=2 extra_acc24_contexts=0 queue_storage=false sram_response=false complete_fc2=false physical_speedup=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_window0_only cp_window1_only cp_both_windows \
        cp_bank_fallthrough cp_all_banks cp_partial_banks cp_pair_last \
        cp_pair_not_last cp_stall_then_accept cp_same_cycle_replace \
        cp_cross_token_attack cp_bad_channel_attack; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M194_FC2_SAME_TOKEN_PAIR_BANK_HEAD_SELECTOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=194025"
    echo "legal_pairs=5004"
    echo "bank_selection_checks=40032"
    echo "stall_hold_checks=3"
    echo "same_cycle_replace_checks=45"
    echo "protocol_attacks=4"
    echo "physical_weight_banks=8"
    echo "resident_windows=2"
    echo "extra_acc24_contexts=0"
    echo "m195_token_flush_replay_opportunity=1.1089684997184623"
    echo "queue_storage=false"
    echo "head_advance=false"
    echo "weight_sram_response=false"
    echo "complete_fc2=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m194_fc2_same_token_pair_bank_head_selector.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M194 FC2 same-token pair bank-head selector VCS sealed at $task_run"
