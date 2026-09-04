#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m204_fc2_descriptor4_paired_window_fixed_bank_frontend_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M204 sealed VCS run" >&2; exit 2; }
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m204/m204_fc2_descriptor4_paired_window_fixed_bank_frontend.sv"]="72ea6af9f9a5723b52521d889079de0ea981a75e5cd9430b19ecd2ff7632a21f"
 ["tb_m204/tb_m204_fc2_descriptor4_paired_window_fixed_bank_frontend.sv"]="d7a89dde9577fc87e36662ba2b398b084c3f390fc865852ca7bdbd88462cc4a5"
 ["verif_m204/m204_fc2_descriptor4_paired_window_fixed_bank_frontend_assertions.sv"]="f71f52f896a1e75ce5164dad47bcb91f26ef390a2d7441131528a541e862e918"
 ["dc_handoff/filelists/date_m204_fc2_descriptor4_paired_window_fixed_bank_frontend_directed_vcs.f"]="1739fc91ea75a61c80dcb42ba607d45159e24c130ac676fed1d2a2076a8d9605"
 ["contracts/m204_fc2_descriptor4_paired_window_fixed_bank_frontend_vcs_contract_r1_20260825.json"]="12563b2e7037dcc3f884ffc2265978873f6ea604760039b3ff6c92f79d69cf46"
 ["results/m203_independent_hammer_review_r1_20260825/SHA256SUMS"]="4e4f82bfe6cbc3956466b241194315d7f0c0f1aefc49e4609935157f18f60c46"
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
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" -f dc_handoff/filelists/date_m204_fc2_descriptor4_paired_window_fixed_bank_frontend_directed_vcs.f -top tb_m204_fc2_descriptor4_paired_window_fixed_bank_frontend -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log" && exit 21 || true
set +e
"$task_run/simv" +ntb_random_seed=204025 -no_save -assert report="$task_run/assert.report" -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M204 descriptor4 paired-window frontend VCS headers=7 packets=8 groups=45 done=5 paired_groups=38 odd_groups=7 group_stalls=10 protocol_attacks=3 complete_fc2=false physical_speedup=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report" && exit 32 || true
declare -A task_covers=(
 [cp_descriptor4]=6 [cp_paired_window]=51 [cp_odd_tail]=23
 [cp_group_stall]=24 [cp_group_accept]=45 [cp_upstream_done]=5
 [cp_token_done]=5 [cp_protocol_attack]=3
)
for task_cover in "${!task_covers[@]}"; do
 grep -Eq "$task_cover, .* ${task_covers[$task_cover]} match" "$task_run/assert.report" || exit 33
done
{
 echo status=PASS_M204_FC2_DESCRIPTOR4_PAIRED_WINDOW_FIXED_BANK_FRONTEND_VCS_SVA
 echo exact_sha=true; echo random_seed=204025; echo accepted_headers=7; echo descriptor_packets=8
 echo accepted_groups=45; echo completed_tokens=5; echo paired_groups=38; echo odd_or_single_groups=7
 echo group_stalls=10; echo protocol_attacks=3; echo descriptor_input_width=4; echo fixed_bank_group_width=8
 echo stage0_w1=true; echo stage1_to_stage3_pair=true; echo m202_to_m204_integrated=false
 echo frozen_payload_cycles=false; echo complete_fc2=false; echo physical_speedup=false; echo system_speedup=false; echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum dc_handoff/scripts/run_vcs_m204_fc2_descriptor4_paired_window_fixed_bank_frontend.sh > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M204 paired-window frontend VCS sealed at $task_run"
