#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m205_m202_to_m204_cycle_cosim_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M205 sealed VCS run" >&2; exit 2; }
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m202/m202_fc2_raw4_to_descriptor4_fresh_bypass_compactor.sv"]="eb9f42ffd4286a4f5c83436acdad30568ddd6e7d90510e725d210a9a35677354"
 ["rtl_m204/m204_fc2_descriptor4_paired_window_fixed_bank_frontend.sv"]="72ea6af9f9a5723b52521d889079de0ea981a75e5cd9430b19ecd2ff7632a21f"
 ["rtl_m205/m205_fc2_raw4_to_paired_window_frontend.sv"]="17dd8458bcdd4f888e46a9425cdec4b52988c6b1931e10a639f299d162ead467"
 ["tb_m205/tb_m205_fc2_raw4_to_paired_window_frontend.sv"]="5f5455b16bc2646798bf85df302c2f1eb111220c039ba7e5b7a132b7e0e59749"
 ["verif_m205/m205_fc2_raw4_to_paired_window_frontend_assertions.sv"]="b1118bc57b0349c89b24393bfd6a494c82158f5c24cef056b0d5a2a8c707c1e7"
 ["dc_handoff/filelists/date_m205_fc2_raw4_to_paired_window_frontend_directed_vcs.f"]="c3eb97dd32293a13883a5d79535f28e3630464ea1d2dc010697338baea869e08"
 ["contracts/m205_m202_to_m204_cycle_cosim_vcs_contract_r1_20260825.json"]="71cde6c459e96c070ebd4d359de0267d7bf06a820b0a102357d7a689d4c99c29"
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
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" -f dc_handoff/filelists/date_m205_fc2_raw4_to_paired_window_frontend_directed_vcs.f -top tb_m205_fc2_raw4_to_paired_window_frontend -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log" && exit 21 || true
set +e
"$task_run/simv" +ntb_random_seed=205025 -no_save -assert report="$task_run/assert.report" -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M205 M202-to-M204 cycle co-sim VCS legal_headers=6 raw_packets=17 groups=104 done=5 descriptor4=10 paired=169 group_stalls=64 raw_backpressure=83 protocol_attacks=2 duplicate_storage=true complete_fc2=false physical_speedup=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout|M205 DEBUG' "$task_run/sim.raw.log" "$task_run/assert.report" && exit 32 || true
declare -A task_covers=(
 [cp_joint_header]=6 [cp_raw4_dense]=17 [cp_raw_backpressure]=83
 [cp_descriptor4]=10 [cp_paired_window]=169 [cp_group_stall]=64
 [cp_group_accept]=104 [cp_compact_done]=5 [cp_token_done]=5
 [cp_protocol_attack]=3
)
for task_cover in "${!task_covers[@]}"; do
 grep -Eq "$task_cover, .* ${task_covers[$task_cover]} match" "$task_run/assert.report" || exit 33
done
{
 echo status=PASS_M205_M202_TO_M204_CYCLE_COSIM_VCS_SVA
 echo exact_sha=true; echo random_seed=205025; echo accepted_headers_including_attack_setup=6; echo accepted_raw_packets=17
 echo accepted_groups=104; echo completed_legal_tokens=5; echo descriptor4_observations=10; echo paired_window_observations=169
 echo group_stalls=64; echo raw_backpressure_cycles=83; echo protocol_attacks=2; echo scoreboard_mismatches=0
 echo full_window_backpressure_is_legal=true; echo duplicate_storage=true; echo fused_storage=false
 echo frozen_payload_cycles=false; echo complete_fc2=false; echo physical_speedup=false; echo system_speedup=false; echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum dc_handoff/scripts/run_vcs_m205_m202_to_m204_cycle_cosim.sh > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M205 M202-to-M204 cycle co-sim VCS sealed at $task_run"
