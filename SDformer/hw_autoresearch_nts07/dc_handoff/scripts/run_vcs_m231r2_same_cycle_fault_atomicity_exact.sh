#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m231r2_same_cycle_fault_atomicity_directed_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" >"$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m231/m231_atlif32_to_fc2_raw4_pingpong_bridge.sv"]="7e129ab7724dcb66c10de51bbb8b444cb0d5e4f3c407b5f910afd1267ddfe89c"
 ["verif_m231/m231_atlif32_to_fc2_raw4_pingpong_bridge_assertions.sv"]="1e32d42c79f8455f5371021e856f222633475cf250a6cd81cb5570d95764da2d"
 ["tb_m231/tb_m231_atlif32_to_fc2_raw4_pingpong_bridge.sv"]="5d8ba794583077b13971ac73061ba5ac056b6bb5b57e0b7668da3dad0f6d6c8f"
 ["dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_directed_vcs.f"]="a7e1fa367f521f7c6215d43afa8cfcc0a3a0f56b803785e966c2430710a45a83"
 ["contracts/m231_checkpoint_bound_atlif_fc2_stream_bridge_contract_r1_20260825.json"]="9e7699a2133b50f80286f352fa2cc69bcabd9277482e897fee1123f2b450ea18"
 ["contracts/m231r2_same_cycle_fault_atomicity_correction_contract_r1_20260825.json"]="1c0ea65de1c71259fa96e4a3e14107eb15440096b58078e509351afff4afc1d5"
 ["results/m231_checkpoint_bound_atlif_fc2_stream_bridge_screen_r1_20260825/m231_checkpoint_bound_atlif_fc2_stream_bridge_screen_r1.json"]="7b03a1fed2844bb487984d2d387aecc544cff9e26602d5292263a48c50e89597"
 ["results/m231_independent_hammer_review_r1_20260825/SHA256SUMS"]="09dc0085cdf35fe1ca7175b2867c5fd444b98712c58d8e70474d0971474ed427"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
 task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" >"$task_run/input_sha256.txt"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
for task_width in 384 768 1536 3072; do
 task_dir="$task_run/w$task_width"
 mkdir "$task_dir"
 set +e
 "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
  +define+M231_INPUT_WIDTH="$task_width" -timescale=1ns/1ps -cm assert \
  -Mdir="$task_dir/csrc" \
  -f dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_directed_vcs.f \
  -top tb_m231_atlif32_to_fc2_raw4_pingpong_bridge \
  -o "$task_dir/simv" >"$task_dir/compile.log" 2>&1
 task_rc=$?
 set -e
 echo "$task_rc" >"$task_dir/compile.rc"
 [[ $task_rc -eq 0 && -x "$task_dir/simv" ]] || exit 20
 grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_dir/compile.log" && exit 21 || true
 set +e
 "$task_dir/simv" +ntb_random_seed="2312$task_width" -no_save -cm assert \
  -assert report="$task_dir/assert.report" >"$task_dir/sim.log" 2>&1
 task_rc=$?
 set -e
 echo "$task_rc" >"$task_dir/sim.rc"
 [[ $task_rc -eq 0 ]] || exit 22
 grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
  "$task_dir/sim.log" "$task_dir/assert.report" && exit 23 || true
 case "$task_width" in
  384) task_packets=6 ;;
  768) task_packets=12 ;;
  1536) task_packets=24 ;;
  3072) task_packets=48 ;;
 esac
 grep -Eq "PASS M231r2 W=$task_width pairs=3 tokens=6 packets=$task_packets header_stalls=[1-9][0-9]* raw_stalls=[1-9][0-9]* full_hits=[1-9][0-9]* attacks=2 fault_atomic=1 cycles=" "$task_dir/sim.log" || exit 30
 for task_cover in cp_pingpong_full cp_header_stall cp_raw_stall cp_fault cp_fault_while_raw_would_accept cp_complete_pair; do
  grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_dir/assert.report" || exit 31
 done
done
{
 echo status=PASS_M231R2_SAME_CYCLE_FAULT_ATOMICITY_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo widths=384_768_1536_3072
 echo pairs_each=3
 echo tokens_each=6
 echo ordinary_protocol_attacks_each=1
 echo ready_raw_concurrent_fault_attacks_each=1
 echo fault_cycle_accepts_each=0
 echo fault_cycle_state_commits_each=0
 echo transpose_mismatches=0
 echo transaction_mismatches=0
 echo assertion_failures=0
 echo complete_ffn=false
 echo system_speedup=false
 echo headline=false
} >"$task_run/m231r2_vcs_receipt_r1.txt"
sha256sum "$task_runner" >"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$task_run/SHA256SUMS"
echo PASS_M231R2_SAME_CYCLE_FAULT_ATOMICITY_EXACT_VCS >"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M231r2 exact VCS sealed at $task_run"
