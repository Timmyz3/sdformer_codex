#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_dc_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m231_atlif32_to_fc2_raw4_pingpong_directed_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]]||exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m231/m231_atlif32_to_fc2_raw4_pingpong_bridge.sv"]="2df1e2deaf2ea397b60fa1632d571349155b0537fbdfe259b9049d4f722135bb"
 ["verif_m231/m231_atlif32_to_fc2_raw4_pingpong_bridge_assertions.sv"]="7358a521ad72c920c6b0e4e0d8620d5f71355c2fea6be56610bc258d53a866d5"
 ["tb_m231/tb_m231_atlif32_to_fc2_raw4_pingpong_bridge.sv"]="151baaf5275f3593dfcca489b60d198fa7221ceb632c0bddd555107808982649"
 ["dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_rtl.f"]="36b3c4cd631f97a58762639044a475904b41297a249b3577e4a240a156196417"
 ["dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_directed_vcs.f"]="a7e1fa367f521f7c6215d43afa8cfcc0a3a0f56b803785e966c2430710a45a83"
 ["contracts/m231_checkpoint_bound_atlif_fc2_stream_bridge_contract_r1_20260825.json"]="9e7699a2133b50f80286f352fa2cc69bcabd9277482e897fee1123f2b450ea18"
 ["system_simulator/scripts/analyze_m231_checkpoint_bound_atlif_fc2_stream_bridge.py"]="87a40bb6f836a40dc7d5fbd0944ab3ba9f93c544ae2e90e9b435fdd231eec9ed"
 ["results/m231_checkpoint_bound_atlif_fc2_stream_bridge_screen_r1_20260825/m231_checkpoint_bound_atlif_fc2_stream_bridge_screen_r1.json"]="7b03a1fed2844bb487984d2d387aecc544cff9e26602d5292263a48c50e89597"
 ["results/m230_independent_hammer_review_r1_20260825/SHA256SUMS"]="7b8e904a873d2b2abf95667a3b6dcff100400f2127db661cd59074905eddadc4"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
:>"$task_run/preflight_sha_checks.txt"
for p in "${!task_expected[@]}";do
 o="$(sha256sum "$p"|awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$p" "${task_expected[$p]}" "$o">>"$task_run/preflight_sha_checks.txt"
 [[ "$o" == "${task_expected[$p]}" ]]||exit 10
done
sha256sum "${!task_expected[@]}">"$task_run/input_sha256.txt"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
for task_width in 384 768 1536 3072;do
 task_dir="$task_run/w$task_width"
 mkdir "$task_dir"
 set +e
 "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
  +define+M231_INPUT_WIDTH="$task_width" -timescale=1ns/1ps -cm assert \
  -Mdir="$task_dir/csrc" \
  -f dc_handoff/filelists/date_m231_atlif32_to_fc2_raw4_pingpong_bridge_directed_vcs.f \
  -top tb_m231_atlif32_to_fc2_raw4_pingpong_bridge \
  -o "$task_dir/simv">"$task_dir/compile.log" 2>&1
 task_rc=$?
 set -e
 echo "$task_rc">"$task_dir/compile.rc"
 [[ $task_rc -eq 0&&-x "$task_dir/simv" ]]||exit 20
 grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_dir/compile.log"&&exit 21||true
 set +e
 "$task_dir/simv" +ntb_random_seed="231$task_width" -no_save -cm assert \
  -assert report="$task_dir/assert.report">"$task_dir/sim.log" 2>&1
 task_rc=$?
 set -e
 echo "$task_rc">"$task_dir/sim.rc"
 [[ $task_rc -eq 0 ]]||exit 22
 grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
  "$task_dir/sim.log" "$task_dir/assert.report"&&exit 23||true
 case "$task_width" in
  384) task_packets=6;;768) task_packets=12;;1536) task_packets=24;;3072) task_packets=48;;
 esac
 grep -Eq "PASS M231 W=$task_width pairs=3 tokens=6 packets=$task_packets header_stalls=[1-9][0-9]* raw_stalls=[1-9][0-9]* full_hits=[1-9][0-9]* attacks=1 cycles=" "$task_dir/sim.log"||exit 30
 for task_cover in cp_pingpong_full cp_header_stall cp_raw_stall cp_fault cp_complete_pair;do
  grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_dir/assert.report"||exit 31
 done
done
{
 echo status=PASS_M231_ATLIF32_TO_FC2_RAW4_PINGPONG_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo widths=384_768_1536_3072
 echo pairs_each=3
 echo tokens_each=6
 echo transpose_mismatches=0
 echo transaction_mismatches=0
 echo assertion_failures=0
 echo protocol_attacks_each=1
 echo maximum_bridge_storage_bytes=1536
 echo frozen_trace_write_plus_read_elision_bytes=875520000
 echo traffic_is_not_cycle_speedup=true
 echo complete_ffn=false
 echo system_speedup=false
 echo headline=false
} >"$task_run/m231_vcs_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' ! -name SHA256SUMS -print0|sort -z|xargs -0 sha256sum>"$task_run/SHA256SUMS"
echo PASS_M231_ATLIF32_TO_FC2_RAW4_PINGPONG_EXACT_VCS>"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M231 exact VCS sealed at $task_run"
