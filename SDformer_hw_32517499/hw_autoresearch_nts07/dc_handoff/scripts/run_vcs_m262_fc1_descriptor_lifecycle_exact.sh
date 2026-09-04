#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_dc_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825"
task_vcs_real="/opt/synopsys/vcs/V-2023.12-SP1"
[[ ! -e "$task_run" ]]||exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m262/m262_fc1_descriptor_lifecycle_wrapper.sv"]="74e81b936cb7abb2f2cebd678c884fa21fe0fdb38039c8dda6e2010a2b02adc8"
 ["verif_m262/m262_fc1_descriptor_lifecycle_assertions.sv"]="8203e6acac8ba5c9d6b67444e94742b22456d75854cf2605070d1bb2a8348da9"
 ["tb_m262/tb_m262_fc1_descriptor_lifecycle_wrapper.sv"]="9f525e06a4c4dc85b30357337f968e271cdc4ca45eafe55e94f24cb6b446b49c"
 ["dc_handoff/filelists/date_m262_fc1_descriptor_lifecycle_rtl.f"]="f86e465cb09bbae6dedb80b5e5fc7aee5d4371167e8dd2b6f377855b66ef486f"
 ["dc_handoff/filelists/date_m262_fc1_descriptor_lifecycle_directed_vcs.f"]="3fc07799fa168ab3cda516c68d1a2af6b9d0cc640681920cdf7bc079b22a7f3d"
 ["contracts/m262_fc1_descriptor_lifecycle_vcs_trace_contract_r1_20260825.json"]="b1bbdee8d0b151af094eef9378b50936358695eced1553398d13a908dc824415"
 ["results/m230_independent_hammer_review_r1_20260825/SHA256SUMS"]="7b8e904a873d2b2abf95667a3b6dcff100400f2127db661cd59074905eddadc4"
 ["results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS"]="133c32c37d6ff61d19ca119634b5604d8a9fe12dd510cd4d9425e59e967247e5"
 ["contracts/m230_h67_fc1_m229_fixed_latency_trace_recurrence_contract_r1_20260825.json"]="2e59a52257b48676c6c667e26c70b1635bc6f5d025849ae1071ff2f9c49b0930"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
:>"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}";do
 task_observed="$(sha256sum "$task_path"|awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$task_path" \
  "${task_expected[$task_path]}" "$task_observed">>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]]||exit 10
done
sha256sum "${!task_expected[@]}">"$task_run/input_sha256.txt"

mkdir "$task_run/vcs_home"
for task_entry in "$task_vcs_real"/*;do
 ln -s "$task_entry" "$task_run/vcs_home/$(basename "$task_entry")"
done
ln -s "$task_vcs_real/linux64" "$task_run/vcs_home/linux"
{
 echo "vcs_real_root=$task_vcs_real"
 echo "vcs_launcher=$task_run/vcs_home/bin/vcs"
 echo "linux_alias_target=$task_vcs_real/linux64"
 echo "vcs_arch_override=linux"
}>"$task_run/vcs_tool_identity.txt"
{
 echo "VCS_ARCH_OVERRIDE=linux VCS_HOME=$task_run/vcs_home"
 echo "-full64 -sverilog -assert svaext -timescale=1ns/1ps -cm assert"
 echo "-f dc_handoff/filelists/date_m262_fc1_descriptor_lifecycle_directed_vcs.f"
 echo "-top tb_m262_fc1_descriptor_lifecycle_wrapper"
}>"$task_run/compile.command.txt"
set +e
env VCS_HOME="$task_run/vcs_home" VCS_ARCH_OVERRIDE=linux \
 "$task_run/vcs_home/bin/vcs" -full64 -sverilog -assert svaext \
 -timescale=1ns/1ps -cm assert -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m262_fc1_descriptor_lifecycle_directed_vcs.f \
 -top tb_m262_fc1_descriptor_lifecycle_wrapper -o "$task_run/simv" \
 >"$task_run/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc">"$task_run/compile.rc"
[[ $task_rc -eq 0&&-x "$task_run/simv" ]]||exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error|^Fatal' "$task_run/compile.log"&&exit 21||true

set +e
"$task_run/simv" +ntb_random_seed=2620825 -no_save -cm assert \
 -assert report="$task_run/assert.report">"$task_run/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc">"$task_run/sim.rc"
[[ $task_rc -eq 0 ]]||exit 22
grep -Eiq 'failed at|Offending|^Error|Fatal:|watchdog' \
 "$task_run/sim.log" "$task_run/assert.report"&&exit 23||true
grep -Eq '^PASS M262 lanes=8 contexts=8 tiles=5 empty=1 desc=18 clean_cycle_checks=14 commits=32 attacks=4 stalls=[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$' \
 "$task_run/sim.log"||exit 30
for task_cover in cp_empty cp_dense cp_bit_sparse cp_factorized \
 cp_factor_stall cp_weight_stall cp_acc_read_stall cp_acc_write_stall \
 cp_commit_stall cp_commit_last cp_abort_stall cp_factor_fault \
 cp_weight_fault cp_acc_fault cp_overflow;do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"||exit 31
done
{
 echo status=PASS_M262_FC1_DESCRIPTOR_LIFECYCLE_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo lanes=8
 echo contexts=8
 echo maximum_descriptors_per_tile=3072
 echo clean_descriptor_cycle_formula=6_plus_3_times_context_popcount
 echo empty_bypass_atomic=true
 echo factor_weight_acc_identity_fail_closed=true
 echo overflow_commit_abort_lifecycle=true
 echo numeric_mismatches=0
 echo transaction_mismatches=0
 echo assertion_failures=0
 echo full_96_lane=false
 echo full_trace_rtl=false
 echo dc=false
 echo complete_fc1=false
 echo complete_ffn=false
 echo system_speedup=false
 echo headline=false
}>"$task_run/m262_vcs_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS -print0|sort -z|xargs -0 sha256sum>"$task_run/SHA256SUMS"
echo PASS_M262_FC1_DESCRIPTOR_LIFECYCLE_EXACT_VCS>"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M262 exact VCS sealed at $task_run"
