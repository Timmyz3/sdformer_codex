#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_dc_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m270_fc1_descriptor_lifecycle_corrective_vcs_r2_exact_20260825"
task_vcs_real="/opt/synopsys/vcs/V-2023.12-SP1"
[[ ! -e "$task_run" ]]||exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m270/m270_fc1_descriptor_lifecycle_wrapper.sv"]="90743313ea0834db6186bbec7d8a5230183300fc4a6aa3ffd0d89a3c500fc725"
 ["verif_m270/m270_fc1_descriptor_lifecycle_assertions.sv"]="1bb351c07062c354d7157481a5f15455759057c9af82f5eee60f2d348fe5491d"
 ["tb_m270/tb_m270_fc1_descriptor_lifecycle_wrapper.sv"]="cc3fed92960881d1ec29b09d707bb216140a01128808364d7c975f473f2c019c"
 ["dc_handoff/filelists/date_m270_fc1_descriptor_lifecycle_rtl.f"]="4e51b47772933db7adf23145cc23e0e8f15e30f8262b10fbd42c478abcad5f9d"
 ["dc_handoff/filelists/date_m270_fc1_descriptor_lifecycle_corrective_vcs.f"]="4416ab2c87ba6386c15c3826ccd1c0ae9b26319aa1ece7b4f2b1330d11a8f1fe"
 ["contracts/m270_fc1_fail_closed_header_correction_vcs_contract_r1_20260825.json"]="196d5d49c616da09060f191fc917e1290dee0cfccedc6fd6af6baeee35b52108"
 ["results/m268_independent_m262_hammer_r1_20260825/SHA256SUMS"]="d3870ff894c11f9986ac1236e71508b2a8c9406df033accf51d221d9d488d4c2"
 ["results/m262_fc1_descriptor_lifecycle_author_r1_20260825/SHA256SUMS"]="a4c33b15e9fa4202f9dd84cde9718325b67b50cdcc3d9829ce3cd4f224e5213e"
 ["results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/SHA256SUMS"]="f60f3fa5639d7e9410a081afd6e285a7d83443867f2f0f4b110e4c4956450245"
 ["results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/SHA256SUMS"]="23f10ed13167d6dc8c6b5c9dbeba42a0777e995becfc7241396746608066b16e"
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
 echo "-f dc_handoff/filelists/date_m270_fc1_descriptor_lifecycle_corrective_vcs.f"
 echo "-top tb_m270_fc1_descriptor_lifecycle_wrapper"
}>"$task_run/compile.command.txt"
set +e
env VCS_HOME="$task_run/vcs_home" VCS_ARCH_OVERRIDE=linux \
 "$task_run/vcs_home/bin/vcs" -full64 -sverilog -assert svaext \
 -timescale=1ns/1ps -cm assert -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m270_fc1_descriptor_lifecycle_corrective_vcs.f \
 -top tb_m270_fc1_descriptor_lifecycle_wrapper -o "$task_run/simv" \
 >"$task_run/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc">"$task_run/compile.rc"
[[ $task_rc -eq 0&&-x "$task_run/simv" ]]||exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error|^Fatal' "$task_run/compile.log"&&exit 21||true

set +e
"$task_run/simv" +ntb_random_seed=2700825 -no_save -cm assert \
 -assert report="$task_run/assert.report">"$task_run/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc">"$task_run/sim.rc"
[[ $task_rc -eq 0 ]]||exit 22
grep -Eiq 'failed at|Offending|^Error|Fatal:|watchdog' \
 "$task_run/sim.log" "$task_run/assert.report"&&exit 23||true
grep -Eq '^PASS M270 lanes=8 contexts=8 tiles=6 empty=1 desc=26 clean_cycle_checks=22 commits=40 attacks=7 stalls=[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$' \
 "$task_run/sim.log"||exit 30
for task_cover in cp_empty cp_dense cp_bit_sparse cp_factorized \
 cp_factor_stall cp_weight_stall cp_acc_read_stall cp_acc_write_stall \
 cp_commit_stall cp_commit_last cp_abort_stall cp_factor_fault \
 cp_weight_fault cp_acc_fault cp_overflow cp_protocol_fault;do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"||exit 31
done
{
 echo status=PASS_M270_FC1_FAIL_CLOSED_HEADER_CORRECTIVE_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo lanes=8
 echo contexts=8
 echo maximum_descriptors_per_tile=3072
 echo clean_descriptor_cycle_formula=6_plus_3_times_context_popcount
 echo empty_bypass_atomic=true
 echo factor_weight_acc_identity_fail_closed=true
 echo malformed_idle_header_fail_closed=true
 echo factor_address_wrap_fail_closed=true
 echo clean_popcounts_one_through_eight=true
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
}>"$task_run/m270_vcs_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS -print0|sort -z|xargs -0 sha256sum>"$task_run/SHA256SUMS"
echo PASS_M270_FC1_FAIL_CLOSED_HEADER_CORRECTIVE_EXACT_VCS>"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M270 exact VCS sealed at $task_run"
