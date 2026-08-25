#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_dc_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]]||exit 2;mkdir -p "$(dirname "$task_run")";mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m229/m229_fc1_dual_held_prefetch_replay_island.sv"]="c36fe753a16fbf76ec9c1654d7ee991ab999964e2b3491d2b86c0badc6cce1e9"
 ["verif_m229/m229_fc1_dual_held_prefetch_replay_assertions.sv"]="8a54bac36493c16c8e8f604e5d953dfe65dc6fb25b2eb519f7aef7d6e92577d5"
 ["tb_m229/tb_m229_fc1_dual_held_prefetch_replay_island.sv"]="e312e6b3ef2904aa881085798d2a99aa6b4b9c73de9cb3f6dfcfe66bad7c3a6e"
 ["dc_handoff/filelists/date_m229_fc1_dual_held_prefetch_replay_rtl.f"]="1feb42fe141d3c60dab9d9d3179fb364a0bc29bd853b255316b65a69b6f6bc58"
 ["dc_handoff/filelists/date_m229_fc1_dual_held_prefetch_replay_directed_vcs.f"]="9a76395eef0a712b6216e4e597e14e7a05e1ee398f652556be2092ac46df8359"
 ["contracts/m229_fc1_dual_held_prefetch_replay_synopsys_contract_r1_20260825.json"]="424aabfe0e36570d221b2aa255414af3a320a6291f82269117376f5448546ee2"
 ["results/m227_independent_hammer_review_r1_20260825/SHA256SUMS"]="c6a1fe78c6c931a89f84aaa6f469ab79433a89a50ce6a77842faee9034f9e62b"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
:>"$task_run/preflight_sha_checks.txt"
for p in "${!task_expected[@]}";do o="$(sha256sum "$p"|awk '{print $1}')";
 printf 'path=%s expected=%s observed=%s\n' "$p" "${task_expected[$p]}" "$o">>"$task_run/preflight_sha_checks.txt";[[ "$o" == "${task_expected[$p]}" ]]||exit 10;done
sha256sum "${!task_expected[@]}">"$task_run/input_sha256.txt"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
for f in 1 2 4;do d="$task_run/f$f";mkdir "$d"
 set +e;"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext +define+M229_FANOUT="$f" -timescale=1ns/1ps -cm assert -Mdir="$d/csrc" -f dc_handoff/filelists/date_m229_fc1_dual_held_prefetch_replay_directed_vcs.f -top tb_m229_fc1_dual_held_prefetch_replay_island -o "$d/simv">"$d/compile.log" 2>&1;rc=$?;set -e;echo "$rc">"$d/compile.rc";[[ $rc -eq 0&&-x "$d/simv" ]]||exit 20
 grep -Eiq 'Warning-\[|Error-\[|^Error' "$d/compile.log"&&exit 21||true
 set +e;"$d/simv" +ntb_random_seed="22902$f" -no_save -cm assert -assert report="$d/assert.report">"$d/sim.log" 2>&1;rc=$?;set -e;echo "$rc">"$d/sim.rc";[[ $rc -eq 0 ]]||exit 22
 grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' "$d/sim.log" "$d/assert.report"&&exit 23||true
 grep -Eq "PASS M229 F=$f groups=3 desc=64 updates=276 attacks=3 overlaps=5[23] req_stalls=[1-9][0-9]* upd_stalls=[1-9][0-9]* cycles=" "$d/sim.log"||exit 30
 for c in cp_full_credit cp_overlap cp_fanout cp_req_stall cp_update_stall cp_fault cp_done;do grep -Eq "$c, .* [1-9][0-9]* match" "$d/assert.report"||exit 31;done
done
{
 echo status=PASS_M229_FC1_DUAL_HELD_PREFETCH_REPLAY_EXACT_VCS
 echo exact_sha=true;echo tool=Synopsys_VCS_V-2023.12-SP1
 echo variants=F1_F2_F4;echo descriptors_each=64;echo context_updates_each=276
 echo current_next_overlap_covered=true;echo protocol_attacks_each=3
 echo numeric_mismatches=0;echo transaction_mismatches=0;echo assertion_failures=0
 echo accumulator_capacity_port_cut_bits=14592;echo complete_fc1=false
 echo complete_ffn=false;echo system_speedup=false;echo headline=false
}>"$task_run/m229_vcs_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' ! -name SHA256SUMS -print0|sort -z|xargs -0 sha256sum>"$task_run/SHA256SUMS"
echo PASS_M229_FC1_DUAL_HELD_PREFETCH_REPLAY_EXACT_VCS>"$task_run/RUN_COMPLETE.txt"
task_complete=1;echo "PASS M229 exact VCS sealed at $task_run"
