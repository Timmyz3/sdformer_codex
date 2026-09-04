#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m202_fc2_raw4_to_descriptor4_fresh_bypass_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M202 sealed VCS run" >&2; exit 2; }
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m202/m202_fc2_raw4_to_descriptor4_fresh_bypass_compactor.sv"]="eb9f42ffd4286a4f5c83436acdad30568ddd6e7d90510e725d210a9a35677354"
 ["tb_m202/m202_m201_interface_adapter.sv"]="162fe95afb5543145e94f572d56f7a733ce566ac7751e25c6a5ad91933a808f6"
 ["tb_m202/tb_m202_fc2_raw4_to_descriptor4_fresh_bypass_compactor.sv"]="5ba67ced1e179f0569fafb7f32a87854d3e495b563adc44591b80790887aacd3"
 ["verif_m202/m202_fc2_raw4_to_descriptor4_fresh_bypass_assertions.sv"]="c7d01f8e21c6572997c4016e15d5d7977b818b3a2ff9bf964b49581dcc180b83"
 ["verif_m201/m201_fc2_raw4_to_descriptor4_stable_compactor_assertions.sv"]="f0eeecc97b50b4fa201123e331ab6cc2b530cbd73611f25c2ffca5a0dedbd656"
 ["tb_m201/tb_m201_fc2_raw4_to_descriptor4_stable_compactor.sv"]="fcf9cf3355b00b3f3b39e49ac937263e133a38571f1752ba2fcd7a7ee022266a"
 ["dc_handoff/filelists/date_m202_fc2_raw4_to_descriptor4_fresh_bypass_directed_vcs.f"]="0100e3042fa5fe45ce6ddca27c0e59995373fad2dabeb295d910538b2653e564"
 ["contracts/m202_fc2_raw4_to_descriptor4_fresh_bypass_vcs_contract_r1_20260825.json"]="c811008745a4edb93e59f3d111226d850fb00b5694e7752f7cdc02161875f132"
 ["results/m199_h67_fc2_decoupled_scanner_compactor_dse_r1_20260825/manifest.sha256"]="e23a72e2a59e4119a3d54eb78bcbf56dd768a165c1e883b364cf6cb4075c0c08"
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
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" -f dc_handoff/filelists/date_m202_fc2_raw4_to_descriptor4_fresh_bypass_directed_vcs.f -top tb_m201_fc2_raw4_to_descriptor4_stable_compactor -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log" && exit 21 || true
set +e
"$task_run/simv" +ntb_random_seed=201025 -no_save -assert report="$task_run/assert.report" -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?; set -e; echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M201 raw4-to-descriptor4 matched compactor VCS tokens=241 raw_packets=911 raw_beats=3643 descriptors=2305 descriptor_packets=1090 descriptor_stalls=251 raw_backpressure=222 simultaneous_push_pop=910 full4=69 zero_tokens=1 protocol_attacks=4 queue_depth=8 physical_speedup=false complete_fc2=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report" && exit 32 || true
for task_cover in cp_raw4_all_nonzero cp_raw4_all_zero cp_descriptor4 cp_window_boundary cp_descriptor_stall cp_raw_backpressure cp_simultaneous_push_pop cp_zero_token_done cp_bad_header_attack cp_bad_raw_attack cp_first_packet_fresh_bypass cp_first_packet_four_fresh; do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done
grep -Eq 'cp_first_packet_fresh_bypass, .* 604 match' "$task_run/assert.report" || exit 34
grep -Eq 'cp_first_packet_four_fresh, .* 8 match' "$task_run/assert.report" || exit 35
{
 echo status=PASS_M202_FC2_RAW4_TO_DESCRIPTOR4_FRESH_BYPASS_VCS_SVA
 echo exact_sha=true; echo random_seed=201025; echo tokens=241; echo raw_packets=911; echo raw_beats=3643
 echo descriptors=2305; echo descriptor_packets=1090; echo descriptor_stalls=251; echo raw_backpressure_cycles=222
 echo simultaneous_push_pop=910; echo fresh_bypass_cover_matches=604; echo fresh_four_nonzero_cover_matches=8
 echo legacy_m201_scoreboard_reused_through_test_only_adapter=true
 echo fresh_arrival_same_cycle_bypass=true; echo cycle_matches_m199_same_cycle_emit_semantics=true
 echo integrated_frontend=false; echo complete_fc2=false; echo physical_speedup=false; echo system_speedup=false; echo paper_ppa_ready=false; echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum dc_handoff/scripts/run_vcs_m202_fc2_raw4_to_descriptor4_fresh_bypass.sh > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M202 fresh-bypass compactor VCS sealed at $task_run"
