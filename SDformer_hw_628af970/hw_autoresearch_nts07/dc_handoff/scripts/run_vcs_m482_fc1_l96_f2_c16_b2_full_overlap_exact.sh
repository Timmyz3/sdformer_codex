#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_dc_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m482_fc1_l96_f2_c16_b2_full_overlap_vcs_r2_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]]||exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m482/m482_fc1_l96_f2_c16_b2_full_overlap_island.sv"]="37c4816a5e1e71ead06e4318a0507dae1a9bcb3cfc7e3a9fa76e27302a3b104a"
 ["verif_m482/m482_fc1_l96_f2_c16_b2_full_overlap_assertions.sv"]="55b8e8ffe63aab432d22729a5e04c1c96f26ef9924ce986cd7ee1d722466cafa"
 ["tb_m482/tb_m482_fc1_l96_f2_c16_b2_full_overlap_island.sv"]="96b20993e7cbf8226a4dcb7dd51d5c15f20039f4a2a766f56eb9de4b769aff49"
 ["dc_handoff/filelists/date_m482_fc1_l96_f2_c16_b2_full_overlap_directed_vcs.f"]="83aca04ddce28138a415337a4be7620d65442ffb17c767a708774c9577567fb3"
 ["system_simulator/scripts/analyze_m482_fc1_l96_f2_c16_b2_full_overlap_recurrence.py"]="6e6b98dec74e7d32c897e6b62811f5986e2238df4834aa64ee6283438d24dff8"
 ["contracts/m482_fc1_l96_f2_c16_b2_full_overlap_vcs_contract_r1_20260827.json"]="26c703cf972765665a7cfd7ff290a7796452e048c9e87f3ec817e5e1eef95888"
 ["results/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826/payload/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2.json"]="2a7a1c917cb2f9aa1adb61092c7619de8d9b495aab5550f1fa41291188006578"
 ["results/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826/SHA256SUMS"]="fe323dc43a90b2fa33d23fb15c2eb55289b6685819da17c7b581ea340a846713"
 ["results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825/SHA256SUMS"]="7591869a0e519f32e309794a5f66d43bfd1b57d059f4cc2261d9be4ae5f9186e"
 ["reviews/m481_fc1_fullwidth_dse_independent_hammer_r1_20260826/SHA256SUMS"]="74aa1de572dbd58cef8cc63e078ee475601abf92b23c4b71354f07d0b2c42691"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
 ["../../../synopsys_date_dual/reviews/m483_open_source_rtl_trick_audit_r1_20260827.json"]="eb60ea57fa065c73587ed2d2d3a315fcc0feb6c5563d1abb6abac282e1316a53"
)
:>"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}";do
 task_observed="$(sha256sum "$task_path"|awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed">>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]]||exit 10
done
sha256sum "${!task_expected[@]}">"$task_run/input_sha256.txt"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
 -cm assert -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m482_fc1_l96_f2_c16_b2_full_overlap_directed_vcs.f \
 -top tb_m482_fc1_l96_f2_c16_b2_full_overlap_island \
 -o "$task_run/simv">"$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc">"$task_run/compile.rc"
[[ $task_rc -eq 0&&-x "$task_run/simv" ]]||exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error|Warning:' "$task_run/compile.log"&&exit 21||true
set +e
"$task_run/simv" +ntb_random_seed=48201 -no_save -cm assert \
 -assert report="$task_run/assert.report">"$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc">"$task_run/sim.rc"
[[ $task_rc -eq 0 ]]||exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
 "$task_run/sim.log" "$task_run/assert.report"&&exit 23||true
grep -Eq 'PASS M482 groups=8 all255_factor_cycles=707 all255_sparse_cycles=1079 all255_ratio=1.526167 factor_rounds=652 sparse_rounds=1024 attacks=1 empty=1 latency_checks=3828 stalls=[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]* commits=64' "$task_run/sim.log"||exit 30
for task_cover in cp_full_credit cp_dual_bank cp_factor_weight_overlap \
 cp_weight_update_overlap cp_triple_overlap cp_factor_stall cp_weight_stall \
 cp_bank_stall cp_commit_stall cp_same_bank_rdw cp_same_address_forward \
 cp_conflict cp_fault cp_done cp_empty;do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"||exit 31
done
python3 system_simulator/scripts/analyze_m482_fc1_l96_f2_c16_b2_full_overlap_recurrence.py \
 --m481-result results/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826/payload/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2.json \
 --m481-seal results/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826/SHA256SUMS \
 --contract contracts/m482_fc1_l96_f2_c16_b2_full_overlap_vcs_contract_r1_20260827.json \
 --m483-review ../../../synopsys_date_dual/reviews/m483_open_source_rtl_trick_audit_r1_20260827.json \
 --docs359 docs/359_DATE终局冻结_20260813.md \
 --vcs-log "$task_run/sim.log" --assert-report "$task_run/assert.report" \
 --output "$task_run/m482_fc1_l96_f2_c16_b2_full_overlap_vcs_recurrence_r1.json" \
 >"$task_run/analyzer.log" 2>&1
grep -Eq 'PASS M482 exact recurrence ratio=1.359896673 envelope=1.044983135 verdict=NO_GO_L96_F2_C16_B2_AS_PERFORMANCE_POINT' "$task_run/analyzer.log"||exit 40
{
 echo status=PASS_M482_VCS__P0_NO_GO_B2_BELOW_1P50
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo geometry=L96_F2_C16_B2
 echo directed_groups=8
 echo all255_definition=masks_1_through_255_not_all_ones
 echo directed_all255_ratio_not_trace=1.526167
 echo serial_view_not_rtl=2.018642790
 echo factor_weight_parallel_view_not_rtl=1.826227656
 echo full_overlap_analytical_not_rtl=1.433143969
 echo exact_frequency_compressed_recurrence_ratio=1.359896673
 echo ideal_scope_corrected_envelope_sensitivity_not_speedup=1.044983135
 echo p0_l96_f2_c16_b2=NO_GO
 echo p1_f2_b4_sensitivity_not_admitted=1.438200567
 echo p1_f4_b4_sensitivity_not_admitted=1.696926427
 echo response_contract=fixed_2cycle_in_order
 echo legal_reorder_supported=false
 echo literal_full_trace_vcs=false
 echo directed_synthetic=true
 echo numeric_mismatches=0
 echo transaction_mismatches=0
 echo assertion_failures=0
 echo dc_run=false
 echo physical_sram_macro=false
 echo measured_performance=false
 echo complete_fc1=false
 echo complete_ffn=false
 echo system_speedup=false
 echo headline=false
 echo paper_ppa_ready=false
}>"$task_run/m482_vcs_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 \
 |sort -z|xargs -0 sha256sum>"$task_run/SHA256SUMS"
sha256sum "$task_run/SHA256SUMS">"$task_run/SHA256SUMS.seal.sha256"
echo PASS_M482_VCS__P0_NO_GO_B2_BELOW_1P50>"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M482 exact VCS sealed at $task_run"
