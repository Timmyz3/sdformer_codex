#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m241r2_elastic_tagged_checkpoint_no_forward_directed_vcs_r1_exact_20260825"
task_vectors="$task_hw_root/results/m241_ordered_checkpoint_subset_r1_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "$task_run" ]] || exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" >"$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m241r2/m241r2_elastic_tagged_checkpoint_no_forward_accumulator.sv"]="d43647ec8e3d7789408c425fa1eed8c19b8b6b723e1f52b70da7f8cb6e53c144"
 ["verif_m241r2/m241r2_elastic_tagged_checkpoint_no_forward_accumulator_assertions.sv"]="7920064956ac8caa0d2a2ab5bea35803525a69f604f3af20e786c82b8bb4e3f7"
 ["tb_m241r2/tb_m241r2_elastic_tagged_checkpoint_no_forward_accumulator.sv"]="f8bff3d401c1c4cf8595bc7a7074d36a0e4941f95eb2e3915a27d4202cd6b459"
 ["dc_handoff/filelists/date_m241r2_elastic_tagged_checkpoint_no_forward_directed_vcs.f"]="37d9fb526eec2209cb036d95042247235fe02b935358450e031f19ac1128d398"
 ["contracts/m241r2_elastic_tagged_checkpoint_no_forward_exact_vcs_contract_r1_20260825.json"]="db85dd197c8f9379e13d1ade7dea77c0b2fa9fb0283a8a8257b2f07e37aef585"
 ["system_simulator/scripts/export_m241_ordered_checkpoint_subset.py"]="60ca8f58f06d100989f4eaa50da29c6ecb22535bb984bdae175fd31a611cd2d3"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/descriptor.mem"]="02865c805b0e79f363b7d29a2c3b045a21d46edb37d7be7c38b57c17f4fded5c"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/weight.mem"]="83d5a850c58173ac8692914216ac57b61127b379be952be6ae45525a755a1be1"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/meta.mem"]="e9e7bc782fbfe6bdfa1c23d1c0940b40a6d0a4245153d441c1e3be167d1a0903"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/m241_ordered_checkpoint_subset.json"]="3e5eebb8c10592744a4a794174d7d6e017cfe3b82dcfffa06bc8e5d889e03ee5"
 ["results/m249_m241_checkpoint_no_forward_independent_hammer_r1_20260825/SHA256SUMS"]="268bea0d1b3462037183a90873fce4bba1002389df21bd50fbf68b5157035713"
 ["results/m241_checkpoint_no_forward_accumulator_milestone_r1_20260825/SHA256SUMS"]="171f3e88cef7fded8062c690e45a4b9bdf8ee8b689768078e87d835e751a5064"
 ["results/m241_checkpoint_no_forward_accumulator_directed_vcs_r1_exact_20260825/SHA256SUMS"]="f6001e7d1d35f2ab8aab75fc07d559aa8b5e3a7aca583ad59ce18b2c85013721"
 ["results/m238_conv_patch_performance_hammer_r1_20260825/SHA256SUMS"]="e5c5a069be47802e006604d9ccabd488e1ced77cce92c7a32356b58e8610a008"
 ["results/m158_source_major_acc19_reorder_exactness_r2_20260824/manifest.sha256"]="22067f8a3bffaae0b00e200a8c3950467c17ee57692862d66d823689ecc14f1e"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: >"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
 task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' \
  "$task_path" "${task_expected[$task_path]}" "$task_observed" \
  >>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" >"$task_run/input_sha256.txt"

(
 cd results/m249_m241_checkpoint_no_forward_independent_hammer_r1_20260825
 sha256sum -c SHA256SUMS
) >"$task_run/m249_independent_review_manifest_check.log"
(
 cd results/m241_checkpoint_no_forward_accumulator_milestone_r1_20260825
 sha256sum -c SHA256SUMS
) >"$task_run/m241_r1_milestone_manifest_check.log"
(
 cd results/m241_checkpoint_no_forward_accumulator_directed_vcs_r1_exact_20260825
 sha256sum -c SHA256SUMS
) >"$task_run/m241_r1_vcs_manifest_check.log"
(
 cd results/m238_conv_patch_performance_hammer_r1_20260825
 sha256sum -c SHA256SUMS
) >"$task_run/m238_manifest_check.log"
(
 cd results/m158_source_major_acc19_reorder_exactness_r2_20260824
 sha256sum -c manifest.sha256
) >"$task_run/m158_proof_manifest_check.log"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
 -timescale=1ns/1ps -cm assert -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m241r2_elastic_tagged_checkpoint_no_forward_directed_vcs.f \
 -top tb_m241r2_elastic_tagged_checkpoint_no_forward_accumulator \
 -o "$task_run/simv" >"$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/compile.rc"
[[ $task_rc -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" && exit 21 || true

set +e
"$task_run/simv" +VECTOR_DIR="$task_vectors" \
 +ntb_random_seed=241220260825 -no_save -cm assert \
 -assert report="$task_run/assert.report" >"$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/sim.rc"
[[ $task_rc -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
 "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true

task_pass='PASS M241r2 scenarios=4 latency_modes=1_2_3_random real_descriptors_each=126 real_writes_each=504 real_lane_checks_each=4032 total_mismatches=0 weight_request_stalls=4 weight_response_stalls=2 acc_request_stalls=2 acc_response_stalls=2 commit_stalls=19 stale_weight_responses=1 stale_acc_responses=1 stale_response_accepts=0 loader_binding_attacks=1 overflow_aborts=1 overflow_success_commits=0 overflow_writes=0 accepted_younger_discarded=2 abort_stalls=2 recovery_commits=1 window_identity=true payload_epoch_binding=true lazy_valid=true overflow_guard=true forwarding_payload_bits=0 m149_instantiated=false real_full_trace=false m238_target_speedup=1.687018 physical_speedup=false system_speedup=false headline=false'
grep -Fx "$task_pass" "$task_run/sim.log" >/dev/null || exit 30
for task_cover in \
 cp_weight_request_stall cp_weight_response_stall \
 cp_acc_request_stall cp_acc_response_stall cp_cache_reuse cp_full4 \
 cp_negate cp_commit_stall cp_alias_interlock cp_stale_weight_response \
 cp_stale_acc_response cp_overflow_abort_with_two_younger cp_abort_stall \
 cp_context_abort cp_window_done; do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" \
  "$task_run/assert.report" || exit 31
done

{
 echo status=PASS_M241R2_ELASTIC_TAGGED_CHECKPOINT_NO_FORWARD_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo m249_independent_review_seal=268bea0d1b3462037183a90873fce4bba1002389df21bd50fbf68b5157035713
 echo repaired_m249_p1_01_fixed_one_cycle=true
 echo repaired_m249_p1_04_overflow_success_commit=true
 echo repaired_m249_p1_05_loader_payload_binding=true
 echo representative_banks=4
 echo representative_lanes=8
 echo accumulator_bits=19
 echo latency_modes=weight_acc_1_1,2_2,3_3,random_1_to_3
 echo real_scenarios=4
 echo real_ordered_descriptors_each=126
 echo real_destination_groups_each=504
 echo real_exact_writes_each=504
 echo real_exact_lane_checks_each=4032
 echo real_integer_mismatches=0
 echo real_weight_macro_read_groups_each=56
 echo real_weight_cache_hits_each=448
 echo bounded_weight_read_work_reduction=9.0x_not_cycle_speedup
 echo weight_request_stalls=4
 echo weight_response_stalls=2
 echo accumulator_request_stalls=2
 echo accumulator_response_stalls=2
 echo stale_weight_responses=1
 echo stale_accumulator_responses=1
 echo stale_response_accepts=0
 echo loader_binding_attacks=1
 echo overflow_aborts=1
 echo overflow_success_commits=0
 echo overflow_writes=0
 echo accepted_younger_discarded=2
 echo abort_stalls=2
 echo post_abort_recovery_commits=1
 echo explicit_window_identity=true
 echo payload_epoch_binding=true
 echo lazy_valid=true
 echo runtime_overflow_guard=true
 echo forwarding_payload_bits=0
 echo m149_instantiated=false
 echo m238_cycle_target=1.687017659x
 echo m238_cycle_target_admitted_by_m241r2=false
 echo real_full_trace=false
 echo dc_launched=false
 echo selected_sram_macro=false
 echo physical_speedup=false
 echo system_speedup=false
 echo paper_ppa_ready=false
 echo headline=false
} >"$task_run/m241r2_vcs_receipt_r1.txt"

sha256sum "$task_runner" >"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' ! -name SHA256SUMS \
 -print0 | sort -z | xargs -0 sha256sum >"$task_run/SHA256SUMS"
echo PASS_M241R2_ELASTIC_TAGGED_CHECKPOINT_NO_FORWARD_EXACT_VCS \
 >"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M241r2 exact VCS sealed at $task_run"
