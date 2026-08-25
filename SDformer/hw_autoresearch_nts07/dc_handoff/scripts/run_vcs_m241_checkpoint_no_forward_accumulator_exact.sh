#!/usr/bin/env bash
set -euo pipefail
task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m241_checkpoint_no_forward_accumulator_directed_vcs_r1_exact_20260825"
task_vectors="$task_hw_root/results/m241_ordered_checkpoint_subset_r1_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" >"$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m241/m241_four_bank_checkpoint_no_forward_accumulator.sv"]="0d0c9a067e71f4a1edfe16d5af090431496c80dac7b7dd4131e4e674a7f38e49"
 ["verif_m241/m241_four_bank_checkpoint_no_forward_accumulator_assertions.sv"]="2bc0cc08d968713a7a0e19000f8e65e646624a72464c9bec3159651edcaa263b"
 ["tb_m241/tb_m241_four_bank_checkpoint_no_forward_accumulator.sv"]="4b0ce232d31ca9fa8d6fc89b4a8b9cff7f79b93b6275a27be92e1f05cdac2236"
 ["dc_handoff/filelists/date_m241_checkpoint_no_forward_accumulator_directed_vcs.f"]="687b1f0ff0579308b34826ec0b26e318117ee7de357d5dc46bad9fde8317ab10"
 ["system_simulator/scripts/export_m241_ordered_checkpoint_subset.py"]="60ca8f58f06d100989f4eaa50da29c6ecb22535bb984bdae175fd31a611cd2d3"
 ["contracts/m241_checkpoint_no_forward_accumulator_exact_vcs_contract_r1_20260825.json"]="6962e59d2f79c17c0388c156cfc1682f4017a1815333094dbde695029ec8fb8c"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/descriptor.mem"]="02865c805b0e79f363b7d29a2c3b045a21d46edb37d7be7c38b57c17f4fded5c"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/weight.mem"]="83d5a850c58173ac8692914216ac57b61127b379be952be6ae45525a755a1be1"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/meta.mem"]="e9e7bc782fbfe6bdfa1c23d1c0940b40a6d0a4245153d441c1e3be167d1a0903"
 ["results/m241_ordered_checkpoint_subset_r1_20260825/m241_ordered_checkpoint_subset.json"]="3e5eebb8c10592744a4a794174d7d6e017cfe3b82dcfffa06bc8e5d889e03ee5"
 ["results/m238_conv_patch_performance_hammer_r1_20260825/SHA256SUMS"]="e5c5a069be47802e006604d9ccabd488e1ced77cce92c7a32356b58e8610a008"
 ["results/m158_source_major_acc19_reorder_exactness_r2_20260824/manifest.sha256"]="22067f8a3bffaae0b00e200a8c3950467c17ee57692862d66d823689ecc14f1e"
 ["results/m158_independent_hammer_review_r2_20260824/manifest.sha256"]="b5a031e03873098bba7f78c41aa31bb2e32d55155f6c5f18f97fd6665639165f"
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
 cd results/m238_conv_patch_performance_hammer_r1_20260825
 sha256sum -c SHA256SUMS
) >"$task_run/m238_manifest_check.log"
(
 cd results/m158_source_major_acc19_reorder_exactness_r2_20260824
 sha256sum -c manifest.sha256
) >"$task_run/m158_proof_manifest_check.log"
(
 cd results/m158_independent_hammer_review_r2_20260824
 sha256sum -c manifest.sha256
) >"$task_run/m158_review_manifest_check.log"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
 -timescale=1ns/1ps -cm assert -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m241_checkpoint_no_forward_accumulator_directed_vcs.f \
 -top tb_m241_four_bank_checkpoint_no_forward_accumulator \
 -o "$task_run/simv" >"$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/compile.rc"
[[ $task_rc -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" && exit 21 || true

set +e
"$task_run/simv" +VECTOR_DIR="$task_vectors" \
 +ntb_random_seed=24120260825 -no_save -cm assert \
 -assert report="$task_run/assert.report" >"$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/sim.rc"
[[ $task_rc -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
 "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true
task_pass='PASS M241 checkpoint descriptors=126 real_groups=504 real_exact_writes=504 real_exact_lanes=4032 real_weight_macro_reads=56 real_cache_hits=448 real_acc_macro_reads=40 total_exact_write_checks=3372 total_exact_lane_checks=26976 mismatches=0 commit_stalls=21 alias_stalls=1 directed_tail_descriptors=4 real_negative_descriptors=2 protocol_attacks=3 younger_fault_atomicity=3 reset_flush=1 overflow_attacks=1 overflow_iterations=2850 banks=4 lanes=8 acc_bits=19 lazy_valid=true overflow_guard=true forwarding_payload_bits=0 dense_high_half_address=true real_full_trace=false m238_target_speedup=1.687018 physical_speedup=false system_speedup=false headline=false'
grep -Fx "$task_pass" "$task_run/sim.log" >/dev/null || exit 30
for task_cover in \
 cp_all_four_weight_banks cp_weight_cache_reuse cp_full4_descriptor \
 cp_tail_descriptor cp_negated_descriptor cp_all_four_accumulator_writes \
 cp_commit_stall cp_rmw_alias_interlock \
 cp_protocol_fault_with_older_commit cp_overflow_fault cp_window_done; do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" \
  "$task_run/assert.report" || exit 31
done

{
 echo status=PASS_M241_CHECKPOINT_NO_FORWARD_ACCUMULATOR_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo representative_banks=4
 echo representative_lanes=8
 echo accumulator_bits=19
 echo real_ordered_descriptors=126
 echo real_destination_groups=504
 echo real_exact_writes=504
 echo real_exact_lane_checks=4032
 echo real_integer_mismatches=0
 echo real_weight_macro_reads=56
 echo real_weight_cache_hits=448
 echo bounded_weight_read_work_reduction=9.0x
 echo real_accumulator_macro_reads=40
 echo real_negative_tuples=8
 echo directed_tail_descriptors=4
 echo commit_stall_cycles=21
 echo same_address_interlock_cycles=1
 echo stale_sequence_attacks=1
 echo replay_order_attacks=1
 echo cache_epoch_alias_attacks=1
 echo younger_protocol_fault_atomicity_checks=3
 echo reset_flush_checks=1
 echo overflow_atomicity_checks=1
 echo lazy_valid=true
 echo runtime_overflow_guard=true
 echo forwarding_payload_bits=0
 echo dense_high_half_address=true
 echo m149_instantiated=false
 echo m238_cycle_target=1.687017659x
 echo m238_cycle_target_admitted_by_m241=false
 echo real_full_trace=false
 echo dc_sta=false
 echo macro_ppa=false
 echo physical_speedup=false
 echo system_speedup=false
 echo headline=false
} >"$task_run/m241_vcs_receipt_r1.txt"
sha256sum "$task_runner" >"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
 >"$task_run/SHA256SUMS"
echo PASS_M241_CHECKPOINT_NO_FORWARD_ACCUMULATOR_EXACT_VCS \
 >"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M241 exact VCS sealed at $task_run"
