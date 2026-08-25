#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m167_dynamic_bn_rank3_three_mode_shared96_kernel_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M167 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m167/m167_dynamic_bn_rank3_three_mode_shared96_kernel.sv"]="9cb7bbeb4ef720c6d0ec09bb67df2a7ebd3438cde055fd7f6412fb55d1a9705c"
    ["verif_m167/m167_dynamic_bn_rank3_three_mode_shared96_kernel_assertions.sv"]="c20457f42988422ef8588ea370fd96cda86b7df6efee5f628b41701d80d5d67f"
    ["tb_m167/tb_m167_dynamic_bn_rank3_three_mode_shared96_kernel.sv"]="7fbd9945470481175b325489389e584ac2743a25fec2fcb438b2fd97d826a2e6"
    ["dc_handoff/filelists/date_m167_three_mode_shared96_kernel_directed_vcs.f"]="7920c39c571a42afbb86b5d4256b516cb844ece88da7e6e738192104e1a09510"
    ["contracts/m167_dynamic_bn_rank3_three_mode_shared96_kernel_vcs_contract_r1_20260824.json"]="5492fb060df91c4f89475c9653598f03ff2bbe04b54f70ec6e06aad156fe2205"
    ["dc_handoff/runs/m166r2_prefolded_rank3_left_atlif_backend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="bde519f9ec0b9110d4bb7d66dab89ebc2a8f1c59c12045dc1d0c8d01e794c5db"
    ["dc_handoff/runs/m166_prefolded_rank3_left_atlif_backend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="2c43b532d1862a0d780c9ad76a00a75f5ae9dab813650b6b2a2cd7f0a0b377ad"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m167_three_mode_shared96_kernel_directed_vcs.f \
    -top tb_m167_dynamic_bn_rank3_three_mode_shared96_kernel \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=1 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M167 dynamic-BN rank3 three-mode shared96 kernel VCS issues=361 results=361 front_issues=121 back_issues=120 prefold_issues=120 main_signed_int8_product_slots=96 front_square_lanes=32 front_products=11616 back_products=11520 prefold_products=11520 front_squares=3872 consecutive_issue_ii1_hits=89 same_cycle_result_replace=[1-9][0-9]* output_stall_cycles=[1-9][0-9]* amplitude_sideband_checks=120 protocol_attacks=1 full_front_tile_issues=5 full_back_tile_issues=5 prefold_products_per_group=640 prefold_issue_cycles_per_group=7 full_rank3_products_per_tile=960 dense_t10_products_per_tile=1600 dense_capacity_lower_bound_cycles=17 conditional_capacity_cycle_boundary=1.7 shared_full_controller=false paft_valid825=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_front_issue cp_back_issue_with_amplitude \
        cp_prefold_issue cp_same_cycle_result_replace \
        cp_stall_then_accept cp_fault_preserves_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_replace="$(sed -n 's/.* same_cycle_result_replace=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_stalls="$(sed -n 's/.* output_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M167_DYNAMIC_BN_RANK3_THREE_MODE_SHARED96_KERNEL_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "accepted_issues=361"
    echo "accepted_results=361"
    echo "front_issues=121"
    echo "back_issues=120"
    echo "prefold_issues=120"
    echo "shared_signed_int8_main_product_slots=96"
    echo "front_signed_square_lanes=32"
    echo "front_products=11616"
    echo "back_products=11520"
    echo "prefold_products=11520"
    echo "front_square_products=3872"
    echo "consecutive_issue_ii1_hits=89"
    echo "same_cycle_result_replace=$task_replace"
    echo "output_stall_cycles=$task_stalls"
    echo "back_amplitude_sideband_checks=120"
    echo "protocol_attacks=1"
    echo "front_issues_per_tile=5"
    echo "back_issues_per_tile=5"
    echo "prefold_products_per_hidden_group=640"
    echo "prefold_issues_per_hidden_group=7"
    echo "dense_t10_products_per_tile=1600"
    echo "rank3_front_plus_back_products_per_tile=960"
    echo "dense_capacity_lower_bound_cycles_at_96_slots=17"
    echo "rank3_front_plus_back_issue_cycles_per_tile=10"
    echo "conditional_dense_to_rank3_capacity_cycle_boundary=1.7"
    echo "shared_main_product_pool=true"
    echo "atlif_amplitude_sideband=true"
    echo "full_controller=false"
    echo "dynamic_bn_rsqrt=false"
    echo "fixed_point_checkpoint_equivalence=false"
    echo "paft_valid825=false"
    echo "full_ffn_cycles=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m167_dynamic_bn_rank3_three_mode_shared96_kernel.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M167 three-mode shared96 kernel VCS sealed at $task_run"
