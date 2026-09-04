#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M184 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m184/m184_fc2_dual_window_k8_fixed_bank_frontend.sv"]="c6212049305faf42cda13f7f3408d5fa478c79a7c76c142501ec01d9f1e01cd6"
    ["verif_m184/m184_fc2_dual_window_k8_fixed_bank_frontend_assertions.sv"]="aeea5d8d5391785f5e49d212f825f637715632fb1756b234c412517cc57f4dbc"
    ["tb_m184/tb_m184_fc2_dual_window_k8_fixed_bank_frontend.sv"]="1443cbc47b746a735da6e91600656e917e915c07b0450325aceb411e6d2b0b16"
    ["dc_handoff/filelists/date_m184_fc2_dual_window_k8_fixed_bank_frontend_directed_vcs.f"]="8a1b606a27f73a37925ead32d36fb9704637481ba27b2d6a0224ee1607613b4c"
    ["contracts/m184_fc2_dual_window_k8_fixed_bank_frontend_vcs_contract_r1_20260825.json"]="64883d54ebb69471198851078bcb17a336dc34f0df0f26b921938687804b8e1e"
    ["contracts/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse_contract_r1_20260824.json"]="4dea36a1ebcb544ea597a84c34fdf7759962adaaf6d6ca2f2ae3a7f511be642a"
    ["contracts/m181_m182_independent_review_semantic_correction_overlay_r1_20260824.json"]="0a54c02958220e32fbc9fd1c4cd766f943352c9e7b6ec9c855ae767b672e9562"
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
    -f dc_handoff/filelists/date_m184_fc2_dual_window_k8_fixed_bank_frontend_directed_vcs.f \
    -top tb_m184_fc2_dual_window_k8_fixed_bank_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=184025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M184 FC2 dual-window K8 fixed-bank frontend VCS headers=15 descriptors=60 tokens=15 bitmap_events=320 unique_groups=57 unique_source_terms=320 replayed_group_results=283 replayed_source_terms=1757 one_source_groups=22 two_source_groups=23 three_source_groups=5 four_source_groups=21 five_source_groups=21 six_source_groups=21 seven_source_groups=1 eight_source_groups=169 descriptor_stall_cycles=77 group_stall_cycles=55 both_windows_closed_cycles=175 release_refill_hits=1 window_replace_hits=3 cross_descriptor_groups=40 consecutive_group_hits=224 stage_windows=2,4,8,8 max_two_buffer_bitmap_bits=1536 protocol_attacks=4 extent_overflow_attacks=1 same_cycle_header_rearm=1 global_topk_sort=false bank_id_payload=false fixed_bank_valid_mask=true token_directory=true native_or_preindexed_source_required=true posthoc_scanner_speedup=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_one_source_group cp_two_source_group \
        cp_three_source_group cp_four_source_group cp_five_source_group \
        cp_six_source_group cp_seven_source_group cp_eight_source_group \
        cp_group_stall_then_accept cp_descriptor_backpressure \
        cp_cross_descriptor_group cp_window_to_window_replace \
        cp_release_and_refill cp_both_windows_closed cp_stage0 cp_stage1 \
        cp_stage2 cp_stage3 cp_zero_token cp_nonzero_token \
        cp_same_cycle_header_rearm; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M184_FC2_DUAL_WINDOW_K8_FIXED_BANK_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=184025"
    echo "headers=15"
    echo "descriptors=60"
    echo "tokens=15"
    echo "bitmap_events=320"
    echo "unique_groups=57"
    echo "unique_source_terms=320"
    echo "replayed_group_results=283"
    echo "replayed_source_terms=1757"
    echo "source_group_histogram_1_to_8=22,23,5,21,21,21,1,169"
    echo "descriptor_stall_cycles=77"
    echo "group_stall_cycles=55"
    echo "both_windows_closed_cycles=175"
    echo "release_refill_hits=1"
    echo "window_replace_hits=3"
    echo "cross_descriptor_groups=40"
    echo "consecutive_group_hits=224"
    echo "stage_windows=2,4,8,8"
    echo "stage_raw_descriptor_extents=4,8,16,32"
    echo "max_two_buffer_bitmap_bits=1536"
    echo "protocol_attacks=4"
    echo "extent_overflow_attacks=1"
    echo "same_cycle_header_rearm=1"
    echo "sva_coverpoints_nonzero=21"
    echo "global_topk_sort=false"
    echo "bank_id_payload=false"
    echo "bank_to_prefix_packing=false"
    echo "fixed_bank_valid_mask=true"
    echo "token_directory=true"
    echo "native_or_preindexed_source_required=true"
    echo "weight_sram_response=false"
    echo "arithmetic=false"
    echo "complete_fc2=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m184_fc2_dual_window_k8_fixed_bank_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M184 FC2 dual-window K8 fixed-bank frontend VCS sealed at $task_run"
