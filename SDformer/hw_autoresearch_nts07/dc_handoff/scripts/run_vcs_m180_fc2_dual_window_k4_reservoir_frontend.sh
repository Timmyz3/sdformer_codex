#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m180_fc2_dual_window_k4_reservoir_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M180 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m180/m180_fc2_dual_window_k4_reservoir_frontend.sv"]="83e72b7bd71f059a1e47dedaaf060d37d8d416979bf24271973910f31ac20a6c"
    ["verif_m180/m180_fc2_dual_window_k4_reservoir_frontend_assertions.sv"]="b1ecc139478e62b38b7bafa3e24e0070a1ff761746102ff0308836b324d4cc69"
    ["tb_m180/tb_m180_fc2_dual_window_k4_reservoir_frontend.sv"]="0c1fc0f773cee61ea17231fed88a32836d8fb8443e3fecada0723f16892c6993"
    ["dc_handoff/filelists/date_m180_fc2_dual_window_k4_reservoir_frontend_directed_vcs.f"]="1f00faa991269b5a6e8f85cbe2ed31f3051feae1d0b3434a3f8f41cc2b57d186"
    ["contracts/m180_fc2_dual_window_k4_reservoir_frontend_vcs_contract_r1_20260824.json"]="8a6cd6ccf2e0f3653d7fbde36f8a480e68f4b9fec7d21c0cc70c063f95b8b02d"
    ["contracts/m179_r1_independent_review_baseline_and_selection_overlay_r1_20260824.json"]="6e8c0b7db0644b6a22545c9660828e311ef4e90b5d6f724b721de69373d40542"
    ["results/m179_independent_hammer_review_r1_20260824/manifest.sha256"]="31f3e4baddcf1d5478d2cb011875154918d6fd9c998bf199c395f52561c3277b"
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
    -f dc_handoff/filelists/date_m180_fc2_dual_window_k4_reservoir_frontend_directed_vcs.f \
    -top tb_m180_fc2_dual_window_k4_reservoir_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=180024 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M180 FC2 dual-window K4 reservoir frontend VCS headers=7 descriptors=52 tokens=7 bitmap_events=263 unique_groups=73 unique_source_terms=263 replayed_group_results=418 replayed_source_terms=1597 descriptor_stall_cycles=[1-9][0-9]* group_stall_cycles=[1-9][0-9]* both_windows_closed_cycles=[1-9][0-9]* release_refill_hits=[1-9][0-9]* window_replace_hits=[1-9][0-9]* cross_descriptor_groups=[1-9][0-9]* consecutive_group_hits=[1-9][0-9]* stage_windows=2,4,8,8 max_two_buffer_payload_bits=1536 protocol_attacks=3 same_cycle_header_rearm=1 token_directory=true native_or_preindexed_source_required=true posthoc_scanner_speedup=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_one_source_group cp_two_source_group \
        cp_three_source_group cp_four_source_group \
        cp_group_stall_then_accept cp_descriptor_backpressure \
        cp_cross_descriptor_group cp_window_to_window_replace \
        cp_release_and_refill cp_both_windows_closed cp_stage0 cp_stage1 \
        cp_stage2 cp_stage3 cp_zero_token cp_nonzero_token \
        cp_same_cycle_header_rearm; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

task_pass_line="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_descriptor_stalls="$(sed -n 's/.* descriptor_stall_cycles=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_group_stalls="$(sed -n 's/.* group_stall_cycles=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_both_closed="$(sed -n 's/.* both_windows_closed_cycles=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_release_refill="$(sed -n 's/.* release_refill_hits=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_window_replace="$(sed -n 's/.* window_replace_hits=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_cross_descriptor="$(sed -n 's/.* cross_descriptor_groups=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"
task_consecutive="$(sed -n 's/.* consecutive_group_hits=\([0-9][0-9]*\).*/\1/p' <<< "$task_pass_line")"

{
    echo "status=PASS_M180_FC2_DUAL_WINDOW_K4_RESERVOIR_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "precommitted_random_seed=180024"
    echo "headers=7"
    echo "descriptors=52"
    echo "tokens=7"
    echo "bitmap_events=263"
    echo "unique_groups=73"
    echo "unique_source_terms=263"
    echo "replayed_group_results=418"
    echo "replayed_source_terms=1597"
    echo "descriptor_stall_cycles=$task_descriptor_stalls"
    echo "group_stall_cycles=$task_group_stalls"
    echo "both_windows_closed_cycles=$task_both_closed"
    echo "release_refill_hits=$task_release_refill"
    echo "window_replace_hits=$task_window_replace"
    echo "cross_descriptor_groups=$task_cross_descriptor"
    echo "consecutive_group_hits=$task_consecutive"
    echo "protocol_attacks=3"
    echo "same_cycle_header_rearm=1"
    echo "sva_coverpoints=17/17"
    echo "sva_failures=0"
    echo "stage_window_depths=2,4,8,8"
    echo "maximum_two_buffer_bitmap_payload_bits_without_metadata=1536"
    echo "constructive_global_top4_selector=true"
    echo "native_or_preindexed_source_required=true"
    echo "token_directory_interface=true"
    echo "token_directory_generation=false"
    echo "posthoc_scanner_speedup=false"
    echo "exact_payload_cycles=false"
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
sha256sum "dc_handoff/scripts/run_vcs_m180_fc2_dual_window_k4_reservoir_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M180 FC2 dual-window K4 reservoir frontend VCS sealed at $task_run"
