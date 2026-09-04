#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M188 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m188/m188_fc2_dual_window_k7_weakest_bank_defer_frontend.sv"]="e5401fce15191b261ccf5c413a221d2e9042daf16c9e0773654b4f27b851f0ee"
    ["verif_m188/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_assertions.sv"]="bc89f0bec31eabda25af92d5cda2be92a246b23a8d94b333bd7e038e4d37a490"
    ["tb_m188/tb_m188_fc2_dual_window_k7_weakest_bank_defer_frontend.sv"]="7bac707a60cce397fda5b70bac7a11023b4d23cfc091db61d284384a42b0efc3"
    ["dc_handoff/filelists/date_m188_fc2_dual_window_k7_weakest_bank_defer_frontend_directed_vcs.f"]="c3430373ce39495072594a8f3e4081864f8e37b2a62cc5b1d4ba79989835f601"
    ["contracts/m188_fc2_dual_window_k7_weakest_bank_defer_frontend_vcs_contract_r1_20260825.json"]="ec1473bb7930cd7b4d274099deb7a570b96b8d07967e082b502c9770cf055fe1"
    ["contracts/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse_contract_r1_20260825.json"]="ad1316e92eedc2b6ca71cc21ca0795f401bbe6f9436652a0e01077a8fc5a1f9d"
    ["results/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse_r1_20260825/m187_h67_fc2_fixed_bank_kcap_exact_payload_dse.json"]="411e61ff9c5e0a8b4ff27e86cf15d5ae87b8ef523fc25e43b0939417d65cd201"
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
    -f dc_handoff/filelists/date_m188_fc2_dual_window_k7_weakest_bank_defer_frontend_directed_vcs.f \
    -top tb_m188_fc2_dual_window_k7_weakest_bank_defer_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=188025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M188 FC2 dual-window K7 weakest-bank-defer frontend VCS headers=15 descriptors=60 tokens=15 bitmap_events=320 unique_groups=59 unique_source_terms=320 replayed_group_results=292 replayed_source_terms=1757 one_source_groups=15 two_source_groups=35 three_source_groups=1 four_source_groups=5 five_source_groups=1 six_source_groups=1 seven_source_groups=234 eight_source_groups=0 descriptor_stall_cycles=75 group_stall_cycles=54 both_windows_closed_cycles=182 release_refill_hits=1 window_replace_hits=3 cross_descriptor_groups=41 consecutive_group_hits=237 stage_windows=2,4,8,8 max_two_buffer_bitmap_bits=1536 protocol_attacks=4 extent_overflow_attacks=1 same_cycle_header_rearm=1 maximum_sources_per_group=7 weakest_bank_defer=true global_topk_sort=false bank_id_payload=false fixed_bank_valid_mask=true token_directory=true native_or_preindexed_source_required=true posthoc_scanner_speedup=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_one_source_group cp_two_source_group \
        cp_three_source_group cp_four_source_group cp_five_source_group \
        cp_six_source_group cp_seven_source_group cp_weakest_bank_defer \
        cp_group_stall_then_accept cp_descriptor_backpressure \
        cp_cross_descriptor_group cp_window_to_window_replace \
        cp_release_and_refill cp_both_windows_closed cp_stage0 cp_stage1 \
        cp_stage2 cp_stage3 cp_zero_token cp_nonzero_token \
        cp_same_cycle_header_rearm; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M188_FC2_DUAL_WINDOW_K7_WEAKEST_BANK_DEFER_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=188025"
    echo "headers=15"
    echo "descriptors=60"
    echo "tokens=15"
    echo "bitmap_events=320"
    echo "unique_groups=59"
    echo "unique_source_terms=320"
    echo "replayed_group_results=292"
    echo "replayed_source_terms=1757"
    echo "source_group_histogram_1_to_8=15,35,1,5,1,1,234,0"
    echo "descriptor_stall_cycles=75"
    echo "group_stall_cycles=54"
    echo "both_windows_closed_cycles=182"
    echo "release_refill_hits=1"
    echo "window_replace_hits=3"
    echo "cross_descriptor_groups=41"
    echo "consecutive_group_hits=237"
    echo "stage_windows=2,4,8,8"
    echo "stage_raw_descriptor_extents=4,8,16,32"
    echo "max_two_buffer_bitmap_bits=1536"
    echo "protocol_attacks=4"
    echo "extent_overflow_attacks=1"
    echo "same_cycle_header_rearm=1"
    echo "sva_coverpoints_nonzero=21"
    echo "maximum_sources_per_group=7"
    echo "weakest_bank_defer=true"
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
sha256sum "dc_handoff/scripts/run_vcs_m188_fc2_dual_window_k7_weakest_bank_defer_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M188 FC2 dual-window K7 weakest-bank-defer frontend VCS sealed at $task_run"
