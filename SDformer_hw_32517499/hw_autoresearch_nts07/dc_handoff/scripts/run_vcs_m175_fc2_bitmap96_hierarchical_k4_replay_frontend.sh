#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m175_fc2_bitmap96_hierarchical_k4_replay_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M175 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m175/m175_fc2_bitmap96_hierarchical_k4_replay_frontend.sv"]="823389677c032ffe675533b904de41671160aaf6672f2effd685cce836f37fbd"
    ["verif_m175/m175_fc2_bitmap96_hierarchical_k4_replay_frontend_assertions.sv"]="3da882e5545d476c1ebdef3c455f11ea7ff6619fd6afe748e0aa02e82fded570"
    ["tb_m175/tb_m175_fc2_bitmap96_hierarchical_k4_replay_frontend.sv"]="0f3adc5f20ec2f9af46c10729c6fdde58168defe3e80a1e4f9874a3036691718"
    ["dc_handoff/filelists/date_m175_fc2_bitmap96_hierarchical_k4_replay_frontend_directed_vcs.f"]="b0aad188e919a76db8005363e006fd277d4fb44590f48f6dacbe0ec8e8a93e31"
    ["contracts/m175_fc2_bitmap96_hierarchical_k4_replay_frontend_vcs_contract_r1_20260824.json"]="d23ad3ed44acc779138f42d947a0fad387113f40a9d04281858f1170c8c41fc7"
    ["results/m173_h67_fc2_scan_width_exact_payload_dse_r1_20260824/manifest.sha256"]="a11cfecff84c8d93bef70f7623b3f45615c87293ecb0e5f136b0cd64db41813a"
    ["results/m173_dse_independent_hammer_review_r1_20260824/manifest.sha256"]="90b6f462f5ee03634020a28a6b2eb055143d426f6866d7ec4b8c0fcb154c7095"
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
    -f dc_handoff/filelists/date_m175_fc2_bitmap96_hierarchical_k4_replay_frontend_directed_vcs.f \
    -top tb_m175_fc2_bitmap96_hierarchical_k4_replay_frontend \
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
task_pass_regex='^PASS M175 FC2 bitmap96 hierarchical K4 replay frontend VCS scan_beats=50 tokens=7 bitmap_events=367 unique_groups=148 unique_source_terms=367 replayed_group_results=644 replayed_source_terms=1519 zero_scan_beats=9 zero_tokens=1 output_stall_cycles=150 prefetch_accepts=5 stage0_consecutive_group_hits=37 consecutive_group_stream_hits=516 same_cycle_token_rearms=1 output_block_extents=1,2,4,8 protocol_attacks=1 scan_width_bits=96 max_sources_per_group=4 shared_hierarchical_selector=true source_group_held_across_output_blocks=true one_raw_beat_prefetch=true weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_four_source_group cp_single_source_group \
        cp_same_cycle_group_replace cp_raw_beat_prefetch_during_replay \
        cp_group_stall_then_accept cp_stage0_final cp_stage1_final \
        cp_stage2_final cp_stage3_final cp_zero_token_done \
        cp_nonzero_token_done cp_same_cycle_token_rearm \
        cp_last_seen_with_pending_work; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M175_FC2_BITMAP96_HIERARCHICAL_K4_REPLAY_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "scan_beats=50"
    echo "tokens=7"
    echo "bitmap_events=367"
    echo "unique_groups=148"
    echo "unique_source_terms=367"
    echo "replayed_group_results=644"
    echo "replayed_source_terms=1519"
    echo "zero_scan_beats=9"
    echo "zero_tokens=1"
    echo "output_stall_cycles=150"
    echo "prefetch_accepts=5"
    echo "stage0_consecutive_group_hits=37"
    echo "consecutive_group_stream_hits=516"
    echo "same_cycle_token_rearms=1"
    echo "output_block_extents=1,2,4,8"
    echo "scan_width_bits=96"
    echo "maximum_sources_per_group=4"
    echo "shared_hierarchical_selector=true"
    echo "source_group_held_across_output_blocks=true"
    echo "raw_bitmap_prefetch_entries=1"
    echo "descriptor_conservation=true"
    echo "continuous_token_stream_frontend=true"
    echo "protocol_attacks=1"
    echo "weight_sram_response=false"
    echo "arithmetic=false"
    echo "complete_fc2=false"
    echo "exact_payload_cycles=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m175_fc2_bitmap96_hierarchical_k4_replay_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M175 FC2 bitmap96 hierarchical K4 replay frontend VCS sealed at $task_run"
