#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m171_fc2_bitmap_k4_group_replay_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M171 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m171/m171_fc2_bitmap_k4_group_replay_frontend.sv"]="a549bc95b54521a3123266ae3c5fc1accbfd2d7a80ccc1891e76f43a4f99ba72"
    ["verif_m171/m171_fc2_bitmap_k4_group_replay_frontend_assertions.sv"]="e2a41c30da7b3d40879688388b026d9f77eef2b9557da21bfde6bbccd200cb13"
    ["tb_m171/tb_m171_fc2_bitmap_k4_group_replay_frontend.sv"]="9953ee52e16f62b5021a4dfa8e43202efab6d98c48afdd284e433dfc7f67c27b"
    ["dc_handoff/filelists/date_m171_fc2_bitmap_k4_group_replay_frontend_directed_vcs.f"]="fe75e4b77b5a5056b00bb7fb88e39db78aaa684e42f38eda546b8d5066366a9f"
    ["contracts/m171_fc2_bitmap_k4_group_replay_frontend_vcs_contract_r1_20260824.json"]="7d376fed5c8e60cb8cb861cb29531872a18ad4f6c45948ba3fdd1fb341364341"
    ["results/m168_dse_independent_hammer_review_r1_20260824/manifest.sha256"]="60a4cb094b3ac318fb5b6fea76c0c770c927c065854eb351d96f7889cbd694db"
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
    -f dc_handoff/filelists/date_m171_fc2_bitmap_k4_group_replay_frontend_directed_vcs.f \
    -top tb_m171_fc2_bitmap_k4_group_replay_frontend \
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
task_pass_regex='^PASS M171 FC2 bitmap K4 group-replay frontend VCS scan_beats=96 tokens=5 bitmap_events=433 unique_groups=144 unique_source_terms=433 replayed_group_results=642 replayed_source_terms=2039 zero_scan_beats=16 zero_tokens=1 output_stall_cycles=154 prefetch_accepts=9 stage0_consecutive_group_hits=27 same_cycle_group_stream_hits=513 output_block_extents=1,2,4,8 protocol_attacks=1 scan_width_bits=64 max_sources_per_group=4 source_group_held_across_output_blocks=true one_raw_beat_prefetch=true weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_four_source_group cp_single_source_group \
        cp_same_cycle_group_replace cp_raw_beat_prefetch_during_replay \
        cp_group_stall_then_accept cp_stage0_final cp_stage1_final \
        cp_stage2_final cp_stage3_final cp_zero_token_done \
        cp_nonzero_token_done cp_last_seen_with_pending_work; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M171_FC2_BITMAP_K4_GROUP_REPLAY_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "scan_beats=96"
    echo "tokens=5"
    echo "bitmap_events=433"
    echo "unique_groups=144"
    echo "unique_source_terms=433"
    echo "replayed_group_results=642"
    echo "replayed_source_terms=2039"
    echo "zero_scan_beats=16"
    echo "zero_tokens=1"
    echo "output_stall_cycles=154"
    echo "prefetch_accepts=9"
    echo "stage0_consecutive_group_hits=27"
    echo "same_cycle_group_stream_hits=513"
    echo "output_block_extents=1,2,4,8"
    echo "scan_width_bits=64"
    echo "maximum_sources_per_group=4"
    echo "source_group_held_across_output_blocks=true"
    echo "raw_bitmap_prefetch_entries=1"
    echo "descriptor_conservation=true"
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
sha256sum "dc_handoff/scripts/run_vcs_m171_fc2_bitmap_k4_group_replay_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M171 FC2 bitmap K4 group-replay frontend VCS sealed at $task_run"
