#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m177_fc2_indexed_nonzero96_k4_replay_frontend_vcs_r2_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M177 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m177/m177_fc2_indexed_nonzero96_k4_replay_frontend.sv"]="ef0e9f6075420f404dcb7617c74e7cc2a36af6db28cada0853c587432703a21f"
    ["verif_m177/m177_fc2_indexed_nonzero96_k4_replay_frontend_assertions.sv"]="b57289f10f3b9c0c8427a9329a46ca6d92e226573caa1451e7e5a6912f9d6b98"
    ["tb_m177/tb_m177_fc2_indexed_nonzero96_k4_replay_frontend.sv"]="5b96f9798bc7adbf7d97167d79af93f5f666cec1c5c03863b5eb82ef2590bf4b"
    ["dc_handoff/filelists/date_m177_fc2_indexed_nonzero96_k4_replay_frontend_directed_vcs.f"]="e37db9805747aefb5adff89e7c800fefde7bb3da6b6bb028e9fba08b7cb42f5f"
    ["contracts/m177_fc2_indexed_nonzero96_k4_replay_frontend_vcs_contract_r1_20260824.json"]="7332a0975a38b16551bf43028e2bb559b1a5d0b7705c85d2c3c2f397327e732a"
    ["contracts/m177_r1_structural_timing_loop_correction_overlay_r2_20260824.json"]="941978606fc89171d7456f78630cb78fe64c9813e4f9e4da9c9b4b9645abfba9"
    ["contracts/m176_r1_beat_index_and_producer_admission_overlay_r1_20260824.json"]="c19ef872a5ca507bc29e2fe625bfe2700e07e671fd1d50ece8ff5342c1396dc9"
    ["results/m176_independent_hammer_review_r1_20260824/manifest.sha256"]="bdc6ae8c0ba3b9ce5712f31107aef618385c09941358a8d460ae3fd898a6ee74"
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
    -f dc_handoff/filelists/date_m177_fc2_indexed_nonzero96_k4_replay_frontend_directed_vcs.f \
    -top tb_m177_fc2_indexed_nonzero96_k4_replay_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=177024 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M177 FC2 indexed-nonzero96 K4 replay frontend VCS raw_beat_opportunities=58 zero_beats_elided=22 payload_descriptors=36 eot_descriptors=7 descriptors=43 tokens=7 bitmap_events=330 unique_groups=114 unique_source_terms=330 replayed_group_results=519 replayed_source_terms=1346 output_stall_cycles=124 prefetch_accepts=1 indexed_gap_accepts=17 eot_with_pending_accepts=5 stage0_consecutive_group_hits=26 consecutive_group_stream_hits=415 same_cycle_token_rearms=1 output_block_extents=1,2,4,8 protocol_attacks=4 legal_done_eot_rearm=1 bitmap_width_bits=96 beat_index_bits=5 stage_extents=4,8,16,32 explicit_eot=true future_prediction=false cross_beat_grouping=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_four_source_group cp_single_source_group \
        cp_same_cycle_group_replace cp_raw_beat_prefetch_during_replay \
        cp_group_stall_then_accept cp_indexed_gap \
        cp_eot_with_pending_work cp_stage0_final cp_stage1_final \
        cp_stage2_final cp_stage3_final cp_zero_token_done \
        cp_nonzero_token_done cp_same_cycle_token_rearm \
        cp_last_seen_with_pending_work; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M177_FC2_INDEXED_NONZERO96_K4_REPLAY_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "precommitted_random_seed=177024"
    echo "raw_beat_opportunities=58"
    echo "zero_beats_elided=22"
    echo "payload_descriptors=36"
    echo "eot_descriptors=7"
    echo "descriptors=43"
    echo "tokens=7"
    echo "bitmap_events=330"
    echo "unique_groups=114"
    echo "unique_source_terms=330"
    echo "replayed_group_results=519"
    echo "replayed_source_terms=1346"
    echo "output_stall_cycles=124"
    echo "prefetch_accepts=1"
    echo "indexed_gap_accepts=17"
    echo "eot_with_pending_accepts=5"
    echo "stage0_consecutive_group_hits=26"
    echo "consecutive_group_stream_hits=415"
    echo "same_cycle_token_rearms=1"
    echo "protocol_attacks=4"
    echo "legal_done_eot_rearm=1"
    echo "sva_coverpoints=15/15"
    echo "sva_failures=0"
    echo "beat_index_times_12=true"
    echo "output_block_extents=1,2,4,8"
    echo "stage_extents_in_beats=4,8,16,32"
    echo "native_or_preindexed_source_required=true"
    echo "posthoc_scanner_speedup=false"
    echo "descriptor_memory_delivery=false"
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
sha256sum "dc_handoff/scripts/run_vcs_m177_fc2_indexed_nonzero96_k4_replay_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M177 FC2 indexed-nonzero96 K4 replay frontend VCS sealed at $task_run"
