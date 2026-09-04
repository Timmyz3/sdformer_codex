#!/usr/bin/env bash
set -euo pipefail

task_review="reviews/m123_w384_signed19_forwarding_accumulator_independent_hammer_r1_20260824"
task_run="$task_review/m120_integrated_vcs"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M123/M120 integrated review evidence" >&2
    exit 2
fi
mkdir "$task_run"

declare -A task_expected=(
    ["rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv"]="2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337"
    ["rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv"]="f37ed1f9ea1f6c26c80327c620e219bbfb3863f29337c754d50ae85068236316"
    ["verif_m120/m120_pwp_tail_mapper_signed19_accumulator_island_assertions.sv"]="89d6d0f8a71e60b2f2b5daa5152ca230bc935aa0390ba4ca858612186d94c908"
    ["$task_review/m118_name_m123_forwarding_shim.sv"]="fadbb8068ede7d673bb7a936bca2c9a2f3268d3849658b5810824c2c40284cb6"
    ["$task_review/m123_m120_integrated_review.f"]="2fc53b2c17cff8520b2b6824a3d391f6e4bd4dcdb8bde003121e09cb4fdc107f"
    ["$task_review/tb_m120_with_m123_integrated_hammer.sv"]="fda4b5dcf9ac810dc6498c2f50241f4b2ef4b54170aac5e05e01187dce53ce79"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_actual="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s actual=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_actual" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_actual" == "${task_expected[$task_path]}" ]]
done

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_review/m123_m120_integrated_review.f" \
    -top tb_m120_with_m123_integrated_hammer \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert -cm_dir "$task_run/simv.vdb" \
    > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -Eq '^PASS M123 integrated M120 counterexample commercial_vcs=true review_only_substitution=true positive_loads=768 positive_weight_reads=768 positive_events=1024 positive_updates=1024 positive_writes=1024 positive_ii1_pairs=768 positive_rw_overlap=768 mapper_lane_checks=98304 tail_bypass_hits=256 negate_events=512 commits=6144 commit_lane_checks=589824 commit_stalls=[1-9][0-9]* stall_releases=[1-9][0-9]* lazy_clear_windows=2 address_minmax=true int8_endpoints=true malformed_beat_attacks=1 malformed_key_attacks=1 early_end_attacks=1 older_update_drain_checks=1 same_address_events_accepted=2 same_address_mapped_updates=2 same_address_updates_written=2 same_address_lane_checks=96 same_address_accept_then_loss_closed=true retry_events_accepted=3 retry_updates_written=3 retry_dedup_absent=true reset_events_accepted=1 reset_updates_written=0 reset_exact_once_undefined=true accumulator_bytes=700416 combined_bytes=725416 directed_legal_and_same_address_exact_once=true heldout_duplicate_retry_reset_exact_once=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false$' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_loads_tail_event, .* 256 match' \
        'cp_event_update_chain, .* 1024 match' \
        'cp_update_ii1, .* 768 match' \
        'cp_lane_read_write_overlap, .* 768 match' \
        'cp_commit_stall_release, .* 1425 match' \
        'cp_full_window, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo 'status=PASS_M123_REVIEW_ONLY_M120_INTEGRATED_COUNTEREXAMPLE_CLOSED'
    echo 'production_m120_wrapper_unmodified=true'
    echo 'review_only_name_shim=true'
    echo 'positive_events_updates_writes=1024/1024/1024'
    echo 'same_address_events_updates_writes=2/2/2'
    echo 'same_address_lane_checks=96'
    echo 'retry_deduplication=false'
    echo 'reset_recovery=false'
    echo 'foundry_macro=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run/preflight_sha_checks.txt" "$task_run/compile.raw.log" \
    "$task_run/sim.raw.log" "$task_run/assert.report" \
    "$task_run/RUN_COMPLETE.txt" > "$task_run/output.sha256"
echo "PASS M123 review-only M120 integrated counterexample replay"
