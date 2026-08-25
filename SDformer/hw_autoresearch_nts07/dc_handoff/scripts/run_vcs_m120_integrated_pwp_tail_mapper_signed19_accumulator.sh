#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M120 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
on_exit() {
    local task_rc="$?"
    if [[ "$task_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$task_rc"
        } > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$task_hw_root"
task_wrapper="rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv"
task_sva="verif_m120/m120_pwp_tail_mapper_signed19_accumulator_island_assertions.sv"
task_tb="tb_m120/tb_m120_pwp_tail_mapper_signed19_accumulator_island.sv"
task_files="dc_handoff/filelists/date_m120_integrated_pwp_tail_mapper_signed19_accumulator_directed_vcs.f"
task_contract="contracts/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_contract_r1_20260824.json"
task_mapper="rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv"
task_accumulator="rtl_m118/m118_w384_signed19_accumulator_frontend.sv"
task_adapter="rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv"
task_m119_contract="contracts/m119_pwp_weight_tail_bypass_mapper_vcs_contract_r1_20260824.json"
task_m119_receipt="dc_handoff/runs/m119_pwp_weight_tail_bypass_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m118_contract="contracts/m118_w384_signed19_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_m118_receipt="dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r2_sealed_20260824/RUN_COMPLETE.txt"
task_m115_review_manifest="reviews/m115r2_pwp_prefix_coefficient_width_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_wrapper"]="f37ed1f9ea1f6c26c80327c620e219bbfb3863f29337c754d50ae85068236316"
    ["$task_sva"]="89d6d0f8a71e60b2f2b5daa5152ca230bc935aa0390ba4ca858612186d94c908"
    ["$task_tb"]="1b3d3ae2b060573ca516906b20c968c17608791f1aef0edaf5ffe82b05c3a758"
    ["$task_files"]="80ca152b62e1dbfae4de9ce7bc5fca63fbc8473ab51f33ee0890defe5f32e982"
    ["$task_contract"]="0ce38d33e4885bd3c5b79f81117acec54df6e0e8b753359b172b6031403a947a"
    ["$task_mapper"]="2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337"
    ["$task_accumulator"]="0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482"
    ["$task_adapter"]="cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af"
    ["$task_m119_contract"]="5ccdebb50ae7149bd51a7b767ae3176758c9617dc6e751d04255814e001e3cd8"
    ["$task_m119_receipt"]="88b36867e9ba4cd67e3d1ff8265351de40a54a42843e4b4cf9c4e7f2a2c9d423"
    ["$task_m118_contract"]="c79f55a15e03bbf26c22e9da2f0eb35d53b1a9795ab02b24a6b3c951c729903e"
    ["$task_m118_receipt"]="f45baa3c322a439377aa9c0c3e919440020294c9392b81343c7fae1bc1e605ff"
    ["$task_m115_review_manifest"]="d0c7067f599c8e24b77099ffec4624c533bbbc098c1d5123bf444ef467237790"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M120 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" \
    -top tb_m120_pwp_tail_mapper_signed19_accumulator_island \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M120 integrated PWP tail mapper signed19 accumulator island VCS windows=2 groups=256 weight_loads=768 weight_reads=768 events=1024 mapped_updates=1024 accumulator_writes=1024 mapped_ii1_pairs=768 lane_rw_overlap=768 tail_bypass_first_events=256 negate_events=512 commits=6144 commit_lane_checks=589824 commit_stalls=734 protocol_attacks=1 weight_port_bits=256 weight_read_latency=1 accumulator_lanes=96 accumulator_bits=19 accumulator_bytes=700416 exact_once_directed=true m117_scheduler_integrated=false heldout_trace_replay=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_loads_tail_event, .* 256 match' \
        'cp_event_update_chain, .* 1024 match' \
        'cp_update_ii1, .* 768 match' \
        'cp_lane_read_write_overlap, .* 768 match' \
        'cp_commit_stall_release, .* 699 match' \
        'cp_full_window, .* 2 match' \
        'cp_fault, .* 1 match' \
        'cp_busy, .* 8680 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M120_INTEGRATED_PWP_TAIL_MAPPER_SIGNED19_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "windows=2"
    echo "active_keys=256"
    echo "fixed_weight_read_latency_cycles=1"
    echo "weight_port_bits=256"
    echo "weight_beats_per_key=3"
    echo "accepted_events=1024"
    echo "mapped_accumulator_updates=1024"
    echo "accumulator_writes=1024"
    echo "tail_bypassed_first_events=256"
    echo "uncounted_tail_bubbles=0_in_directed_scope"
    echo "commit_vectors=6144"
    echo "commit_lane_checks=589824"
    echo "accumulator_signed_bits=19"
    echo "logical_accumulator_bytes=700416"
    echo "directed_event_to_update_exact_once=true"
    echo "m117_scheduler_integrated=false"
    echo "heldout_trace_duplicate_retry_escape_replay=false"
    echo "foundry_weight_sram_macro=false"
    echo "foundry_accumulator_sram_macro=false"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m120_integrated_pwp_tail_mapper_signed19_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M120 integrated PWP tail mapper signed19 accumulator VCS sealed at $task_run"
