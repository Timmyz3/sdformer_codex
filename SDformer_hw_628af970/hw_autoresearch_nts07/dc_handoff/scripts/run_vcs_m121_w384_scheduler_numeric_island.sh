#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m121_w384_scheduler_numeric_island_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M121 sealed VCS run: $task_run" >&2
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
task_wrapper="rtl_m121/m121_w384_scheduler_numeric_island.sv"
task_sva="verif_m121/m121_w384_scheduler_numeric_island_assertions.sv"
task_tb="tb_m121/tb_m121_w384_scheduler_numeric_island.sv"
task_files="dc_handoff/filelists/date_m121_w384_scheduler_numeric_island_directed_vcs.f"
task_contract="contracts/m121_w384_scheduler_numeric_island_vcs_contract_r1_20260824.json"
task_scheduler="rtl_m117/m117_w384_prefetch_transpose_scheduler.sv"
task_mapper="rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv"
task_accumulator="rtl_m118/m118_w384_signed19_accumulator_frontend.sv"
task_adapter="rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv"
task_numeric="rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv"
task_m117_contract="contracts/m117_w384_prefetch_transpose_vcs_contract_r1_20260824.json"
task_m117_receipt="dc_handoff/runs/m117_w384_prefetch_transpose_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m120_contract="contracts/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_contract_r1_20260824.json"
task_m120_receipt="dc_handoff/runs/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m117_review="reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824/manifest.sha256"
task_m119_review="reviews/m119_pwp_weight_tail_bypass_mapper_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_wrapper"]="a448e4cc530a1885f92e413e74f2e9b06df7a5fc5338cc5771f1130bf746be85"
    ["$task_sva"]="84801c33ed2c59d4cc1404cfd9339e4903d01c407bd1e8c7ff2d301470db41a8"
    ["$task_tb"]="23304b03a148daa8c368bebea9b0baff525d206e768357c10fea10be005a39a5"
    ["$task_files"]="0a65fb3ddda0bf430a61ca0d9025688f1ce93404fff66b47c8bca9ff09687d65"
    ["$task_contract"]="a4a2d2aac9838c30cf28c841add479472e3287db087763dc6b1535cc5bcd10ad"
    ["$task_scheduler"]="4e640770349fa2d95ac09731efe7f8587d8bb108bd89169c204200cf41f3983a"
    ["$task_mapper"]="2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337"
    ["$task_accumulator"]="0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482"
    ["$task_adapter"]="cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af"
    ["$task_numeric"]="f37ed1f9ea1f6c26c80327c620e219bbfb3863f29337c754d50ae85068236316"
    ["$task_m117_contract"]="b327f0e14d83ecf1df18fcbedb2d5986a1b53971b54a972892f6552b44ca1fef"
    ["$task_m117_receipt"]="92f991f06f8a4d80ef2fc0d2fdd96cb473a7b6a2e29e687627ac3f531814c927"
    ["$task_m120_contract"]="0ce38d33e4885bd3c5b79f81117acec54df6e0e8b753359b172b6031403a947a"
    ["$task_m120_receipt"]="1cce8b2e7a09bd193baeb703d25e2b25e1d263f80d3cd273f4bedd1a35b032ac"
    ["$task_m117_review"]="3fc112ae72769f4dbba8aed8450fe2b840327292a112825956be8160e93137b2"
    ["$task_m119_review"]="b73e8a8a6d23a12edc62300ca6ad04d5ccd128e89e6e085a04832621d1e43abf"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M121 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m121_w384_scheduler_numeric_island \
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
grep -qx 'PASS M121 W384 scheduler numeric island VCS descriptors=2 ingress_events=98304 active_keys=256 prefetches=256 service_tokens=99072 weight_loads=768 service_events=98304 weight_reads=768 tail_bypass_first_events=256 zero_bubble_key_transitions=254 downstream_backpressure_cycles=0 mapped_updates=98304 mapped_ii1_pairs=98048 accumulator_writes=98304 lane_rw_overlap=98048 descriptor_done=2 commits=3072 commit_lane_checks=294912 commit_stalls=366 protocol_attacks=1 weight_port_bits=256 weight_read_latency=1 accumulator_lanes=96 accumulator_bits=19 accumulator_bytes=700416 directed_end_to_end_service_island=true heldout_trace_replay=false foundry_sram_macro=false module_cycle_projection=2.53546204172554 physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_loads_tail_event, .* 256 match' \
        'cp_event_update_chain, .* 98304 match' \
        'cp_zero_bubble_key_transition, .* 254 match' \
        'cp_update_ii1, .* 98048 match' \
        'cp_lane_rw_overlap, .* 98048 match' \
        'cp_descriptor_done, .* 2 match' \
        'cp_full_commit, .* 1 match' \
        'cp_numeric_fault, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M121_W384_SCHEDULER_NUMERIC_ISLAND_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "descriptors=2"
    echo "ingress_events=98304"
    echo "active_keys=256"
    echo "weight_prefetches=256"
    echo "service_tokens=99072"
    echo "weight_load_tokens=768"
    echo "service_event_tokens=98304"
    echo "mapped_accumulator_updates=98304"
    echo "accumulator_writes=98304"
    echo "tail_bypassed_first_events=256"
    echo "zero_bubble_nonfinal_key_transitions=254"
    echo "numeric_downstream_backpressure_cycles=0"
    echo "scheduler_numeric_accept_agreement_asserted=true"
    echo "commit_vectors=3072"
    echo "commit_lane_checks=294912"
    echo "accumulator_signed_bits=19"
    echo "logical_accumulator_bytes=700416"
    echo "heldout_trace_duplicate_retry_escape_replay=false"
    echo "foundry_weight_sram_macro=false"
    echo "foundry_accumulator_sram_macro=false"
    echo "m109_r2_module_cycle_projection=2.53546204172554"
    echo "module_cycle_projection_admitted=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m121_w384_scheduler_numeric_island.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M121 W384 scheduler numeric island VCS sealed at $task_run"
