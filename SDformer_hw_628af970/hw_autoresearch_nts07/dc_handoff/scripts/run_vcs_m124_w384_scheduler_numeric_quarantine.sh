#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m124_w384_scheduler_numeric_quarantine_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M124 sealed VCS run: $task_run" >&2
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
task_top="rtl_m124/m124_w384_scheduler_numeric_quarantine_island.sv"
task_numeric="rtl_m124/m124_pwp_tail_mapper_signed19_forwarding_accumulator_island.sv"
task_sva="verif_m124/m124_w384_scheduler_numeric_quarantine_island_assertions.sv"
task_tb="tb_m124/tb_m124_w384_scheduler_numeric_quarantine_island.sv"
task_files="dc_handoff/filelists/date_m124_w384_scheduler_numeric_quarantine_directed_vcs.f"
task_contract="contracts/m124_w384_scheduler_numeric_quarantine_vcs_contract_r1_20260824.json"
task_scheduler="rtl_m117/m117_w384_prefetch_transpose_scheduler.sv"
task_mapper="rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv"
task_accumulator="rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"
task_adapter="rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"
task_revocation="contracts/m121_r1_combined_fail_closed_claim_revocation_r1_20260824.json"
task_m123_contract="contracts/m123_w384_signed19_forwarding_accumulator_vcs_contract_r1_20260824.json"
task_m123_receipt="dc_handoff/runs/m123_w384_signed19_forwarding_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m121_review="reviews/m121_w384_scheduler_numeric_island_independent_hammer_r1_20260824/manifest.sha256"
task_m120_review="reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_top"]="e0077bac9fd23aa06002525e198b18133986b0eee942fcf270169e411b1f9606"
    ["$task_numeric"]="32713af2438d29b88160adef8da9c7fbb7b5bf964112ae42351dfd8b7011388e"
    ["$task_sva"]="1407ca52d277864436f7f92e5bd1602993a6d2aee0a031d07b59f9c180e24783"
    ["$task_tb"]="07f049a6844472de04831201f0e46459a24e0e309fbf6656604b328c13494efe"
    ["$task_files"]="66d13a6945ced038b6770f3e7a7574413637a2f1ff468911bb87c105efb15b01"
    ["$task_contract"]="06a4bf9c770e4c4541e1e2f026d35ae18592b0ec73f26359782f86f8bc6b3b69"
    ["$task_scheduler"]="4e640770349fa2d95ac09731efe7f8587d8bb108bd89169c204200cf41f3983a"
    ["$task_mapper"]="2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337"
    ["$task_accumulator"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["$task_adapter"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["$task_revocation"]="d43509bd8fa757c43f280e811240cd825b86a4a348d7e99d9feab286abf66922"
    ["$task_m123_contract"]="63432933d974b277453545118ac02f5d8a803987f8102982e56ee70177eb3f87"
    ["$task_m123_receipt"]="736b989529d1ca6b83bcb705fb87f9f381efb3f7f0809811fda3630006bbc0a8"
    ["$task_m121_review"]="f9d4dfe4172e075cc0604facc6971a6e4391543629dbb4630893d5830a13509b"
    ["$task_m120_review"]="51ad53084fd73b64c3e7bf902ea72313bf0f4df660adaf4124c08cb2cb8116f1"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M124 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m124_w384_scheduler_numeric_quarantine_island \
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
grep -qx 'PASS M124 W384 scheduler numeric island VCS descriptors=2 ingress_events=98304 active_keys=256 prefetches=256 service_tokens=99072 weight_loads=768 service_events=98304 weight_reads=768 tail_bypass_first_events=256 zero_bubble_key_transitions=254 downstream_backpressure_cycles=0 mapped_updates=98304 mapped_ii1_pairs=98048 accumulator_writes=98304 lane_rw_overlap=98048 descriptor_done=2 commits=3072 commit_lane_checks=294912 commit_stalls=366 protocol_attacks=2 weight_port_bits=256 weight_read_latency=1 accumulator_lanes=96 accumulator_bits=19 accumulator_bytes=700416 directed_end_to_end_service_island=true m123_same_address_forwarding=true composite_quarantine=true weight_response_valid=false heldout_trace_replay=false foundry_sram_macro=false module_cycle_projection=2.53546204172554 physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
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
        'cp_numeric_fault, .* 1 match' \
        'cp_scheduler_fault_quarantine, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M124_W384_SCHEDULER_NUMERIC_QUARANTINE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "descriptors=2"
    echo "service_event_tokens=98304"
    echo "mapped_accumulator_updates=98304"
    echo "accumulator_writes=98304"
    echo "m123_same_address_forwarding=true"
    echo "same_cycle_composite_quarantine=true"
    echo "scheduler_fault_attack_passed=true"
    echo "numeric_fault_attack_passed=true"
    echo "post_fault_lifecycle_accepts=0"
    echo "post_fault_weight_requests=0"
    echo "post_fault_commit_outputs=0"
    echo "fixed_weight_read_latency_cycles=1"
    echo "weight_response_valid=false"
    echo "whole_descriptor_retry_deduplication=false"
    echo "heldout_trace_replay=false"
    echo "foundry_sram_macro=false"
    echo "module_cycle_projection_admitted=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m124_w384_scheduler_numeric_quarantine.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M124 W384 scheduler numeric quarantine VCS sealed at $task_run"
