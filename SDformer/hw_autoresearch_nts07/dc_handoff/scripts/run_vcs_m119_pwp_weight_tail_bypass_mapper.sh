#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m119_pwp_weight_tail_bypass_mapper_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M119 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv"
task_sva="verif_m119/m119_pwp_weight_tail_bypass_mapper_assertions.sv"
task_tb="tb_m119/tb_m119_pwp_weight_tail_bypass_mapper.sv"
task_files="dc_handoff/filelists/date_m119_pwp_weight_tail_bypass_mapper_directed_vcs.f"
task_contract="contracts/m119_pwp_weight_tail_bypass_mapper_vcs_contract_r1_20260824.json"
task_m117_review="reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824/m117_w384_prefetch_transpose_independent_hammer_review.json"
task_m117_manifest="reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824/manifest.sha256"
task_m118_review="reviews/m118_w384_signed19_lane_accumulator_independent_hammer_r1_20260824/m118_w384_signed19_lane_accumulator_independent_hammer_review.json"
task_m118_manifest="reviews/m118_w384_signed19_lane_accumulator_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337"
    ["$task_sva"]="dc2270a3538dc7fa803d2c2cc5bc850bea074fdd49acea65946181da25f302e2"
    ["$task_tb"]="fe43f7e9e55234d40ba301311b063a2fbd3da731ecec1b0127999deb5f33cba5"
    ["$task_files"]="440fa48db8ba7075049d556f3b4e8130542c0ab6edb78b1cfcfc4fef1c3e4989"
    ["$task_contract"]="5ccdebb50ae7149bd51a7b767ae3176758c9617dc6e751d04255814e001e3cd8"
    ["$task_m117_review"]="f6e173bd335338eae085efc9e540f9d990e2bd477a228a0708b99d50f7e7eb27"
    ["$task_m117_manifest"]="3fc112ae72769f4dbba8aed8450fe2b840327292a112825956be8160e93137b2"
    ["$task_m118_review"]="f6acfb9fd740409a3146d22992e215486c781c40fc4207a26403cdf8899d93e9"
    ["$task_m118_manifest"]="a757b17b2089fc50fa2cc6a5a3c9ac96956198209f27046ecd504c16f8367bf0"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M119 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m119_pwp_weight_tail_bypass_mapper \
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
grep -qx 'PASS M119 PWP weight tail-bypass mapper VCS groups=129 weight_loads=387 weight_reads=387 events=513 updates=513 lane_checks=49248 tail_bypass_first_events=129 event_ii1_pairs=384 update_stalls=3 negate_events=257 protocol_attacks=1 weight_port_bits=256 weight_beats=3 weight_payload_bits=768 lanes=96 acc_bits=19 delta_bits=1824 fixed_read_latency=1 tail_bypass=true exact_once_directed=true accumulator_integrated=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_three_loads_then_tail_event, .* 129 match' \
        'cp_event_ii1, .* 384 match' \
        'cp_update_stall, .* 1 match' \
        'cp_signed_map_accept, .* 512 match' \
        'cp_fault, .* 1 match' \
        'cp_busy, .* 903 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M119_PWP_WEIGHT_TAIL_BYPASS_MAPPER_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "fixed_weight_read_latency_cycles=1"
    echo "weight_port_bits=256"
    echo "weight_beats_per_key=3"
    echo "weight_payload_bits=768"
    echo "signed_int8_lanes=96"
    echo "signed_accumulator_bits=19"
    echo "mapped_delta_bits=1824"
    echo "tail_bypassed_first_events=129"
    echo "uncounted_tail_bubbles=0_in_directed_scope"
    echo "accepted_events=513"
    echo "mapped_updates=513"
    echo "signed_lane_checks=49248"
    echo "behavioral_weight_memory=true"
    echo "foundry_weight_sram_macro=false"
    echo "m117_rtl_integration=false"
    echo "m118_accumulator_integration=false"
    echo "heldout_trace_exact_once_replay=false"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m119_pwp_weight_tail_bypass_mapper.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M119 PWP weight tail-bypass mapper VCS sealed at $task_run"
