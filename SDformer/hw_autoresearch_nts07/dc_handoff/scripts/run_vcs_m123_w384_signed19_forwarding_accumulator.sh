#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m123_w384_signed19_forwarding_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M123 sealed VCS run: $task_run" >&2
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
task_core="rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"
task_adapter="rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"
task_sva="verif_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_assertions.sv"
task_tb="tb_m123/tb_m123_w384_signed19_forwarding_lane_sliced_accumulator.sv"
task_files="dc_handoff/filelists/date_m123_w384_signed19_forwarding_lane_accumulator_directed_vcs.f"
task_contract="contracts/m123_w384_signed19_forwarding_accumulator_vcs_contract_r1_20260824.json"
task_m120_review="reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_core"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["$task_adapter"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["$task_sva"]="2e4333d7a19f1adfa11f28d0a5ee1ee49efccd32711ea83b845c76032b45137f"
    ["$task_tb"]="7a198caed3e0cb90eb9a07db2fe5168826681795d4fd5717f071a506917a4a58"
    ["$task_files"]="7072f0a32a2efe78d9690adef462fdd70f7c3e07c1aaa55253f0d2e8e2eaaacb"
    ["$task_contract"]="63432933d974b277453545118ac02f5d8a803987f8102982e56ee70177eb3f87"
    ["$task_m120_review"]="51ad53084fd73b64c3e7bf902ea72313bf0f4df660adaf4124c08cb2cb8116f1"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M123 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m123_w384_signed19_forwarding_lane_sliced_accumulator \
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
grep -qx 'PASS M123 W384 forwarding lane-sliced accumulator VCS windows=3 updates=1072 vector_lane_checks=884736 commits=9216 lazy_valid_clears=3 positive_memory_writes=1072 ii1_pairs=1069 same_address_accept_pairs=15 same_address_forward_read_suppressed=15 same_address_chain=16 read_write_overlap=1054 commit_stalls=1101 overflow_attacks=2 lanes=96 vector_bits=1824 accumulator_bytes=700416 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=19 behavioral_macro=true same_address_rdw_mode_independent=true overflow_guard=true reset_recovery=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_update_ii1, .* 1073 match' \
        'cp_same_address_forward_chain, .* 14 match' \
        'cp_read_write_overlap, .* 1058 match' \
        'cp_commit_stall, .* 1050 match' \
        'cp_full_commit, .* 3 match' \
        'cp_fault, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M123_W384_SIGNED19_FORWARDING_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_windows=3"
    echo "accepted_updates=1072"
    echo "positive_memory_writes=1072"
    echo "same_address_chain_length=16"
    echo "same_address_accept_pairs=15"
    echo "same_address_macro_reads_suppressed=15"
    echo "commit_vectors=9216"
    echo "commit_lane_checks=884736"
    echo "macro_rdw_mode_independent=true"
    echo "overflow_fail_closed=true"
    echo "reset_recovery=false"
    echo "transaction_retry_deduplication=false"
    echo "foundry_lane_macro=false"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m123_w384_signed19_forwarding_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M123 W384 signed19 forwarding accumulator VCS sealed at $task_run"
