#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M128 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m128/m128_descriptor_streamed_k4_row_fold.sv"
task_sva="verif_m128/m128_descriptor_streamed_k4_row_fold_assertions.sv"
task_tb="tb_m128/tb_m128_descriptor_streamed_k4_row_fold.sv"
task_files="dc_handoff/filelists/date_m128_descriptor_streamed_k4_row_fold_directed_vcs.f"
task_contract="contracts/m128_descriptor_streamed_k4_row_fold_vcs_contract_r1_20260824.json"
task_correction="contracts/m127_r1_throughput_scope_correction_r1_20260824.json"
task_review="reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="b7c5c4c329bc4f1a7011398c5d3c20933dd8badfc4b2bbf3b213b15efe01e54d"
    ["$task_sva"]="334c366289690bff624e8a3976dd602ed45f6046b7b1ed6314143922e5a06a50"
    ["$task_tb"]="30cc18e83a00173a9f0e17ea5116f5429a340fbea88f3decb4d28073e8cbee94"
    ["$task_files"]="685e547c610acbbf8f9298bb32f9ced1035aff158192d9f882e2c519f5f9cf7c"
    ["$task_contract"]="7b08459cbba96f14666c57b5db274b850b58546c25d7d42e52210bf9e4228bf1"
    ["$task_correction"]="a64a00f443d691b1295a4eb14a92edbc9d41ce448d83fd3a8c3ca4f59d2b365d"
    ["$task_review"]="8bea333f44528044f251a48ebf9d20e261e4919bc63ed9f262b01004d25c7947"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M128 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m128_descriptor_streamed_k4_row_fold \
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
task_pass='PASS M128 descriptor-streamed K4 row fold VCS groups=384 updates=384 sources=1056 lanes=36864 rows_done=170 stalls=98 cross_row_updates=64 cross_row_ii1=63 plus512=1 protocol_attacks=1 reset_attacks=1 cache_bytes=1536 descriptor_predecode_external=true physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_cross_row_replace, .* [1-9][0-9]* match' \
        'cp_k4_descriptor, .* [1-9][0-9]* match' \
        'cp_tail_descriptor, .* [1-9][0-9]* match' \
        'cp_update_stall_release, .* [1-9][0-9]* match' \
        'cp_reset_quiesce, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M128_DESCRIPTOR_STREAMED_K4_ROW_FOLD_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_fill_beats=48"
    echo "positive_descriptors=384"
    echo "positive_updates=384"
    echo "positive_source_contributions=1056"
    echo "numeric_lane_checks=36864"
    echo "cross_row_single_group_updates=64"
    echo "cross_row_adjacent_ii1_intervals=63"
    echo "descriptor_bits=53"
    echo "descriptor_predecode_external=true"
    echo "descriptor_producer_implemented=false"
    echo "foundry_weight_macro=false"
    echo "dc_frequency_improvement=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m128_descriptor_streamed_k4_row_fold.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M128 descriptor-streamed K4 row fold VCS sealed at $task_run"
