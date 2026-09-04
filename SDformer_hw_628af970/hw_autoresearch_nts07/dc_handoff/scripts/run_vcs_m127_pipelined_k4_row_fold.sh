#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m127_pipelined_k4_row_fold_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M127 sealed VCS run: $task_run" >&2
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
task_m125="rtl_m125/m125_block_phased_k4_row_fold.sv"
task_m127="rtl_m127/m127_block_phased_pipelined_k4_row_fold.sv"
task_sva="verif_m127/m127_block_phased_pipelined_k4_row_fold_assertions.sv"
task_tb="tb_m127/tb_m127_block_phased_pipelined_k4_row_fold.sv"
task_files="dc_handoff/filelists/date_m127_pipelined_k4_row_fold_differential_vcs.f"
task_contract="contracts/m127_pipelined_k4_row_fold_vcs_contract_r1_20260824.json"
task_m125_review="reviews/m125_block_phased_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_m125"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["$task_m127"]="5c0c779e8ab463b6589804736bc4d83e77e28cd626a8a117c50caf4a7ea15a5c"
    ["$task_sva"]="f825e7f2ff7f6617d6cd42c81e620e39675164e430dcf528e1e0c7c1986209bb"
    ["$task_tb"]="abb4462609bf8fe719b7eddde077670fff7a2257632144b794935ae4b26d07a6"
    ["$task_files"]="10b1b4c156f68f3442b576b156aca5b57c29ca83bb1fdc2f07dbabff5961de63"
    ["$task_contract"]="2640b4ba5545cffcd0dd55dce002f4cb3d18222a2379c4f41170888a1a0bc293"
    ["$task_m125_review"]="ce917784a653cc9b865bb595a59faaa3b10b228c7760abceb1bb87935a99296e"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M127 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m127_block_phased_pipelined_k4_row_fold \
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
grep -qx 'PASS M127 pipelined K4 row fold VCS fills=99 rows=80 row_done=80 updates=176 selected_sources=606 numeric_lane_checks=16896 full_k4_updates=126 tail_updates=50 ii1_update_pairs=79 update_stalls=38 plus512_checks=2 cycle_exact_checks=507 reset_attacks=1 protocol_attacks=1 pair_pipeline_bits=1920 first_group_extra_cycles=0 m125_cycle_exact_positive=true reset_isolation=true cache_bytes=1536 foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_four_ii1_groups, .* 1 match' \
        'cp_full_k4, .* 126 match' \
        'cp_tail_k1, .* 18 match' \
        'cp_update_stall_release, .* 35 match' \
        'cp_empty_row, .* 2 match' \
        'cp_reset_requests_quiesced, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M127_PIPELINED_K4_ROW_FOLD_DIFFERENTIAL_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_fill_beats=96"
    echo "positive_rows=80"
    echo "positive_updates=176"
    echo "positive_source_contributions=606"
    echo "numeric_lane_checks=16896"
    echo "m125_cycle_exact_checks=507"
    echo "m125_cycle_exact_mismatches=0"
    echo "pair_sum_pipeline_storage_bits=1920"
    echo "first_group_extra_cycles=0"
    echo "four_group_ii1=true"
    echo "reset_isolation=true"
    echo "foundry_weight_macro=false"
    echo "dc_frequency_improvement=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m127_pipelined_k4_row_fold.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M127 pipelined K4 row fold VCS sealed at $task_run"
