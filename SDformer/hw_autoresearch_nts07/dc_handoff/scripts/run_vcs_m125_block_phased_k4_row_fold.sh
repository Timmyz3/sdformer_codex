#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m125_block_phased_k4_row_fold_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M125 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m125/m125_block_phased_k4_row_fold.sv"
task_sva="verif_m125/m125_block_phased_k4_row_fold_assertions.sv"
task_tb="tb_m125/tb_m125_block_phased_k4_row_fold.sv"
task_files="dc_handoff/filelists/date_m125_block_phased_k4_row_fold_directed_vcs.f"
task_contract="contracts/m125_block_phased_k4_row_fold_vcs_contract_r1_20260824.json"
task_m122_correction="contracts/m122_r1_row_fold_admission_and_width_correction_r1_20260824.json"
task_m122_review="reviews/m122_w384_row_synchronous_source_fold_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["$task_sva"]="35f637d853a9760824a638db8757828afe7d4ecfe8e880e578896f082f8432b9"
    ["$task_tb"]="ad90e409d53d5b32a5b1a1f7bd25c6b0bfca9bb2933acf8e216684bf4c450384"
    ["$task_files"]="ee2d94cdea3fa5e1e7b5f6210e61a93bd364f0e84c8e86a2cbbec317e2fcb8cc"
    ["$task_contract"]="0e3512088045a32afa4eafafdf7ff9003f988732e6e01b46c9ed1520da3dbf12"
    ["$task_m122_correction"]="89eedd777da62cb43f6604bc9b6fa5654c8f9d4ff08a72bbc309e3f4a74ef42e"
    ["$task_m122_review"]="db8263d61ddac4dc86b848eb19c09f9840873952fbe1ca91167efce0442a2c77"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M125 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m125_block_phased_k4_row_fold \
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
grep -qx 'PASS M125 block-phased K4 row fold VCS fills=51 rows=66 row_done=66 updates=155 selected_sources=528 numeric_lane_checks=14880 full_k4_updates=105 tail_updates=50 same_row_update_pairs=64 update_stalls=47 negated_minus128_contributions=20 plus512_checks=1 cache_bytes=1536 resident_blocks=1 logical_read_bits_per_update=3072 generic_fold_bits=11 accumulator_delta_bits=19 canonical_select_clear=true fixed8_service_island_projection=3.1725369008459166 projection_only=true m123_integrated=false foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_full_k4, .* 105 match' \
        'cp_tail_k1, .* 14 match' \
        'cp_two_fold_same_row, .* 64 match' \
        'cp_update_stall_release, .* 42 match' \
        'cp_empty_row, .* 1 match' \
        'cp_fault, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M125_BLOCK_PHASED_K4_ROW_FOLD_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_cache_fill_beats=48"
    echo "positive_rows=66"
    echo "row_done=66"
    echo "accepted_updates=155"
    echo "selected_sources=528"
    echo "numeric_lane_checks=14880"
    echo "full_k4_updates=105"
    echo "tail_updates=50"
    echo "consecutive_same_row_update_pairs=64"
    echo "generic_signed11_plus512_boundary=true"
    echo "logical_weight_cache_bytes=1536"
    echo "resident_blocks=1"
    echo "logical_read_bits_per_update=3072"
    echo "m123_accumulator_integrated=false"
    echo "foundry_weight_macro=false"
    echo "fixed8_service_island_projection=3.1725369008459166"
    echo "projection_only=true"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m125_block_phased_k4_row_fold.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M125 block-phased K4 row fold VCS sealed at $task_run"
