#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m126_block_phased_k4_forwarding_accumulator_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M126 sealed VCS run: $task_run" >&2
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
task_m123_core="rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"
task_m123_adapter="rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"
task_m126="rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv"
task_sva="verif_m126/m126_block_phased_k4_forwarding_accumulator_island_assertions.sv"
task_tb="tb_m126/tb_m126_block_phased_k4_forwarding_accumulator_island.sv"
task_files="dc_handoff/filelists/date_m126_block_phased_k4_forwarding_accumulator_directed_vcs.f"
task_contract="contracts/m126_block_phased_k4_forwarding_accumulator_vcs_contract_r1_20260824.json"
task_m122_correction="contracts/m122_r1_row_fold_admission_and_width_correction_r1_20260824.json"
task_m125_review="reviews/m125_block_phased_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"
task_m123_review="reviews/m123_w384_signed19_forwarding_accumulator_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_m125"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["$task_m123_core"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["$task_m123_adapter"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["$task_m126"]="b75c64cfa0803461bef4690025a723df9e039e8d2eef6a0da918fc3b9c063e01"
    ["$task_sva"]="fee69341cb32d960eedcc97646fbf893a1c88e6b220ba6a6c2a05c2be22f64c1"
    ["$task_tb"]="18784c618a86785ae5bf083257a8559059132323ea3b2d13e49962435d0c7cbc"
    ["$task_files"]="890b2870bae08860f47e12afd48258e3f20e1f67168b51105659df3c016e5412"
    ["$task_contract"]="f9a8783e1f3fc915bb690e42703a8547377fbf41c92ecc6276673c9f9ac44889"
    ["$task_m122_correction"]="89eedd777da62cb43f6604bc9b6fa5654c8f9d4ff08a72bbc309e3f4a74ef42e"
    ["$task_m125_review"]="ce917784a653cc9b865bb595a59faaa3b10b228c7760abceb1bb87935a99296e"
    ["$task_m123_review"]="297b24a9877b0efb9da7dd6f388c117bdec97284f9b126ffea0d2ccae1f59f9e"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M126 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m126_block_phased_k4_forwarding_accumulator_island \
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
grep -qx 'PASS M126 K4 fold plus forwarding accumulator VCS starts=2 fills=387 rows=3073 row_done=3072 fold_updates=7327 selected_sources=24803 full_k4_updates=5115 tail_updates=2212 same_row_update_pairs=4262 lane_writes=7326 rw_overlap=0 commits=3072 commit_lane_checks=294912 commit_stalls=401 plus512_checks=1 reset_attacks=1 positive_fold_updates=7326 positive_selected_sources=24802 positive_tail_updates=2211 positive_lane_writes=7326 reset_pending_updates=1 reset_suppressed_writes=1 blocks=8 rows_per_block=384 lanes=96 cache_bytes=1536 fold_bits=11 accumulator_bits=19 m125_m123_integrated=true reset_isolation=true functional_directed_update_compression=3.385476385476 heldout_fixed8_service_projection=3.1725369008459166 projection_only=true foundry_weight_macro=false foundry_accumulator_macro=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_four_consecutive_same_row_folds, .* 160 match' \
        'cp_full_k4_to_write, .* 5115 match' \
        'cp_tail_to_write, .* 2211 match' \
        'cp_commit_stall_release, .* 384 match' \
        'cp_reset_with_prior_update, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M126_BLOCK_PHASED_K4_FORWARDING_ACCUMULATOR_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_rows=3072"
    echo "positive_source_contributions=24802"
    echo "positive_fold_updates=7326"
    echo "positive_accumulator_lane_writes=7326"
    echo "positive_commit_vectors=3072"
    echo "positive_commit_lane_checks=294912"
    echo "consecutive_same_row_update_pairs=4262"
    echo "generic_signed11_plus512_boundary=true"
    echo "reset_pending_updates=1"
    echo "reset_suppressed_physical_writes=1"
    echo "reset_isolation=true"
    echo "functional_directed_update_compression=3.385476385476"
    echo "heldout_fixed8_service_island_projection=3.1725369008459166"
    echo "projection_only=true"
    echo "foundry_weight_macro=false"
    echo "foundry_accumulator_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m126_block_phased_k4_forwarding_accumulator.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M126 K4 fold plus forwarding accumulator VCS sealed at $task_run"
