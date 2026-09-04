#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m130_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M130 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m130/m130_compact_canonical_k4_row_fold.sv"
task_sva="verif_m130/m130_compact_canonical_k4_row_fold_assertions.sv"
task_tb="tb_m130/tb_m130_compact_canonical_k4_row_fold.sv"
task_files="dc_handoff/filelists/date_m130_compact_canonical_k4_row_fold_directed_vcs.f"
task_contract="contracts/m130_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json"
task_review="reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"
task_correction="contracts/m128_r1_independent_review_correction_overlay_r1_20260824.json"
task_m129_contract="contracts/m129_row_admission_bubble_and_descriptor_cost_contract_r1_20260824.json"
task_m129_result="results/m129_row_admission_bubble_and_descriptor_cost_r1_20260824/m129_row_admission_bubble_and_descriptor_cost.json"

declare -A task_expected=(
    ["$task_rtl"]="ff6d10d2fa341a4ef855f8df196542b990fd71fca34b1b3b81b04c5cb7588e96"
    ["$task_sva"]="1c19a210463af9a8afa17e2c8ca7a562066bda1ed60a3b46be0057601884fc19"
    ["$task_tb"]="bbd8569ac2c1c62853ecff8e54551e4fce58765bef58bfcdf4d0fd4fa43094f9"
    ["$task_files"]="f154ed2a9c30dd7ce44c1560ba289f38302cda5b978b670eef931ecb3d959e14"
    ["$task_contract"]="0a67fb7c1466257edc7c6d2cad960565c050916d8456addb3e0330025b8b911b"
    ["$task_review"]="dd1bf2eddec0d7d76468e38e2c712851203db06d53cd4947360c4cdc2476d3c0"
    ["$task_correction"]="e646cc71cc62ce0d50c128c1a57db9a59221909948413ad3493bfc23cf3d44ec"
    ["$task_m129_contract"]="f7bb5038bfea128b87a9df899cd93cf1e66ef66a60226b2e537cd83384c9f777"
    ["$task_m129_result"]="2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M130 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m130_compact_canonical_k4_row_fold \
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
task_pass='PASS M130 compact canonical K4 row fold VCS groups=237 updates=237 sources=691 lanes=22752 done=193 done_overlap=190 stalls=60 long_stall=17 cross_row_updates=64 cross_row_ii1=63 plus512=1 protocol_attacks=4 reset_attacks=1 idle_payload=1 descriptor_bits=35 producer_implemented=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_cross_row_replace, .* [1-9][0-9]* match' \
        'cp_multidescriptor_row, .* [1-9][0-9]* match' \
        'cp_k1_descriptor, .* [1-9][0-9]* match' \
        'cp_k4_descriptor, .* [1-9][0-9]* match' \
        'cp_update_stall_release, .* [1-9][0-9]* match' \
        'cp_tagged_done_overlaps_next_group, .* [1-9][0-9]* match' \
        'cp_reset_quiesce, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M130_COMPACT_CANONICAL_K4_ROW_FOLD_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_fill_beats=192"
    echo "positive_descriptors=237"
    echo "positive_updates=237"
    echo "positive_source_contributions=691"
    echo "numeric_lane_checks=22752"
    echo "tagged_done_tokens=193"
    echo "tagged_done_overlap_next_group=190"
    echo "cross_row_single_group_updates=64"
    echo "cross_row_adjacent_ii1_intervals=63"
    echo "descriptor_payload_bits=35"
    echo "stream_local_strict_source_order=true"
    echo "complete_row_partition_losslessness=false"
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
sha256sum "dc_handoff/scripts/run_vcs_m130_compact_canonical_k4_row_fold.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M130 compact canonical K4 row fold VCS sealed at $task_run"
