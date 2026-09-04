#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M131 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m131/m131_synthesis_safe_compact_canonical_k4_row_fold.sv"
task_sva="verif_m131/m131_synthesis_safe_compact_canonical_k4_row_fold_assertions.sv"
task_tb="tb_m131/tb_m131_synthesis_safe_compact_canonical_k4_row_fold.sv"
task_files="dc_handoff/filelists/date_m131_synthesis_safe_compact_canonical_k4_row_fold_directed_vcs.f"
task_contract="contracts/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json"
task_m130_correction="contracts/m130_r1_dc_elaboration_failure_correction_r1_20260824.json"
task_m130_receipt="dc_handoff/runs/m130_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_rtl"]="82987dd367892213c3f57f0b17b5df4e92603653be9d8a093c9d9b2229cda4ea"
    ["$task_sva"]="17b6493046088f28c6f824e18b3563d703d7c89b4d8d90b6e760135523c79cd4"
    ["$task_tb"]="c81d0cd1a12a5860d1712a71bd04d31960008ce3e21a3914618a30c89488c434"
    ["$task_files"]="f65d8f05819ade452b06a4e8442c47e79ff74a52331afd43c08e63e597fd7013"
    ["$task_contract"]="0e657b5916e428fe09df82588479654185055ab734b74a2782fc9b1ec9bae8ba"
    ["$task_m130_correction"]="9164e6b79846cd6017b03592847d54453d3e2cbfa65549e2cbb9ce281b7fc2ef"
    ["$task_m130_receipt"]="3eec7d86d5129c752f812dd27a9192644349d459d92f4b493bd18ebe0c105135"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M131 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m131_synthesis_safe_compact_canonical_k4_row_fold \
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
task_pass='PASS M131 compact canonical K4 row fold VCS groups=237 updates=237 sources=691 lanes=22752 done=193 done_overlap=190 stalls=60 long_stall=17 cross_row_updates=64 cross_row_ii1=63 plus512=1 protocol_attacks=4 reset_attacks=1 idle_payload=1 descriptor_bits=35 producer_implemented=false physical_speedup=false system_speedup=false headline=false'
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
    echo "status=PASS_M131_SYNTHESIS_SAFE_COMPACT_CANONICAL_K4_ROW_FOLD_VCS_SVA"
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
    echo "negative_predecessor_index_removed=true"
    echo "complete_row_partition_losslessness=false"
    echo "descriptor_producer_implemented=false"
    echo "synopsys_dc_elaboration_clean=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m131_synthesis_safe_compact_canonical_k4_row_fold.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M131 synthesis-safe compact canonical K4 row fold VCS sealed at $task_run"
