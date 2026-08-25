#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m148_destination_tagged_mosaic_k4_packer_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M148 sealed VCS run: $task_run" >&2
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
declare -A task_expected=(
    ["rtl_m148/m148_destination_tagged_mosaic_k4_packer.sv"]="05bd3dfa27a2e5fcc8e05b6fa1cbde7e6710692bf3c030290be044bb83bb2b92"
    ["verif_m148/m148_destination_tagged_mosaic_k4_packer_assertions.sv"]="1cf1c3a0af7467ee04a95a4a8c8e5dd11d9fc1a5447f8b7206f808ce1c43f316"
    ["tb_m148/tb_m148_destination_tagged_mosaic_k4_packer.sv"]="ae0cbe1a8e5d6212abc7ebc03893057479fdcd53c30570f5f1bb427863dbe682"
    ["dc_handoff/filelists/date_m148_destination_tagged_mosaic_k4_packer_directed_vcs.f"]="08897255842074a1670ff8718311ae1818c2d6647ef86e2c8d210e3474ff8a82"
    ["contracts/m148_destination_tagged_mosaic_k4_packer_vcs_contract_r1_20260824.json"]="3ecc3dffc222d7e47bee0779fc93d1cf73cec8192c4d29a34e586856220f6d38"
    ["contracts/m147_independent_review_correction_overlay_r1_20260824.json"]="8a6ba2e9dce906378708a9ecc1cbc71b86153d7d011e1cb9cf8ff5718fa4c9af"
    ["results/m147_independent_hammer_review_r1_20260824/immutable_manifest.sha256"]="ea789dcbd26be622ef68ca900fa367812711bc46daef3fd46b6da713f373f3ce"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M148 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m148_destination_tagged_mosaic_k4_packer_directed_vcs.f \
    -top tb_m148_destination_tagged_mosaic_k4_packer \
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
task_pass='PASS M148 destination-tagged mosaic K4 packer VCS jobs=68 rows=68 events=1901 descriptors=506 block_k4_descriptors=558 descriptor_savings=52 stalls=83 ii1_pairs=419 protocol_attacks=2 stable_order=true exact_tuple_conservation=true zero_row_floor=true first_descriptor_fallthrough=true engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in cp_zero_row cp_fallthrough_first_descriptor \
        cp_full_four_tuple cp_tail_one_tuple cp_tail_two_tuple \
        cp_tail_three_tuple cp_descriptor_stall \
        cp_cross_destination_descriptor; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"
done

{
    echo "status=PASS_M148_DESTINATION_TAGGED_MOSAIC_K4_PACKER_VCS_SVA"
    echo "exact_sha=true"
    echo "jobs=68"
    echo "rows=68"
    echo "source_events=1901"
    echo "mosaic_descriptors=506"
    echo "block_k4_reference_descriptors=558"
    echo "descriptor_savings=52"
    echo "descriptor_stalls=83"
    echo "consecutive_ii1_pairs=419"
    echo "protocol_attacks=2"
    echo "stable_order=true"
    echo "exact_tuple_presence_conservation=true"
    echo "first_descriptor_fallthrough=true"
    echo "signed_negate_metadata=false"
    echo "same_destination_combine_engine=false"
    echo "engine_arithmetic=false"
    echo "sram_macro=false"
    echo "cycle_speedup=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m148_destination_tagged_mosaic_k4_packer.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M148 destination-tagged mosaic K4 packer VCS sealed at $task_run"
