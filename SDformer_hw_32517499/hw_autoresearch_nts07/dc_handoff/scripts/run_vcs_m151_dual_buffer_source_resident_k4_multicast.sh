#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m151_dual_buffer_source_resident_k4_multicast_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M151 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m151/m151_dual_buffer_source_resident_k4_multicast_frontend.sv"]="a20b38e563195e9eea6277627fcd696628cc1077c90a1ef63c014d0d03c7c423"
    ["verif_m151/m151_dual_buffer_source_resident_k4_multicast_frontend_assertions.sv"]="a9984c0ff4d38179423059cd7a03092e2b49e0db0d02ac0f1933005fec3a7808"
    ["tb_m151/tb_m151_dual_buffer_source_resident_k4_multicast_frontend.sv"]="5b1519e70b7a8e63558133a1881b5195aa4ed944019eabccd17bef12fb653294"
    ["dc_handoff/filelists/date_m151_dual_buffer_source_resident_k4_multicast_directed_vcs.f"]="df24818daafc5cc837585ad409714894583d94efa4f29b52d22e5f4b4f47d1c6"
    ["contracts/m151_dual_buffer_source_resident_k4_multicast_vcs_contract_r1_20260824.json"]="3e5f40ca35930a759d0ca82031b27aceb9f10a9fb20e18a13f640686b25da6a6"
    ["contracts/m150_source_stationary_destination_k4_pwp_dse_contract_r1_20260824.json"]="a85fd76796a07326dccbb484428d01127121719f44d78e6608c7cb891bb52237"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m151_dual_buffer_source_resident_k4_multicast_directed_vcs.f \
    -top tb_m151_dual_buffer_source_resident_k4_multicast_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M151 dual-buffer source-resident K4 multicast frontend VCS keys=48 descriptors=96 outputs=96 releases=48 stalls=22 load_descriptor_overlap=42 ii1_pairs=84 protocol_attacks=4 lanes=96 resident_slots=2 source_vector_bits=1056 destination_ports=4 stable_source_vector_once=true independent_negate_metadata=true accumulator_write_ports=false pwp_storage=false sram_macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in cp_both_slots_resident cp_overlap_load_descriptor \
        cp_full_four_destination cp_tail_one_destination \
        cp_tail_two_destination cp_tail_three_destination \
        cp_multicast_stall cp_back_to_back_descriptor \
        cp_release_and_other_slot_live; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"
done

{
    echo "status=PASS_M151_DUAL_BUFFER_SOURCE_RESIDENT_K4_MULTICAST_VCS_SVA"
    echo "exact_sha=true"
    echo "source_keys=48"
    echo "descriptors=96"
    echo "outputs=96"
    echo "releases=48"
    echo "multicast_stalls=22"
    echo "simultaneous_load_descriptor_accepts=42"
    echo "back_to_back_descriptor_pairs=84"
    echo "protocol_attacks=4"
    echo "lanes=96"
    echo "resident_slots=2"
    echo "source_vector_bits_per_lane_signed=11"
    echo "source_vector_payload_bits=1056"
    echo "maximum_distinct_destinations_per_descriptor=4"
    echo "independent_negate_metadata=true"
    echo "source_vector_transmitted_once=true"
    echo "four_destination_accumulator_write_ports=false"
    echo "per_destination_signed_application=false"
    echo "pwp_storage_or_reconstruct=false"
    echo "sram_macro=false"
    echo "m150_cycle_ratio_admitted=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m151_dual_buffer_source_resident_k4_multicast.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M151 dual-buffer source-resident K4 multicast VCS sealed at $task_run"
