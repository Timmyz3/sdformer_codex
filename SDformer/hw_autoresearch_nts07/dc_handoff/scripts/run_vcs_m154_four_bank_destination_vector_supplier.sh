#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m154_four_bank_destination_vector_supplier_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M154 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m154/m154_four_bank_destination_vector_supplier.sv"]="5a791777631dee6bca4c457e634437d818ec1abb1a147d4603a08372aa8acfa6"
    ["verif_m154/m154_four_bank_destination_vector_supplier_assertions.sv"]="8feb9d13a260eafa234b60e193702e960384a3fe54e0012c2aed4d02ec8d9a62"
    ["tb_m154/tb_m154_four_bank_destination_vector_supplier.sv"]="3bc7797e264a08068b7cabbc3371280e92382c52186155c7bf35c4bde9836e9b"
    ["dc_handoff/filelists/date_m154_four_bank_destination_vector_supplier_directed_vcs.f"]="853aa9c18b2828af4e80969384c63d13e2aacb22ce578693cdc1556b458cefb4"
    ["contracts/m154_four_bank_destination_vector_supplier_vcs_contract_r1_20260824.json"]="9d9bd51c79970e87a5dd73b272e7b70fbc528e31d459c9f3a34b3b9d60e1137b"
    ["contracts/m150_m151_m152_cross_destination_vector_identity_correction_overlay_r1_20260824.json"]="9f913b451a80d9938a199c1ce648f45b3bd064641f2c47213c9092bb252a6c99"
    ["results/m150_independent_hammer_review_r1_20260824/manifest.sha256"]="40d6b5b83d959cb2f42a38f807c7541e6e90cad0038dcd6dde084579ee6b382b"
    ["results/m152_independent_hammer_review_r1_20260824/manifest.sha256"]="0cb6ad78be3d027b413ebd2c255bb5711c7a23ecb021db3083db2214cd9dc195"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
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
    -f dc_handoff/filelists/date_m154_four_bank_destination_vector_supplier_directed_vcs.f \
    -top tb_m154_four_bank_destination_vector_supplier \
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
task_pass='PASS M154 four-bank destination-vector supplier VCS contexts=4 descriptors=55 results=55 vector_groups=208 vector_lane_checks=19968 full4=49 tails=6 result_stalls=19 ii1_pairs=40 protocol_attacks=3 banks=4 vectors_per_full_descriptor=4 lanes=96 weight_bits=8 resident_partition_bits=98304 synchronous_read_latency=1 macro_external=true single_vector_multicast=false accumulator_commit=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_all_four_banks cp_low_destination_row \
        cp_high_destination_row cp_back_to_back_descriptor \
        cp_result_stall cp_protocol_fault_with_pending_result; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M154_FOUR_BANK_DESTINATION_VECTOR_SUPPLIER_VCS_SVA"
    echo "exact_sha=true"
    echo "contexts=4"
    echo "descriptors=55"
    echo "results=55"
    echo "independent_vector_groups=208"
    echo "vector_lane_checks=19968"
    echo "full4_descriptors=49"
    echo "tail_descriptors=6"
    echo "result_stalls=19"
    echo "consecutive_ii1_pairs=40"
    echo "protocol_attacks=3"
    echo "banks=4"
    echo "vectors_per_bank=32"
    echo "vector_bits=768"
    echo "resident_partition_bits=98304"
    echo "synchronous_read_latency_cycles=1"
    echo "single_vector_cross_destination_multicast=false"
    echo "accepted_result_survives_younger_fault=true"
    echo "weight_loading_rtl=false"
    echo "real_checkpoint_payload_replay=false"
    echo "sram_macro=false"
    echo "accumulator_commit=false"
    echo "m152_cycle_ratio_admitted=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m154_four_bank_destination_vector_supplier.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M154 four-bank destination-vector supplier VCS sealed at $task_run"
