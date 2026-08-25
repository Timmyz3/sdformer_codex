#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_hw_root/results/m220_m218_m219_l4_cross_module_miter_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M220 sealed VCS run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m218/m218_fc2_tagged_slice_service_island.sv"]="f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1"
 ["rtl_m219/m219_fc2_k1_cropped_tagged_slice_service_island.sv"]="75c4690ec04653084fb59fd75c5ba7ac329807975d76c9ffc43b6304bd4e1d47"
 ["tb_m220/tb_m220_m218_m219_l4_cross_module_miter.sv"]="409d38c878c47726c3fca86fd7011ac48db40c153b03e43f8a925d4e743986e6"
 ["dc_handoff/filelists/date_m220_m218_m219_l4_cross_module_miter_vcs.f"]="f1d7f3e63b39337eb445766c2a7bb36cee26c40ff2f1908a4ccfae370792f8c5"
 ["contracts/m220_m218_m219_l4_cross_module_miter_vcs_contract_r1_20260825.json"]="795af8cafc6a3c5367ec563d25e4c4b4c015a38087cca98ce70259a6d9db202f"
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
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m220_m218_m219_l4_cross_module_miter_vcs.f \
    -top tb_m220_m218_m219_l4_cross_module_miter \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" \
    && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=220025 -no_save \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
grep -Eiq '^Error:|^Fatal:|Fatal:|watchdog|mismatch' \
    "$task_run/sim.log" && exit 23 || true
[[ "$(grep -Fc 'M220 pair blocks=' "$task_run/sim.log")" -eq 33 ]] || exit 30
grep -Fq 'PASS M220 cross-module L4 miter pairs=33 numeric_cases=1 recurrence_checks=3912 M218_M219_bit_exact=true work_conserved=true' \
    "$task_run/sim.log" || exit 31
grep -Fq 'M220 pair blocks=8 sources=8 mode=0 M218_cycles=103 M219_cycles=439 reads=384' \
    "$task_run/sim.log" || exit 32
grep -Fq 'M220 pair blocks=1 sources=2 mode=1 M218_cycles=19 M219_cycles=25 reads=12' \
    "$task_run/sim.log" || exit 33

{
    echo status=PASS_M220_M218_M219_L4_CROSS_MODULE_MITER_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo cross_module_pairs=33
    echo source_counts_covered=1_to_8
    echo output_blocks_covered=1_2_4_8
    echo memory_latency_cycles=4
    echo memory_initiation_interval_cycles=1
    echo outstanding=8
    echo recurrence_checks=3912
    echo numeric_mismatches=0
    echo conservation_mismatches=0
    echo response_latency_mismatches=0
    echo signed_minus128_plus127_dynamic_case=true
    echo m218_m219_bit_exact=true
    echo active_bank_reads_equal=true
    echo small_token_cycles_not_frozen_h67_speedup=true
    echo frozen_h67_service_speedup_recalibrated=false
    echo macro_aware_energy=false
    echo complete_fc2=false
    echo complete_ffn=false
    echo physical_speedup=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/m220_m218_m219_l4_cross_module_miter_vcs_receipt_r1.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$task_run/SHA256SUMS"
printf 'PASS_M220_M218_M219_L4_CROSS_MODULE_MITER_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M220 exact cross-module VCS sealed at $task_run"
