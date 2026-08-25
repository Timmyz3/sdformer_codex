#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_negative="${M273_NEGATIVE_PREFLIGHT_TEST:-0}"
if [[ "$task_negative" == 1 ]]; then
    task_run="$task_hw_root/results/m273_integrated_rank3_atlif_wrong_sha_preflight_r1_20260825"
else
    task_run="$task_hw_root/results/m273_integrated_rank3_atlif_directed_vcs_r1_exact_20260825"
fi
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M273 sealed run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m273/m273_integrated_rank3_atlif.sv"]="f7c42d60f34a0b2454aa64ebc4687ff51259958485624481f3bc1beb3167bbd6"
 ["verif_m273/m273_integrated_rank3_atlif_assertions.sv"]="a3aea55573a93062b0ffdd13cc193a9d6cac3ca3ac7b9b13da3f74c663fb42b2"
 ["tb_m273/tb_m273_integrated_rank3_atlif.sv"]="b775fcb3018c6106e1ea4e22a99c53ef8249a46600f732aeff0f0cb6711e46a4"
 ["dc_handoff/filelists/date_m273_integrated_rank3_atlif_rtl.f"]="c99fe329c43276ce40f7027d54baeaaf747553c9f0b8d4419dcf8e7574b1a02d"
 ["dc_handoff/filelists/date_m273_integrated_rank3_atlif_directed_vcs.f"]="b47c7665ecf029fa996fdda4518d7afc2f494ad3103fbdeec4bdbd8cc9261399"
 ["contracts/m273_integrated_rank3_atlif_vcs_contract_r1_20260825.json"]="e1d219251903c9e9316aafbd6664e5a3d5240de6839196f7a03a6cf2b0de0cb4"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
if [[ "$task_negative" == 1 ]]; then
    task_expected["rtl_m273/m273_integrated_rank3_atlif.sv"]="0000000000000000000000000000000000000000000000000000000000000000"
fi
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
[[ "$task_negative" != 1 ]] || exit 11
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m273_integrated_rank3_atlif_directed_vcs.f \
    -top tb_m273_integrated_rank3_atlif \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" \
    && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=273025 -no_save \
    -assert report="$task_run/assert.report" -cm assert \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true

grep -Fq 'PASS M273 integrated rank3 ATLIF directed clean_contexts=2 pressure_contexts=1 attacks=7 numerical_mismatches=0 clean_cycles_N1=24 clean_cycles_N4=39 pressure_cycles=1618 fifo_peak=16 overlap=1 product_replace=1 full_pop_push=1' \
    "$task_run/sim.log" || exit 30
for task_cover in cp_clean_overlap cp_product_replace cp_result_stall \
        cp_fifo_full cp_full_pop_push cp_raw_backpressure cp_release_wait \
        cp_release cp_context_retire cp_config_fault cp_raw_fault cp_beat4; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 31
done

{
    echo status=PASS_M273_INTEGRATED_RANK3_ATLIF_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo clean_contexts=2
    echo pressure_contexts=1
    echo protocol_attacks=7
    echo checked_result_beats=225
    echo numerical_mismatches=0
    echo conservation_mismatches=0
    echo clean_cycles_n1=24
    echo clean_cycles_n4=39
    echo clean_cycle_formula=5N_plus_19
    echo pressure_cycles=1618
    echo maximum_fifo_occupancy=16
    echo oldest_first_raw_bank=true
    echo oldest_first_intermediate_bank=true
    echo full_simultaneous_pop_push=true
    echo context_drain_release=true
    echo standalone_integrated_rank3_atlif=true
    echo fixed_baseline_source=M31_M265
    echo fixed_baseline_area_matched=false
    echo dc=false
    echo ppa_ready=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/m273_integrated_rank3_atlif_vcs_receipt_r1.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$task_run/SHA256SUMS"
printf 'PASS_M273_INTEGRATED_RANK3_ATLIF_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M273 exact VCS sealed at $task_run"
