#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${task_hw_root}/results/m245_m235_full220800_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
 ["rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv"]="ec0bf05540433ecfc436eac63b41a4cecf4cc53b46533f2fd4f44c7eb70bd611"
 ["verif_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine_assertions.sv"]="2c138fc3902a14d0d53a709e5533e11b686bd724bc986e059e4e841137ac8a45"
 ["tb_m245/tb_m245_m235_full220800.sv"]="442f6786084d203f3688e751af6c747144676d1b624570bccdce5340f0836b5e"
 ["dc_handoff/filelists/date_m245_m235_full220800_vcs.f"]="5af0577d7a0ea21954ebb0181fd4bdc8be92d7ba4042b011ddf6e3504fc17d7e"
 ["system_simulator/scripts/generate_m245_m235_full220800_vectors.py"]="5c2c39c1745d22068e29e1293616a95c7bba6d92e1f895ba5ad0dd885451c169"
 ["results/m245_m235_full220800_vectors_r1_20260825/manifest.sha256"]="ede5ef07816c235714c64f2ef8bf179564f48def5937c19ae7c100771157a5b4"
 ["contracts/m245_m235_full220800_vcs_contract_r1_20260825.json"]="44d08d778f88d284756097b203913fd78aeae1d372fda895ae31fd71c6bff396"
 ["results/m240_bn_pareto_independent_hammer_r1_20260825/SHA256SUMS"]="f9baa9402116b487c6a81be80d6f5d85db2250a0d684712cf1048e7d161d5f09"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "${task_path}" "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
(cd results/m245_m235_full220800_vectors_r1_20260825 && \
    sha256sum -c manifest.sha256) >"${task_run}/nested_vector_manifest_check.txt"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -cm assert -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m245_m235_full220800_vcs.f \
    -top tb_m245_m235_full220800 -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=24520260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq '^PASS M245 M235 checkpoint vectors=220800 mismatches=0 max_latency=[0-9]+ max_unstalled_accept_ii=[0-9]+ result_stalls=[1-9][0-9]* protocol_attacks=1 shared_multiplier_slots=1 multiply_ops_per_pair=5 lut_entries=64 newton_steps=1 tail_extrema_included=6 unchanged_production_rtl=true moment_finalizer=false event_equivalence=false system_speedup=false headline=false$' \
    "${task_run}/sim.log" || exit 30
task_latency="$(sed -n 's/^PASS M245 M235 .*max_latency=\([0-9][0-9]*\).*/\1/p' "${task_run}/sim.log")"
task_interval="$(sed -n 's/^PASS M245 M235 .*max_unstalled_accept_ii=\([0-9][0-9]*\).*/\1/p' "${task_run}/sim.log")"
[[ -n "${task_latency}" && -n "${task_interval}" \
    && "${task_latency}" -le 16 && "${task_interval}" -le 16 ]] || exit 31
for task_cover in cp_result cp_result_stall cp_fault_with_pending_result; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 32
done

{
    echo status=PASS_M245_M235_FULL220800_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo checkpoint_vectors=220800
    echo source_index_min=0
    echo source_index_max=220799
    echo tail_extrema_included=6
    echo integer_output_mismatches=0
    echo maximum_first_result_latency_cycles="${task_latency}"
    echo maximum_unstalled_accept_interval_cycles="${task_interval}"
    echo unchanged_production_rtl=true
    echo system_speedup=false
    echo headline=false
} >"${task_run}/m245_m235_full220800_vcs_receipt_r1.txt"
sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/SHA256SUMS"
echo PASS_M245_M235_FULL220800_EXACT_VCS >"${task_run}/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M245/M235 full220800 exact VCS sealed at ${task_run}"
