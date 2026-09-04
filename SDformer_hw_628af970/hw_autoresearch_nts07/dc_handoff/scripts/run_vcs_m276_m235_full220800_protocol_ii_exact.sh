#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${task_hw_root}/results/m276_m235_full220800_protocol_ii_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv"]="ec0bf05540433ecfc436eac63b41a4cecf4cc53b46533f2fd4f44c7eb70bd611"
    ["verif_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine_assertions.sv"]="2c138fc3902a14d0d53a709e5533e11b686bd724bc986e059e4e841137ac8a45"
    ["verif_m276/m276_m235_full220800_protocol_ii_assertions.sv"]="2b81fa77044c224c4774e2ec41dbdf58899cb2f1c9172fae42e49b3821083871"
    ["tb_m276/tb_m276_m235_full220800_protocol_ii.sv"]="03ae8e5c8854389691b975521755a6459b47d6351d7894ea3776701626d9cdbb"
    ["dc_handoff/filelists/date_m276_m235_full220800_protocol_ii_vcs.f"]="8cd90d30ad419b60481bf9f927bf2809676f19c405d24f72b8994c539c579d63"
    ["system_simulator/scripts/audit_m276_m235_full220800_protocol_ii.py"]="a3f8df315779c357ae25ef6659c482fd7f4ae76930583c519e2bc050ababd428"
    ["dc_handoff/replay/m276_m235_full220800_protocol_ii_REPLAY.md"]="4d2e08de0c6f9c6a0cff04a68dc6e7b237e512d432c6c122afea2d23ffc14572"
    ["contracts/m276_m235_full220800_protocol_ii_vcs_contract_r1_20260825.json"]="f565198e5dcbec3422c88d7d5e20ad6ee027f21b9623ea53af7509506b7bf5a2"
    ["results/m245_m235_full220800_vectors_r1_20260825/m245_m235_full220800_vectors.csv"]="81fbb84952fd79fc03a5b8660e839e27f06dec4e5fcb4b2c0cf770966c42ca29"
    ["results/m245_m235_full220800_vectors_r1_20260825/manifest.sha256"]="ede5ef07816c235714c64f2ef8bf179564f48def5937c19ae7c100771157a5b4"
    ["contracts/m245_m235_full220800_vcs_contract_r1_20260825.json"]="44d08d778f88d284756097b203913fd78aeae1d372fda895ae31fd71c6bff396"
    ["results/m245_m235_full220800_vcs_r1_exact_20260825/SHA256SUMS"]="a10da0a8ffe7b30665cb8fb3270603448166f8ac3f6e51d4831765a210b35272"
    ["results/m246_m245_full220800_independent_hammer_r1_20260825/SHA256SUMS"]="8a0f07a74d49229019dde0ae7c69ea2fdc1040d4723d82d5ccaefe49790795eb"
    ["results/m246_m245_full220800_independent_hammer_r1_20260825/m246_m245_full220800_independent_hammer_r1.json"]="a22ba173dec7cf52133b50e6c168662d66b6976e9dac32a302fcc72384135fc2"
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
sha256sum -c results/m245_m235_full220800_vcs_r1_exact_20260825/SHA256SUMS \
    >"${task_run}/m245_run_manifest_check.txt"
(cd results/m246_m245_full220800_independent_hammer_r1_20260825 && \
    sha256sum -c SHA256SUMS) >"${task_run}/m246_review_manifest_check.txt"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp dc_handoff/replay/m276_m235_full220800_protocol_ii_REPLAY.md \
    "${task_run}/REPLAY.md"
cp contracts/m276_m235_full220800_protocol_ii_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -cm assert -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m276_m235_full220800_protocol_ii_vcs.f \
    -top tb_m276_m235_full220800_protocol_ii -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=27620260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true

python3 system_simulator/scripts/audit_m276_m235_full220800_protocol_ii.py \
    --vectors results/m245_m235_full220800_vectors_r1_20260825/m245_m235_full220800_vectors.csv \
    --sim-log "${task_run}/sim.log" \
    --assert-report "${task_run}/assert.report" \
    --coverage-output "${task_run}/m276_m235_full220800_protocol_ii_coverage_r1.json" \
    --receipt-output "${task_run}/m276_m235_full220800_protocol_ii_vcs_receipt_r1.json"

python3 - "${task_run}/m276_m235_full220800_protocol_ii_vcs_receipt_r1.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert receipt["status"] == "PASS_M276_M235_FULL220800_PROTOCOL_II_EXACT_VCS"
assert receipt["frozen_corpus_vectors"] == 220800
assert receipt["integer_output_mismatches"] == 0
assert receipt["first_result_latency_cycles"] == 8
assert receipt["intrinsic_unstalled_accept_interval_cycles"] == 9
assert receipt["new_speedup"] is False
assert receipt["system_speedup"] is False
assert receipt["headline"] is False
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M276_M235_FULL220800_PROTOCOL_II_EXACT_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"

find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/SHA256SUMS"

task_complete=1
echo "PASS M276/M235 full220800 protocol/II exact VCS sealed at ${task_run}"
