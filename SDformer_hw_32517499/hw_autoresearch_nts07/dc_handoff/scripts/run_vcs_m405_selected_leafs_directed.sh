#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M405_RUN_DIR:-${task_hw_root}/results/m405_selected_leafs_directed_vcs_r1_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}/elastic" "${task_run}/prefix"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv"]="819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744"
    ["rtl_m405/m405_q32_serial16_zero_stop_controller.sv"]="bf509984690e8b6fec83477df7c6c7e223c4c08a984fa82ca3d3348f602af613"
    ["rtl_m405/m405_q32_elastic_selected_slice.sv"]="5bf3c10f892751487688ceecdd9b68a4ad784e3a9a52cb60fe8cb96d66669184"
    ["verif_m405/m405_exact_elastic_pwp_issue_adapter_assertions.sv"]="372aaac64c02cc91ed97029338830cd5712202f7db9c5b550215428f633d1f4e"
    ["verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv"]="4cbb1557dbda9b525893dc4d2a1627e38a5e2d73feda827228e9e5b5d3ad023d"
    ["tb_m405/tb_m405_exact_elastic_pwp_issue_adapter.sv"]="20572e65667258196dd48a67e5bcd4f4ff244ef8568a8f2dfeaacfbc03bc7bb1"
    ["tb_m405/tb_m405_q32_serial16_zero_stop_controller.sv"]="f1c188d9887e62d68126b3c17b44174aa60b4026065f6ba9771cf7d74baf4815"
    ["dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f"]="039288d349194b4e638732cf4dee67772b3d79c4d8df691aca1c5608950a88d5"
    ["dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f"]="0b2e19735cfb8ba8060bd5bbf161b808548e1361b9f8b38cd44600cdb689bbf1"
    ["dc_handoff/filelists/date_m405_selected_slice_rtl.f"]="b3795bbbf04c36d2baad79c2e9dfd93393e8f04315a85864c3b7ca129d72952e"
    ["contracts/m405_selected_leafs_directed_vcs_contract_r1_20260826.json"]="fc1f6eb425a78498c8d2071b6e90ac9f8ad4458892e1f2eb291136400892ab64"
    ["results/m403_m401_elastic_pwp_selected_rtl_prereview_r1_20260826/SHA256SUMS.seal.sha256"]="d97b6132a70e32e676eaada660e4315ef1d62e6577d5e8f7d664e409f0b565ea"
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
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m405_selected_leafs_directed_vcs_contract_r1_20260826.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/elastic/csrc" \
    -f dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f \
    -top tb_m405_exact_elastic_pwp_issue_adapter \
    -o "${task_run}/elastic/simv" \
    >"${task_run}/elastic/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/elastic/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/elastic/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' \
    "${task_run}/elastic/compile.log" && exit 21 || true

set +e
"${task_run}/elastic/simv" +ntb_random_seed=405120260826 \
    -no_save -cm assert \
    -assert report="${task_run}/elastic/assert.report" \
    >"${task_run}/elastic/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/elastic/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/elastic/sim.log" "${task_run}/elastic/assert.report" \
    && exit 23 || true
grep -Eq 'PASS M405A exact elastic PWP blocks=386 narrow=100 wide=286 contributions=672 .* no_gap_wide=66 protocol_attacks=4 atomic_leaks=0' \
    "${task_run}/elastic/sim.log" || exit 24

set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/prefix/csrc" \
    -f dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f \
    -top tb_m405_q32_serial16_zero_stop_controller \
    -o "${task_run}/prefix/simv" \
    >"${task_run}/prefix/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/prefix/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/prefix/simv" ]] || exit 30
grep -Eiq 'Warning-\[|Error-\[|^Error' \
    "${task_run}/prefix/compile.log" && exit 31 || true

set +e
"${task_run}/prefix/simv" +ntb_random_seed=405220260826 \
    -no_save -cm assert \
    -assert report="${task_run}/prefix/assert.report" \
    >"${task_run}/prefix/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/prefix/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 32
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/prefix/sim.log" "${task_run}/prefix/assert.report" \
    && exit 33 || true
grep -Eq 'PASS M405B q32 serial16 rows=64 pass0=64 pass1=[1-9][0-9]* early=[1-9][0-9]* outputs=64 .* protocol_attacks=3 tie_lowest_id=true source_scratch_reads=0 descriptor_scratch=0 task_adjacency_observed=true' \
    "${task_run}/prefix/sim.log" || exit 34

for task_report in "${task_run}/elastic/assert.report" \
                   "${task_run}/prefix/assert.report"; do
    grep -Eq 'Failures: 0' "${task_report}" || exit 40
done

python3 - "${task_run}/elastic/sim.log" "${task_run}/prefix/sim.log" \
    "${task_run}/m405_selected_leafs_directed_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

elastic = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
prefix = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
ma = re.search(r"PASS M405A exact elastic PWP blocks=(\d+) narrow=(\d+) wide=(\d+) contributions=(\d+) stalls=(\d+) no_gap_wide=(\d+) protocol_attacks=(\d+) atomic_leaks=(\d+)", elastic)
mb = re.search(r"PASS M405B q32 serial16 rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) outputs=(\d+) stalls=(\d+) protocol_attacks=(\d+)", prefix)
if not ma or not mb:
    raise SystemExit("missing M405 PASS payload")
a = [int(v) for v in ma.groups()]
b = [int(v) for v in mb.groups()]
receipt = {
    "schema": "m405_selected_leafs_directed_vcs_receipt_v1",
    "status": "PASS_M405_DIRECTED_LEAF_BRINGUP_ONLY",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "elastic": {"blocks": a[0], "narrow_blocks": a[1],
                "wide_blocks": a[2], "contributions": a[3],
                "stall_cycles": a[4], "no_gap_wide_blocks": a[5],
                "protocol_attacks": a[6], "atomic_leaks": a[7]},
    "prefix": {"rows": b[0], "pass0_tasks": b[1],
               "pass1_tasks": b[2], "early_stops": b[3],
               "outputs": b[4], "stall_cycles": b[5],
               "protocol_attacks": b[6]},
    "claim_boundary": {"directed_leaf_rtl": True,
        "full_static_codec_vcs": False, "full_real_q32_miter_vcs": False,
        "m384_q32_regression": False, "integrated_vcs": False,
        "dc": False, "rtl_measured_speedup": False,
        "system_speedup": False, "headline": False}
}
Path(sys.argv[3]).write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n",
                             encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' PASS_M405_SELECTED_EXACT_LEAFS_DIRECTED_SYNOPSYS_VCS \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M405 directed VCS sealed at ${task_run}"
