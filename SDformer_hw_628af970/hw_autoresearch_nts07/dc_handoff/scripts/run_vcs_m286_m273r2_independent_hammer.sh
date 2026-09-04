#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M286_RUN_DIR:-${task_hw_root}/results/m286_m285_m273r2_independent_hammer_r1_exact_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="${M286_EXPECT_RTL_SHA:-11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d}"
    ["verif_m286/m286_m273r2_independent_assertions.sv"]="98cf2d213b551edb66148d29a1a4e7b45176b431cce361f249eac3af79392674"
    ["tb_m286/tb_m286_m273r2_independent_attack.sv"]="72febbcbd51ecc8fc83d3403802226fcb5d06f23df6a118c88a5ff5076ace95d"
    ["dc_handoff/filelists/date_m286_m273r2_independent_vcs.f"]="4124011403a5bd348875f1c995a27279a8e3113dd4a233d5d234148a3659cea7"
    ["contracts/m286_m285_m273r2_independent_hammer_contract_r1_20260825.json"]="ffbba931d9a377cd91ea033bb5b968b4be959904d287f2e34c12baa507a9d439"
    ["contracts/m285_m273r2_glitch_clean_zero_tile_vcs_contract_r1_20260825.json"]="0f0ebe41e70a2a599aa7202622e8fc472a912f8adb019c3e3ddf0357211445df"
    ["results/m285_m273r2_glitch_clean_zero_tile_vcs_r1_exact_20260825/RUN_MANIFEST.seal.sha256"]="98c735ad082417387ede446f02d2f1575f83b0a2d878739805458e0fd78c7252"
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

sha256sum -c \
    results/m285_m273r2_glitch_clean_zero_tile_vcs_r1_exact_20260825/RUN_MANIFEST.seal.sha256 \
    >"${task_run}/m285_author_seal_check.txt"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m286_m285_m273r2_independent_hammer_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m286_m273r2_independent_vcs.f \
    -top tb_m286_m273r2_independent_attack -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=28620260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true

task_pass='PASS M286 independent M285/M273r2 hammer n1=24 n4=39 gapped_n1=26 pressure_n40=1618 legal_protocol_glitches=0 legal_intra_half_changes=0 halfcycle_checks=1618 config_phase_accepts=1/1/1/1/1/1 raw_phase_accepts=40/40/40/40/40 result_phase_accepts=40/40/40/40/40 stage1_checks=9600 stage2_checks=6400 rne_ties=113/149 q8_sat=715 q24_sat=278 cfg_attacks=8 raw_attacks=5 n0_held_cycles=8 fault_edge_fifo_pop_push=1 fault_edge_result_order=1 quarantine_cycles=8 reference_mismatches=0 new_speedup=false dc=false system_speedup=false headline=false'
grep -Fqx "${task_pass}" "${task_run}/sim.log" || exit 30
for task_cover in cp_m286_fault_with_fifo_pop_push cp_m286_n0_held_after_fault \
        cp_m286_sticky_quarantine cp_m286_full_pop_push; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/assert.report" \
    "${task_run}/m286_m285_m273r2_independent_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
covers = {name: int(count) for name, count in re.findall(
    r"u_m286_sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match", report)}
required = {
    "cp_m286_fault_with_fifo_pop_push",
    "cp_m286_n0_held_after_fault",
    "cp_m286_sticky_quarantine",
    "cp_m286_full_pop_push",
}
if set(covers) != required or any(covers[name] < 1 for name in required):
    raise SystemExit("independent cover mismatch: %r" % covers)
receipt = {
    "schema": "m286_m285_m273r2_independent_vcs_receipt_v1",
    "status": "PASS_INDEPENDENT_REPAIR_ADMISSION",
    "reviewer_role": "independent_hammer_reviewer_not_m285_author",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "exact_sha": True,
    "open_source_rtl_tools_invoked": False,
    "production_rtl_sha256": "11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d",
    "verified": {
        "clean_cycles_N1": 24,
        "clean_cycles_N4": 39,
        "pressure_cycles_N40_fixed_one_in_eight_ready": 1618,
        "stage1_accumulator_checks": 9600,
        "stage2_event_checks": 6400,
        "numeric_or_order_mismatches": 0,
        "legal_halfcycle_checks": 1618,
        "legal_protocol_error_pulses": 0,
        "legal_intra_half_signal_changes": 0,
        "config_accept_phase_counts": [1, 1, 1, 1, 1, 1],
        "raw_accept_phase_counts": [40, 40, 40, 40, 40],
        "result_accept_phase_counts": [40, 40, 40, 40, 40],
        "n0_release_held_cycles": 8,
        "fault_edge_fifo_pop_push": 1,
        "fault_edge_result_order_checks": 1,
        "post_fault_quarantine_cycles": 8,
        "assertion_cover_matches": covers,
    },
    "claim_boundary": {
        "repair_admission_only": True,
        "new_speedup": False,
        "area_matched_fixed": False,
        "dc": False,
        "sta": False,
        "energy": False,
        "system_speedup": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M286_M285_M273R2_INDEPENDENT_EXACT_SHA_VCS" \
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
echo "PASS M286 independent exact-SHA VCS sealed at ${task_run}"
