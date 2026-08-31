#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M356_RUN_DIR:-${task_hw_root}/results/m356_failclosed_q128_signed_residual_matcher_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m348/m348_exact_q128_signed_residual_matcher.sv"]="960b72268d526baad7e7d74cd159f2a9fbb286abac025678f239af3f5147eb1f"
    ["rtl_m356/m356_failclosed_q128_signed_residual_matcher.sv"]="3a5dd5f7e3602f4f27f0744ebb13807877b4cc8a45e44dda11afe77dcae67fb8"
    ["verif_m356/m356_failclosed_q128_signed_residual_matcher_assertions.sv"]="f1c5e4712de883591f82c30cf8fc71ed4a35f82023ccfd09c9c529ae3efdb273"
    ["tb_m356/tb_m356_failclosed_q128_signed_residual_matcher.sv"]="a6d28ff242a0b45e7daa98333ab5bb49fafab326345227d14a12d5b957d74887"
    ["dc_handoff/filelists/date_m356_failclosed_q128_signed_residual_matcher_vcs.f"]="b097d985c24b6d4989d34af9d42ce031bd2a70ea204bcf27ca1e81150d8a577c"
    ["contracts/m356_m348_failclosed_q128_signed_residual_matcher_directed_vcs_contract_r1_20260825.json"]="1ba25dba44104be5cc5c2dbe328983440c1f0cbd3607b17d0b8e1244c7091700"
    ["results/m350_m348_q128_exact_signed_residual_matcher_independent_hammer_r1_20260825/m350_independent_hammer_review_r1.json"]="bc382d390fac51da493207caa15acb680a2f9eac1a5f1e22627c901bcd34f75f"
    ["results/m348_exact_q128_signed_residual_matcher_vcs_r1_20260825/m348_exact_q128_signed_residual_matcher_vcs_receipt_r1.json"]="f4c13164fa59803fa48770b762dbbe1ee9da2b62b4f3ee0783d7b8f222f1e084"
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
cp contracts/m356_m348_failclosed_q128_signed_residual_matcher_directed_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m356_failclosed_q128_signed_residual_matcher_vcs.f \
    -top tb_m356_failclosed_q128_signed_residual_matcher \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=35620260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M356 failclosed q128 exact signed matcher transactions=3000 use=[1-9][0-9]* fallback=[1-9][0-9]* mixed=[1-9][0-9]* exact=[1-9][0-9]* transient_ties=[1-9][0-9]* stalls=[1-9][0-9]* protocol_attacks=5 sticky_reconfiguration_attempts=40 max_accept_run=[2-9][0-9][0-9]+ max_retire_run=[1-9][0-9][0-9]+ latency_min=128 latency_max=[1-9][0-9]* mismatches=0 ii1=true center_id=true signed_residual=true exact_fallback=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_config cp_use_pwp cp_fallback \
        cp_positive_signed_residual cp_output_stall; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m356_failclosed_q128_signed_residual_matcher_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"PASS M356 failclosed q128 exact signed matcher transactions=(\d+) use=(\d+) "
    r"fallback=(\d+) mixed=(\d+) exact=(\d+) transient_ties=(\d+) stalls=(\d+) "
    r"protocol_attacks=(\d+) sticky_reconfiguration_attempts=(\d+) "
    r"max_accept_run=(\d+) max_retire_run=(\d+) "
    r"latency_min=(\d+) latency_max=(\d+) mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M356 PASS payload")
values = [int(value) for value in match.groups()]
receipt = {
    "schema": "m356_failclosed_q128_signed_residual_matcher_vcs_receipt_v1",
    "status": "PASS_M356_EXACT_SHA_VCS_STICKY_ERROR_FAILCLOSED_Q128_MATCHER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "transactions": values[0],
    "use_pwp": values[1],
    "fallback_bit_sparse": values[2],
    "mixed_plus_minus_residual": values[3],
    "exact_pattern_use": values[4],
    "transient_best_distance_tie_observations": values[5],
    "stalled_output_cycles": values[6],
    "protocol_attacks": values[7],
    "sticky_reconfiguration_attempts": values[8],
    "sticky_reconfiguration_handshakes": 0,
    "maximum_consecutive_input_accepts": values[9],
    "maximum_consecutive_output_retires": values[10],
    "minimum_observed_latency_cycles": values[11],
    "maximum_observed_latency_cycles": values[12],
    "numeric_or_order_mismatches": values[13],
    "architecture": {
        "patterns": 128,
        "configuration_beats": 8,
        "pipeline_stages": 128,
        "ii1_no_stall": True,
        "tie_break": "lowest center ID",
        "center_id_output": True,
        "plus_minus_residual_masks": True,
        "exact_bit_sparse_fallback": True,
        "sticky_protocol_error_blocks_configuration_until_reset": True,
    },
    "claim_boundary": {
        "functional_matcher_rtl": True,
        "complete_pwp_conv": False,
        "finite_queue_cycle_match": False,
        "dc_area_fmax": False,
        "physical_sram": False,
        "energy": False,
        "system_speedup": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M356_FAILCLOSED_Q128_SIGNED_RESIDUAL_MATCHER_SYNOPSYS_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M356 exact VCS sealed at ${task_run}"
