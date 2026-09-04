#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M285_RUN_DIR:-${task_hw_root}/results/m285_m273r2_glitch_clean_zero_tile_vcs_r1_exact_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="${M285_EXPECT_RTL_SHA:-11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d}"
    ["verif_m273/m273_integrated_rank3_atlif_assertions.sv"]="b5909fd6af6cccb31d77da8376c0a9b2260abbd26d9284691c086419bdf09895"
    ["tb_m273/tb_m273_integrated_rank3_atlif.sv"]="4c7d11e7bbe6c185fcac35fa6950f8778b35c3ecc9f96691399768dfc41157bd"
    ["dc_handoff/filelists/date_m273_integrated_rank3_atlif_rtl.f"]="c99fe329c43276ce40f7027d54baeaaf747553c9f0b8d4419dcf8e7574b1a02d"
    ["dc_handoff/filelists/date_m273_integrated_rank3_atlif_directed_vcs.f"]="b47c7665ecf029fa996fdda4518d7afc2f494ad3103fbdeec4bdbd8cc9261399"
    ["contracts/m285_m273r2_glitch_clean_zero_tile_vcs_contract_r1_20260825.json"]="0f0ebe41e70a2a599aa7202622e8fc472a912f8adb019c3e3ddf0357211445df"
    ["results/m276_m273_integrated_rank3_atlif_independent_hammer_r1_20260825/m276_m273_independent_hammer_review_r1.json"]="d2297f8ba54ea4353cc9b2dfff78f3a86c48bd67c93b0ca8dd7608eadd8276d5"
    ["results/m276_m273_integrated_rank3_atlif_independent_hammer_r1_20260825/RUN_MANIFEST.sha256"]="5d169cc5c5706e6c84a6ab324dee9e7e8a4756d99615911224571c96889c0ccb"
    ["results/m276_m273_integrated_rank3_atlif_independent_hammer_r1_20260825/RUN_MANIFEST.seal.sha256"]="4d2cb1a02284ee9b684c4b2143495c453f5ce79751934e2fa4d822159b2fc89a"
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
    results/m276_m273_integrated_rank3_atlif_independent_hammer_r1_20260825/RUN_MANIFEST.seal.sha256 \
    >"${task_run}/m276_review_seal_check.txt"
printf '%s\n' \
    "NOT_REPLAYED_EXPECTED_SUPERSEDED_SOURCE_DRIFT__SEALED_REVIEW_MANIFEST_FILE_IS_HASH_BOUND" \
    >"${task_run}/m276_review_manifest_scope.txt"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m285_m273r2_glitch_clean_zero_tile_vcs_contract_r1_20260825.json \
    "${task_run}/contract.json"

python3 - <<'PY'
from pathlib import Path

rtl = Path("rtl_m273/m273_integrated_rank3_atlif.sv").read_text(encoding="utf-8")
required = [
    "protocol_error=protocol_error_q;",
    "result_valid=fifo_count_q!=0&&!protocol_error_q;",
    "product_push=product_valid_q&&fifo_credit&&!protocol_error_q;",
    "&&tiles_loaded_q!=0&&work_empty&&!raw_valid;",
    "if(config_accept&&!config_frame_error)",
    "if(raw_accept&&!raw_frame_error)",
    "if(fault_event)protocol_error_q<=1'b1;",
]
for fragment in required:
    if fragment not in rtl:
        raise SystemExit("missing M273r2 mechanism: " + fragment)
for forbidden in (
        "protocol_error=protocol_error_q||fault_event",
        "result_valid=fifo_count_q!=0&&!protocol_error_q&&!fault_event",
        "product_push=product_valid_q&&fifo_credit\n            &&!protocol_error_q&&!fault_event"):
    if forbidden in rtl:
        raise SystemExit("stale combinational fault gating: " + forbidden)
PY

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m273_integrated_rank3_atlif_directed_vcs.f \
    -top tb_m273_integrated_rank3_atlif -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=28520260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true

grep -Fq 'PASS M285 M273r2 integrated rank3 ATLIF directed clean_contexts=2 pressure_contexts=1 attacks=8 numerical_mismatches=0 clean_cycles_N1=24 clean_cycles_N4=39 pressure_cycles=1618 fifo_peak=16 overlap=1 product_replace=1 full_pop_push=1 legal_halfcycle_checks=1681 legal_protocol_error_pulses=0 intra_half_signal_changes=0 legal_config_accepts=18 legal_raw_accepts=225 zero_tile_release_fault=1 formula_domain_N_ge_1=true registered_fault_reporting=true new_speedup=false dc=false system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_clean_overlap cp_product_replace cp_result_stall \
        cp_fifo_full cp_full_pop_push cp_raw_backpressure cp_release_wait \
        cp_release cp_context_retire cp_config_fault cp_raw_fault \
        cp_zero_tile_release_fault cp_beat4; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/assert.report" \
    "${task_run}/m285_m273r2_author_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
covers = {}
for name, count in re.findall(
        r"u_sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match",
        report):
    covers[name] = int(count)
required = {
    "cp_clean_overlap": 34,
    "cp_product_replace": 222,
    "cp_result_stall": 1399,
    "cp_fifo_full": 1462,
    "cp_full_pop_push": 182,
    "cp_raw_backpressure": 1125,
    "cp_release_wait": 313,
    "cp_release": 3,
    "cp_context_retire": 3,
    "cp_config_fault": 4,
    "cp_raw_fault": 3,
    "cp_zero_tile_release_fault": 2,
    "cp_beat4": 45,
}
if covers != required:
    raise SystemExit("assertion cover drift: %r" % covers)
receipt = {
    "schema": "m285_m273r2_glitch_clean_zero_tile_author_vcs_receipt_v1",
    "status": "PASS_M285_M273R2_EXACT_SHA_SYNOPSYS_VCS_AWAITING_INDEPENDENT_HAMMER",
    "role": "author_repair_not_independent_review",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "exact_sha": True,
    "open_source_rtl_tools_invoked": False,
    "production_rtl_sha256":
        "11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d",
    "directed_campaign": {
        "clean_contexts": 2,
        "pressure_contexts": 1,
        "legacy_config_attacks": 4,
        "legacy_raw_attacks": 3,
        "zero_tile_release_attacks": 1,
        "numeric_and_order_mismatches": 0,
        "clean_cycles_N1": 24,
        "clean_cycles_N4": 39,
        "pressure_cycles_N40_fixed_one_in_eight_ready": 1618,
        "fifo_peak": 16,
        "full_fifo_pop_push_cycles": 182,
        "legal_halfcycle_checks": 1681,
        "legal_protocol_error_pulses": 0,
        "legal_intra_half_signal_changes": 0,
        "legal_config_accepts": 18,
        "legal_raw_accepts": 225,
        "assertion_failures": 0,
        "assertion_cover_matches": covers,
    },
    "repairs": {
        "fault_reporting_registered": True,
        "fault_based_combinational_issue_gating_removed": True,
        "fault_based_combinational_result_valid_gating_removed": True,
        "offending_frame_payload_not_committed": True,
        "zero_tile_release_ready": False,
        "zero_tile_release_accept": False,
        "zero_tile_release_registered_sticky_fault": True,
        "clean_cycle_formula": "5*N+19",
        "clean_cycle_formula_domain": "N>=1_and_gap_free_ready_high",
    },
    "claim_boundary": {
        "new_speedup": False,
        "area_matched_fixed": False,
        "dc": False,
        "sta": False,
        "power": False,
        "energy": False,
        "paper_ppa": False,
        "system_speedup": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M285_M273R2_EXACT_SHA_SYNOPSYS_VCS" \
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
echo "PASS M285/M273r2 exact VCS sealed at ${task_run}"
