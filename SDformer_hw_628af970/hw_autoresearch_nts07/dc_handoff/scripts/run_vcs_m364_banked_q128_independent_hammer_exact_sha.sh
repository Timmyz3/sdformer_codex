#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M364_RUN_DIR:-${task_hw_root}/results/m364_m363_banked_q128_independent_hammer_vcs_r1_20260825}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m348/m348_exact_q128_signed_residual_matcher.sv"]="960b72268d526baad7e7d74cd159f2a9fbb286abac025678f239af3f5147eb1f"
    ["rtl_m356/m356_failclosed_q128_signed_residual_matcher.sv"]="3a5dd5f7e3602f4f27f0744ebb13807877b4cc8a45e44dda11afe77dcae67fb8"
    ["rtl_m363/m363_banked_q128_exact_signed_residual_matcher.sv"]="257084916c312c9db4e2d6ad59a4fe20fb604fa6c9d0a0573039339e9614879d"
    ["verif_m363/m363_banked_q128_exact_signed_residual_matcher_assertions.sv"]="8cb774078a5251d9403732cfa40ccf55c6a7db836eeb1a9f5beaa20c34c7e8f6"
    ["verif_m364/m364_banked_q128_independent_hammer_assertions.sv"]="12ef157e3146e8df10d7801467477fece86240627b7a133098ecc193b5d5d4e5"
    ["tb_m364/tb_m364_banked_q128_independent_hammer.sv"]="eea95e4cb1f095ae55d27a064b6d9caf822c7d4bddc492d491c82293ac3f21a4"
    ["dc_handoff/filelists/date_m364_banked_q128_independent_hammer_vcs.f"]="9c9263991d964bdce91c374acb0ffea0601b6a6812563f7e5d0ff28bd9248a54"
    ["contracts/m363_banked_q128_exact_signed_residual_matcher_directed_vcs_contract_r1_20260825.json"]="8eada82433b15ee6b33ebf087eaaf9c4a3d18f73cdfb8dfbcb05153f62ae9704"
    ["contracts/m364_m363_banked_q128_independent_hammer_contract_r1_20260825.json"]="aa943bf3b1fe4fb61bcf44cab63ec8bd7cc704bccabcbef821220ec4fc3eee4b"
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
cp contracts/m364_m363_banked_q128_independent_hammer_contract_r1_20260825.json \
    "${task_run}/contract.json"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m364_banked_q128_independent_hammer_vcs.f \
    -top tb_m364_banked_q128_independent_hammer \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=36420260825 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M364 independent hammer m363_transactions=3000 m356_transactions=3000 .*m363_mismatches=0 m356_mismatches=0 pairwise_mismatches=0 lowest_id_tie_id=2 .*m363_latency_min=4 .*m356_latency_min=128 .*deferred_cfg_handshakes=1 sticky_attack_cycles=64 sticky_cfg_handshakes=0 sticky_input_handshakes=0 mid_pipeline_resets=1 flushed_tokens=4 numeric_signed_order_equivalent=true elastic_ii1=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 30
for task_cover in cp_all_four_elastic_slots_full cp_long_output_stall \
        cp_bubble_refill cp_deferred_configuration cp_sticky_error_freeze; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/sim.log" \
    "${task_run}/m364_m363_banked_q128_independent_hammer_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
pattern = re.compile(
    r"PASS M364 independent hammer m363_transactions=(\d+) "
    r"m356_transactions=(\d+) m363_mismatches=(\d+) "
    r"m356_mismatches=(\d+) pairwise_mismatches=(\d+) "
    r"lowest_id_tie_id=(\d+) use=(\d+) fallback=(\d+) "
    r"mixed_signed=(\d+) exact=(\d+) m363_stalls=(\d+) "
    r"m356_stalls=(\d+) m363_bubbles=(\d+) m356_bubbles=(\d+) "
    r"m363_max_accept_run=(\d+) m363_max_retire_run=(\d+) "
    r"m356_max_accept_run=(\d+) m356_max_retire_run=(\d+) "
    r"m363_latency_min=(\d+) m363_latency_max=(\d+) "
    r"m356_latency_min=(\d+) m356_latency_max=(\d+) "
    r"cfg_block_cycles=(\d+) deferred_cfg_handshakes=(\d+) "
    r"sticky_attack_cycles=(\d+) sticky_cfg_handshakes=(\d+) "
    r"sticky_input_handshakes=(\d+) mid_pipeline_resets=(\d+) "
    r"flushed_tokens=(\d+) numeric_signed_order_equivalent=true "
    r"elastic_ii1=true system_speedup=false headline=false")
match = pattern.search(text)
if not match:
    raise SystemExit("missing M364 PASS payload")
values = [int(value) for value in match.groups()]
keys = [
    "m363_transactions", "m356_transactions", "m363_mismatches",
    "m356_mismatches", "pairwise_mismatches", "lowest_id_tie_id",
    "use_pwp", "fallback", "mixed_signed", "exact", "m363_stalls",
    "m356_stalls", "m363_bubbles", "m356_bubbles",
    "m363_max_accept_run", "m363_max_retire_run",
    "m356_max_accept_run", "m356_max_retire_run",
    "m363_latency_min", "m363_latency_max", "m356_latency_min",
    "m356_latency_max", "cfg_block_cycles", "deferred_cfg_handshakes",
    "sticky_attack_cycles", "sticky_cfg_handshakes",
    "sticky_input_handshakes", "mid_pipeline_resets", "flushed_tokens",
]
metrics = dict(zip(keys, values))
required = {
    "m363_transactions": 3000,
    "m356_transactions": 3000,
    "m363_mismatches": 0,
    "m356_mismatches": 0,
    "pairwise_mismatches": 0,
    "lowest_id_tie_id": 2,
    "m363_latency_min": 4,
    "m356_latency_min": 128,
    "deferred_cfg_handshakes": 1,
    "sticky_attack_cycles": 64,
    "sticky_cfg_handshakes": 0,
    "sticky_input_handshakes": 0,
    "mid_pipeline_resets": 1,
    "flushed_tokens": 4,
}
for key, expected in required.items():
    if metrics[key] != expected:
        raise SystemExit(f"M364 gate failed: {key}={metrics[key]} expected={expected}")
for key in ("use_pwp", "fallback", "mixed_signed", "exact",
            "m363_bubbles", "m356_bubbles"):
    if metrics[key] <= 0:
        raise SystemExit(f"M364 coverage gate failed: {key}")
if metrics["m363_stalls"] < 64 or metrics["m356_stalls"] < 64:
    raise SystemExit("M364 stall gate failed")
if metrics["m363_max_accept_run"] < 256 or metrics["m363_max_retire_run"] < 256:
    raise SystemExit("M364 M363 II1 gate failed")
if metrics["m356_max_accept_run"] < 256 or metrics["m356_max_retire_run"] < 128:
    raise SystemExit("M364 M356 II1 gate failed")

receipt = {
    "schema": "m364_m363_banked_q128_independent_hammer_vcs_receipt_v1",
    "status": "PASS_M364_FRESH_EXACT_SHA_INDEPENDENT_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 36420260825,
    "metrics": metrics,
    "attacks": {
        "four_stage_elastic_long_stall_and_bubbles": True,
        "configuration_waits_for_true_pipeline_empty": True,
        "active_catalog_bad_reload_fail_closed": True,
        "sticky_error_catalog_next_group_active_frozen": True,
        "reset_while_all_four_slots_full": True,
        "post_reset_stale_output": False,
    },
    "semantic_comparison": {
        "independent_q128_reference": True,
        "m356_separate_instance": True,
        "same_3000_stimuli": True,
        "lowest_id_tie": True,
        "signed_plus_minus_masks": True,
        "in_order": True,
        "m363_changes_structure_not_q128_semantics": True,
    },
    "claim_boundary": {
        "functional_matcher_rtl": True,
        "four_cycle_latency_is_speedup": False,
        "system_speedup": False,
        "complete_pwp_conv": False,
        "energy": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' "PASS_M364_M363_BANKED_Q128_INDEPENDENT_HAMMER_VCS" \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M364 independent exact-SHA VCS sealed at ${task_run}"
