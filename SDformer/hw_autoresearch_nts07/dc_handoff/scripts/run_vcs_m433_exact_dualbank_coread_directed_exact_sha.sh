#!/usr/bin/env bash
set -euo pipefail

m433_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m433_hw="$(cd "${m433_dc_root}/.." && pwd)"
m433_runner="$(realpath "${BASH_SOURCE[0]}")"
m433_run="${M433_VCS_RUN_DIR:-${m433_hw}/results/m433_exact_dualbank_coread_directed_vcs_r1_20260826}"
m433_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m433_contract="contracts/m433_exact_dualbank_coread_pwp_adapter_directed_vcs_contract_r1_20260826.json"

m433_sha() { sha256sum "$1" | awk '{print $1}'; }
m433_expect() {
    local m433_path=$1
    local m433_expected=$2
    [[ -f "${m433_path}" ]] || exit 3
    [[ "$(m433_sha "${m433_path}")" == "${m433_expected}" ]] || exit 3
}

[[ ! -e "${m433_run}" ]] || exit 5
mkdir -p "${m433_run}"
m433_complete=0
trap 'm433_rc=$?; if [[ ${m433_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m433_rc}" >"${m433_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m433_hw}"

m433_expect "${m433_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m433_expect rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
m433_expect verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv e5a645a0e256c7d3a72f07f027ecaf2c1d136b433c45e13248592940aba85501
m433_expect tb_m433/tb_m433_exact_dualbank_coread_pwp_adapter.sv c6f2fb68e848b8348d52516bbe2b6c7ac7331ed313108f01703b4c08911f0cf8
m433_expect dc_handoff/filelists/date_m433_exact_dualbank_coread_directed_vcs.f 1e0c2a2730f8aea47e99e53af4a09d0f77d335a40a72f0382d2cfca902beeee5
m433_expect "${m433_contract}" 2d1686ca63136225bf0b92a6f1695daa2fd35c9a39f7599b6a0ecc05f3bcf6ec
m433_expect results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/SHA256SUMS.seal.sha256 befbe0bc715539a3930456ec0b6c2967aea69c5aeb4ebc29dd221b4a1a6dc03b
m433_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m433_run}/m427r3_seal_check.log" 2>&1

sha256sum \
    rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
    verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv \
    tb_m433/tb_m433_exact_dualbank_coread_pwp_adapter.sv \
    dc_handoff/filelists/date_m433_exact_dualbank_coread_directed_vcs.f \
    "${m433_contract}" \
    results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m433_run}/input_sha256.txt"
cp "${m433_contract}" "${m433_run}/contract.json"
export VCS_HOME="${m433_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${m433_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m433_run}/csrc" \
    -f dc_handoff/filelists/date_m433_exact_dualbank_coread_directed_vcs.f \
    -top tb_m433_exact_dualbank_coread_pwp_adapter \
    -o "${m433_run}/simv" >"${m433_run}/compile.log" 2>&1
m433_rc=$?
set -e
printf '%s\n' "${m433_rc}" >"${m433_run}/compile.rc"
[[ ${m433_rc} -eq 0 && -x "${m433_run}/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "${m433_run}/compile.log"; then
    exit 21
fi

set +e
"${m433_run}/simv" +ntb_random_seed=4330120260826 -no_save \
    -cm assert -assert "report=${m433_run}/assert.report" \
    >"${m433_run}/sim.log" 2>&1
m433_rc=$?
set -e
printf '%s\n' "${m433_rc}" >"${m433_run}/sim.rc"
[[ ${m433_rc} -eq 0 ]] || exit 22
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
        "${m433_run}/sim.log" "${m433_run}/assert.report"; then
    exit 23
fi
grep -Eq 'PASS M433 exact dualbank coread standalone requests=[0-9]+ outputs=[0-9]+ narrow=[0-9]+ wide=[0-9]+ signed_boundaries=4 metadata_mismatches=0 arithmetic_mismatches=0 padding_mismatches=0 protocol_attacks=4 failclosed_leaks=0 stall_cycles=[0-9]+ max_stall=[0-9]+ pop_push=[0-9]+ consecutive_ii1=[0-9]+ legal_reloads=1 logical_wide_bytes=144 physical_interface_bytes=160 old_psum_preserved_downstream=true correction_fusion=false accuracy_changed=false cycles=false system_speedup=false ppa=false power=false headline=false' \
    "${m433_run}/sim.log" || exit 24

python3 - "${m433_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
log = (root / "sim.log").read_text(errors="replace")
match = re.search(
    r"PASS M433 exact dualbank coread standalone requests=(\d+) "
    r"outputs=(\d+) narrow=(\d+) wide=(\d+) signed_boundaries=(\d+) "
    r"metadata_mismatches=(\d+) arithmetic_mismatches=(\d+) "
    r"padding_mismatches=(\d+) protocol_attacks=(\d+) failclosed_leaks=(\d+) "
    r"stall_cycles=(\d+) max_stall=(\d+) pop_push=(\d+) "
    r"consecutive_ii1=(\d+) legal_reloads=(\d+)", log)
if not match:
    raise SystemExit("missing M433 PASS payload")
values = [int(x) for x in match.groups()]
keys = ["accepted_legal_requests", "retired_contributions", "narrow_requests",
        "wide_requests", "signed_boundaries", "metadata_mismatches",
        "arithmetic_mismatches", "padding_mismatches", "protocol_attacks",
        "failclosed_leaks", "stall_cycles", "max_stall_cycles",
        "simultaneous_pop_push", "consecutive_ii1_pairs", "legal_empty_reloads"]
directed = dict(zip(keys, values))
if directed["accepted_legal_requests"] < 300:
    raise SystemExit("too few legal requests")
if directed["narrow_requests"] == 0 or directed["wide_requests"] == 0:
    raise SystemExit("width coverage missing")
if directed["signed_boundaries"] != 4:
    raise SystemExit("signed boundary coverage missing")
if directed["max_stall_cycles"] < 8:
    raise SystemExit("long stall coverage missing")
if directed["simultaneous_pop_push"] < 32 or directed["consecutive_ii1_pairs"] < 32:
    raise SystemExit("II=1 elastic coverage missing")
if any(directed[k] for k in ("metadata_mismatches", "arithmetic_mismatches",
                             "padding_mismatches", "failclosed_leaks")):
    raise SystemExit("M433 mismatch or leak")
if directed["protocol_attacks"] != 4 or directed["legal_empty_reloads"] != 1:
    raise SystemExit("protocol coverage drift")
if directed["accepted_legal_requests"] - directed["retired_contributions"] != 1:
    raise SystemExit("expected exactly one buffered output quarantined by reload attack")

receipt = {
    "schema": "m433_exact_dualbank_coread_pwp_adapter_directed_vcs_receipt_v1",
    "status": "PASS_M433_EXACT_DUALBANK_COREAD_STANDALONE_DIRECTED_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "directed": directed,
    "quarantined_buffered_contributions": 1,
    "quarantine_reason": "The reload-busy attack suppresses same-cycle retirement and reset discards the inaccessible entry after sticky fail-closed is confirmed.",
    "interface": {"logical_wide_bytes_per_accept": 144,
                  "physical_interface_bytes_per_accept": 160,
                  "one_entry_elastic_output": True,
                  "simultaneous_pop_push": True,
                  "ready_high_ii": 1},
    "semantics": {"exact_signed12x96_delta": True,
                  "narrow_signed8_sign_extend": True,
                  "wide_high4_low8_concat": True,
                  "downstream_operation": "new_psum=old_psum+update_delta",
                  "seed_fusion": False,
                  "correction_operand": False,
                  "overwrites_old_psum": False,
                  "accuracy_changed": False},
    "verification": {"directed_vcs": True, "sva_failures": 0,
                     "arithmetic_mismatches": 0, "metadata_mismatches": 0,
                     "padding_mismatches": 0, "failclosed_leaks": 0},
    "claim_boundary": {"standalone": True, "exact": True,
                       "full_population_vcs": False,
                       "cycles_measured": False, "system_speedup": False,
                       "dc": False, "formality": False, "primetime": False,
                       "ppa": False, "power": False, "energy": False,
                       "date_headline": False, "dual_port_free_upgrade": False},
}
(root / "m433_exact_dualbank_coread_pwp_adapter_directed_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m433_runner}" >"${m433_run}/runner_sha256.txt"
printf '%s\n' PASS_M433_EXACT_DUALBANK_COREAD_STANDALONE_DIRECTED_VCS \
    >"${m433_run}/RUN_COMPLETE.txt"
find "${m433_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m433_run}/RUN_MANIFEST.sha256"
sha256sum "${m433_run}/RUN_MANIFEST.sha256" \
    >"${m433_run}/RUN_MANIFEST.seal.sha256"
m433_complete=1
echo "PASS_M433_EXACT_DUALBANK_COREAD_DIRECTED_VCS run=${m433_run}"
