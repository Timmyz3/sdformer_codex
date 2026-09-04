#!/usr/bin/env bash
set -euo pipefail

m434_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m434_hw="$(cd "${m434_here}/../.." && pwd)"
m434_runner="$(realpath "${BASH_SOURCE[0]}")"
m434_run="${M434_VCS_RUN_DIR:-${m434_here}/vcs_run_exact_sha_r1b}"
m434_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m434_contract="contracts/m434_m433_independent_hammer_contract_r1_20260826.json"

m434_sha() { sha256sum "$1" | awk '{print $1}'; }
m434_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" ]] || exit 3
    [[ "$(m434_sha "${path}")" == "${expected}" ]] || exit 3
}

[[ ! -e "${m434_run}" ]] || exit 5
mkdir -p "${m434_run}"
m434_complete=0
trap 'm434_rc=$?; if [[ ${m434_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m434_rc}" >"${m434_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m434_hw}"

m434_expect "${m434_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m434_expect rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
m434_expect verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv e5a645a0e256c7d3a72f07f027ecaf2c1d136b433c45e13248592940aba85501
m434_expect tb_m433/tb_m433_exact_dualbank_coread_pwp_adapter.sv c6f2fb68e848b8348d52516bbe2b6c7ac7331ed313108f01703b4c08911f0cf8
m434_expect reviews/m434_m433_independent_hammer_r1_20260826/tb_m434_m433_independent_hammer.sv 245a6a01ed11ef5586b76900a2d9214bc7fc3f62f82238a9c5cc9df04cbd179e
m434_expect reviews/m434_m433_independent_hammer_r1_20260826/m434_m433_independent_vcs.f 1348c77a244ebf07d51f0716039d15edcfed82fcfad608a4b9e906203874fe76
m434_expect "${m434_contract}" a96e48584cdeafa63f971b4c9d495388ad14f1dfd7223a517598715b34a13281
m434_expect results/m433_exact_dualbank_coread_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 d57308dcabd40945f827fa0dfba0f18c7374f5d710722d2121e1084cd5b6d375
m434_expect results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/SHA256SUMS.seal.sha256 befbe0bc715539a3930456ec0b6c2967aea69c5aeb4ebc29dd221b4a1a6dc03b
m434_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd results/m433_exact_dualbank_coread_directed_vcs_r1_20260826
    sha256sum -c RUN_MANIFEST.sha256
    sha256sum -c RUN_MANIFEST.seal.sha256
) >"${m434_run}/m433_seal_check.log" 2>&1
(
    cd results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m434_run}/m427r3_seal_check.log" 2>&1

sha256sum \
    rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
    verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv \
    tb_m433/tb_m433_exact_dualbank_coread_pwp_adapter.sv \
    reviews/m434_m433_independent_hammer_r1_20260826/tb_m434_m433_independent_hammer.sv \
    reviews/m434_m433_independent_hammer_r1_20260826/m434_m433_independent_vcs.f \
    "${m434_contract}" \
    results/m433_exact_dualbank_coread_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md \
    >"${m434_run}/input_sha256.txt"
cp "${m434_contract}" "${m434_run}/contract.json"

export VCS_HOME="${m434_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${m434_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m434_run}/csrc" \
    -f reviews/m434_m433_independent_hammer_r1_20260826/m434_m433_independent_vcs.f \
    -top tb_m434_m433_independent_hammer \
    -o "${m434_run}/simv" >"${m434_run}/compile.log" 2>&1
m434_rc=$?
set -e
printf '%s\n' "${m434_rc}" >"${m434_run}/compile.rc"
[[ ${m434_rc} -eq 0 && -x "${m434_run}/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "${m434_run}/compile.log"; then
    exit 21
fi

set +e
"${m434_run}/simv" +ntb_random_seed=4340120260826 -no_save \
    -cm assert -assert "report=${m434_run}/assert.report" \
    >"${m434_run}/sim.log" 2>&1
m434_rc=$?
set -e
printf '%s\n' "${m434_rc}" >"${m434_run}/sim.rc"
[[ ${m434_rc} -eq 0 ]] || exit 22
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
        "${m434_run}/sim.log" "${m434_run}/assert.report"; then
    exit 23
fi
grep -Eq 'PASS M434 independent M433 hammer wide_codes=4096 narrow_codes=256 accepts=[0-9]+ retires=[0-9]+ explicit_fault_reset_discards=2 arithmetic_mismatches=0 metadata_mismatches=0 order_mismatches=0 attacks=14 same_cycle_leaks=0 sticky_leaks=0 legal_reloads=1 max_stall=[0-9]+ pop_push=[0-9]+ logical_wide_bytes=144 physical_interface_bytes=160 correction_port=false old_psum_port=false seed_fusion=false docs359_unchanged=true dc_go=true formality_go=true full_population_go=true headline=false' \
    "${m434_run}/sim.log" || exit 24

python3 - "${m434_run}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
hw = Path.cwd()
log = (run / "sim.log").read_text(errors="replace")
m = re.search(
    r"PASS M434 independent M433 hammer wide_codes=(\d+) narrow_codes=(\d+) "
    r"accepts=(\d+) retires=(\d+) explicit_fault_reset_discards=(\d+) "
    r"arithmetic_mismatches=(\d+) metadata_mismatches=(\d+) "
    r"order_mismatches=(\d+) attacks=(\d+) same_cycle_leaks=(\d+) "
    r"sticky_leaks=(\d+) legal_reloads=(\d+) max_stall=(\d+) pop_push=(\d+)",
    log)
if not m:
    raise SystemExit("M434 PASS payload missing")
keys = ["wide_codes", "narrow_codes", "accepted", "retired",
        "explicit_fault_reset_discards", "arithmetic_mismatches",
        "metadata_mismatches", "order_mismatches", "attacks",
        "same_cycle_leaks", "sticky_leaks", "legal_reloads",
        "max_stall", "same_cycle_pop_push"]
vcs = dict(zip(keys, map(int, m.groups())))
if vcs["wide_codes"] != 4096 or vcs["narrow_codes"] != 256:
    raise SystemExit("exhaustive code-space gate failed")
if vcs["accepted"] < 640 or vcs["max_stall"] < 16 or vcs["same_cycle_pop_push"] < 100:
    raise SystemExit("elastic stress gate failed")
if vcs["attacks"] != 14 or vcs["legal_reloads"] != 1:
    raise SystemExit("protocol matrix gate failed")
if any(vcs[k] for k in ("arithmetic_mismatches", "metadata_mismatches",
                        "order_mismatches", "same_cycle_leaks", "sticky_leaks")):
    raise SystemExit("independent mismatch/leak")
if vcs["accepted"] - vcs["retired"] != vcs["explicit_fault_reset_discards"]:
    raise SystemExit("unaccounted accepted transaction loss")

rtl_path = hw / "rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv"
rtl = rtl_path.read_text()
port_begin = rtl.index(") (") + 3
port_end = rtl.index("\n);", port_begin)
ports = rtl[port_begin:port_end].lower()
for forbidden in ("old_psum", "correction", "seed_fusion"):
    if forbidden in ports:
        raise SystemExit("forbidden semantic port: " + forbidden)
required = [
    "output logic [1151:0]           contribution_data",
    "high_data[lane*4 +: 4],",
    "low_data[lane*8 +: 8]",
    "{4{low_data[lane*8+7]}}",
    "assign illegal_now_w = illegal_request_w || illegal_reload_w;",
    "assign contribution_valid = !fault_q && !illegal_now_w && output_valid_q;",
    "2'b11: output_valid_q <= 1'b1;",
]
missing = [fragment for fragment in required if fragment not in rtl]
if missing:
    raise SystemExit("required RTL fragment missing: " + repr(missing))

addendum = json.loads((hw / "results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/m427r3_seed_fusion_semantic_audit_r1.json").read_text())
if addendum["decision"]["seed_fusion"] != "REVOKED_NON_EXECUTABLE_OPPORTUNITY":
    raise SystemExit("M427r3 seed-fusion revocation drift")
if addendum["decision"]["dual_coread_semantics"] != "SURVIVES":
    raise SystemExit("M427r3 dual co-read decision drift")

for code in range(4096):
    if (((code >> 8) & 15) << 8 | (code & 255)) != code:
        raise SystemExit("independent wide arithmetic proof failed")
for code in range(256):
    expected = code | (0xF00 if code & 0x80 else 0)
    signed = code - 256 if code & 0x80 else code
    if expected != (signed & 0xFFF):
        raise SystemExit("independent narrow arithmetic proof failed")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt = {
    "schema": "m434_m433_independent_hammer_vcs_receipt_v1",
    "status": "PASS_M434_INDEPENDENT_M433_VCS_HAMMER",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "vcs": vcs,
    "static_semantics": {
        "contribution_is_delta_only": True,
        "old_psum_port": False,
        "correction_port": False,
        "seed_fusion_port": False,
        "wide_lane_equation": "raw12={high4,low8}",
        "narrow_lane_equation": "signed12=sign_extend(signed8_low)",
        "m427r3_seed_fusion_revocation_preserved": True,
    },
    "loss_accounting": {
        "normal_legal_transaction_loss": 0,
        "fault_quarantined_then_reset_discarded": vcs["explicit_fault_reset_discards"],
        "accepted_minus_retired": vcs["accepted"] - vcs["retired"],
        "allowed_only_in_injected_fault_recovery": True,
    },
    "bandwidth": {
        "wide_logical_bytes_per_accept": 144,
        "wide_physical_interface_bytes_per_accept": 160,
        "frozen_shared96_bytes_per_cycle": 96,
        "free_upgrade": False,
        "resource_normalization_required": True,
    },
    "go_no_go": {
        "dc": "GO_STANDALONE_LOGIC_ONLY",
        "formality": "GO_RTL_TO_MAPPED_NETLIST_AFTER_DC",
        "full_population": "GO_INTEGRATION_ONLY_NOT_YET_PASS",
        "paper_or_headline": "NO_GO",
    },
    "claim_boundary": {
        "standalone_directed_vcs": True,
        "full_population_vcs": False,
        "cycles": False,
        "system_speedup": False,
        "ppa": False,
        "power": False,
        "date_headline": False,
    },
    "input_sha256": {
        "rtl": sha(rtl_path),
        "sva": sha(hw / "verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv"),
        "independent_tb": sha(hw / "reviews/m434_m433_independent_hammer_r1_20260826/tb_m434_m433_independent_hammer.sv"),
        "contract": sha(hw / "contracts/m434_m433_independent_hammer_contract_r1_20260826.json"),
        "docs359": sha(hw / "docs/359_DATE终局冻结_20260813.md"),
    },
}
(run / "m434_m433_independent_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m434_runner}" >"${m434_run}/runner_sha256.txt"
printf '%s\n' PASS_M434_INDEPENDENT_M433_VCS_HAMMER \
    >"${m434_run}/RUN_COMPLETE.txt"
m434_complete=1
echo "PASS_M434_INDEPENDENT_M433_VCS_HAMMER run=${m434_run}"
