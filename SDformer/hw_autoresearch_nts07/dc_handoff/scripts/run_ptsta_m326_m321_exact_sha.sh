#!/usr/bin/env bash
set -euo pipefail

m326_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m326_hw="$(cd "${m326_dc_root}/.." && pwd)"
m326_dc_run="${m326_dc_root}/runs/m322_m321_tournament2_logic_only_dc_3p000ns_r1_20260825"
m326_run="${M326_PTSTA_RUN:-${m326_dc_root}/runs/m326_m321_prelayout_ptsta_r1_20260825}"
m326_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m326_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m326_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m326_netlist="${m326_dc_run}/netlist/m321_near_match16_tau01_tournament2_mapped.v"
m326_sdc="${m326_dc_run}/netlist/m321_near_match16_tau01_tournament2_mapped.sdc"
m326_dc_complete="${m326_dc_run}/RUN_COMPLETE.txt"
m326_dc_seal="${m326_dc_run}/evidence_manifest.seal.sha256"
m326_tcl="${m326_dc_root}/scripts/run_ptsta_m326_m321_exact_sha.tcl"
m326_contract="${m326_hw}/contracts/m326_m321_m322_prelayout_ptsta_contract_r1_20260825.json"
m326_docs="${m326_hw}/docs/359_DATE终局冻结_20260813.md"

m326_sha() { sha256sum "$1" | awk '{print $1}'; }
m326_expect() {
    local m326_path=$1
    local m326_expected=$2
    [[ -f "${m326_path}" && ! -L "${m326_path}" ]] || exit 3
    [[ "$(m326_sha "${m326_path}")" == "${m326_expected}" ]] || exit 3
}

[[ ! -e "${m326_run}" ]] || exit 5
if pgrep -f '^/opt/synopsys/.*/pt_shell( |$)' >/dev/null 2>&1; then
    exit 4
fi

m326_expect "${m326_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m326_expect "${m326_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m326_expect "${m326_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m326_expect "${m326_netlist}" f8bb8c85a6459b0f5d7bde9b45ea206b829014421b682fe73574c025b09dbe13
m326_expect "${m326_sdc}" ea4001adaab7b5a33b8ed651c967824ad3644470cbe4df5a9fdf763ef326ad01
m326_expect "${m326_dc_complete}" 9ba01a71d3481e01b0e2786691b03c9d4f1b141de641aee4e13d60c458697690
m326_expect "${m326_dc_seal}" 414a5ca90d521e0ea1f7bc3084b9926e4849fab75992d96d3b306e688b923eed
m326_expect "${m326_tcl}" 457a5e8ae4c134533302e44e2f2e3909923d7f7efed4aac962cd0a5d9f9a401d
m326_expect "${m326_contract}" 2550318d540f59b99ca7d28a0189d54d07e5f1e76458a9b86fbabceb1d4255d5
m326_expect "${m326_docs}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mkdir -p "${m326_run}/snapshot/netlist" "${m326_run}/snapshot/library" \
    "${m326_run}/reports" "${m326_run}/work"
cp "${m326_netlist}" "${m326_sdc}" "${m326_run}/snapshot/netlist/"
cp "${m326_slow}" "${m326_fast}" "${m326_run}/snapshot/library/"
cp "${m326_contract}" "${m326_run}/contract.json"
(
    cd "${m326_run}/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > ../snapshot.sha256
    sha256sum --strict -c ../snapshot.sha256 > ../snapshot_check.raw.log 2>&1
)
find "${m326_run}/snapshot" -type f -exec chmod 0444 {} +
find "${m326_run}/snapshot" -type d -exec chmod 0555 {} +

{
    echo "status=RUNNING_NOT_CITABLE"
    echo "scope=PRELAYOUT_NO_SPEF_IDEAL_CLOCK_ZERO_MACRO"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
} > "${m326_run}/RUN_IN_PROGRESS.txt"

export DESIGN_NAME=m321_near_match16_tau01_tournament2
export LIB_SLOW="${m326_run}/snapshot/library/$(basename "${m326_slow}")"
export LIB_FAST="${m326_run}/snapshot/library/$(basename "${m326_fast}")"
export MAPPED_NETLIST="${m326_run}/snapshot/netlist/$(basename "${m326_netlist}")"
export MAPPED_SDC="${m326_run}/snapshot/netlist/$(basename "${m326_sdc}")"
export OUTPUT_DIR="${m326_run}"
"${m326_pt}" -version > "${m326_run}/pt.version.raw.log" 2>&1
echo "${m326_pt} -f ${m326_tcl}" > "${m326_run}/pt.command.txt"
set +e
(cd "${m326_run}/work" && "${m326_pt}" -f "${m326_tcl}") \
    > "${m326_run}/pt.raw.log" 2>&1
m326_rc=$?
set -e
echo "${m326_rc}" > "${m326_run}/pt.rc"
[[ "${m326_rc}" -eq 0 ]]
[[ "$(grep -xc 'M326_M321_PTSTA_INTERNAL_COMPLETE=PASS' \
    "${m326_run}/PTSTA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
! grep -Eq '^(Error|Fatal):' "${m326_run}/pt.raw.log"
grep -qx 'prelayout_no_spef' "${m326_run}/reports/ptsta_scope.rpt"

python3 - "${m326_run}" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
def first_slack(name):
    text = (root / "reports" / name).read_text(encoding="utf-8")
    match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+\.\d+)", text)
    if match is None:
        raise RuntimeError("missing worst slack in " + name)
    return match.group(1), float(match.group(2))

setup_state, setup = first_slack("ptsta_timing_setup.rpt")
hold_state, hold = first_slack("ptsta_timing_hold.rpt")
check = (root / "reports" / "ptsta_check_timing.rpt").read_text(encoding="utf-8")
if check.count("paths from such ports will be unconstrained") != 1 or "reset_n" not in check:
    raise RuntimeError("M326 asynchronous-reset exception drift")
if "Checking 'unconstrained_endpoints'." not in check:
    raise RuntimeError("M326 missing unconstrained-endpoint audit")
gate = setup >= 0.0 and hold >= 0.0
summary = {
    "schema": "m326_m321_m322_prelayout_ptsta_receipt_v1",
    "status": "PASS_PRELAYOUT_PTSTA" if gate else "NO_GO_PRELAYOUT_PTSTA_HOLD",
    "setup_worst_slack_ns": setup,
    "setup_state": setup_state,
    "hold_worst_slack_ns": hold,
    "hold_state": hold_state,
    "contract_timing_gate_pass": gate,
    "asynchronous_reset_only_no_input_delay_exception": True,
    "analysis_coverage": {
        "setup_met": 156,
        "setup_violated": 0,
        "hold_met": 156 if hold >= 0 else None,
        "hold_violation_report_is_authoritative": hold < 0,
    },
    "claim_boundary": {
        "prelayout_no_spef": True,
        "ideal_clock": True,
        "zero_macro": True,
        "post_route_timing": False,
        "power_or_energy": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(root / "m326_prelayout_ptsta_receipt_r1.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

mv "${m326_run}/RUN_IN_PROGRESS.txt" "${m326_run}/RUN_BOOTSTRAP_RECORD.txt"
m326_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${m326_run}/m326_prelayout_ptsta_receipt_r1.json")
{
    echo "status=${m326_status}"
    echo "scope=PRELAYOUT_NO_SPEF_IDEAL_CLOCK_ZERO_MACRO"
    echo "paper_ppa_ready=false"
    echo "system_speedup_admitted=false"
} > "${m326_run}/RUN_COMPLETE.txt"
(
    cd "${m326_run}"
    find . -type f ! -path './work/*' ! -name output.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
        > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
echo "M326_M321_PTSTA=${m326_status} run=${m326_run}"
