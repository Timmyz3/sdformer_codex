#!/usr/bin/env bash
set -euo pipefail

m424_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m424_hw="$(cd "${m424_dc_root}/.." && pwd)"
m424_result="${m424_hw}/results/m424_m422_ptsta_independent_hammer_r1_20260826"
m424_independent="${m424_result}/independent_ptsta"
m424_m416="${m424_dc_root}/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826"
m424_m420="${m424_dc_root}/runs/m420_m414_dual_formality_r1_20260826"
m424_m421="${m424_hw}/results/m421_m420_dual_formality_independent_hammer_r1_20260826"
m424_m422="${m424_dc_root}/runs/m422_m416_selected_slice_prelayout_ptsta_r1_20260826"
m424_netlist="${m424_m416}/netlist/m405_q32_elastic_selected_slice_mapped.v"
m424_sdc="${m424_m416}/netlist/m405_q32_elastic_selected_slice_mapped.sdc"
m424_tcl="${m424_dc_root}/scripts/run_ptsta_m424_independent_reproduction.tcl"
m424_contract="${m424_hw}/contracts/m424_m422_ptsta_independent_hammer_contract_r1_20260826.json"
m424_pt="/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"
m424_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m424_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

m424_sha() { sha256sum "$1" | awk '{print $1}'; }
m424_expect() { [[ -f "$1" && "$(m424_sha "$1")" == "$2" ]] || exit 3; }

[[ ! -e "${m424_result}" ]] || exit 5
if pgrep -x pt_shell >/dev/null 2>&1; then exit 4; fi

m424_expect "${m424_contract}" 1e9e483aa25feaa278e4863f3132cbe319dc18838d2fe66a3ecc7e4141ad8cdd
m424_expect "${m424_tcl}" 25d9535f00e8b5684e329f85208e14e97ba48f5f22c0757aff70d86981bff726
m424_expect "${m424_pt}" afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef
m424_expect "${m424_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m424_expect "${m424_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m424_expect "${m424_netlist}" 4b07e83eba88508da3fc1aa27187b3fa8ca03a633b165ea68641bcd26b969fe2
m424_expect "${m424_sdc}" b4eb7d225256474c8629cb924db18260efbd691db8a76d3a622d4ae8d0479ed9
m424_expect "${m424_m416}/evidence_manifest.seal.sha256" 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m424_expect "${m424_m420}/output.seal.sha256" cf216915f3f0c8e1ee4e894734e81337d81baf70b36cbd9b51ee23b381e723d7
m424_expect "${m424_m421}/m421_m420_dual_formality_independent_hammer_review_r1.json" 1a449050ebe5967431798ff13638fd27fccca1ae2ec37a636375b34f7c2070a0
m424_expect "${m424_m421}/SHA256SUMS.seal.sha256" 53d71e23ae3f901e98196fb008131847bdd86b0079608d32c5165347a9450554
m424_expect "${m424_m422}/contract.json" c6b5e812039cec1f543461e7f08e19da7bb2513db6ed10a7bdedcda9a184616b
m424_expect "${m424_dc_root}/scripts/run_ptsta_m422_m416_selected_slice_exact_sha.tcl" e8a83147770495b466c8517681880ca4a215f325f619ebcafae8c42d18a128b6
m424_expect "${m424_dc_root}/scripts/run_ptsta_m422_m416_selected_slice_exact_sha.sh" 363aad07abe4a4aef96e93c34c266df44fb7d7ab408d525bfbe93eb3560ee037
m424_expect "${m424_m422}/m422_m416_selected_slice_prelayout_ptsta_receipt_r1.json" 75c643b8878f326b1554cdbca30788e70c4380876b6e0a0c23fdae372157f665
m424_expect "${m424_m422}/output.sha256" 922354bf4d5dc33aad57e00d213ec31992c906052497eed529bc498df9df681d
m424_expect "${m424_m422}/output.seal.sha256" 549cc0f58aadc333775b197a57cac4f7f6fd30133d6ee24b7cd4c9dc48a95a29
m424_expect "${m424_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(cd "${m424_m416}" && sha256sum --strict -c evidence_manifest.seal.sha256 >/dev/null && sha256sum --strict -c evidence_manifest.sha256 >/dev/null)
(cd "${m424_m420}" && sha256sum --strict -c output.seal.sha256 >/dev/null && sha256sum --strict -c output.sha256 >/dev/null)
(cd "${m424_m421}" && sha256sum --strict -c SHA256SUMS.seal.sha256 >/dev/null && sha256sum --strict -c SHA256SUMS >/dev/null)
(cd "${m424_m422}" && sha256sum --strict -c output.seal.sha256 >/dev/null && sha256sum --strict -c output.sha256 >/dev/null)

mkdir -p "${m424_independent}/reports" "${m424_independent}/work"
cp "${m424_contract}" "${m424_independent}/contract.json"
sha256sum "${m424_contract}" "${m424_tcl}" "${m424_pt}" \
    "${m424_slow}" "${m424_fast}" "${m424_netlist}" "${m424_sdc}" \
    "${m424_m416}/evidence_manifest.seal.sha256" \
    "${m424_m420}/output.seal.sha256" \
    "${m424_m421}/SHA256SUMS.seal.sha256" \
    "${m424_m422}/output.seal.sha256" \
    "${m424_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m424_independent}/input_sha256.txt"

export M424_DESIGN_NAME=m405_q32_elastic_selected_slice
export M424_LIB_SLOW="${m424_slow}"
export M424_LIB_FAST="${m424_fast}"
export M424_MAPPED_NETLIST="${m424_netlist}"
export M424_MAPPED_SDC="${m424_sdc}"
export M424_OUTPUT_DIR="${m424_independent}"
"${m424_pt}" -version >"${m424_independent}/pt.version.raw.log" 2>&1
set +e
(cd "${m424_independent}/work" && "${m424_pt}" -f "${m424_tcl}") \
    >"${m424_independent}/pt.raw.log" 2>&1
m424_rc=$?
set -e
echo "${m424_rc}" >"${m424_independent}/pt.rc"
[[ "${m424_rc}" -eq 0 ]]
if grep -Eq '^(Error|Fatal):' "${m424_independent}/pt.raw.log"; then
    exit 31
fi
grep -Fqx "Design 'm405_q32_elastic_selected_slice' was successfully linked." \
    "${m424_independent}/pt.raw.log"
grep -Fqx 'M424_M422_PTSTA_INDEPENDENT_INTERNAL_COMPLETE=PASS' \
    "${m424_independent}/PTSTA_INDEPENDENT_INTERNAL_COMPLETE.txt"

for m424_report in check_timing_independent.rpt analysis_coverage_independent.rpt \
        global_timing_independent.rpt worst_setup_independent.rpt \
        worst_hold_independent.rpt constraint_violators_independent.rpt \
        clock_independent.rpt exceptions_independent.rpt \
        design_independent.rpt libraries_independent.rpt \
        runtime_scope_independent.rpt; do
    [[ -s "${m424_independent}/reports/${m424_report}" ]] || exit 30
done

{
    echo 'status=PASS_M424_INDEPENDENT_PT_EXECUTION_ONLY'
    echo 'interpretation=pending_independent_hammer_review'
} >"${m424_independent}/RUN_COMPLETE.txt"
sha256sum "$(realpath "${BASH_SOURCE[0]}")" >"${m424_independent}/runner_sha256.txt"
echo "PASS_M424_INDEPENDENT_PT_EXECUTION run=${m424_independent}"
