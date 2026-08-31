#!/usr/bin/env bash
set -euo pipefail

m451_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m451_hw="$(cd "${m451_dc_root}/.." && pwd)"
m451_runner="$(realpath "${BASH_SOURCE[0]}")"
m451_run="${M451_VCS_RUN_DIR:-${m451_hw}/results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826}"
m451_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m451_contract="contracts/m451_exact_k1_fused_pwp_correction_directed_vcs_contract_r1_20260826.json"

m451_sha() { sha256sum "$1" | awk '{print $1}'; }
m451_expect() {
    local m451_path=$1
    local m451_expected=$2
    [[ -f "${m451_path}" ]] || exit 3
    [[ "$(m451_sha "${m451_path}")" == "${m451_expected}" ]] || exit 3
}

[[ ! -e "${m451_run}" ]] || exit 5
mkdir -p "${m451_run}"
m451_complete=0
trap 'm451_rc=$?; if [[ ${m451_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m451_rc}" >"${m451_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m451_hw}"

m451_expect "${m451_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m451_expect rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv b09172c5ca5c6fccddad0ccd19f37ffaae032cfe26350297f9ffcb3df65e2307
m451_expect verif_m451/m451_exact_k1_fused_pwp_correction_adapter_assertions.sv e18ec4b28a522085aaa7e844344762ba4b975f878f459f2356792eaf721c15be
m451_expect tb_m451/tb_m451_exact_k1_fused_pwp_correction_adapter.sv c664e9c8aad02427812df94f2474b7d23a94906618f043e97e5ba3e0e5b720fa
m451_expect dc_handoff/filelists/date_m451_exact_k1_fused_pwp_correction_directed_vcs.f 46571faebf588981c5031edb643f77a0a285cc122fda0384961b65bb4a77caf7
m451_expect "${m451_contract}" afcf10562ad37e1fc8c0bb0a0af52c98c189fdce91fc7d1b488f00ce060b4be6
m451_expect results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 a7fe306a91a1efc7b05340fdfa4bfd859e9f7aa830db01e022b046e1fb14b96a
m451_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd results/m449_m447_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m451_run}/m449_seal_check.log" 2>&1

sha256sum \
    rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv \
    verif_m451/m451_exact_k1_fused_pwp_correction_adapter_assertions.sv \
    tb_m451/tb_m451_exact_k1_fused_pwp_correction_adapter.sv \
    dc_handoff/filelists/date_m451_exact_k1_fused_pwp_correction_directed_vcs.f \
    "${m451_contract}" \
    results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m451_run}/input_sha256.txt"
cp "${m451_contract}" "${m451_run}/contract.json"
export VCS_HOME="${m451_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${m451_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m451_run}/csrc" \
    -f dc_handoff/filelists/date_m451_exact_k1_fused_pwp_correction_directed_vcs.f \
    -top tb_m451_exact_k1_fused_pwp_correction_adapter \
    -o "${m451_run}/simv" >"${m451_run}/compile.log" 2>&1
m451_rc=$?
set -e
printf '%s\n' "${m451_rc}" >"${m451_run}/compile.rc"
[[ ${m451_rc} -eq 0 && -x "${m451_run}/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "${m451_run}/compile.log"; then
    exit 21
fi

set +e
"${m451_run}/simv" +ntb_random_seed=4510120260826 -no_save \
    -cm assert -assert "report=${m451_run}/assert.report" \
    >"${m451_run}/sim.log" 2>&1
m451_rc=$?
set -e
printf '%s\n' "${m451_rc}" >"${m451_run}/sim.rc"
[[ ${m451_rc} -eq 0 ]] || exit 22
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
        "${m451_run}/sim.log" "${m451_run}/assert.report"; then
    exit 23
fi
grep -Eq 'PASS M451 exact K1 fused PWP correction adapter requests=[0-9]+ outputs=[0-9]+ plain=[0-9]+ fused=[0-9]+ fused_add=[0-9]+ fused_subtract=[0-9]+ narrow=[0-9]+ wide=[0-9]+ signed_boundary_cases=16 metadata_mismatches=0 arithmetic_mismatches=0 unknown_outputs=0 protocol_attacks=6 failclosed_leaks=0 stall_cycles=[0-9]+ max_stall=[0-9]+ pop_push=[0-9]+ consecutive_ii1=[0-9]+ legal_reloads=1 pwp_physical_bytes=160 correction_existing_bytes=96 new_memory_ports=0 old_psum_preserved_downstream=true accuracy_changed=false cycles=false system_speedup=false ppa=false power=false headline=false' \
    "${m451_run}/sim.log" || exit 24

python3 - "${m451_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
log = (root / "sim.log").read_text(errors="replace")
match = re.search(
    r"PASS M451 exact K1 fused PWP correction adapter requests=(\d+) "
    r"outputs=(\d+) plain=(\d+) fused=(\d+) fused_add=(\d+) "
    r"fused_subtract=(\d+) narrow=(\d+) wide=(\d+) "
    r"signed_boundary_cases=(\d+) metadata_mismatches=(\d+) "
    r"arithmetic_mismatches=(\d+) unknown_outputs=(\d+) "
    r"protocol_attacks=(\d+) failclosed_leaks=(\d+) stall_cycles=(\d+) "
    r"max_stall=(\d+) pop_push=(\d+) consecutive_ii1=(\d+) "
    r"legal_reloads=(\d+)", log)
if not match:
    raise SystemExit("missing M451 PASS payload")
keys = ["accepted_legal_requests", "retired_contributions", "plain_requests",
        "fused_requests", "fused_add_requests", "fused_subtract_requests",
        "narrow_requests", "wide_requests", "signed_boundary_cases",
        "metadata_mismatches", "arithmetic_mismatches", "unknown_outputs",
        "protocol_attacks", "failclosed_leaks", "stall_cycles",
        "max_stall_cycles", "simultaneous_pop_push", "consecutive_ii1_pairs",
        "legal_empty_reloads"]
directed = dict(zip(keys, [int(value) for value in match.groups()]))
if directed["accepted_legal_requests"] < 360:
    raise SystemExit("too few legal requests")
if min(directed[key] for key in ("plain_requests", "fused_add_requests",
                                  "fused_subtract_requests", "narrow_requests",
                                  "wide_requests")) == 0:
    raise SystemExit("mode coverage missing")
if directed["signed_boundary_cases"] < 16 or directed["max_stall_cycles"] < 8:
    raise SystemExit("boundary/stall coverage missing")
if directed["simultaneous_pop_push"] < 32 or directed["consecutive_ii1_pairs"] < 32:
    raise SystemExit("II1 coverage missing")
if any(directed[key] for key in ("metadata_mismatches", "arithmetic_mismatches",
                                 "unknown_outputs", "failclosed_leaks")):
    raise SystemExit("mismatch or leak")
if directed["protocol_attacks"] != 6 or directed["legal_empty_reloads"] != 1:
    raise SystemExit("protocol coverage drift")
if directed["accepted_legal_requests"] - directed["retired_contributions"] != 1:
    raise SystemExit("expected one buffered output quarantined by reload attack")

receipt = {
    "schema": "m451_exact_k1_fused_pwp_correction_directed_vcs_receipt_v1",
    "status": "PASS_M451_EXACT_K1_FUSED_PWP_CORRECTION_DIRECTED_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "directed": directed,
    "quarantined_buffered_contributions": 1,
    "arithmetic": {
        "pwp_input": "signed12x96",
        "correction_input": "signed8x96 with one shared plus/minus sign",
        "output": "exact signed13x96 update_delta",
        "downstream_operation": "new_psum=old_psum+sum(update_delta chunks)",
        "accuracy_changed": False,
    },
    "resource_boundary": {
        "pwp_physical_signal_bytes": 160,
        "correction_existing_memory_read_bytes": 96,
        "concurrent_existing_memory_reads": True,
        "new_memory_ports": 0,
        "incremental_preadder_lanes": 96,
        "power_free": False,
        "fmax_unchanged": False,
    },
    "opportunity": {
        "m430_k1_separate_cycles": 517041352,
        "m447_k1_fused_cycles": 430154216,
        "speedup_vs_m430": 517041352 / 430154216,
        "rtl_measured_speedup": False,
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
    },
    "claim_boundary": {
        "standalone": True, "exact": True, "directed_vcs": True,
        "full_population_vcs": False, "cycle_opportunity_only": True,
        "resource_normalized_speedup": False, "system_speedup": False,
        "dc": False, "formality": False, "primetime": False,
        "ppa": False, "power": False, "energy": False,
        "date_headline": False,
    },
}
(root / "m451_exact_k1_fused_pwp_correction_directed_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m451_runner}" >"${m451_run}/runner_sha256.txt"
printf '%s\n' PASS_M451_EXACT_K1_FUSED_PWP_CORRECTION_DIRECTED_VCS \
    >"${m451_run}/RUN_COMPLETE.txt"
find "${m451_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m451_run}/RUN_MANIFEST.sha256"
sha256sum "${m451_run}/RUN_MANIFEST.sha256" \
    >"${m451_run}/RUN_MANIFEST.seal.sha256"
m451_complete=1
echo "PASS_M451_EXACT_K1_FUSED_PWP_CORRECTION_DIRECTED_VCS run=${m451_run}"
