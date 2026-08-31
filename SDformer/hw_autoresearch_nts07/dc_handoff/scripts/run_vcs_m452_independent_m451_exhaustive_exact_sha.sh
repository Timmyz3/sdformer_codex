#!/usr/bin/env bash
set -euo pipefail

m452_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m452_hw="$(cd "${m452_dc_root}/.." && pwd)"
m452_runner="$(realpath "${BASH_SOURCE[0]}")"
m452_run="${M452_VCS_RUN_DIR:-${m452_hw}/results/m452_m451_independent_hammer_r1_20260826}"
m452_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m452_contract="contracts/m452_m451_independent_hammer_contract_r1_20260826.json"

m452_sha() { sha256sum "$1" | awk '{print $1}'; }
m452_expect() {
    local m452_path=$1
    local m452_expected=$2
    [[ -f "${m452_path}" ]] || exit 3
    [[ "$(m452_sha "${m452_path}")" == "${m452_expected}" ]] || exit 3
}

[[ ! -e "${m452_run}" ]] || exit 5
mkdir -p "${m452_run}"
m452_complete=0
trap 'm452_rc=$?; if [[ ${m452_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m452_rc}" >"${m452_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m452_hw}"

m452_expect "${m452_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m452_expect rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv b09172c5ca5c6fccddad0ccd19f37ffaae032cfe26350297f9ffcb3df65e2307
m452_expect verif_m452/m452_independent_m451_assertions.sv b0d0cf8e8c448f08543142efcd92d020353414d021cf12457c4e0a1e874af341
m452_expect tb_m452/tb_m452_independent_m451_exhaustive.sv 5b1dd7c1541eda7da6810d8128c526ef30b78b8784c337f606efcbf805fdf6b9
m452_expect dc_handoff/filelists/date_m452_independent_m451_exhaustive_vcs.f b9ac9aa320998a86dc7189fa6cecc477341012c9eac5d46ebfc8fd85013fee04
m452_expect "${m452_contract}" f9e2deacd20f16180d34bf97d7885bb5d6416a4d55fcbc3cfe951c95baa9dffa
m452_expect contracts/m451_exact_k1_fused_pwp_correction_directed_vcs_contract_r1_20260826.json afcf10562ad37e1fc8c0bb0a0af52c98c189fdce91fc7d1b488f00ce060b4be6
m452_expect results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 9b6fad46290411d90e9d28e40202981b64d8ccb178f607f23370ce213c6fd3e3
m452_expect results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/m451_exact_k1_fused_pwp_correction_directed_vcs_receipt_r1.json 4d394eb04dec8e145dd3234eec58216fdeb467efe644602db77262190070f3b0
m452_expect results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 a7fe306a91a1efc7b05340fdfa4bfd859e9f7aa830db01e022b046e1fb14b96a
m452_expect results/m449_m447_independent_hammer_r1_20260826/m449_independent_recomputation.json 161c62f851b4afec20f53cd4a6a267104b30a186798f3a1a6cb2ba8f19964524
m452_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826
    sha256sum -c RUN_MANIFEST.sha256
    sha256sum -c RUN_MANIFEST.seal.sha256
) >"${m452_run}/m451_seal_check.log" 2>&1
(
    cd results/m449_m447_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m452_run}/m449_seal_check.log" 2>&1

sha256sum \
    rtl_m451/m451_exact_k1_fused_pwp_correction_adapter.sv \
    verif_m452/m452_independent_m451_assertions.sv \
    tb_m452/tb_m452_independent_m451_exhaustive.sv \
    dc_handoff/filelists/date_m452_independent_m451_exhaustive_vcs.f \
    "${m452_contract}" \
    results/m451_exact_k1_fused_pwp_correction_directed_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m452_run}/input_sha256.txt"
cp "${m452_contract}" "${m452_run}/contract.json"
export VCS_HOME="${m452_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${m452_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m452_run}/csrc" \
    -f dc_handoff/filelists/date_m452_independent_m451_exhaustive_vcs.f \
    -top tb_m452_independent_m451_exhaustive \
    -o "${m452_run}/simv" >"${m452_run}/compile.log" 2>&1
m452_rc=$?
set -e
printf '%s\n' "${m452_rc}" >"${m452_run}/compile.rc"
[[ ${m452_rc} -eq 0 && -x "${m452_run}/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "${m452_run}/compile.log"; then
    exit 21
fi

set +e
"${m452_run}/simv" +ntb_random_seed=4520120260826 -no_save \
    -cm assert -assert "report=${m452_run}/assert.report" \
    >"${m452_run}/sim.log" 2>&1
m452_rc=$?
set -e
printf '%s\n' "${m452_rc}" >"${m452_run}/sim.rc"
[[ ${m452_rc} -eq 0 ]] || exit 22
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
        "${m452_run}/sim.log" "${m452_run}/assert.report"; then
    exit 23
fi
grep -Eq 'PASS M452 independent M451 exhaustive legal_accepts=[0-9]+ retired=[0-9]+ wide_signed_pairs=2097152 narrow_signed_pairs=131072 plain_lanes=[0-9]+ arithmetic_mismatches=0 metadata_mismatches=0 unknown_outputs=0 protocol_attacks=13 failclosed_leaks=0 max_stall=[0-9]+ pop_push=[0-9]+ ii1_pairs=[0-9]+ legal_reloads=1 quarantined=1 signed13_min=-2176 signed13_max=2175 old_psum_external=true memories_absent=true cycles=false system=false power=false' \
    "${m452_run}/sim.log" || exit 24
for m452_cover in cp_plain cp_fused_add cp_fused_sub cp_narrow cp_wide \
        cp_pop_push cp_ii1 cp_stall12 cp_fault; do
    grep -Eq "${m452_cover}.*, [0-9]+ attempts, [1-9][0-9]* match" \
        "${m452_run}/assert.report" || exit 25
done

python3 - "${m452_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
log = (root / "sim.log").read_text(errors="replace")
pattern = (
    r"PASS M452 independent M451 exhaustive legal_accepts=(\d+) "
    r"retired=(\d+) wide_signed_pairs=(\d+) narrow_signed_pairs=(\d+) "
    r"plain_lanes=(\d+) arithmetic_mismatches=(\d+) "
    r"metadata_mismatches=(\d+) unknown_outputs=(\d+) "
    r"protocol_attacks=(\d+) failclosed_leaks=(\d+) max_stall=(\d+) "
    r"pop_push=(\d+) ii1_pairs=(\d+) legal_reloads=(\d+) "
    r"quarantined=(\d+) signed13_min=(-?\d+) signed13_max=(-?\d+)")
match = re.search(pattern, log)
if not match:
    raise SystemExit("M452 PASS payload missing")
keys = ["legal_accepts", "retired", "wide_signed_pairs",
        "narrow_signed_pairs", "plain_lanes", "arithmetic_mismatches",
        "metadata_mismatches", "unknown_outputs", "protocol_attacks",
        "failclosed_leaks", "max_stall", "pop_push", "ii1_pairs",
        "legal_reloads", "quarantined", "signed13_min", "signed13_max"]
directed = dict(zip(keys, (int(value) for value in match.groups())))
if directed["wide_signed_pairs"] != 2 * 4096 * 256:
    raise SystemExit("wide arithmetic extent mismatch")
if directed["narrow_signed_pairs"] != 2 * 256 * 256:
    raise SystemExit("narrow arithmetic extent mismatch")
if any(directed[key] for key in ("arithmetic_mismatches",
                                  "metadata_mismatches",
                                  "unknown_outputs", "failclosed_leaks")):
    raise SystemExit("independent mismatch ledger nonzero")
if (directed["protocol_attacks"] != 13 or
        directed["legal_reloads"] != 1 or
        directed["quarantined"] != 1 or directed["max_stall"] < 12 or
        directed["signed13_min"] != -2176 or
        directed["signed13_max"] != 2175):
    raise SystemExit("independent boundary/protocol coverage drift")
receipt = {
    "schema": "m452_m451_independent_exhaustive_vcs_receipt_v1",
    "status": "PASS_M452_INDEPENDENT_EXHAUSTIVE_ARITHMETIC_PROTOCOL_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "directed": directed,
    "arithmetic": {
        "wide_domain": "all signed12 x signed8 pairs, add and subtract",
        "narrow_domain": "all signed8 x signed8 pairs, add and subtract",
        "full_interface_output_range": [-2176, 2175],
        "signed13_sufficient": True,
        "saturation_rounding_or_drop": False,
    },
    "semantic_boundary": {
        "adapter_output": "update_delta only",
        "downstream_old_psum_present_in_rtl": False,
        "required_integration": "new_psum=old_psum+sum(update_delta chunks)",
        "m426_overwrite_restored": False,
    },
    "resource_boundary": {
        "memory_macros_in_dut": 0,
        "address_generators_in_dut": 0,
        "concurrent_existing_memory_reads_proven_by_this_run": False,
        "instantaneous_payload_bytes": 256,
        "incremental_signed_preadder_lanes": 96,
        "area_timing_power_measured": False,
    },
    "opportunity": {
        "m430_cycles": 517041352,
        "m447_k1_fused_cycles": 430154216,
        "ratio": 517041352 / 430154216,
        "rtl_measured_speedup": False,
        "system_speedup": False,
    },
}
(root / "m452_independent_exhaustive_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m452_runner}" >"${m452_run}/runner_sha256.txt"
printf '%s\n' PASS_M452_INDEPENDENT_EXHAUSTIVE_ARITHMETIC_PROTOCOL_VCS \
    >"${m452_run}/RUN_COMPLETE.txt"
find "${m452_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m452_run}/RUN_MANIFEST.sha256"
sha256sum "${m452_run}/RUN_MANIFEST.sha256" \
    >"${m452_run}/RUN_MANIFEST.seal.sha256"
m452_complete=1
echo "PASS_M452_INDEPENDENT_EXHAUSTIVE_ARITHMETIC_PROTOCOL_VCS run=${m452_run}"
