#!/usr/bin/env bash
set -euo pipefail

m442_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m442_hw="$(cd "${m442_dc_root}/.." && pwd)"
m442_runner="$(realpath "${BASH_SOURCE[0]}")"
m442_run="${M442_VCS_RUN_DIR:-${m442_hw}/results/m442b_m430_full_static_codec_m433_vcs_r1_20260826}"
m442_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m442_contract="contracts/m442_m430_full_static_codec_m433_vcs_contract_r1_20260826.json"
m442_stimulus_dir="results/m442a_m430_full_static_codec_stimulus_r1_20260826"
m442_stimulus="${m442_stimulus_dir}/m442_m430_static_codec_population.hex"

m442_sha() { sha256sum "$1" | awk '{print $1}'; }
m442_expect() {
    local m442_path=$1
    local m442_expected=$2
    [[ -f "${m442_path}" ]] || exit 3
    [[ "$(m442_sha "${m442_path}")" == "${m442_expected}" ]] || exit 3
}

[[ ! -e "${m442_run}" ]] || exit 5
mkdir -p "${m442_run}"
m442_complete=0
trap 'm442_rc=$?; if [[ ${m442_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m442_rc}" >"${m442_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m442_hw}"

m442_expect "${m442_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m442_expect rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
m442_expect verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv e5a645a0e256c7d3a72f07f027ecaf2c1d136b433c45e13248592940aba85501
m442_expect tb_m442/tb_m442_m430_full_static_codec_m433.sv 1bad0b365a890b7498f9fa3f2c7dc453fc913432cd8dc7ff8e35e0f50ae007cf
m442_expect dc_handoff/filelists/date_m442_m430_full_static_codec_m433_vcs.f 7a04d06c3b678a515aee548d0a004f22e3b11ae790ecc352ab8b39680974eae5
m442_expect system_simulator/scripts/build_m442_m430_static_codec_vcs_stimulus.py 9ac9e483a4bc1f0c00a38582e4f8a2158fc3156cb4a5ce511c3969e870d0311e
m442_expect "${m442_contract}" b4f16a8c6342123364f91b9558ba90e8383658ba018486173128d326a05e23f2
m442_expect "${m442_stimulus}" 6afd66512fc8b6fe2b4a7f759bca1299bd0cd825a51d7a5923ebadb84e4d3c1a
m442_expect "${m442_stimulus_dir}/m442_m430_static_codec_stimulus_receipt_r1.json" 94e97f05112b364003312a677ba4bdaca1e7820b4913dcfdc46d1afabb17e331
m442_expect "${m442_stimulus_dir}/SHA256SUMS.seal.sha256" 460700c10c36a846208f10df3140c0898b0c3c72c84ff24c2e5cefecc7ebdee9
m442_expect results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json 6cf413e93d8159d9516ad048eaa26c741e49c2c9a3b330fb1d6dd20ba64dab2a
m442_expect results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256 462501b849f42f1a0690d2fe8dbe3dc226e83ae05dea86f7cb0396d60e9faf7e
m442_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd "${m442_stimulus_dir}"
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m442_run}/stimulus_seal_check.log" 2>&1
(
    cd results/m430b_h67_dualaware_q32_heldout_once_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m442_run}/m430_seal_check.log" 2>&1

sha256sum \
    rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
    verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv \
    tb_m442/tb_m442_m430_full_static_codec_m433.sv \
    dc_handoff/filelists/date_m442_m430_full_static_codec_m433_vcs.f \
    system_simulator/scripts/build_m442_m430_static_codec_vcs_stimulus.py \
    "${m442_contract}" \
    "${m442_stimulus}" \
    "${m442_stimulus_dir}/m442_m430_static_codec_stimulus_receipt_r1.json" \
    "${m442_stimulus_dir}/SHA256SUMS.seal.sha256" \
    results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json \
    results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m442_run}/input_sha256.txt"
cp "${m442_contract}" "${m442_run}/contract.json"
export VCS_HOME="${m442_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${m442_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m442_run}/csrc" \
    -f dc_handoff/filelists/date_m442_m430_full_static_codec_m433_vcs.f \
    -top tb_m442_m430_full_static_codec_m433 \
    -o "${m442_run}/simv" >"${m442_run}/compile.log" 2>&1
m442_rc=$?
set -e
printf '%s\n' "${m442_rc}" >"${m442_run}/compile.rc"
[[ ${m442_rc} -eq 0 && -x "${m442_run}/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "${m442_run}/compile.log"; then
    exit 21
fi

set +e
"${m442_run}/simv" \
    +M442_STIMULUS="${m442_hw}/${m442_stimulus}" \
    +ntb_random_seed=4420120260826 -no_save -cm assert \
    -assert "report=${m442_run}/assert.report" \
    >"${m442_run}/sim.log" 2>&1
m442_rc=$?
set -e
printf '%s\n' "${m442_rc}" >"${m442_run}/sim.rc"
[[ ${m442_rc} -eq 0 ]] || exit 22
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]|unknown_outputs=[1-9]|protocol_faults=[1-9]' \
        "${m442_run}/sim.log" "${m442_run}/assert.report"; then
    exit 23
fi
grep -Eq 'PASS M442 M430 full static codec through M433 blocks=442368 lanes=42467328 narrow=70503 wide=371865 metadata_mismatches=0 arithmetic_mismatches=0 unknown_outputs=0 protocol_faults=0 pop_push=[0-9]+ stall_cycles=[0-9]+ max_queue=[0-9]+ runtime_issue_population=false cycles=false system_speedup=false power=false ppa=false headline=false' \
    "${m442_run}/sim.log" || exit 24

python3 - "${m442_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
log = (root / "sim.log").read_text(errors="replace")
match = re.search(
    r"PASS M442 M430 full static codec through M433 blocks=(\d+) "
    r"lanes=(\d+) narrow=(\d+) wide=(\d+) metadata_mismatches=(\d+) "
    r"arithmetic_mismatches=(\d+) unknown_outputs=(\d+) protocol_faults=(\d+) "
    r"pop_push=(\d+) stall_cycles=(\d+) max_queue=(\d+)", log)
if not match:
    raise SystemExit("missing M442 PASS payload")
values = [int(value) for value in match.groups()]
keys = ["blocks", "lanes", "narrow_blocks", "wide_blocks",
        "metadata_mismatches", "arithmetic_mismatches", "unknown_outputs",
        "protocol_faults", "simultaneous_pop_push", "stall_cycles",
        "maximum_scoreboard_depth"]
population = dict(zip(keys, values))
if population["blocks"] != 442368 or population["lanes"] != 42467328:
    raise SystemExit("M442 population drift")
if population["narrow_blocks"] != 70503 or population["wide_blocks"] != 371865:
    raise SystemExit("M442 width population drift")
if any(population[name] for name in ("metadata_mismatches",
                                      "arithmetic_mismatches",
                                      "unknown_outputs", "protocol_faults")):
    raise SystemExit("M442 mismatch or protocol fault")
if population["simultaneous_pop_push"] < 441368:
    raise SystemExit("M442 insufficient II=1 coverage")
if population["stall_cycles"] < 64 or population["maximum_scoreboard_depth"] < 1:
    raise SystemExit("M442 elasticity coverage drift")
receipt = {
    "schema": "m442_m430_full_static_codec_m433_vcs_receipt_v1",
    "status": "PASS_M442_M430_FULL_STATIC_CODEC_M433_SYNOPSYS_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "population": population,
    "semantics": {
        "real_frozen_h67_ep35_weight_payloads": True,
        "all_m430_static_codec_blocks": True,
        "exact_signed12x96_or_narrow_signed8_reconstruction": True,
        "runtime_repetition_population": False,
        "persistent_old_psum_is_downstream": True,
        "correction_is_downstream": True,
    },
    "verification": {"sva_failures": 0, "vcs": True},
    "claim_boundary": {
        "full_static_codec_population_vcs": True,
        "runtime_issue_population_vcs": False,
        "rtl_measured_cycles": False,
        "system_speedup": False,
        "power": False,
        "energy": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
(root / "m442_m430_full_static_codec_m433_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m442_runner}" >"${m442_run}/runner_sha256.txt"
printf '%s\n' PASS_M442_M430_FULL_STATIC_CODEC_M433_SYNOPSYS_VCS \
    >"${m442_run}/RUN_COMPLETE.txt"
find "${m442_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m442_run}/RUN_MANIFEST.sha256"
sha256sum "${m442_run}/RUN_MANIFEST.sha256" \
    >"${m442_run}/RUN_MANIFEST.seal.sha256"
m442_complete=1
echo "PASS_M442_M430_FULL_STATIC_CODEC_M433_VCS run=${m442_run}"
