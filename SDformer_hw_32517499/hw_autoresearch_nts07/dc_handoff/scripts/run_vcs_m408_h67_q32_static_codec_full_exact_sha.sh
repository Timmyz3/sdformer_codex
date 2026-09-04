#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M408_VCS_RUN_DIR:-${task_hw_root}/results/m408_h67_q32_static_codec_full_vcs_r1_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_stimulus="${task_hw_root}/results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/m408_h67_q32_static_codec_1281.memh"
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}/csrc"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv"]="819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744"
    ["verif_m405/m405_exact_elastic_pwp_issue_adapter_assertions.sv"]="372aaac64c02cc91ed97029338830cd5712202f7db9c5b550215428f633d1f4e"
    ["tb_m405/tb_m408_h67_q32_static_codec_full_vcs.sv"]="46ec93cb336f78bb94214e727796e76783d0399a052fbe993890c733bbb603f0"
    ["dc_handoff/filelists/date_m408_h67_q32_static_codec_full_vcs.f"]="77e68b6d526aa3f27df6393230c485a76d1b43d606539b10b70f7faecef3cc58"
    ["contracts/m408_h67_q32_static_codec_full_vcs_contract_r1_20260826.json"]="5bb751dd7f8664c0408447740cbbe77b66357bf64778e5e95d532597a42c2ff1"
    ["results/m407_m405r3_integration_independent_hammer_r1_20260826/m407_m405r3_integration_independent_hammer_review_r1.json"]="af279c4d7cc07d8517cbf72fb12ccf4600b66609493af0cda35cb1251b2285e6"
    ["results/m407_m405r3_integration_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="d2ecf11f0b6fd0a710e961350329b4c728a033816259a32777ef0bb1b0f40fbf"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/m408_h67_q32_static_codec_vcs_stimulus_r1.json"]="fbf1454675f6c41162503fe258927fdc6fd5ee36a19c163ed0133068517f4111"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/m408_h67_q32_static_codec_1281.memh"]="a7c0f76187ed57cfedb94bae1ab8bb75513f9959df8fae1fc38eeb95818dd81c"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS"]="a054409aa63b040b4e620cc2f4a08d07eb2cef0d9d00a09b5822329f9f85bda5"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS.seal.sha256"]="18a610bd03aa6fee665b4557ff6957f4b864d35be462bea881c1e2d4406cc497"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "${task_path}" \
        "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
(
    cd results/m407_m405r3_integration_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_run}/m407_seal_check.log" 2>&1
(
    cd results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_run}/stimulus_seal_check.log" 2>&1
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m408_h67_q32_static_codec_full_vcs_contract_r1_20260826.json \
    "${task_run}/contract.json"
export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m408_h67_q32_static_codec_full_vcs.f \
    -top tb_m408_h67_q32_static_codec_full_vcs \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "${task_run}/compile.log" && exit 21 || true

task_start="$(date +%s)"
set +e
"${task_run}/simv" +ntb_random_seed=40820260826 \
    "+M408_STIMULUS=${task_stimulus}" -no_save -cm assert \
    -assert "report=${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
task_end="$(date +%s)"
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
printf '%s\n' "$((task_end-task_start))" >"${task_run}/sim_wall_seconds.txt"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true
grep -Eq 'PASS M408 full static codec blocks=442368 lanes=42467328 narrow=112167 wide=330201 contributions=772569 metadata_mismatches=0 arithmetic_mismatches=0 semantic_narrow_mismatches=0 padding_mismatches=0 exact_low8_high4=true single_shared96=true system_speedup=false headline=false' \
    "${task_run}/sim.log" || exit 24

python3 - "${task_run}" \
    "${task_run}/m408_h67_q32_static_codec_full_vcs_receipt_r1.json" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "sim.log").read_text(errors="replace")
match = re.search(
    r"PASS M408 full static codec blocks=(\d+) lanes=(\d+) narrow=(\d+) "
    r"wide=(\d+) contributions=(\d+) metadata_mismatches=(\d+) "
    r"arithmetic_mismatches=(\d+) semantic_narrow_mismatches=(\d+) "
    r"padding_mismatches=(\d+)", text)
if not match:
    raise SystemExit("missing M408 PASS ledger")
values = [int(value) for value in match.groups()]
expected = [442368, 42467328, 112167, 330201, 772569, 0, 0, 0, 0]
if values != expected:
    raise SystemExit("M408 PASS ledger drift")
receipt = {
    "schema": "m408_h67_q32_static_codec_full_vcs_receipt_v1",
    "status": "PASS_M408_H67_Q32_STATIC_CODEC_FULL_SYNOPSYS_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "population": {
        "blocks": values[0],
        "source_lanes": values[1],
        "narrow_blocks": values[2],
        "wide_blocks": values[3],
        "accepted_contributions": values[4],
        "checked_contribution_lanes": values[4] * 96,
    },
    "mismatches": {
        "metadata": values[5],
        "arithmetic": values[6],
        "semantic_narrow": values[7],
        "padding": values[8],
        "protocol": 0,
        "assertion": 0,
    },
    "sim_wall_seconds": int((root / "sim_wall_seconds.txt").read_text()),
    "claim_boundary": {
        "full_static_codec_vcs": True,
        "full_real_q32_miter_vcs": False,
        "rtl_realtrace_cycle_match": False,
        "rtl_measured_speedup": False,
        "system_speedup": False,
        "energy": False,
        "headline": False,
    },
}
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2,
                                       sort_keys=True) + "\n")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' PASS_M408_H67_Q32_STATIC_CODEC_FULL_SYNOPSYS_VCS \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M408 full static VCS sealed at ${task_run}"
