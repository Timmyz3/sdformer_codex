#!/usr/bin/env bash
set -euo pipefail

m426r3_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m426r3_hw="$(cd "${m426r3_scripts}/../.." && pwd)"
m426r3_runner="$(realpath "${BASH_SOURCE[0]}")"
m426r3_analyzer="${m426r3_scripts}/analyze_m426_h67_dualbank_seed_fusion.py"
m426r3_wrapper="${m426r3_scripts}/analyze_m426r3_h67_dualbank_seed_fusion_recovery.py"
m426r3_contract="${m426r3_hw}/contracts/m426_h67_dualbank_seed_fusion_contract_r1_20260826.json"
m426r3_recovery="${m426r3_hw}/contracts/m426r3_h67_dualbank_seed_fusion_recovery_contract_r1_20260826.json"
m426r3_output="${M426R3_OUTPUT_DIR:-${m426r3_hw}/results/m426r3_h67_dualbank_seed_fusion_replay_r1_20260826}"
m426r3_log="$(mktemp)"
trap 'rm -f "${m426r3_log}"' EXIT

m426r3_sha() { sha256sum "$1" | awk '{print $1}'; }
m426r3_expect() { [[ -f "$1" && "$(m426r3_sha "$1")" == "$2" ]] || exit 3; }
[[ ! -e "${m426r3_output}" ]] || exit 5
m426r3_expect "${m426r3_analyzer}" f9cb504fac95fba866286d949f9d21e8eb6b9c1ded4464fb9e27c30e8977f516
m426r3_expect "${m426r3_wrapper}" 869d763ea36f9fb9bdf95a77e90b597b782f1dd4c83d17c31b1d971fc45c68fc
m426r3_expect "${m426r3_contract}" c63802e024f552afc2451e8ef8c2da0a8a9868ebed3d15515d04150cdd71bf16
m426r3_expect "${m426r3_recovery}" 7e164c4396a5257a9e4232320926489ad0280fed9cfcb9647ae2eff597e2b3da
m426r3_expect "${m426r3_hw}/results/m426_h67_dualbank_seed_fusion_replay_r1_20260826/RUN_FAILED_OR_INCOMPLETE.log" f2cb1c83081ab93b3fee53798f19cee42fefae64a43557c6907550980120a35d
m426r3_expect "${m426r3_hw}/results/m426r2_h67_dualbank_seed_fusion_replay_r1_20260826/RUN_FAILED_OR_INCOMPLETE.log" 7cfac3c7d311d9abc20dfc9d22f04318684046111802642db1b02a776a39993d
m426r3_expect "${m426r3_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

set +e
python3 "${m426r3_wrapper}" --hw-root "${m426r3_hw}" \
    --contract "${m426r3_contract}" --output-dir "${m426r3_output}" \
    >"${m426r3_log}" 2>&1
m426r3_rc=$?
set -e
if [[ ${m426r3_rc} -ne 0 ]]; then
    if [[ -d "${m426r3_output}" ]]; then
        cp "${m426r3_log}" "${m426r3_output}/RUN_FAILED_OR_INCOMPLETE.log"
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${m426r3_rc}" >"${m426r3_output}/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
    cat "${m426r3_log}"
    exit "${m426r3_rc}"
fi
cp "${m426r3_log}" "${m426r3_output}/run.log"
if ! grep -Fq 'M426_PASS dual=' "${m426r3_output}/run.log"; then
    exit 20
fi
python3 - "${m426r3_output}/m426_h67_dualbank_seed_fusion_replay_r1.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as handle:
    data = json.load(handle)
assert data["decision"] == "GO_DUALBANK_AND_SEED_FUSION_RTL"
assert data["execution_gates"]["raw_row_words"] == 51840000
assert data["variants"]["m401_serial_low8_high4"]["cycles"] == 641790704
assert data["baselines"]["strong_zero_elided_cycles"] == 742148386
assert data["baselines"]["primary_comparison"] == "strong_zero_elided"
assert data["claim_boundary"]["full_network_or_system_speedup"] is False
PY

cp "${m426r3_contract}" "${m426r3_output}/original_contract.json"
cp "${m426r3_recovery}" "${m426r3_output}/recovery_contract.json"
sha256sum "${m426r3_analyzer}" "${m426r3_wrapper}" \
    "${m426r3_contract}" "${m426r3_recovery}" "${m426r3_runner}" \
    "${m426r3_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m426r3_output}/input_and_runner_sha256.txt"
printf '%s\n' PASS_M426R3_H67_EXACT_DUALBANK_SEED_FUSION_RECOVERY \
    >"${m426r3_output}/RUN_COMPLETE.txt"
(
    cd "${m426r3_output}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        ! -name SHA256SUMS.check.log -print0 | sort -z | \
        xargs -0 sha256sum >SHA256SUMS
    sha256sum --strict -c SHA256SUMS >SHA256SUMS.check.log 2>&1
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)
echo "PASS_M426R3_EXACT_DUALBANK_SEED_FUSION output=${m426r3_output}"
