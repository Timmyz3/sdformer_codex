#!/usr/bin/env bash
set -euo pipefail

m426_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m426_hw="$(cd "${m426_script_dir}/../.." && pwd)"
m426_runner="$(realpath "${BASH_SOURCE[0]}")"
m426_analyzer="${m426_script_dir}/analyze_m426_h67_dualbank_seed_fusion.py"
m426_contract="${m426_hw}/contracts/m426_h67_dualbank_seed_fusion_contract_r1_20260826.json"
m426_output="${M426_OUTPUT_DIR:-${m426_hw}/results/m426_h67_dualbank_seed_fusion_replay_r1_20260826}"
m426_log="$(mktemp)"
trap 'rm -f "${m426_log}"' EXIT

m426_sha() { sha256sum "$1" | awk '{print $1}'; }
m426_expect() { [[ -f "$1" && "$(m426_sha "$1")" == "$2" ]] || exit 3; }
[[ ! -e "${m426_output}" ]] || exit 5

m426_expect "${m426_analyzer}" f9cb504fac95fba866286d949f9d21e8eb6b9c1ded4464fb9e27c30e8977f516
m426_expect "${m426_contract}" c63802e024f552afc2451e8ef8c2da0a8a9868ebed3d15515d04150cdd71bf16
m426_expect "${m426_hw}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

m426_repo="$(dirname "${m426_hw}")"
m426_m410="${m426_hw}/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826"
m426_m401="${m426_hw}/results/m401_h67_q32_elastic_pwp_full_replay_r1_20260826"
m426_m418="${m426_hw}/results/m418_h67_three_mode_exact_cycle_replay_r1_20260826"
m426_m419="${m426_hw}/results/m419_m418_three_mode_independent_hammer_r1_20260826"
(cd "${m426_repo}" && sha256sum --strict -c \
    "${m426_m410}/SHA256SUMS.seal.sha256" >/dev/null && \
    sha256sum --strict -c "${m426_m410}/SHA256SUMS" >/dev/null)
(cd "${m426_repo}" && sha256sum --strict -c \
    "${m426_m401}/SHA256SUMS.seal.sha256" >/dev/null && \
    sha256sum --strict -c "${m426_m401}/SHA256SUMS" >/dev/null)
(cd "${m426_m418}" && sha256sum --strict -c SHA256SUMS.seal.sha256 >/dev/null && \
    sha256sum --strict -c SHA256SUMS >/dev/null)
(cd "${m426_m419}" && sha256sum --strict -c SHA256SUMS.seal.sha256 >/dev/null && \
    sha256sum --strict -c SHA256SUMS >/dev/null)

set +e
python3 "${m426_analyzer}" --hw-root "${m426_hw}" \
    --contract "${m426_contract}" --output-dir "${m426_output}" \
    >"${m426_log}" 2>&1
m426_rc=$?
set -e
if [[ ${m426_rc} -ne 0 ]]; then
    if [[ -d "${m426_output}" ]]; then
        cp "${m426_log}" "${m426_output}/RUN_FAILED_OR_INCOMPLETE.log"
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${m426_rc}" >"${m426_output}/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
    cat "${m426_log}"
    exit "${m426_rc}"
fi
cp "${m426_log}" "${m426_output}/run.log"
if ! grep -Fq 'M426_PASS dual=' "${m426_output}/run.log"; then
    exit 20
fi
python3 - "${m426_output}/m426_h67_dualbank_seed_fusion_replay_r1.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as handle:
    data = json.load(handle)
assert data["status"] == "PASS_M426_H67_EXACT_DUALBANK_SEED_FUSION_REPLAY"
assert data["decision"] == "GO_DUALBANK_AND_SEED_FUSION_RTL"
assert data["execution_gates"]["raw_row_words"] == 51840000
assert data["execution_gates"]["m401_cycle_reproduction_mismatch"] == 0
assert data["execution_gates"]["strong_zero_cycle_reproduction_mismatch"] == 0
assert data["baselines"]["primary_comparison"] == "strong_zero_elided"
assert data["claim_boundary"]["full_network_or_system_speedup"] is False
PY

cp "${m426_contract}" "${m426_output}/contract.json"
sha256sum "${m426_analyzer}" "${m426_contract}" "${m426_runner}" \
    "${m426_hw}/docs/359_DATE终局冻结_20260813.md" \
    >"${m426_output}/input_and_runner_sha256.txt"
printf '%s\n' PASS_M426_H67_EXACT_DUALBANK_SEED_FUSION_REPLAY \
    >"${m426_output}/RUN_COMPLETE.txt"
(
    cd "${m426_output}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        ! -name SHA256SUMS.check.log -print0 | sort -z | \
        xargs -0 sha256sum >SHA256SUMS
    sha256sum --strict -c SHA256SUMS >SHA256SUMS.check.log 2>&1
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)
echo "PASS_M426_EXACT_DUALBANK_SEED_FUSION output=${m426_output}"
