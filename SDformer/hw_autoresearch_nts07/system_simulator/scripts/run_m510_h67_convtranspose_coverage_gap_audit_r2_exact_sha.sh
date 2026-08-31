#!/usr/bin/env bash
set -euo pipefail

m510_runner_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m510_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m510_root}"

m510_python="/opt/anaconda3/envs/pytorch310/bin/python"
m510_analyzer="system_simulator/scripts/audit_m510_h67_convtranspose_coverage_gap.py"
m510_contract="contracts/m510_h67_convtranspose_coverage_gap_contract_r2_20260827.json"
m510_docs510="docs/510_H67反卷积覆盖缺口与EPD立项裁决_20260827.md"
m510_docs359="docs/359_DATE终局冻结_20260813.md"
m510_review="reviews/m510_static_hammer_r2_20260827"
m510_output="results/m510_h67_convtranspose_coverage_gap_audit_r2_20260827"
m510_attempt="results/.m510_h67_convtranspose_coverage_gap_audit_r2_attempt_consumed"

[[ "${m510_runner_abs}" == \
   "${m510_root}/system_simulator/scripts/run_m510_h67_convtranspose_coverage_gap_audit_r2_exact_sha.sh" ]] || {
    echo "M510 runner canonical path drift" >&2
    exit 3
}
[[ -n "${M510_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m510_runner_abs}" | awk '{print $1}')" == \
   "${M510_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M510 caller did not pin the reviewed runner SHA" >&2
    exit 3
}

m510_sha() { sha256sum "$1" | awk '{print $1}'; }
m510_expect() {
    local m510_path=$1 m510_expected=$2
    [[ -f "${m510_path}" ]] || {
        echo "M510 missing ${m510_path}" >&2
        exit 3
    }
    [[ "$(m510_sha "${m510_path}")" == "${m510_expected}" ]] || {
        echo "M510 SHA mismatch ${m510_path}" >&2
        exit 3
    }
}

m510_verify_identities() {
    m510_expect "${m510_analyzer}" 117384e5887d03ef497b40446a20fa673d5a7383eb947679ec14c1f0d2371e7c
    m510_expect "${m510_contract}" 4bda9fb04f8a98138886d6c79e230077106ffa381f1ecdde5fc7471c3ea7e626
    m510_expect "${m510_docs510}" 9406211afc2674aece514bd13b81312a5ae0a8fb984327b0ad6c00ae20c20bbf
    m510_expect "${m510_docs359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
    m510_expect "${m510_review}/SHA256SUMS" 4c6cd8f228a518a511b08a0b1c721167b7c662cd1ec00881760f09d324f3e798
    m510_expect "${m510_review}/SHA256SUMS.seal.sha256" baec50983660a940c5f80be2b1a57671937c317113515490eaaaf0ee582d962c
    (cd "${m510_review}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

[[ -x "${m510_python}" ]] || {
    echo "M510 missing pinned Python ${m510_python}" >&2
    exit 3
}
[[ ! -e "${m510_output}" && ! -e "${m510_attempt}" ]] || {
    echo "M510 one-shot already consumed or output exists" >&2
    exit 5
}
m510_verify_identities

m510_identity_tmp="$(mktemp)"
trap 'rm -f "${m510_identity_tmp}"' EXIT
sha256sum "${m510_runner_abs}" "${m510_analyzer}" "${m510_contract}" \
    "${m510_docs510}" "${m510_docs359}" \
    "${m510_review}/SHA256SUMS.seal.sha256" >"${m510_identity_tmp}"
sha256sum -c "${m510_identity_tmp}" >/dev/null

# Atomic mkdir is the single-owner/one-shot lock. It is deliberately retained
# as the durable attempt receipt after the audit.
mkdir "${m510_attempt}"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_ONE_SHOT_AUDIT"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "output=${m510_output}"
} >"${m510_attempt}/ATTEMPT_CONSUMED.txt"
mv "${m510_identity_tmp}" "${m510_attempt}/identity.sha256"
trap - EXIT
(cd "${m510_attempt}" && sha256sum ATTEMPT_CONSUMED.txt identity.sha256 \
    >SHA256SUMS.initial && sha256sum SHA256SUMS.initial \
    >SHA256SUMS.initial.seal.sha256 && \
    sha256sum -c SHA256SUMS.initial >/dev/null && \
    sha256sum -c SHA256SUMS.initial.seal.sha256 >/dev/null)

"${m510_python}" "${m510_analyzer}" \
    --contract "${m510_contract}" \
    --output-dir "${m510_output}"

(cd "${m510_output}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m510_verify_identities
sha256sum -c "${m510_attempt}/identity.sha256" >/dev/null
{
    echo "status=PASS_ONE_SHOT_AUDIT_AND_FINAL_REHASH"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "output_seal_file_sha256=$(m510_sha "${m510_output}/SHA256SUMS.seal.sha256")"
} >"${m510_attempt}/POSTAUDIT_PASS.txt"
(cd "${m510_attempt}" && sha256sum SHA256SUMS.initial.seal.sha256 \
    POSTAUDIT_PASS.txt >SHA256SUMS && sha256sum SHA256SUMS \
    >SHA256SUMS.seal.sha256 && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
