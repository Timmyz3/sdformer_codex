#!/usr/bin/env bash
set -euo pipefail

[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" ]] || {
    echo "M699 refuses startup hooks" >&2
    exit 2
}
case "$(declare -F)" in *" "*)
    echo "M699 refuses exported shell functions" >&2
    exit 3
esac

m699_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m699_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m699_repo_root="$(cd "${m699_hw_root}/.." && pwd)"
m699_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m699_producer="${m699_repo_root}/neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m699_h67_ep35_multisequence_decoder_payload.py"
m699_contract="${m699_hw_root}/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json"
m699_review="${m699_hw_root}/reviews/m700_m699_multisequence_decoder_capture_fresh_static_hammer_r1_20260828"
m699_output="${m699_hw_root}/system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
m699_attempt="${m699_hw_root}/results/.m699_h67_ep35_multisequence_decoder_payload_r1_attempt_consumed"
m699_quarantine="${m699_output}.runner_quarantine.$$.${RANDOM}.${RANDOM}"
m699_started=0
m699_success=0

m699_die() { echo "M699: $*" >&2; exit 1; }
m699_sha() { sha256sum "$1" | awk '{print $1}'; }
m699_cleanup() {
    local rc=$?
    if [[ "${m699_started}" -eq 1 && "${m699_success}" -ne 1 ]]; then
        if [[ -e "${m699_output}" || -L "${m699_output}" ]]; then
            mv -- "${m699_output}" "${m699_quarantine}" || true
        fi
        if [[ -d "${m699_attempt}" && ! -e "${m699_attempt}/FAILED.txt" ]]; then
            echo "FAIL_CLOSED_NO_CANONICAL_RESULT rc=${rc}" \
                >"${m699_attempt}/FAILED.txt"
        fi
    fi
    exit "${rc}"
}
trap m699_cleanup EXIT

[[ "${m699_runner}" == "${m699_hw_root}/system_handoff/scripts/run_m699_h67_ep35_multisequence_decoder_payload_one_shot.sh" ]] || m699_die "canonical runner path drift"
[[ -n "${M699_EXPECTED_RUNNER_SHA256:-}" &&
   "$(m699_sha "${m699_runner}")" == "${M699_EXPECTED_RUNNER_SHA256}" ]] || m699_die "caller did not pin reviewed runner SHA"
[[ -n "${M699_EXPECTED_CONTRACT_SHA256:-}" &&
   "$(m699_sha "${m699_contract}")" == "${M699_EXPECTED_CONTRACT_SHA256}" ]] || m699_die "caller did not pin reviewed contract SHA"
[[ -n "${M699_EXPECTED_REVIEW_SHA256:-}" &&
   -n "${M699_EXPECTED_REVIEW_OUTER_SEAL_SHA256:-}" ]] || m699_die "caller did not pin fresh review roots"
[[ -x "${m699_python}" && -f "${m699_producer}" &&
   -f "${m699_contract}" ]] || m699_die "missing producer/runtime/contract"
[[ ! -e "${m699_output}" && ! -L "${m699_output}" &&
   ! -e "${m699_attempt}" && ! -L "${m699_attempt}" &&
   ! -e "${m699_quarantine}" && ! -L "${m699_quarantine}" ]] || m699_die "one-shot/output already consumed"
mkdir -p "$(dirname "${m699_output}")" "$(dirname "${m699_attempt}")"
[[ "$(df -Pk "$(dirname "${m699_output}")" | awk 'NR==2 {print $4}')" -ge 2097152 ]] || m699_die "less than 2 GiB output space"

[[ -d "${m699_review}" && ! -L "${m699_review}" &&
   -f "${m699_review}/review.json" &&
   -f "${m699_review}/SHA256SUMS" &&
   -f "${m699_review}/SHA256SUMS.seal.sha256" ]] || m699_die "fresh static review absent"
(cd "${m699_review}" && sha256sum -c SHA256SUMS >/dev/null &&
 sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
[[ "$(m699_sha "${m699_review}/review.json")" == "${M699_EXPECTED_REVIEW_SHA256}" &&
   "$(m699_sha "${m699_review}/SHA256SUMS.seal.sha256")" == "${M699_EXPECTED_REVIEW_OUTER_SEAL_SHA256}" ]] || m699_die "fresh review root drift"
"${m699_python}" - "${m699_review}/review.json" \
    "${M699_EXPECTED_RUNNER_SHA256}" "${M699_EXPECTED_CONTRACT_SHA256}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
if r.get("status") != "GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0":
    raise SystemExit("M699 fresh review status drift")
if r.get("severity") != {"p0":0,"p1":0,"p2":r.get("severity",{}).get("p2")}:
    raise SystemExit("M699 fresh review severity drift")
if not r.get("execution_authorized"):
    raise SystemExit("M699 review did not authorize execution")
if r.get("reviewed_inputs",{}).get("runner_sha256") != sys.argv[2] or \
   r.get("reviewed_inputs",{}).get("contract_sha256") != sys.argv[3]:
    raise SystemExit("M699 reviewed input identity drift")
if r.get("claim_boundary",{}).get("cycles") or \
   r.get("claim_boundary",{}).get("speedup") or \
   r.get("claim_boundary",{}).get("system_speedup"):
    raise SystemExit("M699 review exceeded capture-only boundary")
PY

# Rehash the complete review immediately before consuming the attempt.
(cd "${m699_review}" && sha256sum -c SHA256SUMS >/dev/null &&
 sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
[[ "$(m699_sha "${m699_review}/review.json")" == "${M699_EXPECTED_REVIEW_SHA256}" &&
   "$(m699_sha "${m699_review}/SHA256SUMS.seal.sha256")" == "${M699_EXPECTED_REVIEW_OUTER_SEAL_SHA256}" ]] || m699_die "pre-attempt review replacement detected"

m699_gpu_free=$(/usr/bin/nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits | awk 'NF {print $1}')
[[ "${m699_gpu_free}" =~ ^[0-9]+$ && "${m699_gpu_free}" -ge 20000 ]] || m699_die "requires 20 GiB free GPU memory"
m699_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m699_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m699_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
[[ "${m699_mem_available}" -ge 16777216 &&
   $((m699_commit_limit-m699_committed)) -ge 16777216 ]] || m699_die "host resource gate failed"

mkdir "${m699_attempt}"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_M699_ONE_SHOT"
    echo "runner_sha256=${M699_EXPECTED_RUNNER_SHA256}"
    echo "contract_sha256=${M699_EXPECTED_CONTRACT_SHA256}"
    echo "review_sha256=${M699_EXPECTED_REVIEW_SHA256}"
    echo "review_outer_seal_file_sha256=${M699_EXPECTED_REVIEW_OUTER_SEAL_SHA256}"
    echo "gpu_free_mib=${m699_gpu_free}"
    echo "claim_boundary=PAYLOAD_DENSITY_ONLY_NO_ACCURACY_CYCLES_SPEEDUP_SYSTEM_RTL_EDA_ENERGY_PPA_OR_HEADLINE"
} >"${m699_attempt}/ATTEMPT_CONSUMED.txt"
sha256sum "${m699_runner}" "${m699_producer}" "${m699_contract}" \
    "${m699_review}/SHA256SUMS.seal.sha256" \
    "${m699_hw_root}/docs/359_DATE终局冻结_20260813.md" \
    >"${m699_attempt}/identity.sha256"
(cd "${m699_attempt}" && sha256sum ATTEMPT_CONSUMED.txt identity.sha256 \
    >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m699_started=1
/usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    SDFORMER_USE_MLFLOW=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    M699_EXPECTED_RUNNER_SHA256="${M699_EXPECTED_RUNNER_SHA256}" \
    M699_EXPECTED_CONTRACT_SHA256="${M699_EXPECTED_CONTRACT_SHA256}" \
    M699_RUNNER_PATH="${m699_runner}" \
    M699_ATTEMPT_DIRECTORY="${m699_attempt}" \
    "${m699_python}" "${m699_producer}" \
    --contract "${m699_contract}" --output-dir "${m699_output}" \
    --sequences 3 --samples-per-sequence 10 --num-workers 0 \
    --chunk-elements 8388608

(cd "${m699_output}" && sha256sum -c SHA256SUMS >/dev/null &&
 sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
sha256sum -c "${m699_attempt}/identity.sha256" >/dev/null
echo "PASS_CAPTURE_AND_RUNNER_REHASH" >"${m699_attempt}/POSTCAPTURE_PASS.txt"
(cd "${m699_attempt}" && sha256sum ATTEMPT_CONSUMED.txt identity.sha256 \
    POSTCAPTURE_PASS.txt >SHA256SUMS && sha256sum SHA256SUMS \
    >SHA256SUMS.seal.sha256 && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m699_success=1
trap - EXIT
echo "PASS M699 one-shot capture; fresh result hammer still required"
