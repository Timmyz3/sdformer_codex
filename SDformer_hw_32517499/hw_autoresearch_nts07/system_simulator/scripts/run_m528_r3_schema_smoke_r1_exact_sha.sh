#!/usr/bin/env bash
set -euo pipefail

m528_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m528_hw_root}"

m528_runner="$(realpath "${BASH_SOURCE[0]}")"
m528_analyzer="system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute_r3.py"
m528_execution="contracts/m528_h67_single_port_same_ledger_execution_contract_r3_20260827.json"
m528_python="/opt/anaconda3/envs/pytorch310/bin/python"
m528_author_dir="reviews/m528_single_port_same_ledger_recompute_author_handoff_r3_20260827"
m528_static_dir="reviews/m528_r3_recovery_static_hammer_r1_20260827"
m528_canonical="results/m528_r3_schema_smoke_r1_20260827"
m528_attempt="results/.m528_r3_schema_smoke_r1_20260827.attempt_consumed"
m528_work="results/.m528_r3_schema_smoke_r1_work.$$"
m528_quarantine="${m528_canonical}.failed_or_incomplete.$$.quarantine"
m528_forbidden_production_out="${m528_work}/FORBIDDEN_PRODUCTION_OUTPUT"

m528_sha() { sha256sum "$1" | awk '{print $1}'; }
m528_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m528_sha "${path}")" == "${expected}" ]] || {
        echo "M528 r3 smoke identity mismatch: ${path}" >&2
        exit 3
    }
}

[[ -n "${M528_R3_EXPECTED_SMOKE_RUNNER_SHA256:-}" && \
   "$(m528_sha "${m528_runner}")" == "${M528_R3_EXPECTED_SMOKE_RUNNER_SHA256}" ]] || {
    echo "M528 r3 caller must pin the independently reviewed smoke-runner SHA" >&2
    exit 3
}
[[ -n "${M528_R3_EXPECTED_SMOKE_ADMISSION_PATH:-}" && \
   -n "${M528_R3_EXPECTED_SMOKE_ADMISSION_SHA256:-}" ]] || {
    echo "M528 r3 caller must pin smoke admission path and SHA" >&2
    exit 3
}
[[ "${M528_R3_EXPECTED_SMOKE_ADMISSION_PATH}" != /* && \
   "${M528_R3_EXPECTED_SMOKE_ADMISSION_PATH}" != *".."* ]] || {
    echo "M528 r3 smoke admission path must be repository-relative" >&2
    exit 3
}
m528_admission="${M528_R3_EXPECTED_SMOKE_ADMISSION_PATH}"

m528_expect "${m528_analyzer}" a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae
m528_expect "${m528_execution}" 680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65
m528_expect "${m528_python}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m528_expect "${m528_admission}" "${M528_R3_EXPECTED_SMOKE_ADMISSION_SHA256}"
m528_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

jq -e '
  .schema == "m528_r3_schema_smoke_static_admission_v1" and
  .status == "AUTHORIZED_ONE_M528_R3_PREFLIGHT_ONLY_SCHEMA_SMOKE" and
  .authorization.schema_smoke_runs == 1 and
  .authorization.cpu_production_runs == 0 and
  .authorization.eda_runs == 0 and
  .authorization.gpu_runs == 0 and
  .authorization.rtl == false and
  .identity.smoke_runner_sha256 == env.M528_R3_EXPECTED_SMOKE_RUNNER_SHA256 and
  .identity.analyzer_sha256 == "a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae" and
  .identity.execution_contract_sha256 == "680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65" and
  .expected.area_json_pointer == "generated_view_inventory.slow.area_um2" and
  .expected.corner == "ssg0p9v125c" and
  .expected.pass_token == "PASS_M528_R3_SCHEMA_SMOKE_ONLY" and
  .docs359_sha256 == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
' "${m528_admission}" >/dev/null || exit 3

for key in author_handoff_outer_seal_file_sha256 static_review_outer_seal_file_sha256; do
    value="$(jq -er ".identity.${key}" "${m528_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
m528_expect "${m528_author_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.author_handoff_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_static_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.static_review_outer_seal_file_sha256' "${m528_admission}")"
(
    cd "${m528_author_dir}"
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
(
    cd "${m528_static_dir}"
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

[[ ! -e "${m528_canonical}" && ! -e "${m528_attempt}" && ! -e "${m528_work}" ]] || {
    echo "M528 r3 schema-smoke identity is consumed or colliding" >&2
    exit 5
}
[[ ! -e results/m528_h67_single_port_same_ledger_recompute_r3_20260827 && \
   ! -e results/.m528_h67_single_port_same_ledger_recompute_r3_20260827.attempt_consumed ]] || {
    echo "M528 r3 smoke refuses a pre-existing production identity" >&2
    exit 5
}

mkdir "${m528_work}"
m528_complete=0
m528_cleanup() {
    local rc=$?
    set +e
    if [[ "${m528_complete}" -ne 1 && -d "${m528_work}" ]]; then
        printf 'status=FAILED_SCHEMA_SMOKE_ONLY_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        (
            cd "${m528_work}"
            find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
                -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        )
        mv -T "${m528_work}" "${m528_quarantine}"
    fi
    return "${rc}"
}
trap m528_cleanup EXIT

mkdir "${m528_attempt}"
printf 'status=CONSUMED_AT_FIRST_PREFLIGHT_ONLY_SCHEMA_SMOKE\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m528_canonical}" >"${m528_attempt}/ATTEMPT_CONSUMED.txt"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" "${m528_admission}" \
    >"${m528_attempt}/identity.sha256"
(
    cd "${m528_attempt}"
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)

"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" \
    --schema-smoke-only \
    --smoke-expected-pointer generated_view_inventory.slow.area_um2 \
    --smoke-expected-corner ssg0p9v125c \
    --out "${m528_forbidden_production_out}" \
    >"${m528_work}/positive.stdout" 2>"${m528_work}/positive.stderr"
[[ "$(grep -Fxc PASS_M528_R3_SCHEMA_SMOKE_ONLY "${m528_work}/positive.stdout")" -eq 1 && \
   ! -s "${m528_work}/positive.stderr" ]] || exit 10

set +e
"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" \
    --schema-smoke-only \
    --smoke-expected-pointer generated_view_inventory.fast.area_um2 \
    --smoke-expected-corner ssg0p9v125c \
    --out "${m528_forbidden_production_out}" \
    >"${m528_work}/wrong_pointer.stdout" 2>"${m528_work}/wrong_pointer.stderr"
m528_wrong_pointer_rc=$?
"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" \
    --schema-smoke-only \
    --smoke-expected-pointer generated_view_inventory.slow.area_um2 \
    --smoke-expected-corner ffg1p05vm40c \
    --out "${m528_forbidden_production_out}" \
    >"${m528_work}/wrong_corner.stdout" 2>"${m528_work}/wrong_corner.stderr"
m528_wrong_corner_rc=$?
set -e

[[ "${m528_wrong_pointer_rc}" -ne 0 && "${m528_wrong_corner_rc}" -ne 0 ]] || exit 11
[[ "$(grep -Fc PASS_M528_R3_SCHEMA_SMOKE_ONLY "${m528_work}/wrong_pointer.stdout" || true)" -eq 0 && \
   "$(grep -Fc PASS_M528_R3_SCHEMA_SMOKE_ONLY "${m528_work}/wrong_corner.stdout" || true)" -eq 0 ]] || exit 11
grep -Fq 'schema-smoke expected area pointer mismatch' "${m528_work}/wrong_pointer.stderr" || exit 11
grep -Fq 'schema-smoke expected SRAM corner mismatch' "${m528_work}/wrong_corner.stderr" || exit 11
[[ ! -e "${m528_forbidden_production_out}" ]] || exit 12

jq -n \
    --arg schema m528_r3_schema_smoke_receipt_v1 \
    --arg status PASS_PREFLIGHT_ONLY_SCHEMA_SMOKE_PENDING_INDEPENDENT_RECEIPT_HAMMER \
    --arg runner_sha256 "$(m528_sha "${m528_runner}")" \
    --arg analyzer_sha256 "$(m528_sha "${m528_analyzer}")" \
    --arg execution_contract_sha256 "$(m528_sha "${m528_execution}")" \
    --arg admission_sha256 "$(m528_sha "${m528_admission}")" \
    --argjson wrong_pointer_rc "${m528_wrong_pointer_rc}" \
    --argjson wrong_corner_rc "${m528_wrong_corner_rc}" \
    '{
      schema: $schema,
      date: "2026-08-27",
      status: $status,
      identity: {
        smoke_runner_sha256: $runner_sha256,
        analyzer_sha256: $analyzer_sha256,
        execution_contract_sha256: $execution_contract_sha256,
        smoke_admission_sha256: $admission_sha256
      },
      cases: {
        positive: {exit_code: 0, exact_pass_token_count: 1},
        wrong_pointer: {exit_code: $wrong_pointer_rc, pass_token_count: 0},
        wrong_corner: {exit_code: $wrong_corner_rc, pass_token_count: 0}
      },
      forbidden_activity: {
        process_pool: false,
        row_replay: false,
        production_result: false,
        cpu_production_runs: 0,
        eda_runs: 0,
        gpu_runs: 0,
        rtl: false
      },
      paper_admitted: false,
      system_speedup: false,
      date_headline: false
    }' >"${m528_work}/m528_r3_schema_smoke_receipt_r1.json"
cp "${m528_admission}" "${m528_work}/schema_smoke_admission.json"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" "${m528_admission}" \
    >"${m528_work}/input_identity.sha256"
printf 'status=PASS_PREFLIGHT_ONLY_SCHEMA_SMOKE_PENDING_INDEPENDENT_RECEIPT_HAMMER\npaper_admitted=false\nsystem_speedup=false\n' \
    >"${m528_work}/RUN_COMPLETE.txt"
(
    cd "${m528_work}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
mv -T "${m528_work}" "${m528_canonical}"
m528_complete=1
trap - EXIT
echo "PASS M528 r3 preflight-only schema smoke sealed; independent receipt hammer required"
