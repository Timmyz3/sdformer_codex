#!/usr/bin/env bash
set -euo pipefail

m908_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m908_self="$(realpath "${BASH_SOURCE[0]}")"
m908_contract="${m908_hw_root}/contracts/m908_m892_dc_collision_safe_launch_contract_r1_20260829.json"
m908_shim="${m908_hw_root}/dc_handoff/scripts/m908_collision_safe_path/rg"
m908_inner="${m908_hw_root}/dc_handoff/scripts/run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh"
m908_old_contract="${m908_hw_root}/contracts/m892_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract_r1_20260829.json"
m908_old_release="${m908_hw_root}/contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_release_r1_20260829.json"
m908_old_final="${m908_hw_root}/reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer_r1_20260829/review.json"
m908_final="${m908_hw_root}/reviews/m909_m908_m892_collision_safe_final_launch_hammer_r1_20260829/review.json"
m908_docs359="${m908_hw_root}/docs/359_DATE终局冻结_20260813.md"
m908_result="${m908_hw_root}/dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
m908_attempt="${m908_hw_root}/dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
m908_lock="${m908_hw_root}/dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_launch_lock"

m908_sha() { sha256sum "$1" | awk '{print $1}'; }
m908_expect() {
    local m908_path=$1 m908_expected=$2
    [[ -f "${m908_path}" && ! -L "${m908_path}" && \
       "$(m908_sha "${m908_path}")" == "${m908_expected}" ]] || {
        echo "M908 identity mismatch: ${m908_path}" >&2
        exit 3
    }
}
m908_strict_json() {
    /usr/libexec/platform-python3.6 - "$1" <<'PY'
import json, sys
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: %s" % key)
        out[key] = value
    return out
def nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)
with open(sys.argv[1], "rb") as handle:
    json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique,
               parse_constant=nonfinite)
PY
}
m908_verify_dir_seal() {
    local m908_dir=$1
    [[ -d "${m908_dir}" && ! -L "${m908_dir}" && \
       -f "${m908_dir}/SHA256SUMS" && \
       -f "${m908_dir}/SHA256SUMS.seal.sha256" ]] || exit 3
    (cd "${m908_dir}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
m908_collision_present() {
    local m908_uid m908_proc m908_pid m908_owner m908_exe m908_base m908_comm
    m908_uid="$(id -u)"
    while read -r m908_pid; do
        [[ "${m908_pid}" =~ ^[0-9]+$ ]] || continue
        m908_proc="/proc/${m908_pid}"
        [[ -d "${m908_proc}" ]] || continue
        [[ "${m908_pid}" != "$$" ]] || continue
        m908_owner="$(stat -c '%u' "${m908_proc}" 2>/dev/null || true)"
        [[ "${m908_owner}" == "${m908_uid}" ]] || continue
        m908_exe="$(readlink -f "${m908_proc}/exe" 2>/dev/null || true)"
        m908_base="${m908_exe##*/}"
        m908_comm="$(tr -d '\n' <"${m908_proc}/comm" 2>/dev/null || true)"
        case "${m908_base}:${m908_comm}" in
            dc_shell:*|dc_shell-t:*|common_shell_exec:*|*:dc_shell|*:dc_shell-t|*:common_shell_exec|*:common_shell_exe)
                return 0
                ;;
        esac
    done < <(ps -u "${m908_uid}" -o pid=)
    return 1
}

[[ "$#" -eq 0 ]] || exit 2
bash -n "${m908_self}"
[[ -n "${M908_EXPECTED_WRAPPER_SHA256:-}" && \
   "$(m908_sha "${m908_self}")" == "${M908_EXPECTED_WRAPPER_SHA256}" ]] || {
    echo "M908 caller must pin wrapper SHA" >&2
    exit 3
}
[[ -n "${M908_EXPECTED_CONTRACT_SHA256:-}" ]] || exit 3
[[ -n "${M908_EXPECTED_FINAL_REVIEW_SHA256:-}" ]] || exit 3
[[ -z "${M892_NO_EDA_FULL_PATH_SELFTEST:-}${M892_NO_EDA_SELFTEST_ROOT:-}${M892_NO_EDA_PRODUCTION_SCHEMA_SELFTEST:-}${M892_NO_EDA_SOURCE_REVIEW_FIXTURE:-}${M892_EXPECTED_NO_EDA_SOURCE_REVIEW_SHA256:-}" ]] || {
    echo "M908 forbids M892 no-EDA branch overrides" >&2
    exit 3
}
[[ -z "${OUTPUT_DIR:-}${CLOCK_PERIOD_NS:-}${LIB_DB:-}${MIN_LIB_DB:-}${MACRO_DB:-}${OPERATING_CONDITION:-}" ]] || {
    echo "M908 forbids path, library, clock and corner overrides" >&2
    exit 3
}

m908_expect "${m908_contract}" "${M908_EXPECTED_CONTRACT_SHA256}"
m908_strict_json "${m908_contract}"
jq -e '.schema == "m908_m892_dc_collision_safe_launch_contract_v1"
       and .status == "AUTHORIZED_ADDITIVE_COLLISION_GUARD_FOR_EXISTING_ONE_M892_DC_ATTEMPT"
       and .authorization == {"inherits_m892_attempt":true,"max_new_attempts":0,
            "max_total_m892_attempts":1,"run_dc":true,"run_formality":false,
            "run_pt":false,"run_ptpx":false,"run_remote":false,"run_saif":false,
            "run_vcs":false}
       and .collision_guard.argv_regex_used == false
       and .collision_guard.proc_executable_or_comm_exact_match == true
       and .claim_boundary.system_speedup == false
       and .claim_boundary.paper_ppa_ready == false' "${m908_contract}" >/dev/null || exit 3

while IFS=$'\t' read -r m908_path m908_expected; do
    [[ "${m908_path}" == /* ]] || m908_path="${m908_hw_root}/${m908_path}"
    m908_expect "${m908_path}" "${m908_expected}"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${m908_contract}")
m908_expect "${m908_final}" "${M908_EXPECTED_FINAL_REVIEW_SHA256}"
m908_verify_dir_seal "$(dirname "${m908_final}")"
m908_strict_json "${m908_final}"
jq -e --arg contract_sha "$(m908_sha "${m908_contract}")" \
       --arg wrapper_sha "$(m908_sha "${m908_self}")" \
       --arg shim_sha "$(m908_sha "${m908_shim}")" \
       '.schema == "m909_m908_m892_collision_safe_final_launch_hammer_v1"
        and .status == "PASS100_M908_ADDITIVE_COLLISION_SAFE_FINAL_LAUNCH_HAMMER"
        and .verdict == "PASS" and .score_100 == 100
        and .severity_counts == {"p0":0,"p1":0,"p2":0}
        and .identity.contract_sha256 == $contract_sha
        and .identity.wrapper_sha256 == $wrapper_sha
        and .identity.shim_sha256 == $shim_sha
        and .decision.exactly_one_existing_m892_attempt_authorized == true
        and .decision.new_attempts_authorized == 0
        and .decision.dc_started_by_hammer == false' \
    "${m908_final}" >/dev/null || exit 3
[[ "$(jq -er '.old_m892_authority.inner_runner_sha256' "${m908_contract}")" == \
   "$(m908_sha "${m908_inner}")" ]] || exit 3
[[ "$(jq -er '.old_m892_authority.release_sha256' "${m908_contract}")" == \
   "$(m908_sha "${m908_old_release}")" ]] || exit 3
[[ "$(jq -er '.old_m892_authority.final_review_sha256' "${m908_contract}")" == \
   "$(m908_sha "${m908_old_final}")" ]] || exit 3

[[ ! -e "${m908_result}" && ! -e "${m908_attempt}" && ! -e "${m908_lock}" ]] || {
    echo "M908 refuses consumed/colliding M892 one-shot identity" >&2
    exit 5
}
if m908_collision_present; then
    echo "M908 refuses an exact same-UID Synopsys shell executable collision" >&2
    exit 4
fi

# The original release remains the sole attempt authority.  M908 only replaces
# the defective argv-regex observation with an exact /proc executable/comm
# observation, then enters the fully sealed original M892 runner.
exec env \
    PATH="${m908_hw_root}/dc_handoff/scripts/m908_collision_safe_path:/usr/bin:/bin" \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M892_EXPECTED_DC_RUNNER_SHA256="$(jq -er '.old_m892_authority.inner_runner_sha256' "${m908_contract}")" \
    M892_EXPECTED_DC_ADMISSION_SHA256="$(jq -er '.old_m892_authority.release_sha256' "${m908_contract}")" \
    M892_EXPECTED_DC_FINAL_REVIEW_SHA256="$(jq -er '.old_m892_authority.final_review_sha256' "${m908_contract}")" \
    "${m908_inner}"
