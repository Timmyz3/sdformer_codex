#!/usr/bin/env bash
set -euo pipefail

m1311_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m1311_hw_root="$(cd "${m1311_dc_root}/.." && pwd)"
m1311_wrapper="$(realpath "${BASH_SOURCE[0]}")"
m1311_helper="${m1311_hw_root}/dc_handoff/scripts/check_m1311_python_symlink_entity.sh"
m1311_orchestrator="${m1311_hw_root}/dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.py"
m1311_admission="${m1311_hw_root}/contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_launch_admission_r1_20260831.json"
m1311_helper_sha=25e4bf69d69f9ac6069ce160d252539cbd4c15232284c0a6256fa5d19dcce223
m1311_python_sha=9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f

m1311_sha() { sha256sum "$1" | awk '{print $1}'; }
m1311_expect_regular() {
    [[ -f "$1" && ! -L "$1" && "$(m1311_sha "$1")" == "$2" ]]
}
m1311_sealed_payload_ok() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}" && ! -L "${payload}" && \
       -f "${payload}.sha256" && ! -L "${payload}.sha256" && \
       -f "${payload}.sha256.seal.sha256" && ! -L "${payload}.sha256.seal.sha256" ]] || return 1
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}

[[ -n "${M1311_EXPECTED_WRAPPER_SHA256:-}" && \
   -n "${M1311_EXPECTED_ORCHESTRATOR_SHA256:-}" && \
   -n "${M1311_EXPECTED_ADMISSION_SHA256:-}" ]] || exit 3
m1311_expect_regular "${m1311_wrapper}" "${M1311_EXPECTED_WRAPPER_SHA256}" || exit 3
m1311_expect_regular "${m1311_helper}" "${m1311_helper_sha}" || exit 3
m1311_expect_regular "${m1311_orchestrator}" "${M1311_EXPECTED_ORCHESTRATOR_SHA256}" || exit 3
m1311_expect_regular "${m1311_admission}" "${M1311_EXPECTED_ADMISSION_SHA256}" || exit 3
m1311_sealed_payload_ok "${m1311_admission}" || exit 3
[[ "${PATH:-}" == /usr/bin:/bin && "${LANG:-}" == C.UTF-8 && \
   "${LC_ALL:-}" == C.UTF-8 && -z "${HOME:-}" ]] || exit 3
[[ "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo && \
   "${LM_LICENSE_FILE:-}" == /opt/synopsys/Synopsys.dat ]] || exit 3

/usr/bin/bash "${m1311_helper}" \
    /usr/bin/python3 /etc/alternatives/python3 \
    /etc/alternatives/python3 /usr/bin/python3.6 \
    /usr/bin/python3.6 /usr/libexec/platform-python3.6 \
    /usr/libexec/platform-python3.6 66313 6442661434 755 11872 "${m1311_python_sha}" \
    >/dev/null

# Bind the already-validated regular entity to an inherited descriptor, then
# re-check stat and SHA through that descriptor before using it as interpreter.
exec 9</usr/libexec/platform-python3.6
IFS=: read -r m1311_fd_dev m1311_fd_ino m1311_fd_mode m1311_fd_size \
    < <(stat -Lc '%d:%i:%a:%s' "/proc/${BASHPID}/fd/9")
[[ "${m1311_fd_dev}:${m1311_fd_ino}:${m1311_fd_mode}:${m1311_fd_size}" == \
   "66313:6442661434:755:11872" ]] || exit 3
[[ "$(m1311_sha "/proc/${BASHPID}/fd/9")" == "${m1311_python_sha}" ]] || exit 3

exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
    LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat \
    M1311_EXPECTED_WRAPPER_SHA256="${M1311_EXPECTED_WRAPPER_SHA256}" \
    M1311_EXPECTED_ORCHESTRATOR_SHA256="${M1311_EXPECTED_ORCHESTRATOR_SHA256}" \
    M1311_EXPECTED_ADMISSION_SHA256="${M1311_EXPECTED_ADMISSION_SHA256}" \
    "/proc/${BASHPID}/fd/9" "${m1311_orchestrator}" \
    --admission "${m1311_admission}" \
    --expected-admission-sha "${M1311_EXPECTED_ADMISSION_SHA256}" \
    --expected-wrapper-sha "${M1311_EXPECTED_WRAPPER_SHA256}" \
    --expected-orchestrator-sha "${M1311_EXPECTED_ORCHESTRATOR_SHA256}"
