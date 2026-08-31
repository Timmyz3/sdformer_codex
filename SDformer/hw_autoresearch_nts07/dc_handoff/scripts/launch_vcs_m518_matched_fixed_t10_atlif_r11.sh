#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_wrapper="$(realpath "${BASH_SOURCE[0]}")"
task_runner="${task_dc_root}/scripts/run_vcs_m518_matched_fixed_t10_atlif_r11_exact.sh"
task_admission="${task_hw_root}/contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r11_20260827.json"

if [[ -v M518_RUN_DIR ]]; then
    printf 'env_name=M518_RUN_DIR present=true action=reject_no_value_printed\n' >&2
    exit 40
fi
[[ -f "${task_runner}" ]] || exit 41
[[ -f "${task_admission}" ]] || exit 42
[[ -f "${task_admission}.sha256" ]] || exit 43
[[ -f "${task_admission}.sha256.seal.sha256" ]] || exit 43
(cd "$(dirname "${task_admission}")" && \
    sha256sum -c "$(basename "${task_admission}").sha256" >/dev/null && \
    sha256sum -c "$(basename "${task_admission}").sha256.seal.sha256" >/dev/null) \
    || exit 44

task_runner_sha="$(sha256sum "${task_runner}" | awk '{print $1}')"
task_wrapper_sha="$(sha256sum "${task_wrapper}" | awk '{print $1}')"
task_admission_sha="$(sha256sum "${task_admission}" | awk '{print $1}')"
[[ "${task_runner_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 45
[[ "${task_wrapper_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 46
[[ "${task_admission_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 47

python3 - "${task_admission}" "${task_runner_sha}" \
    "${task_wrapper_sha}" <<'PY'
import json
import math
import sys

def reject(value):
    raise ValueError("non-finite JSON constant: " + value)

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    admission = json.load(handle, parse_constant=reject)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(key)
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)

finite(admission)
if admission.get("authorized_runner_sha256") != sys.argv[2]:
    raise SystemExit("M518 r11 wrapper runner identity mismatch")
if admission.get("authorized_launch_wrapper_sha256") != sys.argv[3]:
    raise SystemExit("M518 r11 wrapper self identity mismatch")
if type(admission.get("authorized_invocations")) is not int or \
        admission["authorized_invocations"] != 1:
    raise SystemExit("M518 r11 wrapper invocation count is not strict integer one")
if admission.get("vcs_authorized") is not True:
    raise SystemExit("M518 r11 wrapper admission does not authorize VCS")
if admission.get("dc_authorized") is not False:
    raise SystemExit("M518 r11 wrapper admission unexpectedly authorizes DC")
if admission.get("required_result_path") != \
        "results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827":
    raise SystemExit("M518 r11 wrapper result identity drift")
if admission.get("required_launch_wrapper_path") != \
        "dc_handoff/scripts/launch_vcs_m518_matched_fixed_t10_atlif_r11.sh":
    raise SystemExit("M518 r11 wrapper path identity drift")
PY

unset M518_EXPECTED_RUNNER_SHA256 || true
unset M518_EXPECTED_STATIC_ADMISSION_SHA256 || true
unset M518_EXPECTED_LAUNCH_WRAPPER_SHA256 || true
exec env -u M518_RUN_DIR \
    M518_EXPECTED_RUNNER_SHA256="${task_runner_sha}" \
    M518_EXPECTED_STATIC_ADMISSION_SHA256="${task_admission_sha}" \
    M518_EXPECTED_LAUNCH_WRAPPER_SHA256="${task_wrapper_sha}" \
    "${task_runner}"
