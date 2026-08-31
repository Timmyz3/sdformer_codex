#!/usr/bin/env bash
set -euo pipefail

# Source-only runner.  This file is authored but MUST NOT be executed until a
# separate, double-sealed VCS launch admission and an independent source static
# hammer both exist and pass the exact semantic checks below.

if [[ $# -ne 0 ]]; then
  echo "ERROR: this exact-SHA runner accepts no path, source, model, top, or output overrides" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"

TOP_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
BINDING_PLAN="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json"
SVA="${HW_ROOT}/verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv"
TB="${HW_ROOT}/tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r3.sv"
CONTRACT="${HW_ROOT}/contracts/m533_m528_dead_write_only_1rw_source_only_contract_r3_20260827.json"
STATIC_REVIEW_DIR="${HW_ROOT}/reviews/m533_m528_dead_write_only_1rw_source_static_hammer_r1_20260827"
STATIC_REVIEW="${STATIC_REVIEW_DIR}/review.json"
VCS_ADMISSION="${HW_ROOT}/contracts/m533_m528_dead_write_only_1rw_vcs_launch_admission_r1_20260827.json"
ASSET_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
ASSET_MANIFEST="${ASSET_ROOT}/SHA256SUMS"
FOUNDRY_SLOW_V="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
FOUNDRY_SLOW_DB="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
RESULT_DIR="${HW_ROOT}/results/m533_m528_dead_write_only_1rw_vcs_r1_20260827"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_sha() {
  local expected="$1"
  local path="$2"
  [[ -f "${path}" ]] || fail "missing frozen file: ${path}"
  local actual
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "SHA drift: ${path}: ${actual} != ${expected}"
}

verify_json_double_seal() {
  local json_path="$1"
  local dir base
  dir="$(dirname -- "${json_path}")"
  base="$(basename -- "${json_path}")"
  [[ -f "${json_path}.sha256" && -f "${json_path}.sha256.seal.sha256" ]] \
    || fail "missing JSON member/outer seal: ${json_path}"
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256")
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256.seal.sha256")
  strict_json_parse "${json_path}"
}

verify_review_double_seal() {
  local review_dir="$1"
  [[ -f "${review_dir}/SHA256SUMS" && -f "${review_dir}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing review member/outer seal: ${review_dir}"
  (cd -- "${review_dir}" && sha256sum -c SHA256SUMS)
  (cd -- "${review_dir}" && sha256sum -c SHA256SUMS.seal.sha256)
  strict_json_parse "${review_dir}/review.json"
}

strict_json_parse() {
  local json_path="$1"
  python3 - "${json_path}" <<'PY'
import json
import sys

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise RuntimeError("duplicate JSON key: " + key)
        result[key] = value
    return result

def reject(token):
    raise RuntimeError("non-standard JSON token: " + token)

with open(sys.argv[1], "r") as handle:
    json.load(handle, object_pairs_hook=pairs, parse_constant=reject)
PY
}

# Frozen author sources and source-only contract.
require_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1 "${TOP_RTL}"
require_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
require_sha db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983 "${BINDING_PLAN}"
require_sha b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b "${SVA}"
require_sha 73b9c6c45f9cd4a8185e386b9a13d674e888af938d3dbcbc29567ad40a558c32 "${TB}"
require_sha 3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b "${CONTRACT}"
verify_json_double_seal "${CONTRACT}"

# The private manifest and selected foundry views are mandatory.  A local
# behavioral SRAM substitute, missing cell, or path override is never accepted.
require_sha c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${ASSET_MANIFEST}"
(cd -- "${ASSET_ROOT}" && sha256sum -c SHA256SUMS)
require_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_SLOW_V}"
require_sha cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${FOUNDRY_SLOW_DB}"
[[ "$(rg -c '^module[[:space:]]+TS1N28HPCPHVTB128X128M4S\b' "${FOUNDRY_SLOW_V}")" == "1" ]] \
  || fail "foundry slow .v does not define exactly one required macro cell"

# Future independent static review is mandatory and is reparsed semantically;
# a sealed FAIL review cannot authorize a launch.
command -v python3 >/dev/null 2>&1 || fail "python3 is unavailable"
command -v jq >/dev/null 2>&1 || fail "jq is unavailable"
verify_review_double_seal "${STATIC_REVIEW_DIR}"
[[ "$(jq -r '.schema' "${STATIC_REVIEW}")" == "m533_m528_dead_write_only_1rw_source_static_hammer_v1" ]] \
  || fail "static review schema mismatch"
[[ "$(jq -r '.status' "${STATIC_REVIEW}")" == "PASS_M533_M528_DW1RW_SOURCE_STATIC_HAMMER" ]] \
  || fail "static review is not PASS"
[[ "$(jq -r '.p0_count' "${STATIC_REVIEW}")" == "0" && "$(jq -r '.p1_count' "${STATIC_REVIEW}")" == "0" ]] \
  || fail "static review has blocking findings"
[[ "$(jq -r '.identity.source_contract_sha256' "${STATIC_REVIEW}")" == "3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b" ]] \
  || fail "static review did not bind the exact source contract"

# A separate launch admission is also mandatory.  Its authorization object is
# a closed schema: exactly one VCS attempt and explicit zero for every other
# HDL/EDA/CPU/GPU/remote class.  Unknown authorization keys are a hard failure.
verify_json_double_seal "${VCS_ADMISSION}"
RUNNER_SHA="$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
STATIC_REVIEW_SHA="$(sha256sum -- "${STATIC_REVIEW}" | awk '{print $1}')"
python3 - "${VCS_ADMISSION}" "${RUNNER_SHA}" "${STATIC_REVIEW_SHA}" <<'PY'
import json
import sys

path, runner_sha, static_sha = sys.argv[1:]
with open(path, "r") as handle:
    admission = json.load(handle)

if admission.get("schema") != "m533_m528_dead_write_only_1rw_vcs_launch_admission_v1":
    raise SystemExit("closed admission schema mismatch")
if admission.get("status") != "AUTHORIZED_ONE_M533_M528_DW1RW_VCS_RUN_AFTER_STATIC_PASS":
    raise SystemExit("closed admission status mismatch")

expected_authorization = {
    "vcs_runs": 1,
    "iverilog_runs": 0,
    "verilator_runs": 0,
    "dc_runs": 0,
    "formality_runs": 0,
    "pt_runs": 0,
    "ptpx_runs": 0,
    "cpu_runs": 0,
    "gpu_runs": 0,
    "network_or_remote_jobs": 0,
}
authorization = admission.get("authorization")
if not isinstance(authorization, dict):
    raise SystemExit("closed authorization object missing")
if set(authorization) != set(expected_authorization):
    unknown = sorted(set(authorization) - set(expected_authorization))
    missing = sorted(set(expected_authorization) - set(authorization))
    raise SystemExit(
        "authorization key set mismatch unknown=%r missing=%r" % (unknown, missing)
    )
if authorization != expected_authorization:
    raise SystemExit("authorization values are not exactly one VCS and all-other-zero")

identity = admission.get("identity")
if not isinstance(identity, dict):
    raise SystemExit("identity object missing")
expected_identity = {
    "runner_sha256": runner_sha,
    "source_contract_sha256":
        "3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b",
    "static_review_sha256": static_sha,
    "foundry_manifest_sha256":
        "c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f",
    "foundry_slow_v_sha256":
        "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
}
for key, expected in expected_identity.items():
    if identity.get(key) != expected:
        raise SystemExit("launch identity mismatch for " + key)
PY

[[ ! -e "${RESULT_DIR}" ]] || fail "refuse to overwrite or reuse result directory: ${RESULT_DIR}"
command -v vcs >/dev/null 2>&1 || fail "Synopsys VCS is unavailable"

mkdir -p -- "${RESULT_DIR}"
cd -- "${RESULT_DIR}"
vcs -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  -debug_access+pp +vcs+lic+wait \
  "${FOUNDRY_SLOW_V}" \
  "${MACRO_RTL}" \
  "${TOP_RTL}" \
  "${SVA}" \
  "${TB}" \
  -top tb_m528_dead_write_only_1rw_product_capture_r3 \
  -o simv 2>&1 | tee compile.log

./simv 2>&1 | tee sim.log
[[ "$(rg -c '^PASS_M533_M528_DW1RW_R3_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)" == "1" ]] \
  || fail "missing or repeated exact PASS token"
[[ "$(rg -c '^COVERAGE_M533_M528_DW1RW_R3 ' sim.log)" == "1" ]] \
  || fail "missing or repeated exact normal-coverage summary token"
COVERAGE_LINE="$(rg '^COVERAGE_M533_M528_DW1RW_R3 ' sim.log)"
for field in dead_plus_read deadline_read_write same_address_forward \
    pending_plus_forward full_no_credit liveness_sequences parent_modes \
    stalled_raw_recovery pingpong_overlap endpoint_rows all_slices; do
  value="$(sed -nE "s/.* ${field}=([0-9]+)( |$).*/\\1/p" <<<"${COVERAGE_LINE}")"
  [[ "${value}" =~ ^[0-9]+$ && "${value}" -ge 1 ]] \
    || fail "normal cover ${field} missed minimum one: ${value:-missing}"
done
[[ "${COVERAGE_LINE}" == *" minima=1 normal_covers=11" ]] \
  || fail "coverage summary does not freeze eleven minima"
[[ "$(rg -c '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)" == "1" ]] \
  || fail "missing or repeated exact P2 strength token"
P2_LINE="$(rg '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)"
P2_PAIRS="$(sed -nE 's/.* consecutive_distinct_reads=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")"
P2_RESPONSES="$(sed -nE 's/.* response_identity_checks=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")"
[[ "${P2_PAIRS}" =~ ^[0-9]+$ && "${P2_PAIRS}" -ge 1 ]] \
  || fail "consecutive distinct read minimum missed"
[[ "${P2_RESPONSES}" =~ ^[0-9]+$ && "${P2_RESPONSES}" -ge 2 ]] \
  || fail "foundry response identity minimum missed"
PASS_LINE="$(rg '^PASS_M533_M528_DW1RW_R3_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)"
for attack_field in dirty_reserved stale_epoch overflow wrong_parent \
    read_before_write parent_only_nonzero; do
  attack_value="$(sed -nE "s/.* ${attack_field}=([0-9]+)( |$).*/\\1/p" <<<"${PASS_LINE}")"
  [[ "${attack_value}" == "1" ]] \
    || fail "protocol attack ${attack_field} did not occur exactly once"
done
[[ "${PASS_LINE}" == *" attacks=6 "* ]] \
  || fail "functional PASS did not freeze six attacks"
if rg -n 'Assertion.*failed|Error-\[SVA|\$error|\$fatal|normal scoreboard errors|protocol attack not detected' compile.log sim.log; then
  fail "VCS/SVA/testbench failure signature detected"
fi

sha256sum -- compile.log sim.log simv > SHA256SUMS
sha256sum -- SHA256SUMS > SHA256SUMS.seal.sha256
echo "PASS_M533_M528_DW1RW_EXACT_SHA_VCS_RUNNER"
