#!/usr/bin/env bash
set -euo pipefail

# M537 source-authored, fail-closed wrapper for the single future M533
# functional VCS attempt.  The fixed launch release, its independent hammer,
# and the r4 runner static hammer intentionally do not exist at authoring time.
# Consequently this script cannot presently reach a result/attempt side effect.

if [[ $# -ne 0 ]]; then
  echo "ERROR: this exact-SHA runner accepts no overrides" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

TOP_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
BINDING_PLAN="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json"
SVA="${HW_ROOT}/verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv"
TB="${HW_ROOT}/tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r3.sv"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m533_m528_dead_write_only_1rw_source_only_contract_r3_20260827.json"
REPAIR_CONTRACT="${HW_ROOT}/contracts/m537_m533_r4_runner_fail_closed_repair_contract_r1_20260827.json"
SOURCE_STATIC_DIR="${HW_ROOT}/reviews/m533_m528_dead_write_only_1rw_source_static_hammer_r1_20260827"
SOURCE_STATIC_REVIEW="${SOURCE_STATIC_DIR}/review.json"
M536_FAIL_DIR="${HW_ROOT}/reviews/m536_m533_r3_functional_vcs_launch_admission_hammer_r1_20260827"
M536_FAIL_REVIEW="${M536_FAIL_DIR}/review.json"

# These fixed paths are deliberately absent in the source-only package.  A
# later independent sequence must create and double-seal them without changing
# this runner.  Missing paths are a hard failure before RESULT_DIR is created.
RUNNER_STATIC_DIR="${HW_ROOT}/reviews/m537_m533_r4_runner_source_static_hammer_r1_20260827"
RUNNER_STATIC_REVIEW="${RUNNER_STATIC_DIR}/review.json"
RELEASE_CANDIDATE="${HW_ROOT}/contracts/m538_m533_m528_dead_write_only_1rw_vcs_launch_release_candidate_r1_20260827.json"
LAUNCH_RELEASE="${HW_ROOT}/contracts/m538_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260827.json"
RELEASE_HAMMER_DIR="${HW_ROOT}/reviews/m538_m533_r4_functional_vcs_launch_release_candidate_hammer_r1_20260827"
RELEASE_HAMMER_REVIEW="${RELEASE_HAMMER_DIR}/review.json"

ASSET_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
ASSET_MANIFEST="${ASSET_ROOT}/SHA256SUMS"
FOUNDRY_SLOW_V="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
FOUNDRY_SLOW_DB="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"

# New identity: the forbidden r3 path is neither reused nor overwritten.
RESULT_DIR="${HW_ROOT}/results/m533_m528_dead_write_only_1rw_vcs_r2_20260827"
PREFLIGHT_DIR=""
MONITOR_PID=""
RESULT_CREATED=0

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${PREFLIGHT_DIR}" && -d "${PREFLIGHT_DIR}" ]]; then
    rm -rf -- "${PREFLIGHT_DIR}"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

strict_json_parse() {
  local json_path=$1
  python3 - "${json_path}" <<'PY'
import json
import math
import sys

def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise RuntimeError("duplicate JSON key: " + key)
        out[key] = value
    return out

def reject(token):
    raise RuntimeError("non-standard JSON token: " + token)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            finite(key)
            finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle, object_pairs_hook=pairs, parse_constant=reject)
finite(value)
PY
}

require_regular_sha() {
  local expected=$1 path=$2 actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/non-regular frozen file: ${path}"
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "SHA drift: ${path}: ${actual} != ${expected}"
}

verify_json_double_seal() {
  local json_path=$1 dir base
  dir="$(dirname -- "${json_path}")"
  base="$(basename -- "${json_path}")"
  [[ -f "${json_path}" && ! -L "${json_path}" && \
     -f "${json_path}.sha256" && ! -L "${json_path}.sha256" && \
     -f "${json_path}.sha256.seal.sha256" && ! -L "${json_path}.sha256.seal.sha256" ]] \
    || fail "missing/non-regular JSON member or outer seal: ${json_path}"
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256" >/dev/null)
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256.seal.sha256" >/dev/null)
  strict_json_parse "${json_path}"
}

verify_review_double_seal() {
  local review_dir=$1
  [[ -d "${review_dir}" && ! -L "${review_dir}" && \
     -f "${review_dir}/review.json" && ! -L "${review_dir}/review.json" && \
     -f "${review_dir}/SHA256SUMS" && ! -L "${review_dir}/SHA256SUMS" && \
     -f "${review_dir}/SHA256SUMS.seal.sha256" && ! -L "${review_dir}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular review or seals: ${review_dir}"
  (cd -- "${review_dir}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${review_dir}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  strict_json_parse "${review_dir}/review.json"
}

command -v python3 >/dev/null 2>&1 || fail "python3 unavailable"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum unavailable"

# Immutable functional sources, models, source contract, and the failed r3
# launch audit are checked before a future release is even considered.
require_regular_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1 "${TOP_RTL}"
require_regular_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
require_regular_sha db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983 "${BINDING_PLAN}"
require_regular_sha b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b "${SVA}"
require_regular_sha 73b9c6c45f9cd4a8185e386b9a13d674e888af938d3dbcbc29567ad40a558c32 "${TB}"
require_regular_sha 3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b "${SOURCE_CONTRACT}"
verify_json_double_seal "${SOURCE_CONTRACT}"
verify_json_double_seal "${REPAIR_CONTRACT}"

require_regular_sha c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${ASSET_MANIFEST}"
(cd -- "${ASSET_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
require_regular_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_SLOW_V}"
require_regular_sha cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${FOUNDRY_SLOW_DB}"
[[ "$(rg -c '^module[[:space:]]+TS1N28HPCPHVTB128X128M4S\b' "${FOUNDRY_SLOW_V}")" == 1 ]] \
  || fail "foundry slow .v lacks the unique required macro cell"
require_regular_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
require_regular_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

verify_review_double_seal "${SOURCE_STATIC_DIR}"
verify_review_double_seal "${M536_FAIL_DIR}"
python3 - "${SOURCE_STATIC_REVIEW}" "${M536_FAIL_REVIEW}" <<'PY'
import json
import sys

source = json.load(open(sys.argv[1], encoding="utf-8"))
failed = json.load(open(sys.argv[2], encoding="utf-8"))
assert source["schema"] == "m533_m528_dead_write_only_1rw_source_static_hammer_v1"
assert source["status"] == "PASS_M533_M528_DW1RW_SOURCE_STATIC_HAMMER"
assert source["verdict"] == "PASS"
assert source["score_100"] == 100
assert [source[k] for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0]
assert source["identity"]["source_contract_sha256"] == \
    "3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b"
assert failed["schema"] == "m536_m533_r3_functional_vcs_launch_admission_hammer_v1"
assert failed["verdict"] == "FAIL"
assert failed["status"].endswith("RUNNER_R4_REQUIRED")
assert failed["p0_count"] == 0 and failed["p1_count"] == 3
assert failed["required_next_gate"]["author_separately_admitted_runner_or_wrapper_r4"] is True
PY
require_regular_sha f1b557991ddf3d845f528ee757e1a9747576ddb7bc2b7ce4dbd9fb5986fb5087 "${M536_FAIL_REVIEW}"

# Future r4 source hammer.  It must be an independent 100/100 PASS that binds
# this live runner and the exact repair contract; a PASS for r3 is insufficient.
verify_review_double_seal "${RUNNER_STATIC_DIR}"
RUNNER_SHA="$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
REPAIR_CONTRACT_SHA="$(sha256sum -- "${REPAIR_CONTRACT}" | awk '{print $1}')"
RUNNER_STATIC_SHA="$(sha256sum -- "${RUNNER_STATIC_REVIEW}" | awk '{print $1}')"
python3 - "${RUNNER_STATIC_REVIEW}" "${RUNNER_SHA}" "${REPAIR_CONTRACT_SHA}" <<'PY'
import json
import sys

review = json.load(open(sys.argv[1], encoding="utf-8"))
assert review["schema"] == "m537_m533_r4_runner_source_static_hammer_v1"
assert review["status"] == "PASS_M537_M533_R4_RUNNER_SOURCE_STATIC_HAMMER"
assert review["verdict"] == "PASS"
assert review["score_100"] == 100
assert [review[k] for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0]
assert review["identity"]["runner_sha256"] == sys.argv[2]
assert review["identity"]["repair_contract_sha256"] == sys.argv[3]
assert review["decision"]["source_only_pass"] is True
assert review["decision"]["vcs_launch_authorized_now"] is False
PY

# The three-stage digest chain avoids an impossible release<->hammer circular
# hash: candidate -> independent hammer -> final launch_now release.  The final
# release binds the exact candidate and hammer members before any side effect.
verify_json_double_seal "${RELEASE_CANDIDATE}"
verify_review_double_seal "${RELEASE_HAMMER_DIR}"
verify_json_double_seal "${LAUNCH_RELEASE}"
RELEASE_CANDIDATE_SHA="$(sha256sum -- "${RELEASE_CANDIDATE}" | awk '{print $1}')"
LAUNCH_RELEASE_SHA="$(sha256sum -- "${LAUNCH_RELEASE}" | awk '{print $1}')"
RELEASE_HAMMER_SHA="$(sha256sum -- "${RELEASE_HAMMER_REVIEW}" | awk '{print $1}')"
python3 - "${RELEASE_CANDIDATE}" "${RELEASE_HAMMER_REVIEW}" "${LAUNCH_RELEASE}" \
  "${RUNNER_SHA}" "${REPAIR_CONTRACT_SHA}" "${RUNNER_STATIC_SHA}" \
  "${RELEASE_CANDIDATE_SHA}" "${LAUNCH_RELEASE_SHA}" "${RELEASE_HAMMER_SHA}" <<'PY'
import json
import sys

candidate_path, hammer_path, release_path = sys.argv[1:4]
runner_sha, repair_sha, static_sha, candidate_sha, release_sha, hammer_sha = sys.argv[4:]
candidate = json.load(open(candidate_path, encoding="utf-8"))
release = json.load(open(release_path, encoding="utf-8"))
hammer = json.load(open(hammer_path, encoding="utf-8"))

expected_auth = {
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
assert candidate["schema"] == \
    "m538_m533_m528_dead_write_only_1rw_vcs_launch_release_candidate_v1"
assert candidate["status"] == "READY_FOR_INDEPENDENT_RELEASE_CANDIDATE_HAMMER"
assert candidate["launch_now"] is False
assert isinstance(candidate.get("authorization"), dict)
assert set(candidate["authorization"]) == set(expected_auth)
assert candidate["authorization"] == expected_auth
candidate_id = candidate["identity"]
assert candidate_id["runner_sha256"] == runner_sha
assert candidate_id["repair_contract_sha256"] == repair_sha
assert candidate_id["runner_static_review_sha256"] == static_sha
assert candidate_id["m536_failed_launch_review_sha256"] == \
    "f1b557991ddf3d845f528ee757e1a9747576ddb7bc2b7ce4dbd9fb5986fb5087"
assert candidate_id["source_contract_sha256"] == \
    "3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b"
assert candidate["unique_attempt"]["result_path"] == \
    "results/m533_m528_dead_write_only_1rw_vcs_r2_20260827"
candidate_policy = candidate["resource_policy"]
assert candidate_policy == {
    "prelaunch_samples": 3,
    "sample_interval_seconds": 2,
    "mem_available_min_kib": 134217728,
    "swap_free_min_kib": 33554432,
    "commit_headroom_min_kib": 33554432,
    "cgroup_version": 1,
    "memory_failcnt_must_not_increase": True,
    "under_oom_must_equal_zero": True,
    "oom_kill_must_equal_zero": True,
    "missing_counter_is_failure": True,
}

assert hammer["schema"] == \
    "m538_m533_r4_functional_vcs_launch_release_candidate_hammer_v1"
assert hammer["status"] == \
    "PASS_M538_M533_R4_FUNCTIONAL_VCS_LAUNCH_RELEASE_CANDIDATE_HAMMER"
assert hammer["verdict"] == "PASS"
assert hammer["score_100"] == 100
assert [hammer[k] for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0]
hammer_id = hammer["identity"]
assert hammer_id["release_candidate_sha256"] == candidate_sha
assert hammer_id["runner_sha256"] == runner_sha
assert hammer_id["repair_contract_sha256"] == repair_sha
assert hammer_id["runner_static_review_sha256"] == static_sha
candidate_decision = hammer["decision"]
assert candidate_decision["release_candidate_pass"] is True
assert candidate_decision["closed_authorization_pass"] is True
assert candidate_decision["source_static_100_p0_p1_p2_zero"] is True
assert candidate_decision["final_launch_release_required"] is True
assert candidate_decision["vcs_launch_authorized_now"] is False

assert release["schema"] == \
    "m538_m533_m528_dead_write_only_1rw_vcs_launch_release_v1"
assert release["status"] == "AUTHORIZED_ONE_M533_M528_DW1RW_R4_VCS_RUN"
assert release["launch_now"] is True
assert isinstance(release.get("authorization"), dict)
assert set(release["authorization"]) == set(expected_auth)
assert release["authorization"] == expected_auth
identity = release["identity"]
assert identity["runner_sha256"] == runner_sha
assert identity["repair_contract_sha256"] == repair_sha
assert identity["runner_static_review_sha256"] == static_sha
assert identity["release_candidate_sha256"] == candidate_sha
assert identity["release_candidate_hammer_review_sha256"] == hammer_sha
assert identity["m536_failed_launch_review_sha256"] == \
    "f1b557991ddf3d845f528ee757e1a9747576ddb7bc2b7ce4dbd9fb5986fb5087"
assert identity["source_contract_sha256"] == \
    "3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b"
assert release["unique_attempt"]["result_path"] == \
    "results/m533_m528_dead_write_only_1rw_vcs_r2_20260827"
assert release["resource_policy"] == candidate_policy
assert release["release_chain"]["candidate_path"] == \
    "contracts/m538_m533_m528_dead_write_only_1rw_vcs_launch_release_candidate_r1_20260827.json"
assert release["release_chain"]["candidate_sha256"] == candidate_sha
assert release["release_chain"]["candidate_hammer_path"] == \
    "reviews/m538_m533_r4_functional_vcs_launch_release_candidate_hammer_r1_20260827/review.json"
assert release["release_chain"]["candidate_hammer_review_sha256"] == hammer_sha
assert release["release_chain"]["digest_chain"] == \
    "candidate_sha256 -> independent_hammer -> final_launch_now_release"
decision = release["decision"]
assert decision["fresh_collision_and_resource_gate_still_required"] is True
assert decision["exactly_one_vcs_attempt_authorized"] is True
assert decision["all_other_runs_authorized"] is False
PY

[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt identity already exists: ${RESULT_DIR}"

# No directory under results has been created above this point.  The temporary
# preflight directory is outside the attempt namespace and is always removed on
# any collision, resource, or release failure.
PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/m537_m533_vcs_preflight.XXXXXXXX")"

scan_same_uid_collisions() {
  local output=$1
  python3 - "${output}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

uid = os.getuid()
self_pid = os.getpid()
runner_pid = os.getppid()
ignored = {self_pid, runner_pid}
matches = []

for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid in ignored:
        continue
    try:
        if entry.stat().st_uid != uid:
            continue
        exe = os.path.basename(os.readlink(entry / "exe"))
        raw = (entry / "cmdline").read_bytes()
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    joined = " ".join(argv)
    # Some Synopsys launchers are interpreter scripts, so /proc/PID/exe can be
    # bash/perl while argv[0] or the first wrapper token is the actual tool.
    # Exact basenames avoid matching this runner's run_vcs_* filename.
    tool_tokens = {os.path.basename(token) for token in argv[:4]}
    tool_tokens.add(exe)
    cls = None
    direct_shells = {"dc_shell", "dc_shell-t", "fm_shell", "fm_shell_exec",
                      "pt_shell", "pt_shell_exec"}
    direct_hits = sorted(tool_tokens & direct_shells)
    if direct_hits:
        cls = direct_hits[0]
    elif "common_shell_exec" in tool_tokens and re.search(
            r"(?:^|\s)-shell\s+(?:dc_shell|dc_shell-t|fm_shell|pt_shell)(?:\s|$)",
            joined):
        cls = "common_shell_exec_for_dc_fm_pt_ptpx"
    elif any(token == "vcs" or token.startswith("vcs.") or
             token in {"vcs1", "vlogan", "vhdlan"} for token in tool_tokens):
        cls = "vcs"
    elif any(token == "simv" or token.startswith("simv.") for token in tool_tokens):
        cls = "simv"
    if cls is not None:
        matches.append({"pid": pid, "class": cls, "exe": exe, "argv": argv})

receipt = {
    "schema": "m537_m533_same_uid_collision_scan_v1",
    "uid": uid,
    "scanner_pid": self_pid,
    "runner_pid": runner_pid,
    "ignored_only_scanner_and_runner_pids": sorted(ignored),
    "forbidden_classes": [
        "dc_shell", "dc_shell-t", "Formality", "PrimeTime", "PTPX",
        "common_shell_exec for DC/FM/PT/PTPX", "vcs", "simv"
    ],
    "matches": matches,
    "verdict": "PASS" if not matches else "FAIL",
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
if matches:
    raise SystemExit(1)
PY
}

resolve_cgroup_v1() {
  local rel
  rel="$(awk -F: '$2 == "memory" {print $3}' /proc/self/cgroup)"
  [[ "${rel}" == /* && "${rel}" != *..* ]] || fail "cgroup-v1 memory controller path unavailable"
  CGROUP_SESSION_DIR="/sys/fs/cgroup/memory${rel}"
  CGROUP_USER_DIR="/sys/fs/cgroup/memory/user.slice"
  local dir file
  for dir in "${CGROUP_SESSION_DIR}" "${CGROUP_USER_DIR}"; do
    [[ -d "${dir}" && ! -L "${dir}" ]] || fail "missing cgroup-v1 directory: ${dir}"
    for file in memory.failcnt memory.oom_control memory.usage_in_bytes; do
      [[ -r "${dir}/${file}" && ! -L "${dir}/${file}" ]] \
        || fail "missing/unreadable cgroup-v1 counter: ${dir}/${file}"
    done
  done
}

read_oom_field() {
  local path=$1 field=$2 value
  value="$(awk -v key="${field}" '$1 == key {print $2}' "${path}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "missing/non-numeric ${field}: ${path}"
  printf '%s\n' "${value}"
}

resource_preflight() {
  local output=$1 sample
  : >"${output}"
  BASE_SESSION_FAILCNT="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"
  BASE_USER_FAILCNT="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
  [[ "${BASE_SESSION_FAILCNT}" =~ ^[0-9]+$ && "${BASE_USER_FAILCNT}" =~ ^[0-9]+$ ]] \
    || fail "non-numeric cgroup-v1 failcnt baseline"
  for sample in 1 2 3; do
    local limit committed available swap headroom session_fail user_fail
    local session_under session_kill user_under user_kill session_usage user_usage
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swap="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    [[ "${limit}" =~ ^[0-9]+$ && "${committed}" =~ ^[0-9]+$ && \
       "${available}" =~ ^[0-9]+$ && "${swap}" =~ ^[0-9]+$ ]] \
      || fail "missing/non-numeric /proc/meminfo resource counter"
    headroom=$((limit - committed))
    session_fail="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"
    user_fail="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
    session_usage="$(<"${CGROUP_SESSION_DIR}/memory.usage_in_bytes")"
    user_usage="$(<"${CGROUP_USER_DIR}/memory.usage_in_bytes")"
    session_under="$(read_oom_field "${CGROUP_SESSION_DIR}/memory.oom_control" under_oom)"
    session_kill="$(read_oom_field "${CGROUP_SESSION_DIR}/memory.oom_control" oom_kill)"
    user_under="$(read_oom_field "${CGROUP_USER_DIR}/memory.oom_control" under_oom)"
    user_kill="$(read_oom_field "${CGROUP_USER_DIR}/memory.oom_control" oom_kill)"
    printf 'sample=%s timestamp=%s mem_available_kib=%s swap_free_kib=%s commit_headroom_kib=%s session_failcnt=%s user_failcnt=%s session_under_oom=%s session_oom_kill=%s user_under_oom=%s user_oom_kill=%s session_usage_bytes=%s user_usage_bytes=%s\n' \
      "${sample}" "$(date --iso-8601=seconds)" "${available}" "${swap}" "${headroom}" \
      "${session_fail}" "${user_fail}" "${session_under}" "${session_kill}" \
      "${user_under}" "${user_kill}" "${session_usage}" "${user_usage}" >>"${output}"
    [[ "${available}" -ge 134217728 ]] || fail "MemAvailable below frozen 128 GiB gate"
    [[ "${swap}" -ge 33554432 ]] || fail "SwapFree below frozen 32 GiB gate"
    [[ "${headroom}" -ge 33554432 ]] || fail "commit headroom below frozen 32 GiB gate"
    [[ "${session_fail}" == "${BASE_SESSION_FAILCNT}" && \
       "${user_fail}" == "${BASE_USER_FAILCNT}" ]] || fail "cgroup-v1 failcnt increased during prelaunch"
    [[ "${session_under}" == 0 && "${session_kill}" == 0 && \
       "${user_under}" == 0 && "${user_kill}" == 0 ]] || fail "cgroup-v1 OOM state is nonzero"
    [[ "${sample}" -eq 3 ]] || sleep 2
  done
}

scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_initial.json" \
  || fail "same-UID Synopsys/VCS/simv collision present"
resolve_cgroup_v1
resource_preflight "${PREFLIGHT_DIR}/resource_prelaunch.log"
# Close the sampling race with a second scan immediately before the atomic
# result/attempt creation.
scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_final.json" \
  || fail "same-UID collision appeared during resource preflight"
[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt appeared during preflight"

# This mkdir is the sole attempt-consumption point.  Every release, seal,
# identity, collision, and resource failure above leaves RESULT_DIR absent.
mkdir -- "${RESULT_DIR}" || fail "atomic result/attempt creation failed"
RESULT_CREATED=1
cp -- "${PREFLIGHT_DIR}/collision_initial.json" "${RESULT_DIR}/"
cp -- "${PREFLIGHT_DIR}/collision_final.json" "${RESULT_DIR}/"
cp -- "${PREFLIGHT_DIR}/resource_prelaunch.log" "${RESULT_DIR}/"

resource_monitor() {
  local output=$1 violation=$2
  : >"${output}"
  while :; do
    local sf uf su sk uu uk
    if [[ ! -r "${CGROUP_SESSION_DIR}/memory.failcnt" || \
          ! -r "${CGROUP_USER_DIR}/memory.failcnt" || \
          ! -r "${CGROUP_SESSION_DIR}/memory.oom_control" || \
          ! -r "${CGROUP_USER_DIR}/memory.oom_control" ]]; then
      echo "missing_runtime_cgroup_counter" >>"${violation}"
      return
    fi
    sf="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"
    uf="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
    su="$(awk '$1 == "under_oom" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"
    sk="$(awk '$1 == "oom_kill" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"
    uu="$(awk '$1 == "under_oom" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"
    uk="$(awk '$1 == "oom_kill" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"
    printf 'timestamp=%s session_failcnt=%s user_failcnt=%s session_under_oom=%s session_oom_kill=%s user_under_oom=%s user_oom_kill=%s\n' \
      "$(date --iso-8601=seconds)" "${sf}" "${uf}" "${su}" "${sk}" "${uu}" "${uk}" >>"${output}"
    if [[ ! "${sf}" =~ ^[0-9]+$ || ! "${uf}" =~ ^[0-9]+$ || \
          ! "${su}" =~ ^[0-9]+$ || ! "${sk}" =~ ^[0-9]+$ || \
          ! "${uu}" =~ ^[0-9]+$ || ! "${uk}" =~ ^[0-9]+$ || \
          "${sf}" != "${BASE_SESSION_FAILCNT}" || "${uf}" != "${BASE_USER_FAILCNT}" || \
          "${su}" != 0 || "${sk}" != 0 || "${uu}" != 0 || "${uk}" != 0 ]]; then
      echo "runtime_failcnt_or_oom_violation" >>"${violation}"
      return
    fi
    sleep 1
  done
}

resource_monitor "${RESULT_DIR}/resource_runtime.log" "${RESULT_DIR}/RESOURCE_VIOLATION" &
MONITOR_PID=$!

cd -- "${RESULT_DIR}"
set +e
"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  -debug_access+pp +vcs+lic+wait \
  "${FOUNDRY_SLOW_V}" "${MACRO_RTL}" "${TOP_RTL}" "${SVA}" "${TB}" \
  -top tb_m528_dead_write_only_1rw_product_capture_r3 \
  -o simv 2>&1 | tee compile.log
compile_rc=${PIPESTATUS[0]}
set -e
[[ "${compile_rc}" -eq 0 ]] || fail "VCS compile failed rc=${compile_rc}; attempt consumed"

set +e
./simv 2>&1 | tee sim.log
sim_rc=${PIPESTATUS[0]}
set -e
[[ "${sim_rc}" -eq 0 ]] || fail "simv failed rc=${sim_rc}; attempt consumed"

kill "${MONITOR_PID}" >/dev/null 2>&1 || true
wait "${MONITOR_PID}" >/dev/null 2>&1 || true
MONITOR_PID=""
[[ ! -e RESOURCE_VIOLATION ]] || fail "runtime cgroup failcnt/OOM gate failed; attempt consumed"

[[ "$(rg -c '^PASS_M533_M528_DW1RW_R3_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)" == 1 ]] \
  || fail "missing/repeated exact functional PASS token"
[[ "$(rg -c '^COVERAGE_M533_M528_DW1RW_R3 ' sim.log)" == 1 ]] \
  || fail "missing/repeated exact coverage token"
COVERAGE_LINE="$(rg '^COVERAGE_M533_M528_DW1RW_R3 ' sim.log)"
for field in dead_plus_read deadline_read_write same_address_forward \
    pending_plus_forward full_no_credit liveness_sequences parent_modes \
    stalled_raw_recovery pingpong_overlap endpoint_rows all_slices; do
  value="$(sed -nE "s/.* ${field}=([0-9]+)( |$).*/\\1/p" <<<"${COVERAGE_LINE}")"
  [[ "${value}" =~ ^[0-9]+$ && "${value}" -ge 1 ]] \
    || fail "normal cover ${field} missed: ${value:-missing}"
done
[[ "${COVERAGE_LINE}" == *" minima=1 normal_covers=11"* ]] \
  || fail "coverage summary does not freeze eleven minima"
[[ "$(rg -c '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)" == 1 ]] \
  || fail "missing/repeated P2 strength token"
P2_LINE="$(rg '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)"
P2_PAIRS="$(sed -nE 's/.* consecutive_distinct_reads=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")"
P2_RESPONSES="$(sed -nE 's/.* response_identity_checks=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")"
[[ "${P2_PAIRS}" =~ ^[0-9]+$ && "${P2_PAIRS}" -ge 1 ]] || fail "distinct read pair missed"
[[ "${P2_RESPONSES}" =~ ^[0-9]+$ && "${P2_RESPONSES}" -ge 2 ]] || fail "response identity minimum missed"
PASS_LINE="$(rg '^PASS_M533_M528_DW1RW_R3_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)"
for field in dirty_reserved stale_epoch overflow wrong_parent read_before_write parent_only_nonzero; do
  value="$(sed -nE "s/.* ${field}=([0-9]+)( |$).*/\\1/p" <<<"${PASS_LINE}")"
  [[ "${value}" == 1 ]] || fail "protocol attack ${field} did not occur exactly once"
done
[[ "${PASS_LINE}" == *" attacks=6 "* ]] || fail "functional PASS did not freeze six attacks"
if rg -n 'Assertion.*failed|Error-\[SVA|\$error|\$fatal|normal scoreboard errors|protocol attack not detected' \
    compile.log sim.log; then
  fail "VCS/SVA/testbench failure signature detected"
fi

sha256sum -- collision_initial.json collision_final.json resource_prelaunch.log \
  resource_runtime.log compile.log sim.log simv > SHA256SUMS
sha256sum -- SHA256SUMS > SHA256SUMS.seal.sha256
echo "PASS_M533_M528_DW1RW_R4_FAIL_CLOSED_EXACT_SHA_VCS_RUNNER"
