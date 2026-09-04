#!/usr/bin/env bash
set -euo pipefail

# M547 source-authored r6 wrapper for one future M533 functional VCS attempt.
# It is intentionally non-executable at authoring time: the independent source
# hammer, admission-candidate hammer, final release, and final-release hammer do
# not yet exist.  Every post-mkdir exit is forced into a double-sealed success
# or FAILED_DO_NOT_CITE terminal state.

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
TB="${HW_ROOT}/tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r4.sv"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m547_m533_m528_dead_write_only_1rw_source_only_contract_r4_20260827.json"
SOURCE_STATIC_DIR="${HW_ROOT}/reviews/m547_m533_r6_source_static_hammer_r1_20260827"
SOURCE_STATIC_REVIEW="${SOURCE_STATIC_DIR}/review.json"
M543_RELEASE_HAMMER_DIR="${HW_ROOT}/reviews/m543_m540_m533_r5_final_launch_release_hammer_r1_20260827"
M543_RELEASE_HAMMER_REVIEW="${M543_RELEASE_HAMMER_DIR}/review.json"
M544_FAILURE_HAMMER_DIR="${HW_ROOT}/reviews/m544_m533_r5_vcs_compile_failure_hammer_r1_20260827"
M544_FAILURE_HAMMER_REVIEW="${M544_FAILURE_HAMMER_DIR}/review.json"
OLD_RESULT_DIR="${HW_ROOT}/results/m533_m528_dead_write_only_1rw_vcs_r3_20260827"

# These fixed paths are deliberately absent in the source-only package.  A
# later independent sequence must create and double-seal them without changing
# this runner.  Missing paths are a hard failure before RESULT_DIR is created.
RUNNER_STATIC_DIR="${HW_ROOT}/reviews/m547_m533_r6_source_static_hammer_r1_20260827"
RUNNER_STATIC_REVIEW="${RUNNER_STATIC_DIR}/review.json"
RELEASE_CANDIDATE="${HW_ROOT}/contracts/m547_m533_m528_dead_write_only_1rw_vcs_launch_admission_candidate_r1_20260827.json"
LAUNCH_RELEASE="${HW_ROOT}/contracts/m547_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260827.json"
RELEASE_HAMMER_DIR="${HW_ROOT}/reviews/m547_m533_r6_vcs_launch_admission_candidate_hammer_r1_20260827"
RELEASE_HAMMER_REVIEW="${RELEASE_HAMMER_DIR}/review.json"
FINAL_RELEASE_HAMMER_DIR="${HW_ROOT}/reviews/m547_m533_r6_vcs_final_launch_release_hammer_r1_20260827"
FINAL_RELEASE_HAMMER_REVIEW="${FINAL_RELEASE_HAMMER_DIR}/review.json"

ASSET_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
ASSET_MANIFEST="${ASSET_ROOT}/SHA256SUMS"
FOUNDRY_SLOW_V="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
FOUNDRY_SLOW_DB="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"

# New identity: the consumed r3 partial is never reused or completed in place.
RESULT_DIR="${HW_ROOT}/results/m547_m533_m528_dead_write_only_1rw_vcs_r4_20260827"
PREFLIGHT_DIR=""
MONITOR_PID=""
RESULT_CREATED=0
TERMINAL_SEALED=0
CURRENT_PHASE="pre_mkdir_source_and_release_gate"
CHILD_RC="not_started"
MONITOR_STATUS="not_started"
FAILURE_MESSAGE=""

fail() {
  FAILURE_MESSAGE="$*"
  echo "ERROR phase=${CURRENT_PHASE}: $*" >&2
  exit 1
}

write_terminal_receipt_and_seal() {
  local kind=$1 runner_rc=$2 receipt marker status
  [[ "${RESULT_CREATED}" -eq 1 && -d "${RESULT_DIR}" ]] || return 1
  [[ "${TERMINAL_SEALED}" -eq 0 ]] || return 0
  if [[ "${kind}" == "failure" ]]; then
    receipt="${RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"
    marker="${RESULT_DIR}/FAILED_DO_NOT_CITE"
    status="FAILED_DO_NOT_CITE"
  else
    receipt="${RESULT_DIR}/RUN_COMPLETE.json"
    marker="${RESULT_DIR}/RUN_COMPLETE.txt"
    status="PASS_FUNCTIONAL_VCS_ONLY"
  fi
  RECEIPT_KIND="${kind}" RECEIPT_STATUS="${status}" RUNNER_RC="${runner_rc}" \
  CURRENT_PHASE_ENV="${CURRENT_PHASE}" CHILD_RC_ENV="${CHILD_RC}" \
  MONITOR_STATUS_ENV="${MONITOR_STATUS}" FAILURE_MESSAGE_ENV="${FAILURE_MESSAGE}" \
  RESULT_DIR_ENV="${RESULT_DIR}" SOURCE_CONTRACT_ENV="${SOURCE_CONTRACT}" \
  TB_ENV="${TB}" TOP_RTL_ENV="${TOP_RTL}" SVA_ENV="${SVA}" \
  MACRO_RTL_ENV="${MACRO_RTL}" BINDING_PLAN_ENV="${BINDING_PLAN}" \
  python3 -I - "${receipt}" <<'PY' || return 11
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR_ENV"])
excluded = {
    "SHA256SUMS", "SHA256SUMS.seal.sha256",
    "RUN_FAILED_OR_INCOMPLETE.json", "RUN_COMPLETE.json",
    "FAILED_DO_NOT_CITE", "RUN_COMPLETE.txt",
}
inventory = []
hashes = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        inventory.append({"path": str(path.relative_to(root)), "type": "symlink_forbidden"})
    elif path.is_file() and path.name not in excluded:
        rel = str(path.relative_to(root))
        data = path.read_bytes()
        inventory.append({"path": rel, "type": "regular", "bytes": len(data)})
        hashes[rel] = hashlib.sha256(data).hexdigest()

def file_hash(name):
    path = root / name
    if not path.is_file() or path.is_symlink():
        return {"present": False, "sha256": None}
    return {"present": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

immutable = {}
for key, env in [
    ("source_contract", "SOURCE_CONTRACT_ENV"), ("testbench_r4", "TB_ENV"),
    ("top_r2", "TOP_RTL_ENV"), ("sva_r2", "SVA_ENV"),
    ("macro_adapter", "MACRO_RTL_ENV"), ("macro_binding_plan", "BINDING_PLAN_ENV"),
]:
    path = Path(os.environ[env])
    immutable[key] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file() and not path.is_symlink() else None,
    }

receipt = {
    "schema": "m547_m533_r6_atomic_terminal_receipt_v1",
    "status": os.environ["RECEIPT_STATUS"],
    "kind": os.environ["RECEIPT_KIND"],
    "paper_citable": False,
    "phase": os.environ["CURRENT_PHASE_ENV"],
    "runner_exit_rc": os.environ["RUNNER_RC"],
    "child_rc": os.environ["CHILD_RC_ENV"],
    "monitor_status": os.environ["MONITOR_STATUS_ENV"],
    "failure_message": os.environ["FAILURE_MESSAGE_ENV"],
    "result_identity": str(root),
    "immutable_source_hashes": immutable,
    "resource_status_and_hashes": {
        name: file_hash(name) for name in [
            "resource_prelaunch.log", "resource_runtime.log", "RESOURCE_HEARTBEAT",
            "RESOURCE_FINAL_REQUEST", "RESOURCE_FINAL_ACK", "RESOURCE_VIOLATION"]
    },
    "collision_status_and_hashes": {
        name: file_hash(name) for name in [
            "collision_initial.json", "collision_final.json", "collision_postmkdir.json"]
    },
    "artifact_inventory_before_receipt": inventory,
    "artifact_sha256_before_receipt": hashes,
    "terminal_rule": "failure is FAILED_DO_NOT_CITE; success is functional-VCS-only and needs independent receipt hammer",
}
Path(os.sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
PY
  if [[ "${kind}" == "failure" ]]; then
    printf 'FAILED_DO_NOT_CITE phase=%s runner_rc=%s child_rc=%s monitor_status=%s\n' \
      "${CURRENT_PHASE}" "${runner_rc}" "${CHILD_RC}" "${MONITOR_STATUS}" >"${marker}" \
      || return 12
  else
    printf 'PASS_FUNCTIONAL_VCS_ONLY phase=%s runner_rc=%s child_rc=%s monitor_status=%s\n' \
      "${CURRENT_PHASE}" "${runner_rc}" "${CHILD_RC}" "${MONITOR_STATUS}" >"${marker}" \
      || return 12
  fi
  (
    cd -- "${RESULT_DIR}"
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
  ) || return 13
  TERMINAL_SEALED=1
}

cleanup() {
  local rc=$?
  local monitor_wait_rc=0
  trap - EXIT INT TERM HUP
  if [[ -n "${MONITOR_PID}" ]]; then
    if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
      kill "${MONITOR_PID}" >/dev/null 2>&1 || :
    fi
    set +e
    wait "${MONITOR_PID}" >/dev/null 2>&1
    monitor_wait_rc=$?
    set -e
    MONITOR_PID=""
    MONITOR_STATUS="cleanup_wait_rc_${monitor_wait_rc}"
  fi
  if [[ "${RESULT_CREATED}" -eq 1 && "${TERMINAL_SEALED}" -eq 0 ]]; then
    set +e
    write_terminal_receipt_and_seal failure "${rc}"
    local seal_rc=$?
    set -e
    if [[ "${seal_rc}" -ne 0 ]]; then
      echo "CRITICAL: unable to double-seal consumed failure result ${RESULT_DIR}" >&2
      rc=125
    fi
  fi
  if [[ -n "${PREFLIGHT_DIR}" && -d "${PREFLIGHT_DIR}" ]]; then
    rm -rf -- "${PREFLIGHT_DIR}"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'CURRENT_PHASE="signal_int"; CHILD_RC="130"; exit 130' INT
trap 'CURRENT_PHASE="signal_term_or_hup"; CHILD_RC="143"; exit 143' TERM HUP

strict_json_parse() {
  local json_path=$1
  python3 -I - "${json_path}" <<'PY'
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
command -v find >/dev/null 2>&1 || fail "find unavailable"
command -v sort >/dev/null 2>&1 || fail "sort unavailable"
command -v xargs >/dev/null 2>&1 || fail "xargs unavailable"
command -v rg >/dev/null 2>&1 || fail "rg unavailable"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be unset or empty"

# Immutable functional source identity.
require_regular_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1 "${TOP_RTL}"
require_regular_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
require_regular_sha db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983 "${BINDING_PLAN}"
require_regular_sha b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b "${SVA}"
require_regular_sha 72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff "${TB}"
require_regular_sha 273426f4b7deb6381596201d15b309b21f2185bb466b687649de9ed15f37d036 "${SOURCE_CONTRACT}"
verify_json_double_seal "${SOURCE_CONTRACT}"

# The consumed r3 partial is immutable failure provenance, never an input that
# may be repaired or completed by this new attempt.
verify_review_double_seal "${M543_RELEASE_HAMMER_DIR}"
verify_review_double_seal "${M544_FAILURE_HAMMER_DIR}"
require_regular_sha 9c4dd1ab0f66cb09ef6f3e959c806efb38beba0c36936d99151afc5da4853a3a \
  "${M543_RELEASE_HAMMER_REVIEW}"
require_regular_sha eefb742d31dd3b9a417dbaf206f513ce7342ca9a5a5cac79e5d69e2911fae947 \
  "${M544_FAILURE_HAMMER_REVIEW}"
[[ -d "${OLD_RESULT_DIR}" && ! -L "${OLD_RESULT_DIR}" ]] \
  || fail "old consumed result identity missing or symlinked"
require_regular_sha 0b0a0fcd88e5dff53e82c4ccfe218aa361c4fcf696aeed1e397d93fffe8b50ef \
  "${OLD_RESULT_DIR}/compile.log"
[[ ! -e "${OLD_RESULT_DIR}/simv" && ! -e "${OLD_RESULT_DIR}/sim.log" && \
   ! -e "${OLD_RESULT_DIR}/SHA256SUMS" && \
   ! -e "${OLD_RESULT_DIR}/SHA256SUMS.seal.sha256" ]] \
  || fail "old FAILED_UNSEALED_DO_NOT_CITE inventory drifted"
python3 -I - "${M543_RELEASE_HAMMER_REVIEW}" "${M544_FAILURE_HAMMER_REVIEW}" <<'PY'
import json
import sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

with open(sys.argv[1], encoding="utf-8") as handle:
    prior_release = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    failure = json.load(handle)
require(prior_release.get("verdict") == "PASS", "M543 prior release hammer mismatch")
require(prior_release.get("score_100") == 100, "M543 prior release score mismatch")
require(failure.get("schema") == "m544_m533_r5_vcs_compile_failure_hammer_v1",
        "M544 schema mismatch")
require(failure.get("verdict") == "FAIL", "M544 verdict mismatch")
require(failure.get("score_100") == 35, "M544 score mismatch")
require([failure.get(k) for k in ("p0_count", "p1_count", "p2_count")] ==
        [1, 1, 1], "M544 P0/P1/P2 mismatch")
result = failure.get("result_identity", {})
require(result.get("attempt_consumed") is True, "M544 attempt-consumed mismatch")
require(result.get("permanent_status") == "FAILED_UNSEALED_DO_NOT_CITE",
        "M544 permanent status mismatch")
require(result.get("reuse_forbidden") is True, "M544 reuse gate mismatch")
PY

# Foundry/VCS identities are checked before any attempt namespace is created.
require_regular_sha c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${ASSET_MANIFEST}"
(cd -- "${ASSET_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
require_regular_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_SLOW_V}"
require_regular_sha cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${FOUNDRY_SLOW_DB}"
[[ "$(rg -c '^module[[:space:]]+TS1N28HPCPHVTB128X128M4S\b' "${FOUNDRY_SLOW_V}")" == 1 ]] \
  || fail "foundry slow .v lacks the unique required macro cell"
require_regular_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
require_regular_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

# Fresh r6 source-static review. It must bind this exact live runner and TB.
verify_review_double_seal "${SOURCE_STATIC_DIR}"
RUNNER_SHA="$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
SOURCE_CONTRACT_SHA="$(sha256sum -- "${SOURCE_CONTRACT}" | awk '{print $1}')"
RUNNER_STATIC_SHA="$(sha256sum -- "${RUNNER_STATIC_REVIEW}" | awk '{print $1}')"
python3 -I - "${RUNNER_STATIC_REVIEW}" "${RUNNER_SHA}" "${SOURCE_CONTRACT_SHA}" <<'PY'
import json
import sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

with open(sys.argv[1], encoding="utf-8") as handle:
    review = json.load(handle)
require(review.get("schema") == "m547_m533_r6_source_static_hammer_v1",
        "r6 source-static schema mismatch")
require(review.get("status") == "PASS_M547_M533_R6_SOURCE_STATIC_HAMMER",
        "r6 source-static status mismatch")
require(review.get("verdict") == "PASS", "r6 source-static verdict mismatch")
require(type(review.get("score_100")) is int and review["score_100"] == 100,
        "r6 source-static score mismatch")
require([review.get(k) for k in ("p0_count", "p1_count", "p2_count")] ==
        [0, 0, 0], "r6 source-static P0/P1/P2 mismatch")
identity = review.get("identity", {})
require(identity.get("runner_sha256") == sys.argv[2],
        "r6 source-static runner binding mismatch")
require(identity.get("source_contract_sha256") == sys.argv[3],
        "r6 source-static contract binding mismatch")
require(identity.get("tb_r4_sha256") ==
        "72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff",
        "r6 source-static TB binding mismatch")
decision = review.get("decision", {})
require(decision.get("source_only_pass") is True,
        "r6 source-static decision mismatch")
require(decision.get("vcs_launch_authorized_now") is False,
        "r6 source-static must not authorize launch")
PY

# Four-stage digest chain:
# launch_now=false candidate -> fresh candidate hammer -> launch_now=true
# release -> fresh final-release hammer. Only the last review may authorize the
# one VCS attempt, and all four members are checked before mkdir.
verify_json_double_seal "${RELEASE_CANDIDATE}"
verify_review_double_seal "${RELEASE_HAMMER_DIR}"
verify_json_double_seal "${LAUNCH_RELEASE}"
verify_review_double_seal "${FINAL_RELEASE_HAMMER_DIR}"
RELEASE_CANDIDATE_SHA="$(sha256sum -- "${RELEASE_CANDIDATE}" | awk '{print $1}')"
RELEASE_HAMMER_SHA="$(sha256sum -- "${RELEASE_HAMMER_REVIEW}" | awk '{print $1}')"
LAUNCH_RELEASE_SHA="$(sha256sum -- "${LAUNCH_RELEASE}" | awk '{print $1}')"
FINAL_RELEASE_HAMMER_SHA="$(sha256sum -- "${FINAL_RELEASE_HAMMER_REVIEW}" | awk '{print $1}')"
python3 -I - "${RELEASE_CANDIDATE}" "${RELEASE_HAMMER_REVIEW}" \
  "${LAUNCH_RELEASE}" "${FINAL_RELEASE_HAMMER_REVIEW}" \
  "${RUNNER_SHA}" "${SOURCE_CONTRACT_SHA}" "${RUNNER_STATIC_SHA}" \
  "${RELEASE_CANDIDATE_SHA}" "${RELEASE_HAMMER_SHA}" \
  "${LAUNCH_RELEASE_SHA}" "${FINAL_RELEASE_HAMMER_SHA}" <<'PY'
import json
import sys

(candidate_path, candidate_hammer_path, release_path, final_hammer_path,
 runner_sha, source_contract_sha, static_sha, candidate_sha,
 candidate_hammer_sha, release_sha, final_hammer_sha) = sys.argv[1:]
with open(candidate_path, encoding="utf-8") as handle:
    candidate = json.load(handle)
with open(candidate_hammer_path, encoding="utf-8") as handle:
    candidate_hammer = json.load(handle)
with open(release_path, encoding="utf-8") as handle:
    release = json.load(handle)
with open(final_hammer_path, encoding="utf-8") as handle:
    final_hammer = json.load(handle)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

expected_auth = {
    "vcs_runs": 1, "iverilog_runs": 0, "verilator_runs": 0,
    "dc_runs": 0, "formality_runs": 0, "pt_runs": 0, "ptpx_runs": 0,
    "cpu_runs": 0, "gpu_runs": 0, "network_or_remote_jobs": 0,
}
require(candidate.get("schema") ==
        "m547_m533_m528_dead_write_only_1rw_vcs_launch_admission_candidate_v1",
        "candidate schema mismatch")
require(candidate.get("status") ==
        "SOURCE_CANDIDATE_ONLY__FRESH_STATIC_AND_ADMISSION_HAMMERS_REQUIRED",
        "candidate status mismatch")
require(candidate.get("launch_now") is False, "candidate launch_now must be false")
require(candidate.get("authorization") == expected_auth,
        "candidate closed authorization mismatch")
expected_policy = {
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
    "same_uid_synopsys_vcs_simv_collision_must_be_zero": True,
}
require(candidate.get("resource_policy") == expected_policy,
        "candidate resource policy mismatch")
candidate_id = candidate.get("identity", {})
require(candidate_id.get("runner_sha256") == runner_sha,
        "candidate runner binding mismatch")
require(candidate_id.get("source_contract_sha256") == source_contract_sha,
        "candidate source-contract binding mismatch")
require(candidate_id.get("tb_r4_sha256") ==
        "72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff",
        "candidate TB binding mismatch")
require(candidate.get("unique_attempt", {}).get("result_path") ==
        "results/m547_m533_m528_dead_write_only_1rw_vcs_r4_20260827",
        "candidate result identity mismatch")

require(candidate_hammer.get("schema") ==
        "m547_m533_r6_vcs_launch_admission_candidate_hammer_v1",
        "candidate-hammer schema mismatch")
require(candidate_hammer.get("status") ==
        "PASS_M547_M533_R6_VCS_LAUNCH_ADMISSION_CANDIDATE_HAMMER",
        "candidate-hammer status mismatch")
require(candidate_hammer.get("verdict") == "PASS" and
        candidate_hammer.get("score_100") == 100,
        "candidate-hammer verdict/score mismatch")
require([candidate_hammer.get(k) for k in ("p0_count", "p1_count", "p2_count")] ==
        [0, 0, 0], "candidate-hammer P0/P1/P2 mismatch")
hammer_id = candidate_hammer.get("identity", {})
require(hammer_id.get("candidate_sha256") == candidate_sha,
        "candidate-hammer candidate binding mismatch")
require(hammer_id.get("runner_sha256") == runner_sha,
        "candidate-hammer runner binding mismatch")
require(hammer_id.get("source_static_review_sha256") == static_sha,
        "candidate-hammer source-static binding mismatch")
require(candidate_hammer.get("decision", {}).get("vcs_launch_authorized_now") is False,
        "candidate hammer must not authorize launch")

require(release.get("schema") ==
        "m547_m533_m528_dead_write_only_1rw_vcs_launch_release_v1",
        "final release schema mismatch")
require(release.get("status") == "READY_FOR_FINAL_RELEASE_HAMMER_ONLY",
        "final release status mismatch")
require(release.get("launch_now") is True, "final release launch_now must be true")
require(release.get("authorization") == expected_auth,
        "final release closed authorization mismatch")
require(release.get("resource_policy") == expected_policy,
        "final release resource policy mismatch")
release_id = release.get("identity", {})
require(release_id.get("runner_sha256") == runner_sha,
        "final release runner binding mismatch")
require(release_id.get("source_contract_sha256") == source_contract_sha,
        "final release source-contract binding mismatch")
require(release_id.get("source_static_review_sha256") == static_sha,
        "final release source-static binding mismatch")
require(release_id.get("candidate_sha256") == candidate_sha,
        "final release candidate binding mismatch")
require(release_id.get("candidate_hammer_review_sha256") == candidate_hammer_sha,
        "final release candidate-hammer binding mismatch")
require(release.get("unique_attempt", {}).get("result_path") ==
        "results/m547_m533_m528_dead_write_only_1rw_vcs_r4_20260827",
        "final release result identity mismatch")

require(final_hammer.get("schema") ==
        "m547_m533_r6_vcs_final_launch_release_hammer_v1",
        "final-release-hammer schema mismatch")
require(final_hammer.get("status") ==
        "PASS_M547_M533_R6_VCS_FINAL_LAUNCH_RELEASE_HAMMER",
        "final-release-hammer status mismatch")
require(final_hammer.get("verdict") == "PASS" and
        final_hammer.get("score_100") == 100,
        "final-release-hammer verdict/score mismatch")
require([final_hammer.get(k) for k in ("p0_count", "p1_count", "p2_count")] ==
        [0, 0, 0], "final-release-hammer P0/P1/P2 mismatch")
final_id = final_hammer.get("identity", {})
require(final_id.get("final_release_sha256") == release_sha,
        "final-release-hammer release binding mismatch")
require(final_id.get("runner_sha256") == runner_sha,
        "final-release-hammer runner binding mismatch")
require(final_id.get("candidate_sha256") == candidate_sha,
        "final-release-hammer candidate binding mismatch")
decision = final_hammer.get("decision", {})
require(decision.get("exactly_one_vcs_attempt_authorized_now") is True,
        "final-release-hammer authorization mismatch")
require(decision.get("all_other_runs_authorized") is False,
        "final-release-hammer non-VCS authorization mismatch")
PY

[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt identity already exists: ${RESULT_DIR}"

# No directory under results has been created above this point.  The temporary
# preflight directory is outside the attempt namespace and is always removed on
# any collision, resource, or release failure.
PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/m547_m533_r6_vcs_preflight.XXXXXXXX")"

scan_same_uid_collisions() {
  local output=$1
  python3 -I - "${output}" <<'PY'
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
    "schema": "m547_m533_r6_same_uid_collision_scan_v1",
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

CURRENT_PHASE="pre_mkdir_collision_initial"
scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_initial.json" \
  || fail "same-UID Synopsys/VCS/simv collision present"
CURRENT_PHASE="pre_mkdir_cgroup_resolution"
resolve_cgroup_v1
CURRENT_PHASE="pre_mkdir_resource_preflight"
resource_preflight "${PREFLIGHT_DIR}/resource_prelaunch.log"
# Close the sampling race with a second scan immediately before the atomic
# result/attempt creation.
CURRENT_PHASE="pre_mkdir_collision_final"
scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_final.json" \
  || fail "same-UID collision appeared during resource preflight"
[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt appeared during preflight"

# This mkdir is the sole attempt-consumption point.  Every release, seal,
# identity, collision, and resource failure above leaves RESULT_DIR absent.
mkdir -- "${RESULT_DIR}" || fail "atomic result/attempt creation failed"
RESULT_CREATED=1
CURRENT_PHASE="post_mkdir_evidence_copy_and_identity_precheck"
cp -- "${PREFLIGHT_DIR}/collision_initial.json" "${RESULT_DIR}/"
cp -- "${PREFLIGHT_DIR}/collision_final.json" "${RESULT_DIR}/"
cp -- "${PREFLIGHT_DIR}/resource_prelaunch.log" "${RESULT_DIR}/"
(cd -- "${RESULT_DIR}" && \
  sha256sum -c <(cd -- "${PREFLIGHT_DIR}" && sha256sum \
    collision_initial.json collision_final.json resource_prelaunch.log) >/dev/null) \
  || fail "post-mkdir copied preflight identity mismatch"
CURRENT_PHASE="post_mkdir_collision_gate"
scan_same_uid_collisions "${RESULT_DIR}/collision_postmkdir.json" \
  || fail "same-UID collision appeared after attempt consumption"

runtime_resource_sample() {
  local output=$1 phase=$2
  local sf uf su sk uu uk session_usage user_usage
  local dir file
  for dir in "${CGROUP_SESSION_DIR}" "${CGROUP_USER_DIR}"; do
    for file in memory.failcnt memory.oom_control memory.usage_in_bytes; do
      [[ -r "${dir}/${file}" && ! -L "${dir}/${file}" ]] || return 1
    done
  done
  sf="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"
  uf="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
  su="$(awk '$1 == "under_oom" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"
  sk="$(awk '$1 == "oom_kill" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"
  uu="$(awk '$1 == "under_oom" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"
  uk="$(awk '$1 == "oom_kill" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"
  session_usage="$(<"${CGROUP_SESSION_DIR}/memory.usage_in_bytes")"
  user_usage="$(<"${CGROUP_USER_DIR}/memory.usage_in_bytes")"
  [[ "${sf}" =~ ^[0-9]+$ && "${uf}" =~ ^[0-9]+$ && \
     "${su}" =~ ^[0-9]+$ && "${sk}" =~ ^[0-9]+$ && \
     "${uu}" =~ ^[0-9]+$ && "${uk}" =~ ^[0-9]+$ && \
     "${session_usage}" =~ ^[0-9]+$ && "${user_usage}" =~ ^[0-9]+$ ]] \
    || return 1
  printf 'phase=%s timestamp=%s epoch=%s session_failcnt=%s user_failcnt=%s session_under_oom=%s session_oom_kill=%s user_under_oom=%s user_oom_kill=%s session_usage_bytes=%s user_usage_bytes=%s\n' \
    "${phase}" "$(date --iso-8601=seconds)" "$(date +%s)" \
    "${sf}" "${uf}" "${su}" "${sk}" "${uu}" "${uk}" \
    "${session_usage}" "${user_usage}" >>"${output}" || return 1
  [[ "${sf}" == "${BASE_SESSION_FAILCNT}" && \
     "${uf}" == "${BASE_USER_FAILCNT}" && \
     "${su}" == 0 && "${sk}" == 0 && "${uu}" == 0 && "${uk}" == 0 ]]
}

resource_monitor() {
  local output=$1 violation=$2 heartbeat=$3 final_request=$4 final_ack=$5
  local sequence=0 heartbeat_tmp="${heartbeat}.tmp.$$" ack_tmp="${final_ack}.tmp.$$"
  : >"${output}"
  while :; do
    if ! runtime_resource_sample "${output}" "periodic"; then
      printf 'runtime_counter_missing_non_numeric_or_drifted\n' >>"${violation}"
      return 64
    fi
    sequence=$((sequence + 1))
    printf 'sequence=%s epoch=%s\n' "${sequence}" "$(date +%s)" >"${heartbeat_tmp}" \
      || return 65
    mv -- "${heartbeat_tmp}" "${heartbeat}" || return 66
    if [[ -e "${final_request}" ]]; then
      if ! runtime_resource_sample "${output}" "final_synchronous"; then
        printf 'final_counter_missing_non_numeric_or_drifted\n' >>"${violation}"
        return 67
      fi
      printf 'final_sample_ack=1 sequence=%s epoch=%s\n' \
        "${sequence}" "$(date +%s)" >"${ack_tmp}" || return 68
      mv -- "${ack_tmp}" "${final_ack}" || return 69
      return 0
    fi
    sleep 1
  done
}

require_monitor_live() {
  local phase=$1 heartbeat_epoch now_epoch
  [[ -n "${MONITOR_PID}" ]] || fail "runtime monitor PID absent at ${phase}"
  kill -0 "${MONITOR_PID}" >/dev/null 2>&1 \
    || fail "runtime monitor exited unexpectedly at ${phase}"
  [[ -f "${RESULT_DIR}/RESOURCE_HEARTBEAT" ]] \
    || fail "runtime monitor heartbeat absent at ${phase}"
  heartbeat_epoch="$(sed -nE 's/.*epoch=([0-9]+).*/\1/p' \
    "${RESULT_DIR}/RESOURCE_HEARTBEAT")"
  [[ "${heartbeat_epoch}" =~ ^[0-9]+$ ]] \
    || fail "runtime monitor heartbeat malformed at ${phase}"
  now_epoch="$(date +%s)"
  [[ $((now_epoch - heartbeat_epoch)) -le 3 ]] \
    || fail "runtime monitor heartbeat stale at ${phase}"
}

finalize_monitor_with_final_sample() {
  local wait_rc ack_seen=0 attempt
  require_monitor_live "final-request"
  : >"${RESULT_DIR}/RESOURCE_FINAL_REQUEST"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -f "${RESULT_DIR}/RESOURCE_FINAL_ACK" ]]; then
      ack_seen=1
      break
    fi
    if ! kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if [[ "${ack_seen}" -ne 1 ]]; then
    if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
      fail "runtime monitor final sample/ack timeout"
    fi
    set +e
    wait "${MONITOR_PID}"
    wait_rc=$?
    set -e
    MONITOR_PID=""
    fail "runtime monitor exited before final ack rc=${wait_rc}"
  fi
  set +e
  wait "${MONITOR_PID}"
  wait_rc=$?
  set -e
  MONITOR_PID=""
  [[ "${wait_rc}" -eq 0 ]] || fail "runtime monitor final exit rc=${wait_rc}"
  [[ ! -e "${RESULT_DIR}/RESOURCE_VIOLATION" ]] \
    || fail "runtime cgroup failcnt/OOM gate failed; attempt consumed"
  [[ "$(rg -c '^final_sample_ack=1 sequence=[0-9]+ epoch=[0-9]+$' \
      "${RESULT_DIR}/RESOURCE_FINAL_ACK")" == 1 ]] \
    || fail "runtime final ack malformed"
  [[ "$(rg -c '^phase=final_synchronous ' \
      "${RESULT_DIR}/resource_runtime.log")" == 1 ]] \
    || fail "runtime final synchronous sample missing/repeated"
  MONITOR_STATUS="final_sample_ack_pass"
}

CURRENT_PHASE="runtime_monitor_start"
resource_monitor "${RESULT_DIR}/resource_runtime.log" \
  "${RESULT_DIR}/RESOURCE_VIOLATION" "${RESULT_DIR}/RESOURCE_HEARTBEAT" \
  "${RESULT_DIR}/RESOURCE_FINAL_REQUEST" "${RESULT_DIR}/RESOURCE_FINAL_ACK" &
MONITOR_PID=$!
MONITOR_STATUS="pid_started_waiting_for_heartbeat"
for monitor_start_try in 1 2 3 4 5; do
  [[ -f "${RESULT_DIR}/RESOURCE_HEARTBEAT" ]] && break
  kill -0 "${MONITOR_PID}" >/dev/null 2>&1 \
    || fail "runtime monitor exited during startup"
  sleep 1
done
require_monitor_live "before-vcs-compile"
MONITOR_STATUS="live_before_compile"

cd -- "${RESULT_DIR}"
CURRENT_PHASE="vcs_compile"
CHILD_RC="running"
set +e
"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  -debug_access+pp +vcs+lic+wait \
  "${FOUNDRY_SLOW_V}" "${MACRO_RTL}" "${TOP_RTL}" "${SVA}" "${TB}" \
  -top tb_m528_dead_write_only_1rw_product_capture_r3 \
  -o simv 2>&1 | tee compile.log
compile_rc=${PIPESTATUS[0]}
set -e
CHILD_RC="${compile_rc}"
if [[ "${compile_rc}" -ne 0 ]]; then
  finalize_monitor_with_final_sample
  fail "VCS compile failed rc=${compile_rc}; attempt consumed"
fi
require_monitor_live "after-vcs-compile"
MONITOR_STATUS="live_after_compile"

CURRENT_PHASE="simv_run"
CHILD_RC="running"
set +e
./simv 2>&1 | tee sim.log
sim_rc=${PIPESTATUS[0]}
set -e
CHILD_RC="${sim_rc}"
finalize_monitor_with_final_sample
[[ "${sim_rc}" -eq 0 ]] || fail "simv failed rc=${sim_rc}; attempt consumed"

CURRENT_PHASE="functional_and_coverage_gate"
CHILD_RC="0"
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

CURRENT_PHASE="success_terminal_seal"
FAILURE_MESSAGE=""
write_terminal_receipt_and_seal success 0 \
  || fail "unable to double-seal successful functional result"
echo "PASS_M547_M533_M528_DW1RW_R6_ATOMIC_TERMINAL_EXACT_SHA_VCS_RUNNER"
