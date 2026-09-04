#!/usr/bin/env bash
set -euo pipefail

# M784 source-authored additive r15 wrapper for one future M533 functional VCS attempt.
# It uses the checksum-identical foundry model's documented UNIT_DELAY mode.
# This is functional VCS only; slow-corner setup/hold remains a separate
# macro-inclusive DC/PT obligation and is never inferred from this run.
# This source-only identity cannot launch until a fresh four-stage independent
# review/release chain exists and is double sealed.  No prior result identity is
# reused.  Every post-attempt exit is forced into a double-sealed terminal
# receipt; a PASS has no fallible shell work after its final verification.

if [[ $# -ne 0 ]]; then
  echo "ERROR: this exact-SHA runner accepts no overrides" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
[[ -f "${RUNNER_PATH}" && ! -L "${RUNNER_PATH}" ]] || {
  echo "ERROR: runner canonical path is not a regular file" >&2; exit 2; }

TOP_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
BINDING_PLAN="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json"
SVA="${HW_ROOT}/verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv"
TB="${HW_ROOT}/tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r7.sv"
SOURCE_CONTRACT="${HW_ROOT}/contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_r1_20260828.json"

M544_DIR="${HW_ROOT}/reviews/m544_m533_r5_vcs_compile_failure_hammer_r1_20260827"
M544_REVIEW="${M544_DIR}/review.json"
M551_DIR="${HW_ROOT}/reviews/m547_m533_r6_source_static_hammer_r1_20260827"
M551_REVIEW="${M551_DIR}/review.json"
M547_HANDOFF_DIR="${HW_ROOT}/reviews/m547_m533_r6_repair_author_handoff_r1_20260827"
M547_HANDOFF="${M547_HANDOFF_DIR}/handoff.json"
M558_DIR="${HW_ROOT}/reviews/m554_m533_r7_source_static_hammer_r1_20260828"
M558_REVIEW="${M558_DIR}/review.json"
OLD_RESULT_DIR="${HW_ROOT}/results/m533_m528_dead_write_only_1rw_vcs_r3_20260827"

# Hard prerequisites for the one new r10 identity: the consumed r8 terminal
# failure and its fresh M717 independent hammer must remain byte-exact and
# double sealed. r10 never mutates, resumes or reuses the r8 result.
R8_FAILED_RESULT_DIR="${HW_ROOT}/results/m560_m533_m528_dead_write_only_1rw_vcs_r6_20260828"
R8_FAILED_RECEIPT="${R8_FAILED_RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"
M717_DIR="${HW_ROOT}/reviews/m717_m560_m533_r8_monitor_start_failure_fresh_hammer_r1_20260828"
M717_REVIEW="${M717_DIR}/review.json"
R9_FAILED_RESULT_DIR="${HW_ROOT}/results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828"
R9_FAILED_MARKER="${R9_FAILED_RESULT_DIR}/FAILED_DO_NOT_CITE"
M726_DIR="${HW_ROOT}/reviews/m726_m719_r9_vcs_compile_failure_fresh_hammer_r1_20260828"
M726_REVIEW="${M726_DIR}/review.json"

# The consumed r10 timing-model failure and its independent M736 hammer are
# mandatory prerequisites.  r14 never edits, resumes, or relabels r10.
R10_FAILED_RESULT_DIR="${HW_ROOT}/results/m729_m533_m528_dead_write_only_1rw_vcs_r10_20260828"
R10_FAILED_RECEIPT="${R10_FAILED_RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"
M736_DIR="${HW_ROOT}/reviews/m736_m729_m533_r10_vcs_timing_failure_fresh_hammer_r1_20260828"
M736_REVIEW="${M736_DIR}/review.json"

# The consumed M737/r11 functional failure and the complete causal TB-repair
# review chain are hard prerequisites.  r14 never reinterprets or relabels the
# old result: it remains FAILED_DO_NOT_CITE and C1 remains unverified there.
R11_FAILED_RESULT_DIR="${HW_ROOT}/results/m737_m533_m528_dead_write_only_1rw_unit_delay_vcs_r11_20260828"
R11_FAILED_RECEIPT="${R11_FAILED_RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"
M738_DIR="${HW_ROOT}/reviews/m738_m737_r11_unit_delay_vcs_raw_failure_fresh_hammer_r1_20260828"
M738_REVIEW="${M738_DIR}/review.json"
M741_DIR="${HW_ROOT}/reviews/m741_m533_tb_r5_raw_monitor_static_hammer_r1_20260828"
M741_REVIEW="${M741_DIR}/review.json"
M743_DIR="${HW_ROOT}/reviews/m743_m533_tb_r6_raw_monitor_fresh_static_hammer_r1_20260828"
M743_REVIEW="${M743_DIR}/review.json"
M744_DIR="${HW_ROOT}/reviews/m744_m533_tb_r7_raw_monitor_fresh_static_hammer_r1_20260828"
M744_REVIEW="${M744_DIR}/review.json"

# The consumed r12 pre-mkdir identity failure is not an HDL attempt.  M757 is
# nevertheless a mandatory, byte-exact causal prerequisite for this additive
# r13 identity: it admitted exactly one repair (the missing b in the M743
# manifest SHA literal).  That frozen lineage remains mandatory while M770
# separately authorizes only the additive r14 environment repair.
M757_DIR="${HW_ROOT}/reviews/m757_m533_r12_premkdir_sha_literal_failure_fresh_hammer_r1_20260828"
M757_REVIEW="${M757_DIR}/review.json"

# The consumed M758/r13 environment failure and M770 fresh causal audit are
# immutable prerequisites.  r15 never resumes, relabels, or cites r13.
R13_FAILED_RESULT_DIR="${HW_ROOT}/results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828"
R13_FAILED_RECEIPT="${R13_FAILED_RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"
M770_DIR="${HW_ROOT}/reviews/m770_m533_r13_vcs_home_failure_fresh_hammer_r1_20260828"
M770_REVIEW="${M770_DIR}/review.json"
# The withdrawn M772/r14 release and M779 launch verdict are historical only.
# M782 is the immutable additive authorization for the one-key r15 repair.
M782_DIR="${HW_ROOT}/reviews/m782_m533_r14_premkdir_launch_boundary_failure_hammer_r1_20260828"
M782_REVIEW="${M782_DIR}/review.json"
AUTHOR_ENV_PREFLIGHT_DIR="${HW_ROOT}/reviews/m772_m533_r14_vcs_environment_preflight_r1_20260828"
AUTHOR_ENV_PREFLIGHT="${AUTHOR_ENV_PREFLIGHT_DIR}/preflight.json"

# All members below are fixed paths.  Only the candidate exists in this
# source-only package.  Every remaining review/release member must be authored
# by fresh agents without changing this runner.
SOURCE_STATIC_DIR="${HW_ROOT}/reviews/m784_m533_r15_unit_delay_source_static_hammer_r1_20260828"
SOURCE_STATIC_REVIEW="${SOURCE_STATIC_DIR}/review.json"
RELEASE_CANDIDATE="${HW_ROOT}/contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_r1_20260828.json"
CANDIDATE_HAMMER_DIR="${HW_ROOT}/reviews/m784_m533_r15_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"
CANDIDATE_HAMMER_REVIEW="${CANDIDATE_HAMMER_DIR}/review.json"
LAUNCH_RELEASE="${HW_ROOT}/contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
FINAL_HAMMER_DIR="${HW_ROOT}/reviews/m784_m533_r15_unit_delay_vcs_final_launch_release_hammer_r1_20260828"
FINAL_HAMMER_REVIEW="${FINAL_HAMMER_DIR}/review.json"

ASSET_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821"
ASSET_MANIFEST="${ASSET_ROOT}/SHA256SUMS"
FOUNDRY_SLOW_V="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
FOUNDRY_SLOW_DB="${ASSET_ROOT}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
VCS_MSG_REPORT="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcsMsgReport"
LICENSE_FILE="/opt/synopsys/Synopsys.dat"
LMUTIL_BIN="/opt/synopsys/scl/2025.03/linux64/bin/lmutil"
EXPECTED_VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
EXPECTED_VCS_ARCH_OVERRIDE="linux"
EXPECTED_SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
EXPECTED_LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"

RESULT_DIR="${HW_ROOT}/results/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_r15_20260828"
PREFLIGHT_DIR=""
MONITOR_PID=""
RESULT_CREATED=0
TERMINAL_SEALED=0
CURRENT_PHASE="pre_mkdir_identity_gate"
CHILD_RC="not_started"
MONITOR_STATUS="not_started"
FAILURE_MESSAGE=""
PREFLIGHT_CLEANUP_RC="not_attempted"

RUNNER_SHA=""
SOURCE_CONTRACT_SHA=""
SOURCE_STATIC_SHA=""
RELEASE_CANDIDATE_SHA=""
CANDIDATE_HAMMER_SHA=""
LAUNCH_RELEASE_SHA=""
FINAL_HAMMER_SHA=""

fail() {
  FAILURE_MESSAGE="$*"
  echo "ERROR phase=${CURRENT_PHASE}: $*" >&2
  exit 1
}

strict_json_parse() {
  local path=$1
  python3 -I - "${path}" <<'PY'
import json, math, sys
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
            finite(key); finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)
with open(sys.argv[1], encoding="utf-8") as handle:
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
  local path=$1 dir base
  dir="$(dirname -- "${path}")"; base="$(basename -- "${path}")"
  [[ -f "${path}" && ! -L "${path}" && -f "${path}.sha256" && ! -L "${path}.sha256" && \
     -f "${path}.sha256.seal.sha256" && ! -L "${path}.sha256.seal.sha256" ]] \
    || fail "missing/non-regular JSON or seal: ${path}"
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256" >/dev/null)
  (cd -- "${dir}" && sha256sum -c -- "${base}.sha256.seal.sha256" >/dev/null)
  strict_json_parse "${path}"
}

verify_review_double_seal() {
  local dir=$1
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/review.json" && ! -L "${dir}/review.json" && \
     -f "${dir}/SHA256SUMS" && ! -L "${dir}/SHA256SUMS" && \
     -f "${dir}/SHA256SUMS.seal.sha256" && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular review package: ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${dir}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  strict_json_parse "${dir}/review.json"
}

verify_handoff_double_seal() {
  local dir=$1
  [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/handoff.json" && ! -L "${dir}/handoff.json" && \
     -f "${dir}/SHA256SUMS" && ! -L "${dir}/SHA256SUMS" && \
     -f "${dir}/SHA256SUMS.seal.sha256" && ! -L "${dir}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular handoff package: ${dir}"
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${dir}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  strict_json_parse "${dir}/handoff.json"
}

verify_r8_failure_and_m717_prerequisite() {
  [[ -d "${R8_FAILED_RESULT_DIR}" && ! -L "${R8_FAILED_RESULT_DIR}" && \
     -f "${R8_FAILED_RECEIPT}" && ! -L "${R8_FAILED_RECEIPT}" && \
     -f "${R8_FAILED_RESULT_DIR}/SHA256SUMS" && ! -L "${R8_FAILED_RESULT_DIR}/SHA256SUMS" && \
     -f "${R8_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" && ! -L "${R8_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular consumed r8 failure package"
  (cd -- "${R8_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${R8_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  require_regular_sha d390ae62512d33e3f959de4dd1cf00546fde2253f9d8a52b0b6de5470568f393 "${R8_FAILED_RECEIPT}"
  require_regular_sha 6061f952794dd8e30b734e123566a2b58aa6fd017f86821af6ca114c505e1d91 "${R8_FAILED_RESULT_DIR}/SHA256SUMS"
  require_regular_sha 5a3f607edf6d0021b4e45ef8eb941465dd45ffe4b145549465b1888ed472eb4b "${R8_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256"
  strict_json_parse "${R8_FAILED_RECEIPT}"
  verify_review_double_seal "${M717_DIR}"
  require_regular_sha ec0f06f3b1f4b112812a6bc101d52ba0078b4f83759928ec715971c7e2ea2bfc "${M717_REVIEW}"
  require_regular_sha 3f6b789d16aab8fc7b866f090619e0c7ec073b42bce42ff15010a0374c09a74b "${M717_DIR}/SHA256SUMS"
  require_regular_sha ef1d93f2faef2f67313620b9d3215b550526ccf8fbdc84bead4d7a131fd27d8e "${M717_DIR}/SHA256SUMS.seal.sha256"
  python3 -I - "${R8_FAILED_RECEIPT}" "${M717_REVIEW}" <<'PY2'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: failure = json.load(h)
with open(sys.argv[2], encoding="utf-8") as h: review = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(failure.get("status") == "FAILED_DO_NOT_CITE", "r8 failure status")
req(failure.get("phase") == "runtime_monitor_start", "r8 failure phase")
req(failure.get("child_rc") == "not_started", "r8 child status")
req(failure.get("runner_exit_rc") == 1, "r8 runner rc")
req(failure.get("paper_citable") is False, "r8 paper boundary")
req(review.get("status") == "ADMIT_FAILURE_RECEIPT__R8_PERMANENTLY_CONSUMED__FUNCTIONAL_NO_CONCLUSION", "M717 status")
req(review.get("decision", {}).get("r8") == "PERMANENTLY_CONSUMED", "M717 consumed")
req(review.get("decision", {}).get("functional_vcs") == "NO_CONCLUSION", "M717 functional boundary")
req(review.get("minimal_r9_fix", {}).get("new_unique_identity_allowed") is True, "M717 r9 permission")
req(review.get("minimal_r9_fix", {}).get("maximum_new_identities_now") == 1, "M717 unique r9")
PY2
}

verify_r9_failure_and_m726_prerequisite() {
  [[ -d "${R9_FAILED_RESULT_DIR}" && ! -L "${R9_FAILED_RESULT_DIR}" &&
     -f "${R9_FAILED_MARKER}" && ! -L "${R9_FAILED_MARKER}" &&
     -f "${R9_FAILED_RESULT_DIR}/SHA256SUMS" && ! -L "${R9_FAILED_RESULT_DIR}/SHA256SUMS" &&
     -f "${R9_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" && ! -L "${R9_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular consumed r9 failure package"
  (cd -- "${R9_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${R9_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  require_regular_sha 24c1b727ed8e3cce11dd0c44ec4966f02c118a546be6967b58de8f51b30c025d "${R9_FAILED_MARKER}"
  require_regular_sha 0953c7ddbeddc4b3b957eaab149d8e3035528e4a1d804fd6c0933b9a20250157 "${R9_FAILED_RESULT_DIR}/SHA256SUMS"
  require_regular_sha b02f6be99240734640d3b59e070a0de2e1768f72030b16db937f7da2e3173af8 "${R9_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256"
  verify_review_double_seal "${M726_DIR}"
  require_regular_sha 729128c0fa9c917f8ce7f357b2e6b302cc13293480010ebe6fab2a31a2392564 "${M726_REVIEW}"
  require_regular_sha 00d907677a134967d42919be436511ffc9d8d82c065a6bdca44ed9078accba81 "${M726_DIR}/SHA256SUMS"
  require_regular_sha 2afe191812ac9ce62a904fa98e5c8594c2130b671d9c48dd16b0d5cd2e7596a4 "${M726_DIR}/SHA256SUMS.seal.sha256"
  python3 -I - "${R9_FAILED_MARKER}" "${M726_REVIEW}" <<'PY2'
import json, sys
marker = open(sys.argv[1], encoding="utf-8").read().strip()
with open(sys.argv[2], encoding="utf-8") as h: review = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(marker == "FAILED_DO_NOT_CITE phase=vcs_compile runner_rc=1 child_rc=vcs_255_tee_0 monitor_status=final_sample_ack_pass", "r9 marker")
req(review.get("status") == "ADMIT_M719_R9_CONSUMED_FAILURE__FUNCTIONAL_NO_CONCLUSION__R10_MINIMAL_REPAIR_ALLOWED", "M726 status")
req(review.get("compile_failure", {}).get("functional_vcs") is False, "M726 function boundary")
req(review.get("compile_failure", {}).get("error_ids") == ["DTINPCIL", "IRFPCA-AUTOVAR"], "M726 root cause")
b = review.get("minimum_r10_boundary", {})
req(b.get("maximum_new_identities") == 1 and b.get("new_unique_result_path_required") is True, "M726 r10 identity")
req(b.get("functional_top_r2_change_allowed") is False and b.get("sva_r2_change_allowed") is False and b.get("macro_adapter_or_binding_change_allowed") is False, "M726 frozen function")
req(b.get("runner_receipt_writer_must_not_mask_binding_failure") is True, "M726 receipt boundary")
PY2
}

verify_r10_failure_and_m736_prerequisite() {
  [[ -d "${R10_FAILED_RESULT_DIR}" && ! -L "${R10_FAILED_RESULT_DIR}" &&
     -f "${R10_FAILED_RECEIPT}" && ! -L "${R10_FAILED_RECEIPT}" &&
     -f "${R10_FAILED_RESULT_DIR}/SHA256SUMS" && ! -L "${R10_FAILED_RESULT_DIR}/SHA256SUMS" &&
     -f "${R10_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" && ! -L "${R10_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular consumed r10 failure package"
  (cd -- "${R10_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${R10_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  require_regular_sha 6b26caeb01df98f194a7a503ae43857c662b95a869646dbde9281697f5652898 "${R10_FAILED_RECEIPT}"
  require_regular_sha a3f3bac4b7b63c351c44eb97a573ec6de0c21dee8f26f33f15ea96038b69f423 "${R10_FAILED_RESULT_DIR}/SHA256SUMS"
  require_regular_sha 32cb998d2dd2efb54e3d5b1a2465b2439592eb6b34d600d82340725002a87e32 "${R10_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256"
  strict_json_parse "${R10_FAILED_RECEIPT}"
  verify_review_double_seal "${M736_DIR}"
  require_regular_sha 8c62408c32531af24e6f894729148c405ee4093f39a6c20147278f67466c4cf0 "${M736_REVIEW}"
  require_regular_sha 0878bb055bc502c6842b6c37dd94acfc84d48088334d80c622aa9833ef733f03 "${M736_DIR}/SHA256SUMS"
  require_regular_sha d8c780106820f6acb5cc13a06d823d3c7fb281dc58204edaf55a001f9e06ba67 "${M736_DIR}/SHA256SUMS.seal.sha256"
  python3 -I - "${R10_FAILED_RECEIPT}" "${M736_REVIEW}" <<'PY2'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: failure = json.load(h)
with open(sys.argv[2], encoding="utf-8") as h: review = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(failure.get("status") == "FAILED_DO_NOT_CITE", "r10 failure status")
req(failure.get("phase") == "functional_and_coverage_gate", "r10 failure phase")
req(failure.get("paper_citable") is False, "r10 paper boundary")
req(review.get("schema") == "m736_m729_m533_r10_vcs_timing_failure_fresh_hammer_v1", "M736 schema")
req(review.get("verdict") == "PASS" and review.get("score_100") == 98, "M736 verdict")
req(review.get("observations", {}).get("timing_violation_total") == 2223, "M736 timing count")
req(review.get("classification", {}).get("functional_vcs_status") == "NO_CONCLUSION", "M736 function boundary")
d = review.get("decision", {})
req(d.get("one_repair_candidate_identity_authorized") is True, "M736 candidate authorization")
req(d.get("vcs_launch_authorized_now") is False, "M736 launch boundary")
req(d.get("authorized_candidate_mode") == "checksum_verified_foundry_v_with_documented_UNIT_DELAY_functional_mode", "M736 mode")
cb = d.get("candidate_claim_boundary", {})
req(cb.get("functional_vcs_only") is True and cb.get("timing_verified") is False, "M736 claim boundary")
PY2
}

verify_r11_failure_and_tb_repair_prerequisites() {
  [[ -d "${R11_FAILED_RESULT_DIR}" && ! -L "${R11_FAILED_RESULT_DIR}" &&
     -f "${R11_FAILED_RECEIPT}" && ! -L "${R11_FAILED_RECEIPT}" &&
     -f "${R11_FAILED_RESULT_DIR}/SHA256SUMS" && ! -L "${R11_FAILED_RESULT_DIR}/SHA256SUMS" &&
     -f "${R11_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" && ! -L "${R11_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256" ]] \
    || fail "missing/non-regular consumed r11 failure package"
  (cd -- "${R11_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${R11_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  require_regular_sha 62d77d2ef19fac3520a328d23deceab2228937752ab02b9fd0813b7c07d7db8b "${R11_FAILED_RECEIPT}"
  require_regular_sha e3381329028dd44f5786a50409b78e7a0ec6dc8ed351ead17d456e07f92399f3 "${R11_FAILED_RESULT_DIR}/SHA256SUMS"
  require_regular_sha f5a9aadc229ca6c480972cc1edb679c835c85cd4259e7c58dbd1517af5247ac8 "${R11_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256"
  strict_json_parse "${R11_FAILED_RECEIPT}"

  verify_review_double_seal "${M738_DIR}"
  verify_review_double_seal "${M741_DIR}"
  verify_review_double_seal "${M743_DIR}"
  verify_review_double_seal "${M744_DIR}"
  require_regular_sha b2ec522258f73749145c4c9c5e62bbc1ae08b9ab902596cafc943de299c9950a "${M738_REVIEW}"
  require_regular_sha d87b1e07a5bcdca0055b3828bd8ad02055d6608f3993bca2a042c31a0964d673 "${M738_DIR}/SHA256SUMS"
  require_regular_sha 8baadc410dff82d156214d015fe7f153e13a082c0d0645637ab6220b18842dc3 "${M738_DIR}/SHA256SUMS.seal.sha256"
  require_regular_sha e8b123040e75b33ded1039c9cb2fcf9abfb9e2e6630cd9527a85997e6815aaf4 "${M741_REVIEW}"
  require_regular_sha 9f29b03f43173c4129eb18323803ed7f8a157ad18144b5f9ad8d665857ad6110 "${M741_DIR}/SHA256SUMS"
  require_regular_sha a191ffc7c1fc26fee0851bf58120500b421b2fe164d12c2e46e10df2a6f6ac53 "${M741_DIR}/SHA256SUMS.seal.sha256"
  require_regular_sha 804d6a368a6733aa358df59f6c7e999f05e928dffe0389659401e62eae55a5bd "${M743_REVIEW}"
  require_regular_sha 626ba66587e86885020031ef5656c3cd971cdacb803bc339b218d1171d796962 "${M743_DIR}/SHA256SUMS"
  require_regular_sha 65d6b8375e46c863e7e1b34d251e173fa506ed8f2bf65fb228e93b78c943a90a "${M743_DIR}/SHA256SUMS.seal.sha256"
  require_regular_sha 6a036e7cf6c0e840018b3f0c148bc68aeeb39034af789a9c447b31e74014ae3e "${M744_REVIEW}"
  require_regular_sha 814ce729586b5c578b90415c5a95d4ee4aa283f348dd45b8113f8220af0169c1 "${M744_DIR}/SHA256SUMS"
  require_regular_sha 174c0ff28bbd7bfee735a7fd6ad7d180d5795573562522e4b3ac4acba3f22f15 "${M744_DIR}/SHA256SUMS.seal.sha256"

  python3 -I - "${R11_FAILED_RECEIPT}" "${M738_REVIEW}" "${M741_REVIEW}" \
    "${M743_REVIEW}" "${M744_REVIEW}" <<'PY2'
import json, sys
docs = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        docs.append(json.load(handle))
failure, m738, m741, m743, m744 = docs
def req(cond, message):
    if not cond: raise RuntimeError(message)
req(failure.get("status") == "FAILED_DO_NOT_CITE", "r11 failure status")
req(failure.get("phase") == "functional_and_coverage_gate", "r11 failure phase")
req(failure.get("failure_message") == "functional token", "r11 failure token gate")
req(failure.get("paper_citable") is False, "r11 paper boundary")
req(m738.get("status") == "PASS_M738_FAILURE_CLASSIFICATION__R11_FAILED_DO_NOT_CITE", "M738 status")
req(m738.get("verdict") == "PASS" and m738.get("score_100") == 98, "M738 score")
req(m738.get("classification", {}).get("functional_status") == "NO_PASS_DUE_TO_TB_ORACLE_FALSE_POSITIVE", "M738 classification")
req(m738.get("classification", {}).get("c1_rtl_verified") is False, "M738 C1 boundary")
req(m738.get("decision", {}).get("r11_status") == "FAILED_DO_NOT_CITE", "M738 old result frozen")
req(m738.get("decision", {}).get("one_new_candidate_identity_allowed") is True, "M738 candidate")
req(m738.get("decision", {}).get("vcs_launch_authorized_now") is False, "M738 no launch")
req(m741.get("verdict") == "FAIL" and m741.get("score_100") == 94, "M741 failure review")
req(m741.get("decision", {}).get("r12_runner_candidate_authoring_allowed_from_current_r5") is False, "M741 no premature r12")
req(m743.get("verdict") == "FAIL_ONE_TB_ONLY_CAUSAL_PRIORITY_REPAIR_REQUIRED" and m743.get("score") == 96, "M743 failure review")
req(m743.get("authorization", {}).get("r12_runner_candidate_from_r6") is False, "M743 no premature r12")
req(m744.get("verdict") == "PASS_ALLOW_ONE_R12_RUNNER_CANDIDATE__NO_LAUNCH_NOW", "M744 verdict")
req(m744.get("score") == 100 and [m744.get(k) for k in ("p0","p1","p2")] == [0,0,0], "M744 score")
req(m744.get("frozen_sources", {}).get("tb_r7", {}).get("sha256") == "d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2", "M744 TB r7")
auth = m744.get("authorization", {})
req(auth.get("one_r12_runner_candidate_authoring") is True, "M744 candidate authorization")
req(auth.get("vcs_or_simv_launch_now") is False and auth.get("eda_launch_now") is False, "M744 no launch")
PY2
}

verify_m757_r12_premkdir_failure_prerequisite() {
  verify_review_double_seal "${M757_DIR}"
  require_regular_sha 8b84530b0666b2b52617d25e8cf2e5fd2f0f2fe45b0c5242f45528d38803f991 "${M757_REVIEW}"
  require_regular_sha 0f717acbe3d392a7d9fdb3c9f5115c836226263449650159b35a139beb8718f8 "${M757_DIR}/SHA256SUMS"
  require_regular_sha 34d91ce36c0ce6a533afd6576c7dd6ee7196a91eeb28b83809239864eaf9f5ed "${M757_DIR}/SHA256SUMS.seal.sha256"
  python3 -I - "${M757_REVIEW}" <<'PY2'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    review = json.load(handle)
def req(cond, message):
    if not cond: raise RuntimeError(message)
req(review.get("status") == "PASS_FAILURE_AUDIT__M746_R12_BLOCKED_PRE_MKDIR__M743_MANIFEST_SHA_LITERAL_MISSING_ONE_NIBBLE__ADDITIVE_R13_ONLY", "M757 status")
req(review.get("verdict") == "PASS" and review.get("score_out_of_100") == 100, "M757 score")
finding = review.get("finding", {})
req(finding.get("expected_literal_length") == 63 and finding.get("actual_sha256_length") == 64, "M757 literal lengths")
req(finding.get("actual_m743_manifest_sha256") == "626ba66587e86885020031ef5656c3cd971cdacb803bc339b218d1171d796962", "M757 corrected M743 SHA")
req(finding.get("single_missing_character") == "b" and finding.get("missing_character_position_1_based") == 40, "M757 one-nibble repair")
missed = review.get("missed_static_edge_analysis", {})
req(missed.get("m749_and_m753_other_checks_revoked") is False, "M757 preserves unrelated hammer checks")
req("M749" in missed.get("m749_specific_gap", "") and "M743 SHA256SUMS" in missed.get("m749_specific_gap", ""), "M757 M749 cross-edge gap")
req("final-release hammer" in missed.get("m753_specific_gap", ""), "M757 M753 cross-edge gap")
successor = review.get("required_additive_successor", {})
req(successor.get("edit_r12_in_place") is False, "M757 additive identity")
req(successor.get("new_static_rule_required", "").startswith("Before release, enumerate all require_regular_sha calls"), "M757 complete cross-edge rule")
auth = review.get("authorization", {})
req(auth.get("author_additive_r13_source_package") is True, "M757 r13 source authorization")
req(auth.get("run_r13_now") is False and auth.get("run_vcs") is False and auth.get("run_simv") is False, "M757 no launch")
PY2
}

verify_r13_failure_m770_m782_and_author_preflight_prerequisites() {
  [[ -d "${R13_FAILED_RESULT_DIR}" && ! -L "${R13_FAILED_RESULT_DIR}" ]] ||
    fail "missing/non-regular consumed r13 result"
  (cd -- "${R13_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${R13_FAILED_RESULT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  require_regular_sha df9e70c0f382139dc5c35b95cbee9e7aa7af9e9466d81cb9c8d563660fbe243b "${R13_FAILED_RECEIPT}"
  require_regular_sha bb35c3732e6970e1e0d7a79c26b2999002bdb9fe9e38a2a4667ac8967e1d7a06 "${R13_FAILED_RESULT_DIR}/SHA256SUMS"
  require_regular_sha c78ec408a5490b9624618da0a7bd343180959aefaef010ca7de03e0d90c76b6a "${R13_FAILED_RESULT_DIR}/SHA256SUMS.seal.sha256"
  strict_json_parse "${R13_FAILED_RECEIPT}"

  verify_review_double_seal "${M770_DIR}"
  require_regular_sha caba813792a8df3b1b9b72a7ddb7ec053096acab6188645b9d3c59a2ca8c3192 "${M770_REVIEW}"
  require_regular_sha b3dc9493170604ff3cbea305e55abbbbefc5b645a8355840ecc7f821f05f4895 "${M770_DIR}/SHA256SUMS"
  require_regular_sha 1c5c89225131e7678ec21d1ddcd72fa5bcfa30a468e863e129494a845f4c6a40 "${M770_DIR}/SHA256SUMS.seal.sha256"

  verify_review_double_seal "${M782_DIR}"
  require_regular_sha ff7498279990537c7e60f886d44a3a6ec919aeb39d2fe5a9294a049f9a79bf6b "${M782_REVIEW}"
  require_regular_sha 3cf622455ea68a5df7fe511ebc7897c2e78f68488d4696ffc23b4ade685d448b "${M782_DIR}/SHA256SUMS"
  require_regular_sha e6dbb6250e913a56b58741374f1b8ac1ce5b20e0653713635c47f91fd2d5d740 "${M782_DIR}/SHA256SUMS.seal.sha256"

  [[ -d "${AUTHOR_ENV_PREFLIGHT_DIR}" && ! -L "${AUTHOR_ENV_PREFLIGHT_DIR}" &&
     -f "${AUTHOR_ENV_PREFLIGHT}" && ! -L "${AUTHOR_ENV_PREFLIGHT}" &&
     -f "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS" && ! -L "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS" &&
     -f "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS.seal.sha256" &&
     ! -L "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS.seal.sha256" ]] ||
    fail "missing/non-regular author environment preflight package"
  (cd -- "${AUTHOR_ENV_PREFLIGHT_DIR}" && sha256sum -c SHA256SUMS >/dev/null)
  (cd -- "${AUTHOR_ENV_PREFLIGHT_DIR}" && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
  strict_json_parse "${AUTHOR_ENV_PREFLIGHT}"
  require_regular_sha dd7500d8d5deaa8bc4d0d02113218c6d6b92bdfe27dca1b8d0c3724b125b2c9f "${AUTHOR_ENV_PREFLIGHT}"
  require_regular_sha 473bf4553070c2b6914bb44ce5bfae388dee8de0b855f61d71be054506d51546 "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS"
  require_regular_sha 159f9b779c5edf15d2a9b5434545300947e64c1bfd2227ef8da96a7b2185dd10 "${AUTHOR_ENV_PREFLIGHT_DIR}/SHA256SUMS.seal.sha256"

  python3 -I - "${R13_FAILED_RECEIPT}" "${M770_REVIEW}" "${M782_REVIEW}" "${AUTHOR_ENV_PREFLIGHT}" <<'PY2'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: failure = json.load(h)
with open(sys.argv[2], encoding="utf-8") as h: audit = json.load(h)
with open(sys.argv[3], encoding="utf-8") as h: m782 = json.load(h)
with open(sys.argv[4], encoding="utf-8") as h: preflight = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(failure.get("status") == "FAILED_DO_NOT_CITE", "r13 failure status")
req(failure.get("phase") == "vcs_compile", "r13 failure phase")
req(failure.get("child_rc") == "vcs_1_tee_0", "r13 child rc")
req(failure.get("paper_citable") is False, "r13 paper boundary")
req(audit.get("verdict") == "PASS" and audit.get("score_out_of_100") == 100, "M770 verdict")
req(audit.get("decision", {}).get("one_additive_r14_source_package_authorized") is True, "M770 r14 source authorization")
req(audit.get("decision", {}).get("r14_launch_authorized_now") is False, "M770 launch boundary")
req(m782.get("verdict") == "PASS_FAILURE_AUDIT" and m782.get("score_out_of_100") == 100, "M782 verdict")
req(m782.get("decision", {}).get("one_additive_r15_source_package_authorized") is True, "M782 r15 source authorization")
req(m782.get("decision", {}).get("r15_launch_authorized_now") is False, "M782 launch boundary")
req(m782.get("r14_release_disposition", {}).get("release_status_after_audit") == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "r14 release withdrawn")
req(preflight.get("status") == "PASS_READ_ONLY_FULL64_ID_AND_LICENSE_STATUS__NO_HOME__NO_COMPILE__NO_SEAT_CHECKOUT", "author preflight status")
req(preflight.get("environment", {}).get("HOME") == "UNSET_BY_ENV_I", "HOME boundary")
req(preflight.get("probes", {}).get("vcs_full64_id", {}).get("pass") is True, "full64 ID")
req(preflight.get("probes", {}).get("compiler_license", {}).get("free", 0) > 0, "compiler free seat")
req(preflight.get("probes", {}).get("runtime_license", {}).get("free", 0) > 0, "runtime free seat")
b = preflight.get("boundary", {})
req(b.get("hdl_compile") is False and b.get("simv") is False, "preflight no compile")
req(b.get("result_directory_created") is False and b.get("r14_attempt_consumed") is False, "preflight no attempt")
PY2
}

vcs_license_preflight() {
  local id_out=$1 compiler_out=$2 runtime_out=$3
  local -a clean_env=(
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C
    VCS_HOME="${EXPECTED_VCS_HOME}"
    VCS_ARCH_OVERRIDE="${EXPECTED_VCS_ARCH_OVERRIDE}"
    SNPSLMD_LICENSE_FILE="${EXPECTED_SNPSLMD_LICENSE_FILE}"
    LM_LICENSE_FILE="${EXPECTED_LM_LICENSE_FILE}"
  )
  "${clean_env[@]}" "${VCS_BIN}" -full64 -ID >"${id_out}" 2>&1 ||
    fail "VCS full64 identity probe failed"
  [[ "$(rg -c '^Compiler version = VCS V-2023\.12-SP1_Full64$' "${id_out}")" == 1 ]] ||
    fail "VCS full64 identity mismatch"
  ! rg -n 'Error-|ERROR|Fatal|Cannot find' "${id_out}" >/dev/null ||
    fail "VCS identity probe diagnostic"

  "${clean_env[@]}" "${LMUTIL_BIN}" lmstat -c "${EXPECTED_SNPSLMD_LICENSE_FILE}"     -f VCSCompiler_Net >"${compiler_out}" 2>&1 || fail "VCSCompiler_Net query failed"
  "${clean_env[@]}" "${LMUTIL_BIN}" lmstat -c "${EXPECTED_SNPSLMD_LICENSE_FILE}"     -f VCSRuntime_Net >"${runtime_out}" 2>&1 || fail "VCSRuntime_Net query failed"

  python3 -I - "${compiler_out}" "${runtime_out}" <<'PY2'
import re, sys
for path, feature in zip(sys.argv[1:], ("VCSCompiler_Net", "VCSRuntime_Net")):
    text = open(path, encoding="utf-8").read()
    if "license server UP (MASTER)" not in text or "snpslmd: UP" not in text:
        raise RuntimeError(feature + " server/daemon unavailable")
    pattern = rf"Users of {feature}:  \(Total of ([0-9]+) licenses issued;  Total of ([0-9]+) licenses in use\)"
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise RuntimeError(feature + " ambiguous usage line")
    total, used = map(int, found[0])
    if total <= used:
        raise RuntimeError(feature + " has no free seat")
PY2
}

verify_old_partial_closed_inventory() {
  python3 -I - "${OLD_RESULT_DIR}" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {
    "RESOURCE_FINAL_ACK": "36a29fd702bf1726f3bdb52580cb0e0c6cbfdacfa3334a2b01436014acfbc5e8",
    "RESOURCE_FINAL_REQUEST": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "RESOURCE_HEARTBEAT": "fca5512812e8468f9285b1bded77590a5e6423900551b35d1542e29874f1cdde",
    "collision_final.json": "60c8a58122f9abf4a7010382e72b81892564f3f6e4bb718e0ea8e03d7366cf1b",
    "collision_initial.json": "c2fcc19eb1b83300fafbecd7a95088d3e1a5e09231f984a4e0248250aa9c3804",
    "compile.log": "0b0a0fcd88e5dff53e82c4ccfe218aa361c4fcf696aeed1e397d93fffe8b50ef",
    "resource_prelaunch.log": "7a76b23d7a1347a6a99f0a71c13b10089a5429faf00f9204816676050e7f57a6",
    "resource_runtime.log": "f4b32c5c016cdfaa0a0403cec575bcf8b2dd81afc28daf395c080219195695f2",
}
if not root.is_dir() or root.is_symlink():
    raise RuntimeError("old consumed result missing or symlinked")
entries = list(os.scandir(root))
names = {entry.name for entry in entries}
if names != set(expected):
    raise RuntimeError(f"old partial closed inventory mismatch: {sorted(names)}")
for entry in entries:
    mode = entry.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode) or entry.is_symlink():
        raise RuntimeError("old partial member is not a plain regular file: " + entry.name)
    digest = hashlib.sha256(Path(entry.path).read_bytes()).hexdigest()
    if digest != expected[entry.name]:
        raise RuntimeError("old partial SHA drift: " + entry.name)
PY
}

cleanup_preflight_preserve_rc() {
  local cleanup_rc=0
  if [[ -n "${PREFLIGHT_DIR}" && -e "${PREFLIGHT_DIR}" ]]; then
    if rm -rf -- "${PREFLIGHT_DIR}"; then
      cleanup_rc=0
    else
      cleanup_rc=$?
    fi
  fi
  PREFLIGHT_CLEANUP_RC="${cleanup_rc}"
  if [[ "${cleanup_rc}" -eq 0 ]]; then
    PREFLIGHT_DIR=""
  fi
  return "${cleanup_rc}"
}

build_artifact_inventory() {
  local terminal_kind=$1
  TERMINAL_KIND="${terminal_kind}" RESULT_DIR_ENV="${RESULT_DIR}" python3 -I - <<'PY'
import hashlib, json, os, stat
from pathlib import Path
root = Path(os.environ["RESULT_DIR_ENV"])
kind = os.environ["TERMINAL_KIND"]
excluded = {"ARTIFACT_INVENTORY.json", "RUN_COMPLETE.json", "RUN_COMPLETE.txt",
            "RUN_FAILED_OR_INCOMPLETE.json", "FAILED_DO_NOT_CITE",
            "SHA256SUMS", "SHA256SUMS.seal.sha256"}
items = []
strict_errors = []
root_resolved = root.resolve(strict=True)
for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
    rel = str(path.relative_to(root))
    if rel in excluded:
        continue
    st = path.lstat()
    if stat.S_ISDIR(st.st_mode):
        items.append({"path": rel, "type": "directory"})
    elif stat.S_ISREG(st.st_mode):
        data = path.read_bytes()
        items.append({"path": rel, "type": "regular", "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    elif stat.S_ISLNK(st.st_mode):
        raw = os.readlink(path)
        try:
            resolved = path.resolve(strict=True)
            inside = resolved == root_resolved or root_resolved in resolved.parents
            target_regular = resolved.is_file() and not resolved.is_symlink()
        except (FileNotFoundError, RuntimeError, OSError):
            resolved = None; inside = False; target_regular = False
        item = {"path": rel, "type": "symlink", "readlink_target": raw,
                "resolved_inside_result": inside,
                "resolved_target": str(resolved.relative_to(root_resolved)) if inside else None,
                "target_regular": target_regular}
        if inside and target_regular:
            data = resolved.read_bytes()
            item.update({"target_bytes": len(data),
                         "target_sha256": hashlib.sha256(data).hexdigest()})
        else:
            strict_errors.append(rel)
        items.append(item)
    else:
        items.append({"path": rel, "type": "unsupported", "mode": oct(st.st_mode)})
        strict_errors.append(rel)
if kind == "success" and strict_errors:
    raise RuntimeError("PASS cannot seal external/broken/special objects: " + ",".join(strict_errors))
payload = {
    "schema": "m784_m533_r15_unit_delay_artifact_inventory_v1",
    "terminal_kind": kind,
    "symlink_policy": "path and raw readlink target are sealed; PASS additionally requires an internal regular target whose bytes and SHA are sealed",
    "all_pass_symlinks_internal_regular_and_content_bound": kind != "success" or not strict_errors,
    "items": items,
}
(root / "ARTIFACT_INVENTORY.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_terminal_receipt() {
  local kind=$1 runner_rc=$2 receipt status marker
  if [[ "${kind}" == "success" ]]; then
    receipt="${RESULT_DIR}/RUN_COMPLETE.json"; marker="${RESULT_DIR}/RUN_COMPLETE.txt"
    status="PASS_FUNCTIONAL_VCS_ONLY"
  else
    receipt="${RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json"; marker="${RESULT_DIR}/FAILED_DO_NOT_CITE"
    status="FAILED_DO_NOT_CITE"
  fi
  RECEIPT_KIND="${kind}" RECEIPT_STATUS="${status}" RUNNER_RC="${runner_rc}" \
  CURRENT_PHASE_ENV="${CURRENT_PHASE}" CHILD_RC_ENV="${CHILD_RC}" \
  MONITOR_STATUS_ENV="${MONITOR_STATUS}" FAILURE_MESSAGE_ENV="${FAILURE_MESSAGE}" \
  PREFLIGHT_CLEANUP_RC_ENV="${PREFLIGHT_CLEANUP_RC}" RESULT_DIR_ENV="${RESULT_DIR}" \
  RUNNER_ENV="${RUNNER_PATH}" VCS_ENV="${VCS_BIN}" VCS_MSG_ENV="${VCS_MSG_REPORT}" \
  LICENSE_FILE_ENV="${LICENSE_FILE}" LMUTIL_ENV="${LMUTIL_BIN}" \
  ASSET_MANIFEST_ENV="${ASSET_MANIFEST}" \
  FOUNDRY_V_ENV="${FOUNDRY_SLOW_V}" FOUNDRY_DB_ENV="${FOUNDRY_SLOW_DB}" \
  SOURCE_CONTRACT_ENV="${SOURCE_CONTRACT}" SOURCE_STATIC_ENV="${SOURCE_STATIC_REVIEW}" \
  CANDIDATE_ENV="${RELEASE_CANDIDATE}" CANDIDATE_HAMMER_ENV="${CANDIDATE_HAMMER_REVIEW}" \
  RELEASE_ENV="${LAUNCH_RELEASE}" FINAL_HAMMER_ENV="${FINAL_HAMMER_REVIEW}" \
  TB_ENV="${TB}" TOP_ENV="${TOP_RTL}" SVA_ENV="${SVA}" MACRO_ENV="${MACRO_RTL}" \
  BINDING_ENV="${BINDING_PLAN}" M544_ENV="${M544_REVIEW}" M551_ENV="${M551_REVIEW}" \
  M547_HANDOFF_ENV="${M547_HANDOFF}" M558_ENV="${M558_REVIEW}" \
  R8_FAILED_ENV="${R8_FAILED_RECEIPT}" M717_ENV="${M717_REVIEW}" \
  R9_FAILED_ENV="${R9_FAILED_MARKER}" M726_ENV="${M726_REVIEW}" \
  R10_FAILED_ENV="${R10_FAILED_RECEIPT}" M736_ENV="${M736_REVIEW}" \
  R11_FAILED_ENV="${R11_FAILED_RECEIPT}" M738_ENV="${M738_REVIEW}" \
  M741_ENV="${M741_REVIEW}" M743_ENV="${M743_REVIEW}" M744_ENV="${M744_REVIEW}" \
  M757_ENV="${M757_REVIEW}" R13_FAILED_ENV="${R13_FAILED_RECEIPT}" \
  M770_ENV="${M770_REVIEW}" M782_ENV="${M782_REVIEW}" AUTHOR_PREFLIGHT_ENV="${AUTHOR_ENV_PREFLIGHT}" \
  python3 -I - "${receipt}" <<'PY' || return $?
import hashlib, json, os, sys
from pathlib import Path
def bind(env):
    path = Path(os.environ[env])
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("non-regular receipt binding: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
bindings = {}
for key, env in [
    ("runner_r15_unit_delay", "RUNNER_ENV"), ("vcs_binary", "VCS_ENV"),
    ("vcs_msg_report", "VCS_MSG_ENV"), ("license_file", "LICENSE_FILE_ENV"),
    ("lmutil", "LMUTIL_ENV"),
    ("foundry_asset_manifest", "ASSET_MANIFEST_ENV"), ("foundry_slow_v", "FOUNDRY_V_ENV"),
    ("foundry_slow_db", "FOUNDRY_DB_ENV"), ("source_contract", "SOURCE_CONTRACT_ENV"),
    ("source_static_review", "SOURCE_STATIC_ENV"), ("launch_candidate", "CANDIDATE_ENV"),
    ("candidate_hammer_review", "CANDIDATE_HAMMER_ENV"), ("launch_release", "RELEASE_ENV"),
    ("final_release_hammer_review", "FINAL_HAMMER_ENV"), ("testbench_r7", "TB_ENV"),
    ("top_r2", "TOP_ENV"), ("sva_r2", "SVA_ENV"), ("macro_adapter", "MACRO_ENV"),
    ("macro_binding_plan", "BINDING_ENV"), ("m544_failure_review", "M544_ENV"),
    ("m551_r6_failure_review", "M551_ENV"), ("m547_r6_author_handoff", "M547_HANDOFF_ENV"),
    ("m558_r7_failure_review", "M558_ENV"), ("consumed_r8_failure_receipt", "R8_FAILED_ENV"),
    ("m717_r8_failure_fresh_hammer", "M717_ENV"),
    ("consumed_r9_failure_marker", "R9_FAILED_ENV"),
    ("m726_r9_failure_fresh_hammer", "M726_ENV"),
    ("consumed_r10_failure_receipt", "R10_FAILED_ENV"),
    ("m736_r10_timing_failure_fresh_hammer", "M736_ENV"),
    ("consumed_r11_failure_receipt", "R11_FAILED_ENV"),
    ("m738_r11_failure_classification", "M738_ENV"),
    ("m741_tb_r5_failure_review", "M741_ENV"),
    ("m743_tb_r6_failure_review", "M743_ENV"),
    ("m744_tb_r7_source_static_pass", "M744_ENV"),
    ("m757_r12_premkdir_sha_literal_failure_audit", "M757_ENV"),
    ("consumed_r13_environment_failure", "R13_FAILED_ENV"),
    ("m770_r13_environment_failure_audit", "M770_ENV"),
    ("m782_r14_launch_boundary_failure_audit", "M782_ENV"),
    ("m772_author_readonly_environment_preflight_reused", "AUTHOR_PREFLIGHT_ENV")]:
    bindings[key] = bind(env)
root = Path(os.environ["RESULT_DIR_ENV"])
inv = root / "ARTIFACT_INVENTORY.json"
receipt = {
    "schema": "m784_m533_r15_unit_delay_atomic_terminal_receipt_v1",
    "status": os.environ["RECEIPT_STATUS"], "kind": os.environ["RECEIPT_KIND"],
    "paper_citable": False, "phase": os.environ["CURRENT_PHASE_ENV"],
    "runner_exit_rc": int(os.environ["RUNNER_RC"]), "child_rc": os.environ["CHILD_RC_ENV"],
    "monitor_status": os.environ["MONITOR_STATUS_ENV"],
    "failure_message": os.environ["FAILURE_MESSAGE_ENV"],
    "preflight_cleanup_rc": os.environ["PREFLIGHT_CLEANUP_RC_ENV"],
    "environment_policy": {
        "clean_env": True, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
        "VCS_ARCH_OVERRIDE": "linux",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
        "HOME": "UNSET", "identity_probe": "vcs -full64 -ID",
        "license_features": ["VCSCompiler_Net", "VCSRuntime_Net"],
    },
    "exact_live_launch_bindings": bindings,
    "old_partial_gate": "exact eight names, eight regular-file SHAs, and no extras revalidated twice before mkdir",
    "artifact_inventory": {"path": "ARTIFACT_INVENTORY.json",
        "sha256": hashlib.sha256(inv.read_bytes()).hexdigest()},
    "symlink_policy": "PASS permits only internal regular targets and binds path, raw target, resolved target, target bytes and target SHA in ARTIFACT_INVENTORY.json",
    "macro_model_mode": "foundry_UNIT_DELAY_functional",
    "claim_boundary": {"functional_vcs_only": os.environ["RECEIPT_KIND"] == "success",
        "timing_verified": False, "paper_citable_timing": False,
        "speedup": False, "ppa": False, "energy": False,
        "system_or_paper_headline": False},
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  printf '%s phase=%s runner_rc=%s child_rc=%s monitor_status=%s\n' \
    "${status}" "${CURRENT_PHASE}" "${runner_rc}" "${CHILD_RC}" "${MONITOR_STATUS}" >"${marker}"
}

verify_artifact_inventory_live() {
  RESULT_DIR_ENV="${RESULT_DIR}" python3 -I - <<'PY'
import hashlib, json, os, stat
from pathlib import Path
root = Path(os.environ["RESULT_DIR_ENV"])
inventory_path = root / "ARTIFACT_INVENTORY.json"
with inventory_path.open(encoding="utf-8") as handle:
    inventory = json.load(handle)
expected = {item["path"]: item for item in inventory["items"]}
excluded = {"ARTIFACT_INVENTORY.json", "RUN_COMPLETE.json", "RUN_COMPLETE.txt",
            "RUN_FAILED_OR_INCOMPLETE.json", "FAILED_DO_NOT_CITE",
            "SHA256SUMS", "SHA256SUMS.seal.sha256"}
live = {}
for path in root.rglob("*"):
    rel = str(path.relative_to(root))
    if rel not in excluded:
        live[rel] = path
if set(live) != set(expected):
    raise RuntimeError("artifact inventory path-set drift")
root_resolved = root.resolve(strict=True)
for rel, path in live.items():
    item = expected[rel]; mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        if item["type"] != "directory": raise RuntimeError("directory type drift: " + rel)
    elif stat.S_ISREG(mode):
        data = path.read_bytes()
        if item["type"] != "regular" or item["bytes"] != len(data) or item["sha256"] != hashlib.sha256(data).hexdigest():
            raise RuntimeError("regular content drift: " + rel)
    elif stat.S_ISLNK(mode):
        if item["type"] != "symlink" or item["readlink_target"] != os.readlink(path):
            raise RuntimeError("symlink object drift: " + rel)
        if item.get("target_sha256") is not None:
            resolved = path.resolve(strict=True)
            inside = resolved == root_resolved or root_resolved in resolved.parents
            data = resolved.read_bytes() if inside and resolved.is_file() else None
            if (not inside or data is None or item.get("resolved_target") != str(resolved.relative_to(root_resolved))
                    or item.get("target_bytes") != len(data)
                    or item.get("target_sha256") != hashlib.sha256(data).hexdigest()):
                raise RuntimeError("PASS symlink target drift: " + rel)
        elif inventory["terminal_kind"] == "success":
            raise RuntimeError("PASS symlink lacks content binding: " + rel)
    else:
        if item["type"] != "unsupported" or item["mode"] != oct(mode):
            raise RuntimeError("special object drift: " + rel)
PY
}

seal_terminal_members() {
  verify_artifact_inventory_live
  (
    cd -- "${RESULT_DIR}"
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
  )
}

reset_terminal_members() {
  rm -f -- "${RESULT_DIR}/ARTIFACT_INVENTORY.json" \
    "${RESULT_DIR}/RUN_COMPLETE.json" "${RESULT_DIR}/RUN_COMPLETE.txt" \
    "${RESULT_DIR}/RUN_FAILED_OR_INCOMPLETE.json" "${RESULT_DIR}/FAILED_DO_NOT_CITE" \
    "${RESULT_DIR}/SHA256SUMS" "${RESULT_DIR}/SHA256SUMS.seal.sha256"
}

cleanup() {
  local original_rc=$? effective_rc monitor_wait_rc=0 cleanup_rc=0 seal_rc=0
  effective_rc="${original_rc}"
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1
    wait "${MONITOR_PID}" >/dev/null 2>&1
    monitor_wait_rc=$?
    MONITOR_PID=""
    MONITOR_STATUS="cleanup_wait_rc_${monitor_wait_rc}"
  fi
  cleanup_preflight_preserve_rc
  cleanup_rc=$?
  if [[ "${cleanup_rc}" -ne 0 ]]; then
    FAILURE_MESSAGE="${FAILURE_MESSAGE}; preflight_cleanup_rc=${cleanup_rc}"
  fi
  if [[ "${RESULT_CREATED}" -eq 1 && "${TERMINAL_SEALED}" -eq 0 ]]; then
    # An unsealed post-mkdir EOF can never be a successful exit.  Translate
    # only that impossible zero into a failure code; ordinary and cleanup
    # return codes remain exactly as captured.
    if [[ "${effective_rc}" -eq 0 ]]; then
      effective_rc=124
      FAILURE_MESSAGE="${FAILURE_MESSAGE}; unexpected_unsealed_zero_exit"
    fi
    if reset_terminal_members && \
       build_artifact_inventory failure && \
       write_terminal_receipt failure "${effective_rc}" && \
       seal_terminal_members; then
      seal_rc=0
    else
      seal_rc=$?
    fi
    if [[ "${seal_rc}" -eq 0 ]]; then
      TERMINAL_SEALED=1
    else
      echo "CRITICAL: unable to double-seal consumed failure ${RESULT_DIR}" >&2
      effective_rc=125
    fi
  fi
  # A temporary-directory cleanup error is recorded but never overwrites the
  # already captured runner return code.  Only terminal sealing failure uses 125.
  exit "${effective_rc}"
}
trap cleanup EXIT

signal_exit() {
  local name=$1 rc=$2
  CURRENT_PHASE="signal_${name}"
  CHILD_RC="${rc}"
  FAILURE_MESSAGE="caught_${name}"
  exit "${rc}"
}
trap 'signal_exit int 130' INT
trap 'signal_exit term 143' TERM
trap 'signal_exit hup 129' HUP

command -v python3 >/dev/null 2>&1 || fail "python3 unavailable"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum unavailable"
command -v find >/dev/null 2>&1 || fail "find unavailable"
command -v sort >/dev/null 2>&1 || fail "sort unavailable"
command -v xargs >/dev/null 2>&1 || fail "xargs unavailable"
command -v rg >/dev/null 2>&1 || fail "rg unavailable"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be unset or empty"

require_regular_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1 "${TOP_RTL}"
require_regular_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
require_regular_sha db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983 "${BINDING_PLAN}"
require_regular_sha b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b "${SVA}"
require_regular_sha d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2 "${TB}"
verify_json_double_seal "${SOURCE_CONTRACT}"

verify_review_double_seal "${M544_DIR}"
verify_review_double_seal "${M551_DIR}"
verify_handoff_double_seal "${M547_HANDOFF_DIR}"
verify_review_double_seal "${M558_DIR}"
require_regular_sha eefb742d31dd3b9a417dbaf206f513ce7342ca9a5a5cac79e5d69e2911fae947 "${M544_REVIEW}"
require_regular_sha 55ea13bec1de6e1ae4dcc603bf4ba99600729c89be45d0c1deb062758cd0c43f "${M551_REVIEW}"
require_regular_sha ca191e2949b3f6c005b28f8c5c87468f9a2f1574ffe6d85aebe25edbe8284f06 "${M551_DIR}/SHA256SUMS"
require_regular_sha e530ee4786997a5ec61578335f32bb59caa0a2dff296969f49127b867e598caf "${M547_HANDOFF}"
require_regular_sha 058ac0624a02b84d1aaf56766834cdbe9e484c66064453254e331e477b517e15 "${M558_REVIEW}"
require_regular_sha 0a5d789415f265dc334a91c432e9b1939aa477c13adadd36bfbb7140e352bf46 "${M558_DIR}/SHA256SUMS"

verify_old_partial_closed_inventory
verify_r8_failure_and_m717_prerequisite
verify_r9_failure_and_m726_prerequisite
verify_r10_failure_and_m736_prerequisite
verify_r11_failure_and_tb_repair_prerequisites
verify_m757_r12_premkdir_failure_prerequisite
verify_r13_failure_m770_m782_and_author_preflight_prerequisites
python3 -I - "${M544_REVIEW}" "${M551_REVIEW}" "${M558_REVIEW}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: old = json.load(h)
with open(sys.argv[2], encoding="utf-8") as h: r6 = json.load(h)
with open(sys.argv[3], encoding="utf-8") as h: r7 = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(old.get("status") == "FAIL_M544_M533_R5_VCS_COMPILE_FAILURE_HAMMER__FAILED_UNSEALED_DO_NOT_CITE", "M544 status")
req(old.get("result_identity", {}).get("regular_file_count") == 8, "M544 inventory count")
req(r6.get("status") == "FAIL_M547_M533_R6_SOURCE_STATIC_HAMMER__REPAIR_REQUIRED", "M551 status")
req([r6.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0,3,1], "M551 findings")
req(r7.get("status") == "FAIL_M554_M533_R7_SOURCE_STATIC_HAMMER__REPAIR_REQUIRED", "M558 status")
req([r7.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0,1,0], "M558 findings")
req(r7.get("p1_findings", [{}])[0].get("id") == "M558-P1-01", "M558 finding identity")
PY

require_regular_sha c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f "${ASSET_MANIFEST}"
(cd -- "${ASSET_ROOT}" && sha256sum -c SHA256SUMS >/dev/null)
require_regular_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_SLOW_V}"
require_regular_sha cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf "${FOUNDRY_SLOW_DB}"
require_regular_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
require_regular_sha b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b "${VCS_MSG_REPORT}"
require_regular_sha fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490 "${LICENSE_FILE}"
require_regular_sha e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07 "${LMUTIL_BIN}"
[[ "${VCS_HOME-}" == "${EXPECTED_VCS_HOME}" ]] || fail "VCS_HOME exact environment required"
[[ "${VCS_ARCH_OVERRIDE-}" == "${EXPECTED_VCS_ARCH_OVERRIDE}" ]] || fail "VCS_ARCH_OVERRIDE exact environment required"
[[ "${SNPSLMD_LICENSE_FILE-}" == "${EXPECTED_SNPSLMD_LICENSE_FILE}" ]] || fail "SNPSLMD_LICENSE_FILE exact environment required"
[[ "${LM_LICENSE_FILE-}" == "${EXPECTED_LM_LICENSE_FILE}" ]] || fail "LM_LICENSE_FILE exact environment required"
[[ ! -v HOME ]] || fail "HOME must remain unset under env-i"
export VCS_HOME VCS_ARCH_OVERRIDE SNPSLMD_LICENSE_FILE LM_LICENSE_FILE
verify_r13_failure_m770_m782_and_author_preflight_prerequisites
require_regular_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA="$(sha256sum -- "${RUNNER_PATH}" | awk '{print $1}')"
SOURCE_CONTRACT_SHA="$(sha256sum -- "${SOURCE_CONTRACT}" | awk '{print $1}')"

verify_review_double_seal "${SOURCE_STATIC_DIR}"
SOURCE_STATIC_SHA="$(sha256sum -- "${SOURCE_STATIC_REVIEW}" | awk '{print $1}')"
python3 -I - "${SOURCE_STATIC_REVIEW}" "${RUNNER_SHA}" "${SOURCE_CONTRACT_SHA}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: r = json.load(h)
def req(c, m):
    if not c: raise RuntimeError(m)
req(r.get("schema") == "m784_m533_r15_unit_delay_source_static_hammer_v1", "r15 static schema")
req(r.get("status") == "PASS_M784_M533_R15_UNIT_DELAY_SOURCE_STATIC_HAMMER", "r15 static status")
req(r.get("verdict") == "PASS" and r.get("score_100") == 100, "r15 static score")
req([r.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0,0,0], "r15 static findings")
i = r.get("identity", {})
req(i.get("runner_sha256") == sys.argv[2], "r15 static runner binding")
req(i.get("source_contract_sha256") == sys.argv[3], "r15 static contract binding")
req(i.get("tb_r7_sha256") == "d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2", "r15 static TB binding")
st = r.get("static_selftest", {})
req(st.get("m738_m744_m757_m770_and_m782_prerequisites_pass") is True, "r15 prerequisite chain")
req(st.get("all_require_regular_sha_literals_67_lowerhex_and_live_match_pass") is True, "r15 complete SHA cross-edge audit")
req(st.get("runner_m770_executable_heredoc_predicate_pass") is True, "r15 executable M770 predicate")
req(st.get("runner_m770_missing_or_wrong_key_negative_test_pass") is True, "r15 M770 predicate negative test")
req(st.get("unit_delay_define_pass") is True, "r15 UNIT_DELAY mode")
req(st.get("forbidden_timing_bypass_absent_pass") is True, "r15 no timing bypass")
req(st.get("claim_separation_pass") is True, "r15 claim separation")
req(st.get("new_result_path_absent") is True, "r15 result absence")
req(r.get("decision", {}).get("vcs_launch_authorized_now") is False, "static review may not launch")
PY

verify_json_double_seal "${RELEASE_CANDIDATE}"
verify_review_double_seal "${CANDIDATE_HAMMER_DIR}"
verify_json_double_seal "${LAUNCH_RELEASE}"
verify_review_double_seal "${FINAL_HAMMER_DIR}"
RELEASE_CANDIDATE_SHA="$(sha256sum -- "${RELEASE_CANDIDATE}" | awk '{print $1}')"
CANDIDATE_HAMMER_SHA="$(sha256sum -- "${CANDIDATE_HAMMER_REVIEW}" | awk '{print $1}')"
LAUNCH_RELEASE_SHA="$(sha256sum -- "${LAUNCH_RELEASE}" | awk '{print $1}')"
FINAL_HAMMER_SHA="$(sha256sum -- "${FINAL_HAMMER_REVIEW}" | awk '{print $1}')"
python3 -I - "${RELEASE_CANDIDATE}" "${CANDIDATE_HAMMER_REVIEW}" "${LAUNCH_RELEASE}" \
  "${FINAL_HAMMER_REVIEW}" "${RUNNER_SHA}" "${SOURCE_CONTRACT_SHA}" "${SOURCE_STATIC_SHA}" \
  "${RELEASE_CANDIDATE_SHA}" "${CANDIDATE_HAMMER_SHA}" "${LAUNCH_RELEASE_SHA}" <<'PY'
import json, sys
p = sys.argv[1:]
with open(p[0], encoding="utf-8") as h: c = json.load(h)
with open(p[1], encoding="utf-8") as h: ch = json.load(h)
with open(p[2], encoding="utf-8") as h: r = json.load(h)
with open(p[3], encoding="utf-8") as h: fh = json.load(h)
runner, contract, static, candidate, hammer, release = p[4:]
def req(x, m):
    if not x: raise RuntimeError(m)
auth = {"vcs_runs":1,"simv_runs":1,"iverilog_runs":0,"verilator_runs":0,
        "dc_runs":0,"formality_runs":0,"pt_runs":0,"ptpx_runs":0,
        "cpu_runs":0,"gpu_runs":0,"network_or_remote_jobs":0}
policy = {"prelaunch_samples":3,"sample_interval_seconds":2,
          "mem_available_min_kib":134217728,"swap_free_min_kib":33554432,
          "commit_headroom_min_kib":33554432,"cgroup_version":1,
          "memory_failcnt_must_not_increase":True,"under_oom_must_equal_zero":True,
          "oom_kill_must_equal_zero":True,"missing_counter_is_failure":True,
          "same_uid_synopsys_vcs_simv_collision_must_be_zero":True}
env_policy = {"clean_env":True,"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C",
              "VCS_HOME":"/opt/synopsys/vcs/V-2023.12-SP1",
              "VCS_ARCH_OVERRIDE":"linux","SNPSLMD_LICENSE_FILE":"27030@ic.ismd-nemo",
              "LM_LICENSE_FILE":"/opt/synopsys/Synopsys.dat","HOME":"UNSET",
              "identity_probe":"vcs -full64 -ID",
              "license_features":["VCSCompiler_Net","VCSRuntime_Net"]}
req(c.get("schema") == "m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_v1", "candidate schema")
req(c.get("launch_now") is False, "candidate launch_now")
req(c.get("authorization") == auth, "candidate closed authorization")
req(c.get("resource_policy") == policy, "candidate resource policy")
req(c.get("environment_policy") == env_policy, "candidate environment policy")
req(c.get("identity", {}).get("runner_sha256") == runner, "candidate runner")
req(c.get("identity", {}).get("source_contract_sha256") == contract, "candidate contract")
req(c.get("macro_model_mode") == "foundry_UNIT_DELAY_functional", "candidate macro mode")
ccb = c.get("claim_boundary", {})
req(ccb.get("functional_vcs_only") is True and ccb.get("timing_verified") is False, "candidate claim separation")
req(ch.get("schema") == "m784_m533_r15_unit_delay_vcs_launch_admission_candidate_hammer_v1", "candidate hammer schema")
req(ch.get("verdict") == "PASS" and ch.get("score_100") == 100, "candidate hammer score")
req([ch.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0,0,0], "candidate hammer findings")
req(ch.get("identity", {}).get("candidate_sha256") == candidate, "candidate hammer binding")
req(ch.get("identity", {}).get("runner_sha256") == runner, "candidate hammer runner")
req(ch.get("identity", {}).get("source_static_review_sha256") == static, "candidate hammer static")
req(ch.get("decision", {}).get("vcs_launch_authorized_now") is False, "candidate hammer may not launch")
req(r.get("schema") == "m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_v1", "release schema")
req(r.get("launch_now") is True, "release launch_now")
req(r.get("authorization") == auth, "release closed authorization")
req(r.get("resource_policy") == policy, "release resource policy")
req(r.get("environment_policy") == env_policy, "release environment policy")
req(r.get("macro_model_mode") == "foundry_UNIT_DELAY_functional", "release macro mode")
rcb = r.get("claim_boundary", {})
req(rcb.get("functional_vcs_only") is True and rcb.get("timing_verified") is False, "release claim separation")
ri = r.get("identity", {})
req(ri.get("runner_sha256") == runner and ri.get("source_contract_sha256") == contract, "release source identity")
req(ri.get("source_static_review_sha256") == static, "release static")
req(ri.get("candidate_sha256") == candidate and ri.get("candidate_hammer_review_sha256") == hammer, "release candidate chain")
req(r.get("unique_attempt", {}).get("result_path") == "results/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_r15_20260828", "release result")
req(fh.get("schema") == "m784_m533_r15_unit_delay_vcs_final_launch_release_hammer_v1", "final hammer schema")
req(fh.get("verdict") == "PASS" and fh.get("score_100") == 100, "final hammer score")
req([fh.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0,0,0], "final hammer findings")
fi = fh.get("identity", {})
req(fi.get("final_release_sha256") == release and fi.get("runner_sha256") == runner, "final hammer release binding")
req(fi.get("candidate_sha256") == candidate, "final hammer candidate binding")
req(fh.get("decision", {}).get("exactly_one_vcs_attempt_authorized_now") is True, "final authorization")
req(fh.get("decision", {}).get("all_other_runs_authorized") is False, "closed non-VCS auth")
PY

[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt already exists: ${RESULT_DIR}"
PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/m784_m533_r15_unit_delay_vcs_preflight.XXXXXXXX")"

scan_same_uid_collisions() {
  local output=$1
  python3 -I - "${output}" <<'PY'
import json, os, re, sys
from pathlib import Path
uid = os.getuid(); self_pid = os.getpid(); runner_pid = os.getppid()
ignored = {self_pid, runner_pid}; matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit(): continue
    pid = int(entry.name)
    if pid in ignored: continue
    try:
        if entry.stat().st_uid != uid: continue
        exe = os.path.basename(os.readlink(entry / "exe"))
        argv = [x.decode("utf-8", "replace") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
        starttime = (entry / "stat").read_text().split()[21]
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError): continue
    tokens = {os.path.basename(x) for x in argv[:6]}; tokens.add(exe); joined = " ".join(argv)
    cls = None
    hits = sorted(tokens & {"dc_shell","dc_shell-t","fm_shell","fm_shell_exec","pt_shell","pt_shell_exec"})
    if hits: cls = hits[0]
    elif "common_shell_exec" in tokens and re.search(r"(?:^|\s)-shell\s+(?:dc_shell|dc_shell-t|fm_shell|pt_shell)(?:\s|$)", joined): cls = "common_shell_exec_for_dc_fm_pt_ptpx"
    elif any(x == "vcs" or x.startswith("vcs.") or x in {"vcs1","vlogan","vhdlan"} for x in tokens): cls = "vcs"
    elif any(x == "simv" or x.startswith("simv.") for x in tokens): cls = "simv"
    if cls: matches.append({"pid": pid, "starttime": starttime, "class": cls, "exe": exe, "argv": argv})
payload = {"schema":"m784_m533_r15_unit_delay_collision_scan_v1","uid":uid,"scanner_pid":self_pid,
           "runner_pid":runner_pid,"matches":matches,"verdict":"PASS" if not matches else "FAIL"}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if matches: raise SystemExit(1)
PY
}

resolve_cgroup_v1() {
  local rel dir file
  rel="$(awk -F: '$2 == "memory" {print $3}' /proc/self/cgroup)"
  [[ "${rel}" == /* && "${rel}" != *..* ]] || fail "cgroup-v1 memory path unavailable"
  CGROUP_SESSION_DIR="/sys/fs/cgroup/memory${rel}"
  CGROUP_USER_DIR="/sys/fs/cgroup/memory/user.slice"
  for dir in "${CGROUP_SESSION_DIR}" "${CGROUP_USER_DIR}"; do
    [[ -d "${dir}" && ! -L "${dir}" ]] || fail "missing cgroup directory: ${dir}"
    for file in memory.failcnt memory.oom_control memory.usage_in_bytes; do
      [[ -r "${dir}/${file}" && ! -L "${dir}/${file}" ]] || fail "missing cgroup counter: ${dir}/${file}"
    done
  done
}

read_oom_field() {
  local path=$1 field=$2 value
  value="$(awk -v k="${field}" '$1 == k {print $2}' "${path}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "bad ${field}: ${path}"
  printf '%s\n' "${value}"
}

resource_preflight() {
  local output=$1 sample limit committed available swap headroom sf uf su sk uu uk
  : >"${output}"
  BASE_SESSION_FAILCNT="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"
  BASE_USER_FAILCNT="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
  for sample in 1 2 3; do
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"; committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"; swap="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    [[ "${limit}" =~ ^[0-9]+$ && "${committed}" =~ ^[0-9]+$ && "${available}" =~ ^[0-9]+$ && "${swap}" =~ ^[0-9]+$ ]] || fail "bad meminfo"
    headroom=$((limit - committed)); sf="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"; uf="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
    su="$(read_oom_field "${CGROUP_SESSION_DIR}/memory.oom_control" under_oom)"; sk="$(read_oom_field "${CGROUP_SESSION_DIR}/memory.oom_control" oom_kill)"
    uu="$(read_oom_field "${CGROUP_USER_DIR}/memory.oom_control" under_oom)"; uk="$(read_oom_field "${CGROUP_USER_DIR}/memory.oom_control" oom_kill)"
    printf 'sample=%s timestamp=%s mem_available_kib=%s swap_free_kib=%s commit_headroom_kib=%s session_failcnt=%s user_failcnt=%s session_under_oom=%s session_oom_kill=%s user_under_oom=%s user_oom_kill=%s\n' \
      "${sample}" "$(date --iso-8601=seconds)" "${available}" "${swap}" "${headroom}" "${sf}" "${uf}" "${su}" "${sk}" "${uu}" "${uk}" >>"${output}"
    [[ "${available}" -ge 134217728 && "${swap}" -ge 33554432 && "${headroom}" -ge 33554432 ]] || fail "resource threshold failed"
    [[ "${sf}" == "${BASE_SESSION_FAILCNT}" && "${uf}" == "${BASE_USER_FAILCNT}" && "${su}" == 0 && "${sk}" == 0 && "${uu}" == 0 && "${uk}" == 0 ]] || fail "cgroup resource gate failed"
    [[ "${sample}" -eq 3 ]] || sleep 2
  done
}

CURRENT_PHASE="pre_mkdir_collision_initial"; scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_initial.json" || fail "initial collision"
CURRENT_PHASE="pre_mkdir_cgroup"; resolve_cgroup_v1
CURRENT_PHASE="pre_mkdir_resource"; resource_preflight "${PREFLIGHT_DIR}/resource_prelaunch.log"
CURRENT_PHASE="pre_mkdir_collision_final"; scan_same_uid_collisions "${PREFLIGHT_DIR}/collision_final.json" || fail "final collision"
CURRENT_PHASE="pre_mkdir_old_partial_final_revalidation"; verify_old_partial_closed_inventory
CURRENT_PHASE="pre_mkdir_r8_m717_final_revalidation"; verify_r8_failure_and_m717_prerequisite
CURRENT_PHASE="pre_mkdir_r9_m726_final_revalidation"; verify_r9_failure_and_m726_prerequisite
CURRENT_PHASE="pre_mkdir_r10_m736_final_revalidation"; verify_r10_failure_and_m736_prerequisite
CURRENT_PHASE="pre_mkdir_r11_tb_repair_final_revalidation"; verify_r11_failure_and_tb_repair_prerequisites
CURRENT_PHASE="pre_mkdir_m757_r12_premkdir_failure_final_revalidation"; verify_m757_r12_premkdir_failure_prerequisite
CURRENT_PHASE="pre_mkdir_r13_m770_environment_failure_final_revalidation"; verify_r13_failure_m770_and_author_preflight_prerequisites
CURRENT_PHASE="pre_mkdir_vcs_full64_identity_and_license_status"
vcs_license_preflight "${PREFLIGHT_DIR}/vcs_full64_id.txt" \
  "${PREFLIGHT_DIR}/lmstat_VCSCompiler_Net.txt" "${PREFLIGHT_DIR}/lmstat_VCSRuntime_Net.txt"

# r15 retains M558-P1-01 closure: the final absence recheck, atomic mkdir and ownership
# publication are one catchable-signal critical section.  A failed mkdir never
# claims or removes an identity that could belong to another process.
CURRENT_PHASE="pre_mkdir_atomic_attempt_publication"
trap '' INT TERM HUP
if [[ -e "${RESULT_DIR}" ]]; then
  trap 'signal_exit int 130' INT; trap 'signal_exit term 143' TERM; trap 'signal_exit hup 129' HUP
  fail "result appeared during preflight"
fi
if mkdir -- "${RESULT_DIR}"; then
  RESULT_CREATED=1
  trap 'signal_exit int 130' INT; trap 'signal_exit term 143' TERM; trap 'signal_exit hup 129' HUP
else
  mkdir_rc=$?
  trap 'signal_exit int 130' INT; trap 'signal_exit term 143' TERM; trap 'signal_exit hup 129' HUP
  [[ ! -e "${RESULT_DIR}" ]] || fail "atomic attempt creation failed rc=${mkdir_rc}; unowned result exists"
  fail "atomic attempt creation failed rc=${mkdir_rc}; no result created"
fi
CURRENT_PHASE="post_mkdir_evidence_copy"
cp -- "${PREFLIGHT_DIR}/collision_initial.json" "${PREFLIGHT_DIR}/collision_final.json" \
  "${PREFLIGHT_DIR}/resource_prelaunch.log" "${PREFLIGHT_DIR}/vcs_full64_id.txt" \
  "${PREFLIGHT_DIR}/lmstat_VCSCompiler_Net.txt" "${PREFLIGHT_DIR}/lmstat_VCSRuntime_Net.txt" "${RESULT_DIR}/"
printf '%s\n' \
  'macro_model_mode=foundry_UNIT_DELAY_functional' \
  'functional_vcs_only=true' \
  'timing_verified=false' \
  'paper_citable_timing=false' > "${RESULT_DIR}/MACRO_MODEL_MODE.txt"
CURRENT_PHASE="post_mkdir_collision"; scan_same_uid_collisions "${RESULT_DIR}/collision_postmkdir.json" || fail "post-mkdir collision"

runtime_resource_sample() {
  local output=$1 phase=$2 sf uf su sk uu uk
  sf="$(<"${CGROUP_SESSION_DIR}/memory.failcnt")"; uf="$(<"${CGROUP_USER_DIR}/memory.failcnt")"
  su="$(awk '$1=="under_oom" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"; sk="$(awk '$1=="oom_kill" {print $2}' "${CGROUP_SESSION_DIR}/memory.oom_control")"
  uu="$(awk '$1=="under_oom" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"; uk="$(awk '$1=="oom_kill" {print $2}' "${CGROUP_USER_DIR}/memory.oom_control")"
  [[ "${sf}" =~ ^[0-9]+$ && "${uf}" =~ ^[0-9]+$ && "${su}" =~ ^[0-9]+$ && "${sk}" =~ ^[0-9]+$ && "${uu}" =~ ^[0-9]+$ && "${uk}" =~ ^[0-9]+$ ]] || return 1
  printf 'phase=%s timestamp=%s epoch=%s session_failcnt=%s user_failcnt=%s session_under_oom=%s session_oom_kill=%s user_under_oom=%s user_oom_kill=%s\n' \
    "${phase}" "$(date --iso-8601=seconds)" "$(date +%s)" "${sf}" "${uf}" "${su}" "${sk}" "${uu}" "${uk}" >>"${output}" || return 1
  [[ "${sf}" == "${BASE_SESSION_FAILCNT}" && "${uf}" == "${BASE_USER_FAILCNT}" && "${su}" == 0 && "${sk}" == 0 && "${uu}" == 0 && "${uk}" == 0 ]]
}

resource_monitor() {
  local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0
  local tmp="${heartbeat}.tmp.$$"
  : >"${output}"
  while :; do
    runtime_resource_sample "${output}" periodic || { printf 'resource_violation\n' >>"${violation}"; return 64; }
    seq=$((seq+1)); printf 'sequence=%s epoch=%s\n' "${seq}" "$(date +%s)" >"${tmp}"; mv -- "${tmp}" "${heartbeat}"
    if [[ -e "${request}" ]]; then
      runtime_resource_sample "${output}" final_synchronous || { printf 'final_resource_violation\n' >>"${violation}"; return 67; }
      printf 'final_sample_ack=1 sequence=%s epoch=%s\n' "${seq}" "$(date +%s)" >"${ack}"
      return 0
    fi
    sleep 1
  done
}

require_monitor_live() {
  local where=$1 epoch now
  [[ -n "${MONITOR_PID}" ]] || fail "monitor absent ${where}"
  kill -0 "${MONITOR_PID}" >/dev/null 2>&1 || fail "monitor dead ${where}"
  [[ -f "${RESULT_DIR}/RESOURCE_HEARTBEAT" ]] || fail "heartbeat absent ${where}"
  epoch="$(sed -nE 's/.*epoch=([0-9]+).*/\1/p' "${RESULT_DIR}/RESOURCE_HEARTBEAT")"
  [[ "${epoch}" =~ ^[0-9]+$ ]] || fail "heartbeat malformed"
  now="$(date +%s)"; [[ $((now-epoch)) -le 3 ]] || fail "heartbeat stale"
}

finalize_monitor() {
  local i wait_rc ack=0
  require_monitor_live final_request
  : >"${RESULT_DIR}/RESOURCE_FINAL_REQUEST"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    [[ -f "${RESULT_DIR}/RESOURCE_FINAL_ACK" ]] && { ack=1; break; }
    kill -0 "${MONITOR_PID}" >/dev/null 2>&1 || break
    sleep 1
  done
  [[ "${ack}" -eq 1 ]] || fail "monitor final ack absent"
  set +e; wait "${MONITOR_PID}"; wait_rc=$?; set -e
  MONITOR_PID=""
  [[ "${wait_rc}" -eq 0 && ! -e "${RESULT_DIR}/RESOURCE_VIOLATION" ]] || fail "monitor final failure rc=${wait_rc}"
  [[ "$(rg -c '^final_sample_ack=1 sequence=[0-9]+ epoch=[0-9]+$' "${RESULT_DIR}/RESOURCE_FINAL_ACK")" == 1 ]] || fail "bad final ack"
  [[ "$(rg -c '^phase=final_synchronous ' "${RESULT_DIR}/resource_runtime.log")" == 1 ]] || fail "bad final sample"
  MONITOR_STATUS="final_sample_ack_pass"
}

CURRENT_PHASE="runtime_monitor_start"
resource_monitor "${RESULT_DIR}/resource_runtime.log" "${RESULT_DIR}/RESOURCE_VIOLATION" \
  "${RESULT_DIR}/RESOURCE_HEARTBEAT" "${RESULT_DIR}/RESOURCE_FINAL_REQUEST" "${RESULT_DIR}/RESOURCE_FINAL_ACK" &
MONITOR_PID=$!; MONITOR_STATUS="starting"
for i in 1 2 3 4 5; do [[ -f "${RESULT_DIR}/RESOURCE_HEARTBEAT" ]] && break; kill -0 "${MONITOR_PID}" >/dev/null 2>&1 || fail "monitor startup"; sleep 1; done
require_monitor_live before_compile; MONITOR_STATUS="live_before_compile"

cd -- "${RESULT_DIR}"
CURRENT_PHASE="vcs_compile"; CHILD_RC="running"
set +e
"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext -debug_access+pp \
  +define+UNIT_DELAY +vcs+lic+wait \
  "${FOUNDRY_SLOW_V}" "${MACRO_RTL}" "${TOP_RTL}" "${SVA}" "${TB}" \
  -top tb_m528_dead_write_only_1rw_product_capture_r3 -o simv 2>&1 | tee compile.log
pipe_rc=("${PIPESTATUS[@]}")
set -e
CHILD_RC="vcs_${pipe_rc[0]}_tee_${pipe_rc[1]}"
if [[ "${pipe_rc[0]}" -ne 0 || "${pipe_rc[1]}" -ne 0 ]]; then finalize_monitor; fail "VCS/tee failed ${CHILD_RC}"; fi
require_monitor_live after_compile; MONITOR_STATUS="live_after_compile"

CURRENT_PHASE="simv_run"; CHILD_RC="running"
set +e
./simv 2>&1 | tee sim.log
pipe_rc=("${PIPESTATUS[@]}")
set -e
CHILD_RC="simv_${pipe_rc[0]}_tee_${pipe_rc[1]}"
finalize_monitor
[[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]}" -eq 0 ]] || fail "simv/tee failed ${CHILD_RC}"

CURRENT_PHASE="functional_and_coverage_gate"; CHILD_RC="0"
[[ "$(rg -c '^PASS_M533_M528_DW1RW_R7_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)" == 1 ]] || fail "functional token"
[[ "$(rg -c '^COVERAGE_M533_M528_DW1RW_R7 ' sim.log)" == 1 ]] || fail "coverage token"
COVERAGE_LINE="$(rg '^COVERAGE_M533_M528_DW1RW_R7 ' sim.log)"
for field in dead_plus_read deadline_read_write same_address_forward pending_plus_forward full_no_credit liveness_sequences parent_modes stalled_raw_recovery stalled_raw_forward_recovery stalled_raw_response_recovery pingpong_overlap endpoint_rows all_slices; do
  value="$(sed -nE "s/.* ${field}=([0-9]+)( |$).*/\\1/p" <<<"${COVERAGE_LINE}")"
  [[ "${value}" =~ ^[0-9]+$ && "${value}" -ge 1 ]] || fail "cover ${field}"
done
[[ "${COVERAGE_LINE}" == *" minima=1 normal_covers=13"* ]] || fail "coverage minima"
P2_LINE="$(rg '^P2_STRENGTH_M533_M528_DW1RW_R3 ' sim.log)"
[[ "$(sed -nE 's/.* consecutive_distinct_reads=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")" -ge 1 ]] || fail "P2 pair"
[[ "$(sed -nE 's/.* response_identity_checks=([0-9]+)( |$).*/\1/p' <<<"${P2_LINE}")" -ge 2 ]] || fail "P2 responses"
PASS_LINE="$(rg '^PASS_M533_M528_DW1RW_R7_DIRECTED_RANDOM_AND_ATTACKS ' sim.log)"
for field in dirty_reserved stale_epoch overflow wrong_parent read_before_write parent_only_nonzero; do
  [[ "$(sed -nE "s/.* ${field}=([0-9]+)( |$).*/\\1/p" <<<"${PASS_LINE}")" == 1 ]] || fail "attack ${field}"
done
[[ "${PASS_LINE}" == *" attacks=6 "* ]] || fail "attack count"
if rg -n 'Timing violation|Assertion.*failed|Error-\[SVA|\$error|\$fatal|normal scoreboard errors|protocol attack not detected' compile.log sim.log; then fail "failure signature"; fi

# Close catchable-signal and temporary-cleanup races before the PASS seal.
CURRENT_PHASE="success_preseal_cleanup"
trap '' INT TERM HUP
if ! cleanup_preflight_preserve_rc; then
  trap 'signal_exit int 130' INT; trap 'signal_exit term 143' TERM; trap 'signal_exit hup 129' HUP
  fail "preflight cleanup failed before success seal rc=${PREFLIGHT_CLEANUP_RC}"
fi
CURRENT_PHASE="success_terminal_seal"; FAILURE_MESSAGE=""
reset_terminal_members
build_artifact_inventory success
write_terminal_receipt success 0
seal_terminal_members
# From here to exit there are only non-fallible shell-state changes and exit.
TERMINAL_SEALED=1
trap - EXIT
exit 0
