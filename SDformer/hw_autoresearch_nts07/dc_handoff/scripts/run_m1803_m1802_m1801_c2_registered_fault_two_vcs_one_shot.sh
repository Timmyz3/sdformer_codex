#!/usr/bin/env bash
set -euo pipefail

dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "${dc_root}/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
release_rel="contracts/m1803_m1802_m1801_c2_registered_fault_two_vcs_launch_release_r1_20260902.json"
release="${hw_root}/${release_rel}"
contract_rel="contracts/m1801_m1797_c2_registered_public_fault_evidence_successor_source_contract_r1_20260902.json"
review_rel="reviews/m1802_m1801_c2_registered_public_fault_evidence_successor_source_hammer_r1_20260902"
result="${hw_root}/results/m1803_m1802_m1801_c2_registered_fault_two_vcs_r1_20260902"
attempt="${hw_root}/results/.m1803_m1802_m1801_c2_registered_fault_two_vcs_attempt_consumed"
work="${hw_root}/results/.m1803_m1802_m1801_c2_registered_fault_two_vcs_work.$$"
quarantine="${hw_root}/results/m1803_m1802_m1801_c2_registered_fault_two_vcs_r1_20260902.failed_or_incomplete.$$.quarantine"
vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
python36=/usr/libexec/platform-python3.6
complete=0
attempt_consumed=0

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1803 gate failure: %s\n' "$*" >&2; exit 3; }
seal_dir() {
    local dir=$1
    (cd "${dir}" && find . -type f ! -name SHA256SUMS \
        ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum \
        >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 \
        && sha256sum -c SHA256SUMS >/dev/null \
        && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
verify_dir_seal() {
    local rel=$1
    (cd "${hw_root}/${rel}" && sha256sum -c SHA256SUMS >/dev/null \
        && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) \
        || fail "directory seal ${rel}"
}
verify_file_seal() {
    local rel=$1 base="${hw_root}/$1"
    (cd "$(dirname "${base}")" \
        && sha256sum -c "$(basename "${base}").sha256" >/dev/null \
        && sha256sum -c "$(basename "${base}").sha256.seal.sha256" >/dev/null) \
        || fail "file seal ${rel}"
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${attempt_consumed}" -eq 1 && "${complete}" -ne 1 ]]; then
        mkdir -p "${work}"
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\n' "${rc}" \
            >"${work}/RUN_FAILED_OR_INCOMPLETE.txt"
        seal_dir "${work}" || true
        [[ -e "${quarantine}" ]] || mv "${work}" "${quarantine}" || true
    fi
    exit "${rc}"
}
collision_gate() {
    if pgrep -x vcs1 >/dev/null || pgrep -x vlogan >/dev/null \
        || pgrep -f '(^|/)(vcs)( |$)' >/dev/null \
        || pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null; then
        fail "VCS/DC/FM/PT collision"
    fi
}
resource_gate() {
    local available committed limit
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    [[ "${available}" -ge 8388608 && $((limit-committed)) -ge 8388608 ]] \
        || fail "less than 8 GiB headroom"
}
verify_authority() {
    [[ -f "${release}" && ! -L "${release}" ]] || fail "missing release"
    verify_file_seal "${release_rel}"
    verify_file_seal "${contract_rel}"
    verify_dir_seal "${review_rel}"
    "${python36}" - "${release}" "${runner}" \
        "${hw_root}/${contract_rel}" "${hw_root}/${review_rel}" <<'PY'
import hashlib, json, sys
from pathlib import Path
release_p, runner_p, contract_p, review_d = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d = json.loads(release_p.read_text(encoding="utf-8"))
if d.get("schema") != "m1803_m1802_m1801_c2_registered_fault_two_vcs_launch_release_r1_v1":
    raise SystemExit(10)
if d.get("status") != "AUTHORIZE_EXACTLY_ONE_M1803_TWO_VCS_CAMPAIGN":
    raise SystemExit(11)
if d.get("fresh_execution_budget") != {"vcs_compile_runs": 2, "simv_runs": 2,
        "dc_runs": 0, "ptpx_runs": 0, "automatic_retry": False}:
    raise SystemExit(12)
review_p = review_d / "review.json"
manifest_p = review_d / "SHA256SUMS"
outer_p = review_d / "SHA256SUMS.seal.sha256"
want = {"runner_sha256": sha(runner_p), "source_contract_sha256": sha(contract_p),
        "source_review_sha256": sha(review_p),
        "source_review_manifest_sha256": sha(manifest_p),
        "source_review_outer_seal_file_sha256": sha(outer_p)}
if d.get("identity") != want:
    raise SystemExit(13)
review = json.loads(review_p.read_text(encoding="utf-8"))
if review.get("status") != ("PASS_M1802_M1801_C2_REGISTERED_PUBLIC_FAULT_"
        "EVIDENCE_SUCCESSOR_SOURCE_HAMMER__P0_0_P1_0__EXACTLY_TWO_VCS_"
        "CAMPAIGNS_AUTHORIZED__NO_EDA"):
    raise SystemExit(14)
if review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
    raise SystemExit(15)
if d.get("claim_boundary") != {"rtl_functionality": False,
        "root_cause_confirmed": False, "mapped_functionality": False,
        "performance": False, "ppa": False, "power": False, "energy": False,
        "system_speedup": False, "paper_citable": False}:
    raise SystemExit(16)
PY
}
absolute_filelist() {
    local src=$1 dst=$2 line
    : >"${dst}"
    while IFS= read -r line || [[ -n "${line}" ]]; do
        if [[ -z "${line}" || "${line}" == +* || "${line}" == -* ]]; then
            printf '%s\n' "${line}" >>"${dst}"
        else
            printf '%s/%s\n' "${hw_root}" "${line}" >>"${dst}"
        fi
    done <"${hw_root}/${src}"
}
run_campaign() {
    local name=$1 filelist=$2 top=$3 seed=$4 pass_re=$5
    local dir="${work}/${name}" rc
    mkdir "${dir}"
    absolute_filelist "${filelist}" "${dir}/files.abs.f"
    (cd "${dir}" && "${vcs}" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir=csrc -f files.abs.f -top "${top}" -o simv \
        >compile.log 2>&1); rc=$?
    printf '%s\n' "${rc}" >"${dir}/compile.rc"
    [[ "${rc}" -eq 0 && -x "${dir}/simv" ]] || return 20
    ! grep -Eiq 'Error-\[|^Error|^Fatal|Fatal:' "${dir}/compile.log" || return 21
    (cd "${dir}" && ./simv "+ntb_random_seed=${seed}" -no_save \
        -assert report=assert.report -cm assert >sim.log 2>&1); rc=$?
    printf '%s\n' "${rc}" >"${dir}/sim.rc"
    [[ "${rc}" -eq 0 ]] || return 22
    [[ "$(grep -Ec "${pass_re}" "${dir}/sim.log")" -eq 1 ]] || return 23
    ! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "${dir}/sim.log" "${dir}/assert.report" || return 24
}

verify_authority
[[ -x "${vcs}" && -x "${python36}" ]] || fail "tool identity"
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
    || fail "execution namespace consumed"
collision_gate
resource_gate
mkdir "${attempt}"
attempt_consumed=1
mkdir "${work}"
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf 'status=ONE_M1803_TWO_VCS_ATTEMPT_CONSUMED\nrunner_sha256=%s\n' \
    "$(sha "${runner}")" >"${attempt}/ATTEMPT.txt"

run_campaign unit \
    dc_handoff/filelists/iscas_m1801_c2_registered_public_fault_export_directed_vcs.f \
    tb_m1801_c2_registered_public_fault_export_directed 180301 \
    '^PASS M1801 registered public fault export .*attack_classes=4 .*public_fault_binary=true .*$'
run_campaign full \
    dc_handoff/filelists/iscas_m1801_c2_registered_public_fault_k8_vs_k1x8_vcs.f \
    tb_m1801_c2_registered_public_fault_k8_vs_k1x8_raw4_acc24 180302 \
    '^PASS M1801 channel-split cutthrough-8bank equal-bandwidth FC2 VCS .*protocol_attacks=5 .*$'

"${python36}" - "${work}" "${runner}" "${release}" <<'PY'
import hashlib, json, sys
from pathlib import Path
work, runner, release = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {
  "schema": "m1803_m1802_m1801_c2_registered_fault_two_vcs_result_r1_v1",
  "status": "PASS_M1803_M1801_REGISTERED_PUBLIC_FAULT_TWO_VCS",
  "campaigns": 2, "vcs_compiles": 2, "simv_runs": 2,
  "unit_pass_tokens": 1, "full_pass_tokens": 1,
  "assertion_failures": 0, "automatic_retry": False,
  "runner_sha256": sha(runner), "release_sha256": sha(release),
  "claim_boundary": {"rtl_functionality": True,
    "root_cause_confirmed": True, "mapped_functionality": False,
    "performance": False, "ppa": False, "power": False, "energy": False,
    "system_speedup": False, "paper_citable": False}}
(work / "result.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n",
                                  encoding="utf-8")
(work / "RUN_COMPLETE.txt").write_text(receipt["status"]+"\n", encoding="utf-8")
PY
seal_dir "${work}"
mv "${work}" "${result}"
complete=1
trap - EXIT INT TERM HUP
printf 'PASS M1803 result=%s\n' "${result}"
