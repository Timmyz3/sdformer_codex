#!/usr/bin/env bash
set -euo pipefail

m523_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m523_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m523_hw="$(cd "${m523_dc_root}/.." && pwd)"
m523_canonical="${m523_hw}/results/m523_c2d_k8_polyphase_tap_bundler_vcs_r2_20260827"
m523_attempt="${m523_hw}/results/.m523_c2d_k8_polyphase_tap_bundler_vcs_r2_attempt_consumed"
m523_work="${m523_hw}/results/.m523_c2d_k8_polyphase_tap_bundler_vcs_r2_work.$$"
m523_vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
m523_rtl=rtl_m523/m523_c2d_k8_polyphase_tap_bundler.sv
m523_sva=verif_m523/m523_c2d_k8_polyphase_tap_bundler_assertions.sv
m523_tb=tb_m523/tb_m523_c2d_k8_polyphase_tap_bundler.sv
m523_filelist=dc_handoff/filelists/date_m523_c2d_k8_polyphase_tap_bundler_directed_vcs.f
m523_contract=contracts/m523_c2d_k8_polyphase_tap_bundler_vcs_contract_draft_r3_20260827.json
m523_review_dir="${m523_hw}/reviews/m523_c2d_k8_bundler_runner_static_hammer_r2_20260827"
m523_review_json="${m523_review_dir}/m523_c2d_k8_bundler_runner_static_hammer_r2_20260827.json"
m523_review_md="${m523_review_dir}/m523_c2d_k8_bundler_runner_static_hammer_r2_20260827.md"
m523_failure_review_dir="${m523_hw}/reviews/m523_vcs_exit31_failure_receipt_hammer_r1_20260827"
m523_failure_review_json="${m523_failure_review_dir}/m523_vcs_exit31_failure_receipt_hammer_r1_20260827.json"
m523_expected_failure_review_outer="${M523_R2_EXPECTED_FAILURE_REVIEW_OUTER_SEAL_SHA256:-}"
m523_expected_runner="${M523_R2_EXPECTED_RUNNER_SHA256:-}"
m523_observed_runner="$(sha256sum "${m523_runner}" | awk '{print $1}')"

# Caller self-SHA is the first admission gate and precedes every result,
# attempt, quarantine, review, resource, or Synopsys side effect.  Exit 10 is
# reserved for the automatic wrong-runner negative preflight.
[[ "${m523_expected_runner}" =~ ^[0-9a-f]{64}$ && \
   "${m523_observed_runner}" == "${m523_expected_runner}" ]] || exit 10

m523_sha() { sha256sum "$1" | awk '{print $1}'; }
m523_expect_regular() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && \
       "$(m523_sha "${path}")" == "${expected}" ]]
}

m523_verify_failure_review() {
    [[ "${m523_expected_failure_review_outer}" == \
        b3c2ec802dc053c84e1154369ee23045724f303585a9ebea3260022b6b96b0ad ]]
    [[ -d "${m523_failure_review_dir}" && ! -L "${m523_failure_review_dir}" ]]
    [[ "$(find -P "${m523_failure_review_dir}" -mindepth 1 -maxdepth 1 \
        -printf '%y %f\n' | LC_ALL=C sort)" == \
        $'f RUN_COMPLETE\nf SHA256SUMS\nf SHA256SUMS.seal.sha256\nf m523_vcs_exit31_failure_receipt_hammer_r1_20260827.json\nf m523_vcs_exit31_failure_receipt_hammer_r1_20260827.md\nf mechanical_evidence.txt' ]]
    [[ "$(m523_sha "${m523_failure_review_dir}/SHA256SUMS.seal.sha256")" \
        == "${m523_expected_failure_review_outer}" ]]
    (
        cd "${m523_failure_review_dir}"
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
    python3 - "${m523_failure_review_json}" <<'PY'
import json
import math
import sys
from pathlib import Path

def reject(value):
    raise ValueError("non-finite JSON constant: " + value)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            finite(key)
            finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)

review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),
                    parse_constant=reject)
finite(review)
assert review["schema"] == "m523_vcs_exit31_failure_receipt_hammer_v1"
assert review["status"] == \
    "FAILURE_DIAGNOSTIC_CONFIRMED__FUNCTIONAL_PASS_UNADMITTED__TERMINAL_COVER_SETTLEMENT_RACE__NEW_IDENTITY_REQUIRED"
assert review["p0_count"] == 0
repair = review["minimum_repair"]
assert repair["rtl_change_required"] is False
assert repair["sva_change_required"] is False
assert repair["old_attempt_reuse_forbidden"] is True
assert repair["old_quarantine_promotion_forbidden"] is True
decision = review["decision"]
for key in (
    "old_run_admitted", "old_run_citable",
    "old_runner_reexecution_authorized", "new_runner_authorized",
    "vcs_authorized_now", "dc_authorized", "formality_authorized",
    "pt_or_ptpx_authorized", "functional_claim_authorized",
    "performance_or_energy_claim_authorized", "direct_c2_claim_authorized",
    "ppa_claim_authorized", "system_or_headline_claim_authorized",
):
    assert decision[key] is False
PY
}

m523_verify_static_review() {
    local outer_expected="${M523_R2_EXPECTED_STATIC_REVIEW_OUTER_SEAL_SHA256:-}"
    [[ "${outer_expected}" =~ ^[0-9a-f]{64}$ ]]
    [[ -d "${m523_review_dir}" && ! -L "${m523_review_dir}" ]]
    [[ -f "${m523_review_json}" && ! -L "${m523_review_json}" && \
       -f "${m523_review_md}" && ! -L "${m523_review_md}" && \
       -f "${m523_review_dir}/RUN_COMPLETE" && \
       ! -L "${m523_review_dir}/RUN_COMPLETE" && \
       -f "${m523_review_dir}/SHA256SUMS" && \
       ! -L "${m523_review_dir}/SHA256SUMS" && \
       -f "${m523_review_dir}/SHA256SUMS.seal.sha256" && \
       ! -L "${m523_review_dir}/SHA256SUMS.seal.sha256" ]]
    [[ "$(m523_sha "${m523_review_dir}/SHA256SUMS.seal.sha256")" \
        == "${outer_expected}" ]]
    [[ "$(find -P "${m523_review_dir}" -mindepth 1 -maxdepth 1 \
        -printf '%y %f\n' | LC_ALL=C sort)" == \
        $'f RUN_COMPLETE\nf SHA256SUMS\nf SHA256SUMS.seal.sha256\nf m523_c2d_k8_bundler_runner_static_hammer_r2_20260827.json\nf m523_c2d_k8_bundler_runner_static_hammer_r2_20260827.md' ]]
    (
        cd "${m523_review_dir}"
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
    python3 - "${m523_review_dir}" "${m523_observed_runner}" \
        "$(m523_sha "${m523_contract}")" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
runner_sha = sys.argv[2]
contract_sha = sys.argv[3]

def reject(value):
    raise ValueError("non-finite JSON constant: " + value)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            finite(key)
            finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)

review = json.loads((root /
    "m523_c2d_k8_bundler_runner_static_hammer_r2_20260827.json").read_text(
        encoding="utf-8"), parse_constant=reject)
finite(review)
assert review["schema"] == "m523_c2d_k8_bundler_runner_static_hammer_v2"
assert review["status"] == "STATIC_GO__EXACT_SHA_ONE_SHOT_VCS_AUTHORIZED"
assert review["p0_count"] == 0
assert review["candidate_identity"]["runner"]["path"] == \
    "dc_handoff/scripts/run_vcs_m523_c2d_k8_polyphase_tap_bundler_r2_exact_sha.sh"
assert review["candidate_identity"]["runner"]["sha256"] == runner_sha
assert review["candidate_identity"]["contract"]["sha256"] == contract_sha
decision = review["decision"]
assert decision["authorized_runner_sha256"] == runner_sha
assert decision["authorized_runner_invocations"] == 1
assert decision["vcs_authorized"] is True
assert decision["dc_authorized"] is False
assert decision["formality_authorized"] is False
assert decision["pt_or_ptpx_authorized"] is False
assert decision["performance_or_energy_claim_authorized"] is False
assert decision["system_or_headline_claim_authorized"] is False
assert decision["direct_c2_claim_authorized"] is False
assert decision["ppa_claim_authorized"] is False
PY
}

m523_verify_inputs() {
    m523_expect_regular "${m523_runner}" "${m523_observed_runner}"
    m523_expect_regular "${m523_vcs}" \
        0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
    m523_expect_regular "${m523_rtl}" \
        ad6def7cd81e5f3cd1570ef23fd062da19ee8b2a35498d6deca1c010522a0920
    m523_expect_regular "${m523_sva}" \
        91360568b5b105f9bd86ee3d615e87a848b42b8aafc522c0a5546abaeaf77f7c
    m523_expect_regular "${m523_tb}" \
        3b468b7247ddbb0f292a653ba15f0021ca5926354128858208ccf6147d6ff5cc
    m523_expect_regular "${m523_filelist}" \
        f2cc54336820235f0585cee0d790044681485696f87c1c829b1e42bfe5acf8d0
    m523_expect_regular "${m523_contract}" \
        6dac33f9fe035c0ed1c14ddd7dbc7d9ebfabcdec279cf027ce07cf0774baa415
    m523_expect_regular docs/359_DATE终局冻结_20260813.md \
        dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
    m523_verify_failure_review
    m523_verify_static_review
}

m523_resource_gate() {
    local log=$1 failures=0
    : >"${log}"
    for sample in 1 2 3; do
        local limit committed available swap headroom failcnt under_oom oom_kill
        limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
        committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
        headroom=$((limit - committed))
        failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
        under_oom=$(awk '/^under_oom / {print $2}' \
            /sys/fs/cgroup/memory/user.slice/memory.oom_control)
        oom_kill=$(awk '/^oom_kill / {print $2}' \
            /sys/fs/cgroup/memory/user.slice/memory.oom_control)
        printf 'sample=%s timestamp=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s failcnt=%s under_oom=%s oom_kill=%s\n' \
            "${sample}" "$(date --iso-8601=seconds)" "${headroom}" \
            "${available}" "${swap}" "${failcnt}" "${under_oom}" \
            "${oom_kill}" >>"${log}"
        if [[ "${headroom}" -lt 33554432 || \
              "${available}" -lt 134217728 || \
              "${swap}" -lt 33554432 || "${failcnt}" -ne 0 || \
              "${under_oom}" -ne 0 || "${oom_kill}" -ne 0 ]]; then
            failures=$((failures + 1))
        fi
        if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
                pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null || \
                pgrep -x vcs >/dev/null || pgrep -x vcs1 >/dev/null || \
                pgrep -x vlogan >/dev/null || pgrep -x vhdlan >/dev/null || \
                pgrep -f '/common_shell_exec -shell (dc_shell|fm_shell|pt_shell) ' \
                    >/dev/null; then
            printf 'sample=%s forbidden_process=synopsys_eda\n' "${sample}" \
                >>"${log}"
            failures=$((failures + 1))
        fi
        local pid owner state cpu rss current_user
        current_user=$(id -un)
        while read -r pid; do
            [[ -n "${pid}" ]] || continue
            owner=$(ps -o user= -p "${pid}" | xargs)
            state=$(ps -o stat= -p "${pid}" | xargs)
            cpu=$(ps -o pcpu= -p "${pid}" | xargs)
            rss=$(ps -o rss= -p "${pid}" | xargs)
            if [[ "${owner}" == "${current_user}" || ! "${state}" =~ ^[SI] ]] || \
                    ! awk -v cpu="${cpu}" -v rss="${rss}" \
                        'BEGIN {exit !(cpu <= 0.5 && rss <= 262144)}'; then
                printf 'sample=%s forbidden_simv pid=%s owner=%s stat=%s pcpu=%s rss_kib=%s\n' \
                    "${sample}" "${pid}" "${owner}" "${state}" "${cpu}" \
                    "${rss}" >>"${log}"
                failures=$((failures + 1))
            else
                printf 'sample=%s allowed_foreign_idle_simv pid=%s owner=%s stat=%s pcpu=%s rss_kib=%s\n' \
                    "${sample}" "${pid}" "${owner}" "${state}" "${cpu}" \
                    "${rss}" >>"${log}"
            fi
        done < <(pgrep -x simv || true)
        if pgrep -f '(^|[ /])[^ ]*(analyze|independent|sweep|dse|simulate)_m[0-9][^ ]*[.]py( |$)' \
                >/dev/null; then
            printf 'sample=%s forbidden_process=project_cpu_dse\n' "${sample}" \
                >>"${log}"
            failures=$((failures + 1))
        fi
        [[ "${sample}" -eq 3 ]] || sleep 5
    done
    [[ "${failures}" -eq 0 ]]
}

m523_require_no_match() {
    local pattern=$1
    shift
    local rc
    set +e
    grep -Eiq -- "${pattern}" "$@"
    rc=$?
    set -e
    [[ "${rc}" -eq 1 ]]
}

m523_verify_run_tree() {
    python3 - "$1" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def reject(value):
    raise ValueError("non-finite JSON constant: " + value)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            finite(key)
            finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)

assert root.is_dir() and not root.is_symlink()
manifest = root / "SHA256SUMS"
outer = root / "SHA256SUMS.seal.sha256"
assert manifest.is_file() and not manifest.is_symlink()
assert outer.is_file() and not outer.is_symlink()
outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
assert len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == outer_fields[0]

entries = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    normalized = Path(name).as_posix()
    assert normalized not in entries and not normalized.startswith("/")
    assert ".." not in Path(normalized).parts
    path = root / normalized
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    entries[normalized] = digest
actual_regular = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and not path.is_symlink()
    and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
}
assert set(entries) == actual_regular

topology = json.loads((root / "TOPOLOGY.json").read_text(encoding="utf-8"),
                      parse_constant=reject)
links = json.loads((root / "VCS_SYMLINKS.json").read_text(encoding="utf-8"),
                   parse_constant=reject)
finite(topology)
finite(links)
assert topology["schema"] == "m523_vcs_output_topology_v2"
assert topology["symlink_profile"] == "historical_vcs_exact2"
assert set(topology["regular_files"]) == actual_regular
assert links["schema"] == "m523_vcs_symlink_inventory_v2"
assert links["profile"] == "historical_vcs_exact2"
assert len(links["links"]) == 2
observed = {}
for path in root.rglob("*"):
    if path.is_symlink():
        rel = path.relative_to(root).as_posix()
        raw = os.readlink(path)
        resolved = path.resolve(strict=True)
        assert resolved.is_file() and not resolved.is_symlink()
        try:
            resolved_relative = resolved.relative_to(root.resolve())
        except ValueError:
            raise AssertionError("M523 external result-tree symlink: " + rel)
        observed[rel] = {
            "path": rel,
            "raw_target": raw,
            "resolved_relative_path": resolved_relative.as_posix(),
            "resolved_target_sha256": hashlib.sha256(
                resolved.read_bytes()).hexdigest(),
        }
assert observed == {item["path"]: item for item in links["links"]}
archive = [name for name in observed
           if re.fullmatch(r"csrc/_[0-9]+_archive_1[.]so", name)]
shape = "simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml"
assert len(archive) == 1 and shape in observed
assert observed[shape]["raw_target"] == \
    "../../common/assert.verilog.shape.xml"
archive_name = Path(archive[0]).name
assert observed[archive[0]]["raw_target"] == \
    ".//../simv.daidir//" + archive_name
PY
}

cd "${m523_hw}"
m523_verify_inputs
[[ ! -e "${m523_canonical}" && ! -e "${m523_attempt}" && \
   ! -e "${m523_work}" ]] || exit 4

mkdir "${m523_work}"
m523_complete=0
m523_attempt_live=0
m523_preflight_quarantine="${m523_canonical}.preflight_failed.$$.quarantine"
m523_failed_quarantine="${m523_canonical}.failed_or_incomplete.$$.quarantine"
m523_cleanup() {
    local rc=$?
    if [[ "${m523_complete}" -ne 1 ]]; then
        local failed_path=""
        if [[ -d "${m523_work}" ]]; then
            failed_path="${m523_work}"
        elif [[ -d "${m523_canonical}" ]]; then
            failed_path="${m523_canonical}"
        fi
        if [[ -n "${failed_path}" ]]; then
            printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
                "${rc}" >"${failed_path}/RUN_FAILED_OR_INCOMPLETE.txt"
            if [[ "${m523_attempt_live}" -eq 0 ]]; then
                [[ ! -e "${m523_preflight_quarantine}" ]]
                mv -T "${failed_path}" "${m523_preflight_quarantine}"
            else
                [[ ! -e "${m523_failed_quarantine}" ]]
                mv -T "${failed_path}" "${m523_failed_quarantine}"
            fi
        fi
    fi
    return "${rc}"
}
trap m523_cleanup EXIT

cd "${m523_hw}"
m523_resource_gate "${m523_work}/resource_preflight.log"
m523_verify_inputs

# Automatic wrong-runner negative: the child must exit 10 at the first gate,
# before creating a work/attempt/canonical path or querying VCS.
mkdir "${m523_work}/wrong_runner_preflight"
m523_wrong_sha=0000000000000000000000000000000000000000000000000000000000000000
set +e
M523_R2_EXPECTED_RUNNER_SHA256="${m523_wrong_sha}" \
M523_R2_EXPECTED_STATIC_REVIEW_OUTER_SEAL_SHA256="${M523_R2_EXPECTED_STATIC_REVIEW_OUTER_SEAL_SHA256}" \
M523_R2_EXPECTED_FAILURE_REVIEW_OUTER_SEAL_SHA256="${M523_R2_EXPECTED_FAILURE_REVIEW_OUTER_SEAL_SHA256}" \
    "${m523_runner}" \
    >"${m523_work}/wrong_runner_preflight/child.stdout" \
    2>"${m523_work}/wrong_runner_preflight/child.stderr"
m523_negative_rc=$?
set -e
[[ "${m523_negative_rc}" -eq 10 && ! -e "${m523_canonical}" && \
   ! -e "${m523_attempt}" ]] || exit 11
python3 - "${m523_work}/wrong_runner_preflight" \
    "${m523_observed_runner}" "${m523_wrong_sha}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = {
    "schema": "m523_wrong_runner_preflight_receipt_v2",
    "status": "PASS_WRONG_RUNNER_SHA_EXIT10_NO_ATTEMPT_NO_VCS",
    "observed_runner_sha256": sys.argv[2],
    "supplied_wrong_runner_sha256": sys.argv[3],
    "child_exit_code": 10,
    "canonical_created": False,
    "attempt_consumed": False,
    "vcs_invocations": 0,
    "dc_authorized": False,
}
path = root / "m523_wrong_runner_preflight_receipt_v2.json"
path.write_text(json.dumps(receipt, allow_nan=False, indent=2,
                           sort_keys=True) + "\n", encoding="utf-8")
round_trip = json.loads(path.read_text(encoding="utf-8"))
assert round_trip == receipt
PY
printf 'PASS_WRONG_RUNNER_SHA_EXIT10_NO_ATTEMPT_NO_VCS\n' \
    >"${m523_work}/wrong_runner_preflight/RUN_COMPLETE.txt"
(
    cd "${m523_work}/wrong_runner_preflight"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

m523_verify_inputs
sha256sum "${m523_runner}" "${m523_vcs}" "${m523_rtl}" "${m523_sva}" \
    "${m523_tb}" "${m523_filelist}" "${m523_contract}" \
    docs/359_DATE终局冻结_20260813.md \
    >"${m523_work}/input_sha256.txt"
find -P "${m523_failure_review_dir}" -maxdepth 1 -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum \
    >>"${m523_work}/input_sha256.txt"
find -P "${m523_review_dir}" -maxdepth 1 -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum \
    >>"${m523_work}/input_sha256.txt"
cp "${m523_contract}" "${m523_work}/contract.json"

mkdir "${m523_work}/.attempt_staging"
printf 'status=CONSUMED_BEFORE_EXACT_VCS_ID_AND_COMPILE\ncanonical=%s\n' \
    "${m523_canonical}" \
    >"${m523_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m523_runner}" "${m523_vcs}" "${m523_rtl}" "${m523_sva}" \
    "${m523_tb}" "${m523_filelist}" "${m523_contract}" \
    docs/359_DATE终局冻结_20260813.md \
    "${m523_failure_review_dir}/SHA256SUMS.seal.sha256" \
    "${m523_review_dir}/SHA256SUMS.seal.sha256" \
    >"${m523_work}/.attempt_staging/identity.sha256"
(
    cd "${m523_work}/.attempt_staging"
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
mv -T "${m523_work}/.attempt_staging" "${m523_attempt}"
m523_attempt_live=1
[[ "$(find -P "${m523_attempt}" -mindepth 1 -maxdepth 1 -printf '%y %f\n' | \
    LC_ALL=C sort)" == \
    $'f ATTEMPT_CONSUMED.txt\nf SHA256SUMS\nf SHA256SUMS.seal.sha256\nf identity.sha256' ]]

export VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
export VCS_ARCH_OVERRIDE=linux
set +e
"${m523_vcs}" -full64 -ID >"${m523_work}/vcs_id.txt" 2>&1
m523_rc=$?
set -e
printf '%s\n' "${m523_rc}" >"${m523_work}/vcs_id.rc"
[[ "${m523_rc}" -eq 0 ]] || exit 19
grep -Fq 'V-2023.12-SP1' "${m523_work}/vcs_id.txt" || exit 19

set +e
"${m523_vcs}" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${m523_work}/csrc" -f "${m523_filelist}" \
    -top tb_m523_c2d_k8_polyphase_tap_bundler \
    -o "${m523_work}/simv" >"${m523_work}/compile.log" 2>&1
m523_rc=$?
set -e
printf '%s\n' "${m523_rc}" >"${m523_work}/compile.rc"
[[ "${m523_rc}" -eq 0 && -x "${m523_work}/simv" ]] || exit 20
m523_require_no_match \
    'Warning-\[|^[[:space:]]*Warning:|Error-\[|^[[:space:]]*Error|^[[:space:]]*Fatal|Fatal:' \
    "${m523_work}/compile.log" || exit 21

set +e
"${m523_work}/simv" +ntb_random_seed=523027 -no_save -cm assert \
    -assert report="${m523_work}/assert.report" \
    >"${m523_work}/sim.log" 2>&1
m523_rc=$?
set -e
printf '%s\n' "${m523_rc}" >"${m523_work}/sim.rc"
[[ "${m523_rc}" -eq 0 ]] || exit 22
m523_require_no_match \
    'failed at|Offending|^[[:space:]]*Error|^[[:space:]]*Fatal|Fatal:|watchdog|timeout' \
    "${m523_work}/sim.log" "${m523_work}/assert.report" || exit 23

m523_pass=$(grep -E '^PASS M523 events=6 bundles=8 taps=43 full8=4 tails1=1 stalls=[1-9][0-9]* replacements=[1-9][0-9]* boundaries=6 cross_event=2 tag_flush=1 time_flush=1 stream_flush=2 stream_iso=1 fifo_max=18 phases=6/10/10/17 protocol_attack=1$' \
    "${m523_work}/sim.log")
[[ "$(printf '%s\n' "${m523_pass}" | grep -c '^PASS M523')" -eq 1 ]] || \
    exit 30
m523_required_covers=(
    cp_full_eight_tap_bundle cp_one_tap_boundary_flush
    cp_cross_event_tail_fill cp_stream_last_flush cp_partial_bundle_flush
    cp_stall cp_same_edge_input_output cp_fifo_full
    cp_protocol_fault_during_busy cp_fault_drain_complete
)
for cover in "${m523_required_covers[@]}"; do
    grep -Eq "sva\\.${cover}, .* [1-9][0-9]* match" \
        "${m523_work}/assert.report" || exit 31
done

python3 - "${m523_work}" "${m523_pass}" "${m523_observed_runner}" \
    "$(m523_sha "${m523_contract}")" \
    "$(m523_sha "${m523_review_dir}/SHA256SUMS.seal.sha256")" \
    "$(m523_sha "${m523_failure_review_dir}/SHA256SUMS.seal.sha256")" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
match = re.fullmatch(
    r"PASS M523 events=(\d+) bundles=(\d+) taps=(\d+) full8=(\d+) "
    r"tails1=(\d+) stalls=(\d+) replacements=(\d+) boundaries=(\d+) "
    r"cross_event=(\d+) tag_flush=(\d+) time_flush=(\d+) "
    r"stream_flush=(\d+) stream_iso=(\d+) fifo_max=(\d+) "
    r"phases=(\d+)/(\d+)/(\d+)/(\d+) "
    r"protocol_attack=(\d+)", sys.argv[2])
if match is None:
    raise SystemExit("M523 pass-line parse failure")
values = [int(value) for value in match.groups()]
expected_fixed = [6, 8, 43, 4, 1]
if values[:5] != expected_fixed or values[5] < 1 or values[6] < 1 or \
        values[7:] != [6, 2, 1, 1, 2, 1, 18, 6, 10, 10, 17, 1]:
    raise SystemExit("M523 measured ledger drift: {!r}".format(values))
receipt = {
    "schema": "m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_v2",
    "status": "PASS_M523_EXACT_FUNCTIONAL_CROSS_EVENT_PACKING_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1 Full64",
    "seed": 523027,
    "identity": {
        "runner_sha256": sys.argv[3],
        "contract_r3_sha256": sys.argv[4],
        "authorizing_static_review_outer_seal_file_sha256": sys.argv[5],
        "failure_review_outer_seal_file_sha256": sys.argv[6],
        "wrong_runner_preflight_exit_code": 10,
    },
    "measured": {
        "events": values[0],
        "bundles": values[1],
        "taps": values[2],
        "full_eight_bundles": values[3],
        "one_tap_boundary_flushes": values[4],
        "stall_cycles": values[5],
        "same_edge_input_output": values[6],
        "event_boundaries": values[7],
        "cross_event_bundles": values[8],
        "tag_boundary_flushes": values[9],
        "time_boundary_flushes": values[10],
        "stream_last_flushes": values[11],
        "same_context_stream_last_isolations": values[12],
        "maximum_fifo_occupancy": values[13],
        "phase_counts_00_01_10_11": values[14:18],
        "protocol_attacks": values[18],
        "assertion_failures": 0,
    },
    "functional_scope": {
        "atomic_event_fanout_4_6_9": True,
        "same_tag_time_non_stream_cross_event_pack": True,
        "per_lane_event_boundary_exact": True,
        "tag_and_time_boundary_flush": True,
        "stream_last_crossing_forbidden": True,
        "accepted_work_fault_drain": True,
        "maximum_legal_size32_source31_destination63": True,
        "direct_m218_c2_bank_integration": False,
    },
    "claim_boundary": {
        "directed_functional_completeness": True,
        "frontend_bundle_reduction": False,
        "decoder_cycle_speedup": False,
        "dc": False,
        "formality": False,
        "sta": False,
        "power": False,
        "energy": False,
        "ppa": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
path = root / "m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_v2.json"
path.write_text(json.dumps(receipt, allow_nan=False, indent=2,
                           sort_keys=True) + "\n", encoding="utf-8")
round_trip = json.loads(path.read_text(encoding="utf-8"),
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError("non-finite JSON constant: " + value)))
if round_trip != receipt:
    raise SystemExit("M523 strict finite receipt round-trip drift")
PY

# Inventory the two VCS-generated symlink classes explicitly.  Symlinks are
# not manifest members; their paths/raw targets/in-tree resolved targets and
# resolved-target hashes are sealed in VCS_SYMLINKS.json instead.
python3 - "${m523_work}" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
links = []
for path in root.rglob("*"):
    if not path.is_symlink():
        continue
    rel = path.relative_to(root).as_posix()
    raw = os.readlink(path)
    resolved = path.resolve(strict=True)
    try:
        resolved_relative = resolved.relative_to(root)
    except ValueError:
        raise SystemExit("M523 external VCS symlink: " + rel)
    if not resolved.is_file() or resolved.is_symlink():
        raise SystemExit("M523 unsafe VCS symlink: " + rel)
    links.append({
        "path": rel,
        "raw_target": raw,
        "resolved_relative_path": resolved_relative.as_posix(),
        "resolved_target_sha256": hashlib.sha256(
            resolved.read_bytes()).hexdigest(),
    })
links.sort(key=lambda item: item["path"])
if len(links) != 2:
    raise SystemExit("M523 expected exact2 VCS symlinks, got {!r}".format(links))
archive = [item for item in links
           if re.fullmatch(r"csrc/_[0-9]+_archive_1[.]so", item["path"])]
shape_path = "simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml"
shape = [item for item in links if item["path"] == shape_path]
if len(archive) != 1 or len(shape) != 1:
    raise SystemExit("M523 VCS symlink profile drift")
if archive[0]["raw_target"] != ".//../simv.daidir//" + \
        Path(archive[0]["path"]).name:
    raise SystemExit("M523 archive symlink target drift")
if shape[0]["raw_target"] != "../../common/assert.verilog.shape.xml":
    raise SystemExit("M523 coverage symlink target drift")
inventory = {
    "schema": "m523_vcs_symlink_inventory_v2",
    "profile": "historical_vcs_exact2",
    "links": links,
}
(root / "VCS_SYMLINKS.json").write_text(
    json.dumps(inventory, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY

printf 'PASS_M523_EXACT_FUNCTIONAL_CROSS_EVENT_PACKING_VCS\n' \
    >"${m523_work}/RUN_COMPLETE.txt"
python3 - "${m523_work}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
regular = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and not path.is_symlink()
    and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256",
                          "TOPOLOGY.json"})
regular.append("TOPOLOGY.json")
topology = {
    "schema": "m523_vcs_output_topology_v2",
    "regular_files": sorted(regular),
    "symlink_profile": "historical_vcs_exact2",
    "symlink_inventory": "VCS_SYMLINKS.json",
    "manifest_excludes": ["SHA256SUMS", "SHA256SUMS.seal.sha256"],
    "manifest_symlink_policy": "Symlinks are excluded as files and represented exactly by the sealed VCS_SYMLINKS.json inventory.",
}
(root / "TOPOLOGY.json").write_text(
    json.dumps(topology, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
(
    cd "${m523_work}"
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

m523_verify_inputs
sha256sum -c "${m523_work}/input_sha256.txt" >/dev/null
(
    cd "${m523_attempt}"
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
sha256sum -c "${m523_attempt}/identity.sha256" >/dev/null
m523_verify_run_tree "${m523_work}"
[[ ! -e "${m523_canonical}" ]]
mv -T "${m523_work}" "${m523_canonical}"
m523_verify_run_tree "${m523_canonical}"
m523_complete=1
trap - EXIT
echo "PASS M523 r2 exact VCS sealed at ${m523_canonical}"
