#!/usr/bin/env bash
set -euo pipefail

m931_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m931_self="$(realpath "${BASH_SOURCE[0]}")"
m931_contract="${m931_hw_root}/contracts/m931_m912_c1_metadata_pipeline_macro_aware_dc_source_contract_r1_20260829.json"
m931_filelist="${m931_hw_root}/dc_handoff/filelists/date_m931_m912_c1_metadata_pipeline_macro_aware_dc.f"
m931_sdc="${m931_hw_root}/dc_handoff/constraints/date_m931_m912_c1_metadata_pipeline_macro_aware_3ns.sdc"
m931_tcl="${m931_hw_root}/dc_handoff/scripts/run_dc_m931_m912_c1_metadata_pipeline_macro_aware_candidate.tcl"
m931_result="${m931_hw_root}/dc_handoff/runs/m931_m912_c1_metadata_pipeline_macro_aware_dc_3p000ns_r1_20260829"
m931_attempt="${m931_hw_root}/dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_attempt_consumed"
m931_lock="${m931_hw_root}/dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_launch_lock"
m931_work="${m931_hw_root}/dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_work.$$"
m931_quarantine="${m931_result}.failed_or_incomplete.$$.quarantine"
m931_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m931_lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
m931_license=/opt/synopsys/Synopsys.dat
m931_std_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m931_std_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m931_macro_root=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821
m931_macro_manifest="${m931_macro_root}/SHA256SUMS"
m931_macro_slow="${m931_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
m931_macro_fast="${m931_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
m931_forbidden_macro_v="${m931_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
m931_docs="${m931_hw_root}/docs/359_DATE终局冻结_20260813.md"
m931_attempt_consumed=0
m931_lock_owned=0

m931_sha() { sha256sum "$1" | awk '{print $1}'; }
m931_verify_file() {
    local path=$1 expected=$2
    [[ -f "$path" && ! -L "$path" && "$(m931_sha "$path")" == "$expected" ]] || {
        echo "M931 identity mismatch: $path" >&2
        exit 3
    }
}
m931_verify_linkable_file() {
    local path=$1 expected=$2
    [[ -f "$path" && "$(m931_sha "$path")" == "$expected" ]] || {
        echo "M931 tool or library identity mismatch: $path" >&2
        exit 3
    }
}
m931_verify_file_seal() {
    local payload=$1 dir base
    dir="$(dirname "$payload")"; base="$(basename "$payload")"
    [[ -f "$payload.sha256" && -f "$payload.sha256.seal.sha256" ]] || return 1
    (cd "$dir" && sha256sum -c "$base.sha256" >/dev/null && \
        sha256sum -c "$base.sha256.seal.sha256" >/dev/null)
}
m931_verify_dir_seal() {
    local dir=$1
    [[ -d "$dir" && -f "$dir/SHA256SUMS" && -f "$dir/SHA256SUMS.seal.sha256" ]] || return 1
    (cd "$dir" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
m931_seal_dir() {
    local dir=$1
    (
        cd "$dir"
        find . -type l -print -quit | grep -q . && exit 1
        find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 \
            | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m931_collision_present() {
    local uid pid proc owner exe base comm
    uid="$(id -u)"
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ && "$pid" != "$$" ]] || continue
        proc="/proc/$pid"; [[ -d "$proc" ]] || continue
        owner="$(stat -c '%u' "$proc" 2>/dev/null || true)"
        [[ "$owner" == "$uid" ]] || continue
        exe="$(readlink -f "$proc/exe" 2>/dev/null || true)"; base="${exe##*/}"
        comm="$(tr -d '\n' <"$proc/comm" 2>/dev/null || true)"
        case "$base:$comm" in
            dc_shell:*|dc_shell-t:*|common_shell_exec:*|*:dc_shell|*:dc_shell-t|*:common_shell_exec|*:common_shell_exe) return 0 ;;
        esac
    done < <(ps -u "$uid" -o pid=)
    return 1
}
m931_cleanup() {
    local rc=$?
    set +e
    if [[ "$rc" -ne 0 && "$m931_attempt_consumed" -eq 1 && -d "$m931_work" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\n' "$rc" >"$m931_work/RUN_FAILED_OR_INCOMPLETE.txt"
        m931_seal_dir "$m931_work" || true
        mv -T -- "$m931_work" "$m931_quarantine" || true
    fi
    if [[ "$m931_lock_owned" -eq 1 ]]; then
        rmdir "$m931_lock" 2>/dev/null || true
        m931_lock_owned=0
    fi
    return "$rc"
}
trap m931_cleanup EXIT INT TERM

[[ "$#" -eq 0 ]] || exit 2
bash -n "$m931_self"
[[ -n "${M931_EXPECTED_RUNNER_SHA256:-}" && "$(m931_sha "$m931_self")" == "$M931_EXPECTED_RUNNER_SHA256" ]] || exit 3
[[ -n "${M931_EXPECTED_CONTRACT_SHA256:-}" ]] || exit 3
[[ -n "${M931_EXPECTED_REVIEW_DIR:-}" && -n "${M931_EXPECTED_REVIEW_SHA256:-}" ]] || exit 3
[[ -z "${OUTPUT_DIR:-}${CLOCK_PERIOD_NS:-}${LIB_DB:-}${MIN_LIB_DB:-}${MACRO_DB:-}${OPERATING_CONDITION:-}" ]] || exit 3
[[ ! -e "$m931_result" && ! -e "$m931_attempt" && ! -e "$m931_lock" ]] || exit 5

m931_verify_file "$m931_contract" "$M931_EXPECTED_CONTRACT_SHA256"
m931_verify_file_seal "$m931_contract"
python3 -m json.tool "$m931_contract" >/dev/null
jq -e '.schema == "m931_m912_c1_metadata_pipeline_macro_aware_dc_source_contract_v1"
       and .status == "SOURCE_ONLY__INDEPENDENT_HAMMER_AND_RELEASE_REQUIRED__NO_EDA_AUTHORIZED"
       and .authorization.run_dc_now == false
       and .claim_boundary.system_speedup == false
       and .claim_boundary.paper_ppa_ready == false
       and .physical_point.clock_period_ns == 3.0
       and .physical_point.expected_macro_count == 9' "$m931_contract" >/dev/null
while IFS=$'\t' read -r rel expected; do
    m931_verify_file "$m931_hw_root/$rel" "$expected"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "$m931_contract")
[[ "$(m931_sha "$m931_docs")" == dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 ]] || exit 3
[[ "$(m931_sha "$m931_macro_manifest")" == "$(jq -r '.foundry_views.macro_manifest_sha256' "$m931_contract")" ]] || exit 3
(cd "$m931_macro_root" && sha256sum -c SHA256SUMS >/dev/null)
[[ -f "$m931_forbidden_macro_v" ]] || exit 3
! rg -F "$(basename "$m931_forbidden_macro_v")" "$m931_filelist" >/dev/null || exit 3
m931_verify_linkable_file "$m931_dc" "$(jq -r '.tool_identity.dc_shell_sha256' "$m931_contract")"
m931_verify_linkable_file "$m931_lmutil" "$(jq -r '.tool_identity.lmutil_sha256' "$m931_contract")"
m931_verify_file "$m931_license" "$(jq -r '.tool_identity.license_file_sha256' "$m931_contract")"
m931_verify_file "$m931_std_slow" "$(jq -r '.foundry_views.std_slow_sha256' "$m931_contract")"
m931_verify_file "$m931_std_fast" "$(jq -r '.foundry_views.std_fast_sha256' "$m931_contract")"
m931_verify_file "$m931_macro_slow" "$(jq -r '.foundry_views.macro_slow_sha256' "$m931_contract")"
m931_verify_file "$m931_macro_fast" "$(jq -r '.foundry_views.macro_fast_sha256' "$m931_contract")"
m931_verify_dir_seal "$M931_EXPECTED_REVIEW_DIR"
[[ "$(m931_sha "$M931_EXPECTED_REVIEW_DIR/review.json")" == "$M931_EXPECTED_REVIEW_SHA256" ]] || exit 3
jq -e '.verdict == "PASS" and .score_100 >= 95 and .severity_counts.p0 == 0 and .severity_counts.p1 == 0
       and .decision.authorize_exactly_one_dc_attempt == true and .decision.eda_started_by_reviewer == false' \
    "$M931_EXPECTED_REVIEW_DIR/review.json" >/dev/null
[[ -x "$m931_dc" && -x "$m931_lmutil" ]] || exit 4
if m931_collision_present; then echo 'M931 same-UID Synopsys shell collision' >&2; exit 4; fi
mkdir "$m931_lock"
m931_lock_owned=1
"$m931_lmutil" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null
"$m931_lmutil" lmstat -c 27030@ic.ismd-nemo -f DC-Ultra-Opt >/dev/null
mkdir "$m931_attempt"
printf 'status=M931_ATTEMPT_CONSUMED\nmax_attempts=1\nretry=false\n' >"$m931_attempt/ATTEMPT_CONSUMED.txt"
m931_seal_dir "$m931_attempt"
m931_attempt_consumed=1
mkdir "$m931_work"

set +e
env LANG=C.UTF-8 LC_ALL=C.UTF-8 SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="$m931_license" \
    HW_ROOT="$m931_hw_root" RTL_FILELIST="$m931_filelist" SDC_FILE="$m931_sdc" OUTPUT_DIR="$m931_work" \
    STD_SLOW_DB="$m931_std_slow" STD_FAST_DB="$m931_std_fast" \
    MACRO_SLOW_DB="$m931_macro_slow" MACRO_FAST_DB="$m931_macro_fast" \
    "$m931_dc" -no_gui -f "$m931_tcl" 2>&1 | tee "$m931_work/dc.log"
m931_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$m931_rc" >"$m931_work/dc.rc"
[[ "$m931_rc" -eq 0 && -f "$m931_work/TCL_PASS_TERMINAL.txt" ]] || exit 6

python3 - "$m931_work" "$m931_contract" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
contract = pathlib.Path(sys.argv[2])
area = (root / "reports/area_hierarchy.rpt").read_text(errors="replace")
timing = (root / "reports/timing_setup.rpt").read_text(errors="replace")
qor = (root / "reports/qor.rpt").read_text(errors="replace")
area_m = re.findall(r"Total cell area:\s*([-+0-9.eE]+)", area)
slack_m = re.findall(r"slack \((?:MET|VIOLATED)\)\s*([-+0-9.eE]+)", timing)
mac = (root / "reports/macro_binding_audit.txt").read_text(errors="replace")
receipt = {
  "schema": "m931_m912_c1_metadata_pipeline_macro_aware_dc_raw_receipt_v1",
  "status": "RAW_DC_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED",
  "identity": {"source_contract_sha256": __import__('hashlib').sha256(contract.read_bytes()).hexdigest()},
  "physical_point": {"technology_nm": 28, "clock_period_ns": 3.0, "ideal_clock": True,
      "wireload": "ZeroWireload", "macro_count": 9, "cell_area_um2": float(area_m[-1]) if area_m else None,
      "worst_setup_slack_ns": min(map(float, slack_m)) if slack_m else None},
  "raw_checks": {"macro_binding_pass": "PASS_M931_RESOLVED_LIBRARY_MACRO_STRUCTURE" in mac,
      "tcl_terminal_pass": (root / "TCL_PASS_TERMINAL.txt").exists(), "qor_present": bool(qor)},
  "claim_boundary": {"independent_result_hammered": False, "setup_admitted": False,
      "hold_signoff": False, "power": False, "energy": False, "ppa": False,
      "speedup": False, "system_speedup": False, "paper_ppa_ready": False, "headline": False}
}
(root / "m931_m912_c1_metadata_pipeline_macro_aware_dc_raw_receipt_r1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
PY
printf 'status=RUN_COMPLETE_RAW_DC__RESULT_HAMMER_REQUIRED\n' >"$m931_work/RUN_COMPLETE.txt"
m931_seal_dir "$m931_work"
mv -T -- "$m931_work" "$m931_result"
m931_attempt_consumed=0
rmdir "$m931_lock"
m931_lock_owned=0
trap - EXIT INT TERM
printf 'M931_RAW_DC_COMPLETE result=%s\n' "$m931_result"
