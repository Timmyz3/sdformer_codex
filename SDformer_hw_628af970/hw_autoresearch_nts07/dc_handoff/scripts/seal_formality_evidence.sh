#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 RUN_DIR ATTEMPT_TAG EXPECTED_PASSING_POINTS" >&2
    exit 2
fi

formal_run_dir="$(realpath "$1")"
formal_attempt_tag="$2"
formal_expected_points="$3"
formal_log="$formal_run_dir/formality_${formal_attempt_tag}.log"
formal_exit="$formal_run_dir/formality_${formal_attempt_tag}.exit_status"
formal_status="$formal_run_dir/reports/formality_status.txt"
formal_unmatched="$formal_run_dir/reports/formality_unmatched.rpt"
formal_verify="$formal_run_dir/reports/formality_verify.rpt"
formal_manifest="$formal_run_dir/formality_run_manifest.json"
formal_admission="$formal_run_dir/formality_admission_${formal_attempt_tag}.txt"
formal_ledger="$formal_run_dir/formality_evidence_${formal_attempt_tag}.sha256"
formal_script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
formal_python="${PYTHON_BIN:-python3}"
formal_machine_audit="$formal_run_dir/formality_machine_audit_${formal_attempt_tag}.json"

for formal_file in "$formal_log" "$formal_exit" "$formal_status" \
        "$formal_unmatched" "$formal_verify" "$formal_manifest"; do
    test -s "$formal_file"
done
if ! command -v "$formal_python" >/dev/null 2>&1; then
    echo "Formality evidence sealer Python interpreter is unavailable" >&2
    exit 3
fi
if [[ -e "$formal_admission" || -e "$formal_ledger" ]]; then
    echo "refusing to overwrite existing Formality sealed evidence" >&2
    exit 4
fi
test "$(tr -d '[:space:]' < "$formal_exit")" = "0"
test "$(tr -d '[:space:]' < "$formal_status")" = "PASS"
grep -q '^Verification SUCCEEDED$' "$formal_log"
grep -Eq "^[[:space:]]+${formal_expected_points} Passing compare points$" "$formal_log"
grep -Eq '^[[:space:]]*Failing \(not equivalent\)[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]+0$' "$formal_log"
grep -q '0(0) Unmatched reference(implementation) compare points' "$formal_log"

formal_design="$($formal_python - "$formal_manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], "r"))["design_name"])
PY
)"
if [[ "$formal_design" == "qfit_atlif_unified_t10_t2_stream_core" ]]; then
    if [[ -e "$formal_machine_audit" ]]; then
        echo "refusing to overwrite existing M31 Formality machine audit" >&2
        exit 5
    fi
    "$formal_python" \
        "$formal_script_root/scripts/audit_m31_r4_formality.py" \
        --run-dir "$formal_run_dir" --attempt "$formal_attempt_tag" \
        --expected-passing "$formal_expected_points" \
        --output "$formal_machine_audit"
    formal_unread_fields="$($formal_python - "$formal_machine_audit" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], "r"))
if result.get("status") != (
        "PASS_M31_R4_RTL_TO_FRESH_MAPPED_NETLIST_FORMALITY_STRICT"):
    raise SystemExit("M31 Formality machine audit status drift")
verification = result["verification"]
print("unread_reference_points={}".format(
    verification["unread_reference_points"]))
print("unread_implementation_points={}".format(
    verification["unread_implementation_points"]))
print("fmr_elab_147_diagnostics={}".format(
    verification["fmr_elab_147_diagnostics"]))
print("logic_simulator_disagreement_warnings={}".format(
    verification["logic_simulator_disagreement_warnings"]))
PY
)"
else
    formal_unread_fields="strict_m31_machine_audit=NOT_APPLICABLE"
fi

{
    echo "status=PASS_RTL_TO_MAPPED_NETLIST_FORMALITY"
    echo "attempt_tag=$formal_attempt_tag"
    echo "passing_compare_points=$formal_expected_points"
    echo "failing_compare_points=0"
    echo "unmatched_compare_points=0"
    printf '%s\n' "$formal_unread_fields"
} > "$formal_admission"

sha256sum "$formal_log" "$formal_exit" "$formal_status" \
    "$formal_unmatched" "$formal_verify" "$formal_manifest" \
    "$formal_admission" "$formal_script_root/run_formality.sh" \
    "$formal_script_root/scripts/run_formality.tcl" \
    "$formal_script_root/scripts/write_synopsys_run_manifest.py" \
    "$formal_script_root/scripts/seal_formality_evidence.sh" \
    > "$formal_ledger"
if [[ "$formal_design" == "qfit_atlif_unified_t10_t2_stream_core" ]]; then
    sha256sum "$formal_machine_audit" \
        "$formal_script_root/scripts/audit_m31_r4_formality.py" \
        "$formal_script_root/scripts/build_m31_r4_synopsys_receipt.py" \
        >> "$formal_ledger"
fi
sha256sum -c "$formal_ledger"
echo "FORMALITY_EVIDENCE_SEALED run=$formal_run_dir attempt=$formal_attempt_tag"
