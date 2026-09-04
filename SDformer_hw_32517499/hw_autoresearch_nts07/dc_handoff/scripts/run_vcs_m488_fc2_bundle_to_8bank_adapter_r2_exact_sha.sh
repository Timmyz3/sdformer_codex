#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m488_fc2_bundle_to_8bank_adapter_directed_vcs_r2b_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "$task_run" ]] || exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" >"$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m488/m488_fc2_bundle_to_8bank_adapter.sv"]="b9024112bb3e3f27ebed60c92437aa136a23fd954568c89413e05724931d4c1b"
 ["verif_m488/m488_fc2_bundle_to_8bank_adapter_assertions.sv"]="fd49748432a286a44ddc99ec58e4ca0d8bcbbf33a17e3ce1d642308d257946ba"
 ["tb_m488/tb_m488_fc2_bundle_to_8bank_adapter.sv"]="d0dbd2411361aacfa2b7d876945bf4e999c7274fc3596bc1f5ad24a26878a65f"
 ["dc_handoff/filelists/date_m488_fc2_bundle_to_8bank_adapter_directed_vcs.f"]="b2cca114de212844f708b7a08ffe37e8e07d77cb1cd125841bbf5d9ce0631fcf"
 ["contracts/m488_fc2_bundle_to_8bank_adapter_directed_vcs_contract_r2_20260827.json"]="a88b0d1fd6460aec2ec0758a0578dc20796430db2919312d31a4496dee74a567"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: >"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
 task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' \
  "$task_path" "${task_expected[$task_path]}" "$task_observed" \
  >>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" >"$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
 +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
 -Mdir="$task_run/csrc" \
 -f dc_handoff/filelists/date_m488_fc2_bundle_to_8bank_adapter_directed_vcs.f \
 -top tb_m488_fc2_bundle_to_8bank_adapter \
 -o "$task_run/simv" >"$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/compile.rc"
[[ $task_rc -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=48820260827 -no_save -cm assert \
 -assert report="$task_run/assert.report" >"$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" >"$task_run/sim.rc"
[[ $task_rc -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
 "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true
grep -Eq 'PASS M488 bundle-to-8bank adapter requests=98 bank_beats=341 partial=[1-9][0-9]* request_stalls=[1-9][0-9]* response_stalls=[1-9][0-9]* out_of_order=[1-9][0-9]* attack=1 cycles=[1-9][0-9]* headline=false system_speedup=false' \
 "$task_run/sim.log" || exit 30

for task_cover in \
 cp_full_eight_bank_request cp_partial_request_distribution \
 cp_pending_request_stall cp_eight_responses_same_cycle \
 cp_out_of_order_bundle_response cp_core_response_stall \
 cp_retire_then_slot_reuse cp_protocol_attack; do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" \
  "$task_run/assert.report" || exit 31
done

python3 - "$task_run" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
text = (root / "sim.log").read_text()
match = re.search(
    r"PASS M488 bundle-to-8bank adapter requests=(\d+) bank_beats=(\d+) "
    r"partial=(\d+) request_stalls=(\d+) response_stalls=(\d+) "
    r"out_of_order=(\d+) attack=1 cycles=(\d+)", text)
if not match:
    raise SystemExit("M488 R2 PASS line not found")
requests, beats, partial, req_stalls, rsp_stalls, ooo, cycles = map(
    int, match.groups())
receipt = {
    "schema": "m488_fc2_bundle_to_8bank_adapter_vcs_receipt_v2",
    "status": "PASS_M488_FC2_BUNDLE_TO_8BANK_ADAPTER_R2_EXACT_VCS",
    "exact_sha": True,
    "tool": "Synopsys VCS V-2023.12-SP1",
    "bundle_requests": requests,
    "bank_request_beats": beats,
    "bank_response_beats": beats,
    "partial_request_distributions": partial,
    "request_stalls": req_stalls,
    "response_stalls": rsp_stalls,
    "out_of_order_bundle_responses": ooo,
    "cycles": cycles,
    "retire_then_slot_reuse_covered": True,
    "same_cycle_slot_reuse": False,
    "combinational_loop_removed_by_construction": True,
    "stale_response_attacks": 1,
    "numeric_mismatches": 0,
    "identity_mismatches": 0,
    "assertion_failures": 0,
    "m342_k8_integration": False,
    "m349_k1x8_integration": False,
    "dc": False,
    "system_speedup": False,
    "headline": False,
    "paper_ppa_ready": False,
}
(root / "m488_vcs_receipt_r2.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(root / "m488_vcs_receipt_r2.txt").write_text("\n".join([
    f"{key}={str(value).lower() if isinstance(value, bool) else value}"
    for key, value in receipt.items() if not isinstance(value, dict)
]) + "\n")
PY

sha256sum "$task_runner" >"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
 >"$task_run/SHA256SUMS"
echo PASS_M488_FC2_BUNDLE_TO_8BANK_ADAPTER_R2_EXACT_VCS \
 >"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M488 R2 exact VCS sealed at $task_run"
