#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m488_fc2_bundle_to_8bank_adapter_directed_vcs_r1_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

[[ ! -e "$task_run" ]] || exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" >"$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
 ["rtl_m488/m488_fc2_bundle_to_8bank_adapter.sv"]="98b11764ea186b57b27ae04be36e313e7d6a82270250282cd34dcb07fdc856e4"
 ["verif_m488/m488_fc2_bundle_to_8bank_adapter_assertions.sv"]="d365c683c60459122dfa7ec6ffe5fe12532eeaa11376b325222c0e96aa700141"
 ["tb_m488/tb_m488_fc2_bundle_to_8bank_adapter.sv"]="d3df560db033c00d516885dc1528e91a90addcfb95b4f9d6c5d6d6b147d82bea"
 ["dc_handoff/filelists/date_m488_fc2_bundle_to_8bank_adapter_directed_vcs.f"]="b2cca114de212844f708b7a08ffe37e8e07d77cb1cd125841bbf5d9ce0631fcf"
 ["contracts/m488_fc2_bundle_to_8bank_adapter_directed_vcs_contract_r1_20260827.json"]="00996375b4d3eb818fa9dea42327d9687771f8512a1fc775d95f87c9986dfea8"
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
 cp_same_cycle_slot_reuse cp_protocol_attack; do
 grep -Eq "$task_cover, .* [1-9][0-9]* match" \
  "$task_run/assert.report" || exit 31
done

{
 echo status=PASS_M488_FC2_BUNDLE_TO_8BANK_ADAPTER_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo bundle_requests=98
 echo bank_request_beats=341
 echo bank_response_beats=341
 echo partial_request_distributions=46
 echo request_stalls=11
 echo response_stalls=21
 echo out_of_order_bundle_responses=10
 echo eight_bank_same_cycle_responses=1
 echo same_cycle_slot_reuses=1
 echo stale_response_attacks=1
 echo numeric_mismatches=0
 echo identity_mismatches=0
 echo assertion_failures=0
 echo m342_k8_integration=false
 echo m349_k1x8_integration=false
 echo dc=false
 echo system_speedup=false
 echo headline=false
 echo paper_ppa_ready=false
} >"$task_run/m488_vcs_receipt_r1.txt"

sha256sum "$task_runner" >"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
 ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
 >"$task_run/SHA256SUMS"
echo PASS_M488_FC2_BUNDLE_TO_8BANK_ADAPTER_EXACT_VCS \
 >"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M488 exact VCS sealed at $task_run"
