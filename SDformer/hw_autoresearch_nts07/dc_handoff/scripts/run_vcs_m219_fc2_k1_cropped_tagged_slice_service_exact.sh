#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_hw_root/results/m219_fc2_k1_cropped_tagged_slice_service_directed_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M219 sealed VCS run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m219/m219_fc2_k1_cropped_tagged_slice_service_island.sv"]="75c4690ec04653084fb59fd75c5ba7ac329807975d76c9ffc43b6304bd4e1d47"
 ["verif_m219/m219_fc2_k1_cropped_tagged_slice_service_assertions.sv"]="378a81dcd9fc258dd568d8ee283be842b80d632c56315a9126cac074948bd93c"
 ["tb_m219/tb_m219_fc2_k1_cropped_tagged_slice_service_island.sv"]="a6e6bfcff24d959b5d574507368682f32f974006c0cddf9adf084db388fdadcc"
 ["dc_handoff/filelists/date_m219_fc2_k1_cropped_tagged_slice_service_rtl.f"]="2c07a9c9d9912698ce7e6d6870a443df9285400f6246e350a010cce22e472d7a"
 ["dc_handoff/filelists/date_m219_fc2_k1_cropped_tagged_slice_service_directed_vcs.f"]="658ab4f20240d5628b21dc13e5e3bf0c2467e9ffce664fb2cb32594f3418d012"
 ["contracts/m219_fc2_k1_cropped_tagged_slice_service_directed_vcs_contract_r1_20260825.json"]="8d9d74127a34492d277a659d6c9800d7932f7d5848899fe9d1a3da45940915e3"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m219_fc2_k1_cropped_tagged_slice_service_directed_vcs.f \
    -top tb_m219_fc2_k1_cropped_tagged_slice_service_island \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" \
    && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=219025 -no_save \
    -assert report="$task_run/assert.report" -cm assert \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true

grep -Fq 'PASS M219 cropped K1 directed numeric/protocol clean_groups=144 clean_requests=864 clean_responses=864 clean_contexts=864 clean_results=102 clean_done=5 clean_bank_reads=864' \
    "$task_run/sim.log" || exit 30
grep -Fq 'max_fifo=4 max_outstanding=8' "$task_run/sim.log" || exit 31
grep -Fq 'identity_attacks=4 duplicate_attacks=1 timeouts=1' \
    "$task_run/sim.log" || exit 32
grep -Eq 'slot_reuse=[1-9][0-9]* context_reuse=[1-9][0-9]* ooo_retirements=[1-9][0-9]* request_stalls=[1-9][0-9]* result_stalls=[1-9][0-9]*' \
    "$task_run/sim.log" || exit 33
grep -Eq 'cp_k1_request, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 34
grep -Eq 'cp_same_cycle_replace, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 35
grep -Eq 'cp_flush, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 36
grep -Eq 'cp_protocol_fault_rise, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 37

{
    echo status=PASS_M219_FC2_K1_CROPPED_TAGGED_SLICE_SERVICE_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo clean_groups=144
    echo clean_requests=864
    echo clean_responses=864
    echo clean_context_writes=864
    echo clean_result_beats=102
    echo clean_token_done=5
    echo clean_active_bank_reads=864
    echo numeric_mismatches=0
    echo conservation_mismatches=0
    echo native_128bit_k1_response=true
    echo retained_context_bits=18432
    echo maximum_fifo_occupancy=4
    echo maximum_outstanding=8
    echo identity_attacks=4
    echo delayed_duplicate_attacks=1
    echo flush_ack_timeout_attacks=1
    echo post_flush_b_zero_pollution=true
    echo standalone_sparse_fc2_update_service=true
    echo fair_area_sensitivity_baseline=false
    echo complete_fc2=false
    echo complete_ffn=false
    echo macro_aware_ppa=false
    echo physical_speedup=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/m219_fc2_k1_cropped_tagged_slice_service_vcs_receipt_r1.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$task_run/SHA256SUMS"
printf 'PASS_M219_FC2_K1_CROPPED_TAGGED_SLICE_SERVICE_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M219 exact VCS sealed at $task_run"
