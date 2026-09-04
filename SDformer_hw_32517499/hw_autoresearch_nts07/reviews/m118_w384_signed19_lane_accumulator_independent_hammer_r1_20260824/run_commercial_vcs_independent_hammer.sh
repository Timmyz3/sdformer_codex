#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw="$(cd "$task_review/../.." && pwd)"
task_run="$task_review/vcs_run_r1"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite independent M118 run: $task_run" >&2
    exit 2
fi
mkdir "$task_run"
task_complete=0
on_exit() {
    local task_rc="$?"
    if [[ "$task_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$task_rc"
        } > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$task_hw"
task_core="rtl_m118/m118_w384_signed19_accumulator_frontend.sv"
task_adapter="rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv"
task_tb="reviews/m118_w384_signed19_lane_accumulator_independent_hammer_r1_20260824/tb_m118_signed19_independent_hammer.sv"
task_m115_result="results/m115r2_pwp_prefix_coefficient_width_r1_20260824/m115r2_pwp_prefix_coefficient_width.json"
task_m115_contract="contracts/m115r2_pwp_prefix_coefficient_width_contract_r1_20260824.json"
task_m115_manifest="results/m115r2_pwp_prefix_coefficient_width_r1_20260824/SHA256SUMS.complete_r1.txt"
task_m118_contract="contracts/m118_w384_signed19_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_r2_receipt="dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r2_sealed_20260824/RUN_COMPLETE.txt"
task_r1_failure="dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r1_sealed_20260824/RUN_FAILED_OR_INCOMPLETE.txt"
task_doc359="docs/359_DATE终局冻结_20260813.md"

declare -A task_expected=(
    ["$task_core"]="0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482"
    ["$task_adapter"]="cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af"
    ["$task_tb"]="06ec7e609fa64723918b74fc691ef9ecabcc3f0fd129b7cf901f2ef2d14c7c84"
    ["$task_m115_result"]="b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708"
    ["$task_m115_contract"]="9edd6aac10186e24f21fffa5ce1b5a28da292258ad30df1d6934a7b1d1927eec"
    ["$task_m115_manifest"]="6b9af5e9e7de61edc770e1d4d738d6c0b0070e7947f6aec12633da7181f96326"
    ["$task_m118_contract"]="c79f55a15e03bbf26c22e9da2f0eb35d53b1a9795ab02b24a6b3c951c729903e"
    ["$task_r2_receipt"]="f45baa3c322a439377aa9c0c3e919440020294c9392b81343c7fae1bc1e605ff"
    ["$task_r1_failure"]="a1bbaa0205b4cbe7d793e5525ca93da242f0e14e11e64eb7383903559c0126a0"
    ["$task_doc359"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: > "$task_run/input_sha256.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf '%s  %s\n' "$task_observed" "$task_path" \
        >> "$task_run/input_sha256.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M118 independent exact-SHA mismatch: $task_path" >&2
        exit 10
    fi
done
grep -qx 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE' "$task_r1_failure"
grep -qx 'integrated_accepted_transaction_exact_once_miter=false' \
    "$task_r2_receipt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" "$task_core" "$task_adapter" "$task_tb" \
    -top tb_m118_signed19_independent_hammer \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" -no_save > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -q '^PASS M118 independent signed19 hammer commercial_vcs=true exact_source_sha=true ' \
    "$task_run/sim.raw.log"
grep -q 'negative_overflow_attacks=1' "$task_run/sim.raw.log"
grep -q 'mathematical_candidate=true integrated_exact_once=false' \
    "$task_run/sim.raw.log"
grep -q 'foundry_macro=false ppa=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false$' \
    "$task_run/sim.raw.log"
if grep -Eiq '^Error|^Fatal|watchdog|failure|mismatch|corrupted|not suppressed' \
        "$task_run/sim.raw.log"; then
    exit 31
fi

{
    echo "status=PASS_M118_INDEPENDENT_SIGNED19_COMMERCIAL_VCS_HAMMER"
    echo "exact_source_sha=true"
    echo "tool=Synopsys_VCS_V-2023.12-SP1_Full64"
    echo "independent_testbench=true"
    echo "signed19_positive_negative_boundary=true"
    echo "rmw_and_nonconflicting_ii1=true"
    echo "lazy_clear=true"
    echo "commit_stall_and_order=true"
    echo "same_address_rdw_fail_closed=true"
    echo "positive_overflow_fail_closed=true"
    echo "negative_overflow_fail_closed=true"
    echo "lane_geometry=96x3072x19"
    echo "vector_bits=1824"
    echo "payload_bytes=700416"
    echo "combined_logical_bytes=725416"
    echo "saving_vs_signed24_payload_bytes=184320"
    echo "m115r2_mathematical_candidate=true"
    echo "integrated_accepted_transaction_exact_once_miter=false"
    echo "r1_receipt_citable=false"
    echo "behavioral_macro=true"
    echo "foundry_macro=false"
    echo "ppa=false"
    echo "cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run/compile.raw.log" "$task_run/sim.raw.log" \
    "$task_run/RUN_COMPLETE.txt" > "$task_run/output_sha256.txt"
sha256sum "$task_review/run_commercial_vcs_independent_hammer.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M118 independent commercial VCS hammer sealed"
