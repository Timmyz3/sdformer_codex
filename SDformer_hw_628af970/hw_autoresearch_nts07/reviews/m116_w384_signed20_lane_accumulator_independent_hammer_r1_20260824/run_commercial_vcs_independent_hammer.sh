#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw="$(cd "$task_review/../.." && pwd)"
task_run="$task_review/vcs_run_r1"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite sealed independent M116 hammer: $task_run" >&2
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
task_core="rtl_m116/m116_w384_signed20_accumulator_frontend.sv"
task_adapter="rtl_m116/m116_w384_signed20_lane_sliced_accumulator_adapter.sv"
task_tb="reviews/m116_w384_signed20_lane_accumulator_independent_hammer_r1_20260824/tb_m116_signed20_independent_hammer.sv"
task_m115="contracts/m115_pwp_transient_accumulator_width_contract_r1_20260824.json"
task_m116="contracts/m116_w384_signed20_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
task_prod_receipt="dc_handoff/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_doc359="docs/359_DATE终局冻结_20260813.md"

declare -A task_expected=(
    ["$task_core"]="dd7e52e9ab3739972ca160283406c17f5d1a2947a3dd2456608a782b640c47b0"
    ["$task_adapter"]="074735e1f583d3dbef8e6dbee28f1ffb5a82bcda7a7328c8b520c5efc3c53a16"
    ["$task_tb"]="43d90648b4d4569accebb55bca493e9972deedc6e2db5d1443a2f1a00aca7053"
    ["$task_m115"]="ba730fcb6612fd8aa5c8e8c7d1aba976b759de54cbab05779ca409dadf9af9c8"
    ["$task_m116"]="bb245aa111d9646ff6b772c65a3362ae266d2f492691dac83d1782789912b721"
    ["$task_prod_receipt"]="0d29ccffbca08254487cc99a91d4bfd8005496f6b5fc9186bb63ebc21998f602"
    ["$task_doc359"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: > "$task_run/input_sha256.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf '%s  %s\n' "$task_observed" "$task_path" \
        >> "$task_run/input_sha256.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M116 independent exact-SHA mismatch: $task_path" >&2
        exit 10
    fi
done

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" \
    "$task_core" "$task_adapter" "$task_tb" \
    -top tb_m116_signed20_independent_hammer \
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
grep -q '^PASS M116 independent signed20 hammer commercial_vcs=true exact_source_sha=true ' \
    "$task_run/sim.raw.log"
grep -q 'negative_overflow_attacks=1' "$task_run/sim.raw.log"
grep -q 'foundry_macro=false ppa=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false$' \
    "$task_run/sim.raw.log"
if grep -Eiq '^Error|^Fatal|watchdog|failure|mismatch|corrupted|not suppressed' \
        "$task_run/sim.raw.log"; then
    exit 31
fi

{
    echo "status=PASS_M116_INDEPENDENT_SIGNED20_COMMERCIAL_VCS_HAMMER"
    echo "exact_source_sha=true"
    echo "tool=Synopsys_VCS_V-2023.12-SP1_Full64"
    echo "independent_testbench=true"
    echo "signed20_positive_negative_boundary=true"
    echo "rmw_and_nonconflicting_ii1=true"
    echo "lazy_clear=true"
    echo "commit_stall_and_order=true"
    echo "same_address_rdw_fail_closed=true"
    echo "positive_overflow_fail_closed=true"
    echo "negative_overflow_fail_closed=true"
    echo "lane_geometry=96x3072x20"
    echo "vector_bits=1920"
    echo "payload_bytes=737280"
    echo "saving_vs_signed24_bytes=147456"
    echo "checkpoint_only=true"
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
echo "PASS M116 independent commercial VCS hammer sealed"
