#!/usr/bin/env bash
set -euo pipefail

task_review="reviews/m111_w384_signed24_accumulator_independent_hammer_r1_20260824"
task_run="$task_review/vcs_run"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_rtl="rtl_m111/m111_w384_signed24_accumulator_frontend.sv"
task_expected_rtl="354e0de95ee4380098c09fac67af3e137b3ab8bb9f88ac706d62fe201179b43a"

mkdir -p "$task_run"
task_observed_rtl="$(sha256sum "$task_rtl" | awk '{print $1}')"
if [[ "$task_observed_rtl" != "$task_expected_rtl" ]]; then
    echo "M111 independent exact-SHA RTL mismatch" >&2
    exit 10
fi
printf 'path=%s expected=%s observed=%s\n' \
    "$task_rtl" "$task_expected_rtl" "$task_observed_rtl" \
    > "$task_run/production_rtl_sha_check.txt"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -Mdir="$task_run/csrc" -f "$task_review/independent_vcs.f" \
    -top tb_m111_independent_hammer -o "$task_run/simv" \
    > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi
set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -q '^PASS M111 INDEPENDENT HAMMER ' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 31
fi
printf '%s\n' 'status=PASS_M111_INDEPENDENT_COMMERCIAL_VCS_SVA' \
    'production_modified=false' 'docs_359_modified=false' \
    'scheduled_cycle_ratio=false' 'physical_speedup=false' \
    'system_speedup=false' 'headline=false' > "$task_run/RUN_COMPLETE.txt"
