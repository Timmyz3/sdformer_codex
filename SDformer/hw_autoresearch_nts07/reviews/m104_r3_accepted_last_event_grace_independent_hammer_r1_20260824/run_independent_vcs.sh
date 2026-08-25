#!/usr/bin/env bash
set -euo pipefail

task_review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review_dir/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_rtl="rtl_m104/m104_held_weight_correction_broadcaster.sv"
task_rtl_sha="7ea7978f431e917ee1a7835b8474af59e8f294587b1f115441388de8fb9c1ec5"

cd "$task_hw_root"
task_observed="$(sha256sum "$task_rtl" | awk '{print $1}')"
if [[ "$task_observed" != "$task_rtl_sha" ]]; then
    echo "M104 r3 independent RTL SHA mismatch" >&2
    exit 10
fi

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    -timescale=1ns/1ps -cm assert \
    -Mdir="$task_review_dir/csrc" \
    -f "$task_review_dir/filelist.f" \
    -top tb_m104_r3_independent_hammer \
    -o "$task_review_dir/simv" \
    > "$task_review_dir/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_review_dir/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_review_dir/simv" ]]; then
    exit 20
fi

set +e
"$task_review_dir/simv" -no_save \
    -assert report="$task_review_dir/assert.report" -cm assert \
    > "$task_review_dir/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_review_dir/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then
    exit 30
fi
grep -qx 'PASS M104 r3 independent VCS last_linger=1 nonlast_linger=1 between_edge_low_high=1 identity_mutations=5 older_buffer_quarantine=1 sticky_checks=11 reset_recoveries=10 accepted_events=10 macros=0' \
    "$task_review_dir/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_review_dir/sim.raw.log" "$task_review_dir/assert.report"; then
    exit 31
fi

echo 'status=PASS_M104_R3_INDEPENDENT_VCS_SVA' > "$task_review_dir/VCS_RUN_COMPLETE.txt"
