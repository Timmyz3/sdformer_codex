#!/usr/bin/env bash
set -euo pipefail

review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "$review_dir/../.." && pwd)"
run_dir="$review_dir/vcs_adversarial_run_r3"
vcs_root="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
rtl_rel="rtl_m104/m104_held_weight_correction_broadcaster.sv"
expected_rtl_sha="37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a"

if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite independent M104 VCS run: $run_dir" >&2
    exit 2
fi
mkdir "$run_dir"
cd "$hw_root"

observed_rtl_sha="$(sha256sum "$rtl_rel" | awk '{print $1}')"
printf 'path=%s expected=%s observed=%s\n' \
    "$rtl_rel" "$expected_rtl_sha" "$observed_rtl_sha" \
    > "$run_dir/preflight_sha_checks.txt"
if [[ "$observed_rtl_sha" != "$expected_rtl_sha" ]]; then
    exit 10
fi
sha256sum "$rtl_rel" \
    "$review_dir/m104_independent_adversarial_assertions.sv" \
    "$review_dir/tb_m104_independent_adversarial.sv" \
    > "$run_dir/input_sha256.txt"

export VCS_HOME="$vcs_root" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$vcs_root/bin/vcs" -full64 -sverilog -assert svaext \
    -cm assert -cm_dir "$run_dir/simv.vdb" \
    -timescale=1ns/1ps \
    -Mdir="$run_dir/csrc" \
    "$rtl_rel" \
    "$review_dir/m104_independent_adversarial_assertions.sv" \
    "$review_dir/tb_m104_independent_adversarial.sv" \
    -top tb_m104_independent_adversarial \
    -o "$run_dir/simv" > "$run_dir/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$run_dir/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$run_dir/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$run_dir/compile.raw.log"; then exit 21; fi

set +e
"$run_dir/simv" -no_save -assert report="$run_dir/assert.report" \
    -cm assert -cm_dir "$run_dir/simv.vdb" \
    > "$run_dir/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$run_dir/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M104 independent adversarial VCS signed_codes=256 lanes=96 signs=2 ready_release_fault=1 sticky_cycles=3 reset_recovery=1 ii1_turnovers=4 load_gap=1 last_wait=1' "$run_dir/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$run_dir/sim.raw.log" "$run_dir/assert.report"; then
    exit 31
fi
for cover in \
        cp_illegal_plus_ready_release \
        cp_legal_stalled_last_then_release \
        cp_reset_recovery; do
    grep -Eq "$cover, .* [1-9][0-9]* match" "$run_dir/assert.report"
done

{
    echo 'status=PASS_M104_INDEPENDENT_ADVERSARIAL_VCS_SVA'
    echo 'production_rtl_exact_sha=true'
    echo 'signed_int8_codes=256'
    echo 'signed_output_lanes=96'
    echo 'positive_and_negative=true'
    echo 'same_cycle_invalid_plus_ready_release_quarantine=true'
    echo 'sticky_fault_reset_only=true'
    echo 'ready_release_turnover=true'
    echo 'last_wait_then_accept=true'
    echo 'load_gaps_between_three_beats=true'
    echo 'scheduled_cycle_speedup=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > "$run_dir/RUN_COMPLETE.txt"
sha256sum "$run_dir"/*.raw.log "$run_dir"/*.report \
    "$run_dir/RUN_COMPLETE.txt" > "$run_dir/output_sha256.txt"
echo "PASS M104 independent adversarial VCS/SVA"
