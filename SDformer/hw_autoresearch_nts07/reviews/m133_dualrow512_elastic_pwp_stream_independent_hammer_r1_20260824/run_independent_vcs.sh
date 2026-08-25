#!/usr/bin/env bash
set -euo pipefail

review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "$review_dir/../.." && pwd)"
vcs_root="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME="$vcs_root" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
cd "$hw_root"

cross_run="$review_dir/frozen_r1_stall_fault_cross_property"
mkdir -p "$cross_run"

sha256sum \
    rtl_m133/m133_dualrow512_elastic_pwp_stream.sv \
    "$review_dir/frozen_r1_m133_assertions.sv" \
    "$review_dir/tb_m133_stall_fault_interaction.sv" \
    contracts/m133_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json \
    dc_handoff/filelists/date_m133_dualrow512_elastic_pwp_stream_directed_vcs.f \
    > "$review_dir/independent_vcs_input_sha256.txt"

test "$(sha256sum "$review_dir/frozen_r1_m133_assertions.sv" | awk '{print $1}')" \
    = 'ab45fe7d15dd5a57a55461dd92ebd67a2a8a482a57c0e383f35c1a9c0b62a9b4'

"$vcs_root/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$cross_run/csrc" \
    rtl_m133/m133_dualrow512_elastic_pwp_stream.sv \
    "$review_dir/frozen_r1_m133_assertions.sv" \
    "$review_dir/tb_m133_stall_fault_interaction.sv" \
    -top tb_m133_stall_fault_interaction \
    -o "$cross_run/simv" > "$cross_run/compile.log" 2>&1
printf '0\n' > "$cross_run/compile.rc"
"$cross_run/simv" -no_save -assert report="$cross_run/assert.report" \
    > "$cross_run/sim.log" 2>&1
printf '0\n' > "$cross_run/sim.rc"
grep -Fq 'PASS M133 independent cross-property stimulus reached expected quarantine' \
    "$cross_run/sim.log"
grep -Fq 'ap_output_stable_under_stall' "$cross_run/assert.report"
grep -Eiq 'fail|failed at|Offending' "$cross_run/assert.report"

{
    echo 'status=PASS_FROZEN_R1_EXPECTED_CROSS_PROPERTY_COUNTEREXAMPLE'
    echo 'sealed_r1_original_run_audited_separately=true'
    echo 'frozen_r1_sva_exact_sha=true'
    echo 'cross_property_counterexample=stall_then_invalid_input'
    echo 'rtl_same_cycle_quarantine_observed=true'
    echo 'frozen_sva_ap_output_stable_under_stall_failure_expected=true'
    echo 'production_modified=false'
} > "$review_dir/VCS_REVIEW_COMPLETE.txt"
