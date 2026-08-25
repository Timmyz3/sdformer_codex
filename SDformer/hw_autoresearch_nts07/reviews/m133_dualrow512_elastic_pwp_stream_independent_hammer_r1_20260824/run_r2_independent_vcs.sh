#!/usr/bin/env bash
set -euo pipefail

review_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "$review_dir/../.." && pwd)"
vcs_root="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME="$vcs_root" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
cd "$hw_root"

prod_run="$review_dir/independent_r2_production_rerun"
cross_run="$review_dir/independent_r2_stall_fault_cross_property"
mkdir -p "$prod_run" "$cross_run"

declare -A expected=(
    ["rtl_m133/m133_dualrow512_elastic_pwp_stream.sv"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv"]="564fc8184977f352d4d841164583f0dc694ce8ba33fd3d2d6f871a3c2cbc6cea"
    ["tb_m133/tb_m133_dualrow512_elastic_pwp_stream.sv"]="3b73c0ea7d572382521e112a7962febe9c9733899b3a1ca30fa282b97708a742"
    ["dc_handoff/filelists/date_m133_dualrow512_elastic_pwp_stream_directed_vcs.f"]="575a3171e12b701f58709a68703a18eb0a4d111e215e7e4393921c2a4f347c31"
    ["contracts/m133r2_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json"]="75d827342d36a82318a29f3efe7149a87b73eeacd576f94c9c533d9cb4c2020f"
    ["contracts/m133_r1_stall_fault_composition_correction_r1_20260824.json"]="a32d3bab8faddf0a318c6ba6a3a1b36cb2ac579c08b3d327b56f4e599f59feff"
)
: > "$review_dir/independent_r2_input_sha256.txt"
for path in "${!expected[@]}"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    test "$observed" = "${expected[$path]}"
    printf '%s  %s\n' "$observed" "$path" \
        >> "$review_dir/independent_r2_input_sha256.txt"
done

"$vcs_root/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$prod_run/csrc" \
    -f dc_handoff/filelists/date_m133_dualrow512_elastic_pwp_stream_directed_vcs.f \
    -top tb_m133_dualrow512_elastic_pwp_stream \
    -o "$prod_run/simv" > "$prod_run/compile.log" 2>&1
printf '0\n' > "$prod_run/compile.rc"
"$prod_run/simv" -no_save -assert report="$prod_run/assert.report" \
    > "$prod_run/sim.log" 2>&1
printf '0\n' > "$prod_run/sim.rc"
grep -Fqx \
    'PASS M133r2 dualrow512 elastic PWP stream VCS vectors=105 outputs=105 beats=236 lanes=9696 escapes=4 ii_checks=63 stalls=43 long_stall=23 boundaries=5 protocol_attacks=2 stall_fault_overlap=1 reset_attacks=1 idle_payload=1 cycles_8_9_10_11=2_2_2_3 input_bits=512 bank_mapper=false macro=false physical_speedup=false system_speedup=false headline=false' \
    "$prod_run/sim.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal' \
        "$prod_run/sim.log" "$prod_run/assert.report"; then
    exit 21
fi
grep -Eq 'cp_stall_to_fault_quarantine, .* [1-9][0-9]* match' \
    "$prod_run/assert.report"

"$vcs_root/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$cross_run/csrc" \
    rtl_m133/m133_dualrow512_elastic_pwp_stream.sv \
    verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv \
    "$review_dir/tb_m133_stall_fault_interaction.sv" \
    -top tb_m133_stall_fault_interaction \
    -o "$cross_run/simv" > "$cross_run/compile.log" 2>&1
printf '0\n' > "$cross_run/compile.rc"
"$cross_run/simv" -no_save -assert report="$cross_run/assert.report" \
    > "$cross_run/sim.log" 2>&1
printf '0\n' > "$cross_run/sim.rc"
grep -Fq 'PASS M133 independent cross-property stimulus reached expected quarantine and 3-cycle sticky fault' \
    "$cross_run/sim.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal' \
        "$cross_run/sim.log" "$cross_run/assert.report"; then
    exit 31
fi
grep -Eq 'cp_stall_to_fault_quarantine, .* [1-9][0-9]* match' \
    "$cross_run/assert.report"

{
    echo 'status=PASS_M133R2_INDEPENDENT_PRODUCTION_AND_CROSS_PROPERTY_VCS'
    echo 'exact_sha=true'
    echo 'production_suite_pass=true'
    echo 'production_assertion_failures=false'
    echo 'independent_stall_fault_cross_pass=true'
    echo 'independent_three_cycle_sticky_fault_pass=true'
    echo 'independent_assertion_failures=false'
    echo 'r1_counterexample_closed=true'
    echo 'rtl_unchanged=true'
} > "$review_dir/R2_VCS_REVIEW_COMPLETE.txt"
