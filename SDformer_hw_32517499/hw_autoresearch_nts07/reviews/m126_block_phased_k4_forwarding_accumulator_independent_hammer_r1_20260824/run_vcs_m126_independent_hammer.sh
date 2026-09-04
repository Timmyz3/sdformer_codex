#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_sealed="$task_review/sealed_vcs_rerun"
task_hammer="$task_review/independent_vcs"
task_boundary="$task_review/boundary_vcs"

cd "$task_hw_root"
declare -A task_expected=(
    ["contracts/m126_block_phased_k4_forwarding_accumulator_vcs_contract_r1_20260824.json"]="f9a8783e1f3fc915bb690e42703a8547377fbf41c92ecc6276673c9f9ac44889"
    ["rtl_m125/m125_block_phased_k4_row_fold.sv"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv"]="b75c64cfa0803461bef4690025a723df9e039e8d2eef6a0da918fc3b9c063e01"
    ["verif_m126/m126_block_phased_k4_forwarding_accumulator_island_assertions.sv"]="fee69341cb32d960eedcc97646fbf893a1c88e6b220ba6a6c2a05c2be22f64c1"
    ["tb_m126/tb_m126_block_phased_k4_forwarding_accumulator_island.sv"]="18784c618a86785ae5bf083257a8559059132323ea3b2d13e49962435d0c7cbc"
    ["dc_handoff/filelists/date_m126_block_phased_k4_forwarding_accumulator_directed_vcs.f"]="890b2870bae08860f47e12afd48258e3f20e1f67168b51105659df3c016e5412"
)
: > "$task_review/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_review/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]]
done

rm -rf "$task_sealed" "$task_hammer" "$task_boundary"
mkdir -p "$task_sealed" "$task_hammer" "$task_boundary"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_sealed/csrc" \
    -f dc_handoff/filelists/date_m126_block_phased_k4_forwarding_accumulator_directed_vcs.f \
    -top tb_m126_block_phased_k4_forwarding_accumulator_island \
    -o "$task_sealed/simv" > "$task_sealed/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_sealed/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_sealed/compile.raw.log"
set +e
"$task_sealed/simv" -no_save -assert report="$task_sealed/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_sealed/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -qx 'PASS M126 K4 fold plus forwarding accumulator VCS starts=2 fills=387 rows=3073 row_done=3072 fold_updates=7327 selected_sources=24803 full_k4_updates=5115 tail_updates=2212 same_row_update_pairs=4262 lane_writes=7326 rw_overlap=0 commits=3072 commit_lane_checks=294912 commit_stalls=401 plus512_checks=1 reset_attacks=1 positive_fold_updates=7326 positive_selected_sources=24802 positive_tail_updates=2211 positive_lane_writes=7326 reset_pending_updates=1 reset_suppressed_writes=1 blocks=8 rows_per_block=384 lanes=96 cache_bytes=1536 fold_bits=11 accumulator_bits=19 m125_m123_integrated=true reset_isolation=true functional_directed_update_compression=3.385476385476 heldout_fixed8_service_projection=3.1725369008459166 projection_only=true foundry_weight_macro=false foundry_accumulator_macro=false physical_speedup=false system_speedup=false headline=false' "$task_sealed/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_sealed/sim.raw.log" "$task_sealed/assert.report"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_hammer/csrc" \
    -f "$task_review/m126_independent.f" \
    -top tb_m126_independent_hammer \
    -o "$task_hammer/simv" > "$task_hammer/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_hammer/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_hammer/compile.raw.log"
set +e
"$task_hammer/simv" -no_save -assert report="$task_hammer/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_hammer/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -q '^PASS M126 independent hammer ' "$task_hammer/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report"

# Overflow/identity characterization intentionally omits production SVA:
# an accepted overflowing update is specified to raise a sticky fault and not
# write, which intentionally violates the positive-only conservation property.
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    -Mdir="$task_boundary/csrc" -f "$task_review/m126_boundary.f" \
    -top tb_m126_overflow_identity_hammer \
    -o "$task_boundary/simv" > "$task_boundary/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_boundary/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_boundary/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_boundary/compile.raw.log"
set +e
"$task_boundary/simv" -no_save > "$task_boundary/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_boundary/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -q '^PASS M126 overflow identity hammer ' "$task_boundary/sim.raw.log"
! grep -Eiq '^Error|^Fatal|watchdog timeout' "$task_boundary/sim.raw.log"

sha256sum "$task_sealed"/{compile.raw.log,sim.raw.log,assert.report} \
    "$task_hammer"/{compile.raw.log,sim.raw.log,assert.report} \
    "$task_boundary"/{compile.raw.log,sim.raw.log} \
    > "$task_review/vcs_output.sha256"
{
    echo 'status=PASS_M126_INDEPENDENT_VCS_HAMMER'
    echo 'reset_high_external_handshakes=0'
    echo 'reset_edge_physical_writes=0'
    echo 'positive_source_update_write_commit_conservation=true'
    echo 'overflow_fail_closed=true'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
} > "$task_review/RUN_COMPLETE.txt"
echo "PASS M126 independent hammer outputs at $task_review"
