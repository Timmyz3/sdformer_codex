#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_sealed="$task_review/sealed_vcs_rerun"
task_hammer="$task_review/independent_vcs"

cd "$task_hw_root"
declare -A task_expected=(
    ["contracts/m125_block_phased_k4_row_fold_vcs_contract_r1_20260824.json"]="0e3512088045a32afa4eafafdf7ff9003f988732e6e01b46c9ed1520da3dbf12"
    ["rtl_m125/m125_block_phased_k4_row_fold.sv"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["verif_m125/m125_block_phased_k4_row_fold_assertions.sv"]="35f637d853a9760824a638db8757828afe7d4ecfe8e880e578896f082f8432b9"
    ["tb_m125/tb_m125_block_phased_k4_row_fold.sv"]="ad90e409d53d5b32a5b1a1f7bd25c6b0bfca9bb2933acf8e216684bf4c450384"
    ["dc_handoff/filelists/date_m125_block_phased_k4_row_fold_directed_vcs.f"]="ee2d94cdea3fa5e1e7b5f6210e61a93bd364f0e84c8e86a2cbbec317e2fcb8cc"
)

: > "$task_review/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_review/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]]
done

rm -rf "$task_sealed" "$task_hammer"
mkdir -p "$task_sealed" "$task_hammer"
export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_sealed/csrc" \
    -f dc_handoff/filelists/date_m125_block_phased_k4_row_fold_directed_vcs.f \
    -top tb_m125_block_phased_k4_row_fold \
    -o "$task_sealed/simv" > "$task_sealed/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_sealed/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_sealed/compile.raw.log"

set +e
"$task_sealed/simv" -no_save \
    -assert report="$task_sealed/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_sealed/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -qx 'PASS M125 block-phased K4 row fold VCS fills=51 rows=66 row_done=66 updates=155 selected_sources=528 numeric_lane_checks=14880 full_k4_updates=105 tail_updates=50 same_row_update_pairs=64 update_stalls=47 negated_minus128_contributions=20 plus512_checks=1 cache_bytes=1536 resident_blocks=1 logical_read_bits_per_update=3072 generic_fold_bits=11 accumulator_delta_bits=19 canonical_select_clear=true fixed8_service_island_projection=3.1725369008459166 projection_only=true m123_integrated=false foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false' "$task_sealed/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_sealed/sim.raw.log" "$task_sealed/assert.report"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_hammer/csrc" \
    -f "$task_review/m125_independent.f" \
    -top tb_m125_independent_hammer \
    -o "$task_hammer/simv" > "$task_hammer/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_hammer/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_hammer/compile.raw.log"

set +e
"$task_hammer/simv" -no_save \
    -assert report="$task_hammer/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_hammer/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer/sim.rc"
[[ "$task_rc" -eq 0 ]]
grep -q '^PASS M125 independent hammer ' "$task_hammer/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report"

sha256sum \
    "$task_sealed/compile.raw.log" "$task_sealed/sim.raw.log" \
    "$task_sealed/assert.report" "$task_hammer/compile.raw.log" \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report" \
    > "$task_review/vcs_output.sha256"
echo 'status=PASS_M125_INDEPENDENT_VCS_HAMMER_WITH_RESET_FINDING' \
    > "$task_review/RUN_COMPLETE.txt"
echo 'reset_quiescence=false' >> "$task_review/RUN_COMPLETE.txt"
echo 'standalone_reset_free_admission=true' >> "$task_review/RUN_COMPLETE.txt"
echo 'physical_speedup=false' >> "$task_review/RUN_COMPLETE.txt"
echo 'system_speedup=false' >> "$task_review/RUN_COMPLETE.txt"
echo "PASS M125 independent hammer outputs at $task_review"
