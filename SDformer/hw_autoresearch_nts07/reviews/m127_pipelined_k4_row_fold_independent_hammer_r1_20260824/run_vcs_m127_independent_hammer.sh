#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_sealed="$task_review/sealed_vcs_replay"
task_hammer="$task_review/independent_vcs"

if [[ -e "$task_sealed" || -e "$task_hammer" ]]; then
    echo "refusing to overwrite existing M127 independent VCS evidence" >&2
    exit 2
fi
mkdir "$task_sealed" "$task_hammer"
cd "$task_hw_root"

declare -A task_expected=(
    ["contracts/m127_pipelined_k4_row_fold_vcs_contract_r1_20260824.json"]="2640b4ba5545cffcd0dd55dce002f4cb3d18222a2379c4f41170888a1a0bc293"
    ["rtl_m125/m125_block_phased_k4_row_fold.sv"]="cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"
    ["rtl_m127/m127_block_phased_pipelined_k4_row_fold.sv"]="5c0c779e8ab463b6589804736bc4d83e77e28cd626a8a117c50caf4a7ea15a5c"
    ["verif_m127/m127_block_phased_pipelined_k4_row_fold_assertions.sv"]="f825e7f2ff7f6617d6cd42c81e620e39675164e430dcf528e1e0c7c1986209bb"
    ["tb_m127/tb_m127_block_phased_pipelined_k4_row_fold.sv"]="abb4462609bf8fe719b7eddde077670fff7a2257632144b794935ae4b26d07a6"
    ["dc_handoff/filelists/date_m127_pipelined_k4_row_fold_differential_vcs.f"]="10b1b4c156f68f3442b576b156aca5b57c29ca83bb1fdc2f07dbabff5961de63"
    ["dc_handoff/scripts/run_vcs_m127_pipelined_k4_row_fold.sh"]="36043ea63b23ecbfad15adb64e9314dc132a3f2aa90d18d105213b968652a255"
    ["reviews/m125_block_phased_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"]="ce917784a653cc9b865bb595a59faaa3b10b228c7760abceb1bb87935a99296e"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

: > "$task_review/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_review/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]]
done

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_sealed/csrc" \
    -f dc_handoff/filelists/date_m127_pipelined_k4_row_fold_differential_vcs.f \
    -top tb_m127_block_phased_pipelined_k4_row_fold \
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
grep -qx 'PASS M127 pipelined K4 row fold VCS fills=99 rows=80 row_done=80 updates=176 selected_sources=606 numeric_lane_checks=16896 full_k4_updates=126 tail_updates=50 ii1_update_pairs=79 update_stalls=38 plus512_checks=2 cycle_exact_checks=507 reset_attacks=1 protocol_attacks=1 pair_pipeline_bits=1920 first_group_extra_cycles=0 m125_cycle_exact_positive=true reset_isolation=true cache_bytes=1536 foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false' "$task_sealed/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_sealed/sim.raw.log" "$task_sealed/assert.report"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_hammer/csrc" \
    -f "$task_review/m127_independent.f" \
    -top tb_m127_independent_hammer \
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
grep -q '^PASS M127 independent hammer ' "$task_hammer/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report"

sha256sum \
    "$task_sealed/compile.raw.log" "$task_sealed/sim.raw.log" \
    "$task_sealed/assert.report" "$task_hammer/compile.raw.log" \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report" \
    > "$task_review/vcs_output.sha256"
{
    echo 'status=PASS_M127_INDEPENDENT_VCS_HAMMER_WITH_THROUGHPUT_SCOPE_FINDING'
    echo 'exact_sha_production_replay=true'
    echo 'accepted_cycle_equivalence=true'
    echo 'valid_numeric_equivalence=true'
    echo 'intra_row_four_group_ii1=true'
    echo 'cross_row_single_group_ii1=false'
    echo 'pair_sum_payload_bits=1920'
    echo 'full_elastic_stage_bits_at_least=1950'
    echo 'dc_frequency_improvement=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > "$task_review/RUN_COMPLETE.txt"
echo "PASS M127 independent hammer outputs at $task_review"
