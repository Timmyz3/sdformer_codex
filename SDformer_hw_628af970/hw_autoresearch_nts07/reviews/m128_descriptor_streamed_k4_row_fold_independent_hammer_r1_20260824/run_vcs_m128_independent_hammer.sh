#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_sealed="$task_review/sealed_vcs_replay"
task_hammer="$task_review/independent_vcs"

if [[ -e "$task_sealed" || -e "$task_hammer" ]]; then
    echo "refusing to overwrite existing M128 independent VCS evidence" >&2
    exit 2
fi
mkdir "$task_sealed" "$task_hammer"
cd "$task_hw_root"

declare -A task_expected=(
    ["contracts/m128_descriptor_streamed_k4_row_fold_vcs_contract_r1_20260824.json"]="7b08459cbba96f14666c57b5db274b850b58546c25d7d42e52210bf9e4228bf1"
    ["contracts/m127_r1_throughput_scope_correction_r1_20260824.json"]="a64a00f443d691b1295a4eb14a92edbc9d41ce448d83fd3a8c3ca4f59d2b365d"
    ["rtl_m128/m128_descriptor_streamed_k4_row_fold.sv"]="b7c5c4c329bc4f1a7011398c5d3c20933dd8badfc4b2bbf3b213b15efe01e54d"
    ["verif_m128/m128_descriptor_streamed_k4_row_fold_assertions.sv"]="334c366289690bff624e8a3976dd602ed45f6046b7b1ed6314143922e5a06a50"
    ["tb_m128/tb_m128_descriptor_streamed_k4_row_fold.sv"]="30cc18e83a00173a9f0e17ea5116f5429a340fbea88f3decb4d28073e8cbee94"
    ["dc_handoff/filelists/date_m128_descriptor_streamed_k4_row_fold_directed_vcs.f"]="685e547c610acbbf8f9298bb32f9ced1035aff158192d9f882e2c519f5f9cf7c"
    ["dc_handoff/scripts/run_vcs_m128_descriptor_streamed_k4_row_fold.sh"]="d4fa2311c4d7674fc808a3ad1dc09c9f266000660bb913cb17908c4d2098931c"
    ["reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256"]="8bea333f44528044f251a48ebf9d20e261e4919bc63ed9f262b01004d25c7947"
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
    -f dc_handoff/filelists/date_m128_descriptor_streamed_k4_row_fold_directed_vcs.f \
    -top tb_m128_descriptor_streamed_k4_row_fold \
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
grep -Fqx 'PASS M128 descriptor-streamed K4 row fold VCS groups=384 updates=384 sources=1056 lanes=36864 rows_done=170 stalls=98 cross_row_updates=64 cross_row_ii1=63 plus512=1 protocol_attacks=1 reset_attacks=1 cache_bytes=1536 descriptor_predecode_external=true physical_speedup=false system_speedup=false headline=false' "$task_sealed/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_sealed/sim.raw.log" "$task_sealed/assert.report"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_hammer/csrc" \
    -f "$task_review/m128_independent.f" \
    -top tb_m128_independent_hammer \
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
grep -q '^PASS M128 independent hammer ' "$task_hammer/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    "$task_hammer/sim.raw.log" "$task_hammer/assert.report"

sha256sum "$task_sealed"/{compile.raw.log,sim.raw.log,assert.report} \
    "$task_hammer"/{compile.raw.log,sim.raw.log,assert.report} \
    > "$task_review/vcs_output.sha256"
{
    echo 'status=PASS_M128_INDEPENDENT_VCS_HAMMER_WITH_EXTERNAL_PREDECODE_BOUNDARY'
    echo 'exact_sha_production_replay=true'
    echo 'descriptor_bits=53'
    echo 'cross_row_descriptor_ii1=true'
    echo 'canonical_order_enforced=false'
    echo 'cross_descriptor_source_conservation_enforced=false'
    echo 'descriptor_predecode_external=true'
    echo 'descriptor_predecode_cost_modeled=false'
    echo 'dc_frequency_improvement=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > "$task_review/RUN_COMPLETE.txt"
echo "PASS M128 independent hammer outputs at $task_review"
