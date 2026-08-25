#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_prod="$task_review/production_vcs_replay"
task_ind="$task_review/independent_vcs"
task_param="$task_review/parameter_attacks"

if [[ -e "$task_prod" || -e "$task_ind" || -e "$task_param" ]]; then
    echo "refusing to overwrite existing M134 independent evidence" >&2
    exit 2
fi
mkdir "$task_prod" "$task_ind" "$task_param"

cd "$task_root"
declare -A task_expected=(
    ["contracts/m134_conflict_free_16bank_dualrow_mapper_vcs_contract_r1_20260824.json"]="5536ddc291254f2daea2169aad6160e9be8b36299da00a0002cd671e1a64e6da"
    ["contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json"]="82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9"
    ["rtl_m134/m134_conflict_free_16bank_dualrow_mapper.sv"]="497eb7ac803d08692352ac0d77db54f585cfb597ddd081632d53ca0ff91fdbe3"
    ["verif_m134/m134_conflict_free_16bank_dualrow_mapper_assertions.sv"]="0d626b4ef1038d046b128e9a1d04fcb121ca2e0ccca2a978b5175c13884032c8"
    ["tb_m134/tb_m134_conflict_free_16bank_dualrow_mapper.sv"]="b274eae135db56492ebda13ff2a25e6a3f4bcf690d6d7bbafa299e8d2559d91b"
    ["dc_handoff/filelists/date_m134_conflict_free_16bank_dualrow_mapper_directed_vcs.f"]="11cc9888135e5226ffeded5e29290f5e0e8953e3f78d22a368339d040d132f4c"
    ["dc_handoff/filelists/date_m134_conflict_free_16bank_dualrow_mapper_logic_only_dc.f"]="76d4e88ef1b7bfd60c383ab8e18579742fbf7b60349f87a6fe34f648930da9f3"
    ["dc_handoff/scripts/run_vcs_m134_conflict_free_16bank_dualrow_mapper.sh"]="35e127051f3f973179df6055087b58dbb8b593125cfd106955cba9c7c75de3fb"
    ["dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="e8981a5fb623f76df044225513d8334b03b65b3fcd73620eeee57d6707b2dc49"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="047dec485d9c5e748d2a98cb10cc65a946d6c39b4b7085e9363a78cb6958f17d"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/input_sha256.txt"]="bbc0c4a2a2d137ffe16c1be5cee8ee7df94a09e1b0e5ea1a743b01a7e4a5bcfd"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/output_sha256.txt"]="c00246c2ef56cf005b999317e53ef4003d27333688ca1440558a44686938521b"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/preflight_sha_checks.txt"]="fc485bfcdb46fee94b64ebb2382a6bac6a5a46e818719a76d4b1e75907b0e959"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/runner_sha256.txt"]="d8f3513a852256c6abb0a5f60b38e4e9b6456bc6e152fbe89150435f6c84b52b"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/compile.raw.log"]="f31578696f9515ac9b132a6a202b5cf9171f2f4637d330cdc2cca40b631536ff"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/sim.raw.log"]="8f3cb3d31a642f5d0f4459310494af4b11d8f81a74ee64b42511552abcc25b13"
    ["dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/assert.report"]="172625326c29857eca7a5b801b6c4d6d815b3d2fb3d77b9da4259fd71915c1fd"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    ["reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/m134_production_replay.f"]="4faa4a9fb7619aa8d368ce2d127dddd220f86ba30db13ecdd85128c333942b53"
    ["reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/m134_independent.f"]="37791f6f44c10d6cffae5051b0635dd9c4e40a100d052aea5f97d14d984e425d"
    ["reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/m134_parameter_attack.f"]="fe9eee7095c07d7ee787b10965a2b1af188de335eeb0eeb9d0c9601a7b26ec8d"
    ["reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/tb_m134_independent_hammer.sv"]="503e0bcfc494f2ec3a4e84b53d5166f64d43675b8c42aa92cd526d2ef3b132ec"
    ["reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/tb_m134_parameter_attack.sv"]="1922b002c4334974c5ad9006fce306d862281b59b02c2bf09881aee4a7306537"
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
cd "$task_prod"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir=csrc \
    -F "$task_review/m134_production_replay.f" \
    -top tb_m134_conflict_free_16bank_dualrow_mapper -o simv \
    > compile.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > compile.rc
[[ "$task_rc" -eq 0 && -x simv ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' compile.raw.log
set +e
./simv -no_save -assert report=assert.report \
    -cm line+cond+tgl+fsm+assert > sim.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > sim.rc
[[ "$task_rc" -eq 0 ]]
grep -Fqx 'PASS M134 conflict-free 16-bank dualrow mapper VCS legal_windows=3665 logical_words=58640 physical_bank_reads=58640 row_crossings=3435 base_offsets=16 illegal_windows=3 words=3680 banks=16 word_bits=32 service_bits=512 reads_per_bank=1 macro=false physical_speedup=false system_speedup=false headline=false' sim.raw.log
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' sim.raw.log assert.report

cd "$task_ind"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir=csrc \
    -F "$task_review/m134_independent.f" \
    -top tb_m134_independent_hammer -o simv > compile.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > compile.rc
[[ "$task_rc" -eq 0 && -x simv ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' compile.raw.log
set +e
./simv -no_save -assert report=assert.report \
    -cm line+cond+tgl+fsm+assert > sim.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > sim.rc
[[ "$task_rc" -eq 0 ]]
grep -Fqx 'PASS M134 independent hammer legal_windows=3665 illegal_windows=431 logical_words=58640 physical_addresses=58640 one_read_per_bank_checks=58640 row_crossings=3435 crossed_bank_addresses=27480 base_offset0=230 other_base_offsets=229 valid_low_payload_checks=64 stale_or_skewed_data_undetected=1 x_base_not_fail_closed=1 words=3680 rows_per_bank=230 banks=16 word_bits=32 service_bits=512 exposed_address_bits=128 exposed_bank_data_bits=512 macro=false macro_latency=false response_tag=false parameter_guard_synthesis_hard=false physical_speedup=false system_speedup=false headline=false' sim.raw.log
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' sim.raw.log assert.report

for task_attack in WORDS BANKS WORD_W BASE_W ROW_W; do
    task_dir="$task_param/${task_attack,,}_guard"
    mkdir "$task_dir"
    cd "$task_dir"
    set +e
    "$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
        "+define+ATTACK_${task_attack}" -Mdir=csrc \
        -F "$task_review/m134_parameter_attack.f" \
        -top tb_m134_parameter_attack -o simv > compile.raw.log 2>&1
    task_compile_rc="$?"
    set -e
    printf '%s\n' "$task_compile_rc" > compile.rc
    [[ "$task_compile_rc" -eq 0 && -x simv ]]
    set +e
    ./simv -no_save > sim.raw.log 2>&1
    task_sim_rc="$?"
    set -e
    printf '%s\n' "$task_sim_rc" > sim.rc
    # VCS maps this time-zero $fatal/$finish to process RC 0; the exact fatal
    # text, not the shell RC, is the fail-closed evidence.
    [[ "$task_sim_rc" -eq 0 ]]
    grep -Fq 'M134 production geometry drift' sim.raw.log
done

task_dir="$task_param/banks8_synthesis_guard_bypass"
mkdir "$task_dir"
cd "$task_dir"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -timescale=1ns/1ps \
    +define+SYNTHESIS+ATTACK_BANKS -Mdir=csrc \
    -F "$task_review/m134_parameter_attack.f" \
    -top tb_m134_parameter_attack -o simv > compile.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > compile.rc
[[ "$task_rc" -eq 0 && -x simv ]]
set +e
./simv -no_save > sim.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > sim.rc
[[ "$task_rc" -eq 0 ]]
grep -Fqx 'PASS M134 synthesis-define parameter guard bypass banks=8 guard_active=false hardcoded_modulo16_unknown=true production_geometry_only=true' sim.raw.log

cd "$task_review"
sha256sum production_vcs_replay/{compile.raw.log,sim.raw.log,assert.report} \
    independent_vcs/{compile.raw.log,sim.raw.log,assert.report} \
    > vcs_output.sha256
find parameter_attacks -mindepth 2 -maxdepth 2 \
    \( -name compile.raw.log -o -name compile.rc -o -name sim.raw.log -o -name sim.rc \) \
    -type f -print0 | sort -z | xargs -0 sha256sum > parameter_attack_output.sha256
sha256sum run_m134_independent_hammer.sh > review_runner_sha256.txt
{
    echo 'status=PASS_M134_INDEPENDENT_MAPPING_AND_BOUNDARY_HAMMER'
    echo 'production_exact_sha_vcs_replay=true'
    echo 'independent_exhaustive_legal_windows=3665'
    echo 'independent_exhaustive_illegal_windows=431'
    echo 'one_read_per_bank_mapping=true'
    echo 'logical_reorder_exact=true'
    echo 'row_address_bounds=true'
    echo 'simulation_geometry_guard_attacks=5'
    echo 'synthesis_parameter_guard_hard=false'
    echo 'stale_or_skewed_response_detected=false'
    echo 'unknown_base_fail_closed=false'
    echo 'foundry_macro=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > RUN_COMPLETE.txt
echo "PASS M134 independent hammer at $task_review"
