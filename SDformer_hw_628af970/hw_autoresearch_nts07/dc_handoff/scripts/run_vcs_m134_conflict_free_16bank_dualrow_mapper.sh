#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M134 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
on_exit() {
    local task_rc="$?"
    if [[ "$task_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$task_rc"
        } > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$task_hw_root"
task_rtl="rtl_m134/m134_conflict_free_16bank_dualrow_mapper.sv"
task_sva="verif_m134/m134_conflict_free_16bank_dualrow_mapper_assertions.sv"
task_tb="tb_m134/tb_m134_conflict_free_16bank_dualrow_mapper.sv"
task_files="dc_handoff/filelists/date_m134_conflict_free_16bank_dualrow_mapper_directed_vcs.f"
task_contract="contracts/m134_conflict_free_16bank_dualrow_mapper_vcs_contract_r1_20260824.json"
task_m132_correction="contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json"
task_m133r2_receipt="dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_rtl"]="497eb7ac803d08692352ac0d77db54f585cfb597ddd081632d53ca0ff91fdbe3"
    ["$task_sva"]="0d626b4ef1038d046b128e9a1d04fcb121ca2e0ccca2a978b5175c13884032c8"
    ["$task_tb"]="b274eae135db56492ebda13ff2a25e6a3f4bcf690d6d7bbafa299e8d2559d91b"
    ["$task_files"]="11cc9888135e5226ffeded5e29290f5e0e8953e3f78d22a368339d040d132f4c"
    ["$task_contract"]="5536ddc291254f2daea2169aad6160e9be8b36299da00a0002cd671e1a64e6da"
    ["$task_m132_correction"]="82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9"
    ["$task_m133r2_receipt"]="e8981a5fb623f76df044225513d8334b03b65b3fcd73620eeee57d6707b2dc49"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M134 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" \
    -top tb_m134_conflict_free_16bank_dualrow_mapper \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
task_pass='PASS M134 conflict-free 16-bank dualrow mapper VCS legal_windows=3665 logical_words=58640 physical_bank_reads=58640 row_crossings=3435 base_offsets=16 illegal_windows=3 words=3680 banks=16 word_bits=32 service_bits=512 reads_per_bank=1 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_crosses_physical_row, .* [1-9][0-9]* match' \
        'cp_last_legal_window, .* [1-9][0-9]* match' \
        'cp_first_illegal_window, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done
for task_offset in $(seq 0 15); do
    grep -Eq "offset_covers\\[$task_offset\\]\\.cp_every_base_bank, .* [1-9][0-9]* match" \
        "$task_run/assert.report"
done

{
    echo "status=PASS_M134_CONFLICT_FREE_16BANK_DUALROW_MAPPER_VCS_SVA"
    echo "exact_sha=true"
    echo "legal_base_windows=3665"
    echo "logical_word_checks=58640"
    echo "physical_bank_address_checks=58640"
    echo "row_crossing_windows=3435"
    echo "base_bank_offsets_covered=16"
    echo "illegal_windows=3"
    echo "logical_words_total=3680"
    echo "banks=16"
    echo "word_bits=32"
    echo "service_bits=512"
    echo "reads_per_bank_per_service=1"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m134_conflict_free_16bank_dualrow_mapper.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M134 conflict-free 16-bank dualrow mapper VCS sealed at $task_run"
