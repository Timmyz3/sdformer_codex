#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m135r2_conflict_free_16bank_pwp_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M135r2 sealed VCS run: $task_run" >&2
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
task_m134="rtl_m134/m134_conflict_free_16bank_dualrow_mapper.sv"
task_m133="rtl_m133/m133_dualrow512_elastic_pwp_stream.sv"
task_m135="rtl_m135/m135_conflict_free_16bank_pwp_frontend.sv"
task_sva="verif_m135/m135_conflict_free_16bank_pwp_frontend_assertions.sv"
task_tb="tb_m135/tb_m135_conflict_free_16bank_pwp_frontend.sv"
task_files="dc_handoff/filelists/date_m135_conflict_free_16bank_pwp_frontend_directed_vcs.f"
task_contract="contracts/m135r2_conflict_free_16bank_pwp_frontend_vcs_contract_r1_20260824.json"
task_correction="contracts/m135_r1_dc_check_design_failure_correction_r1_20260824.json"
task_review_overlay="contracts/m133r2_independent_review_identity_supersession_overlay_r1_20260824.json"
task_m134_receipt="dc_handoff/runs/m134_conflict_free_16bank_dualrow_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m133r2_receipt="dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m134"]="497eb7ac803d08692352ac0d77db54f585cfb597ddd081632d53ca0ff91fdbe3"
    ["$task_m133"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["$task_m135"]="7896f3384cc647fbe033ecbc5af909231be5f1aa4c9c2e9c6fe1aeed5a8f3f56"
    ["$task_sva"]="f509664ad212c71ade77c8656ce7367446aeded101304f102ccb73fe0412c12b"
    ["$task_tb"]="310e99030468d6283ed3e9d812cae480ad66277d1d73809bf529ffa058f60916"
    ["$task_files"]="abdfe7338993b8fa9625747ebb4a35e45a3f87d9080fd3548cccf20485422f12"
    ["$task_contract"]="d28a49741bbc97d054e4d031c21da64f18f745242bcdca1bd7d37e9389227911"
    ["$task_correction"]="479ac0f4112dd399ead6969c48ce1ed6d4fa55ae082911223178e3dbb4a13154"
    ["$task_review_overlay"]="a043e194ee83d40e97914526d091e7f981527b248f2667fc49a3c6254940f9dc"
    ["$task_m134_receipt"]="047dec485d9c5e748d2a98cb10cc65a946d6c39b4b7085e9363a78cb6958f17d"
    ["$task_m133r2_receipt"]="e8981a5fb623f76df044225513d8334b03b65b3fcd73620eeee57d6707b2dc49"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M135r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m135_conflict_free_16bank_pwp_frontend \
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
task_pass='PASS M135 conflict-free 16-bank PWP frontend VCS vectors=96 outputs=96 beats=212 lanes=8832 escapes=4 ii_checks=63 stalls=15 row_crossings=194 base_offsets=16 invalid_base_attacks=1 reset_attacks=1 cycles_8_9_10_11=2_2_2_3 banks=16 service_bits=512 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_cross_row_bank_mapping, .* [1-9][0-9]* match' \
        'cp_last_legal_bank_window, .* [1-9][0-9]* match' \
        'cp_output_stall_release, .* [1-9][0-9]* match' \
        'cp_invalid_base_quarantine, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M135R2_CONFLICT_FREE_16BANK_PWP_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "vectors=96"
    echo "outputs=96"
    echo "accepted_beats=212"
    echo "numeric_lane_checks=8832"
    echo "escapes=4"
    echo "exact_start_interval_checks=63"
    echo "unaligned_row_crossing_beats=194"
    echo "base_bank_offsets_covered=16"
    echo "invalid_base_attacks=1"
    echo "cycles_8_9_10_11=2_2_2_3"
    echo "banks=16"
    echo "service_bits=512"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m135r2_conflict_free_16bank_pwp_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M135r2 conflict-free 16-bank PWP frontend VCS sealed at $task_run"
