#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m102_r4_symmetric_accepted_request_grace_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M102 r4 sealed VCS run: $task_run" >&2
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
task_m82="rtl_m82/zero_bubble_elastic_pwp_stream.sv"
task_base_rtl="rtl_m102/m102_bit_sparse_weight_stream.sv"
task_base_sva="verif_m102/m102_bit_sparse_weight_stream_assertions.sv"
task_base_tb="tb_m102/tb_m102_bit_sparse_weight_stream.sv"
task_base_files="dc_handoff/filelists/date_m102_bit_sparse_weight_stream_directed_vcs.f"
task_base_dc_files="dc_handoff/filelists/date_m102_bit_sparse_weight_stream_logic_only_dc.f"
task_cand_rtl="rtl_m102/m102_combined_candidate_service_top.sv"
task_cand_sva="verif_m102/m102_combined_candidate_service_assertions.sv"
task_cand_tb="tb_m102/tb_m102_combined_candidate_service_top.sv"
task_cand_files="dc_handoff/filelists/date_m102_combined_candidate_directed_vcs.f"
task_cand_dc_files="dc_handoff/filelists/date_m102_combined_candidate_service_logic_only_dc.f"
task_contract="contracts/m102_r4_symmetric_accepted_request_grace_vcs_contract_r1_20260824.json"
task_r3_review="reviews/m102_r3_same_cycle_fault_quarantine_independent_hammer_r1_20260824/m102_r3_same_cycle_fault_quarantine_independent_hammer_review.json"
task_r3_manifest="reviews/m102_r3_same_cycle_fault_quarantine_independent_hammer_r1_20260824/manifest.sha256"
task_r3_complete="reviews/m102_r3_same_cycle_fault_quarantine_independent_hammer_r1_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$task_base_rtl"]="a6ffae955d9288ad3f3b3db9320bb87c52da7c68a073cf2e85407ae741dc438b"
    ["$task_base_sva"]="655c4ec9bef3951ef1f9e611c0514d860dac090de50374be281a5f717f3f0e95"
    ["$task_base_tb"]="44458e325cb3fb01c80bcc6ebcf5d32a3081ba3b5d351c2470f30d6fd5b63d60"
    ["$task_base_files"]="dd2254796aaec5e31e364edfddfecdf3b2783b3e7b766b3af00a8c2143241829"
    ["$task_base_dc_files"]="189bef8461da76c61493d66d842166c48307755daa0aa12d5d3da0cb65d04189"
    ["$task_cand_rtl"]="5c5074cdbfa52c332a6416c33c5e0af26fc55fc75f7efc441c3639b3cd084c91"
    ["$task_cand_sva"]="9c429aba817307797c9226948cdd796c51e7b0f88a2e72ad51f452a64ea1107e"
    ["$task_cand_tb"]="eec21efc848a2fc05ca2a85e1b7a68ff4588f168a5b756341fda900fdb410796"
    ["$task_cand_files"]="3b996c426840999fa94b3cb128c2859694d5bfb4e34e0c65b523be82af3726dc"
    ["$task_cand_dc_files"]="e297647b4231465a0cf37ca028b12f016f3f2d3c9ad70745bccc991ac418dfcc"
    ["$task_contract"]="aae0aafda5816ad654f44354a36c3dffb2d0dca47d0b04566063257a207a5481"
    ["$task_r3_review"]="c1565e480849e095555642a8ec39511b2bffd1d25f96d8591fcd74fb88223873"
    ["$task_r3_manifest"]="9f4d8b6d32a5fd91420b3961ecf71dbfadce6c805c09e9f17b814d77855151e6"
    ["$task_r3_complete"]="a84e77189d88a944d4d5976699d71d718f441b056de490cd7257a494e623158f"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M102 r4 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc_baseline" \
    -f "$task_base_files" -top tb_m102_bit_sparse_weight_stream \
    -o "$task_run/simv_baseline" \
    > "$task_run/compile_baseline.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile_baseline.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv_baseline" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile_baseline.raw.log"; then exit 21; fi
set +e
"$task_run/simv_baseline" -no_save \
    -assert report="$task_run/assert_baseline.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim_baseline.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim_baseline.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M102 bit-sparse-r4 baseline vectors=90 beats=277 starts=95 ii3_checks=23 lanes=96 signed_min=-128 signed_max=127 stalls=30 attacks=7 same_cycle_release_attacks=1 accepted_grace_holds=1 resets=8 precompacted=true macros=0' "$task_run/sim_baseline.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim_baseline.raw.log" "$task_run/assert_baseline.report"; then exit 31; fi
for task_cover in \
        'cp_exact_ii3, .* 70 match' \
        'cp_output_stall, .* 30 match' \
        'cp_signed_boundaries, .* 120 match' \
        'cp_protocol_fault, .* 21 match' \
        'cp_same_cycle_release_quarantine, .* 1 match' \
        'cp_accepted_request_grace, .* 1 match' \
        'cp_fault_quarantines_buffered_output, .* 3 match' \
        'cp_fault_reset_recovery, .* 14 match'; do
    grep -Eq "$task_cover" "$task_run/assert_baseline.report"
done

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc_candidate" \
    -f "$task_cand_files" -top tb_m102_combined_candidate_service_top \
    -o "$task_run/simv_candidate" \
    > "$task_run/compile_candidate.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile_candidate.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv_candidate" ]]; then exit 40; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile_candidate.raw.log"; then exit 41; fi
set +e
"$task_run/simv_candidate" -no_save \
    -assert report="$task_run/assert_candidate.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim_candidate.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim_candidate.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 50; fi
grep -qx 'PASS M102 combined-r4 parser_cycles=1792 vectors=8 beats=28 pwp=4 correction=2 fallback=2 stalls=3 protocol_attacks=12 continuation_attacks=6 metadata_attacks=1 same_cycle_release_attacks=1 phase_reload_attacks=1 accepted_grace_holds=1 shared_slot_ii_checks=7' "$task_run/sim_candidate.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim_candidate.raw.log" "$task_run/assert_candidate.report"; then exit 51; fi
for task_cover in \
        'cp_pwp, .* 8 match' \
        'cp_positive_correction, .* 1 match' \
        'cp_negative_correction, .* 1 match' \
        'cp_fallback, .* 2 match' \
        'cp_stall, .* 4 match' \
        'cp_protocol_fault, .* 46 match' \
        'cp_fault_quarantines_buffered_output, .* 6 match' \
        'cp_same_cycle_release_quarantine, .* 1 match' \
        'cp_accepted_request_grace, .* 1 match' \
        'cp_fault_blocks_phase_reload, .* 1 match' \
        'cp_metadata_error, .* 1 match' \
        'cp_pwp_to_correction_seam, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert_candidate.report"
done

{
    echo "status=PASS_M102_R4_SYMMETRIC_ACCEPTED_REQUEST_GRACE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "baseline_and_candidate_same_cycle_quarantine=true"
    echo "accepted_request_grace_exact_identity=true"
    echo "changed_request_faults_same_cycle=true"
    echo "request_fault_reset_only=true"
    echo "same_clock_service_slot_work_ratio=1.4094204844392757"
    echo "actual_record_replay=false"
    echo "physical_fmax_speedup=false"
    echo "equal_area=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/*.raw.log "$task_run"/*.report \
    "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m102_r4_symmetric_accepted_request_grace.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M102 r4 symmetric accepted-request grace sealed at $task_run"
