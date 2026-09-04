#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m102_r2_fail_closed_matched_vector_service_islands_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M102 r2 sealed VCS run: $task_run" >&2
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
task_cand_rtl="rtl_m102/m102_combined_candidate_service_top.sv"
task_cand_sva="verif_m102/m102_combined_candidate_service_assertions.sv"
task_cand_tb="tb_m102/tb_m102_combined_candidate_service_top.sv"
task_cand_files="dc_handoff/filelists/date_m102_combined_candidate_directed_vcs.f"
task_contract="contracts/m102_r2_fail_closed_matched_vector_service_islands_vcs_contract_r1_20260824.json"
task_m88="results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/m88_bounded_sync_bank_double_buffer.json"
task_preflight="reviews/m102_bit_sparse_physical_baseline_preflight_independent_hammer_r1_20260824/RUN_COMPLETE.txt"
task_preflight_manifest="reviews/m102_bit_sparse_physical_baseline_preflight_independent_hammer_r1_20260824/manifest.sha256"
task_hammer="reviews/m102_matched_vector_service_islands_independent_hammer_r1_20260824/m102_matched_vector_service_islands_independent_hammer_review.json"
task_hammer_manifest="reviews/m102_matched_vector_service_islands_independent_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_m82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$task_base_rtl"]="29862d377b6226cdc10af60f8c7af287cadb0ff846511496fc21b620f2ccd97e"
    ["$task_base_sva"]="cb97eee9f7eb2a7d0bcc75d4eace716fd7c45aa05e69da8ac84af6d374efba93"
    ["$task_base_tb"]="471b81af7df6793004db4f6e162d81ae1f412c666196240eb30d46651760e21b"
    ["$task_base_files"]="dd2254796aaec5e31e364edfddfecdf3b2783b3e7b766b3af00a8c2143241829"
    ["$task_cand_rtl"]="426f0374b06e194c182f81553392414c30dfee57eacd56818aa2bd8e1f00742e"
    ["$task_cand_sva"]="5b36b76be98f56ec27533c17f24e27566beed66280b6e6fc97f84c5cb4e2fdcf"
    ["$task_cand_tb"]="bdaaffdc1c9c3a3afd641269db5eb4ee67dbee7d6ff2f7a005010b7690101cc7"
    ["$task_cand_files"]="3b996c426840999fa94b3cb128c2859694d5bfb4e34e0c65b523be82af3726dc"
    ["$task_contract"]="d104a8affd17ca8f456816db92ea7b4d81dc499846f2640faa3e085ef7745bff"
    ["$task_m88"]="36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483"
    ["$task_preflight"]="78cd8fd7f6cb013c19a004ae5883751cc0d74f53f5df337f25f823ac3276c78d"
    ["$task_preflight_manifest"]="1ff9cc0490f189fb18e03dcb5248cfdce7b346354d35a943e12e9675dc961830"
    ["$task_hammer"]="511b2b8bb46a4b1288d97de63ce686a775a8ec4e289e53d189b97e91cce1e745"
    ["$task_hammer_manifest"]="40065f9a0d9484a10179e93312630247d168cc63d9e949706ce93f11dfdd9c80"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "$task_m82" "$task_base_rtl" "$task_base_sva" \
        "$task_base_tb" "$task_base_files" "$task_cand_rtl" \
        "$task_cand_sva" "$task_cand_tb" "$task_cand_files" \
        "$task_contract" "$task_m88" "$task_preflight" \
        "$task_preflight_manifest" "$task_hammer" "$task_hammer_manifest"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M102 r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "$task_m82" "$task_base_rtl" "$task_base_sva" \
    "$task_base_tb" "$task_base_files" "$task_cand_rtl" \
    "$task_cand_sva" "$task_cand_tb" "$task_cand_files" \
    "$task_contract" "$task_m88" "$task_preflight" \
    "$task_preflight_manifest" "$task_hammer" "$task_hammer_manifest" \
    > "$task_run/input_sha256.txt"

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
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv_baseline" ]]; then
    echo "M102 r2 baseline VCS compile failed rc=$task_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile_baseline.raw.log"; then
    echo "M102 r2 baseline compile warning/error signature" >&2
    exit 21
fi
set +e
"$task_run/simv_baseline" -no_save \
    -assert report="$task_run/assert_baseline.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim_baseline.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim_baseline.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
grep -qx 'PASS M102 bit-sparse weight baseline vectors=90 beats=274 starts=94 ii3_checks=23 lanes=96 signed_min=-128 signed_max=127 stalls=28 attacks=6 resets=7 precompacted=true macros=0' \
    "$task_run/sim_baseline.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim_baseline.raw.log" "$task_run/assert_baseline.report"; then
    echo "M102 r2 baseline functional/SVA failure" >&2
    exit 31
fi
for task_cover in \
        'cp_exact_ii3, .* 70 match' \
        'cp_output_stall, .* 28 match' \
        'cp_signed_boundaries, .* 118 match' \
        'cp_protocol_fault, .* 12 match' \
        'cp_fault_reset_recovery, .* 12 match'; do
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
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv_candidate" ]]; then
    echo "M102 r2 candidate VCS compile failed rc=$task_rc" >&2
    exit 40
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile_candidate.raw.log"; then
    echo "M102 r2 candidate compile warning/error signature" >&2
    exit 41
fi
set +e
"$task_run/simv_candidate" -no_save \
    -assert report="$task_run/assert_candidate.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim_candidate.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim_candidate.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 50; fi
grep -qx 'PASS M102 combined-r2 parser_cycles=1792 vectors=8 beats=28 pwp=4 correction=2 fallback=2 stalls=3 protocol_attacks=12 continuation_attacks=6 metadata_attacks=1 fault_stall_attacks=1 shared_slot_ii_checks=7' \
    "$task_run/sim_candidate.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim_candidate.raw.log" "$task_run/assert_candidate.report"; then
    echo "M102 r2 candidate functional/SVA failure" >&2
    exit 51
fi
for task_cover in \
        'cp_pwp, .* 7 match' \
        'cp_positive_correction, .* 2 match' \
        'cp_negative_correction, .* 1 match' \
        'cp_fallback, .* 2 match' \
        'cp_stall, .* 4 match' \
        'cp_protocol_fault, .* 32 match' \
        'cp_fault_quarantines_buffered_output, .* 3 match' \
        'cp_metadata_error, .* 1 match' \
        'cp_pwp_to_correction_seam, .* 2 match'; do
    grep -Eq "$task_cover" "$task_run/assert_candidate.report"
done

{
    echo "status=PASS_M102_R2_FAIL_CLOSED_MATCHED_VECTOR_SERVICE_ISLANDS_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "independent_hammer_p0_repaired=true"
    echo "buffered_output_quarantined_under_top_fault=true"
    echo "continuation_identity_attacks=6"
    echo "metadata_poison_attacks=1"
    echo "pwp_to_correction_seam_cover=2"
    echo "same_clock_service_slot_work_ratio=1.4094204844392757"
    echo "current_single_context_parser_and_load_inclusive_ratio_upper_bound=1.407436500047485"
    echo "actual_record_replay=false"
    echo "physical_fmax_speedup=false"
    echo "equal_area=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/*.raw.log "$task_run"/*.report \
    "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m102_r2_fail_closed_matched_vector_service_islands.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M102 r2 fail-closed matched vector-service islands sealed at $task_run"
