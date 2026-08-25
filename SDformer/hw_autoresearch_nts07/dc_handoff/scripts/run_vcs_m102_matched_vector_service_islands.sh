#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="$DC_HANDOFF_ROOT/runs/m102_matched_vector_service_islands_vcs_r1_sealed_20260824"
VCS_ROOT="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M102 sealed VCS run: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
run_complete=0
on_exit() {
    local rc="$?"
    if [[ "$run_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$rc"
        } > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"
M82="rtl_m82/zero_bubble_elastic_pwp_stream.sv"
BASE_RTL="rtl_m102/m102_bit_sparse_weight_stream.sv"
BASE_SVA="verif_m102/m102_bit_sparse_weight_stream_assertions.sv"
BASE_TB="tb_m102/tb_m102_bit_sparse_weight_stream.sv"
BASE_FILELIST="dc_handoff/filelists/date_m102_bit_sparse_weight_stream_directed_vcs.f"
CAND_RTL="rtl_m102/m102_combined_candidate_service_top.sv"
CAND_SVA="verif_m102/m102_combined_candidate_service_assertions.sv"
CAND_TB="tb_m102/tb_m102_combined_candidate_service_top.sv"
CAND_FILELIST="dc_handoff/filelists/date_m102_combined_candidate_directed_vcs.f"
CONTRACT="contracts/m102_matched_vector_service_islands_vcs_contract_r1_20260824.json"
M88_RESULT="results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/m88_bounded_sync_bank_double_buffer.json"
PREFLIGHT_COMPLETE="reviews/m102_bit_sparse_physical_baseline_preflight_independent_hammer_r1_20260824/RUN_COMPLETE.txt"
PREFLIGHT_MANIFEST="reviews/m102_bit_sparse_physical_baseline_preflight_independent_hammer_r1_20260824/manifest.sha256"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$BASE_RTL"]="29862d377b6226cdc10af60f8c7af287cadb0ff846511496fc21b620f2ccd97e"
    ["$BASE_SVA"]="cb97eee9f7eb2a7d0bcc75d4eace716fd7c45aa05e69da8ac84af6d374efba93"
    ["$BASE_TB"]="471b81af7df6793004db4f6e162d81ae1f412c666196240eb30d46651760e21b"
    ["$BASE_FILELIST"]="dd2254796aaec5e31e364edfddfecdf3b2783b3e7b766b3af00a8c2143241829"
    ["$CAND_RTL"]="e0e6444cbcf695245a2b6b3caa220ae7e93f124fb4ba0109d7de30a0d09ed419"
    ["$CAND_SVA"]="91825e9e284eddad5f50b52b1c0b4a23f475824f5137714c9a5ede3c1d72c312"
    ["$CAND_TB"]="28ff6b98b2d444ef9433224896cf6ae3cda86ed1af08474eb283220cfbe7a353"
    ["$CAND_FILELIST"]="3b996c426840999fa94b3cb128c2859694d5bfb4e34e0c65b523be82af3726dc"
    ["$CONTRACT"]="24b49136151bde04cc809122101b928aa7a778c100b44aefed0db1732b1e7ac2"
    ["$M88_RESULT"]="36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483"
    ["$PREFLIGHT_COMPLETE"]="78cd8fd7f6cb013c19a004ae5883751cc0d74f53f5df337f25f823ac3276c78d"
    ["$PREFLIGHT_MANIFEST"]="1ff9cc0490f189fb18e03dcb5248cfdce7b346354d35a943e12e9675dc961830"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$BASE_RTL" "$BASE_SVA" "$BASE_TB" \
        "$BASE_FILELIST" "$CAND_RTL" "$CAND_SVA" "$CAND_TB" \
        "$CAND_FILELIST" "$CONTRACT" "$M88_RESULT" \
        "$PREFLIGHT_COMPLETE" "$PREFLIGHT_MANIFEST"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M102 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$M82" "$BASE_RTL" "$BASE_SVA" "$BASE_TB" \
    "$BASE_FILELIST" "$CAND_RTL" "$CAND_SVA" "$CAND_TB" \
    "$CAND_FILELIST" "$CONTRACT" "$M88_RESULT" \
    "$PREFLIGHT_COMPLETE" "$PREFLIGHT_MANIFEST" \
    > "$RUN_DIR/input_sha256.txt"

export VCS_HOME="$VCS_ROOT" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$VCS_ROOT/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$RUN_DIR/csrc_baseline" \
    -f "$BASE_FILELIST" -top tb_m102_bit_sparse_weight_stream \
    -o "$RUN_DIR/simv_baseline" \
    > "$RUN_DIR/compile_baseline.raw.log" 2>&1
compile_baseline_rc="$?"
set -e
printf '%s\n' "$compile_baseline_rc" > "$RUN_DIR/compile_baseline.rc"
if [[ "$compile_baseline_rc" -ne 0 || ! -x "$RUN_DIR/simv_baseline" ]]; then
    echo "M102 baseline VCS compile failed rc=$compile_baseline_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile_baseline.raw.log"; then
    echo "M102 baseline compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv_baseline" -no_save \
    -assert report="$RUN_DIR/assert_baseline.report" \
    -cm line+cond+tgl+fsm+assert \
    > "$RUN_DIR/sim_baseline.raw.log" 2>&1
sim_baseline_rc="$?"
set -e
printf '%s\n' "$sim_baseline_rc" > "$RUN_DIR/sim_baseline.rc"
if [[ "$sim_baseline_rc" -ne 0 ]]; then
    echo "M102 baseline VCS simulation failed rc=$sim_baseline_rc" >&2
    exit 30
fi
grep -qx 'PASS M102 bit-sparse weight baseline vectors=90 beats=274 starts=94 ii3_checks=23 lanes=96 signed_min=-128 signed_max=127 stalls=28 attacks=6 resets=7 precompacted=true macros=0' \
    "$RUN_DIR/sim_baseline.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim_baseline.raw.log" "$RUN_DIR/assert_baseline.report"; then
    echo "M102 baseline functional/SVA failure signature found" >&2
    exit 31
fi
grep -Eq 'cp_exact_ii3, .* 70 match' "$RUN_DIR/assert_baseline.report"
grep -Eq 'cp_output_stall, .* 28 match' "$RUN_DIR/assert_baseline.report"
grep -Eq 'cp_signed_boundaries, .* 118 match' "$RUN_DIR/assert_baseline.report"
grep -Eq 'cp_protocol_fault, .* 12 match' "$RUN_DIR/assert_baseline.report"
grep -Eq 'cp_fault_reset_recovery, .* 12 match' "$RUN_DIR/assert_baseline.report"

set +e
"$VCS_ROOT/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$RUN_DIR/csrc_candidate" \
    -f "$CAND_FILELIST" -top tb_m102_combined_candidate_service_top \
    -o "$RUN_DIR/simv_candidate" \
    > "$RUN_DIR/compile_candidate.raw.log" 2>&1
compile_candidate_rc="$?"
set -e
printf '%s\n' "$compile_candidate_rc" > "$RUN_DIR/compile_candidate.rc"
if [[ "$compile_candidate_rc" -ne 0 || ! -x "$RUN_DIR/simv_candidate" ]]; then
    echo "M102 candidate VCS compile failed rc=$compile_candidate_rc" >&2
    exit 40
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile_candidate.raw.log"; then
    echo "M102 candidate compile log contains warning/error signature" >&2
    exit 41
fi

set +e
"$RUN_DIR/simv_candidate" -no_save \
    -assert report="$RUN_DIR/assert_candidate.report" \
    -cm line+cond+tgl+fsm+assert \
    > "$RUN_DIR/sim_candidate.raw.log" 2>&1
sim_candidate_rc="$?"
set -e
printf '%s\n' "$sim_candidate_rc" > "$RUN_DIR/sim_candidate.rc"
if [[ "$sim_candidate_rc" -ne 0 ]]; then
    echo "M102 candidate VCS simulation failed rc=$sim_candidate_rc" >&2
    exit 50
fi
grep -qx 'PASS M102 combined parser_cycles=896 vectors=8 beats=28 pwp=4 correction=2 fallback=2 stalls=3 protocol_attacks=6 shared_slot_ii_checks=7' \
    "$RUN_DIR/sim_candidate.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim_candidate.raw.log" "$RUN_DIR/assert_candidate.report"; then
    echo "M102 candidate functional/SVA failure signature found" >&2
    exit 51
fi
grep -Eq 'cp_pwp, .* 7 match' "$RUN_DIR/assert_candidate.report"
grep -Eq 'cp_positive_correction, .* 2 match' "$RUN_DIR/assert_candidate.report"
grep -Eq 'cp_fallback, .* 2 match' "$RUN_DIR/assert_candidate.report"
grep -Eq 'cp_stall, .* 3 match' "$RUN_DIR/assert_candidate.report"
grep -Eq 'cp_protocol_fault, .* 11 match' "$RUN_DIR/assert_candidate.report"

{
    echo "status=PASS_M102_MATCHED_VECTOR_SERVICE_ISLANDS_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "baseline_vectors=90"
    echo "candidate_vectors=8"
    echo "baseline_fixed8_ii_cycles=3"
    echo "candidate_pwp_service_cycles=3,4,4,5"
    echo "candidate_correction_fallback_service_cycles=3"
    echo "aggregate_service_slots=1"
    echo "precompacted_input=true"
    echo "actual_record_replay=false"
    echo "physical_fmax_speedup=false"
    echo "equal_area=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
sha256sum "$RUN_DIR"/*.raw.log "$RUN_DIR"/*.report \
    "$RUN_DIR"/RUN_COMPLETE.txt > "$RUN_DIR/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m102_matched_vector_service_islands.sh" \
    > "$RUN_DIR/runner_sha256.txt"
run_complete=1
echo "PASS M102 matched vector-service islands VCS/SVA sealed at $RUN_DIR"
