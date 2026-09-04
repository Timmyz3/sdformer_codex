#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_vcs_r12_exact_sha_20260823"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M37-r12 exact-SHA VCS run: $RUN_DIR" >&2
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
            echo "run_dir=$RUN_DIR"
        } > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"

RTL="rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv"
ASSERTIONS="verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv"
TESTBENCH="tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv"
FILELIST="dc_handoff/filelists/date_m37_r12_csd_reconstruct_t10_vcs.f"
RUNNER="dc_handoff/scripts/run_vcs_m37_r12_exact_sha_sva.sh"
CONTRACT="contracts/m37_r12_exact_sha_vcs_contract_r1_20260823.json"
R11_VALIDATOR="dc_handoff/scripts/validate_m37_r11_independent_hammer_review.py"
R11_REVIEW="results/m37_r11_independent_hammer_review_20260822/m37_r11_independent_hammer_review.json"
R11_PIN="contracts/m37_r11_evidence_pin_r1_20260822.json"
R9_RECEIPT="contracts/m37_output_receipt_r4_20260822.json"
MATH_CONTRACT="contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json"
MATH_RESULT="results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json"

RTL_SHA="f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd"
ASSERTIONS_SHA="7492af816161febbd0b0e62a1f8e697151d15202e4ad71dd79d721f66a874fe0"
TESTBENCH_SHA="bd92f8ebac83fee446b3fbebadbcb928031706ed99641bc248b459e1786da5cc"
FILELIST_SHA="8dec6f37de7483ce8458fd13072578efa7543ad3c73927d75292cbe146834e2b"
CONTRACT_SHA="8d9a335995a96fca84602cda60fcad83b23218a35d7413647db0c6525f05aaab"
R11_VALIDATOR_SHA="d145e1561ab14484833b2ffbef7d3a42609d5934698238a3f254a7f8337bb080"
R11_REVIEW_SHA="cd798e84365a3601d32a854dffb425a16a18c7fed5fc46a1023584f5fb22e7a3"
R11_PIN_SHA="9410b3418a001b84cfe035b9ebe9fef6190284db916d0ef6c7b1d806d46a09b4"
R9_RECEIPT_SHA="7ba9b180705cbc61bc8188e09935ca9cdd86edddd13b5adef0053332941993c1"
MATH_CONTRACT_SHA="790c8a6e7d0fafacf5fcf64b1f4cb106d12fdb93464d68b8592ce5b14125d144"
MATH_RESULT_SHA="9b5b080aeb198d54df92ab6bd21741dfb5e05cbe24a2b81fe5d39843e82d47e6"
GOLDEN_VECTOR_SHA="2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627"

check_sha() {
    local expected="$1"
    local path="$2"
    local observed
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "$expected" "$observed" >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "$expected" ]]; then
        echo "exact-SHA preflight mismatch: $path" >&2
        return 1
    fi
}

: > "$RUN_DIR/preflight_sha_checks.txt"
check_sha "$RTL_SHA" "$RTL"
check_sha "$ASSERTIONS_SHA" "$ASSERTIONS"
check_sha "$TESTBENCH_SHA" "$TESTBENCH"
check_sha "$FILELIST_SHA" "$FILELIST"
check_sha "$CONTRACT_SHA" "$CONTRACT"
check_sha "$R11_VALIDATOR_SHA" "$R11_VALIDATOR"
check_sha "$R11_REVIEW_SHA" "$R11_REVIEW"
check_sha "$R11_PIN_SHA" "$R11_PIN"
check_sha "$R9_RECEIPT_SHA" "$R9_RECEIPT"
check_sha "$MATH_CONTRACT_SHA" "$MATH_CONTRACT"
check_sha "$MATH_RESULT_SHA" "$MATH_RESULT"

# Rebuild the independent review before compilation.  Its deterministic stdout
# must be byte-identical to the externally pinned review, not merely exit zero.
set +e
/usr/bin/python3.6 "$R11_VALIDATOR" \
    > "$RUN_DIR/r11_review_validation.raw.log" \
    2> "$RUN_DIR/r11_review_validation.stderr.raw.log"
r11_validation_rc="$?"
set -e
printf '%s\n' "$r11_validation_rc" > "$RUN_DIR/r11_review_validation.rc"
if [[ "$r11_validation_rc" -ne 0 ]]; then
    echo "M37-r12 independent r11 review rebuild failed" >&2
    exit 10
fi
if [[ -s "$RUN_DIR/r11_review_validation.stderr.raw.log" ]]; then
    echo "M37-r12 independent r11 review emitted stderr" >&2
    exit 11
fi
check_sha "$R11_REVIEW_SHA" "$RUN_DIR/r11_review_validation.raw.log"
if ! cmp -s "$RUN_DIR/r11_review_validation.raw.log" "$R11_REVIEW"; then
    echo "M37-r12 rebuilt review is not byte-identical to pinned review" >&2
    exit 12
fi

sha256sum \
    "$RTL" \
    "$ASSERTIONS" \
    "$TESTBENCH" \
    "$FILELIST" \
    "$RUNNER" \
    "$CONTRACT" \
    "$R11_VALIDATOR" \
    "$R11_REVIEW" \
    "$R11_PIN" \
    "$R9_RECEIPT" \
    "$MATH_CONTRACT" \
    "$MATH_RESULT" \
    > "$RUN_DIR/input_sha256.txt"

set +e
sha256sum --strict -c "$RUN_DIR/input_sha256.txt" \
    > "$RUN_DIR/input_manifest_check.raw.log" 2>&1
input_manifest_rc="$?"
set -e
printf '%s\n' "$input_manifest_rc" > "$RUN_DIR/input_manifest_check.rc"
if [[ "$input_manifest_rc" -ne 0 ]]; then
    echo "M37-r12 input manifest verification failed" >&2
    exit 13
fi

{
    echo "VCS_ARCH_OVERRIDE=linux vcs -full64 -sverilog -assert svaext -debug_access+pp"
    echo "-Mdir=$RUN_DIR/csrc"
    echo "+define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED"
    echo "-timescale=1ns/1ps -f $FILELIST"
    echo "-top tb_qfit_atlif_csd_reconstruct_t10 -o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

export VCS_ARCH_OVERRIDE=linux
set +e
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f "$FILELIST" \
    -top tb_qfit_atlif_csd_reconstruct_t10 \
    -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M37-r12 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eq '^(Warning|Error)-' "$RUN_DIR/compile.raw.log" \
        || grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
            "$RUN_DIR/compile.raw.log"; then
    echo "M37-r12 compile log contains warning/error/fatal signature" >&2
    exit 21
fi
echo "M37_R12_COMPILE=PASS rc=0 simv_executable=1" \
    > "$RUN_DIR/compile.success.marker"

echo "$RUN_DIR/simv +M37_VECTOR_FILE=$RUN_DIR/vectors.txt" \
    > "$RUN_DIR/sim.command.txt"
set +e
"$RUN_DIR/simv" +M37_VECTOR_FILE="$RUN_DIR/vectors.txt" \
    > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M37-r12 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi

pass_count="$(awk '/^M37_PASS / {count++} END {print count+0}' \
    "$RUN_DIR/sim.raw.log")"
if [[ "$pass_count" -ne 1 ]]; then
    echo "M37-r12 requires exactly one internal M37_PASS line, got $pass_count" >&2
    exit 31
fi
sed -n '/^M37_PASS /p' "$RUN_DIR/sim.raw.log" \
    > "$RUN_DIR/tb_internal_pass.marker"

grep -qx 'M37_PASS total_tiles=245 nominal_tiles=96 dut_unique_signed_input_coefficient_product_pairs=65536 product_miters=117600 bit_miters=39200 arithmetic_issues=1225 no_data_multiplier=1' \
    "$RUN_DIR/tb_internal_pass.marker"
grep -qx 'M37_UNIQUENESS unique_tile_payloads=96 unique_expected_product_fingerprints=96 unique_expected_bitmaps=96 consecutive_identical=0 nominal_unique_signed_inputs=256' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_FLOW conditional_standalone_accept_ii5_matches=69 phase4_chain_accepts=220 max_fifo=16 fifo_full_cycles=249 full_pop_push=116 stalls=1001/147 done_with_fifo_pending=245' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_CONFIG config_load_release_reload=15/15/14 release_reject_busy_fifo_input=599/599/599/571 live_pin_perturbations=96 legal_zero_min_max=1' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_ILLEGAL illegal_matrix=210/210 illegal_classes=30,30,30,30,30,30,30' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_THRESHOLD index=0 value=-8388608 equal=48 just_below_raw=16 positive_saturation=16 negative_saturation=32' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_THRESHOLD index=1 value=-12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_THRESHOLD index=2 value=0 equal=112 just_below_raw=16 positive_saturation=16 negative_saturation=16' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_THRESHOLD index=3 value=12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_THRESHOLD index=4 value=8388607 equal=32 just_below_raw=16 positive_saturation=16 negative_saturation=16' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M37_DIVERSITY generic_saturation=80/96 diversity=19740/19460' \
    "$RUN_DIR/sim.raw.log"
grep -q 'Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64' \
    "$RUN_DIR/sim.raw.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.raw.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.raw.log"
grep -q 'M37_SVA_BOUND=1' "$RUN_DIR/sim.raw.log"
grep -q 'M37_RANDOM_SEED=0x4d370203' "$RUN_DIR/sim.raw.log"

mapfile -t cover_counts < <(
    sed -nE 's/^.*,[[:space:]]+2758 attempts,[[:space:]]+([0-9]+) match$/\1/p' \
        "$RUN_DIR/sim.raw.log"
)
if [[ "${cover_counts[*]}" != "220 1271 249 117 245 571 133 210" ]]; then
    echo "M37-r12 SVA cover vector drift: ${cover_counts[*]}" >&2
    exit 32
fi
printf '%s\n' "${cover_counts[*]}" > "$RUN_DIR/sva_cover_counts.txt"

if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M37-r12 functional/SVA failure signature found" >&2
    exit 33
fi

vector_sha="$(sha256sum "$RUN_DIR/vectors.txt" | awk '{print $1}')"
if [[ "$vector_sha" != "$GOLDEN_VECTOR_SHA" ]]; then
    echo "M37-r12 vector SHA mismatch expected=$GOLDEN_VECTOR_SHA got=$vector_sha" >&2
    exit 34
fi

{
    echo "status=PASS_R12_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER"
    echo "headline_admitted=false"
    echo "review_required=true"
    echo "candidate_rtl_sha256=$RTL_SHA"
    echo "r11_validator_sha256=$R11_VALIDATOR_SHA"
    echo "r11_review_sha256=$R11_REVIEW_SHA"
    echo "compile_rc=$compile_rc"
    echo "sim_rc=$sim_rc"
    echo "internal_m37_pass_count=$pass_count"
    echo "functional_mismatch_count=0"
    echo "sva_assertion_failure_count=0"
    echo "DC_STA_Formality_PPA_power_energy_system_admitted=false"
} > "$RUN_DIR/runner_status.txt"
echo "M37_R12_SIMULATION=PASS rc=0 functional_mismatch=0 sva_failure=0" \
    > "$RUN_DIR/simulation.success.marker"

sha256sum \
    "$RUN_DIR/preflight_sha_checks.txt" \
    "$RUN_DIR/r11_review_validation.raw.log" \
    "$RUN_DIR/r11_review_validation.stderr.raw.log" \
    "$RUN_DIR/r11_review_validation.rc" \
    "$RUN_DIR/input_sha256.txt" \
    "$RUN_DIR/input_manifest_check.raw.log" \
    "$RUN_DIR/input_manifest_check.rc" \
    "$RUN_DIR/compile.command.txt" \
    "$RUN_DIR/compile.raw.log" \
    "$RUN_DIR/compile.rc" \
    "$RUN_DIR/compile.success.marker" \
    "$RUN_DIR/simv" \
    "$RUN_DIR/sim.command.txt" \
    "$RUN_DIR/sim.raw.log" \
    "$RUN_DIR/sim.rc" \
    "$RUN_DIR/tb_internal_pass.marker" \
    "$RUN_DIR/sva_cover_counts.txt" \
    "$RUN_DIR/vectors.txt" \
    "$RUN_DIR/simulation.success.marker" \
    "$RUN_DIR/runner_status.txt" \
    > "$RUN_DIR/output_sha256.txt"

set +e
sha256sum --strict -c "$RUN_DIR/output_sha256.txt" \
    > "$RUN_DIR/output_manifest_check.raw.log" 2>&1
output_manifest_rc="$?"
set -e
printf '%s\n' "$output_manifest_rc" > "$RUN_DIR/output_manifest_check.rc"
if [[ "$output_manifest_rc" -ne 0 ]]; then
    echo "M37-r12 output manifest verification failed" >&2
    exit 40
fi

(
    cd "$RUN_DIR"
    sha256sum input_sha256.txt output_sha256.txt runner_status.txt \
        output_manifest_check.raw.log output_manifest_check.rc \
        > run_local_seal.sha256
)
set +e
(
    cd "$RUN_DIR"
    sha256sum --strict -c run_local_seal.sha256
) > "$RUN_DIR/run_local_seal_check.raw.log" 2>&1
seal_rc="$?"
set -e
printf '%s\n' "$seal_rc" > "$RUN_DIR/run_local_seal_check.rc"
if [[ "$seal_rc" -ne 0 ]]; then
    echo "M37-r12 local seal verification failed" >&2
    exit 41
fi

{
    echo "M37_R12_RUN_COMPLETE=PASS"
    echo "all_functional_and_SVA_checks_complete=true"
    echo "all_input_output_and_local_SHA_checks_complete=true"
    echo "claim_scope=VCS_ONLY_NO_DC_FORMALITY_PPA"
} > "$RUN_DIR/RUN_COMPLETE.txt"
(
    cd "$RUN_DIR"
    sha256sum run_local_seal.sha256 run_local_seal_check.raw.log \
        run_local_seal_check.rc RUN_COMPLETE.txt > completion_seal.sha256
    sha256sum --strict -c completion_seal.sha256 \
        > completion_seal_check.raw.log 2>&1
)

run_complete=1
echo "M37_R12_EXACT_SHA_VCS_SVA_RUN=PASS run_dir=$RUN_DIR"
