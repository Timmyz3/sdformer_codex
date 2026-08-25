#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/m37_csd_reconstruct_t10_vcs_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/simv" || -e "$RUN_DIR/compile.log" \
        || -e "$RUN_DIR/sim.log" || -e "$RUN_DIR/vectors.txt" ]]; then
    echo "refusing to overwrite M37 VCS run: $RUN_DIR" >&2
    exit 2
fi
cd "$HW_ROOT"

M37_MATH_CONTRACT_SHA256="790c8a6e7d0fafacf5fcf64b1f4cb106d12fdb93464d68b8592ce5b14125d144"
M37_MATH_RESULT_SHA256="9b5b080aeb198d54df92ab6bd21741dfb5e05cbe24a2b81fe5d39843e82d47e6"
test "$(sha256sum contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json | awk '{print $1}')" \
    = "$M37_MATH_CONTRACT_SHA256"
test "$(sha256sum results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json | awk '{print $1}')" \
    = "$M37_MATH_RESULT_SHA256"

if grep -Eq '[[:space:]][*][[:space:]]' \
        rtl_m37/qfit_atlif_csd_reconstruct_t10.sv; then
    echo "M37 DUT contains a data multiplication operator" >&2
    exit 6
fi
if grep -Fq 'selected_row*RANK' \
        rtl_m37/qfit_atlif_csd_reconstruct_t10.sv \
    || [[ "$(grep -Fc 'selected_coefficient = (selected_row << 1)' \
        rtl_m37/qfit_atlif_csd_reconstruct_t10.sv)" -ne 1 ]]; then
    echo "M37 runtime rank-3 control index is not the exact shift-add rewrite" >&2
    exit 8
fi

sha256sum \
    rtl_m37/qfit_atlif_csd_reconstruct_t10.sv \
    verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv \
    tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv \
    dc_handoff/filelists/date_m37_csd_reconstruct_t10_vcs.f \
    dc_handoff/scripts/run_vcs_m37_csd_reconstruct_t10_sva.sh \
    contracts/m37_csd_reconstruct_t10_vcs_contract_r3_20260822.json \
    contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json \
    results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json \
    > "$RUN_DIR/input_sha256.txt"

export VCS_ARCH_OVERRIDE=linux
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f dc_handoff/filelists/date_m37_csd_reconstruct_t10_vcs.f \
    -top tb_qfit_atlif_csd_reconstruct_t10 \
    -o "$RUN_DIR/simv" 2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" +M37_VECTOR_FILE="$RUN_DIR/vectors.txt" \
    2>&1 | tee "$RUN_DIR/sim.log"

grep -q '^M37_PASS total_tiles=245 nominal_tiles=96 dut_unique_signed_input_coefficient_product_pairs=65536 product_miters=117600 bit_miters=39200 arithmetic_issues=1225 no_data_multiplier=1$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_UNIQUENESS unique_tile_payloads=96 unique_expected_product_fingerprints=96 unique_expected_bitmaps=96 consecutive_identical=0 nominal_unique_signed_inputs=256$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_FLOW conditional_standalone_accept_ii5_matches=69 phase4_chain_accepts=220 max_fifo=16 fifo_full_cycles=249 full_pop_push=116 stalls=1001/147 done_with_fifo_pending=245$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_CONFIG config_load_release_reload=15/15/14 release_reject_busy_fifo_input=599/599/599/571 live_pin_perturbations=96 legal_zero_min_max=1$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_ILLEGAL illegal_matrix=210/210 illegal_classes=30,30,30,30,30,30,30$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_THRESHOLD index=0 value=-8388608 equal=48 just_below_raw=16 positive_saturation=16 negative_saturation=32$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_THRESHOLD index=1 value=-12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_THRESHOLD index=2 value=0 equal=112 just_below_raw=16 positive_saturation=16 negative_saturation=16$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_THRESHOLD index=3 value=12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_THRESHOLD index=4 value=8388607 equal=32 just_below_raw=16 positive_saturation=16 negative_saturation=16$' \
    "$RUN_DIR/sim.log"
grep -q '^M37_DIVERSITY generic_saturation=80/96 diversity=19740/19460$' \
    "$RUN_DIR/sim.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.log"
grep -q 'M37_SVA_BOUND=1' "$RUN_DIR/sim.log"
grep -q 'M37_RANDOM_SEED=0x4d370203' "$RUN_DIR/sim.log"
if grep -Eq ', [0-9]+ attempts, 0 match$' "$RUN_DIR/sim.log"; then
    echo "M37 uncovered SVA cover property" >&2
    exit 5
fi
if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.log"; then
    echo "M37 assertion failure signature found" >&2
    exit 3
fi
if grep -Eq '^(Warning|Error)-' "$RUN_DIR/compile.log"; then
    echo "M37 VCS compile warning/error signature found" >&2
    exit 7
fi
if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/compile.log"; then
    echo "M37 VCS compile log contains a broad warning/error/fatal signature" >&2
    exit 7
fi

M37_GOLDEN_VECTOR_SHA256="2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627"
M37_VECTOR_SHA256="$(sha256sum "$RUN_DIR/vectors.txt" | awk '{print $1}')"
if [[ "$M37_VECTOR_SHA256" != "$M37_GOLDEN_VECTOR_SHA256" ]]; then
    echo "M37 vector SHA256 mismatch: expected $M37_GOLDEN_VECTOR_SHA256 got $M37_VECTOR_SHA256" >&2
    exit 4
fi

{
    echo "status=PASS_EXACT_RANK3_CONTROL_INDEX_SHIFT_ADD_SOURCE_INTENT"
    echo "old_runtime_expression=selected_row_times_RANK"
    echo "old_dc_observation=2_DW02_mult_A4_B2_control_index_resources"
    echo "old_dc_run_state=FAIL_RESOURCE_AUDIT_DO_NOT_CITE"
    echo "new_runtime_expression=(selected_row_shift_left_1)_plus_selected_row_plus_rank_index"
    echo "required_shift_add_match_count=1"
    echo "forbidden_selected_row_times_RANK_match_count=0"
    echo "source_star_tokens_begin"
    grep -n '\*' rtl_m37/qfit_atlif_csd_reconstruct_t10.sv
    echo "source_star_tokens_end"
} > "$RUN_DIR/rtl_multiplier_intent_audit.txt"
{
    echo "status=PASS_R8_VCS_SVA_PENDING_INDEPENDENT_REVIEW"
    echo "review_required=true"
    echo "headline_admitted=false"
    echo "prior_r7_receipt_state=STALE_SUPERSEDED_DO_NOT_CITE"
    echo "prior_dc_3ns_r1_state=FAIL_RESOURCE_AUDIT_DO_NOT_CITE"
} > "$RUN_DIR/runner_status.txt"

sha256sum "$RUN_DIR/compile.log" "$RUN_DIR/sim.log" \
    "$RUN_DIR/vectors.txt" "$RUN_DIR/rtl_multiplier_intent_audit.txt" \
    "$RUN_DIR/runner_status.txt" > "$RUN_DIR/output_sha256.txt"
(cd "$RUN_DIR" && sha256sum --strict -c output_sha256.txt)
(
    cd "$RUN_DIR"
    sha256sum input_sha256.txt output_sha256.txt runner_status.txt \
        > run_local_seal.sha256
    sha256sum --strict -c run_local_seal.sha256
)
