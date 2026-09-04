#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_vcs_r9_20260822}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/simv" || -e "$RUN_DIR/compile.log" \
        || -e "$RUN_DIR/sim.log" || -e "$RUN_DIR/vectors.txt" ]]; then
    echo "refusing to overwrite M37-r9 VCS run: $RUN_DIR" >&2
    exit 2
fi
cd "$HW_ROOT"

M37_R9_RTL_SHA256="a5f42567fc5262a99152ef04699c9062cbedc70075c0a91397ce8d00dc4397ed"
M37_R9_CONTRACT_SHA256="1d8644e3e964bdbb83bf02fc51f41a4669ca21ad6eeb61d9a62a451026d82b77"
M37_R9_AUDITOR_SHA256="f8045d9ff9dddf26202e2d3cde0997fe8cd7f89f5d58dacd170da9e6ef802aa9"
M37_R8_AUDITOR_SHA256="6fcf221ac018e38283723b687852e1809941aabdbbfa031dd812da14113cc856"
M37_R8_ADMISSION_SHA256="f133b96a458686e17f94ecf52c26db3c9b753ef7145f4b396a9f047acfda0fa2"
M37_R8_VALIDATOR_SHA256="7be9c7e5bba4ffb0fb972be948019dce5354362bc1da4e8d3e68057b0c4cce07"
M37_R8_SNAPSHOT_SHA256="ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd"
M37_R8_PROVENANCE_SHA256="f7b88ceafe4447ad7dc1abb11751bead49d3170293ffec1ea6f521aac0c99f99"
M37_R8_LEDGER_SHA256="01dc86fcda8ba3627e2de27fbab26866ca794b0e3e8da05d6fbd563cf72364a3"
M37_MATH_CONTRACT_SHA256="790c8a6e7d0fafacf5fcf64b1f4cb106d12fdb93464d68b8592ce5b14125d144"
M37_MATH_RESULT_SHA256="9b5b080aeb198d54df92ab6bd21741dfb5e05cbe24a2b81fe5d39843e82d47e6"

check_sha() {
    local expected="$1"
    local path="$2"
    local observed
    observed="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$observed" != "$expected" ]]; then
        echo "SHA256 mismatch for $path: expected $expected got $observed" >&2
        exit 9
    fi
}

check_sha "$M37_R9_RTL_SHA256" rtl_m37/qfit_atlif_csd_reconstruct_t10.sv
check_sha "$M37_R9_CONTRACT_SHA256" contracts/m37_csd_reconstruct_t10_vcs_contract_r4_20260822.json
check_sha "$M37_R9_AUDITOR_SHA256" dc_handoff/scripts/audit_m37_r9_source_intent.py
check_sha "$M37_R8_AUDITOR_SHA256" dc_handoff/scripts/audit_m37_r8_source_intent.py
check_sha "$M37_R8_ADMISSION_SHA256" contracts/m37_r8_independent_vcs_source_intent_admission_r1_20260822.json
check_sha "$M37_R8_VALIDATOR_SHA256" dc_handoff/scripts/validate_m37_r8_vcs_source_intent_admission.py
check_sha "$M37_R8_SNAPSHOT_SHA256" evidence_snapshots/m37_r8_ab7d73a6_20260822/qfit_atlif_csd_reconstruct_t10.sv
check_sha "$M37_R8_PROVENANCE_SHA256" evidence_snapshots/m37_r8_ab7d73a6_20260822/README.provenance.txt
check_sha "$M37_R8_LEDGER_SHA256" evidence_snapshots/m37_r8_ab7d73a6_20260822/snapshot_contents.sha256
check_sha "$M37_MATH_CONTRACT_SHA256" contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json
check_sha "$M37_MATH_RESULT_SHA256" results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json

python3 dc_handoff/scripts/validate_m37_r8_vcs_source_intent_admission.py \
    contracts/m37_r8_independent_vcs_source_intent_admission_r1_20260822.json \
    > "$RUN_DIR/r8_admission_validation.txt"
python3 dc_handoff/scripts/audit_m37_r9_source_intent.py \
    rtl_m37/qfit_atlif_csd_reconstruct_t10.sv \
    "$RUN_DIR/r9_source_intent_audit.txt"
grep -q '^status=PASS_M37_R9_STATIC_INDEX_SOURCE_AUDIT$' \
    "$RUN_DIR/r9_source_intent_audit.txt"
test "$(grep -c '^counterexample=.* result=REJECT ' "$RUN_DIR/r9_source_intent_audit.txt")" -eq 6

sha256sum \
    rtl_m37/qfit_atlif_csd_reconstruct_t10.sv \
    verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv \
    tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv \
    dc_handoff/filelists/date_m37_csd_reconstruct_t10_vcs.f \
    dc_handoff/scripts/run_vcs_m37_csd_reconstruct_t10_r9_sva.sh \
    dc_handoff/scripts/audit_m37_r9_source_intent.py \
    dc_handoff/scripts/audit_m37_r8_source_intent.py \
    contracts/m37_csd_reconstruct_t10_vcs_contract_r4_20260822.json \
    contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json \
    results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json \
    contracts/m37_r8_independent_vcs_source_intent_admission_r1_20260822.json \
    dc_handoff/scripts/validate_m37_r8_vcs_source_intent_admission.py \
    evidence_snapshots/m37_r8_ab7d73a6_20260822/qfit_atlif_csd_reconstruct_t10.sv \
    evidence_snapshots/m37_r8_ab7d73a6_20260822/README.provenance.txt \
    evidence_snapshots/m37_r8_ab7d73a6_20260822/snapshot_contents.sha256 \
    > "$RUN_DIR/input_sha256.txt"
sha256sum --strict -c "$RUN_DIR/input_sha256.txt"

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
grep -q 'Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64' \
    "$RUN_DIR/sim.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.log"
grep -q 'M37_SVA_BOUND=1' "$RUN_DIR/sim.log"
grep -q 'M37_RANDOM_SEED=0x4d370203' "$RUN_DIR/sim.log"

mapfile -t cover_counts < <(
    sed -nE 's/^.*,[[:space:]]+2758 attempts,[[:space:]]+([0-9]+) match$/\1/p' \
        "$RUN_DIR/sim.log"
)
if [[ "${cover_counts[*]}" != "220 1271 249 117 245 571 133 210" ]]; then
    echo "M37-r9 SVA cover vector drift: ${cover_counts[*]}" >&2
    exit 5
fi
if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.log"; then
    echo "M37-r9 assertion failure signature found" >&2
    exit 3
fi
if grep -Eq '^(Warning|Error)-' "$RUN_DIR/compile.log"; then
    echo "M37-r9 VCS compile warning/error signature found" >&2
    exit 7
fi
if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/compile.log"; then
    echo "M37-r9 VCS compile log contains a broad warning/error/fatal signature" >&2
    exit 7
fi

M37_GOLDEN_VECTOR_SHA256="2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627"
M37_VECTOR_SHA256="$(sha256sum "$RUN_DIR/vectors.txt" | awk '{print $1}')"
if [[ "$M37_VECTOR_SHA256" != "$M37_GOLDEN_VECTOR_SHA256" ]]; then
    echo "M37-r9 vector SHA256 mismatch: expected $M37_GOLDEN_VECTOR_SHA256 got $M37_VECTOR_SHA256" >&2
    exit 4
fi

{
    echo "status=PASS_R9_STATIC_INDEX_VCS_SVA_PENDING_INDEPENDENT_HAMMER_NO_DC_CLAIM"
    echo "review_required=true"
    echo "headline_admitted=false"
    echo "r9_rtl_sha256=$M37_R9_RTL_SHA256"
    echo "r8_dc_zero_multiplier_state=HISTORICAL_R8_ONLY_NOT_R9_EVIDENCE"
    echo "r8_formality_state=FAIL_FMR_ELAB_147_DO_NOT_CITE_AS_CLOSED"
    echo "r9_dc_sta_formality_ppa_power_energy_system_admitted=false"
} > "$RUN_DIR/runner_status.txt"

sha256sum "$RUN_DIR/compile.log" "$RUN_DIR/sim.log" \
    "$RUN_DIR/vectors.txt" "$RUN_DIR/r9_source_intent_audit.txt" \
    "$RUN_DIR/runner_status.txt" > "$RUN_DIR/output_sha256.txt"
(cd "$RUN_DIR" && sha256sum --strict -c output_sha256.txt)
(
    cd "$RUN_DIR"
    sha256sum input_sha256.txt output_sha256.txt runner_status.txt \
        > run_local_seal.sha256
    sha256sum --strict -c run_local_seal.sha256
)

echo "M37_R9_VCS_SVA_RUN=PASS run_dir=$RUN_DIR"
