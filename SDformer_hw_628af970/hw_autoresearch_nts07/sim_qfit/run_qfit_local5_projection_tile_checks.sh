#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_projection_tile"
OUT="${ROOT}/results/qfit_local5_projection_tile_yosys_20260731"
mkdir -p "${BUILD}" "${OUT}"
rm -f "${OUT}/status.tsv" "${OUT}/report.md" "${OUT}/cycle_evidence.json"

(
  cd "${ROOT}"
  python3 -m unittest tests.test_qfsa_exact_reference
) 2>&1 | tee "${OUT}/python_exact_reference.log"

{
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  yosys -V
  uname -srvmo
  g++ --version | sed -n '1p'
} >"${OUT}/tool_versions.txt"

python3 "${ROOT}/scripts/generate_local5_masked_integer_vectors.py" \
  --count 1024 \
  --seed 0x66d5 \
  --output "${BUILD}/local5_score_shiftmax_vectors.txt" \
  >"${OUT}/score_shiftmax_vector_generation.log"
sha256sum "${BUILD}/local5_score_shiftmax_vectors.txt" \
  >"${OUT}/generated_vector_sha256.txt"
python3 "${ROOT}/scripts/generate_local5_fullchain_oracle.py" \
  --inputs "${BUILD}/local5_fullchain_inputs.txt" \
  --expected "${BUILD}/local5_fullchain_expected.txt" \
  >"${OUT}/fullchain_oracle_generation.log"
python3 "${ROOT}/scripts/generate_local5_fullchain_oracle.py" \
  --seed 0x5eed1234 \
  --inputs "${BUILD}/local5_fullchain_random_inputs.txt" \
  --expected "${BUILD}/local5_fullchain_random_expected.txt" \
  >"${OUT}/fullchain_random_oracle_generation.log"
sha256sum \
  "${BUILD}/local5_fullchain_inputs.txt" \
  "${BUILD}/local5_fullchain_expected.txt" \
  "${BUILD}/local5_fullchain_random_inputs.txt" \
  "${BUILD}/local5_fullchain_random_expected.txt" \
  >"${OUT}/fullchain_oracle_sha256.txt"
FULLCHAIN_ARGS=(
  "+PY_INPUTS=${BUILD}/local5_fullchain_inputs.txt"
  "+PY_EXPECTED=${BUILD}/local5_fullchain_expected.txt"
)
FULLCHAIN_RANDOM_ARGS=(
  "+PY_INPUTS=${BUILD}/local5_fullchain_random_inputs.txt"
  "+PY_EXPECTED=${BUILD}/local5_fullchain_random_expected.txt"
)
iverilog -g2012 -s tb_local5_score_shiftmax_vectors \
  -o "${BUILD}/score_shiftmax_ref_iv" \
  "${ROOT}/rtl_local5/local5_axnor_score_q7.sv" \
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv" \
  "${ROOT}/tb_local5/tb_local5_score_shiftmax_vectors.sv"
vvp "${BUILD}/score_shiftmax_ref_iv" \
  "+VECTORS=${BUILD}/local5_score_shiftmax_vectors.txt" \
  "+EXPECTED=1024" \
  | tee "${OUT}/score_shiftmax_pyref_iverilog.log"
rm -rf "${BUILD}/obj_score_shiftmax_ref"
verilator --binary --timing -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_local5_score_shiftmax_vectors \
  --Mdir "${BUILD}/obj_score_shiftmax_ref" \
  "${ROOT}/rtl_local5/local5_axnor_score_q7.sv" \
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv" \
  "${ROOT}/tb_local5/tb_local5_score_shiftmax_vectors.sv"
"${BUILD}/obj_score_shiftmax_ref/Vtb_local5_score_shiftmax_vectors" \
  "+VECTORS=${BUILD}/local5_score_shiftmax_vectors.txt" \
  "+EXPECTED=1024" \
  | tee "${OUT}/score_shiftmax_pyref_verilator.log"

COMMON=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_local5/local5_axnor_score_q7.sv"
  "${ROOT}/rtl_local5/local5_stencil_token.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_role_sharded_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
)
SYNTH=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_role_sharded_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
)

for backend in tcfm5 affine4 linear5 role_sharded; do
  case "${backend}" in
    tcfm5) kind=0 ;;
    affine4) kind=1 ;;
    linear5) kind=2 ;;
    role_sharded) kind=3 ;;
  esac
  iverilog -g2012 -s tb_qfit_local5_projection_tile \
    -P "tb_qfit_local5_projection_tile.BACKEND_KIND=${kind}" \
    -o "${BUILD}/tile_${backend}_iv" \
    "${COMMON[@]}" \
    "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
  vvp "${BUILD}/tile_${backend}_iv" \
    "${FULLCHAIN_ARGS[@]}" \
    "+TERM_TRACE=${OUT}/ordered_term_trace.csv" \
    | tee "${OUT}/${backend}_iverilog.log"
done
cp "${OUT}/tcfm5_iverilog.log" "${OUT}/iverilog.log"
vvp "${BUILD}/tile_tcfm5_iv" \
  "${FULLCHAIN_RANDOM_ARGS[@]}" \
  | tee "${OUT}/fullchain_random_iverilog.log"

for seed in 1 44257 48879; do
  iverilog -g2012 -s tb_qfit_local5_projection_tile \
    -P "tb_qfit_local5_projection_tile.BACKEND_KIND=0" \
    -o "${BUILD}/tile_tcfm5_bp_${seed}_iv" \
    "${COMMON[@]}" \
    "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
  vvp "${BUILD}/tile_tcfm5_bp_${seed}_iv" \
    "${FULLCHAIN_ARGS[@]}" \
    "+TERM_STALL_SEED=${seed}" \
    | tee "${OUT}/tcfm5_backpressure_seed_${seed}_iverilog.log"
done

iverilog -g2012 -s tb_qfit_local5_projection_protocol \
  -o "${BUILD}/tile_protocol_iv" \
  "${COMMON[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_protocol.sv"
vvp "${BUILD}/tile_protocol_iv" \
  | tee "${OUT}/protocol_iverilog.log"

rm -rf "${BUILD}/obj"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_projection_tile \
  --Mdir "${BUILD}/obj" \
  "${COMMON[@]}" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
"${BUILD}/obj/Vtb_qfit_local5_projection_tile" \
  "${FULLCHAIN_ARGS[@]}" \
  | tee "${OUT}/verilator.log"
"${BUILD}/obj/Vtb_qfit_local5_projection_tile" \
  "${FULLCHAIN_RANDOM_ARGS[@]}" \
  | tee "${OUT}/fullchain_random_verilator_sva.log"
for seed in 1 44257 48879; do
  "${BUILD}/obj/Vtb_qfit_local5_projection_tile" \
    "${FULLCHAIN_ARGS[@]}" \
    "+TERM_STALL_SEED=${seed}" \
    | tee "${OUT}/tcfm5_backpressure_seed_${seed}_verilator_sva.log"
done

rm -rf "${BUILD}/obj_protocol"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_projection_protocol \
  --Mdir "${BUILD}/obj_protocol" \
  "${COMMON[@]}" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_protocol.sv"
"${BUILD}/obj_protocol/Vtb_qfit_local5_projection_protocol" \
  | tee "${OUT}/protocol_verilator_sva.log"

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${SYNTH[*]};
  hierarchy -top qfit_local5_projection_tile;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/hier_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/stat.json stat -json
"

sha256sum \
  "${COMMON[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_protocol.sv" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/scripts/report_qfit_local5_projection_tile.py" \
  "${ROOT}/scripts/analyze_qfit_value_quotient_trace.py" \
  "${ROOT}/tests/test_qfsa_exact_reference.py" \
  "${ROOT}/scripts/qfsa_exact_reference.py" \
  "${ROOT}/scripts/generate_local5_masked_integer_vectors.py" \
  "${ROOT}/scripts/generate_local5_fullchain_oracle.py" \
  "${ROOT}/tb_local5/tb_local5_score_shiftmax_vectors.sv" \
  "${ROOT}/sim_qfit/run_qfit_local5_projection_tile_checks.sh" \
  >"${OUT}/source_sha256.txt"
printf 'Icarus end-to-end integer golden\tPASS\n' >"${OUT}/status.tsv"
printf 'Python QFSA score exact reference 6/6\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Python-generated score/Shiftmax vectors 1024 Icarus/Verilator\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Independent Python Q/K-to-Acc32 directed/random fullchain Icarus/Verilator-SVA\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Icarus Affine4/Linear5 same-producer exact\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Icarus Role-Sharded same-producer exact\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Icarus TCFM-5 fixed-seed random backpressure x3\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Verilator/SVA TCFM-5 fixed-seed backpressure x3\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Icarus adversarial protocol negative/last-term/nonzero consecutive\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Verilator/SVA adversarial protocol negative/last-term/nonzero consecutive\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'Verilator/SVA end-to-end\tPASS\n' >>"${OUT}/status.tsv"
printf 'Yosys check\tPASS\n' >>"${OUT}/status.tsv"
"${ROOT}/scripts/report_qfit_local5_projection_tile.py"
"${ROOT}/scripts/analyze_qfit_value_quotient_trace.py"
printf 'PASS qfit Local5 end-to-end projection tile checks\n'
