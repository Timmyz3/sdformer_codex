#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_projection_t450_shell"
OUT="${ROOT}/results/qfit_local5_projection_t450_shell_20260809"
mkdir -p "${BUILD}" "${OUT}"
rm -f \
  "${OUT}/status.tsv" \
  "${OUT}/report.md" \
  "${OUT}/report.json" \
  "${OUT}/oracle_sha256.txt" \
  "${OUT}/source_sha256.txt" \
  "${OUT}/oracle_hashes.sha256" \
  "${OUT}/oracle_hash_check.log" \
  "${OUT}/source_hashes.sha256" \
  "${OUT}/source_hash_check.log" \
  "${OUT}/tcfm5_backpressure_iverilog.log"

python3 "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  --seed 0x45052026 \
  --out-dim 2 \
  --inputs "${BUILD}/t450_inputs.txt" \
  --expected "${BUILD}/t450_expected.txt" \
  >"${OUT}/oracle_generation.log"
for seed in 0x45052027 0x45052028; do
  tag="${seed#0x}"
  python3 "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
    --seed "${seed}" \
    --out-dim 2 \
    --inputs "${BUILD}/t450_inputs_${tag}.txt" \
    --expected "${BUILD}/t450_expected_${tag}.txt" \
    >"${OUT}/oracle_generation_${tag}.log"
done
python3 "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  --seed 0x45052026 \
  --out-dim 32 \
  --inputs "${BUILD}/t450_out32_inputs.txt" \
  --expected "${BUILD}/t450_out32_expected.txt" \
  >"${OUT}/oracle_generation_out32.log"

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
PARAMS=(
  -P tb_qfit_local5_projection_tile.HEIGHT=15
  -P tb_qfit_local5_projection_tile.WIDTH=15
  -P tb_qfit_local5_projection_tile.TIME_PLANES=2
  -P tb_qfit_local5_projection_tile.HEAD_DIM=32
  -P tb_qfit_local5_projection_tile.OUT_DIM=2
)
ORACLE_ARGS=(
  "+PY_INPUTS=${BUILD}/t450_inputs.txt"
  "+PY_EXPECTED=${BUILD}/t450_expected.txt"
)

for spec in "tcfm5:0" "linear5:2"; do
  name="${spec%%:*}"
  kind="${spec##*:}"
  iverilog -g2012 -s tb_qfit_local5_projection_tile \
    "${PARAMS[@]}" \
    -P "tb_qfit_local5_projection_tile.BACKEND_KIND=${kind}" \
    -o "${BUILD}/${name}.vvp" \
    "${COMMON[@]}" "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
  vvp "${BUILD}/${name}.vvp" "${ORACLE_ARGS[@]}" \
    | tee "${OUT}/${name}_iverilog.log"
done

for seed in 1 17717 48879; do
  vvp "${BUILD}/tcfm5.vvp" "${ORACLE_ARGS[@]}" \
    "+TERM_STALL_SEED=${seed}" \
    | tee "${OUT}/tcfm5_backpressure_seed_${seed}_iverilog.log"
done

for mode in 2 3 4; do
  vvp "${BUILD}/tcfm5.vvp" "${ORACLE_ARGS[@]}" \
    "+TERM_STALL_MODE=${mode}" \
    | tee "${OUT}/tcfm5_directed_mode_${mode}_iverilog.log"
done

for seed in 0x45052027 0x45052028; do
  tag="${seed#0x}"
  vvp "${BUILD}/tcfm5.vvp" \
    "+PY_INPUTS=${BUILD}/t450_inputs_${tag}.txt" \
    "+PY_EXPECTED=${BUILD}/t450_expected_${tag}.txt" \
    | tee "${OUT}/tcfm5_data_seed_${tag}_iverilog.log"
done

iverilog -g2012 -s tb_qfit_local5_projection_tile \
  -P tb_qfit_local5_projection_tile.HEIGHT=15 \
  -P tb_qfit_local5_projection_tile.WIDTH=15 \
  -P tb_qfit_local5_projection_tile.TIME_PLANES=2 \
  -P tb_qfit_local5_projection_tile.HEAD_DIM=32 \
  -P tb_qfit_local5_projection_tile.OUT_DIM=32 \
  -P tb_qfit_local5_projection_tile.BACKEND_KIND=0 \
  -o "${BUILD}/tcfm5_out32.vvp" \
  "${COMMON[@]}" "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
vvp "${BUILD}/tcfm5_out32.vvp" \
  "+PY_INPUTS=${BUILD}/t450_out32_inputs.txt" \
  "+PY_EXPECTED=${BUILD}/t450_out32_expected.txt" \
  | tee "${OUT}/tcfm5_out32_iverilog.log"

rm -rf "${BUILD}/verilator"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_projection_tile \
  --Mdir "${BUILD}/verilator" \
  -GHEIGHT=15 -GWIDTH=15 -GTIME_PLANES=2 -GHEAD_DIM=32 -GOUT_DIM=2 \
  -GBACKEND_KIND=0 \
  "${COMMON[@]}" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
"${BUILD}/verilator/Vtb_qfit_local5_projection_tile" "${ORACLE_ARGS[@]}" \
  | tee "${OUT}/tcfm5_verilator_sva.log"
"${BUILD}/verilator/Vtb_qfit_local5_projection_tile" "${ORACLE_ARGS[@]}" \
  +TERM_STALL_SEED=17717 \
  | tee "${OUT}/tcfm5_backpressure_verilator_sva.log"
for seed in 1 48879; do
  "${BUILD}/verilator/Vtb_qfit_local5_projection_tile" "${ORACLE_ARGS[@]}" \
    "+TERM_STALL_SEED=${seed}" \
    | tee "${OUT}/tcfm5_backpressure_seed_${seed}_verilator_sva.log"
done
for seed in 0x45052027 0x45052028; do
  tag="${seed#0x}"
  "${BUILD}/verilator/Vtb_qfit_local5_projection_tile" \
    "+PY_INPUTS=${BUILD}/t450_inputs_${tag}.txt" \
    "+PY_EXPECTED=${BUILD}/t450_expected_${tag}.txt" \
    | tee "${OUT}/tcfm5_data_seed_${tag}_verilator_sva.log"
done
for mode in 2 3 4; do
  "${BUILD}/verilator/Vtb_qfit_local5_projection_tile" "${ORACLE_ARGS[@]}" \
    "+TERM_STALL_MODE=${mode}" \
    | tee "${OUT}/tcfm5_directed_mode_${mode}_verilator_sva.log"
done

rm -rf "${BUILD}/verilator_out32"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_local5_projection_tile \
  --Mdir "${BUILD}/verilator_out32" \
  -GHEIGHT=15 -GWIDTH=15 -GTIME_PLANES=2 -GHEAD_DIM=32 -GOUT_DIM=32 \
  -GBACKEND_KIND=0 \
  "${COMMON[@]}" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv"
"${BUILD}/verilator_out32/Vtb_qfit_local5_projection_tile" \
  "+PY_INPUTS=${BUILD}/t450_out32_inputs.txt" \
  "+PY_EXPECTED=${BUILD}/t450_out32_expected.txt" \
  | tee "${OUT}/tcfm5_out32_verilator_sva.log"

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${SYNTH[*]};
  chparam -set HEIGHT 15 -set WIDTH 15 -set TIME_PLANES 2 -set HEAD_DIM 32 -set OUT_DIM 32 -set BACKEND_KIND 0 qfit_local5_projection_tile;
  hierarchy -top qfit_local5_projection_tile;
  proc; opt; memory_collect; check -assert;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/stat.json stat -json;
  write_json ${OUT}/yosys_netlist.json
"

sha256sum \
  "${BUILD}"/t450_*.txt \
  >"${OUT}/oracle_hashes.sha256"
sha256sum -c "${OUT}/oracle_hashes.sha256" \
  >"${OUT}/oracle_hash_check.log"

sha256sum \
  "${COMMON[@]}" \
  "${ROOT}/tb_qfit/tb_qfit_local5_projection_tile.sv" \
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_sync_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "${ROOT}/verif_qfit/qfit_local5_projection_tile_assertions.sv" \
  "${ROOT}/scripts/generate_local5_t450_fullchain_oracle.py" \
  "${ROOT}/scripts/generate_local5_fullchain_oracle.py" \
  "${ROOT}/scripts/report_qfit_local5_projection_t450_shell.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_projection_t450_shell_checks.sh" \
  >"${OUT}/source_hashes.sha256"
sha256sum -c "${OUT}/source_hashes.sha256" \
  >"${OUT}/source_hash_check.log"

python3 "${ROOT}/scripts/report_qfit_local5_projection_t450_shell.py"
printf 'Synthetic T450 Python-to-Acc32 Icarus/Verilator-SVA\tPASS\n' \
  >"${OUT}/status.tsv"
printf 'T450 fixed-seed backpressure cross-simulator\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'T450 multi-data-seed and directed-stall regression\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'T450 OUT32 Python-to-Acc32 production-width smoke\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'T450 Linear-5 equal-bank RTL baseline\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'T450 Yosys parameterized check\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'T450 oracle/source SHA-256 self-check\tPASS\n' \
  >>"${OUT}/status.tsv"
printf 'PASS Local5 synthetic T450 deployment shell\n'
