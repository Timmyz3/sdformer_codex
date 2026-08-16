#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_legal1rw_inplace"
OUT="${ROOT}/results/qfit_local5_legal1rw_inplace_20260810"
ORACLE="${BUILD}/oracle"
mkdir -p "${BUILD}" "${OUT}" "${ORACLE}"
rm -rf "${BUILD}"/obj_*
rm -f "${OUT}"/*.log "${OUT}"/*.json "${OUT}"/*.txt \
  "${OUT}"/*.tsv "${OUT}"/*.md

RTL=(
  "${ROOT}/rtl_hitflow/gatestack_output_tile_scheduler.sv"
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_retirement_scheduler.sv"
  "${ROOT}/rtl_qfit/qfit_sync_1r1w_bank.sv"
  "${ROOT}/rtl_qfit/qfit_relation_transpose_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_sync_relation_bank.sv"
  "${ROOT}/rtl_qfit/qfit_exposure_relation_vault.sv"
  "${ROOT}/rtl_qfit/qfit_fcsr_relation_memo_top.sv"
  "${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${ROOT}/rtl_qfit/qfit_direct_1rw_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_fcsr_relation_memo_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_memo_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv"
  "${ROOT}/verif_qfit/qfit_exposure_relation_vault_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_single_port_acc_memory_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_memo_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv"
TB_LEAF="${ROOT}/tb_qfit/tb_qfit_tcfm5_inplace_accumulate.sv"
SEEDS=(17717 44257 48879)
CANDIDATES=(b0_scalar_recompute b1_scalar_memo b2_inplace_recompute b3_inplace_memo)

python3 "${ROOT}/scripts/generate_local5_memo_multitile_oracle.py" \
  --out-dir "${ORACLE}" >"${OUT}/oracle.log"
ARGS=(
  "+INPUT_H0=${ORACLE}/head0_inputs.txt"
  "+INPUT_H1=${ORACLE}/head1_inputs.txt"
  "+INPUT_H2=${ORACLE}/head2_inputs.txt"
  "+EXPECTED=${ORACLE}/expected_all_tiles.txt"
)

{
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  yosys -V
  uname -srvmo
} >"${OUT}/tool_versions.txt"

iverilog -g2012 -s tb_qfit_tcfm5_inplace_accumulate \
  -Ptb_qfit_tcfm5_inplace_accumulate.ACC_BACKEND_KIND=1 \
  -o "${BUILD}/leaf_iv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv" \
  "${ROOT}/rtl_qfit/qfit_direct_1rw_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" "${TB_LEAF}" \
  >"${OUT}/iverilog_leaf_build.log" 2>&1
for mode in 0 1; do
  vvp "${BUILD}/leaf_iv" "+MODE=${mode}" \
    >"${OUT}/leaf_mode_${mode}_iverilog.log" 2>&1
  grep -q '^PASS TCFM5 ' "${OUT}/leaf_mode_${mode}_iverilog.log"
done

for candidate in "${CANDIDATES[@]}"; do
  memo=0
  inplace=0
  [[ "${candidate}" == *memo ]] && memo=1
  [[ "${candidate}" == *inplace* ]] && inplace=1
  iverilog -g2012 \
    -Ptb_qfit_local5_memo_multitile_cross_head.USE_MEMO="${memo}" \
    -Ptb_qfit_local5_memo_multitile_cross_head.USE_INPLACE="${inplace}" \
    -Ptb_qfit_local5_memo_multitile_cross_head.ACC_BACKEND_KIND=1 \
    -Ptb_qfit_local5_memo_multitile_cross_head.TRANSACTION_INDEXED_SERVICE=1 \
    -s tb_qfit_local5_memo_multitile_cross_head \
    -o "${BUILD}/${candidate}_iv" "${RTL[@]}" "${TB}" \
    >"${OUT}/iverilog_${candidate}_build.log" 2>&1
  for seed in "${SEEDS[@]}"; do
    vvp "${BUILD}/${candidate}_iv" "${ARGS[@]}" "+SERVICE_SEED=${seed}" \
      >"${OUT}/${candidate}_seed_${seed}_iverilog.log" 2>&1
    grep -q '^PASS Local5 multi-tile ' \
      "${OUT}/${candidate}_seed_${seed}_iverilog.log"
  done
done

for candidate in "${CANDIDATES[@]}"; do
  memo=0
  inplace=0
  [[ "${candidate}" == *memo ]] && memo=1
  [[ "${candidate}" == *inplace* ]] && inplace=1
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
    --top-module tb_qfit_local5_memo_multitile_cross_head \
    -GUSE_MEMO="${memo}" -GUSE_INPLACE="${inplace}" \
    -GACC_BACKEND_KIND=1 -GTRANSACTION_INDEXED_SERVICE=1 \
    --Mdir "${BUILD}/obj_${candidate}" \
    "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
    >"${OUT}/verilator_${candidate}_build.log" 2>&1
  for seed in "${SEEDS[@]}"; do
    "${BUILD}/obj_${candidate}/Vtb_qfit_local5_memo_multitile_cross_head" \
      "${ARGS[@]}" "+SERVICE_SEED=${seed}" \
      >"${OUT}/${candidate}_seed_${seed}_verilator_sva.log" 2>&1
    grep -q '^PASS Local5 multi-tile ' \
      "${OUT}/${candidate}_seed_${seed}_verilator_sva.log"
  done
done

for candidate in "${CANDIDATES[@]}"; do
  memo=0
  inplace=0
  [[ "${candidate}" == *memo ]] && memo=1
  [[ "${candidate}" == *inplace* ]] && inplace=1
  yosys -q -l "${OUT}/yosys_${candidate}.log" -p "
    read_verilog -sv ${RTL[*]};
    chparam -set USE_RELATION_MEMO ${memo} qfit_local5_cross_head_tile_executor;
    chparam -set USE_INPLACE_CROSS_HEAD_ACC ${inplace} qfit_local5_cross_head_tile_executor;
    chparam -set ACC_BACKEND_KIND 1 qfit_local5_cross_head_tile_executor;
    hierarchy -check -top qfit_local5_cross_head_tile_executor;
    proc; opt; memory_collect; check -assert;
    flatten; opt; memory_collect; check -assert
  "
done

sha256sum "${ORACLE}"/* "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${TB}" "${TB_LEAF}" \
  "${ROOT}/scripts/generate_local5_memo_multitile_oracle.py" \
  "${ROOT}/scripts/report_qfit_local5_legal1rw_inplace.py" \
  "${ROOT}/tests/test_report_qfit_local5_legal1rw_inplace.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_legal1rw_inplace_checks.sh" \
  >"${OUT}/source_sha256.txt"

python3 -m unittest tests.test_report_qfit_local5_legal1rw_inplace \
  >"${OUT}/report_unittest.log" 2>&1
python3 "${ROOT}/scripts/report_qfit_local5_legal1rw_inplace.py" \
  --output-dir "${OUT}"

printf '合法1RW叶级原位与fail-closed\tPASS\n' >"${OUT}/status.tsv"
printf 'B0-B3三seed Icarus bit-exact\tPASS\n' >>"${OUT}/status.tsv"
printf 'B0-B3三seed Verilator/SVA bit-exact\tPASS\n' >>"${OUT}/status.tsv"
printf 'transaction-indexed服务账本\tPASS\n' >>"${OUT}/status.tsv"
printf 'B0-B3 Yosys综合可读\tPASS\n' >>"${OUT}/status.tsv"
printf 'PASS Local5 legal-1RW in-place checks\n'
