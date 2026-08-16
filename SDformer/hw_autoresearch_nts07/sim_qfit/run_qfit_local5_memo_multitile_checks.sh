#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_qfit/local5_memo_multitile"
OUT="${ROOT}/results/qfit_local5_memo_multitile_20260809"
ORACLE="${BUILD}/oracle"
mkdir -p "${BUILD}" "${OUT}" "${ORACLE}"
rm -rf "${BUILD}/obj_memo" "${BUILD}/obj_baseline"
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
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_fcsr_relation_memo_projection_top.sv"
  "${ROOT}/rtl_qfit/qfit_local5_memo_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_projection_tile.sv"
  "${ROOT}/rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${ROOT}/rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_relation_transpose_assertions.sv"
  "${ROOT}/verif_qfit/qfit_exposure_relation_vault_assertions.sv"
  "${ROOT}/verif_qfit/qfit_source_multicast_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_assertions.sv"
  "${ROOT}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_memo_tagged_t450_job_engine_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv"
TB_FAULT="${ROOT}/tb_qfit/tb_qfit_local5_cross_head_partial_faults.sv"
SEEDS=(17717 44257 48879)

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

for mode in memo baseline; do
  value=1
  if [[ "${mode}" == baseline ]]; then value=0; fi
  iverilog -g2012 \
    -Ptb_qfit_local5_memo_multitile_cross_head.USE_MEMO="${value}" \
    -s tb_qfit_local5_memo_multitile_cross_head \
    -o "${BUILD}/${mode}_iv" "${RTL[@]}" "${TB}" \
    >"${OUT}/iverilog_${mode}_build.log" 2>&1
  for seed in "${SEEDS[@]}"; do
    vvp "${BUILD}/${mode}_iv" "${ARGS[@]}" "+SERVICE_SEED=${seed}" \
      | tee "${OUT}/${mode}_seed_${seed}_iverilog.log"
  done
done

iverilog -g2012 -s tb_qfit_local5_cross_head_partial_faults \
  -o "${BUILD}/fault_iv" "${RTL[@]}" "${TB_FAULT}" \
  >"${OUT}/iverilog_fault_build.log" 2>&1
for fault in 0 1 2 3; do
  vvp "${BUILD}/fault_iv" "+FAULT_MODE=${fault}" \
    | tee "${OUT}/partial_fault_${fault}_iverilog.log"
done

for mode in memo baseline; do
  value=1
  if [[ "${mode}" == baseline ]]; then value=0; fi
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
    --top-module tb_qfit_local5_memo_multitile_cross_head \
    -GUSE_MEMO="${value}" --Mdir "${BUILD}/obj_${mode}" \
    "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
    >"${OUT}/verilator_${mode}_build.log" 2>&1
  for seed in "${SEEDS[@]}"; do
    "${BUILD}/obj_${mode}/Vtb_qfit_local5_memo_multitile_cross_head" \
      "${ARGS[@]}" "+SERVICE_SEED=${seed}" \
      | tee "${OUT}/${mode}_seed_${seed}_verilator_sva.log"
  done
done

verilator --lint-only --timing -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module qfit_local5_cross_head_tile_executor \
  -GUSE_RELATION_MEMO=1 "${RTL[@]}" \
  >"${OUT}/verilator_memo_executor_lint.log" 2>&1

yosys -q -l "${OUT}/yosys_memo_executor.log" -p "
  read_verilog -sv ${RTL[*]};
  chparam -set USE_RELATION_MEMO 1 qfit_local5_cross_head_tile_executor;
  hierarchy -check -top qfit_local5_cross_head_tile_executor;
  proc; opt; memory_collect; check -assert;
  tee -o ${OUT}/memo_executor_stat.json stat -json;
  flatten; opt; memory_collect; check -assert;
  tee -o ${OUT}/memo_executor_flat_stat.json stat -json
"

python3 "${ROOT}/scripts/report_qfit_local5_memo_multitile.py"
sha256sum "${ORACLE}"/* "${RTL[@]}" "${ASSERTIONS[@]}" \
  "${TB}" "${TB_FAULT}" \
  "${ROOT}/scripts/generate_local5_memo_multitile_oracle.py" \
  "${ROOT}/scripts/report_qfit_local5_memo_multitile.py" \
  "${ROOT}/sim_qfit/run_qfit_local5_memo_multitile_checks.sh" \
  >"${OUT}/source_sha256.txt"

printf '三 seed memo/recompute Icarus bit-exact 对照\tPASS\n' >"${OUT}/status.tsv"
printf '三 seed memo/recompute Verilator/SVA 对照\tPASS\n' >>"${OUT}/status.tsv"
printf '四种 partial 故障 fail-closed\tPASS\n' >>"${OUT}/status.tsv"
printf 'memo executor lint 与 Yosys 开放映射\tPASS\n' >>"${OUT}/status.tsv"
printf 'PASS Local5 memo multi-tile checks\n'
