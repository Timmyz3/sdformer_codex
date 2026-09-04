#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/qfit_projection_baseline_matrix_20260731"
mkdir -p "${OUT}"

run_one() {
  local name="$1"
  local top="$2"
  local source="$3"
  yosys -q -l "${OUT}/${name}_yosys.log" -p "
    read_verilog -sv \
      ${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv \
      ${ROOT}/rtl_qfit/${source};
    hierarchy -top ${top};
    flatten;
    proc; opt; memory_collect; memory_dff; opt;
    check -assert;
    tee -o ${OUT}/${name}_stat.json stat -json
  "
  grep -q 'Found and reported 0 problems.' "${OUT}/${name}_yosys.log"
}

run_one \
  tcfm5 \
  qfit_tcfm5_projection_top \
  qfit_tcfm5_projection_top.sv
run_one \
  affine4 \
  qfit_affine4_projection_top \
  qfit_affine4_projection_top.sv
run_one \
  linear5 \
  qfit_linear5_projection_top \
  qfit_linear5_projection_top.sv
run_one \
  role_sharded \
  qfit_role_sharded_projection_top \
  qfit_role_sharded_projection_top.sv

python3 "${ROOT}/scripts/report_qfit_projection_baseline_matrix.py" \
  --input-dir "${OUT}" \
  --output-dir "${OUT}" \
  --cycle-evidence \
    "${ROOT}/results/qfit_local5_projection_tile_yosys_20260731/cycle_evidence.json"

sha256sum \
  "${ROOT}/rtl_qfit/qfit_tcfm5_acc_bank.sv" \
  "${ROOT}/rtl_qfit/qfit_tcfm5_projection_top.sv" \
  "${ROOT}/rtl_qfit/qfit_affine4_projection_top.sv" \
  "${ROOT}/rtl_qfit/qfit_linear5_projection_top.sv" \
  "${ROOT}/rtl_qfit/qfit_role_sharded_projection_top.sv" \
  "${ROOT}/scripts/report_qfit_projection_baseline_matrix.py" \
  "${ROOT}/sim_qfit/run_qfit_projection_baseline_structural_compare.sh" \
  >"${OUT}/input_sha256.txt"

printf 'PASS qfit Local5 projection baseline structural comparison\n'
