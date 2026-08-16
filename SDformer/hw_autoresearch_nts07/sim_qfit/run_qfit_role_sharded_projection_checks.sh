#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d /tmp/qfit_role_sharded_projection.XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

ACC_RTL="${ROOT_DIR}/rtl_qfit/qfit_tcfm5_acc_bank.sv"
TOP_RTL="${ROOT_DIR}/rtl_qfit/qfit_role_sharded_projection_top.sv"
TB="${ROOT_DIR}/tb_qfit/tb_qfit_role_sharded_projection_top.sv"
TOP_SVA="${ROOT_DIR}/verif_qfit/qfit_role_sharded_projection_assertions.sv"
ACC_SVA="${ROOT_DIR}/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"

iverilog -g2012 \
  -s tb_qfit_role_sharded_projection_top \
  -o "${BUILD_DIR}/role_sharded.vvp" \
  "${ACC_RTL}" \
  "${TOP_RTL}" \
  "${TB}"
vvp "${BUILD_DIR}/role_sharded.vvp"

verilator --lint-only --Wall -Wno-fatal \
  --top-module qfit_role_sharded_projection_top \
  "${ACC_RTL}" \
  "${TOP_RTL}"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb_qfit_role_sharded_projection_top \
  --Mdir "${BUILD_DIR}/obj_role_sharded" \
  "${ACC_RTL}" \
  "${TOP_RTL}" \
  "${TB}" \
  "${TOP_SVA}" \
  "${ACC_SVA}" \
  --exe
"${BUILD_DIR}/obj_role_sharded/Vtb_qfit_role_sharded_projection_top"

yosys -q -l "${BUILD_DIR}/yosys.log" -p "
  read_verilog -sv ${ACC_RTL} ${TOP_RTL};
  hierarchy -check -top qfit_role_sharded_projection_top;
  proc; opt; memory_collect; memory_dff; opt;
  check -assert;
  stat
"

printf '%s\n' \
  "PASS role-sharded projection: Icarus exact scoreboard, Verilator SVA, Yosys check"
