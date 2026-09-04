#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhumd/work/sdformer_codex/SDformer}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/dual_line_stateful_tile_vcs_sva_20260821}"

source /home/zhumd/work/synopsys_date_dual/env.sh
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

vcs -full64 -sverilog -debug_access+all -assert svaext \
  -timescale=1ns/1ps \
  "${REPO_ROOT}/hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_tile_selector.sv" \
  "${REPO_ROOT}/hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_source_streamer.sv" \
  "${REPO_ROOT}/hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_tile_executor.sv" \
  "${REPO_ROOT}/hw_autoresearch_nts07/rtl_qfit/qfit_dual_line_stateful_tile_top.sv" \
  "${REPO_ROOT}/hw_autoresearch_nts07/verif_qfit/qfit_dual_line_stateful_tile_assertions.sv" \
  "${REPO_ROOT}/hw_autoresearch_nts07/tb_qfit/tb_qfit_dual_line_stateful_tile.sv" \
  -top tb_qfit_dual_line_stateful_tile \
  -o simv \
  -l compile.log

./simv -l simulation.log
sha256sum compile.log simulation.log > evidence.sha256
