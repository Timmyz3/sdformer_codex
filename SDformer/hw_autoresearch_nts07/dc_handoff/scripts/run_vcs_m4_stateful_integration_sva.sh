#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m4_stateful_integration_vcs_sva_20260821}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all \
  -timescale=1ns/1ps +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  -top tb_qfit_dual_line_descriptor_stateful_engine \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_stateful_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_stateful_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_stateful_engine.sv" \
  -o simv 2>&1 | tee compile.log

./simv -assert report="$RUN_DIR/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M4_STATE_RESULT outputs=8 local=4 motion=4" simulation.log
grep -q "PASS: M4 descriptor-resident Local absolute plus Motion delta shared-state miter exact" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" \
  simulation.log assertion_report.txt
sha256sum \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_stateful_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_stateful_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_stateful_engine.sv" \
  "$ROOT/dc_handoff/scripts/run_vcs_m4_stateful_integration_sva.sh" \
  simv compile.log simulation.log assertion_report.txt > evidence.sha256
echo "PASS Synopsys VCS/SVA M4 stateful Local/Motion integration"
