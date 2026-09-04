#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m9_dual_granularity_state_vcs_sva_20260821}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  -top tb_qfit_dual_granularity_temporal_state_engine \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_granularity_temporal_state_engine.sv" \
  -o simv 2>&1 | tee compile.log
./simv -assert report="$RUN_DIR/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M9_1_RESULT wide=12 narrow=14 wide_local=2 wide_motion=10 narrow_local=8 narrow_motion=6 abort=1 wide_errors=1 narrow_errors=1 rmw_stalls=3 reset_block_checks=3 domain_fault_checks=1" simulation.log
grep -q "PASS: Synopsys VCS M9.1 SRAM-realistic atomic Local/Motion shared temporal state exact" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt
sha256sum \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/dc_handoff/scripts/run_vcs_m9_dual_granularity_state_sva.sh" \
  compile.log simulation.log assertion_report.txt > evidence.sha256
echo "PASS Synopsys VCS/SVA M9.1 dual-granularity temporal state"
