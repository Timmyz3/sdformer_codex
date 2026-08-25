#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m8_temporal_destination_commit_vcs_sva_20260821}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  -top tb_qfit_temporal_destination_commit_engine \
  "$ROOT/rtl_qfit/qfit_temporal_destination_commit_engine.sv" \
  "$ROOT/verif_qfit/qfit_temporal_destination_commit_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_temporal_destination_commit_engine.sv" \
  -o simv 2>&1 | tee compile.log
./simv -assert report="$RUN_DIR/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "TEMPORAL_COMMIT_RESULT legal=409 local=183 motion=226 rejected=21 protocol_errors=21 abort=1 abort_rejected=2 reset_blocked=1" simulation.log
grep -q "M8_SVA_COVERAGE accepted=409 local=183 motion=226 rejected=21 abort=1 abort_rejected=2" simulation.log
grep -q "PASS: Synopsys VCS M8.2 reset-fenced Local/Motion temporal destination commit exact" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt
sha256sum "$ROOT/rtl_qfit/qfit_temporal_destination_commit_engine.sv" \
  "$ROOT/verif_qfit/qfit_temporal_destination_commit_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_temporal_destination_commit_engine.sv" \
  "$ROOT/dc_handoff/scripts/run_vcs_m8_temporal_commit_sva.sh" \
  compile.log simulation.log assertion_report.txt > evidence.sha256
echo "PASS Synopsys VCS/SVA M8 temporal destination commit"
