#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_hitflow/vectors/h67_fullres_ep35_postconvergence_t450_20260805_checkpoint_atlif}"
IDENTITY="${IDENTITY:-h67_ep35}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m7_atlif_l16_${IDENTITY}_vcs_sva_20260821}"

test -s "$VECTOR_DIR/manifest.json"
test -s "$VECTOR_DIR/vector_contract.svh"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

vcs -full64 -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  +incdir+"$VECTOR_DIR" \
  -top tb_checkpoint_atlif_dptme_l16_segmented \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/verif_hitflow/hitflow_dptme_assertions.sv" \
  "$ROOT/verif_hitflow/bind_hitflow_dptme_assertions.sv" \
  "$ROOT/tb_hitflow/tb_checkpoint_atlif_dptme_l16_segmented.sv" \
  -o simv 2>&1 | tee compile.log

(cd "$VECTOR_DIR" && "$RUN_DIR/simv" +ntb_random_seed=20260821) \
  2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "t10_segment_commands=90 t2_segment_commands=72" simulation.log
grep -q "hidden=25920 hidden_mismatches=0 events=25920 event_mismatches=0 sampled_protocol_errors=0" simulation.log
grep -q "PASS: Synopsys VCS checkpoint-bound ATLIF L16 S10/S2 exact" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log

mkdir -p "$RUN_DIR/directed_protocol"
cd "$RUN_DIR/directed_protocol"
vcs -full64 -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  -top tb_hitflow_dptme_array \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/verif_hitflow/hitflow_dptme_assertions.sv" \
  "$ROOT/verif_hitflow/bind_hitflow_dptme_assertions.sv" \
  "$ROOT/tb_hitflow/tb_hitflow_dptme_array.sv" \
  -o simv 2>&1 | tee compile.log
./simv -assert report="$RUN_DIR/directed_protocol/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log
grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "DPTME_PROTOCOL_RESULT sampled_protocol_errors=3 tag_reject=1 early_last_reject=1 single_step_reject=1 state_advance_errors=0" simulation.log
grep -q "PASS: HIT-Flow DP-TME array" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt

cd "$RUN_DIR"
sha256sum "$VECTOR_DIR/manifest.json" "$VECTOR_DIR/vector_contract.svh" \
  "$VECTOR_DIR/meta.mem" "$VECTOR_DIR/x.mem" "$VECTOR_DIR/weight.mem" \
  "$VECTOR_DIR/bias.mem" "$VECTOR_DIR/threshold.mem" \
  "$VECTOR_DIR/expected_hidden.mem" "$VECTOR_DIR/expected_event.mem" \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/verif_hitflow/hitflow_dptme_assertions.sv" \
  "$ROOT/verif_hitflow/bind_hitflow_dptme_assertions.sv" \
  "$ROOT/tb_hitflow/tb_checkpoint_atlif_dptme_l16_segmented.sv" \
  "$ROOT/tb_hitflow/tb_hitflow_dptme_array.sv" \
  "$ROOT/dc_handoff/scripts/run_vcs_m7_atlif_l16_checkpoint.sh" \
  compile.log simulation.log directed_protocol/compile.log \
  directed_protocol/simulation.log directed_protocol/assertion_report.txt \
  > evidence.sha256

echo "PASS Synopsys VCS/SVA M7 ATLIF L16 checkpoint regression identity=$IDENTITY"
