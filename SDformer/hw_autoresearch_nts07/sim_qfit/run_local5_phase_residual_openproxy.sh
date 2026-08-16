#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/local5_phase_residual_openproxy_20260814}"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
BUILD="$OUT/build"
LOGS="$OUT/logs"
mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
)

bash sim_qfit/run_qfit_score_leaf_checks.sh >"$LOGS/score_leaf_regression.log" 2>&1
grep -q '^PASS Verilator SVA simulation$' "$LOGS/score_leaf_regression.log"

cat >"$OUT/constraint_3ns.sdc" <<'SDC'
create_clock -name clk_core -period 3.000 [get_ports clk_core]
set_input_delay 0.200 -clock clk_core [get_ports {rst_core in_valid in_tag in_q in_k in_valid_mask out_ready}]
set_output_delay 0.200 -clock clk_core [get_ports {in_ready out_valid out_tag out_score_q7 out_gate_q17 out_k_self out_valid_mask perf_service_cycles perf_route_direct_mask}]
set_load 0.010 [get_ports {in_ready out_valid out_tag out_score_q7 out_gate_q17 out_k_self out_valid_mask perf_service_cycles perf_route_direct_mask}]
SDC

cat >"$OUT/run_sta.tcl" <<'TCL'
read_liberty $env(LIB)
read_verilog $env(NETLIST)
link_design qfit_local5_score_leaf
read_sdc $env(SDC)
check_setup
report_checks -path_delay max -group_count 5 -endpoint_count 5 -fields {slew cap input_pins} -format full_clock_expanded -digits 6
TCL

for variant in absolute phase_residual; do
  phase=0
  if [[ "$variant" == "phase_residual" ]]; then phase=1; fi
  yosys -l "$LOGS/nangate45_${variant}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    chparam -set ARCH_QFSA 1 -set PIPE_COMPACTOR 1 -set XBF_BANKED 1 -set ARCH_PHASE_RESIDUAL $phase -set USE_THRESHOLD_ROUTE 1 -set ROUTE_THRESHOLD 8 -set USE_BANK_PRESSURE_ROUTE 1 -set BANK_PRESSURE_THRESHOLD 2 qfit_local5_score_leaf;
    hierarchy -check -top qfit_local5_score_leaf;
    proc; flatten; opt; memory; opt; techmap; opt;
    dfflibmap -liberty $LIB; abc -D 3000 -liberty $LIB;
    clean; check -assert; stat -liberty $LIB;
    write_verilog -noattr $BUILD/${variant}_mapped.v
  " >/dev/null
  LIB="$LIB" NETLIST="$BUILD/${variant}_mapped.v" SDC="$OUT/constraint_3ns.sdc" \
    sta "$OUT/run_sta.tcl" >"$LOGS/sta_${variant}.log" 2>&1
  grep -q 'data arrival time' "$LOGS/sta_${variant}.log"
done

python3 -m unittest scripts.test_report_local5_phase_residual_openproxy \
  >"$LOGS/python_unit_tests.log" 2>&1
python3 scripts/report_local5_phase_residual_openproxy.py --output-dir "$OUT" \
  >"$LOGS/report_stdout.log" 2>&1

sha256sum "${RTL[@]}" tb_qfit/tb_qfit_local5_score_leaf.sv \
  verif_qfit/qfit_score_leaf_assertions.sv \
  scripts/report_local5_phase_residual_openproxy.py \
  scripts/test_report_local5_phase_residual_openproxy.py \
  sim_qfit/run_local5_phase_residual_openproxy.sh "$LIB" \
  >"$OUT/source_input_sha256.txt"
sha256sum "$OUT"/logs/*.log "$OUT"/build/*.v \
  "$OUT"/report.json "$OUT"/report.md "$OUT"/constraint_3ns.sdc \
  "$OUT"/run_sta.tcl "$OUT"/source_input_sha256.txt \
  >"$OUT/result_sha256.txt"

echo 'PASS Local5 phase-residual open proxy'
