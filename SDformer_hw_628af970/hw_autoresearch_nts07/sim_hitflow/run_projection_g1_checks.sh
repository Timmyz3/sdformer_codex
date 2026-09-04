#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/projection_g1"
mkdir -p "$BUILD"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_hitflow_nmf_g1_builder \
  -o "$BUILD/tb_nmf_g1.vvp" \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  tb_hitflow/tb_hitflow_nmf_g1_builder.sv
vvp "$BUILD/tb_nmf_g1.vvp"

iverilog -g2012 -Wall -s tb_hitflow_gate_product_engine \
  -o "$BUILD/tb_product.vvp" \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  tb_hitflow/tb_hitflow_gate_product_engine.sv
vvp "$BUILD/tb_product.vvp"

iverilog -g2012 -Wall -s tb_hitflow_segmented_multicast \
  -o "$BUILD/tb_multicast.vvp" \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  tb_hitflow/tb_hitflow_segmented_multicast.sv
vvp "$BUILD/tb_multicast.vvp"

iverilog -g2012 -Wall -s tb_hitflow_banked_accumulator \
  -o "$BUILD/tb_accumulator.vvp" \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  tb_hitflow/tb_hitflow_banked_accumulator.sv
vvp "$BUILD/tb_accumulator.vvp"

iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top \
  -o "$BUILD/tb_g1_top.vvp" \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  rtl_hitflow/hitflow_g1_projection_top.sv \
  tb_hitflow/tb_hitflow_g1_projection_top.sv
vvp "$BUILD/tb_g1_top.vvp"

verilator --lint-only --sv -Wall --Wno-fatal \
  --top-module hitflow_nmf_g1_builder \
  rtl_hitflow/hitflow_nmf_g1_builder.sv
verilator --lint-only --sv -Wall --Wno-fatal \
  --top-module hitflow_gate_product_engine \
  rtl_hitflow/hitflow_gate_product_engine.sv
verilator --lint-only --sv -Wall --Wno-fatal \
  --top-module hitflow_segmented_multicast \
  rtl_hitflow/hitflow_segmented_multicast.sv
verilator --lint-only --sv -Wall --Wno-fatal \
  --top-module hitflow_banked_accumulator \
  rtl_hitflow/hitflow_banked_accumulator.sv
verilator --lint-only --sv -Wall --Wno-fatal --Wno-UNOPTFLAT \
  --top-module hitflow_g1_projection_top \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  rtl_hitflow/hitflow_g1_projection_top.sv

NMF_ASSERT_BUILD="$BUILD/verilator_nmf_assertions"
rm -rf "$NMF_ASSERT_BUILD"
verilator --binary --assert --timing --sv -Wall --Wno-fatal \
  -Wno-BLKSEQ --top-module tb_hitflow_nmf_g1_builder \
  --Mdir "$NMF_ASSERT_BUILD" \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  verif_hitflow/hitflow_nmf_g1_assertions.sv \
  verif_hitflow/bind_hitflow_nmf_g1_assertions.sv \
  tb_hitflow/tb_hitflow_nmf_g1_builder.sv
"$NMF_ASSERT_BUILD/Vtb_hitflow_nmf_g1_builder"

PRODUCT_ASSERT_BUILD="$BUILD/verilator_product_assertions"
rm -rf "$PRODUCT_ASSERT_BUILD"
verilator --binary --assert --timing --sv -Wall --Wno-fatal \
  --top-module tb_hitflow_gate_product_engine \
  --Mdir "$PRODUCT_ASSERT_BUILD" \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  verif_hitflow/hitflow_gate_product_assertions.sv \
  verif_hitflow/bind_hitflow_gate_product_assertions.sv \
  tb_hitflow/tb_hitflow_gate_product_engine.sv
"$PRODUCT_ASSERT_BUILD/Vtb_hitflow_gate_product_engine"

MULTICAST_ASSERT_BUILD="$BUILD/verilator_multicast_assertions"
rm -rf "$MULTICAST_ASSERT_BUILD"
verilator --binary --assert --timing --sv -Wall --Wno-fatal \
  --top-module tb_hitflow_segmented_multicast \
  --Mdir "$MULTICAST_ASSERT_BUILD" \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  verif_hitflow/hitflow_segmented_multicast_assertions.sv \
  verif_hitflow/bind_hitflow_segmented_multicast_assertions.sv \
  tb_hitflow/tb_hitflow_segmented_multicast.sv
"$MULTICAST_ASSERT_BUILD/Vtb_hitflow_segmented_multicast"

ACC_ASSERT_BUILD="$BUILD/verilator_accumulator_assertions"
rm -rf "$ACC_ASSERT_BUILD"
verilator --binary --assert --timing --sv -Wall --Wno-fatal \
  --top-module tb_hitflow_banked_accumulator \
  --Mdir "$ACC_ASSERT_BUILD" \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  verif_hitflow/hitflow_banked_accumulator_assertions.sv \
  verif_hitflow/bind_hitflow_banked_accumulator_assertions.sv \
  tb_hitflow/tb_hitflow_banked_accumulator.sv
"$ACC_ASSERT_BUILD/Vtb_hitflow_banked_accumulator"

yosys -q -l "$BUILD/yosys_nmf_g1.log" \
  -p 'read_verilog -sv -defer rtl_hitflow/hitflow_nmf_g1_builder.sv; hierarchy -check -top hitflow_nmf_g1_builder; proc; opt; memory; opt; check; stat'
yosys -q -l "$BUILD/yosys_product.log" \
  -p 'read_verilog -sv -defer rtl_hitflow/hitflow_gate_product_engine.sv; hierarchy -check -top hitflow_gate_product_engine; proc; opt; memory; opt; check; stat'
yosys -q -l "$BUILD/yosys_multicast.log" \
  -p 'read_verilog -sv -defer rtl_hitflow/hitflow_segmented_multicast.sv; hierarchy -check -top hitflow_segmented_multicast; proc; opt; memory; opt; check; stat'
yosys -q -l "$BUILD/yosys_accumulator.log" \
  -p 'read_verilog -sv -defer rtl_hitflow/hitflow_banked_accumulator.sv; hierarchy -check -top hitflow_banked_accumulator; proc; opt; memory -nomap; opt; check; stat'
yosys -q -l "$BUILD/yosys_g1_top.log" \
  -p 'read_verilog -sv -defer rtl_hitflow/hitflow_nmf_g1_builder.sv rtl_hitflow/hitflow_gate_product_engine.sv rtl_hitflow/hitflow_segmented_multicast.sv rtl_hitflow/hitflow_banked_accumulator.sv rtl_hitflow/hitflow_g1_projection_top.sv; hierarchy -check -top hitflow_g1_projection_top; proc; opt; memory -nomap; opt; check; stat'

for log in "$BUILD/yosys_nmf_g1.log" "$BUILD/yosys_product.log" \
           "$BUILD/yosys_multicast.log" "$BUILD/yosys_accumulator.log" \
           "$BUILD/yosys_g1_top.log"; do
  if grep -q '^Warning:' "$log"; then
    grep '^Warning:' "$log"
    exit 1
  fi
  grep -E 'Found and reported|Number of cells|\$mul' "$log" | tail -10
done

PYTHONPATH=scripts python -m unittest \
  scripts/test_class_gate_multicast_projection_reference.py -v
