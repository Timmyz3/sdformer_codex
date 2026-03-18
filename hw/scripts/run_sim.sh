#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

iverilog -g2012 -o hw/tb/tb_top.out \
  hw/rtl/top.v \
  hw/rtl/controller.v \
  hw/rtl/attention_unit.v \
  hw/rtl/token_mixer.v \
  hw/rtl/spike_unit.v \
  hw/rtl/sram_if.v \
  hw/rtl/pe_array.v \
  hw/tb/tb_top.sv

vvp hw/tb/tb_top.out

