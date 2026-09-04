#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

yosys -p "read_verilog hw/rtl/top.v hw/rtl/controller.v hw/rtl/attention_unit.v hw/rtl/token_mixer.v hw/rtl/spike_unit.v hw/rtl/sram_if.v hw/rtl/pe_array.v; synth -top top; stat"

