#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
BUILD="$ROOT/build_hitflow/nangate45_dctf_frontend_mapping"
OUT="$ROOT/results/gatestack_dctf_frontend_20260720"
mkdir -p "$BUILD" "$OUT"
cd "$ROOT"

map_one() {
  local name="$1"
  local rtl="$2"
  local top="$3"
  local chparam="$4"
  timeout 1200 yosys -q -l "$BUILD/${name}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv $rtl; $chparam hierarchy -check -top $top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check; stat -liberty $LIB"
}

map_one adapter \
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv \
  gatestack_dctf_term_event_adapter ""
map_one fabric_q2 \
  rtl_hitflow/gatestack_dctf_term_fabric.sv \
  gatestack_dctf_term_fabric \
  "chparam -set Q 2 gatestack_dctf_term_fabric;"

python3 scripts/summarize_gatestack_dctf_frontend.py \
  --mapping-dir "$BUILD" --output-dir "$OUT"

echo "PASS: DCTF adapter与Q2 fabric开放逻辑映射完成；memory面积未计"
