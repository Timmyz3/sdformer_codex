#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${NANGATE45_LIB:-$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib}"
OUT="$ROOT/results/gatestack_dctf32_bank_executor_20260720"
MAPPING="$OUT/mapping"
mkdir -p "$MAPPING"
cd "$ROOT"

if [[ ! -s "$LIB" ]]; then
  echo "缺少Nangate45 Liberty: $LIB" >&2
  exit 1
fi

# 两个候选读取完全相同的源文件，并固定为相同的32个乘积lane。
RTL=(
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
)

map_one() {
  local name="$1"
  local top="$2"
  local parameters="$3"
  timeout 1200 yosys -q -l "$MAPPING/${name}.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; $parameters hierarchy -check -top $top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB; write_verilog -noattr $MAPPING/${name}_mapped.v"
}

yosys -V >"$OUT/yosys_version.txt"
sha256sum "$LIB" "${RTL[@]}" >"$OUT/input_sha256.txt"

map_one product_engine_32 gatestack_decoupled_product_engine \
  "chparam -set OUT_TILE 32 gatestack_decoupled_product_engine; chparam -set TAG_W 36 gatestack_decoupled_product_engine;"
map_one executor_32 gatestack_dctf32_bank_executor \
  "chparam -set OUT_TILE 32 gatestack_dctf32_bank_executor;"

python3 scripts/summarize_gatestack_dctf32_bank_executor.py \
  --mapping-dir "$MAPPING" \
  --output-dir "$OUT"

echo "PASS: DCTF32 executor与32-lane product-engine同源Nangate45无约束逻辑映射完成"
