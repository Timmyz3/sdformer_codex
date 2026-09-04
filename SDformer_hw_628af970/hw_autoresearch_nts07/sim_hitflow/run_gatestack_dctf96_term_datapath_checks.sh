#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf96_term_datapath_top"
OUT="$ROOT/results/gatestack_dctf96_illegal_metadata_fix_20260722"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
ENGINE="rtl_hitflow/gatestack_decoupled_product_engine.sv"
EXECUTOR="rtl_hitflow/gatestack_dctf32_bank_executor.sv"
PPDI_EXECUTOR="rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv"
ADAPTER="rtl_hitflow/gatestack_dctf_term_event_adapter.sv"
ADAPTER2C="rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv"
PPDI_ADAPTER2C="rtl_hitflow/gatestack_ppdi_term_event_adapter_2c.sv"
PPDI_TOKEN_BANK="rtl_hitflow/gatestack_ppdi_token_bank.sv"
FABRIC="rtl_hitflow/gatestack_dctf_term_fabric.sv"
PPDI_FABRIC="rtl_hitflow/gatestack_ppdi_dctf_term_fabric.sv"
RTL="rtl_hitflow/gatestack_dctf96_term_datapath_top.sv"
TB="tb_hitflow/tb_gatestack_dctf96_term_datapath_top.sv"
SVA="verif_hitflow/gatestack_dctf96_term_datapath_top_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf96_term_datapath_top_assertions.sv"

mkdir -p "$BUILD" "$OUT"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_dctf96_term_datapath_top \
  -o "$BUILD/tb.vvp" \
  "$ENGINE" "$EXECUTOR" "$PPDI_EXECUTOR" "$ADAPTER" "$ADAPTER2C" \
  "$PPDI_TOKEN_BANK" "$PPDI_ADAPTER2C" "$FABRIC" "$PPDI_FABRIC" \
  "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
    "$BUILD/iverilog_build.log"; then
  cat "$BUILD/iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
if grep -Eiq '(^|[^[:alpha:]])(error|fatal|assertion failed)([^[:alpha:]]|$)' \
      "$BUILD/iverilog.log" ||
   ! grep -q '^PASS DCTF96 TERM DATAPATH ' "$BUILD/iverilog.log"; then
  exit 1
fi

rm -rf "$BUILD/verilator_obj"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf96_term_datapath_top \
  -Mdir "$BUILD/verilator_obj" \
  "$ENGINE" "$EXECUTOR" "$PPDI_EXECUTOR" "$ADAPTER" "$ADAPTER2C" \
  "$PPDI_TOKEN_BANK" "$PPDI_ADAPTER2C" "$FABRIC" "$PPDI_FABRIC" \
  "$RTL" "$TB" \
  "$SVA" "$BIND" >"$BUILD/verilator_build.log" 2>&1
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
    "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_dctf96_term_datapath_top" \
  | tee "$BUILD/verilator.log"
if grep -Eiq '(^|[^[:alpha:]])(error|fatal|assertion failed)([^[:alpha:]]|$)' \
      "$BUILD/verilator.log" ||
   ! grep -q '^PASS DCTF96 TERM DATAPATH ' "$BUILD/verilator.log"; then
  exit 1
fi

iverilog -g2012 -Wall -s tb_gatestack_dctf96_term_datapath_top \
  -Ptb_gatestack_dctf96_term_datapath_top.ADAPTER_CONTEXTS=2 \
  -o "$BUILD/tb_2c.vvp" \
  "$ENGINE" "$EXECUTOR" "$PPDI_EXECUTOR" "$ADAPTER" "$ADAPTER2C" \
  "$PPDI_TOKEN_BANK" "$PPDI_ADAPTER2C" "$FABRIC" "$PPDI_FABRIC" \
  "$RTL" "$TB" \
  >"$BUILD/iverilog_2c_build.log" 2>&1
if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
    "$BUILD/iverilog_2c_build.log"; then
  cat "$BUILD/iverilog_2c_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb_2c.vvp" | tee "$BUILD/iverilog_2c.log"
grep -q '^PASS DCTF96 TERM DATAPATH ' "$BUILD/iverilog_2c.log"

rm -rf "$BUILD/verilator_2c_obj"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf96_term_datapath_top \
  -GADAPTER_CONTEXTS=2 -Mdir "$BUILD/verilator_2c_obj" \
  "$ENGINE" "$EXECUTOR" "$PPDI_EXECUTOR" "$ADAPTER" "$ADAPTER2C" \
  "$PPDI_TOKEN_BANK" "$PPDI_ADAPTER2C" "$FABRIC" "$PPDI_FABRIC" \
  "$RTL" "$TB" \
  "$SVA" "$BIND" >"$BUILD/verilator_2c_build.log" 2>&1
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
    "$BUILD/verilator_2c_build.log"; then
  cat "$BUILD/verilator_2c_build.log" >&2
  exit 1
fi
"$BUILD/verilator_2c_obj/Vtb_gatestack_dctf96_term_datapath_top" \
  | tee "$BUILD/verilator_2c.log"
grep -q '^PASS DCTF96 TERM DATAPATH ' "$BUILD/verilator_2c.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $ENGINE $EXECUTOR $PPDI_EXECUTOR $ADAPTER $ADAPTER2C $PPDI_TOKEN_BANK $PPDI_ADAPTER2C $FABRIC $PPDI_FABRIC $RTL; hierarchy -check -top gatestack_dctf96_term_datapath_top; proc; opt; check; stat"
grep -Ei '(^Warning:|ERROR:|fatal|assert)' "$BUILD/yosys.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1
python "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_tb.log" ||
   grep -Eiq '(\[ERROR\]|\[WARNING\]|fatal|traceback)' \
       "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log"; then
  cat "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log" >&2
  exit 1
fi

iverilog -V >"$OUT/iverilog_version.txt" 2>&1
verilator --version >"$OUT/verilator_version.txt"
yosys -V >"$OUT/yosys_version.txt"
sha256sum "$RTL" "$TB" "$SVA" "$BIND" \
  "$ADAPTER" "$ADAPTER2C" "$PPDI_ADAPTER2C" "$PPDI_TOKEN_BANK" \
  "$FABRIC" "$PPDI_FABRIC" "$EXECUTOR" "$PPDI_EXECUTOR" "$ENGINE" \
  >"$OUT/input_sha256.txt"
python3 - "$BUILD" "$OUT" <<'PY'
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

build = Path(sys.argv[1])
out = Path(sys.argv[2])
pattern = re.compile(
    r"PASS DCTF96 TERM DATAPATH cycles=(\d+) issued=(\d+) "
    r"completed=\{(\d+),(\d+),(\d+)\}.*stale=\{(\d+),(\d+),(\d+)\}"
)
rows = {}
for name, filename in (
    ("Icarus_1C", "iverilog.log"),
    ("Verilator_SVA_1C", "verilator.log"),
    ("Icarus_2C", "iverilog_2c.log"),
    ("Verilator_SVA_2C", "verilator_2c.log"),
):
    match = pattern.search((build / filename).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{name}缺少PASS计数")
    values = [int(value) for value in match.groups()]
    rows[name] = {
        "cycles": values[0],
        "issued_terms": values[1],
        "completed_terms": values[2:5],
        "stale_responses": values[5:8],
        "mismatch": 0,
    }
report = {
    "status": "PASS",
    "runs": rows,
    "coverage": [
        "非法channel握手与payload drain",
        "非法supertile握手与payload drain",
        "零destination非法term立即恢复",
        "非法drain中flush恢复",
        "clear_error与新非法term同拍时new-error-wins",
        "2C合法context在途与非法drain隔离后完整恢复并核对结果",
        "1C/2C合法term完整计算回归",
    ],
    "log_paths": {
        name: str(build / filename) for name, filename in (
            ("Icarus_1C", "iverilog.log"),
            ("Verilator_SVA_1C", "verilator.log"),
            ("Icarus_2C", "iverilog_2c.log"),
            ("Verilator_SVA_2C", "verilator_2c.log"),
            ("Yosys", "yosys.log"),
            ("Erie_RTL", "erie_rtl.log"),
            ("Erie_TB", "erie_tb.log"),
        )
    },
    "provenance": {
        "input_sha256": str(out / "input_sha256.txt"),
        "tool_versions": [
            str(out / "iverilog_version.txt"),
            str(out / "verilator_version.txt"),
            str(out / "yosys_version.txt"),
        ],
    },
    "limits": [
        "动态SVA不是完整formal证明",
        "非法非零term仍要求source最终提供event_term_last或flush",
        "未覆盖随机长序列malformed/flush",
    ],
}
(out / "验证结果.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# DCTF96 非法 Metadata 活性修复验证",
    "",
    "| 运行 | 周期 | issued term | completed | stale | mismatch |",
    "|---|---:|---:|---|---|---:|",
]
for name, row in rows.items():
    lines.append(
        f"| {name} | {row['cycles']} | {row['issued_terms']} | "
        f"{row['completed_terms']} | {row['stale_responses']} | "
        f"{row['mismatch']} |"
    )
lines += [
    "", "## 覆盖", "",
    *[f"- {item}；" for item in report["coverage"]],
    "", "## 可追溯性", "",
    "- input_sha256.txt 固化RTL、TB、SVA、bind及依赖叶模块哈希；",
    "- 单独保存Icarus、Verilator和Yosys版本；",
    "- 验证结果.json记录四个运行的日志路径与mismatch字段；",
    "", "## 证据边界", "",
    *[f"- {item}；" for item in report["limits"]],
    "",
]
(out / "验证报告.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY

echo "PASS: DCTF96 term datapath 1C/2C Icarus、Verilator --assert、Yosys hierarchy/check/stat、Erie RTL+TB 0 error/warning"
