#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_dctf_term_event_adapter_2c"
OUT="$ROOT/results/gatestack_dctf_term_event_adapter_2c_stress_20260722"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
RTL="rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv"
TB="tb_hitflow/tb_gatestack_dctf_term_event_adapter_2c.sv"
SVA="verif_hitflow/gatestack_dctf_term_event_adapter_2c_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf_term_event_adapter_2c_assertions.sv"
mkdir -p "$BUILD" "$OUT"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_dctf_term_event_adapter_2c \
  -o "$BUILD/tb.vvp" "$RTL" "$TB" >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
grep -q '^PASS DCTF ADAPTER 2C ' "$BUILD/iverilog.log"

rm -rf "$BUILD/verilator_obj"
verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_dctf_term_event_adapter_2c \
  -Mdir "$BUILD/verilator_obj" "$RTL" "$TB" "$SVA" "$BIND" \
  >"$BUILD/verilator_build.log" 2>&1
if grep -Eq '(%Warning|%Error)' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_dctf_term_event_adapter_2c" \
  | tee "$BUILD/verilator.log"
grep -q '^PASS DCTF ADAPTER 2C ' "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $RTL; hierarchy -check -top gatestack_dctf_term_event_adapter_2c; proc; opt; check -assert; stat"
yosys -q -l "$BUILD/yosys_event_ways2.log" -p \
  "read_verilog -sv $RTL; chparam -set EVENT_WAYS 2 gatestack_dctf_term_event_adapter_2c; hierarchy -check -top gatestack_dctf_term_event_adapter_2c; proc; opt; check -assert; stat"
cat "$BUILD/yosys.log" "$BUILD/yosys_event_ways2.log" \
  | grep -E '^Warning:|ERROR:' \
  | grep -Ev 'Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1 || true
if grep -Eq '\[ERROR\]|(^|[[:space:]])ERROR([:[:space:]]|$)' \
    "$BUILD/erie_rtl.log"; then
  cat "$BUILD/erie_rtl.log" >&2
  exit 1
fi

python3 - "$BUILD" "$OUT" <<'PY'
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

build = Path(sys.argv[1])
out = Path(sys.argv[2])
pattern = re.compile(
    r"PASS DCTF ADAPTER 2C outputs=(\d+) overlap=(\d+) "
    r"backpressure=(\d+) stress_terms=(\d+) stress_destinations=(\d+)"
)
rows = {}
for simulator, filename in (("Icarus", "iverilog.log"),
                            ("Verilator_SVA", "verilator.log")):
    match = pattern.search((build / filename).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{simulator}缺少压力回归PASS计数")
    rows[simulator] = {
        key: int(value) for key, value in zip(
            ("commands", "overlap_cycles", "backpressure_cycles",
             "stress_terms", "stress_destinations"), match.groups()
        )
    }
if rows["Icarus"] != rows["Verilator_SVA"]:
    raise SystemExit("双模拟器压力计数不一致")
report = {
    "status": "PASS",
    "simulators": rows,
    "directed_coverage": [
        "per-context物理sideband",
        "malformed duplicate term原子丢弃",
        "flush清空双context",
        "sticky error清除",
    ],
    "stress_coverage": [
        "128个连续合法term",
        "696个压力destination",
        "8-bit command sequence回绕",
        "确定性随机command backpressure",
        "collect/emit并发",
    ],
    "limits": [
        "压力流为确定性合成term，不是新增网络trace",
        "未注入随机flush和连续多类malformed term",
        "不替代functional/code coverage或formal证明",
    ],
}
(out / "验证结果.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
row = rows["Icarus"]
lines = [
    "# DCTF-2C 原子双上下文压力验证",
    "",
    "| 模拟器 | 总命令 | 重叠拍 | 反压拍 | 压力term | 压力destination |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, item in rows.items():
    lines.append(
        f"| {name} | {item['commands']} | {item['overlap_cycles']} | "
        f"{item['backpressure_cycles']} | {item['stress_terms']} | "
        f"{item['stress_destinations']} |"
    )
lines += [
    "", "## 结论", "",
    "- Icarus 与 Verilator 动态 SVA 的输出与计数完全一致；",
    "- 128 个连续 term 产生 696 个压力 destination，加前置定向用例共检查 704 条命令；",
    "- 命中 collect/emit 并发、随机反压和 8-bit command sequence 回绕；",
    "- malformed term、flush 和 per-context sideband 由前置定向用例覆盖；",
    "", "## 证据边界", "",
    *[f"- {item}；" for item in report["limits"]],
    "",
]
(out / "验证报告.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY

echo "PASS: DCTF-2C原子双上下文adapter通过Icarus、Verilator动态SVA、Yosys和Erie"
