#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ppdi_term_event_adapter_2c"
OUT="$ROOT/results/gatestack_ppdi_term_event_adapter_2c_20260722"
LOGS="$OUT/logs"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
BANK="rtl_hitflow/gatestack_ppdi_token_bank.sv"
RTL="rtl_hitflow/gatestack_ppdi_term_event_adapter_2c.sv"
TB="tb_hitflow/tb_gatestack_ppdi_term_event_adapter_2c.sv"
TB162="tb_hitflow/tb_gatestack_ppdi_term_event_adapter_2c_tokens162.sv"
SVA="verif_hitflow/gatestack_ppdi_term_event_adapter_2c_assertions.sv"
BIND="verif_hitflow/bind_gatestack_ppdi_term_event_adapter_2c_assertions.sv"
RUNNER="sim_hitflow/run_gatestack_ppdi_term_event_adapter_2c_checks.sh"

need_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "FAIL: missing tool: $1" >&2
    exit 1
  fi
}

for tool in iverilog vvp verilator yosys python3; do
  need_tool "$tool"
done
if [[ ! -r "$LINTER" ]]; then
  echo "FAIL: missing Erie linter: $LINTER" >&2
  exit 1
fi

mkdir -p "$BUILD" "$LOGS"
rm -rf "$BUILD/verilator_obj" "$BUILD/verilator_obj_162"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_ppdi_term_event_adapter_2c \
  -o "$BUILD/tb.vvp" "$BANK" "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
if grep -Eiq 'warning:|error:' "$BUILD/iverilog_build.log"; then
  cat "$BUILD/iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
grep -Eq '^PASS PPDI ADAPTER 2C .*gate_zero_legal=1 ' \
  "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ppdi_term_event_adapter_2c \
  -Mdir "$BUILD/verilator_obj" \
  "$BANK" "$RTL" "$SVA" "$BIND" "$TB" \
  >"$BUILD/verilator_build.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_ppdi_term_event_adapter_2c" \
  | tee "$BUILD/verilator.log"
grep -Eq '^PASS PPDI ADAPTER 2C .*gate_zero_legal=1 ' \
  "$BUILD/verilator.log"

icarus_pass="$(grep '^PASS PPDI ADAPTER 2C ' "$BUILD/iverilog.log")"
verilator_pass="$(grep '^PASS PPDI ADAPTER 2C ' "$BUILD/verilator.log")"
if [[ "$icarus_pass" != "$verilator_pass" ]]; then
  echo "FAIL: Icarus and Verilator counters differ" >&2
  printf 'Icarus: %s\nVerilator: %s\n' \
    "$icarus_pass" "$verilator_pass" >&2
  exit 1
fi

iverilog -g2012 -Wall -s tb_gatestack_ppdi_term_event_adapter_2c_tokens162 \
  -o "$BUILD/tb_tokens162.vvp" "$BANK" "$RTL" "$TB162" \
  >"$BUILD/iverilog_tokens162_build.log" 2>&1
if grep -Eiq 'warning:|error:' "$BUILD/iverilog_tokens162_build.log"; then
  cat "$BUILD/iverilog_tokens162_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb_tokens162.vvp" | tee "$BUILD/iverilog_tokens162.log"
grep -q '^PASS PPDI ADAPTER TOKENS162 commands=243 destinations=324 ' \
  "$BUILD/iverilog_tokens162.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ppdi_term_event_adapter_2c_tokens162 \
  -Mdir "$BUILD/verilator_obj_162" \
  "$BANK" "$RTL" "$SVA" "$BIND" "$TB162" \
  >"$BUILD/verilator_tokens162_build.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/verilator_tokens162_build.log"; then
  cat "$BUILD/verilator_tokens162_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj_162/Vtb_gatestack_ppdi_term_event_adapter_2c_tokens162" \
  | tee "$BUILD/verilator_tokens162.log"
grep -q '^PASS PPDI ADAPTER TOKENS162 commands=243 destinations=324 ' \
  "$BUILD/verilator_tokens162.log"
if [[ "$(grep '^PASS PPDI ADAPTER TOKENS162 ' "$BUILD/iverilog_tokens162.log")" != \
      "$(grep '^PASS PPDI ADAPTER TOKENS162 ' "$BUILD/verilator_tokens162.log")" ]]; then
  echo "FAIL: TOKENS162 Icarus and Verilator counters differ" >&2
  exit 1
fi

yosys -q -l "$BUILD/yosys_default.log" -p \
  "read_verilog -sv $BANK $RTL; hierarchy -check -top gatestack_ppdi_term_event_adapter_2c; proc; opt; check -assert; stat"
yosys -q -l "$BUILD/yosys_tokens11.log" -p \
  "read_verilog -sv $BANK $RTL; chparam -set TOKENS 11 -set EVENT_WAYS 3 gatestack_ppdi_term_event_adapter_2c; hierarchy -check -top gatestack_ppdi_term_event_adapter_2c; proc; opt; check -assert; stat"
yosys -q -l "$BUILD/yosys_token_bank.log" -p \
  "read_verilog -sv $BANK; chparam -set DEPTH 21 gatestack_ppdi_token_bank; hierarchy -check -top gatestack_ppdi_token_bank; proc; opt; check -assert; stat"
cat "$BUILD/yosys_default.log" "$BUILD/yosys_tokens11.log" \
    "$BUILD/yosys_token_bank.log" \
  | grep -E '^Warning:|ERROR:' \
  | grep -Ev 'Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/yosys_unexpected.log" || true
if [[ -s "$BUILD/yosys_unexpected.log" ]]; then
  cat "$BUILD/yosys_unexpected.log" >&2
  exit 1
fi
grep -q 'Found and reported 0 problems' "$BUILD/yosys_default.log"
grep -q 'Found and reported 0 problems' "$BUILD/yosys_tokens11.log"
grep -q 'Found and reported 0 problems' "$BUILD/yosys_token_bank.log"
grep -Eq 'gatestack_ppdi_token_bank[[:space:]]+16' \
  "$BUILD/yosys_default.log"
grep -Eq '\$memwr_v2[[:space:]]+1$' "$BUILD/yosys_token_bank.log"

python3 "$LINTER" "$BANK" >"$BUILD/erie_bank.log" 2>&1
python3 "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1
python3 "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
python3 "$LINTER" --mode tb "$TB162" >"$BUILD/erie_tb162.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' \
     "$BUILD/erie_bank.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' \
     "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' \
     "$BUILD/erie_tb.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' \
     "$BUILD/erie_tb162.log"; then
  cat "$BUILD/erie_bank.log" "$BUILD/erie_rtl.log" \
      "$BUILD/erie_tb.log" "$BUILD/erie_tb162.log" >&2
  exit 1
fi

cp "$BUILD/iverilog_build.log" "$BUILD/iverilog.log" \
   "$BUILD/iverilog_tokens162_build.log" \
   "$BUILD/iverilog_tokens162.log" \
   "$BUILD/verilator_build.log" "$BUILD/verilator.log" \
   "$BUILD/verilator_tokens162_build.log" \
   "$BUILD/verilator_tokens162.log" \
   "$BUILD/yosys_default.log" "$BUILD/yosys_tokens11.log" \
   "$BUILD/yosys_token_bank.log" "$BUILD/yosys_unexpected.log" \
   "$BUILD/erie_bank.log" "$BUILD/erie_rtl.log" \
   "$BUILD/erie_tb.log" "$BUILD/erie_tb162.log" "$LOGS/"
sha256sum "$BANK" "$RTL" "$TB" "$TB162" "$SVA" "$BIND" "$RUNNER" \
  >"$OUT/输入SHA256.txt"
{
  printf 'Icarus: %s\n' "$(iverilog -V 2>&1 | sed -n '1p')"
  printf 'VVP: %s\n' "$(vvp -V 2>&1 | sed -n '1p')"
  printf 'Verilator: %s\n' "$(verilator --version)"
  printf 'Yosys: %s\n' "$(yosys -V)"
  printf 'Python: %s\n' "$(python3 --version 2>&1)"
  printf 'Erie linter: %s\n' "$LINTER"
  printf 'Erie linter SHA256: %s\n' \
    "$(sha256sum "$LINTER" | cut -d' ' -f1)"
} >"$OUT/工具版本.txt"
(cd "$OUT" && sha256sum logs/*.log >"日志SHA256.txt")

python3 - "$OUT" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
logs = out / "logs"
pattern = re.compile(
    r"PASS PPDI ADAPTER 2C commands=(\d+) destinations=(\d+) "
    r"overlap=(\d+) full=(\d+) backpressure=(\d+) wraps=(\d+) "
    r"gate_zero_legal=(\d+) paired=(\d+) only_even=(\d+) "
    r"only_odd=(\d+) malformed=(\d+) flush=(\d+) stress_terms=(\d+)"
)
keys = (
    "commands", "destinations", "overlap_cycles", "full_context_cycles",
    "backpressure_cycles", "sequence_wraps", "gate_zero_legal",
    "paired_commands", "only_even_commands", "only_odd_commands",
    "malformed_terms", "flushes", "stress_terms",
)
simulators = {}
for name, filename in (
    ("Icarus", "iverilog.log"),
    ("Verilator_动态SVA", "verilator.log"),
):
    match = pattern.search((logs / filename).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{name}日志缺少PPDI-2C PASS计数")
    simulators[name] = {
        key: int(value) for key, value in zip(keys, match.groups())
    }
if simulators["Icarus"] != simulators["Verilator_动态SVA"]:
    raise SystemExit("Icarus与Verilator计数不一致")

pattern162 = re.compile(
    r"PASS PPDI ADAPTER TOKENS162 commands=(\d+) destinations=(\d+) "
    r"paired=(\d+) only_even=(\d+) only_odd=(\d+) backpressure=(\d+) "
    r"stable_checks=(\d+) flush=(\d+)"
)
capacity_simulators = {}
for name, filename in (
    ("Icarus", "iverilog_tokens162.log"),
    ("Verilator_动态SVA", "verilator_tokens162.log"),
):
    match = pattern162.search((logs / filename).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{name}日志缺少TOKENS162 PASS计数")
    capacity_simulators[name] = {
        key: int(value) for key, value in zip(
            ("commands", "destinations", "paired", "only_even", "only_odd",
             "backpressure_cycles", "stable_checks", "flushes"),
            match.groups(),
        )
    }
if capacity_simulators["Icarus"] != capacity_simulators["Verilator_动态SVA"]:
    raise SystemExit("TOKENS162双模拟器计数不一致")

def parse_sha(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        values[filename.lstrip("* ")] = digest
    return values

coverage = [
    "balanced奇偶配对、only-even和only-odd",
    "TOKENS=11时even容量6与odd容量5边界",
    "多beat非数值顺序输入且各parity保持收集顺序",
    "双context collect/emit重叠与双context同时占用",
    "确定性随机command反压及完整payload稳定",
    "duplicate、out-of-range、event metadata错误零partial command",
    "gate-zero term保持标量adapter合同并正常提交",
    "完整context加部分收集context的共同flush",
    "sticky protocol_error跨flush保持、clear_error清除及同拍新错误优先",
    "5-bit全局command sequence连续和回绕",
    "逐command mask/token/term边界及destination总量无重无漏",
    "TOKENS=162全容量、81 only-even、81 only-odd及compact index80",
    "四路append-index条带化token bank且每个物理bank单写口",
]
limits = [
    "本结果只覆盖PPDI parity-partition双context adapter叶模块，不代表完整PPDI fabric、executor或顶层集成已完成。",
    "本结果没有目标工艺映射、STA、功耗或真实工作负载性能测量，不据此声称性能收益。",
    "动态SVA和定向/确定性随机仿真不等同于形式证明，也不替代functional coverage与code coverage闭合。",
    "Yosys stat只作为结构可综合性与层次/check证据，不作为面积比较结论。",
    "当前token-bank采用异步读、同步单写的寄存器阵列风格；尚未证明目标同步SRAM的读延迟、端口和时序合同。",
    "TOKENS=162时每个context/parity有3个未使用条带槽，两个context合计96 bit物理padding；该开销必须进入后续面积与能量分账。",
    "16个单写token-bank的结论只覆盖token payload；adapter中的seen bitmap和metadata仍是行为级存储。",
]
result = {
    "状态": "PASS",
    "对象": "gatestack_ppdi_term_event_adapter_2c",
    "模拟器": simulators,
    "生产容量模拟器": capacity_simulators,
    "静态检查": {
        "Verilator_assert_零warning": True,
        "Yosys_default_hierarchy_check_stat": "PASS",
        "Yosys_TOKENS11_EVENT_WAYS3_hierarchy_check_stat": "PASS",
        "Yosys_16个token_bank每bank单写口": "PASS",
        "Erie_token_bank": "0 error, 0 warning",
        "Erie_RTL": "0 error, 0 warning",
        "Erie_TB": "0 error, 0 warning",
        "Erie_TB162": "0 error, 0 warning",
    },
    "覆盖": coverage,
    "边界": limits,
    "输入SHA256": parse_sha(out / "输入SHA256.txt"),
    "日志SHA256": parse_sha(out / "日志SHA256.txt"),
    "工具版本": (out / "工具版本.txt").read_text(
        encoding="utf-8"
    ).splitlines(),
}
(out / "验证结果.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# PPDI 奇偶分区双上下文 Adapter 验证报告",
    "",
    "## 结论",
    "",
    "`gatestack_ppdi_term_event_adapter_2c` 在本报告限定的叶模块边界内通过验证。Icarus 与启用动态 SVA 的 Verilator 输出计数完全一致，Verilator `-Wall --assert` 构建为零 warning；Yosys 默认参数及 `TOKENS=11/EVENT_WAYS=3` 参数均通过 `hierarchy -check`、`check -assert` 和 `stat`；Erie RTL/TB 均为 0 error、0 warning。",
    "",
    "event 收集时按 append index 对 even/odd token 进行四路条带化，每个 context/parity/bank 只有一个写入口；只有 whole-term 校验完成后 context 才对 command 可见。发射条数为 `max(even_count, odd_count)`，没有验证后 build/reorder 阶段。",
    "",
    "## 动态结果",
    "",
    "| 模拟器 | command | destination | 重叠拍 | 双context满拍 | 反压拍 | 配对 | 仅偶 | 仅奇 | malformed | flush | 回绕 | 压力term |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for name, item in simulators.items():
    lines.append(
        f"| {name} | {item['commands']} | {item['destinations']} | "
        f"{item['overlap_cycles']} | {item['full_context_cycles']} | "
        f"{item['backpressure_cycles']} | {item['paired_commands']} | "
        f"{item['only_even_commands']} | {item['only_odd_commands']} | "
        f"{item['malformed_terms']} | {item['flushes']} | "
        f"{item['sequence_wraps']} | {item['stress_terms']} |"
    )
lines += [
    "", "## 生产容量动态结果", "",
    "| 模拟器 | command | destination | 配对 | 仅偶 | 仅奇 | 反压拍 | 稳定检查 | flush |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for name, item in capacity_simulators.items():
    lines.append(
        f"| {name} | {item['commands']} | {item['destinations']} | "
        f"{item['paired']} | {item['only_even']} | {item['only_odd']} | "
        f"{item['backpressure_cycles']} | {item['stable_checks']} | "
        f"{item['flushes']} |"
    )
lines += [
    "",
    "## 覆盖内容",
    "",
    *[f"- {item}。" for item in coverage],
    "",
    "## 工具证据",
    "",
    "- Verilator：构建日志未出现 `%Warning` 或 `%Error`。",
    "- Yosys：两组 adapter 参数均为 0 problems；默认结构含 16 个 token-bank 实例，独立 token-bank 只有一个 `$memwr_v2`。",
    "- Erie：token-bank、adapter RTL 与两套 TB 均为 0 error、0 warning。",
    "- 输入哈希、工具版本、日志及日志哈希分别见同目录对应文本文件和 `logs/`。",
    "",
    "## 结果边界",
    "",
    *[f"- {item}" for item in limits],
    "",
]
(out / "验证报告.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"状态": "PASS", **simulators["Icarus"]},
                 ensure_ascii=False))
PY

printf 'TOOLS Icarus="%s" Verilator="%s" Yosys="%s"\n' \
  "$(iverilog -V 2>&1 | sed -n '1p')" \
  "$(verilator --version)" "$(yosys -V)"
echo "PASS: PPDI-2C adapter passed Icarus, Verilator assertions, Yosys, and Erie RTL/TB lint"
