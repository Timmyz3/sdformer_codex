#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ppdi_dctf32_bank_executor"
RESULT="$ROOT/results/gatestack_ppdi_dctf32_bank_executor_20260722"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
ENGINE="rtl_hitflow/gatestack_decoupled_product_engine.sv"
RTL="rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv"
TB="tb_hitflow/tb_gatestack_ppdi_dctf32_bank_executor.sv"
SVA="verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv"
BIND="verif_hitflow/bind_gatestack_ppdi_dctf32_bank_executor_assertions.sv"
ACC="rtl_hitflow/hitflow_banked_accumulator.sv"
ACC_TB="tb_hitflow/tb_gatestack_ppdi_executor_acc_flush.sv"
ACC_SVA="verif_hitflow/hitflow_banked_accumulator_assertions.sv"
ACC_BIND="verif_hitflow/bind_hitflow_banked_accumulator_assertions.sv"

mkdir -p "$BUILD" "$RESULT"
rm -rf "$BUILD/verilator_obj" "$BUILD/verilator_acc_flush_obj"
cd "$ROOT"

iverilog -g2012 -Wall -s tb_gatestack_ppdi_dctf32_bank_executor \
  -o "$BUILD/tb.vvp" "$ENGINE" "$RTL" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
if grep -Eiq 'warning:|error:' "$BUILD/iverilog_build.log"; then
  cat "$BUILD/iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
grep -q '^PASS PPDI DCTF32 BANK EXECUTOR ' "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ppdi_dctf32_bank_executor \
  -Mdir "$BUILD/verilator_obj" \
  "$ENGINE" "$RTL" "$SVA" "$BIND" "$TB" \
  >"$BUILD/verilator_build.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  cat "$BUILD/verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/Vtb_gatestack_ppdi_dctf32_bank_executor" \
  | tee "$BUILD/verilator.log"
grep -q '^PASS PPDI DCTF32 BANK EXECUTOR ' "$BUILD/verilator.log"

iverilog -g2012 -Wall -s tb_gatestack_ppdi_executor_acc_flush \
  -o "$BUILD/acc_flush.vvp" "$ENGINE" "$RTL" "$ACC" "$ACC_TB" \
  >"$BUILD/acc_flush_iverilog_build.log" 2>&1
if grep -Eiq 'warning:|error:' "$BUILD/acc_flush_iverilog_build.log"; then
  cat "$BUILD/acc_flush_iverilog_build.log" >&2
  exit 1
fi
vvp "$BUILD/acc_flush.vvp" | tee "$BUILD/acc_flush_iverilog.log"
grep -q '^PASS PPDI EXECUTOR ACC FLUSH ' "$BUILD/acc_flush_iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_ppdi_executor_acc_flush \
  -Mdir "$BUILD/verilator_acc_flush_obj" \
  "$ENGINE" "$RTL" "$ACC" "$SVA" "$BIND" \
  "$ACC_SVA" "$ACC_BIND" "$ACC_TB" \
  >"$BUILD/acc_flush_verilator_build.log" 2>&1
if grep -Eq '%Warning|%Error' "$BUILD/acc_flush_verilator_build.log"; then
  cat "$BUILD/acc_flush_verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_acc_flush_obj/Vtb_gatestack_ppdi_executor_acc_flush" \
  | tee "$BUILD/acc_flush_verilator.log"
grep -q '^PASS PPDI EXECUTOR ACC FLUSH ' "$BUILD/acc_flush_verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv $ENGINE $RTL; hierarchy -check -top gatestack_ppdi_dctf32_bank_executor; proc; opt; check; stat"
if grep -Eq '^Warning:|ERROR:' "$BUILD/yosys.log"; then
  grep -E '^Warning:|ERROR:' "$BUILD/yosys.log" >&2
  exit 1
fi

python "$LINTER" "$RTL" >"$BUILD/erie_rtl.log" 2>&1
python "$LINTER" --mode tb "$TB" >"$BUILD/erie_tb.log" 2>&1
python "$LINTER" --mode tb "$ACC_TB" >"$BUILD/erie_acc_tb.log" 2>&1
if ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_rtl.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_tb.log" ||
   ! grep -q 'Summary: 0 error(s), 0 warning(s)' "$BUILD/erie_acc_tb.log"; then
  cat "$BUILD/erie_rtl.log" "$BUILD/erie_tb.log" \
      "$BUILD/erie_acc_tb.log" >&2
  exit 1
fi

iverilog -V >"$RESULT/iverilog_version.txt" 2>&1
verilator --version >"$RESULT/verilator_version.txt"
yosys -V >"$RESULT/yosys_version.txt"
sha256sum "$ENGINE" "$RTL" "$TB" "$SVA" "$BIND" \
  "$ACC" "$ACC_TB" "$ACC_SVA" "$ACC_BIND" \
  >"$RESULT/input_sha256.txt"
cp "$BUILD/iverilog.log" "$RESULT/leaf_iverilog.log"
cp "$BUILD/iverilog_build.log" "$RESULT/leaf_iverilog_build.log"
cp "$BUILD/verilator.log" "$RESULT/leaf_verilator_sva.log"
cp "$BUILD/verilator_build.log" "$RESULT/leaf_verilator_build.log"
cp "$BUILD/yosys.log" "$RESULT/leaf_yosys.log"
cp "$BUILD/erie_rtl.log" "$RESULT/leaf_erie_rtl.log"
cp "$BUILD/erie_tb.log" "$RESULT/leaf_erie_tb.log"
cp "$BUILD/acc_flush_iverilog.log" "$RESULT/acc_flush_iverilog.log"
cp "$BUILD/acc_flush_iverilog_build.log" \
  "$RESULT/acc_flush_iverilog_build.log"
cp "$BUILD/acc_flush_verilator.log" "$RESULT/acc_flush_verilator_sva.log"
cp "$BUILD/acc_flush_verilator_build.log" \
  "$RESULT/acc_flush_verilator_build.log"
cp "$BUILD/erie_acc_tb.log" "$RESULT/acc_flush_erie_tb.log"
(cd "$RESULT" && sha256sum ./*.log >log_sha256.txt)

python - "$BUILD" "$RESULT" <<'PY'
import json
import re
import sys
from pathlib import Path

build = Path(sys.argv[1])
result = Path(sys.argv[2])
result_rel = Path("results") / result.name
pattern = re.compile(
    r"PASS PPDI DCTF32 BANK EXECUTOR cycles=(\d+) commands=(\d+) "
    r"weight_req=(\d+) acc=\{(\d+),(\d+)\} done=(\d+) stale=(\d+) "
    r"zero=\{(\d+),(\d+)\}"
)

runs = {}
for name, filename in (("Icarus", "iverilog.log"),
                       ("Verilator_SVA", "verilator.log")):
    text = (build / filename).read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"无法解析{name} PASS行")
    values = [int(value) for value in match.groups()]
    runs[name] = {
        "cycles": values[0],
        "commands": values[1],
        "weight_requests": values[2],
        "acc_commits": values[3:5],
        "term_done": values[5],
        "stale_responses": values[6],
        "zero_gate_commands": values[7],
        "zero_gate_term_done": values[8],
        "mismatch": 0,
    }

acc_pattern = re.compile(
    r"PASS PPDI EXECUTOR ACC FLUSH old_partial=(\d+)/(\d+) "
    r"replacement_final_token2=(\d+) bias=(\d+) updates=(\d+) writes=(\d+)"
)
acc_runs = {}
for name, filename in (("Icarus", "acc_flush_iverilog.log"),
                       ("Verilator_SVA", "acc_flush_verilator.log")):
    text = (build / filename).read_text(encoding="utf-8")
    match = acc_pattern.search(text)
    if match is None:
        raise SystemExit(f"无法解析{name} Acc flush PASS行")
    values = [int(value) for value in match.groups()]
    acc_runs[name] = {
        "old_partial_commits_even_odd": values[0:2],
        "replacement_final_token2": values[2],
        "bias_commits": values[3],
        "updates": values[4],
        "writes": values[5],
        "mismatch": 0,
    }

report = {
    "status": "PASS",
    "scope": "PPDI单bank双目的executor叶模块",
    "runs": runs,
    "executor_accumulator_flush": acc_runs,
    "coverage": [
        "偶端口先提交、奇端口后提交且无重复",
        "偶奇双端口同拍提交",
        "only-even与only-odd命令",
        "两个命令复用一次term product",
        "空mask与奇偶编码错误拒绝",
        "部分提交时flush",
        "epoch迟到weight响应隔离",
        "Acc valid受阻期间stale响应并行drain且valid不撤回",
        "child旧sticky error单拍clear后parent与child同时清除",
        "paired non-last到paired last只释放一次product",
        "奇端口先提交后偶端口保持",
        "缩小epoch参数下pending-generation满表阻塞与drain恢复",
        "pending stale错误身份不清generation、完整身份才释放",
        "单命令paired零gate次拍retire、head_last且零weight/Acc",
        "多命令only-even/only-odd零gate保持sequence并精确done",
        "零gate term中途flush无done、weight或Acc",
        "真实Banked Acc同拍flush隔离旧partial write并同tag/token恢复",
    ],
    "provenance": {
        "input_sha256": str(result_rel / "input_sha256.txt"),
        "tool_versions": [
            str(result_rel / "iverilog_version.txt"),
            str(result_rel / "verilator_version.txt"),
            str(result_rel / "yosys_version.txt"),
        ],
        "logs": {
            "Icarus": str(result_rel / "leaf_iverilog.log"),
            "Icarus_Build": str(result_rel / "leaf_iverilog_build.log"),
            "Verilator_SVA": str(result_rel / "leaf_verilator_sva.log"),
            "Verilator_Build": str(result_rel / "leaf_verilator_build.log"),
            "Yosys": str(result_rel / "leaf_yosys.log"),
            "Erie_RTL": str(result_rel / "leaf_erie_rtl.log"),
            "Erie_TB": str(result_rel / "leaf_erie_tb.log"),
            "Acc_Flush_Icarus": str(result_rel / "acc_flush_iverilog.log"),
            "Acc_Flush_Icarus_Build": str(result_rel / "acc_flush_iverilog_build.log"),
            "Acc_Flush_Verilator_SVA": str(result_rel / "acc_flush_verilator_sva.log"),
            "Acc_Flush_Verilator_Build": str(result_rel / "acc_flush_verilator_build.log"),
            "Erie_Acc_TB": str(result_rel / "acc_flush_erie_tb.log"),
            "Log_SHA256": str(result_rel / "log_sha256.txt"),
        },
    },
    "limits": [
        "当前仅为单bank叶模块，尚未连接PPDI adapter/fabric",
        "动态SVA不是formal证明",
        "未形成H67四stage周期或目标工艺PPA结论",
    ],
}
(result / "验证结果.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# PPDI 双目的 Bank Executor 叶模块验证",
    "",
    "| 运行 | 周期 | command | weight | Acc偶/奇 | done | stale | 零gate command/done | mismatch |",
    "|---|---:|---:|---:|---|---:|---:|---|---:|",
]
for name, item in runs.items():
    lines.append(
        f"| {name} | {item['cycles']} | {item['commands']} | "
        f"{item['weight_requests']} | {item['acc_commits']} | "
        f"{item['term_done']} | {item['stale_responses']} | "
        f"{item['zero_gate_commands']}/{item['zero_gate_term_done']} | 0 |"
    )
lines += [
    "",
    "## 结论",
    "",
    "- 一个term product可被偶/奇两个目的提交复用；",
    "- 两端口可分拍握手，已完成端口不会再次拉高valid；",
    "- command只在全部有效目的提交后retire，term-last同拍产生done；",
    "- 零gate term逐command retire，省略weight访问和加零Acc更新；",
    "- Icarus、Verilator动态SVA、Yosys和Erie全部通过。",
    "- 真实Banked Acc集成在两个模拟器下证明共同flush后旧partial write不可见。",
    "",
    "## 证据边界",
    "",
    "- 当前仅为单bank叶模块，尚未连接PPDI adapter/fabric；",
    "- 动态SVA不是formal证明；",
    "- 不据此声称H67加速、面积或能耗收益。",
]
(result / "验证报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY

echo "PASS: PPDI DCTF32 executor Icarus、Verilator动态SVA、Yosys、Erie 0 error/warning"
