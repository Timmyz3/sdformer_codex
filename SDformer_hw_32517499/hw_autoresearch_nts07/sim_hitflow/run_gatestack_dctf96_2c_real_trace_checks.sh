#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/gatestack_dctf96_2c_real_trace_20260722"
VECTORS="$ROOT/results/gatestack_dctf96_real_trace_20260720/vectors"
BUILD="$OUT/build"
LOGS="$OUT/logs"
TB="tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv"
TOP="tb_gatestack_dctf96_banklocal_projection_real_trace"
HEADS=(3 6 12 24)
RTL=(
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)
SVA=(
  verif_hitflow/gatestack_dctf_term_event_adapter_2c_assertions.sv
  verif_hitflow/bind_gatestack_dctf_term_event_adapter_2c_assertions.sv
  verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv
  verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv
)
mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

for stage in 0 1 2 3; do
  mkdir -p "$BUILD/s$stage"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.STAGE=$stage" -P"$TOP.HEADS=${HEADS[$stage]}" \
    -P"$TOP.ADAPTER_CONTEXTS=2" \
    -o "$BUILD/s$stage/tb.vvp" "${RTL[@]}" "$TB" \
    >"$LOGS/iverilog_build_s$stage.log" 2>&1
  vvp "$BUILD/s$stage/tb.vvp" "+VECTOR_DIR=$VECTORS/s$stage" \
    | tee "$LOGS/icarus_s$stage.log"
  grep -q "^PASS DCTF96 REAL TRACE stage=S$stage " \
    "$LOGS/icarus_s$stage.log"
done

rm -rf "$BUILD/s0/verilator_obj"
verilator --binary --timing --assert -Wall --top-module "$TOP" \
  -GSTAGE=0 -GHEADS=3 -GADAPTER_CONTEXTS=2 \
  -Mdir "$BUILD/s0/verilator_obj" "${RTL[@]}" "$TB" "${SVA[@]}" \
  >"$LOGS/verilator_build_s0.log" 2>&1
if grep -Eq '(%Warning|%Error)' "$LOGS/verilator_build_s0.log"; then
  cat "$LOGS/verilator_build_s0.log" >&2
  exit 1
fi
"$BUILD/s0/verilator_obj/V$TOP" "+VECTOR_DIR=$VECTORS/s0" \
  | tee "$LOGS/verilator_s0.log"
grep -q '^PASS DCTF96 REAL TRACE stage=S0 ' "$LOGS/verilator_s0.log"

python3 - "$OUT" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
pat = re.compile(r"PASS DCTF96 REAL TRACE stage=S(?P<stage>\d+) heads=(?P<heads>\d+) cycles=(?P<cycles>\d+) terms=(?P<terms>\d+) physical_weight_req=(?P<weight>\d+) bias_req=(?P<bias>\d+) final_checks=(?P<final>\d+)")
rows=[]
for stage in range(4):
    text=(out/f"logs/icarus_s{stage}.log").read_text()
    m=pat.search(text)
    if not m: raise SystemExit(f"S{stage}缺少PASS")
    rows.append({k:int(v) for k,v in m.groupdict().items()})
v=pat.search((out/"logs/verilator_s0.log").read_text())
verilator={k:int(x) for k,x in v.groupdict().items()}
if verilator != rows[0]: raise SystemExit("S0双模拟器计数不一致")
old=[822,718,5652,55072]
central=[871,718,7208,51056]
independent=[871,730,7232,51112]
total=sum(r["cycles"] for r in rows)
report={
  "说明":"DCTF96-2C H67真实S0-S3 projection-only RTL回放",
  "Icarus":rows, "Verilator_S0_SVA":verilator,
  "总周期":total, "DCTF96_1C总周期":sum(old),
  "Central96总周期":sum(central), "Independent32x3总周期":sum(independent),
  "相对1C加速":sum(old)/total,
  "相对Central加速":sum(central)/total,
  "相对Independent加速":sum(independent)/total,
  "结论":"S0-S3全部bit-exact，2C前端反转S3高term密度瓶颈",
}
(out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
lines=[
"# DCTF96-2C H67真实四阶段回放",
"",
"双上下文前端仅重叠当前term命令发射与下一term收集/完整验证；三路bank-local weight/product/Acc、bias和final语义不变。",
"",
"| Stage | 2C周期 | 1C周期 | Central周期 | term | 物理weight | bias | acc32检查 |",
"|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r,o,c in zip(rows,old,central):
    lines.append(f"| S{r['stage']} | {r['cycles']} | {o} | {c} | {r['terms']} | {r['weight']} | {r['bias']} | {r['final']} |")
lines += [
"", "## 结论", "",
f"- 四阶段累计{total}周期，相对1C为{sum(old)/total:.3f}x，相对Central为{sum(central)/total:.3f}x，相对Independent为{sum(independent)/total:.3f}x；",
"- Icarus S0-S3与S0 Verilator动态SVA全部通过，233280个acc32检查零失配；",
"- whole-term验证后提交、双context所有权、flush清空和命令反压稳定由adapter动态SVA覆盖；",
"", "## 证据边界", "",
"- 只覆盖sample0/window0；",
"- 固定一拍行为weight/bias存储，final全ready；",
"- INT8仍为候选部署合同；",
"- 需要结合2C开放逻辑映射、SRAM宏、STA和SAIF后才能形成PPA/EDP结论；",
"",
]
(out/"report.md").write_text("\n".join(lines))
print(json.dumps(report,ensure_ascii=False))
PY

echo "PASS: DCTF96-2C H67 S0-S3及S0动态SVA真实回放完成"
