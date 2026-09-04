#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT="$ROOT/results/gatestack_three_independent32_term_real_trace_20260722"
VECTOR_ROOT="$RESULT/three_independent32_term_real_trace_vectors"
BUILD="$RESULT/three_independent32_term_real_trace_build"
LOGS="$RESULT/three_independent32_term_real_trace_logs"
SOURCE_MANIFEST="$ROOT/results/h67_real_bit_trace_20260717/manifest.json"
VECTOR_MANIFEST="$RESULT/three_independent32_term_real_trace_vectors_manifest.json"
DCTF_RESULT="$ROOT/results/gatestack_dctf96_real_trace_20260720/实测结果.json"
TB="tb_hitflow/tb_gatestack_three_independent32_term_real_trace.sv"
SVA=(
  verif_hitflow/gatestack_multihead_tile_projection_assertions.sv
  verif_hitflow/bind_gatestack_multihead_tile_projection_assertions.sv
  verif_hitflow/gatestack_three_independent32_term_real_trace_assertions.sv
  verif_hitflow/bind_gatestack_three_independent32_term_real_trace_assertions.sv
)
TOP="tb_gatestack_three_independent32_term_real_trace"
RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_transposed_bitmap_bank.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_three_independent32_term_projection_top.sv
)
HEADS=(3 6 12 24)

mkdir -p "$BUILD" "$LOGS" "$VECTOR_ROOT"
cd "$ROOT"

iverilog -V >"$RESULT/three_independent32_term_real_trace_iverilog_version_full.txt" 2>&1 || true
{
  sed -n '1p' "$RESULT/three_independent32_term_real_trace_iverilog_version_full.txt"
  verilator --version
  python3 --version
} >"$RESULT/three_independent32_term_real_trace_tool_versions.txt"

PYTHONPATH=scripts python3 -m unittest -v \
  scripts/test_generate_gatestack_dctf_real_trace_vectors.py \
  >"$LOGS/three_independent32_term_real_trace_generator_unittest.log" 2>&1

PYTHONPATH=scripts python3 scripts/generate_gatestack_dctf_real_trace_vectors.py \
  --manifest "$SOURCE_MANIFEST" \
  --output-root "$VECTOR_ROOT" \
  --result "$VECTOR_MANIFEST" \
  >"$LOGS/three_independent32_term_real_trace_vector_generation.log" 2>&1

for stage in 0 1 2 3; do
  stage_build="$BUILD/s$stage"
  mkdir -p "$stage_build"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.STAGE=$stage" \
    -P"$TOP.HEADS=${HEADS[$stage]}" \
    -o "$stage_build/three_independent32_term_real_trace.vvp" \
    "${RTL[@]}" "$TB" \
    >"$LOGS/three_independent32_term_real_trace_iverilog_build_s$stage.log" 2>&1
  vvp "$stage_build/three_independent32_term_real_trace.vvp" \
    "+VECTOR_DIR=$VECTOR_ROOT/s$stage" \
    | tee "$LOGS/three_independent32_term_real_trace_icarus_s$stage.log"
  grep -q "^PASS THREE_INDEPENDENT32 TERM REAL TRACE stage=S$stage " \
    "$LOGS/three_independent32_term_real_trace_icarus_s$stage.log"
done

S0_BUILD="$BUILD/s0/three_independent32_term_real_trace_verilator_obj"
rm -rf "$S0_BUILD"
verilator --binary --timing --assert -Wall \
  --top-module "$TOP" \
  -GSTAGE=0 -GHEADS=3 \
  -Mdir "$S0_BUILD" \
  "${RTL[@]}" "$TB" "${SVA[@]}" \
  >"$LOGS/three_independent32_term_real_trace_verilator_build_s0.log" 2>&1
if grep -Eq '(%Warning|%Error)' \
    "$LOGS/three_independent32_term_real_trace_verilator_build_s0.log"; then
  cat "$LOGS/three_independent32_term_real_trace_verilator_build_s0.log" >&2
  exit 1
fi
"$S0_BUILD/V$TOP" "+VECTOR_DIR=$VECTOR_ROOT/s0" \
  | tee "$LOGS/three_independent32_term_real_trace_verilator_s0.log"
grep -q '^PASS THREE_INDEPENDENT32 TERM REAL TRACE stage=S0 ' \
  "$LOGS/three_independent32_term_real_trace_verilator_s0.log"

python3 - "$RESULT" "$DCTF_RESULT" <<'PY'
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

result = Path(sys.argv[1])
dctf_result_path = Path(sys.argv[2])
manifest = json.loads(
    (result / "three_independent32_term_real_trace_vectors_manifest.json").read_text(
        encoding="utf-8"
    )
)
pattern = re.compile(
    r"^PASS THREE_INDEPENDENT32 TERM REAL TRACE stage=S(?P<stage>\d+) "
    r"heads=(?P<heads>\d+) cycles=(?P<cycles>\d+) "
    r"logical_terms=(?P<logical_terms>\d+) "
    r"term_port_reads=(?P<term_reads>\d+) "
    r"event_port_reads=(?P<event_reads>\d+) "
    r"source_done_ports=(?P<source_done>\d+) "
    r"weight_req=(?P<weight>\d+) bias_req=(?P<bias>\d+) "
    r"final_checks=(?P<final>\d+)$",
    re.MULTILINE,
)


def parse_log(path: Path) -> dict[str, int]:
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"缺少PASS计数: {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def read_memh(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


rows = []
for stage in range(4):
    rows.append(
        parse_log(
            result
            / "three_independent32_term_real_trace_logs"
            / f"three_independent32_term_real_trace_icarus_s{stage}.log"
        )
    )

verilator_row = parse_log(
    result
    / "three_independent32_term_real_trace_logs"
    / "three_independent32_term_real_trace_verilator_s0.log"
)
if verilator_row != rows[0]:
    raise SystemExit("S0 Icarus与Verilator计数不一致")

for row, record in zip(rows, manifest["records"]):
    vector_dir = Path(record["vector_dir"])
    counts = read_memh(vector_dir / "term_destination_counts.memh")
    event_beats = sum((count + 3) // 4 for count in counts)
    expected = {
        "heads": record["heads"],
        "logical_terms": record["expected_issued_terms"],
        "term_reads": record["expected_issued_terms"] * 3,
        "event_reads": event_beats * record["logical_supertiles"] * 3,
        "source_done": record["heads"] * record["logical_supertiles"] * 3,
        "weight": record["expected_physical_weight_requests"],
        "bias": record["expected_bias_requests"],
        "final": record["expected_final_checks"],
    }
    for key, value in expected.items():
        if row[key] != value:
            raise SystemExit(
                f"S{row['stage']}实测{key}={row[key]}，向量期望={value}"
            )
    row["logical_event_beats"] = event_beats * record["logical_supertiles"]

dctf_rows = {}
if dctf_result_path.exists():
    dctf_payload = json.loads(dctf_result_path.read_text(encoding="utf-8"))
    dctf_rows = {int(row["stage"]): row for row in dctf_payload["Icarus"]}

comparison = []
for row in rows:
    dctf_cycles = dctf_rows.get(row["stage"], {}).get("cycles")
    comparison.append(
        {
            "stage": row["stage"],
            "three_independent32_cycles": row["cycles"],
            "dctf96_cycles": dctf_cycles,
            "cycle_delta_vs_dctf96": (
                row["cycles"] - dctf_cycles if dctf_cycles is not None else None
            ),
        }
    )

external_cost = {
    "边界": "term client/slot读口位于本wrapper之外，不计入三套projection RTL内部面积",
    "结构读口数": 3,
    "DCTF共享输入边界读口基准": 1,
    "增量读口数": 2,
    "端口结构倍率": 3.0,
    "非空term/event动态读访问倍率": 3.0,
    "相对DCTF动态读访问增量百分比": 200.0,
    "说明": "这是端口数和握手访问次数代理，不是SRAM宏综合面积或焦耳能耗实测",
}

lines = [
    "# 3xIndependent32 H67真实S0-S3 Projection-only RTL对照",
    "",
    f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    "- 数据证据：与DCTF96相同的H67 sample0/window0真实term/event向量、INT8 weight和acc32 bias。",
    "- 结构：三套独立gatestack_multihead_tile_projection_top，OUT_TILE=32、BANKS=2；只共享clock/reset。",
    "- 握手：三个term client/slot读口并行供给相同流，每路valid保持到本路ready握手；无共享decoded-term逻辑。",
    "- 存储模型：三套weight和bias端口均为固定一拍响应；六个final端口逐acc32检查。",
    "",
    "| Stage | Heads | Icarus cycles | DCTF96 cycles | cycle差 | 逻辑terms | 三口term读 | 三口event beat读 | source_done | weight | bias | final检查 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row, delta in zip(rows, comparison):
    dctf_cycles = delta["dctf96_cycles"]
    cycle_delta = delta["cycle_delta_vs_dctf96"]
    lines.append(
        f"| S{row['stage']} | {row['heads']} | {row['cycles']} | "
        f"{dctf_cycles if dctf_cycles is not None else 'N/A'} | "
        f"{cycle_delta if cycle_delta is not None else 'N/A'} | "
        f"{row['logical_terms']} | {row['term_reads']} | {row['event_reads']} | "
        f"{row['source_done']} | {row['weight']} | {row['bias']} | {row['final']} |"
    )
lines.extend(
    [
        "",
        "## 三读口外部代价",
        "",
        "- 结构面积代理：独立基线需要3个term client/slot读口，相对DCTF共享输入边界增加2个读口，端口结构为3x。",
        "- 动态能量代理：非空stage中term descriptor与event beat均发生3份独立握手访问，相对单读口流量为3x，即+200%。",
        "- S1真实trace为空流，动态term/event访问为0；三个物理读口的结构代价仍存在。",
        "- 上述值只记录wrapper外部端口和握手访问代理；未给出SRAM宏面积或焦耳值，因为本轮未对外部slot存储做物理综合/功耗标定。",
        "",
        "## 验证结论",
        "",
        "- Icarus S0-S3全部通过，六个final端口所有acc32元素bit-exact。",
        f"- Verilator S0 + SVA通过；cycles={verilator_row['cycles']}，与Icarus一致，-Wall无warning。",
        "- SVA覆盖三路term/event/source_done stall稳定、相同流内容、固定一拍weight/bias响应和final stall稳定。",
        "- Python DCTF真实向量生成器单测通过；DUT protocol_error和accumulator_overflow均未触发。",
        "",
        "## 边界",
        "",
        "本结果仅覆盖projection execution slice。INT8量化沿用源manifest候选合同，不替代valid825部署精度冻结；三读口物理面积/能量仍需带具体SRAM宏综合评估。",
        "",
    ]
)
(result / "three_independent32_term_real_trace_中文汇总.md").write_text(
    "\n".join(lines), encoding="utf-8"
)

run_result = {
    "说明": "3xIndependent32 H67真实S0-S3 projection-only RTL对照",
    "向量合同": "与DCTF96相同H67 term/event输入向量",
    "Icarus": rows,
    "Verilator_S0_SVA": verilator_row,
    "DCTF96周期对照": comparison,
    "三读口外部面积能量代价": external_cost,
    "结论": "S0-S3全部通过，六final逐acc32 bit-exact",
}
(result / "three_independent32_term_real_trace_results.json").write_text(
    json.dumps(run_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "PASS: 3xIndependent32 H67真实S0-S3 Icarus及S0 Verilator+SVA全部通过"
