#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT="$ROOT/results/gatestack_central96_term_real_trace_20260722"
DCTF_RESULT="$ROOT/results/gatestack_dctf96_real_trace_20260720"
VECTOR_ROOT="$DCTF_RESULT/vectors"
VECTOR_MANIFEST="$DCTF_RESULT/vectors_manifest.json"
DCTF_RUN_JSON="$DCTF_RESULT/实测结果.json"
BUILD="$RESULT/build"
LOGS="$RESULT/logs"
TB="tb_hitflow/tb_gatestack_central96_term_projection_real_trace.sv"
SVA="verif_hitflow/gatestack_multihead_tile_projection_assertions.sv"
BIND="verif_hitflow/bind_gatestack_multihead_tile_projection_assertions.sv"
TOP="tb_gatestack_central96_term_projection_real_trace"
RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
)
HEADS=(3 6 12 24)

mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

test -f "$VECTOR_MANIFEST"
test -f "$DCTF_RUN_JSON"
iverilog -V >"$RESULT/iverilog_version_full.txt" 2>&1 || true
{
  sed -n '1p' "$RESULT/iverilog_version_full.txt"
  verilator --version
  python3 --version
} >"$RESULT/tool_versions.txt"

PYTHONPATH=scripts python3 -m unittest -v \
  scripts/test_generate_gatestack_dctf_real_trace_vectors.py \
  >"$LOGS/generator_unittest.log" 2>&1

python3 - "$VECTOR_MANIFEST" "$VECTOR_ROOT" <<'PY' \
  >"$LOGS/vector_reuse_check.log"
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
vector_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for record in manifest["records"]:
    stage_dir = vector_root / f"s{record['stage']}"
    for name, expected in record["files"].items():
        path = stage_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise SystemExit(f"向量SHA256不匹配: {path}")
print("PASS: DCTF S0-S3向量目录逐文件SHA256复用检查通过")
PY

for stage in 0 1 2 3; do
  stage_build="$BUILD/s$stage"
  mkdir -p "$stage_build"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.STAGE=$stage" \
    -P"$TOP.HEADS=${HEADS[$stage]}" \
    -o "$stage_build/tb.vvp" "${RTL[@]}" "$TB" \
    >"$LOGS/iverilog_build_s$stage.log" 2>&1
  vvp "$stage_build/tb.vvp" \
    "+VECTOR_DIR=$VECTOR_ROOT/s$stage" \
    | tee "$LOGS/icarus_s$stage.log"
  grep -q "^PASS CENTRAL96 REAL TRACE stage=S$stage " \
    "$LOGS/icarus_s$stage.log"
done

S0_BUILD="$BUILD/s0/verilator_obj"
rm -rf "$S0_BUILD"
verilator --binary --timing --assert -Wall \
  --top-module "$TOP" \
  -GSTAGE=0 -GHEADS=3 \
  -Mdir "$S0_BUILD" \
  "${RTL[@]}" "$TB" "$SVA" "$BIND" \
  >"$LOGS/verilator_build_s0.log" 2>&1
if grep -Eq '(%Warning|%Error)' "$LOGS/verilator_build_s0.log"; then
  cat "$LOGS/verilator_build_s0.log" >&2
  exit 1
fi
"$S0_BUILD/V$TOP" "+VECTOR_DIR=$VECTOR_ROOT/s0" \
  | tee "$LOGS/verilator_s0.log"
grep -q '^PASS CENTRAL96 REAL TRACE stage=S0 ' "$LOGS/verilator_s0.log"

python3 - "$RESULT" "$VECTOR_MANIFEST" "$DCTF_RUN_JSON" <<'PY'
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

result = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
dctf_result = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
pattern = re.compile(
    r"^PASS CENTRAL96 REAL TRACE stage=S(?P<stage>\d+) heads=(?P<heads>\d+) "
    r"cycles=(?P<cycles>\d+) terms=(?P<terms>\d+) "
    r"logical_weight_req=(?P<logical_weight>\d+) "
    r"physical_weight_access=(?P<physical_weight>\d+) "
    r"logical_bias_req=(?P<logical_bias>\d+) "
    r"physical_bias_access=(?P<physical_bias>\d+) "
    r"final_beats=(?P<final_beats>\d+) "
    r"final_checks=(?P<final_checks>\d+)$",
    re.MULTILINE,
)
rows = []
for stage in range(4):
    log = (result / "logs" / f"icarus_s{stage}.log").read_text(
        encoding="utf-8"
    )
    match = pattern.search(log)
    if match is None:
        raise SystemExit(f"缺少S{stage} Icarus PASS计数")
    row = {key: int(value) for key, value in match.groupdict().items()}
    record = manifest["records"][stage]
    expected = {
        "heads": record["heads"],
        "terms": record["expected_issued_terms"],
        "logical_weight": record["expected_physical_weight_requests"] // 3,
        "physical_weight": record["expected_physical_weight_requests"],
        "logical_bias": record["expected_bias_requests"] // 3,
        "physical_bias": record["expected_bias_requests"],
        "final_beats": record["tokens"] * record["logical_supertiles"],
        "final_checks": record["expected_final_checks"],
    }
    for key, value in expected.items():
        if row[key] != value:
            raise SystemExit(
                f"S{stage}实测{key}={row[key]}，DCTF向量期望={value}"
            )
    dctf_row = dctf_result["Icarus"][stage]
    if dctf_row["stage"] != stage:
        raise SystemExit("DCTF实测结果stage顺序错误")
    row["dctf_cycles"] = int(dctf_row["cycles"])
    row["central_over_dctf_cycles"] = round(
        row["cycles"] / row["dctf_cycles"], 6
    )
    rows.append(row)

verilator_log = (result / "logs" / "verilator_s0.log").read_text(
    encoding="utf-8"
)
verilator_match = pattern.search(verilator_log)
if verilator_match is None:
    raise SystemExit("缺少S0 Verilator PASS计数")
verilator_row = {
    key: int(value) for key, value in verilator_match.groupdict().items()
}
icarus_s0 = {key: rows[0][key] for key in verilator_row}
if verilator_row != icarus_s0:
    raise SystemExit("S0 Icarus与Verilator计数不一致")

lines = [
    "# Central96 H67真实S0-S3 Projection-only RTL对照",
    "",
    f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    "- DUT：`gatestack_multihead_tile_projection_top`，`OUT_TILE=96`、`BANKS=2`。",
    "- 输入：直接复用DCTF S0-S3向量目录及其生成器单测；按logical supertile重放相同head、term和event负载。",
    "- 存储时序：weight与bias请求均在上升沿采样，固定下一拍返回；final两个96-wide端口全ready。",
    "- 物理访问口径：每次96-wide逻辑weight或bias握手统计为3次32-lane物理bank access。",
    "- 响应数据：行为模型仅在接口上显式拼接三个真实32-lane INT8/acc32向量切片。",
    "",
    "| Stage | Heads | Central cycles | DCTF cycles | Central/DCTF | terms | weight逻辑/物理 | bias逻辑/物理 | 96-wide final beats | acc32检查 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| S{row['stage']} | {row['heads']} | {row['cycles']} | "
        f"{row['dctf_cycles']} | {row['central_over_dctf_cycles']:.3f} | "
        f"{row['terms']} | {row['logical_weight']}/{row['physical_weight']} | "
        f"{row['logical_bias']}/{row['physical_bias']} | "
        f"{row['final_beats']} | {row['final_checks']} |"
    )
lines.extend(
    [
        "",
        "## 验证结论",
        "",
        "- Icarus S0-S3全部通过，两个96-wide final端口逐acc32 bit-exact，无重无漏。",
        f"- Verilator S0 + projection SVA通过；cycles={verilator_row['cycles']}，与Icarus一致。",
        "- DCTF向量逐文件SHA256检查和原生成器3项单测通过。",
        "- DUT `protocol_error` 与 `accumulator_overflow` 均未触发。",
        "",
        "## 证据边界",
        "",
        "这里的96-wide memory是固定一拍TB行为接口模型，不是已完成3个物理SRAM宏拆分、布局或时序签核的证据。3次物理bank access只作为与DCTF一致的统计口径；响应拼接明确来自三个32-lane真实向量切片。结果仅覆盖projection execution slice，INT8候选量化合同仍需valid825部署精度冻结。",
        "",
    ]
)
(result / "实测汇总.md").write_text("\n".join(lines), encoding="utf-8")

run_result = {
    "说明": "Central96与DCTF同term/event边界的H67真实S0-S3 projection-only RTL对照",
    "DUT参数": {"OUT_TILE": 96, "BANKS": 2},
    "存储模型": {
        "响应延迟拍数": 1,
        "weight逻辑请求对应物理bank_access": 3,
        "bias逻辑请求对应物理bank_access": 3,
        "响应拼接": "3个32-lane真实INT8/acc32切片",
        "物理实现声明": False,
    },
    "Icarus": rows,
    "Verilator_S0_projection_SVA": verilator_row,
    "结论": "全部通过，两个96-wide final逐acc32 bit-exact",
    "限制": "行为96-wide memory不能视为已物理拆宏",
}
(result / "实测结果.json").write_text(
    json.dumps(run_result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "PASS: Central96 H67真实S0-S3 Icarus及S0 Verilator+projection SVA全部通过"
