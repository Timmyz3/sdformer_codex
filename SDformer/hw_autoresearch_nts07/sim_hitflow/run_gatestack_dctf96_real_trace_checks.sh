#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_ABS="$(readlink -f "${BASH_SOURCE[0]}")"
RESULT="${RESULT_DIR:-$ROOT/results/gatestack_dctf96_real_trace_20260720}"
VECTOR_ROOT="$RESULT/vectors"
BUILD="$RESULT/build"
LOGS="$RESULT/logs"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/results/h67_real_bit_trace_20260717/manifest.json}"
VECTOR_MANIFEST="$RESULT/vectors_manifest.json"
PYTHON="${PYTHON:-python3}"
TB="tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv"
SVA="verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv"
BIND="verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv"
TOP="tb_gatestack_dctf96_banklocal_projection_real_trace"
RTL=(
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)
mkdir -p "$BUILD" "$LOGS" "$VECTOR_ROOT"
cd "$ROOT"

SOURCE_SET_SHA256="$("$PYTHON" - "$RUNNER_ABS" "$ROOT/$TB" "$ROOT/$SVA" "$ROOT/$BIND" "${RTL[@]/#/$ROOT/}" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
for value in sys.argv[1:]:
    path = Path(value).resolve()
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    digest.update(str(path).encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_digest.encode("ascii"))
    digest.update(b"\n")
print(digest.hexdigest())
PY
)"

iverilog -V >"$RESULT/iverilog_version_full.txt" 2>&1 || true
{
  sed -n '1p' "$RESULT/iverilog_version_full.txt"
  verilator --version
  "$PYTHON" --version
} >"$RESULT/tool_versions.txt"

PYTHONPATH=scripts "$PYTHON" -m unittest -v \
  scripts/test_generate_gatestack_dctf_real_trace_vectors.py \
  >"$LOGS/generator_unittest.log" 2>&1

PYTHONPATH=scripts "$PYTHON" scripts/generate_gatestack_dctf_real_trace_vectors.py \
  --manifest "$SOURCE_MANIFEST" \
  --output-root "$VECTOR_ROOT" \
  --result "$VECTOR_MANIFEST" \
  >"$LOGS/vector_generation.log" 2>&1

mapfile -t RECORD_ROWS < <("$PYTHON" - "$VECTOR_MANIFEST" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for record in manifest["records"]:
    vector_dir = Path(record["vector_dir"])
    vector_aggregate = hashlib.sha256(
        json.dumps(
            record["files"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    print(
        f"{record['stage']}\t{record['heads']}\t{record['tokens']}\t"
        f"{record['token_id_width']}\t{vector_dir.name}\t{vector_dir}\t"
        f"{record['name']}\t{vector_aggregate}"
    )
PY
)
if [[ ${#RECORD_ROWS[@]} -eq 0 ]]; then
  echo "No projection records in $VECTOR_MANIFEST" >&2
  exit 1
fi

for row in "${RECORD_ROWS[@]}"; do
  IFS=$'\t' read -r stage heads tokens token_id_width record_id vector_dir vector_name vector_aggregate <<<"$row"
  record_build="$BUILD/$record_id"
  mkdir -p "$record_build"
  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.STAGE=$stage" \
    -P"$TOP.HEADS=$heads" \
    -P"$TOP.TOKENS=$tokens" \
    -P"$TOP.TOKEN_ID_W=$token_id_width" \
    -o "$record_build/tb.vvp" "${RTL[@]}" "$TB" \
    >"$LOGS/iverilog_build_$record_id.log" 2>&1
  {
    printf 'RUN_RECEIPT simulator=icarus assertions=none vector_name=%s vector_id=%s vector_aggregate_sha256=%s source_set_sha256=%s stage=%s heads=%s tokens=%s\n' \
      "$vector_name" "$record_id" "$vector_aggregate" "$SOURCE_SET_SHA256" \
      "$stage" "$heads" "$tokens"
    vvp "$record_build/tb.vvp" "+VECTOR_DIR=$vector_dir"
  } | tee "$LOGS/icarus_$record_id.log"
  grep -q "^PASS DCTF96 REAL TRACE stage=S$stage " \
    "$LOGS/icarus_$record_id.log"
done

declare -A VERILATOR_STAGE_BUILT=()
for row in "${RECORD_ROWS[@]}"; do
  IFS=$'\t' read -r stage heads tokens token_id_width record_id vector_dir vector_name vector_aggregate <<<"$row"
  stage_build="$BUILD/verilator_s$stage"
  if [[ -z "${VERILATOR_STAGE_BUILT[$stage]:-}" ]]; then
    rm -rf "$stage_build"
    verilator --binary --timing --assert -Wall \
      --top-module "$TOP" \
      -GSTAGE="$stage" -GHEADS="$heads" \
      -GTOKENS="$tokens" -GTOKEN_ID_W="$token_id_width" \
      -Mdir "$stage_build" \
      "${RTL[@]}" "$TB" "$SVA" "$BIND" \
      >"$LOGS/verilator_build_s$stage.log" 2>&1
    if grep -Eq '(%Warning|%Error)' "$LOGS/verilator_build_s$stage.log"; then
      cat "$LOGS/verilator_build_s$stage.log" >&2
      exit 1
    fi
    VERILATOR_STAGE_BUILT[$stage]=1
  fi
  {
    printf 'RUN_RECEIPT simulator=verilator assertions=enabled vector_name=%s vector_id=%s vector_aggregate_sha256=%s source_set_sha256=%s stage=%s heads=%s tokens=%s\n' \
      "$vector_name" "$record_id" "$vector_aggregate" "$SOURCE_SET_SHA256" \
      "$stage" "$heads" "$tokens"
    "$stage_build/V$TOP" "+VECTOR_DIR=$vector_dir"
  } | tee "$LOGS/verilator_$record_id.log"
  grep -q "^PASS DCTF96 REAL TRACE stage=S$stage " \
    "$LOGS/verilator_$record_id.log"
done

"$PYTHON" - "$RESULT" "$ROOT" "$RUNNER_ABS" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

result = Path(sys.argv[1])
root = Path(sys.argv[2])
runner = Path(sys.argv[3])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"provenance source missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def source_set_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(resolved).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def vector_aggregate(record: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            record["files"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


manifest = json.loads((result / "vectors_manifest.json").read_text(encoding="utf-8"))
rtl_paths = [
    "rtl_hitflow/gatestack_decoupled_product_engine.sv",
    "rtl_hitflow/gatestack_dctf32_bank_executor.sv",
    "rtl_hitflow/gatestack_dctf_term_event_adapter.sv",
    "rtl_hitflow/gatestack_dctf_term_fabric.sv",
    "rtl_hitflow/gatestack_dctf96_term_datapath_top.sv",
    "rtl_hitflow/hitflow_banked_accumulator.sv",
    "rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv",
]
source_set_paths = [
    runner,
    root / "tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv",
    root / "verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv",
    root / "verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv",
    *(root / path for path in rtl_paths),
]
current_source_set_sha256 = source_set_digest(source_set_paths)
source_manifest_path = Path(manifest["source_manifest"])
source_manifest_sha = sha256_file(source_manifest_path)
if source_manifest_sha != manifest["source_manifest_sha256"]:
    raise SystemExit("source trace manifest changed after vector generation")
pattern = re.compile(
    r"^PASS DCTF96 REAL TRACE stage=S(?P<stage>\d+) heads=(?P<heads>\d+) "
    r"cycles=(?P<cycles>\d+) terms=(?P<terms>\d+) "
    r"physical_weight_req=(?P<weight>\d+) bias_req=(?P<bias>\d+) "
    r"final_checks=(?P<final>\d+)$",
    re.MULTILINE,
)
receipt_pattern = re.compile(
    r"^RUN_RECEIPT simulator=(?P<simulator>\w+) assertions=(?P<assertions>\w+) "
    r"vector_name=(?P<vector_name>\S+) vector_id=(?P<vector_id>\S+) "
    r"vector_aggregate_sha256=(?P<vector_sha>[0-9a-f]{64}) "
    r"source_set_sha256=(?P<source_sha>[0-9a-f]{64}) "
    r"stage=(?P<stage>\d+) heads=(?P<heads>\d+) tokens=(?P<tokens>\d+)$",
    re.MULTILINE,
)


def validate_receipt(
    log: str,
    record: dict[str, object],
    record_id: str,
    simulator: str,
    assertions: str,
) -> dict[str, object]:
    matches = receipt_pattern.findall(log)
    if len(matches) != 1:
        raise SystemExit(f"{record['name']} {simulator} receipt数量不是1")
    match = receipt_pattern.search(log)
    assert match is not None
    receipt = match.groupdict()
    expected = {
        "simulator": simulator,
        "assertions": assertions,
        "vector_name": str(record["name"]),
        "vector_id": record_id,
        "vector_sha": vector_aggregate(record),
        "source_sha": current_source_set_sha256,
        "stage": str(record["stage"]),
        "heads": str(record["heads"]),
        "tokens": str(record["tokens"]),
    }
    for key, value in expected.items():
        if receipt[key] != value:
            raise SystemExit(
                f"{record['name']} {simulator} receipt {key}={receipt[key]} != {value}"
            )
    return {
        "name": record["name"],
        "simulator": simulator,
        "assertions": assertions,
        "vector_id": record_id,
        "vector_aggregate_sha256": receipt["vector_sha"],
        "source_set_sha256": receipt["source_sha"],
    }
rows = []
run_receipts = []
for record in manifest["records"]:
    record_id = Path(record["vector_dir"]).name
    log = (result / "logs" / f"icarus_{record_id}.log").read_text(encoding="utf-8")
    match = pattern.search(log)
    if match is None:
        raise SystemExit(f"缺少{record['name']} Icarus PASS计数")
    row = {key: int(value) for key, value in match.groupdict().items()}
    row["name"] = record["name"]
    row["vector_id"] = record_id
    rows.append(row)
    run_receipts.append(
        validate_receipt(log, record, record_id, "icarus", "none")
    )

verilator_rows = []
for record, icarus_row in zip(manifest["records"], rows):
    record_id = Path(record["vector_dir"]).name
    verilator_log = (
        result / "logs" / f"verilator_{record_id}.log"
    ).read_text(encoding="utf-8")
    verilator_match = pattern.search(verilator_log)
    if verilator_match is None:
        raise SystemExit(f"缺少{record['name']} Verilator PASS计数")
    verilator_row = {
        key: int(value) for key, value in verilator_match.groupdict().items()
    }
    if any(verilator_row[key] != icarus_row[key] for key in verilator_row):
        raise SystemExit(f"{record['name']} Icarus与Verilator计数不一致")
    verilator_row["name"] = record["name"]
    verilator_row["vector_id"] = record_id
    verilator_rows.append(verilator_row)
    run_receipts.append(
        validate_receipt(
            verilator_log, record, record_id, "verilator", "enabled"
        )
    )

simulation_logs = []
for record in manifest["records"]:
    record_id = Path(record["vector_dir"]).name
    simulation_logs.append(
        {
            "name": record["name"],
            "icarus": file_binding(result / "logs" / f"icarus_{record_id}.log"),
            "verilator": file_binding(
                result / "logs" / f"verilator_{record_id}.log"
            ),
        }
    )

for row, record in zip(rows, manifest["records"]):
    expected = {
        "heads": record["heads"],
        "terms": record["expected_issued_terms"],
        "weight": record["expected_physical_weight_requests"],
        "bias": record["expected_bias_requests"],
        "final": record["expected_final_checks"],
    }
    for key, value in expected.items():
        if row[key] != value:
            raise SystemExit(
                f"S{row['stage']}实测{key}={row[key]}，向量期望={value}"
            )

lines = [
    "# DCTF96 H67真实S0-S3 Projection-only RTL回放实测",
    "",
    f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    "- 数据证据：H67 sample0、window0真实K/gate trace，真实INT8 projection weight与acc32 bias。",
    "- 映射：S0/S1/S2/S3分别为3/6/12/24 heads；每个逻辑supertile对应3个物理32-lane bank。",
    "- 存储模型：三个bank独立握手，请求在上升沿采样，响应寄存后于下一拍可见；weight与bias均为固定一拍。",
    "- 判定：六个final端口每次握手检查32个acc32元素，并检查每个输出head的162个token无重无漏。",
    "",
    "| Stage | Heads | Icarus cycles | 逻辑terms | 物理weight请求 | bias请求 | final逐元素检查 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['name']} | {row['heads']} | {row['cycles']} | "
        f"{row['terms']} | {row['weight']} | {row['bias']} | {row['final']} |"
    )
lines.extend(
    [
        "",
        "## 验证结论",
        "",
        f"- Icarus all-attention ({len(rows)} records)：全部通过，所有final元素bit-exact。",
        f"- Verilator all-attention + SVA：{len(verilator_rows)}条全部通过，逐条与Icarus一致。",
        "- Python向量生成器单测：3项全部通过。",
        "- stale weight/bias响应计数均为0，DUT protocol_error与accumulator_overflow均未触发。",
        "",
        "## 边界",
        "",
        "本结果只覆盖projection execution slice。INT8量化仍沿用源manifest标注的候选合同，不能替代valid825部署精度冻结。",
        "",
    ]
)
(result / "实测汇总.md").write_text("\n".join(lines), encoding="utf-8")

run_result = {
    "说明": "DCTF96 H67真实all-attention projection-only RTL回放实测",
    "Icarus": rows,
    "Verilator_all_records_SVA": verilator_rows,
    "结论": "全部通过，final逐元素bit-exact",
}
(result / "实测结果.json").write_text(
    json.dumps(run_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

run_context = manifest.get("source_run_context") or {}
artifact_identity = run_context.get("artifact_identity") or {}
checkpoint_sha = artifact_identity.get("checkpoint_sha256")

vector_payloads = []
for record in manifest["records"]:
    vector_dir = Path(record["vector_dir"])
    payload_files = []
    for name, expected in sorted((record.get("files") or {}).items()):
        path = vector_dir / name
        binding = file_binding(path)
        if binding["sha256"] != expected.get("sha256") or binding["bytes"] != int(
            expected.get("bytes", -1)
        ):
            raise SystemExit(f"projection vector payload changed: {path}")
        payload_files.append(binding)
    vector_payloads.append(
        {
            "name": record["name"],
            "record_manifest": file_binding(vector_dir / "manifest.json"),
            "files": payload_files,
        }
    )

source_artifacts = {
    "source_trace_manifest": file_binding(source_manifest_path),
    "vector_manifest": file_binding(result / "vectors_manifest.json"),
    "vector_generator": file_binding(
        root / "scripts/generate_gatestack_dctf_real_trace_vectors.py"
    ),
    "vector_generator_tests": file_binding(
        root / "scripts/test_generate_gatestack_dctf_real_trace_vectors.py"
    ),
    "runner": file_binding(runner),
    "testbench": file_binding(
        root / "tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv"
    ),
    "sva": file_binding(
        root / "verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv"
    ),
    "bind": file_binding(
        root / "verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv"
    ),
    "tool_versions": file_binding(result / "tool_versions.txt"),
    "generator_unittest_log": file_binding(
        result / "logs/generator_unittest.log"
    ),
    "vector_generation_log": file_binding(
        result / "logs/vector_generation.log"
    ),
    "simulation_logs": simulation_logs,
    "rtl": [file_binding(root / path) for path in rtl_paths],
}
report = {
    "schema": "h67_checkpoint_projection_rtl_exact_v4",
    "status": "PASS" if checkpoint_sha else "PASS_LEGACY_UNBOUND",
    "scope": "checkpoint_bound_real_weight_projection_component_rtl_exact_not_full_network",
    "checkpoint_identity": {
        "checkpoint_path": artifact_identity.get("checkpoint_path"),
        "checkpoint_sha256": checkpoint_sha,
        "config_path": artifact_identity.get("config_path"),
        "config_sha256": artifact_identity.get("config_sha256"),
    },
    "source_manifest": manifest["source_manifest"],
    "source_manifest_sha256": manifest["source_manifest_sha256"],
    "vector_manifest": str((result / "vectors_manifest.json").resolve()),
    "vector_manifest_sha256": source_artifacts["vector_manifest"]["sha256"],
    "source_run_context": run_context,
    "source_artifacts": source_artifacts,
    "source_set_sha256": current_source_set_sha256,
    "run_receipts": run_receipts,
    "report_generation": {
        "reused_completed_simulation_logs": os.environ.get(
            "HITFLOW_REPORT_REUSE", "0"
        )
        == "1",
        "note": (
            "若为true，仅重新解析并哈希已完成的双仿真器日志；"
            "不声称重新执行RTL。"
        ),
    },
    "vector_payloads": vector_payloads,
    "record_count": len(rows),
    "temporal_tokens": manifest["temporal_tokens"],
    "token_id_width": manifest["token_id_width"],
    "required_stage_coverage": sorted({row["stage"] for row in rows}),
    "weight_mode": "checkpoint_dyadic_int8_projection_weight",
    "verification": {
        "icarus_all_records_bit_exact": True,
        "verilator_all_records_sva_bit_exact": True,
        "record_count": len(rows),
    },
    "records": rows,
    "limits": [
        "只覆盖attention projection execution slice，不代表完整encoder或full-network RTL exact",
        "INT8 dyadic量化合同来自同一checkpoint生成的bit trace，仍需独立部署精度冻结",
    ],
}
(result / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "PASS: DCTF96 H67真实all-attention Icarus及Verilator+SVA逐record检查全部通过"
