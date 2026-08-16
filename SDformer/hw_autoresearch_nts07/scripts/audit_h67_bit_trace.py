#!/usr/bin/env python3
"""审计H67真实位级trace，并重算GateStack workload与容量统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def unpack_bits(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    bits = np.unpackbits(packed, bitorder="little")
    if bits.size < count:
        raise ValueError(f"packed bit不足: {bits.size} < {count}")
    return bits[:count].reshape(shape).astype(np.bool_)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def row_stats(k_row: np.ndarray, gate_row: np.ndarray) -> dict[str, Any]:
    if k_row.ndim != 2 or gate_row.ndim != 1:
        raise ValueError("row必须为K[tokens,lanes]与gate[tokens]")
    if k_row.shape[0] != gate_row.shape[0]:
        raise ValueError("row token数不一致")
    tokens, lanes = k_row.shape
    counts: dict[tuple[int, int], int] = {}
    for token in range(tokens):
        gate_code = int(gate_row[token])
        for lane in np.flatnonzero(k_row[token]):
            key = (gate_code, int(lane))
            counts[key] = counts.get(key, 0) + 1
    events = int(k_row.sum())
    terms = len(counts)
    classes = len({gate for gate, _lane in counts})
    max_fanout = max(counts.values(), default=0)
    ipd_bits = 128 + math.ceil(terms / 2) * 64 + events * 8
    raw_bits = tokens * (lanes + 9)
    return {
        "events": events,
        "terms": terms,
        "classes": classes,
        "max_fanout": max_fanout,
        "ipd_bits": ipd_bits,
        "raw_bits": raw_bits,
        "mode": "IPD32W" if ipd_bits <= raw_bits else "RAW41",
    }


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["file"])
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if digest != record["sha256"]:
        raise ValueError(f"SHA256不匹配: {path}")
    with np.load(path) as payload:
        q_shape = tuple(int(value) for value in payload["q_shape"])
        k_shape = tuple(int(value) for value in payload["k_shape"])
        q_bits = unpack_bits(payload["q_bits_packed"], q_shape)
        k_bits = unpack_bits(payload["k_bits_packed"], k_shape)
        gate = payload["gate_q17"]
        weight_float = payload["projection_weight_float32"]
        weight_int8 = payload["projection_weight_int8"]
        weight_exp = payload["projection_weight_scale_exp2"]
        bias_float = payload["projection_bias_float32"]
        bias_acc = payload["projection_bias_acc_int64"]

    if len(q_shape) != 5 or len(k_shape) != 5:
        raise ValueError("Q/K shape必须是[2,W,H,N,D]")
    if q_shape != k_shape or q_shape[0] != 2:
        raise ValueError("Q/K shape必须一致且T=2")
    _, windows, heads, spatial_tokens, lanes = q_shape
    if tuple(gate.shape) != (windows, heads, 2 * spatial_tokens):
        raise ValueError("gate shape与Q/K不一致")
    if gate.min(initial=0) < 0 or gate.max(initial=0) > 256:
        raise ValueError("gate Q1.7越界")
    dim = heads * lanes
    if tuple(weight_float.shape) != (dim, dim):
        raise ValueError(f"projection weight应为[{dim},{dim}]")
    if weight_int8.shape != weight_float.shape or weight_exp.shape != (dim,):
        raise ValueError("INT8 weight/scale shape错误")
    if bias_float.shape != (dim,) or bias_acc.shape != (dim,):
        raise ValueError("bias shape错误")
    scale = np.exp2(weight_exp.astype(np.float32))[:, None]
    restored = weight_int8.astype(np.float32) * scale
    quant_error = np.abs(restored - weight_float)
    if np.any(quant_error > scale / 2 + 1e-6):
        raise ValueError("dyadic INT8权重量化误差超过半个量化步长")

    row_records: list[dict[str, Any]] = []
    for window in range(windows):
        for head in range(heads):
            k_row = k_bits[:, window, head].reshape(2 * spatial_tokens, lanes)
            row = row_stats(k_row, gate[window, head])
            row.update({"window": window, "head": head})
            row_records.append(row)
    total_events = sum(row["events"] for row in row_records)
    total_terms = sum(row["terms"] for row in row_records)
    direct_work = total_events
    reduction = 1.0 - total_terms / direct_work if direct_work else 0.0
    return {
        "name": record["name"],
        "sample_id": int(record["sample_id"]),
        "file": str(path),
        "sha256_ok": True,
        "shape": {
            "windows": windows,
            "heads": heads,
            "spatial_tokens": spatial_tokens,
            "temporal_tokens": 2 * spatial_tokens,
            "lanes": lanes,
            "dim": dim,
        },
        "q_active_bits": int(q_bits.sum()),
        "k_active_bits": int(k_bits.sum()),
        "gate_nonzero": int(np.count_nonzero(gate)),
        "direct_active_lane_work": direct_work,
        "gatestack_equivalent_terms": total_terms,
        "equivalent_term_reduction_ratio": reduction,
        "ipd_rows": sum(row["mode"] == "IPD32W" for row in row_records),
        "raw_rows": sum(row["mode"] == "RAW41" for row in row_records),
        "max_terms_per_row": max((row["terms"] for row in row_records), default=0),
        "max_fanout": max((row["max_fanout"] for row in row_records), default=0),
        "active_classes_p95": float(
            np.percentile([row["classes"] for row in row_records], 95)
        ),
        "weight_quant_max_abs_error": float(quant_error.max(initial=0.0)),
        "weight_scale_exp2_min": int(weight_exp.min(initial=0)),
        "weight_scale_exp2_max": int(weight_exp.max(initial=0)),
        "rows": row_records,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# H67真实位级Trace数据质量与GateStack统计审计",
        "",
        "## 结论",
        "",
        f"- 审计状态：**{result['status']}**",
        f"- 记录数：{len(result['records'])}",
        f"- stage覆盖：{result['coverage']['stages']}",
        f"- 四stage完整：{result['coverage']['four_stage_complete']}",
        "- Q/K/gate属于真实网络位级trace；INT8 projection weight仍是候选合同，未经valid825验证前不得称为冻结部署量化。",
        "",
        "## 逐记录统计",
        "",
        "| 模块 | 样本 | 窗口 | head | direct work | 等价term | work减少 | IPD行 | RAW行 | max fanout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["records"]:
        shape = row["shape"]
        lines.append(
            f"| {row['name']} | {row['sample_id']} | {shape['windows']} | {shape['heads']} | "
            f"{row['direct_active_lane_work']} | {row['gatestack_equivalent_terms']} | "
            f"{row['equivalent_term_reduction_ratio']:.4%} | {row['ipd_rows']} | "
            f"{row['raw_rows']} | {row['max_fanout']} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 本报告的work减少是从真实K位图与真实Q1.7 gate重算的逻辑乘积项，不是周期、功耗或EDP。",
            "- 必须将同一trace送入Direct、no-residency和GateStack RTL，才能形成公平性能消融。",
            "- projection权重的dyadic INT8误差仅证明编码器算术自洽，不代表网络精度通过。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(
    manifest_path: Path,
    *,
    require_four_stages: bool = False,
    require_records: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [audit_record(record) for record in manifest.get("records", [])]
    stages = sorted(
        {
            int(row["name"].split(".")[0][1:])
            for row in records
            if row["name"].startswith("S")
        }
    )
    four_stage_complete = stages == [0, 1, 2, 3]
    if require_four_stages and not four_stage_complete:
        raise ValueError(f"四stage覆盖不完整: {stages}")
    if require_records is not None and len(records) != require_records:
        raise ValueError(f"trace记录数不匹配: {len(records)} != {require_records}")
    return {
        "status": "PASS",
        "source_manifest": str(manifest_path),
        "coverage": {
            "stages": stages,
            "four_stage_complete": four_stage_complete,
        },
        "records": records,
        "limits": manifest.get("limits", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-four-stages", action="store_true")
    parser.add_argument("--require-records", type=int, default=None)
    args = parser.parse_args()
    result = audit(
        args.manifest,
        require_four_stages=args.require_four_stages,
        require_records=args.require_records,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    md_path = args.output_dir / "audit.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(md_path, result)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
