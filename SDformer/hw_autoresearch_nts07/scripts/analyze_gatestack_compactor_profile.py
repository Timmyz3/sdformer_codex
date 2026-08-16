#!/usr/bin/env python3
"""从既有逐token K-count trace重建GateStack event compactor周期。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_hit_flow_ordered_profiles import decode_count_trace, percentile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/gatestack_compactor_profile_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_compactor_profile_20260715.md"


def reconstruct_row_k_counts(record: dict[str, Any]) -> np.ndarray:
    encoded = record["pair_k_count_ordered_trace"]
    shape = tuple(int(value) for value in encoded["shape"])
    if len(shape) != 4 or shape[0] != 2:
        raise ValueError(f"K-count trace不是[2,B,H,N]: {shape}")
    flat = np.asarray(decode_count_trace(encoded), dtype=np.int16)
    temporal = flat.reshape(shape)
    rows = temporal.transpose(1, 2, 0, 3).reshape(shape[1], shape[2], -1)
    if rows.shape[2] != int(record["tokens"]):
        raise ValueError("重建token数与record不一致")
    return rows


def compactor_cycles_by_row(record: dict[str, Any], width: int) -> list[int]:
    if width <= 0:
        raise ValueError("compactor width必须为正")
    rows = reconstruct_row_k_counts(record).astype(np.int64)
    cycles = ((rows + width - 1) // width).sum(axis=2)
    return cycles.reshape(-1).astype(np.int64).tolist()


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    by_stage: dict[int, dict[str, list[int]]] = {
        stage: {"active_tokens": [], "max_k": [], **{f"r{r}": [] for r in (1, 2, 4, 8)}}
        for stage in range(4)
    }
    active_sum_check = 0
    projection_sum_check = 0
    ideal_packed = {width: 0 for width in (1, 2, 4, 8)}
    exact_cycles = {width: 0 for width in (1, 2, 4, 8)}

    for record in profile["summary"]["h60_records"]:
        rows = reconstruct_row_k_counts(record).astype(np.int64)
        projection_active = decode_count_trace(
            record["projection_baseline_active_lanes_ordered_trace"]
        )
        active_by_row = rows.sum(axis=2).reshape(-1).tolist()
        if active_by_row != projection_active:
            raise ValueError("逐token K-count与projection active lanes不一致")
        stage = int(record["stage"])
        active_tokens = np.count_nonzero(rows, axis=2).reshape(-1).tolist()
        max_k = rows.max(axis=2).reshape(-1).tolist()
        by_stage[stage]["active_tokens"].extend(active_tokens)
        by_stage[stage]["max_k"].extend(max_k)
        active_sum_check += sum(active_by_row)
        projection_sum_check += sum(projection_active)
        for width in (1, 2, 4, 8):
            cycles = ((rows + width - 1) // width).sum(axis=2).reshape(-1).tolist()
            by_stage[stage][f"r{width}"].extend(cycles)
            exact_cycles[width] += sum(cycles)
            ideal_packed[width] += sum(math.ceil(value / width) for value in active_by_row)

    all_fields = {
        key: [value for stage in range(4) for value in by_stage[stage][key]]
        for key in by_stage[0]
    }
    return {
        "active_sum_exact": active_sum_check == projection_sum_check,
        "active_lanes": active_sum_check,
        "all": {key: _summary(values) for key, values in all_fields.items()},
        "stages": {
            stage: {key: _summary(values) for key, values in fields.items()}
            for stage, fields in by_stage.items()
        },
        "exact_vs_ideal": {
            width: {
                "exact_cycles": exact_cycles[width],
                "ideal_cross_token_packed_cycles": ideal_packed[width],
                "overhead": exact_cycles[width] / ideal_packed[width]
                if ideal_packed[width]
                else 1.0,
            }
            for width in (1, 2, 4, 8)
        },
    }


def render_md(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    lines = [
        "# GateStack Event Compactor真实逐Token Profile",
        "",
        f"输入：`{result['profile']}`。本报告直接解码既有 `pair_k_count_ordered_trace`，不重新占用GPU。",
        "",
        f"逐token K-count重建与projection active lanes一致：`{'PASS' if analysis['active_sum_exact'] else 'FAIL'}`。",
        "",
        "## 1. 全量分布",
        "",
        "| 指标 | mean | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "active_tokens": "活动token/head",
        "max_k": "单token最大K lane",
        "r1": "R1提取周期",
        "r2": "R2提取周期",
        "r4": "R4提取周期",
        "r8": "R8提取周期",
    }
    for key, label in labels.items():
        row = analysis["all"][key]
        lines.append(
            f"| {label} | {row['mean']:.3f} | {row['p50']:.1f} | {row['p95']:.1f} | "
            f"{row['p99']:.1f} | {row['max']} |"
        )
    lines += [
        "",
        "## 2. 不允许跨token理想打包的开销",
        "",
        "| R | 精确周期 | ceil(total/R)理想周期 | 开销倍率 |",
        "|---:|---:|---:|---:|",
    ]
    for width, row in analysis["exact_vs_ideal"].items():
        lines.append(
            f"| {width} | {row['exact_cycles']} | {row['ideal_cross_token_packed_cycles']} | "
            f"{row['overhead']:.4f}x |"
        )
    lines += [
        "",
        "## 3. 分stage的R4周期",
        "",
        "| Stage | active token mean/p99 | max K mean/p99 | R4 mean/p99/max |",
        "|---|---:|---:|---:|",
    ]
    for stage, fields in analysis["stages"].items():
        lines.append(
            f"| {stage} | {fields['active_tokens']['mean']:.2f}/{fields['active_tokens']['p99']:.0f} | "
            f"{fields['max_k']['mean']:.2f}/{fields['max_k']['p99']:.0f} | "
            f"{fields['r4']['mean']:.2f}/{fields['r4']['p99']:.0f}/{fields['r4']['max']} |"
        )
    lines += [
        "",
        "## 4. 架构含义",
        "",
        "- compactor不需要新GPU profile；逐token分布已经冻结在原始ordered trace中；",
        "- R路提取必须按每token `ceil(k_count/R)` 求和，不能用 `ceil(total_active/R)`；",
        "- R=4与R=8的边际收益需结合lane extractor和slot-counter多写口面积；",
        "- 本统计仍未包含packed slot SRAM的写bank冲突，该项必须由RTL随机trace replay给出。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "analysis": analyze(profile),
        "evidence": "[prof逐token K-count重建]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
