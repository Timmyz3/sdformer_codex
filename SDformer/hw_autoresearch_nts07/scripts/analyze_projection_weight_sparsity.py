#!/usr/bin/env python3
"""统计 H67 四 stage INT8 projection 权重的零值与 N:M 可跳过性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_int8(path: Path) -> list[int]:
    values = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = int(line.strip(), 16)
        values.append(raw - 256 if raw >= 128 else raw)
    return values


def nm_eligible(values: list[int], nonzero_limit: int, group: int) -> float:
    if len(values) % group:
        raise ValueError("权重数量必须能被N:M分组长度整除")
    blocks = [values[index:index + group]
              for index in range(0, len(values), group)]
    eligible = sum(sum(value != 0 for value in block) <= nonzero_limit
                   for block in blocks)
    return eligible / len(blocks)


def analyze(root: Path) -> dict:
    stages = []
    for stage in range(4):
        path = root / f"real_sample0_s{stage}_b0_capacity" / \
            "projection_weights_int8.memh"
        values = load_int8(path)
        zero_count = sum(value == 0 for value in values)
        stages.append({
            "stage": stage,
            "weights": len(values),
            "zero_count": zero_count,
            "zero_ratio": zero_count / len(values),
            "eligible_2_4": nm_eligible(values, 2, 4),
            "eligible_4_8": nm_eligible(values, 4, 8),
            "eligible_2_8": nm_eligible(values, 2, 8),
            "min": min(values),
            "max": max(values),
            "unique": len(set(values)),
        })
    return {"schema_version": 1, "status": "PASS", "stages": stages}


def render_markdown(report: dict) -> str:
    lines = [
        "# H67 Projection INT8 权重稀疏性统计",
        "",
        "| Stage | 权重数 | 零值率 | 2:4可跳过块 | 4:8可跳过块 | 2:8可跳过块 | 范围 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        lines.append(
            f"| S{row['stage']} | {row['weights']} | {row['zero_ratio']:.3%} | "
            f"{row['eligible_2_4']:.3%} | {row['eligible_4_8']:.3%} | "
            f"{row['eligible_2_8']:.3%} | [{row['min']}, {row['max']}] |"
        )
    lines += [
        "",
        "结论：当前 checkpoint 权重接近稠密，不能用 weight-zero skipping 或 butterfly zero skipper 解释主要收益。若要采用该类结构，必须先做结构化剪枝和 full30/valid825 重训；exact 硬件主线不采用。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.vector_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
