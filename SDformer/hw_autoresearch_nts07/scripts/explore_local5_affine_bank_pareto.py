#!/usr/bin/env python3
"""穷举 Local5 仿射 bank 着色的 bank-count/replay Pareto。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROLES = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1))


def evaluate_mapping(
    *,
    height: int,
    width: int,
    planes: int,
    banks: int,
    y_coefficient: int,
) -> dict[str, object]:
    bank_population = [0] * banks
    replay_histogram: Counter[int] = Counter()
    total_sources = height * width * planes
    total_replay_cycles = 0

    for plane in range(planes):
        for y in range(height):
            for x in range(width):
                bank = (x + y_coefficient * y) % banks
                bank_population[bank] += 1
                occupancy = [0] * banks
                for dy, dx in ROLES:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        occupancy[(nx + y_coefficient * ny) % banks] += 1
                replay = max(occupancy)
                replay_histogram[replay] += 1
                total_replay_cycles += replay

    depth = max(bank_population)
    allocated_entries = banks * depth
    return {
        "banks": banks,
        "formula": f"(x+{y_coefficient}*y)%{banks}",
        "y_coefficient": y_coefficient,
        "bank_population": bank_population,
        "bank_depth": depth,
        "allocated_entries": allocated_entries,
        "storage_utilization": total_sources / allocated_entries,
        "maximum_replay": max(replay_histogram),
        "mean_replay": total_replay_cycles / total_sources,
        "replay_histogram": {
            str(cycles): replay_histogram[cycles]
            for cycles in sorted(replay_histogram)
        },
    }


def explore(height: int, width: int, planes: int) -> dict[str, object]:
    candidates = []
    best_by_bank = {}
    for banks in range(2, 6):
        rows = [
            evaluate_mapping(
                height=height,
                width=width,
                planes=planes,
                banks=banks,
                y_coefficient=coefficient,
            )
            for coefficient in range(banks)
        ]
        rows.sort(
            key=lambda row: (
                row["mean_replay"],
                row["maximum_replay"],
                -row["storage_utilization"],
                row["y_coefficient"],
            )
        )
        best_by_bank[str(banks)] = rows[0]
        candidates.extend(rows)
    return {
        "height": height,
        "width": width,
        "planes": planes,
        "tokens": height * width * planes,
        "contract": (
            "bank=(x+b*y) mod B；每bank每拍最多一条同步1R1W更新；"
            "同一term冲突按最大bank占用精确replay"
        ),
        "best_by_bank_count": best_by_bank,
        "all_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--planes", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qfit_local5_affine_bank_pareto_20260731"),
    )
    args = parser.parse_args()
    if min(args.height, args.width, args.planes) <= 0:
        raise SystemExit("height/width/planes 必须为正数")

    result = explore(args.height, args.width, args.planes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Local5 仿射拓扑着色的 bank/replay Pareto",
        "",
        f"- 窗口：`T={args.planes}×{args.height}×{args.width}`；",
        f"- token：{result['tokens']}；",
        f"- 合同：{result['contract']}。",
        "",
        "| bank数 | 最佳公式 | 最大replay | 平均replay | bank深度 | 存储利用率 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for banks in range(2, 6):
        row = result["best_by_bank_count"][str(banks)]
        lines.append(
            f"| {banks} | `{row['formula']}` | {row['maximum_replay']} | "
            f"{row['mean_replay']:.4f} | {row['bank_depth']} | "
            f"{row['storage_utilization']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 架构解释",
            "",
            "- 5 bank 的最佳仿射着色达到一 term 一拍且零 replay；",
            "- 4 bank 因内点五邻域形成 K5，最坏至少两拍；",
            "- 4 bank 是否有更优 EDP 取决于真实 gate-equivalence mask，"
            "不能由全五角色上界决定；",
            "- 因此硬件候选应保留 `B=5 exact-zero-replay` 与 "
            "`B=4 exact-replay` 两点，并由 post-G0 trace 和同宏 PPA 选择。",
            "",
            "以上仅为固定拓扑穷举，不是模型 workload 周期。",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("PASS Local5 affine bank Pareto")


if __name__ == "__main__":
    main()
