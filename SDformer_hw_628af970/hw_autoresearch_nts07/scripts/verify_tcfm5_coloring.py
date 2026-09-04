#!/usr/bin/env python3
"""穷举验证 TCFM-5 着色、地址单射与最小 bank 下界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLES = {
    "self": (0, 0),
    "south": (1, 0),
    "north": (-1, 0),
    "west": (0, -1),
    "east": (0, 1),
}


def color(y: int, x: int) -> int:
    return (x + 2 * y) % 5


def verify(height: int, width: int, planes: int) -> dict[str, object]:
    x_groups = (width + 4) // 5
    seen: set[tuple[int, int]] = set()
    full_neighborhoods = 0
    max_neighborhood = 0
    linear_conflict_hist = {cycles: 0 for cycles in range(1, 6)}

    for plane in range(planes):
        for y in range(height):
            for x in range(width):
                bank = color(y, x)
                address = (
                    plane * height * x_groups
                    + y * x_groups
                    + x // 5
                )
                key = (bank, address)
                if key in seen:
                    raise AssertionError(
                        f"bank/address collision: p={plane} y={y} x={x}"
                    )
                seen.add(key)

                neighbors = []
                linear_counts = [0] * 5
                for dy, dx in ROLES.values():
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbors.append(color(ny, nx))
                        linear_bank = (
                            plane * height * width + ny * width + nx
                        ) % 5
                        linear_counts[linear_bank] += 1
                if len(neighbors) != len(set(neighbors)):
                    raise AssertionError(
                        f"neighborhood color conflict: p={plane} y={y} x={x}"
                    )
                max_neighborhood = max(max_neighborhood, len(neighbors))
                if len(neighbors) == 5:
                    full_neighborhoods += 1
                linear_conflict_hist[max(linear_counts)] += 1

    tokens = height * width * planes
    if len(seen) != tokens:
        raise AssertionError("bank/address mapping is not injective")
    if height >= 3 and width >= 3 and max_neighborhood != 5:
        raise AssertionError("interior K5 lower-bound witness is missing")

    return {
        "height": height,
        "width": width,
        "planes": planes,
        "tokens": tokens,
        "banks": 5,
        "x_groups": x_groups,
        "allocated_entries": 5 * planes * height * x_groups,
        "address_utilization": tokens
        / (5 * planes * height * x_groups),
        "injective_bank_address": True,
        "conflict_free_all_neighborhoods": True,
        "maximum_neighborhood_size": max_neighborhood,
        "interior_k5_witnesses": full_neighborhoods,
        "minimum_banks_for_one_cycle": max_neighborhood,
        "linear5_replay_histogram": linear_conflict_hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--planes", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qfit_tcfm5_coloring_proof_20260731"
        ),
    )
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.planes <= 0:
        raise SystemExit("height/width/planes 必须为正数")

    result = verify(args.height, args.width, args.planes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "proof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    histogram = result["linear5_replay_histogram"]
    lines = [
        "# TCFM-5 穷举着色与地址证明",
        "",
        f"- 部署窗口：`T={args.planes}×{args.height}×{args.width}`，"
        f"共 {result['tokens']} 个 token；",
        "- `bank(x,y)=(x+2y) mod 5` 对所有有效五点邻域均无颜色冲突；",
        "- `(bank, local_address)` 对全部 token 为单射，无覆盖或别名；",
        f"- 内点 K5 见证数：{result['interior_k5_witnesses']}；",
        f"- 单周期、每 bank 单写口模型的最小 bank 数："
        f"{result['minimum_banks_for_one_cycle']}；",
        f"- bank 存储槽利用率：{result['address_utilization']:.2%}；",
        "",
        "## Linear-5 同端口映射的邻域 replay 分布",
        "",
        "| 每个五点集合所需周期 | source 数 |",
        "|---:|---:|",
    ]
    for cycles in range(1, 6):
        lines.append(f"| {cycles} | {histogram[cycles]} |")
    lines.extend(
        [
            "",
            "该分布只证明拓扑冲突，不包含 gate 等价类与 K 稀疏度；"
            "真实 product-term 周期必须由 post-G0 trace 统计。",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(
        "PASS TCFM-5 coloring "
        f"tokens={result['tokens']} k5={result['interior_k5_witnesses']}"
    )


if __name__ == "__main__":
    main()
