#!/usr/bin/env python3
"""建立 Local5 role-sharded+final-reduction 的公平存储/读回下界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(
    *,
    height: int,
    width: int,
    planes: int,
    out_dim: int,
    acc_width: int,
) -> dict[str, object]:
    tokens = height * width * planes
    role_entries = {
        "self": tokens,
        "north": planes * max(height - 1, 0) * width,
        "south": planes * max(height - 1, 0) * width,
        "west": planes * height * max(width - 1, 0),
        "east": planes * height * max(width - 1, 0),
    }
    valid_edges = sum(role_entries.values())
    vector_width = out_dim * acc_width
    final_vector_adds = valid_edges - tokens
    return {
        "height": height,
        "width": width,
        "planes": planes,
        "tokens": tokens,
        "out_dim": out_dim,
        "acc_width": acc_width,
        "vector_width": vector_width,
        "tcfm5": {
            "acc_entries": tokens,
            "acc_bits": tokens * vector_width,
            "read_vectors": tokens,
            "final_vector_adds": 0,
        },
        "role_sharded_compressed_boundary": {
            "role_entries": role_entries,
            "acc_entries": valid_edges,
            "acc_bits": valid_edges * vector_width,
            "entry_ratio_vs_tcfm5": valid_edges / tokens,
            "read_vectors": valid_edges,
            "read_ratio_vs_tcfm5": valid_edges / tokens,
            "final_vector_adds": final_vector_adds,
            "final_scalar_adds": final_vector_adds * out_dim,
            "delivery_cycles_per_product_term": 1,
        },
        "contract": (
            "每个角色独立1R1W partial-Acc bank；边界无效edge不分配；"
            "读回时对同一destination的有效角色partial做exact整数归约"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--planes", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=4)
    parser.add_argument("--acc-width", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qfit_local5_role_sharded_baseline_20260731"
        ),
    )
    args = parser.parse_args()
    if min(
        args.height,
        args.width,
        args.planes,
        args.out_dim,
        args.acc_width,
    ) <= 0:
        raise SystemExit("所有尺寸参数必须为正数")

    result = evaluate(
        height=args.height,
        width=args.width,
        planes=args.planes,
        out_dim=args.out_dim,
        acc_width=args.acc_width,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    role = result["role_sharded_compressed_boundary"]
    tcfm = result["tcfm5"]
    lines = [
        "# Local5 Role-Sharded 强基线分账",
        "",
        f"- 部署窗口：`T={args.planes}×{args.height}×{args.width}`；",
        f"- Acc vector：`{args.out_dim}×{args.acc_width}="
        f"{result['vector_width']} bit`；",
        f"- 合同：{result['contract']}。",
        "",
        "## 存储与读回",
        "",
        "| 架构 | Acc entries | Acc bits | 读向量 | 最终向量加法 |",
        "|---|---:|---:|---:|---:|",
        f"| TCFM-5 | {tcfm['acc_entries']} | {tcfm['acc_bits']} | "
        f"{tcfm['read_vectors']} | 0 |",
        f"| Role-sharded，边界压缩 | {role['acc_entries']} | "
        f"{role['acc_bits']} | {role['read_vectors']} | "
        f"{role['final_vector_adds']} |",
        "",
        f"Role-sharded 的 Acc entry 与读向量均为 TCFM-5 的 "
        f"{role['entry_ratio_vs_tcfm5']:.3f} 倍；最终还需 "
        f"{role['final_vector_adds']} 次128-bit向量加法，即 "
        f"{role['final_scalar_adds']} 次32-bit标量加法。",
        "",
        "## 公平结论",
        "",
        "- 两者在无反压时都可做到一 product term 一 delivery 周期；",
        "- Role-sharded 通过复制 destination partial state 消除写冲突；",
        "- TCFM-5 通过拓扑着色让每个 destination 只保留一份 Acc，"
        "同时不需要最终归约；",
        "- 因此 Role-sharded 是吞吐相同但存储与读回更重的强基线，"
        "仍需RTL和同宏PPA确认控制/布线差异。",
        "",
        "该报告是结构下界模型，不是SRAM面积或能量结果。",
        "",
    ]
    (args.output_dir / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("PASS Local5 role-sharded baseline model")


if __name__ == "__main__":
    main()
