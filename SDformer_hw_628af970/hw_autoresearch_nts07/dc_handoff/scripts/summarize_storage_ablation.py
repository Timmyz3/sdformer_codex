#!/usr/bin/env python3
"""汇总H67/H68精确深度与二次幂填充深度的通用综合对照。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "dc_handoff/runs/storage_ablation"
RESULTS = ROOT / "results"


def load(design: str, tag: str) -> dict[str, int]:
    raw = json.loads((RUN / f"{design}_{tag}.json").read_text(encoding="utf-8"))
    module = raw["modules"][f"\\{design}"]
    types = module["num_cells_by_type"]
    return {
        "总单元": int(module["num_cells"]),
        "触发器": sum(int(value) for name, value in types.items() if "DFF" in name),
        "多路器": int(types.get("$_MUX_", 0)),
        "组合单元": int(module["num_cells"]) - sum(
            int(value) for name, value in types.items() if "DFF" in name
        ),
    }


def reduction(exact: int, padded: int) -> float:
    return 1.0 - exact / padded


def main() -> int:
    rows = []
    for design, label, exact_bits, padded_bits in (
        ("h67_attention_top", "H67", 162 * 56 + 35 * 8, 256 * 56 + 64 * 8),
        ("h68_castling_deploy_top", "H68", 162 * 56 + 3 * 8, 256 * 56 + 4 * 8),
    ):
        exact = load(design, "exact")
        padded = load(design, "padded")
        rows.append(
            {
                "设计": label,
                "精确深度": exact,
                "二次幂填充": padded,
                "存储位_精确": exact_bits,
                "存储位_填充": padded_bits,
                "存储位下降": reduction(exact_bits, padded_bits),
                "总单元下降": reduction(exact["总单元"], padded["总单元"]),
                "触发器下降": reduction(exact["触发器"], padded["触发器"]),
                "多路器下降": reduction(exact["多路器"], padded["多路器"]),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "h67_h68_storage_ablation.json"
    md_path = RESULTS / "h67_h68_storage_ablation.md"
    json_path.write_text(json.dumps({"状态": "通过", "结果": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H67/H68物理存储深度综合对照",
        "",
        "## 结果",
        "",
        "| 设计 | 配置 | 存储位 | 总通用单元 | 触发器 | 多路器 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        exact = row["精确深度"]
        padded = row["二次幂填充"]
        lines.append(
            f"| {row['设计']} | 精确162深度 | {row['存储位_精确']} | {exact['总单元']} | "
            f"{exact['触发器']} | {exact['多路器']} |"
        )
        lines.append(
            f"| {row['设计']} | 填充256深度 | {row['存储位_填充']} | {padded['总单元']} | "
            f"{padded['触发器']} | {padded['多路器']} |"
        )
        lines.append(
            f"| {row['设计']} | 精确深度下降 | {row['存储位下降']:.2%} | {row['总单元下降']:.2%} | "
            f"{row['触发器下降']:.2%} | {row['多路器下降']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该结果使用Yosys通用门集合，比较同一RTL、同一算法接口下的编译期物理容量。"
            "它能证明消除无效地址容量会减少寄存器和选择逻辑，但不能换算为um2、MHz或mW。",
            "",
            "精确深度活动项库已经合并score、K和token，并由求和/发射阶段共享一个读端口。"
            "正式SRAM宏替换仍需同步读时序重排和目标宏模型回灌。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
