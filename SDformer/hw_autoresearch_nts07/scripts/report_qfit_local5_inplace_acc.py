#!/usr/bin/env python3
"""汇总 Local5 标量/原位跨头累加四候选的可审计 RTL 对照。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qfit_local5_inplace_acc_20260809"
CANDIDATES = (
    "b0_scalar_recompute",
    "b1_scalar_memo",
    "b2_inplace_recompute",
    "b3_inplace_memo",
)
SEEDS = (17717, 44257, 48879)
PATTERN = re.compile(
    r"PASS Local5 multi-tile memo=(?P<memo>\d+) "
    r"inplace=(?P<inplace>\d+) seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) token=(?P<token>\d+) "
    r"hits=(?P<hits>\d+) fallback=(?P<fallback>\d+) "
    r"replay_records=(?P<replay_records>\d+) "
    r"partial=(?P<partial>\d+) final=(?P<final>\d+) "
    r"child_results=(?P<child_results>\d+) "
    r"weight_cycles=(?P<weight_cycles>\d+) "
    r"frontend_cycles=(?P<frontend_cycles>\d+) "
    r"readout_cycles=(?P<readout_cycles>\d+) "
    r"release_cycles=(?P<release_cycles>\d+) "
    r"rmw_cycles=(?P<rmw_cycles>\d+) "
    r"drain_cycles=(?P<drain_cycles>\d+) "
    r"scheduler_cycles=(?P<scheduler_cycles>\d+)"
)


def parse_log(path: Path) -> dict[str, int]:
    match = PATTERN.search(path.read_text())
    if not match:
        raise RuntimeError(f"无法解析 {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def yosys_stat(candidate: str) -> dict[str, int]:
    payload = json.loads((OUT / f"{candidate}_flat_stat.json").read_text())
    module_names = [
        name for name in payload["modules"]
        if name.endswith("\\qfit_local5_cross_head_tile_executor")
    ]
    if len(module_names) != 1:
        raise RuntimeError(
            f"Yosys 顶层匹配数错误 candidate={candidate}: {module_names}"
        )
    module = payload["modules"][module_names[0]]
    cells = module.get("num_cells_by_type", {})
    return {
        "generic_cells": int(module["num_cells"]),
        "wire_bits": int(module["num_wire_bits"]),
        "mem_v2": int(cells.get("$mem_v2", 0)),
    }


def main() -> None:
    rows: list[dict[str, int | str]] = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            reference = None
            for simulator in ("iverilog", "verilator_sva"):
                row = parse_log(
                    OUT / f"{candidate}_seed_{seed}_{simulator}.log"
                )
                if reference is None:
                    reference = row
                elif row != reference:
                    raise RuntimeError(
                        f"跨模拟器不一致 candidate={candidate} seed={seed}"
                    )
                rows.append({"candidate": candidate, "simulator": simulator, **row})

    primary = {
        candidate: [
            row for row in rows
            if row["candidate"] == candidate
            and row["simulator"] == "iverilog"
        ]
        for candidate in CANDIDATES
    }
    means = {
        candidate: {
            field: mean(float(row[field]) for row in candidate_rows)
            for field in (
                "cycles", "token", "partial", "final", "child_results",
                "weight_cycles", "frontend_cycles", "readout_cycles",
                "release_cycles", "rmw_cycles", "drain_cycles",
                "scheduler_cycles",
            )
        }
        for candidate, candidate_rows in primary.items()
    }
    cycle = {key: value["cycles"] for key, value in means.items()}
    comparisons = {
        "B2_vs_B0_inplace_only": cycle["b0_scalar_recompute"]
        / cycle["b2_inplace_recompute"],
        "B3_vs_B1_inplace_with_memo": cycle["b1_scalar_memo"]
        / cycle["b3_inplace_memo"],
        "B3_vs_B0_joint": cycle["b0_scalar_recompute"]
        / cycle["b3_inplace_memo"],
        "B1_vs_B0_memo_scalar": cycle["b0_scalar_recompute"]
        / cycle["b1_scalar_memo"],
        "B3_vs_B2_memo_inplace": cycle["b2_inplace_recompute"]
        / cycle["b3_inplace_memo"],
    }
    mapping = {candidate: yosys_stat(candidate) for candidate in CANDIDATES}
    summary = {
        "evidence": "rtl",
        "rows": rows,
        "means": means,
        "comparisons": comparisons,
        "storage_contract_bits": {
            "tcfm5_vector_acc_all_candidates": 460800,
            "extra_scalar_cross_head_acc_B0_B1": 460800,
            "extra_cross_head_acc_B2_B3": 0,
        },
        "scalar_cross_head_access_per_test": {
            "reads_B0_B1": 129600,
            "writes_B0_B1": 129600,
            "reads_B2_B3": 0,
            "writes_B2_B3": 0,
        },
        "yosys_open_proxy": mapping,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# Local5 TCFM5 原位跨头累加四候选 RTL 对照",
        "",
        "## 结论",
        "",
        (
            f"在相同三输入头、三输出 tile、T450/OUT32、定向 Q/K/weight 和随机反压下，"
            f"B2 原位 recompute 相对 B0 scalar recompute 平均加速 "
            f"{comparisons['B2_vs_B0_inplace_only']:.4f}x `[rtl]`；B3 原位 memo "
            f"相对 B0 平均加速 {comparisons['B3_vs_B0_joint']:.4f}x `[rtl]`。"
        ),
        (
            "原位路径删除独立 460800-bit 跨头 Acc 合同、129600 次中间 partial、"
            "129600 次额外 Acc 读和 129600 次额外 Acc 写 `[rtl合同]`。"
        ),
        "",
        "## 四候选周期",
        "",
        "| 候选 | mean 周期 | Token | child result | 跨头 partial | 相对 B0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        item = means[candidate]
        lines.append(
            f"| {candidate} | {item['cycles']:.1f} | {item['token']:.0f} | "
            f"{item['child_results']:.0f} | {item['partial']:.0f} | "
            f"{cycle['b0_scalar_recompute'] / item['cycles']:.4f}x |"
        )
    lines.extend([
        "",
        "## 周期分账",
        "",
        "| 候选 | weight | frontend | child readout | release | cross RMW* | external drain | scheduler |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for candidate in CANDIDATES:
        item = means[candidate]
        lines.append(
            f"| {candidate} | {item['weight_cycles']:.1f} | "
            f"{item['frontend_cycles']:.1f} | {item['readout_cycles']:.1f} | "
            f"{item['release_cycles']:.1f} | "
            f"{item['rmw_cycles']:.1f} | {item['drain_cycles']:.1f} | "
            f"{item['scheduler_cycles']:.1f} |"
        )
    lines.extend([
        "",
        "`cross RMW` 是 `child readout` 内部的诊断子集，不能再次加入总周期；"
        "其余列是近似互斥阶段，边界握手会造成少量计数差。",
        "",
        "## 开放工具边界",
        "",
        "| 候选 | generic cell | wire bit | $mem_v2 |",
        "|---|---:|---:|---:|",
    ])
    for candidate in CANDIDATES:
        item = mapping[candidate]
        lines.append(
            f"| {candidate} | {item['generic_cells']} | "
            f"{item['wire_bits']} | {item['mem_v2']} |"
        )
    lines.extend([
        "",
        "以上 Yosys 数字只用于结构差分，不是 ASIC 面积。行为级 1024-bit TCFM5 字"
        "仍需 SRAM macro banking、时序和活动功耗验证 `[待验证]`。",
        "",
        "## 证据边界",
        "",
        "- 四候选均与同一独立 Python oracle 逐 Acc32 零失配 `[rtl]`。",
        "- 当前是定向三 head/三 output-tile workload，不代表 fullres resident 分布。",
        "- 原位累加的核心收益来自删除中间物化；final 仍为 43200 个标量结果，"
        "不能把逻辑事务下降直接当作功耗或 PPA。",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
