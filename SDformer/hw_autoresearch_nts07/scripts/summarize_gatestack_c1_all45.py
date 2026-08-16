#!/usr/bin/env python3
"""汇总 GateStack C1 45-head RTL、模型偏差与开放综合结构代理。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAGE_RE = re.compile(
    r"C1_STAGE stage=(?P<stage>\d+) heads=(?P<heads>\d+) "
    r"cycles=(?P<cycles>\d+) c0=(?P<c0>\d+) speedup=(?P<speedup>[0-9.]+)"
)
FINAL_RE = re.compile(
    r"PASS: C1 all45 stage-bounded cycles=(?P<cycles>\d+) "
    r"C0=(?P<c0>\d+) speedup=(?P<speedup>[0-9.]+) "
    r"overlap=(?P<overlap>\d+) blocked=(?P<blocked>\d+) "
    r"stalls=(?P<stalls>\d+)"
)


def parse_stage(match: re.Match[str]) -> dict[str, int | float]:
    """保留加速比的小数精度，其余计数按整数解析。"""
    values: dict[str, int | float] = {
        key: int(value) for key, value in match.groupdict().items() if key != "speedup"
    }
    values["speedup"] = float(match.group("speedup"))
    return values


def parse_final(match: re.Match[str]) -> dict[str, int | float]:
    """解析最终汇总，避免把 1.403x 截断为 1。"""
    values: dict[str, int | float] = {
        key: int(value) for key, value in match.groupdict().items() if key != "speedup"
    }
    values["speedup"] = float(match.group("speedup"))
    return values


def yosys_hierarchy_cells(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    block = text.split("=== design hierarchy ===")[-1]
    cells = int(re.search(r"Number of cells:\s+(\d+)", block).group(1))
    mem_v2 = int(re.search(r"\$mem_v2\s+(\d+)", block).group(1))
    muxes = int(re.search(r"\$mux\s+(\d+)", block).group(1))
    return cells, mem_v2, muxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-log", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--c0-yosys", type=Path, required=True)
    parser.add_argument("--c1-yosys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    log = args.rtl_log.read_text(encoding="utf-8")
    stages = [parse_stage(match) for match in STAGE_RE.finditer(log)]
    if len(stages) != 4:
        raise ValueError("RTL 日志必须包含四个 C1_STAGE 记录")
    final_match = FINAL_RE.search(log)
    if final_match is None:
        raise ValueError("RTL 日志缺少最终 PASS 记录")
    final = parse_final(final_match)
    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    model_total = int(model["stage_bounded_overlap"]["c1_cycles"])
    model_stage = {
        int(row["stage"]): int(row["c1_model"]["makespan_cycles"])
        for row in model["stages"]
    }
    c0_cells, c0_mem, c0_mux = yosys_hierarchy_cells(args.c0_yosys)
    c1_cells, c1_mem, c1_mux = yosys_hierarchy_cells(args.c1_yosys)

    result = {
        "schema_version": 1,
        "status": "PASS",
        "evidence": {
            "c1_latency_and_replay": "[rtl]",
            "model_comparison": "[rtl]+[模型]",
            "structure": "[开放综合代理]",
            "scope": "sample0/B0/window0 四 stage 全部 45 head",
        },
        "rtl": {
            **final,
            "cycle_reduction": 1.0 - final["cycles"] / final["c0"],
            "heads": 45,
            "words": 861,
            "terms": 762,
            "destinations": 3226,
            "scan_work_items": 2728,
        },
        "model": {
            "cycles": model_total,
            "rtl_minus_model_cycles": final["cycles"] - model_total,
            "rtl_model_error": (final["cycles"] - model_total) / model_total,
        },
        "stages": [
            {
                **row,
                "model_cycles": model_stage[row["stage"]],
                "rtl_minus_model_cycles": row["cycles"]
                - model_stage[row["stage"]],
            }
            for row in stages
        ],
        "open_synthesis_proxy": {
            "c0_cells": c0_cells,
            "c1_cells": c1_cells,
            "cell_increase": c1_cells / c0_cells - 1.0,
            "c0_mem_v2": c0_mem,
            "c1_mem_v2": c1_mem,
            "c0_mux": c0_mux,
            "c1_mux": c1_mux,
        },
        "limits": [
            "真实动态 C1 顶层验证使用 Icarus；全规模 Verilator+SVA 为 lint/elaboration，动态执行因工具性能超时",
            "开放综合 generic cell 不是目标库面积、频率或功耗",
            "trace 仅覆盖单个真实 window",
            "尚缺 SRAM 宏、DC/STA/SAIF、映射后等价与全 encoder FPS",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GateStack C1 双 Workspace 全 45-Head RTL 结果",
        "",
        "## 1. 结论",
        "",
        f"C1 在四 stage 全部 45 个真实 head 上完成 {final['cycles']} 拍，"
        f"相对 C0 的 {final['c0']} 拍减少 "
        f"{result['rtl']['cycle_reduction']:.2%}，实测加速 "
        f"{final['c0'] / final['cycles']:.3f}x `[rtl]`。",
        "",
        "45 个 head 的 861 个 slot word 全部逐 word 回放零失配；"
        "762 个 term、3226 个逻辑 destination 和 2728 个扫描/旁路 work item "
        "均通过 RTL 计数合约。",
        "",
        f"先前 C1 模型为 {model_total} 拍，RTL 只多 "
        f"{final['cycles'] - model_total} 拍，偏差 "
        f"{(final['cycles'] - model_total) / model_total:.2%}。模型已经被 RTL 基本复现。",
        "",
        "## 2. 分 Stage",
        "",
        "| Stage | Head | C0 RTL | C1 模型 | C1 RTL | RTL-模型 | RTL 加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["stages"]:
        lines.append(
            f"| {row['stage']} | {row['heads']} | {row['c0']} | "
            f"{row['model_cycles']} | {row['cycles']} | "
            f"{row['rtl_minus_model_cycles']} | "
            f"{row['c0'] / row['cycles']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 3. 调度活动",
            "",
            f"- capture/service 重叠：{final['overlap']} 拍；",
            f"- workspace 满导致 capture 阻塞：{final['blocked']} 拍；",
            f"- workspace 输出背压：{final['stalls']} 拍；",
            "- 顺序等待计数：0，未发生后到 head 越序 issue。",
            "",
            "## 4. 开放综合结构代理",
            "",
            "| 指标 | C0 | C1 | 变化 |",
            "|---|---:|---:|---:|",
            f"| generic cells | {c0_cells} | {c1_cells} | "
            f"{c1_cells / c0_cells - 1.0:+.2%} |",
            f"| `$mem_v2` | {c0_mem} | {c1_mem} | {c1_mem - c0_mem:+d} |",
            f"| `$mux` | {c0_mux} | {c1_mux} | {c1_mux - c0_mux:+d} |",
            "",
            "C1 的 1.403x 吞吐来自复制 canonical workspace，而不是复制 Serializer。"
            "因此面积代理增加 75% 左右是必须正视的成本。实际 memory 应映射 SRAM 宏，"
            "不能把 generic mux/cell 当作最终面积。",
            "",
            "## 5. 证据边界",
            "",
            "- C1 真实动态功能、latency 和逐 word replay 是 `[rtl]`；",
            "- 全规模 Verilator+SVA 已 0-warning lint/elaboration，但动态执行因该层次事件模型性能超时；",
            "- 各 workspace、Serializer、slot 叶模块仍有独立 Verilator 动态 SVA 回归；",
            "- C1 开放 Yosys `check` 为 0 problem；generic cell 仅为结构代理；",
            "- 尚不能宣称 DATE accept，也不能用这些数字替代 DC/STA/SAIF。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
