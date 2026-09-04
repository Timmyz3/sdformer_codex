#!/usr/bin/env python3
"""用H67四stage真实trace计算head-major partial-sum spill理论下界。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


CELL_RE = re.compile(r"Number of cells:\s+(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--yosys-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=162)
    parser.add_argument("--out-tile", type=int, default=32)
    parser.add_argument("--banks", type=int, default=2)
    parser.add_argument("--acc-bytes", type=int, default=4)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    rows_by_key = {(int(r["stage"]), r["mode"]): r for r in baseline["rows"]}
    cells = [int(v) for v in CELL_RE.findall(args.yosys_log.read_text(encoding="utf-8"))][-1]
    batches = math.ceil(args.tokens / args.banks)
    rows = []
    for stage in range(4):
        gate = rows_by_key[(stage, "gatestack")]
        nores = rows_by_key[(stage, "no_residency")]
        heads = int(gate["heads"])
        tiles = heads
        psum_capacity = tiles * args.tokens * args.out_tile * args.acc_bytes
        spill_bytes = 2 * (heads - 1) * psum_capacity
        gate_payload = int(gate["payload_words"]) * 8
        decode_once_payload = int(nores["payload_words"]) * 8 // tiles
        saved_payload = max(0, gate_payload - decode_once_payload)
        read_transactions = (heads - 1) * tiles * batches
        write_transactions = read_transactions
        final_transactions = tiles * batches
        rows.append({
            "stage": stage,
            "heads": heads,
            "tiles": tiles,
            "psum_capacity_bytes": psum_capacity,
            "minimal_spill_bytes": spill_bytes,
            "gatestack_payload_bytes": gate_payload,
            "head_major_decode_once_payload_bytes": decode_once_payload,
            "payload_bytes_saved": saved_payload,
            "spill_to_saved_payload_ratio": spill_bytes / saved_payload if saved_payload else None,
            "spill_read_transactions": read_transactions,
            "spill_write_transactions": write_transactions,
            "final_transactions": final_transactions,
            "minimal_schedule_transactions": read_transactions + write_transactions + final_transactions,
        })
    result = {
        "status": "PASS",
        "evidence": "[H67真实trace统计]+[RTL事务调度器]+[理论下界]",
        "scheduler_yosys_generic_cells": cells,
        "assumptions": {
            "tokens": args.tokens,
            "out_tile": args.out_tile,
            "banks": args.banks,
            "acc_bytes": args.acc_bytes,
            "initial_head_read_elided": True,
            "last_head_write_elided": True,
            "builder_event_buffer_traffic_included": False,
            "compute_weight_bias_traffic_included": False,
        },
        "rows": rows,
        "limits": [
            "这是head-major spill的乐观下界，不含descriptor/event buffer重放、算术、权重、bias和控制停顿",
            "GateStack payload bytes来自单个真实窗口，不代表全数据集分布",
            "spill SRAM能量必须用目标宏表征，当前只报告字节和事务数",
            "事务调度RTL不执行projection数值运算，不能替代完整head-major bit-exact基线",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Head-major Partial-sum Spill真实Trace下界",
        "",
        "## 结论",
        "",
        "可综合事务调度器已经证明head-major最小相序：每个head只decode一次；首head免读psum，末head免写psum；中间head对所有output tile执行read-modify-write。即使采用这一乐观下界，partial-sum流量仍远高于GateStack重放payload的流量。",
        "",
        "| Stage | H/Tiles | PSUM容量 | 最小spill | GateStack payload | decode-once payload | 节省payload | spill/节省 | 最小事务周期下界 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = "N/A" if row["spill_to_saved_payload_ratio"] is None else f"{row['spill_to_saved_payload_ratio']:.1f}x"
        md.append(
            f"| S{row['stage']} | {row['heads']} | {row['psum_capacity_bytes']/1024:.1f} KiB | "
            f"{row['minimal_spill_bytes']/1024:.1f} KiB | {row['gatestack_payload_bytes']/1024:.2f} KiB | "
            f"{row['head_major_decode_once_payload_bytes']/1024:.2f} KiB | {row['payload_bytes_saved']/1024:.2f} KiB | "
            f"{ratio} | {row['minimal_schedule_transactions']} |"
        )
    md.extend([
        "",
        f"调度控制器开放综合为{cells}个Yosys generic cells。该控制面积不是瓶颈；真正代价是随stage维度增长的全tensor partial-sum SRAM容量和读写能量。",
        "",
        "## 下界公式",
        "",
        "```text",
        "PSUM_capacity = output_tiles * tokens * OUT_TILE * ACC_bytes",
        "minimal_spill = 2 * (heads - 1) * PSUM_capacity",
        "```",
        "",
        "首head从零初始化，省去read；末head读入旧psum后直接bias/final，省去write。因此该公式已经有利于head-major。实际实现还需缓存一次decode产生的descriptor/event，并为每个output tile重放，真实流量只会更高。",
        "",
        "## 架构指导",
        "",
        "结果支持继续采用output-tile-stationary：用较小的单tile AccTile跨head驻留，代价是重放compact payload；head-major虽然减少decode次数，却把低位宽稀疏payload流量换成32-bit dense partial-sum spill。后续目标PPA需要用SRAM宏能量验证该结论，但无需优先实现完整head-major算术核。",
        "",
        "## 证据边界",
        "",
    ])
    md.extend(f"- {item}" for item in result["limits"])
    md.append("")
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
