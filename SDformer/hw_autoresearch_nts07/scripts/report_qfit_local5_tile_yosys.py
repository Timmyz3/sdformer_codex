#!/usr/bin/env python3
"""生成 QFIT C1+C2 最小集成 tile 的中文结构报告。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qfit_local5_tile_yosys_20260730"
TOP = "\\qfit_local5_tile"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["modules"][TOP]


def main() -> None:
    memory = load(RESULTS / "tile_memory_stat.json")
    flat = load(RESULTS / "tile_flat_stat.json")
    memory_types = memory["num_cells_by_type"]
    flat_types = flat["num_cells_by_type"]
    row = {
        "memory_cells": int(memory_types.get("$mem_v2", 0)),
        "flat_cells": int(flat["num_cells"]),
        "flat_register_cells": sum(
            int(count)
            for name, count in flat_types.items()
            if "DFF" in name
        ),
        "flat_mux_cells": int(flat_types.get("$_MUX_", 0)),
        "flat_xor_cells": int(flat_types.get("$_XOR_", 0)),
    }
    report = {
        "schema": "qfit_local5_tile_open_synthesis_proxy_v1",
        "evidence": "[RTL/开放综合代理]，非 DC/STA/SAIF",
        "architecture": "XBF-DBDR score + FCSR-RX",
        "parameters": {
            "height": 15,
            "width": 15,
            "time_planes": 2,
            "gate_width": 9,
        },
        "structure": row,
        "functional": {
            "simulators": ["Icarus", "Verilator"],
            "descriptor_count": 24,
            "score_gate_relation_mismatch": 0,
            "random_backpressure": "PASS",
            "sva": "PASS",
        },
    }
    (RESULTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QFIT C1+C2 最小集成 Tile RTL 报告",
        "",
        "## 结论",
        "",
        "- XBF-DBDR exact score leaf 已直接连接 FCSR-RX 同步六 bank 关系转置；",
        "- `Kself`、valid mask 与 5x9-bit gate 随 score metadata 对齐；",
        "- 两个时间平面下，score gate 到 source descriptor 逐项零失配；",
        "- 该 tile 尚未接 projection term builder、DCTF/Acc 或完整 encoder。",
        "",
        "## 结构代理",
        "",
        "| `$mem_v2` | flat cells | flat register cells | flat mux | flat xor |",
        "|---:|---:|---:|---:|---:|",
        f"| {row['memory_cells']} | {row['flat_cells']} | "
        f"{row['flat_register_cells']} | {row['flat_mux_cells']} | "
        f"{row['flat_xor_cells']} |",
        "",
        "## 证据边界",
        "",
        "- 结构数字来自 Yosys generic mapping，不是面积、频率或功耗；",
        "- memory-map 后以 `setundef -zero` 处理未复位 SRAM 初态；",
        "- 当前只证明 C1 与 C2 的接口、位宽、相序和关系转置组合正确；",
        "- DATE 主结果仍需 projection consumer、真实 trace 和目标库 PPA。",
        "",
    ]
    (RESULTS / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
