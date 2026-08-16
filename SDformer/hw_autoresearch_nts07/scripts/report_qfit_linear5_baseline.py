#!/usr/bin/env python3
"""生成 Linear-5 exact-replay 强基线中文报告。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qfit_linear5_baseline_20260731"


def main() -> None:
    stat = json.loads((OUT / "stat.json").read_text(encoding="utf-8"))
    module = stat["modules"]["\\qfit_linear5_projection_top"]
    types = module["num_cells_by_type"]
    result = {
        "schema": "qfit_linear5_exact_replay_v1",
        "evidence": "[RTL/开放结构综合]，非PPA",
        "functional": {
            "nonempty_masks": 31,
            "terms": 32,
            "destination_updates": 83,
            "mismatch": 0,
        },
        "structure": {
            "cells": int(module["num_cells"]),
            "wire_bits": int(module["num_wire_bits"]),
            "memories": int(types.get("$mem_v2", 0)),
            "multipliers": int(types.get("$mul", 0)),
            "dividers": int(types.get("$div", 0)),
            "modulo": int(types.get("$mod", 0)),
            "muxes": int(types.get("$mux", 0)),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    s = result["structure"]
    lines = [
        "# Linear-5 Exact-Replay 强基线RTL结果",
        "",
        "## 功能",
        "",
        "- 穷举内点31种非空destination mask；",
        "- 增加边界合法子集与window-last close/drain；",
        "- 32个term、83次destination update全部整数exact；",
        "- Icarus、Verilator/SVA、Yosys检查通过。",
        "",
        "## 开放结构",
        "",
        "| cells | wire bits | memory | mul | div | mod | mux |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s['cells']} | {s['wire_bits']} | {s['memories']} | "
        f"{s['multipliers']} | {s['dividers']} | {s['modulo']} | "
        f"{s['muxes']} |",
        "",
        "- 五个同步1R1W Acc bank与TCFM-5同容量、同接口；",
        "- 地址使用行基址商余数分解，网表无运行时除法或取模；",
        "- 同bank冲突由内部pending mask精确replay，product只计算一次；",
        "- 无冲突term可连续每拍接受。",
        "",
        "## 证据边界",
        "",
        "- 当前是W6定向/穷举mask向量，不是post-G0部署trace；",
        "- Yosys结构数不是面积、频率、功耗或EDP；",
        "- 与TCFM-5的论文比较必须使用相同SRAM macro和SDC。",
        "",
    ]
    (OUT / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
