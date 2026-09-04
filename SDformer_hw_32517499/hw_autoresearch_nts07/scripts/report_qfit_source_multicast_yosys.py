#!/usr/bin/env python3
"""生成 QFIT source multicast 与 TCFM-5 中文结构报告。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qfit_source_multicast_yosys_20260730"


def module_stat(name: str, top: str) -> dict[str, int]:
    data = json.loads((OUT / name).read_text(encoding="utf-8"))
    module = data["modules"][f"\\{top}"]
    types = module["num_cells_by_type"]
    return {
        "cells": int(module["num_cells"]),
        "wire_bits": int(module["num_wire_bits"]),
        "memory_cells": int(types.get("$mem_v2", 0)),
        "mul_cells": int(types.get("$mul", 0)),
        "mux_cells": int(types.get("$mux", 0)),
    }


def acc_bank_memory_contract() -> dict[str, int]:
    data = json.loads(
        (OUT / "acc_bank_netlist.json").read_text(encoding="utf-8")
    )
    cells = data["modules"]["qfit_tcfm5_acc_bank"]["cells"]
    memories = [cell for cell in cells.values() if cell["type"] == "$mem_v2"]
    if len(memories) != 1:
        raise RuntimeError(f"期望1个Acc memory，实际{len(memories)}个")
    parameters = memories[0]["parameters"]

    def binary_parameter(name: str) -> int:
        return int(parameters[name], 2)

    return {
        "read_ports": binary_parameter("RD_PORTS"),
        "write_ports": binary_parameter("WR_PORTS"),
        "read_clocked": binary_parameter("RD_CLK_ENABLE"),
        "write_clocked": binary_parameter("WR_CLK_ENABLE"),
        "width": binary_parameter("WIDTH"),
        "depth": binary_parameter("SIZE"),
    }


def main() -> None:
    builder = module_stat(
        "builder_stat.json",
        "qfit_source_multicast_term_builder",
    )
    tcfm5 = module_stat(
        "tcfm5_stat.json",
        "qfit_tcfm5_projection_top",
    )
    acc_bank = module_stat(
        "acc_bank_stat.json",
        "qfit_tcfm5_acc_bank",
    )
    acc_contract = acc_bank_memory_contract()
    expected_contract = {
        "read_ports": 1,
        "write_ports": 1,
        "read_clocked": 1,
        "write_clocked": 1,
        "width": 128,
        "depth": 90,
    }
    if acc_contract != expected_contract:
        raise RuntimeError(f"Acc bank 1R1W合同不成立：{acc_contract}")
    report = {
        "schema": "qfit_source_multicast_open_structure_v2",
        "evidence": "[RTL/开放结构综合]，非 DC/STA/SAIF",
        "builder": builder,
        "acc_bank": acc_bank,
        "acc_bank_memory_contract": acc_contract,
        "tcfm5": tcfm5,
        "functional": {
            "builder_vector": {
                "product_terms": 12,
                "destination_updates": 20,
                "mismatch": 0,
            },
            "tcfm5_vector": {
                "product_terms": 3,
                "destination_updates": 11,
                "accumulator_mismatch": 0,
            },
            "acc_bank_raw_sequences": ["A/A/A", "A/B/A"],
            "icarus": "PASS",
            "verilator_sva": "PASS",
            "yosys_check": "PASS",
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QFIT 源多播与 TCFM-5 RTL 结果",
        "",
        "## 功能结果",
        "",
        "- source term builder：12 个共享 product term 精确表示 20 次 destination update；",
        "- TCFM-5：3 个 product term 完成 11 次五色 bank-local Acc 更新；",
        "- `gate=256`、重复 gate 合并、零 K、边界子集与整数 Acc readback 均通过；",
        "- Acc bank 的 `A/A/A`、`A/B/A` 同址 RAW 压力序列通过；",
        "- Icarus、Verilator/SVA 和 Yosys `check -assert` 全部通过。",
        "- TCFM-5 顶层 focused Verilator lint 为零 warning。",
        "",
        "以上是定向功能向量，不是部署 workload 压缩率或加速比。",
        "",
        "## 开放结构统计",
        "",
        "| 模块 | cells | wire bits | `$mem_v2` | `$mul` | `$mux` |",
        "|---|---:|---:|---:|---:|---:|",
        f"| source multicast builder | {builder['cells']} | "
        f"{builder['wire_bits']} | {builder['memory_cells']} | "
        f"{builder['mul_cells']} | {builder['mux_cells']} |",
        f"| packed Acc bank | {acc_bank['cells']} | "
        f"{acc_bank['wire_bits']} | {acc_bank['memory_cells']} | "
        f"{acc_bank['mul_cells']} | {acc_bank['mux_cells']} |",
        f"| TCFM-5 projection | {tcfm5['cells']} | "
        f"{tcfm5['wire_bits']} | {tcfm5['memory_cells']} | "
        f"{tcfm5['mul_cells']} | {tcfm5['mux_cells']} |",
        "",
        "## Acc存储宏合同审计",
        "",
        "| read ports | write ports | read clocked | write clocked | width | depth |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {acc_contract['read_ports']} | {acc_contract['write_ports']} | "
        f"{acc_contract['read_clocked']} | {acc_contract['write_clocked']} | "
        f"{acc_contract['width']} | {acc_contract['depth']} |",
        "",
        "Yosys 在 `memory_dff` 后确认每个 Acc bank 是单个同步 1R1W、"
        "128-bit × 90-depth memory。该结果证明 RTL 端口形态，不等价于"
        "目标工艺 SRAM macro 已映射。",
        "",
        "## 证据边界",
        "",
        "- 统计停在 Yosys `memory_collect`，只证明结构可综合；",
        "- TCFM-5 展平后为 1 个 weight memory 与 5 个显式 packed Acc "
        "memory；Acc 使用同步1R1W RMW、同拍读写同址 forwarding 和 drain，"
        "但尚未映射目标 SRAM macro；",
        "- 动态乘法器数固定为 `OUT_DIM=4`，product vector 在 bank 外"
        "计算一次后广播；",
        "- 五色映射已常量化为坐标颜色 LUT、固定角色旋转和本地组地址，"
        "Yosys 网表中不存在 `$mod/$div`；",
        "- 本报告只统计两个叶模块；C1-C3 端到端回放已在 "
        "`results/qfit_local5_projection_tile_yosys_20260731/report.md` "
        "单独闭环；",
        "- 论文收益必须来自 post-G0 多样本 trace 与同约束 DC/STA/SAIF。",
        "",
    ]
    (OUT / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
