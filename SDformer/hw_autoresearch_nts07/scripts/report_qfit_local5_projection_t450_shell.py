#!/usr/bin/env python3
"""生成 Local5 当前单顶层的合成 T450 部署壳中文报告。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qfit_local5_projection_t450_shell_20260809"
BUILD = ROOT / "build_qfit/local5_projection_t450_shell"
PATTERN = re.compile(
    r"backend=(\d+) tile_cycles=(\d+) descriptors=(\d+) "
    r"terms=(\d+) updates=(\d+) stalls=(\d+) "
    r"issue_seed=(\d+) issue_stall_cycles=(\d+) "
    r"issue_block_hits=(\d+) lane_gate_products=(\d+) "
    r"lane1_hits=(\d+) lane1_misses=(\d+) lane2_hits=(\d+) "
    r"lane2_misses=(\d+) dm16_hits=(\d+) dm16_misses=(\d+) "
    r"linear5_cycles=(\d+) affine4_cycles=(\d+) "
    r"single_cycles=(\d+) acc32_mismatch=(\d+) "
    r"python_fullchain_miter=(\d+) python_acc32_mismatch=(\d+)"
)
STALL_PATTERN = re.compile(
    r"LOCAL5_STALL_COVERAGE mode=(\d+) first_hold=(\d+) "
    r"final_hold=(\d+) final_done=(\d+)"
)
GENERATION_PATTERN = re.compile(
    r"seed=(\d+) out_dim=(\d+) inputs=(\d+) acc32=(\d+) "
    r"terms=(\d+) updates=(\d+)"
)


def parse_log(name: str) -> tuple[int, ...]:
    log = (OUT / name).read_text(encoding="utf-8")
    match = PATTERN.search(log)
    if not match:
        raise RuntimeError(f"无法解析 {name} 的完整 T450 计数")
    values = tuple(map(int, match.groups()))
    if values[19] != 0 or values[20:] != (1, 0):
        raise RuntimeError(f"{name} 未通过 Python Acc32 全链 miter")
    return values


def parse_stall_coverage(name: str) -> tuple[int, int, int, int]:
    log = (OUT / name).read_text(encoding="utf-8")
    match = STALL_PATTERN.search(log)
    if not match:
        raise RuntimeError(f"无法解析 {name} 的定向反压 coverage")
    return tuple(map(int, match.groups()))


def parse_generation(name: str) -> tuple[int, ...]:
    text = (OUT / name).read_text(encoding="utf-8")
    match = GENERATION_PATTERN.search(text)
    if not match:
        raise RuntimeError(f"无法解析 {name} 的 T450 oracle 账本")
    return tuple(map(int, match.groups()))


def parameter_int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 2)


def validate_out32_observability() -> None:
    columns: dict[int, list[int]] = {out: [] for out in range(32)}
    with (BUILD / "t450_out32_expected.txt").open(encoding="ascii") as handle:
        for line in handle:
            plane, y, x, out, value = map(int, line.split())
            if not (0 <= plane < 2 and 0 <= y < 15 and 0 <= x < 15):
                raise RuntimeError("T450 OUT32 expected 地址越界")
            if out not in columns:
                raise RuntimeError("T450 OUT32 expected 输出地址越界")
            columns[out].append(value)
    if any(len(values) != 450 for values in columns.values()):
        raise RuntimeError("T450 OUT32 expected 未覆盖每个输出的 450 个 Acc32")
    if len({tuple(values) for values in columns.values()}) != 32:
        raise RuntimeError("T450 OUT32 输出列不可区分，无法杀死地址混叠")


def main() -> int:
    oracle_hashes = (OUT / "oracle_hashes.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    source_hashes = (OUT / "source_hashes.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    oracle_hash_check = (OUT / "oracle_hash_check.log").read_text(
        encoding="utf-8"
    ).splitlines()
    source_hash_check = (OUT / "source_hash_check.log").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(oracle_hashes) != 8 or len(oracle_hash_check) != 8:
        raise RuntimeError("T450 Python oracle SHA-256 账本不完整")
    if not source_hashes or len(source_hashes) != len(source_hash_check):
        raise RuntimeError("T450 RTL/验证源码 SHA-256 账本不完整")
    if any(not line.endswith(": OK") for line in oracle_hash_check + source_hash_check):
        raise RuntimeError("T450 SHA-256 自校验失败")

    seed, out_dim, inputs, acc32, oracle_terms, oracle_updates = parse_generation(
        "oracle_generation.log"
    )
    if (seed, out_dim, inputs, acc32) != (0x45052026, 2, 450, 900):
        raise RuntimeError("T450 Python oracle 几何或随机种子错误")

    tcfm = parse_log("tcfm5_iverilog.log")
    linear = parse_log("linear5_iverilog.log")
    verilator = parse_log("tcfm5_verilator_sva.log")
    bp_iverilog = parse_log("tcfm5_backpressure_seed_17717_iverilog.log")
    bp_verilator = parse_log("tcfm5_backpressure_verilator_sva.log")
    lfsr_runs = {
        seed_value: parse_log(f"tcfm5_backpressure_seed_{seed_value}_iverilog.log")
        for seed_value in (1, 17717, 48879)
    }
    directed_runs = {
        mode: parse_log(f"tcfm5_directed_mode_{mode}_iverilog.log")
        for mode in (2, 3, 4)
    }
    directed_verilator = {
        mode: parse_log(f"tcfm5_directed_mode_{mode}_verilator_sva.log")
        for mode in (2, 3, 4)
    }
    directed_coverage = {
        mode: parse_stall_coverage(f"tcfm5_directed_mode_{mode}_iverilog.log")
        for mode in (2, 3, 4)
    }
    directed_verilator_coverage = {
        mode: parse_stall_coverage(f"tcfm5_directed_mode_{mode}_verilator_sva.log")
        for mode in (2, 3, 4)
    }
    data_runs = {}
    for seed_value in (0x45052027, 0x45052028):
        tag = f"{seed_value:x}"
        generated = parse_generation(f"oracle_generation_{tag}.log")
        run = parse_log(f"tcfm5_data_seed_{tag}_iverilog.log")
        if generated[:4] != (seed_value, 2, 450, 900):
            raise RuntimeError(f"附加 T450 数据种子几何错误: {tag}")
        if run[2:5] != (450, generated[4], generated[5]):
            raise RuntimeError(f"附加 T450 数据种子工作量不一致: {tag}")
        data_runs[tag] = run
    out32_generation = parse_generation("oracle_generation_out32.log")
    out32 = parse_log("tcfm5_out32_iverilog.log")
    out32_verilator = parse_log("tcfm5_out32_verilator_sva.log")
    if out32_generation[:4] != (0x45052026, 32, 450, 14400):
        raise RuntimeError("T450 OUT32 oracle 几何错误")
    if out32[2:5] != (450, out32_generation[4], out32_generation[5]):
        raise RuntimeError("T450 OUT32 RTL 工作量与 Python 不一致")
    if out32 != out32_verilator:
        raise RuntimeError("T450 OUT32 Icarus 与 Verilator/SVA 完整计数不一致")
    validate_out32_observability()

    if tcfm != verilator:
        raise RuntimeError("T450 基线 Icarus 与 Verilator/SVA 完整计数不一致")
    if bp_iverilog != bp_verilator:
        raise RuntimeError("T450 反压 Icarus 与 Verilator/SVA 完整计数不一致")
    if tcfm[2:5] != (450, oracle_terms, oracle_updates):
        raise RuntimeError("T450 Python 与 TCFM-5 工作量账本不一致")
    if linear[2:5] != tcfm[2:5] or linear[0] != 2:
        raise RuntimeError("Linear-5 强基线没有消费相同 T450 工作量")
    if bp_iverilog[2:5] != tcfm[2:5]:
        raise RuntimeError("T450 反压改变了有效工作量")
    if bp_iverilog[6] != 17717 or bp_iverilog[7] <= 0 or bp_iverilog[8] <= 0:
        raise RuntimeError("T450 固定种子反压未命中真实 term 边界")
    for seed_value, run in lfsr_runs.items():
        if run[2:5] != tcfm[2:5] or run[6] != seed_value or run[8] <= 0:
            raise RuntimeError(f"T450 LFSR 反压种子 {seed_value} 未保持工作量")
        verilator_name = (
            "tcfm5_backpressure_verilator_sva.log"
            if seed_value == 17717
            else f"tcfm5_backpressure_seed_{seed_value}_verilator_sva.log"
        )
        if run != parse_log(verilator_name):
            raise RuntimeError(f"T450 LFSR 反压种子 {seed_value} 跨仿真器不一致")
    for tag, run in data_runs.items():
        if run != parse_log(f"tcfm5_data_seed_{tag}_verilator_sva.log"):
            raise RuntimeError(f"T450 数据种子 {tag} 跨仿真器不一致")
    for mode in (2, 3, 4):
        if directed_runs[mode] != directed_verilator[mode]:
            raise RuntimeError(f"T450 定向反压 mode={mode} 跨仿真器不一致")
        if directed_runs[mode][2:5] != tcfm[2:5] or directed_runs[mode][8] <= 0:
            raise RuntimeError(f"T450 定向反压 mode={mode} 未保持工作量")
        if directed_coverage[mode] != directed_verilator_coverage[mode]:
            raise RuntimeError(f"T450 定向反压 mode={mode} coverage 跨仿真器不一致")
        if directed_coverage[mode][0] != mode:
            raise RuntimeError(f"T450 定向反压 mode={mode} coverage 标签错误")
    if directed_coverage[2][1] != 1:
        raise RuntimeError("T450 首 term 全停模式未命中")
    if directed_coverage[4][2:] != (1, 1):
        raise RuntimeError("T450 最后 term 定向停顿未完成")

    stat = json.loads((OUT / "stat.json").read_text(encoding="utf-8"))
    module = stat["modules"]["\\qfit_local5_projection_tile"]
    cell_types = module["num_cells_by_type"]
    netlist = json.loads((OUT / "yosys_netlist.json").read_text(encoding="utf-8"))
    netlist_module = netlist["modules"]["qfit_local5_projection_tile"]
    memories = []
    for cell_name, cell in netlist_module["cells"].items():
        if cell["type"] != "$mem_v2":
            continue
        params = cell["parameters"]
        memories.append(
            {
                "name": cell_name,
                "size": parameter_int(params["SIZE"]),
                "width": parameter_int(params["WIDTH"]),
                "rd_ports": parameter_int(params["RD_PORTS"]),
                "wr_ports": parameter_int(params["WR_PORTS"]),
            }
        )
    acc_memories = [
        item for item in memories
        if item["size"] == 90 and item["width"] == 32 * 32
        and item["rd_ports"] == 1 and item["wr_ports"] == 1
    ]
    relation_k = [
        item for item in memories
        if item["size"] == 45 and item["width"] == 32
        and item["rd_ports"] == 1 and item["wr_ports"] == 1
    ]
    relation_gate = [
        item for item in memories
        if item["size"] == 45 and item["width"] == 10
        and item["rd_ports"] == 1 and item["wr_ports"] == 1
    ]
    weight_memories = [
        item for item in memories
        if item["size"] == 32 and item["width"] == 32 * 8
        and item["rd_ports"] == 1 and item["wr_ports"] == 1
    ]
    if len(memories) != 23 or len(acc_memories) != 5:
        raise RuntimeError("T450 OUT32 accumulator memory 数量/深度/位宽/端口错误")
    if len(relation_k) != 1 or len(relation_gate) != 5:
        raise RuntimeError("T450 relation memory 数量/深度/位宽/端口错误")
    if len(weight_memories) != 1:
        raise RuntimeError("T450 OUT32 weight row memory 不是 32x256 1R1W")
    if (OUT / "tcfm5_backpressure_iverilog.log").exists():
        raise RuntimeError("T450 结果目录仍包含旧命名 backpressure 日志")
    report = {
        "schema": "local5_projection_t450_synthetic_shell_v1",
        "status": "PASS",
        "evidence": "[rtl] synthetic T450; not checkpoint-bound or profile",
        "geometry": {
            "height": 15,
            "width": 15,
            "time_planes": 2,
            "descriptors": 450,
            "source_id_width": 9,
            "out_dim": 2,
            "production_width_smoke_out_dim": 32,
        },
        "oracle": {
            "seed": hex(seed),
            "inputs": inputs,
            "acc32_outputs": acc32,
            "product_terms": oracle_terms,
            "destination_updates": oracle_updates,
            "acc32_mismatch": 0,
        },
        "rtl": {
            "tcfm5_cycles": tcfm[1],
            "linear5_cycles": linear[1],
            "tcfm5_relation_stalls": tcfm[5],
            "linear5_relation_stalls": linear[5],
            "linear5_over_tcfm5": linear[1] / tcfm[1],
            "backpressure_cycles": bp_iverilog[1],
            "backpressure_issue_stalls": bp_iverilog[7],
            "backpressure_term_hits": bp_iverilog[8],
            "lfsr_stall_seeds": sorted(lfsr_runs),
            "directed_stall_modes": sorted(directed_runs),
            "additional_data_seeds": sorted(data_runs),
            "out32_cycles": out32[1],
            "out32_acc32_outputs": out32_generation[3],
            "out32_columns_distinguishable": True,
            "all_seed_runs_cross_simulator": True,
            "icarus_verilator_equal": True,
        },
        "yosys_proxy": {
            "cells": module["num_cells"],
            "wire_bits": module["num_wire_bits"],
            "mem_v2": cell_types.get("$mem_v2", 0),
            "memory_contract": {
                "total_mem_v2": len(memories),
                "acc_90x1024_1r1w": len(acc_memories),
                "relation_k_45x32_1r1w": len(relation_k),
                "relation_gate_45x10_1r1w": len(relation_gate),
                "weight_row_32x256_1r1w": len(weight_memories),
            },
        },
        "reproducibility": {
            "oracle_hashes": len(oracle_hashes),
            "source_hashes": len(source_hashes),
            "sha256_self_check": True,
        },
        "pending": [
            "final checkpoint theta/Q7/Q1.7/invalid-mask freeze",
            "checkpoint and payload SHA binding",
            "real full-resolution T450 trace",
            "Direct and TCFM-5 same-checkpoint Acc32 miter",
            "variable SRAM latency and multi-window p95/p99",
            "DC/STA/SAIF/PTPX",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reduction = (linear[1] - tcfm[1]) / linear[1] * 100
    lines = [
        "# Local5 当前单顶层合成 T450 部署壳回归",
        "",
        "## 结论",
        "",
        "当前 `score/Shiftmax5 -> relation transpose -> source-major term -> "
        "TCFM-5 -> Acc32` 单顶层已在 `H15×W15×T2=450` 几何下跑通。"
        "本工作包只证明控制、地址宽度、存储深度、close/drain 和反压在 T450 下"
        "可工作，不是最终 checkpoint-bound 证据。",
        "",
        "- `[rtl]` Python 独立 oracle：450 输入、900 个 Acc32、"
        f"{oracle_terms} term、{oracle_updates} update；",
        "- `[rtl]` TCFM-5 Icarus 与 Verilator/SVA 完整计数一致，"
        "逐 Acc32 零失配；",
        "- `[rtl]` 固定种子反压在 Icarus/Verilator 下完整计数一致，"
        f"命中 {bp_iverilog[8]} 个可发 term 边界；",
        "- `[rtl]` 等 5-bank Linear-5 真实 RTL 使用相同输入和工作量，"
        "逐 Acc32 零失配；",
        "- `[模型]` Yosys 参数化 `check -assert` 通过；结构统计不是 ASIC PPA。",
        f"- `[rtl]` 8 个 oracle 文件和 {len(source_hashes)} 个 RTL/验证源文件的 "
        "SHA-256 自校验通过。",
        "- `[rtl]` OUT_DIM=32 生产宽度 Icarus/Verilator-SVA smoke 完成 14400 个 "
        "Acc32，32 个输出列逐列可区分且失配为 0；"
        "三组 Q/K 数据种子、三组 LFSR 反压、首 term 全停、长 burst 和最后 term "
        "定向停顿均跨仿真器通过。",
        "",
        "## 周期与工作量",
        "",
        "| 实现 | tile 周期 | relation stall | term | update |",
        "|---|---:|---:|---:|---:|",
        f"| TCFM-5 | {tcfm[1]} | {tcfm[5]} | {tcfm[3]} | {tcfm[4]} |",
        f"| Linear-5 | {linear[1]} | {linear[5]} | {linear[3]} | {linear[4]} |",
        f"| TCFM-5 + 固定种子反压 | {bp_iverilog[1]} | {bp_iverilog[5]} | "
        f"{bp_iverilog[3]} | {bp_iverilog[4]} |",
        "",
        f"在该合成 T450 流上，TCFM-5 相对等 5-bank Linear-5 减少 "
        f"{reduction:.2f}% tile 周期。该数字是 `[rtl]` 合成向量结果，不能外推"
        "到真实 full-resolution workload。",
        "",
        "## Yosys 结构代理",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| cells | {module['num_cells']} |",
        f"| wire bits | {module['num_wire_bits']} |",
        f"| `$mem_v2` | {cell_types.get('$mem_v2', 0)} |",
        f"| Acc memory `90x1024 1R1W` | {len(acc_memories)} |",
        f"| Relation K `45x32 1R1W` | {len(relation_k)} |",
        f"| Relation gate `45x10 1R1W` | {len(relation_gate)} |",
        f"| Weight row `32x256 1R1W` | {len(weight_memories)} |",
        "",
        "以上仅为 `[模型]` 开放结构统计。没有目标库、目标 SRAM、时钟约束、"
        "SAIF 或布线寄生，因此不得称为 ASIC 面积、频率、功耗或 PPA。",
        "",
        "## 证据边界与下一步",
        "",
        "该包使用固定种子合成 Q/K 和运行时 invalid-candidate mask，没有绑定"
        "训练 checkpoint、theta-folded 权重或真实 profile。最终 rank-1 释放后必须：",
        "",
        "1. 冻结 theta、Q7、Q1.7 和 invalid-mask 合同；",
        "2. 绑定 checkpoint/config/payload/trace SHA；",
        "3. 用真实 full-resolution T450 trace 替换本合成向量；",
        "4. 在相同 checkpoint 和权重下完成 Direct/TCFM-5 逐 Acc32 miter；",
        "5. 再加入可变 SRAM 延迟、多窗口 mean/p95/p99 和目标工艺 PPA。",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
