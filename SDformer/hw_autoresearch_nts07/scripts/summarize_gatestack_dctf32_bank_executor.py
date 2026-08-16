#!/usr/bin/env python3
"""汇总 DCTF32 bank executor 与 product-engine 叶模块的开放库映射。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


AREA_RE = re.compile(r"Chip area for module .*: ([0-9]+(?:\.[0-9]+)?)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
MEM_RE = re.compile(r"\$mem_v2\s+([0-9]+)")


def last_match(
    pattern: re.Pattern[str], text: str, default: int | None = None
) -> int | float:
    values = pattern.findall(text)
    if not values:
        if default is not None:
            return default
        raise RuntimeError(f"无法匹配 {pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def build_report(mapping_dir: Path) -> dict:
    rows = []
    for name, label, boundary in (
        (
            "product_engine_32",
            "32-lane product-engine叶模块（36-bit tag）",
            "单term锁存、一次权重请求与响应匹配、32-lane乘积寄存、输出握手及五组可观察计数器",
        ),
        (
            "executor_32",
            "DCTF32 bank executor",
            "包含product engine，并增加command协议、epoch隔离、term内乘积驻留复用、奇偶Acc路由与完成控制",
        ),
    ):
        text = (mapping_dir / f"{name}.log").read_text()
        rows.append(
            {
                "name": name,
                "label": label,
                "out_tile": 32,
                "functional_boundary": boundary,
                "logic_area": last_match(AREA_RE, text),
                "cells": last_match(CELLS_RE, text),
                "mem_v2": last_match(MEM_RE, text, 0),
            }
        )

    leaf, executor = rows
    comparison = {
        "logic_area_delta": executor["logic_area"] - leaf["logic_area"],
        "logic_area_ratio": executor["logic_area"] / leaf["logic_area"],
        "logic_area_increase": executor["logic_area"] / leaf["logic_area"] - 1.0,
        "cell_delta": executor["cells"] - leaf["cells"],
        "cell_ratio": executor["cells"] / leaf["cells"],
        "cell_increase": executor["cells"] / leaf["cells"] - 1.0,
        "interpretation": (
            "差值混合了command协议、epoch迟到响应隔离、乘积驻留控制、"
            "奇偶Acc路由、完成控制和顶层可观察输出优化；功能边界不同，"
            "不是纯路由面积"
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "开放库无约束logic proxy",
        "library": "NangateOpenCellLibrary_typical.lib",
        "same_source_settings": {
            "rtl_sources": [
                "rtl_hitflow/gatestack_decoupled_product_engine.sv",
                "rtl_hitflow/gatestack_dctf32_bank_executor.sv",
            ],
            "out_tile": 32,
            "aligned_engine_tag_w": 36,
            "mapping_flow": "相同Yosys流程、相同Liberty、无时序约束",
        },
        "rtl_verification": {
            "icarus": "PASS",
            "verilator_dynamic_sva": "PASS",
            "yosys": "PASS",
            "erie": "PASS，0 error / 0 warning",
            "aba_test": (
                "旧epoch响应先到并被ready/drop，不产生Acc更新或term_done；"
                "随后新epoch响应完成，stale_rsp=1"
            ),
        },
        "epoch_constraint": {
            "default_epoch_w": 4,
            "epoch_states": 16,
            "requirement": (
                "旧响应必须在epoch计数回绕到同值前全部排空；否则仅按epoch相等"
                "判断会出现ABA别名，需扩大epoch或增加未决请求生命周期约束"
            ),
        },
        "rows": rows,
        "comparison": comparison,
        "limits": [
            "无SDC、STA、SAIF、SRAM macro和DC",
            "未估计布局布线、时钟、互连、功耗或存储宏",
            "不得称为ASIC PPA或签核结果",
            "两个顶层功能边界不同，差值不是纯路由面积",
        ],
    }


def render_markdown(report: dict) -> str:
    leaf, executor = report["rows"]
    comparison = report["comparison"]
    epoch = report["epoch_constraint"]
    lines = [
        "# DCTF32 Bank Executor 开放逻辑映射代理",
        "",
        "证据等级仅为**开放库无约束logic proxy**。两次映射读取同一组RTL，product engine均固定为`OUT_TILE=32、TAG_W=36`，并使用同一Nangate45 Liberty和同一Yosys流程。",
        "",
        "| 顶层 | 功能边界 | 库面积值 | 标准单元数 | `$mem_v2` |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['functional_boundary']} | "
            f"{row['logic_area']:.3f} | {row['cells']} | {row['mem_v2']} |"
        )
    lines += [
        "",
        "## 同源差值",
        "",
        f"Executor相对叶模块增加库面积值`{comparison['logic_area_delta']:.3f}`（`{comparison['logic_area_increase'] * 100:.3f}%`），增加`{comparison['cell_delta']}`个标准单元（`{comparison['cell_increase'] * 100:.3f}%`）。",
        "",
        "Executor包含叶模块，还包含command身份与顺序协议、epoch迟到响应隔离、整条term的乘积驻留复用、奇偶Acc端口路由以及term完成控制。叶模块顶层暴露五组计数器；嵌入executor后对应内部计数器未使用并可被优化，所以这里得到的是两个不同功能边界的净差值，不是wrapper毛开销、纯路由面积或物理互连开销。",
        "",
        "## 已有RTL验证事实",
        "",
        "- Icarus：PASS；",
        "- Verilator动态SVA：PASS；",
        "- Yosys：PASS；",
        "- Erie：PASS，RTL与TB均为`0 error / 0 warning`；",
        "- ABA用例先发出旧epoch请求并flush，再以相同SRAM身份发出新epoch请求；旧epoch响应先到时被ready/drop，不产生Acc更新或`term_done`，随后新epoch响应正常完成，最终`stale_rsp=1`。",
        "",
        "## Epoch有限回绕约束",
        "",
        f"RTL默认`EPOCH_W={epoch['default_epoch_w']}`，只有`{epoch['epoch_states']}`个epoch状态。正确性要求旧响应必须在epoch计数回绕到同值前全部排空；否则旧响应可能与新请求发生ABA别名。系统集成必须限制迟到响应生命周期和连续flush次数，或扩大epoch并增加未决请求跟踪。",
        "",
        "## 证据边界",
        "",
        "本结果没有SDC、STA、SAIF、SRAM macro或DC，也没有布局布线、时钟树、互连和功耗分析。库面积值只用于同源逻辑结构筛选，不得称为ASIC PPA或签核结果。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.mapping_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
