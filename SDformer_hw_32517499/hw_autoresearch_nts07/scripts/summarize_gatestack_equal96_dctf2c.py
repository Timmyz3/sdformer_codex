#!/usr/bin/env python3
"""汇总四种96-lane term边界结构的真实周期与开放逻辑代理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(
    equal_cycles: dict,
    equal_mapping: dict,
    two_context_trace: dict,
    two_context_mapping: dict,
) -> dict:
    cycle_summary = equal_cycles["summary"]["cycles"]
    mapping_rows = {row["name"]: row for row in equal_mapping["rows"]}
    two_row = two_context_mapping["mapping"]
    two_stage = {
        item["stage"]: item["cycles"] for item in two_context_trace["Icarus"]
    }
    definitions = (
        ("central96", "Central96", "central96_term", 1),
        ("independent32x3", "3xIndependent32", "independent32x3_term", 3),
        ("dctf96", "DCTF96-1C", "dctf96_term", 1),
    )
    rows = []
    for key, label, mapping_key, clients in definitions:
        mapping = mapping_rows[mapping_key]
        stage_cycles = [item["cycles"][key] for item in equal_cycles["rows"]]
        rows.append({
            "key": key,
            "label": label,
            "term_clients": clients,
            "stage_cycles": stage_cycles,
            "total_cycles": cycle_summary[key],
            "logic_area": mapping["logic_area"],
            "cells": mapping["cells"],
            "mem_v2": mapping["mem_v2"],
            "extra_state_bits": None,
        })
    rows.append({
        "key": "dctf96_2c",
        "label": "DCTF96-2C",
        "term_clients": 1,
        "stage_cycles": [two_stage[stage] for stage in range(4)],
        "total_cycles": two_context_trace["总周期"],
        "logic_area": two_row["logic_area"],
        "cells": two_row["cells"],
        "mem_v2": two_row["mem_v2"],
        "extra_state_bits": two_row["total_architectural_state_bits"],
    })

    central = rows[0]
    for row in rows:
        row["speedup_vs_central"] = (
            central["total_cycles"] / row["total_cycles"]
        )
        row["logic_area_reduction_vs_central"] = (
            1.0 - row["logic_area"] / central["logic_area"]
        )
        row["area_normalized_throughput_vs_central"] = (
            central["total_cycles"] * central["logic_area"]
        ) / (row["total_cycles"] * row["logic_area"])

    two = rows[-1]
    one = rows[-2]
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "boundary": "H67 sample0/window0，term/event到projection final",
        "product_lanes": 96,
        "rows": rows,
        "dctf2c": {
            "speedup_vs_dctf1c": one["total_cycles"] / two["total_cycles"],
            "logic_area_delta_vs_dctf1c": (
                two["logic_area"] / one["logic_area"] - 1.0
            ),
            "s3_cycle_reduction_vs_dctf1c": (
                1.0 - two["stage_cycles"][3] / one["stage_cycles"][3]
            ),
            "architectural_state_bits": two["extra_state_bits"],
        },
        "workload": {
            "logical_terms": equal_cycles["summary"]["total_logical_terms"],
            "s3_term_count": equal_cycles["rows"][3]["logical_terms"],
            "s3_destinations": equal_cycles["rows"][3]["destinations"],
            "s3_destinations_per_term":
                equal_cycles["rows"][3]["destinations_per_term"],
        },
        "limits": [
            "只覆盖H67 sample0/window0，不代表全数据集分布",
            "固定一拍行为weight/bias存储，final全ready",
            "开放Nangate45映射无SDC、STA、SAIF、DC或P&R",
            "$mem_v2不计入logic area，mem_v2个数不能换算SRAM面积",
            "Central的宽Acc仍需在物理签核中统一拆为六个32-lane宏",
            "面积归一吞吐不是EDP，也不是目标工艺PPA",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 四种96-Lane架构等边界总表",
        "",
        "全部候选使用同一H67 sample0/window0 term/event输入、96个INT8 product lane和完整projection final边界。",
        "",
        "| 结构 | S0 | S1 | S2 | S3 | 总周期 | 相对Central加速 | 逻辑面积值 | cells | `$mem_v2` | 已分账状态位 | 面积归一吞吐 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        stages = " | ".join(str(value) for value in row["stage_cycles"])
        state_bits = ("-" if row["extra_state_bits"] is None else
                      str(row["extra_state_bits"]))
        lines.append(
            f"| {row['label']} | {stages} | {row['total_cycles']} | "
            f"{row['speedup_vs_central']:.3f}x | {row['logic_area']:.3f} | "
            f"{row['cells']} | {row['mem_v2']} | {state_bits} | "
            f"{row['area_normalized_throughput_vs_central']:.3f}x |"
        )
    dctf = report["dctf2c"]
    workload = report["workload"]
    lines += [
        "",
        "## 结果解释",
        "",
        f"- DCTF96-2C相对1C总周期提速{dctf['speedup_vs_dctf1c']:.3f}x，S3周期降低{dctf['s3_cycle_reduction_vs_dctf1c'] * 100:.3f}%；",
        f"- DCTF96-2C相对1C开放逻辑面积变化{dctf['logic_area_delta_vs_dctf1c'] * 100:.3f}%，但另有{dctf['architectural_state_bits']} bit架构状态，不能用逻辑面积下降掩盖memory代价；",
        f"- S3含{workload['s3_term_count']}个term、{workload['s3_destinations']}个destination，平均每term {workload['s3_destinations_per_term']:.3f}个destination，是双上下文重叠收益的主要来源；",
        "- 3xIndependent32周期接近Central，但三套term client与更大开放逻辑面积使其面积归一吞吐最低；",
        "",
        "## 证据边界",
        "",
        *[f"- {item}；" for item in report["limits"]],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-cycles", type=Path, required=True)
    parser.add_argument("--equal-mapping", type=Path, required=True)
    parser.add_argument("--two-context-trace", type=Path, required=True)
    parser.add_argument("--two-context-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        json.loads(path.read_text(encoding="utf-8")) for path in (
            args.equal_cycles,
            args.equal_mapping,
            args.two_context_trace,
            args.two_context_mapping,
        )
    ]
    report = build_report(*inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
