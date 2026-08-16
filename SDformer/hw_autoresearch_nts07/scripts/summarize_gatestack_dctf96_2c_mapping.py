#!/usr/bin/env python3
"""汇总DCTF96-2C开放逻辑映射与等边界基线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_gatestack_equal96_term_boundary import parse_mapping


def dctf2c_state_bits() -> dict[str, int]:
    """返回当前TOKENS=162、TOKEN_ID_W=8配置的架构状态位分账。"""
    token_bits = 2 * 162 * 8
    seen_bits = 2 * 162
    # 每个context包含valid/complete以及term元数据和接收计数。
    context_metadata_bits = 2 * (
        2 + 32 + 9 + 5 + 8 + 13 + 1 + 10 + 8 + 8
    )
    # fill状态/所有权指针5b、emit index 8b、command sequence 16b、error 1b。
    shared_control_bits = 5 + 8 + 16 + 1
    return {
        "context_token_bits": token_bits,
        "context_seen_bits": seen_bits,
        "context_metadata_bits": context_metadata_bits,
        "shared_control_bits": shared_control_bits,
        "total_architectural_state_bits": (
            token_bits + seen_bits + context_metadata_bits + shared_control_bits
        ),
    }


def build_report(output_dir: Path, baseline_path: Path) -> dict:
    row = parse_mapping(
        output_dir / "mapping/dctf96_2c.log",
        output_dir / "mapping/dctf96_2c_mapped.v",
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_rows = {item["name"]: item for item in baseline["rows"]}
    central = base_rows["central96_term"]
    dctf_1c = base_rows["dctf96_term"]
    cycles = {"central96": 59853, "dctf96_1c": 62264, "dctf96_2c": 53910}
    row.update({
        "name": "dctf96_2c",
        "adapter_contexts": 2,
        **dctf2c_state_bits(),
    })
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "mapping": row,
        "baselines": {"central96": central, "dctf96_1c": dctf_1c},
        "cycles": cycles,
        "dctf2c_area_delta_vs_1c_pct":
            (row["logic_area"] / dctf_1c["logic_area"] - 1.0) * 100.0,
        "dctf2c_area_reduction_vs_central_pct":
            (1.0 - row["logic_area"] / central["logic_area"]) * 100.0,
        "dctf2c_area_normalized_throughput_vs_central":
            (cycles["central96"] * central["logic_area"]) /
            (cycles["dctf96_2c"] * row["logic_area"]),
        "limits": [
            "无SDC、STA、SAIF、DC、P&R或SRAM宏",
            "$mem_v2不计入logic area，双context存储位单独报告",
            "面积归一吞吐组合开放逻辑面积与RTL周期，不是EDP",
            "只使用H67 sample0/window0周期",
        ],
    }


def render_markdown(report: dict) -> str:
    row = report["mapping"]
    central = report["baselines"]["central96"]
    one = report["baselines"]["dctf96_1c"]
    storage = row["total_architectural_state_bits"]
    return "\n".join([
        "# DCTF96-2C开放逻辑映射代理",
        "",
        "本报告比较同一term/event输入边界下的Central96、DCTF96单上下文与DCTF96-2C。所有面积均为Nangate45无约束逻辑代理。",
        "",
        "| 结构 | 周期 | 逻辑面积值 | cells | `$mem_v2` |",
        "|---|---:|---:|---:|---:|",
        f"| Central96 | 59853 | {central['logic_area']:.3f} | {central['cells']} | {central['mem_v2']} |",
        f"| DCTF96-1C | 62264 | {one['logic_area']:.3f} | {one['cells']} | {one['mem_v2']} |",
        f"| DCTF96-2C | 53910 | {row['logic_area']:.3f} | {row['cells']} | {row['mem_v2']} |",
        "",
        "## 有限结论",
        "",
        f"- 2C相对1C开放逻辑面积变化{report['dctf2c_area_delta_vs_1c_pct']:.3f}%；",
        f"- 2C相对Central开放逻辑面积降低{report['dctf2c_area_reduction_vs_central_pct']:.3f}%；",
        f"- 以Central为1，2C开放逻辑面积归一吞吐为{report['dctf2c_area_normalized_throughput_vs_central']:.3f}x；",
        f"- 双context架构状态合同为{storage} bit，其中token数组{row['context_token_bits']} bit、seen bitmap {row['context_seen_bits']} bit、context元数据{row['context_metadata_bits']} bit、共享控制{row['shared_control_bits']} bit；这些位不能因开放映射未计memory而忽略；",
        "",
        "## 证据边界",
        "",
        *[f"- {item}；" for item in report["limits"]],
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.output_dir, args.baseline)
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
