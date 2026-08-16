#!/usr/bin/env python3
"""从真实 supertile RTL 日志统计逻辑事务与物理 32-lane bank 流量。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT\s+(.*)$")
WIDTHS = (32, 64, 96, 128)
STAGES = (0, 1, 2, 3)
WEIGHT_W = 8
BIAS_W = 32
PHYSICAL_BANK_LANES = 32


def parse_result_line(text: str) -> dict[str, str]:
    for line in text.splitlines():
        match = RESULT_RE.match(line)
        if not match:
            continue
        fields: dict[str, str] = {}
        for item in match.group(1).split():
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        return fields
    raise ValueError("日志中没有 RESULT 行")


def load_rows(sweep_dir: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for width in WIDTHS:
        for stage in STAGES:
            path = sweep_dir / f"w{width}_s{stage}" / "iverilog.log"
            fields = parse_result_line(path.read_text(encoding="utf-8"))
            if fields.get("status") != "PASS":
                raise ValueError(f"{path} 不是 PASS")
            rows.append(
                {
                    "width": width,
                    "stage": stage,
                    "terms": int(fields["projection_terms"]),
                    "bias_commits": int(fields["bias"]),
                    "total_cycles": int(fields["total_cycles"]),
                    "mismatches": int(fields["mismatches"]),
                }
            )
    return rows


def summarize(rows: list[dict[str, int | str]]) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    baseline_weight_bits = 0
    baseline_bias_bits = 0
    baseline_logical_weight_requests = 0
    for width in WIDTHS:
        selected = [row for row in rows if row["width"] == width]
        terms = sum(int(row["terms"]) for row in selected)
        bias_commits = sum(int(row["bias_commits"]) for row in selected)
        lane_banks = math.ceil(width / PHYSICAL_BANK_LANES)
        weight_bits = terms * width * WEIGHT_W
        bias_bits = bias_commits * width * BIAS_W
        if width == 32:
            baseline_weight_bits = weight_bits
            baseline_bias_bits = bias_bits
            baseline_logical_weight_requests = terms
        summaries.append(
            {
                "width": width,
                "lane_banks_per_request": lane_banks,
                "logical_weight_requests": terms,
                "physical_weight_bank_accesses": terms * lane_banks,
                "weight_payload_bits": weight_bits,
                "bias_logical_responses": bias_commits,
                "physical_bias_bank_accesses": bias_commits * lane_banks,
                "bias_payload_bits": bias_bits,
                "total_cycles": sum(int(row["total_cycles"]) for row in selected),
            }
        )
    for item in summaries:
        item["logical_weight_request_reduction"] = (
            baseline_logical_weight_requests / int(item["logical_weight_requests"])
            if int(item["logical_weight_requests"])
            else 0.0
        )
        item["weight_padding_overhead_pct"] = (
            (int(item["weight_payload_bits"]) / baseline_weight_bits - 1.0) * 100.0
            if baseline_weight_bits
            else 0.0
        )
        item["bias_padding_overhead_pct"] = (
            (int(item["bias_payload_bits"]) / baseline_bias_bits - 1.0) * 100.0
            if baseline_bias_bits
            else 0.0
        )
    return summaries


def render_markdown(summaries: list[dict[str, float | int]]) -> str:
    lines = [
        "# HATF Supertile 权重与 Bias 流量分账",
        "",
        "## 结论",
        "",
        "HATF96 将逻辑 weight request 数降低为 32-lane 路径的三分之一，但每次并行访问三个 32-lane bank。四个 stage 的 physical bank access 和 payload bit 与 32-lane 基线一致，说明收益来自 replay/decoder/term 控制复用与并行时延，而不是删除稠密权重读取。",
        "",
        "| 宽度 | 每请求32-lane bank | logical weight req | physical weight bank access | weight payload bit | weight padding | bias payload bit | bias padding | 总周期 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['width']} | {row['lane_banks_per_request']} | "
            f"{row['logical_weight_requests']} | "
            f"{row['physical_weight_bank_accesses']} | "
            f"{row['weight_payload_bits']} | "
            f"{row['weight_padding_overhead_pct']:.2f}% | "
            f"{row['bias_payload_bits']} | "
            f"{row['bias_padding_overhead_pct']:.2f}% | "
            f"{row['total_cycles']} |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- 统计输入是 sample0/B0/window0 的 S0-S3 真实 RTL RESULT 行。",
            "- physical bank 固定为 32 lane；width=96 对应每逻辑请求并行访问 3 个 bank。",
            "- weight payload bit = projection_terms × width × 8。",
            "- bias payload bit = bias_commits × width × 32。",
            "- 这里没有 SRAM 电容、地址译码、时钟树或布线数据，不能由 payload bit 直接推出能量。",
            "",
            "## 投稿使用边界",
            "",
            "可以声明 HATF96 在当前 trace 上将逻辑请求和 term 控制事务减少 3 倍，同时保持 32-lane physical bank access 与 payload bit 不变。不能声明权重 SRAM 能量减少 3 倍；该结论必须等待 SRAM macro 与 mapped SAIF。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("build_hitflow/gatestack_projection_supertile_sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hatf_supertile_traffic_20260720"),
    )
    args = parser.parse_args()
    rows = load_rows(args.sweep_dir)
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "source": "[rtl] sample0/B0/window0 S0-S3",
        "physical_bank_lanes": PHYSICAL_BANK_LANES,
        "rows": summaries,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(summaries), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
