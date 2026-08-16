#!/usr/bin/env python3
"""汇总Motion真实四stage的PPDI/IBF完整投影RTL消融。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS_RE = re.compile(
    r"^PASS DCTF96 REAL TRACE stage=S(?P<stage>\d+) "
    r"heads=(?P<heads>\d+) cycles=(?P<cycles>\d+) "
    r"terms=(?P<terms>\d+) physical_weight_req=(?P<weight>\d+) "
    r"bias_req=(?P<bias>\d+) final_checks=(?P<final>\d+)$",
    re.MULTILINE,
)
MODES = ("scalar_rmw", "ppdi_rmw", "scalar_ibf", "ppdi_ibf")


def parse_log(path: Path) -> dict[str, int]:
    match = PASS_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{path}缺少真实trace PASS计数")
    return {key: int(value) for key, value in match.groupdict().items()}


def build_report(log_root: Path, mapping_report: Path | None = None) -> dict:
    rows = {
        mode: [
            parse_log(log_root / mode / f"icarus_s{stage}.log")
            for stage in range(4)
        ]
        for mode in MODES
    }
    totals = {
        mode: sum(row["cycles"] for row in stage_rows)
        for mode, stage_rows in rows.items()
    }
    baseline = totals["scalar_rmw"]
    bias_requests = {
        mode: sum(row["bias"] for row in stage_rows)
        for mode, stage_rows in rows.items()
    }
    verilator = parse_log(log_root / "ppdi_ibf" / "verilator_s0.log")
    if verilator != rows["ppdi_ibf"][0]:
        raise ValueError("PPDI+IBF S0的Icarus与Verilator计数不一致")
    for stage in range(4):
        reference = rows["scalar_rmw"][stage]
        for mode in MODES[1:]:
            candidate = rows[mode][stage]
            for field in ("stage", "heads", "terms", "weight", "final"):
                if candidate[field] != reference[field]:
                    raise ValueError(f"{mode} S{stage}字段{field}与基线不一致")
    report = {
        "schema_version": 1,
        "status": "RTL_REAL_TRACE_BIT_EXACT",
        "scope": "H67 Motion sample0/window0，S0-S3完整projection顶层",
        "stages": rows,
        "total_cycles": totals,
        "speedup_vs_scalar_rmw": {
            mode: baseline / cycles for mode, cycles in totals.items()
        },
        "total_bias_requests": bias_requests,
        "bias_request_reduction": {
            mode: 1.0 - requests / bias_requests["scalar_rmw"]
            for mode, requests in bias_requests.items()
        },
        "verilator_ppdi_ibf_s0": verilator,
        "limits": [
            "只有Motion sample0/window0，不代表多样本mean/p95/p99",
            "weight与bias均为固定一拍存储模型，final始终ready",
            "周期不包含attention前端、ATLIF、skip和片外IO",
            "动态SVA不是formal，开放库映射不是DC PPA",
        ],
    }
    if mapping_report is not None:
        mapping = json.loads(mapping_report.read_text(encoding="utf-8"))
        area_ratios = mapping["open_logic_mapping"]["area_ratios"]
        report["open_logic_cross_evidence"] = {
            "area_ratios": area_ratios,
            "area_normalized_throughput": {
                mode: report["speedup_vs_scalar_rmw"][mode] / area_ratios[mode]
                for mode in MODES
            },
            "warning": "真实T162周期与开放库逻辑面积的跨证据代理，不是ASIC PPA",
        }
    return report


def render_markdown(report: dict) -> str:
    rows = report["stages"]
    totals = report["total_cycles"]
    speedups = report["speedup_vs_scalar_rmw"]
    bias = report["total_bias_requests"]
    bias_reduction = report["bias_request_reduction"]
    labels = {
        "scalar_rmw": "标量+RMW",
        "ppdi_rmw": "PPDI+RMW",
        "scalar_ibf": "标量+IBF",
        "ppdi_ibf": "PPDI+IBF",
    }
    lines = [
        "# PPDI+IBF Motion真实四阶段RTL回放",
        "",
        f"状态：{report['status']}。四种模式使用同一组真实term、INT8权重、INT32偏置和逐元素golden。",
        "",
        "## 周期消融",
        "",
        "| 配置 | S0 | S1 | S2 | S3 | 总周期 | 相对标量RMW |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        stage_cycles = [row["cycles"] for row in rows[mode]]
        lines.append(
            f"| {labels[mode]} | {stage_cycles[0]} | {stage_cycles[1]} | "
            f"{stage_cycles[2]} | {stage_cycles[3]} | {totals[mode]} | "
            f"{speedups[mode]:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 偏置请求",
            "",
            "| 配置 | 四阶段偏置请求 | 降低 |",
            "|---|---:|---:|",
        ]
    )
    for mode in MODES:
        lines.append(
            f"| {labels[mode]} | {bias[mode]} | "
            f"{bias_reduction[mode] * 100:.3f}% |"
        )
    lines.extend(
        [
            "",
            "所有模式的term数、物理weight请求和233280个INT32 final元素一致；组合模式S0的Icarus与Verilator动态SVA计数一致。",
            "",
            "## 限制",
            "",
            *[f"- {item}" for item in report["limits"]],
            "",
        ]
    )
    if "open_logic_cross_evidence" in report:
        cross = report["open_logic_cross_evidence"]
        lines.extend(
            [
                "",
                "## 开放逻辑跨证据代理",
                "",
                "| 配置 | 面积倍率 | 真实trace面积归一吞吐 |",
                "|---|---:|---:|",
            ]
        )
        for mode in MODES:
            lines.append(
                f"| {labels[mode]} | {cross['area_ratios'][mode]:.3f}x | "
                f"{cross['area_normalized_throughput'][mode]:.3f}x |"
            )
        lines.extend(["", cross["warning"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping-report", type=Path)
    args = parser.parse_args()
    report = build_report(args.log_root, args.mapping_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
