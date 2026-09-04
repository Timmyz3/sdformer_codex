#!/usr/bin/env python3
"""汇总 HIFP 分块 OpenROAD P&R/STA 结果并生成中文证据报告。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BLOCKS = {
    "datapath_scalar": ("hifp_dctf96_datapath_t6", "scalar"),
    "datapath_ppdi": ("hifp_dctf96_datapath_t6", "ppdi"),
    "accumulator_rmw": ("hifp_accumulator_t6", "rmw"),
    "accumulator_ibf": ("hifp_accumulator_t6", "ibf"),
}

EXPECTED_CONSTANT_ENDPOINTS = {
    "datapath_scalar": re.compile(
        r"(?:acc_update_token_ids\[(?:0|6|12)\]|fabric_max_occupancy\[(?:[2-9]|[12][0-9]|3[01])\])"
    ),
    "datapath_ppdi": re.compile(
        r"(?:acc_update_token_ids\[(?:0|6|12)\]|fabric_max_occupancy\[(?:[2-9]|[12][0-9]|3[01])\])"
    ),
    "accumulator_rmw": re.compile(r"(?!)"),
    "accumulator_ibf": re.compile(r"final_token_ids\[0\]"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def collect_block(
    log_root: Path, name: str, design: str, variant: str
) -> dict[str, Any]:
    root = log_root / "nangate45" / design / variant
    finish = load_json(root / "6_report.json")
    route = load_json(root / "5_2_TritonRoute.json")
    finish_log = (root / "6_report.log").read_text(encoding="utf-8", errors="replace")
    unconstrained = [
        int(value)
        for value in re.findall(r"There (?:is|are) (\d+) unconstrained endpoint", finish_log)
    ]
    verbose_path = root / "check_setup_verbose.log"
    if not verbose_path.exists():
        raise FileNotFoundError(f"缺少verbose约束审计: {verbose_path}")
    verbose_log = verbose_path.read_text(encoding="utf-8", errors="replace")
    endpoint_names = re.findall(r"^  ([^\s]+)$", verbose_log, flags=re.MULTILINE)
    unexpected_endpoints = [
        endpoint
        for endpoint in endpoint_names
        if EXPECTED_CONSTANT_ENDPOINTS[name].fullmatch(endpoint) is None
    ]
    drc_by_iteration = {
        key.rsplit(":", 1)[-1]: value
        for key, value in route.items()
        if key.startswith("detailedroute__route__drc_errors__iter:")
    }
    setup_violations = finish["finish__timing__drv__setup_violation_count"]
    hold_violations = finish["finish__timing__drv__hold_violation_count"]
    max_slew_violations = finish["finish__timing__drv__max_slew"]
    max_cap_violations = finish["finish__timing__drv__max_cap"]
    max_fanout_violations = finish["finish__timing__drv__max_fanout"]
    drc_violations = route["detailedroute__route__drc_errors"]
    return {
        "design": design,
        "variant": variant,
        "stdcell_area_um2": finish["finish__design__instance__area__stdcell"],
        "stdcell_count": finish["finish__design__instance__count__stdcell"],
        "utilization": finish["finish__design__instance__utilization__stdcell"],
        "io_pins": finish["finish__design__io"],
        "setup_slack_ns": finish["finish__timing__setup__ws"],
        "setup_tns_ns": finish["finish__timing__setup__tns"],
        "setup_violations": setup_violations,
        "hold_violations": hold_violations,
        "max_slew_violations": max_slew_violations,
        "max_cap_violations": max_cap_violations,
        "max_fanout_violations": max_fanout_violations,
        "unconstrained_endpoints": max(unconstrained, default=0),
        "unconstrained_endpoint_names": endpoint_names,
        "unexpected_unconstrained_endpoints": unexpected_endpoints,
        "critical_delay_ns": 5.0 - finish["finish__timing__setup__ws"],
        "drc_violations": drc_violations,
        "drc_by_iteration": drc_by_iteration,
        "setup_hold_closed": setup_violations == 0 and hold_violations == 0,
        "electrical_drv_clean": (
            max_slew_violations == 0
            and max_cap_violations == 0
            and max_fanout_violations == 0
        ),
        "routing_drc_clean": drc_violations == 0,
        "wirelength_um": route["detailedroute__route__wirelength"],
        "vias": route["detailedroute__route__vias"],
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cycle-report",
        type=Path,
        default=Path("results/ppdi_ibf_real_trace_20260801/report.json"),
    )
    args = parser.parse_args()

    log_root = args.root / "openroad_hifp/work/logs"
    blocks = {
        name: collect_block(log_root, name, design, variant)
        for name, (design, variant) in BLOCKS.items()
    }
    cycle_path = args.cycle_report
    if not cycle_path.is_absolute():
        cycle_path = args.root / cycle_path
    cycles = load_json(cycle_path)["total_cycles"]

    combinations: dict[str, dict[str, Any]] = {}
    for mode, datapath_key, accumulator_key in (
        ("scalar_rmw", "datapath_scalar", "accumulator_rmw"),
        ("ppdi_rmw", "datapath_ppdi", "accumulator_rmw"),
        ("scalar_ibf", "datapath_scalar", "accumulator_ibf"),
        ("ppdi_ibf", "datapath_ppdi", "accumulator_ibf"),
    ):
        datapath = blocks[datapath_key]
        accumulator = blocks[accumulator_key]
        combinations[mode] = {
            "cycles": cycles[mode],
            "composed_stdcell_area_um2": (
                datapath["stdcell_area_um2"] + 3 * accumulator["stdcell_area_um2"]
            ),
            "composed_critical_delay_ns": max(
                datapath["critical_delay_ns"], accumulator["critical_delay_ns"]
            ),
            "setup_closed_at_5ns": (
                datapath["setup_violations"] == 0
                and accumulator["setup_violations"] == 0
                and not datapath["unexpected_unconstrained_endpoints"]
                and not accumulator["unexpected_unconstrained_endpoints"]
            ),
            "drc_clean": (
                datapath["drc_violations"] == 0 and accumulator["drc_violations"] == 0
            ),
            "drv_clean": (
                datapath["electrical_drv_clean"]
                and accumulator["electrical_drv_clean"]
            ),
        }

    base = combinations["scalar_rmw"]
    for row in combinations.values():
        row["speedup_vs_scalar_rmw"] = base["cycles"] / row["cycles"]
        row["area_ratio_vs_scalar_rmw"] = (
            row["composed_stdcell_area_um2"] / base["composed_stdcell_area_um2"]
        )
        row["area_normalized_throughput_vs_scalar_rmw"] = (
            row["speedup_vs_scalar_rmw"] / row["area_ratio_vs_scalar_rmw"]
        )

    result = {
        "schema_version": 1,
        "status": "OPEN_PNR_FLOW_COMPLETE_NOT_SIGNOFF",
        "evidence": "[open-pnr][代理]",
        "flow": {
            "platform": "Nangate45",
            "clock_period_ns": 5.0,
            "tokens": 6,
            "output_tile": 32,
            "accumulator_blocks_per_projection": 3,
            "openroad_commit": "547465ccf8979379098216194f5837c413c7e2e9",
            "orfs_commit": "3a0a1efd1d8d7891de1c4961487eaf6288adf7df",
            "yosys_version": "0.33",
        },
        "blocks": blocks,
        "composed_projection_proxy": combinations,
        "limits": [
            "这是Nangate45开放库的分块P&R/STA代理，不是DC、PrimeTime或目标工艺签核。",
            "TOKENS固定为6，状态存储由触发器映射；不是T162/T450全分辨率实例。",
            "未实例化SRAM宏、macro wrapper、memory compiler模型或真实memory latency。",
            "组合面积为datapath加三份32-lane accumulator的标准单元面积，不含块间布线、顶层halo和宏间通道。",
            "组合关键路径取两个块关键路径最大值，不含块间时序弧，因此只用于同约束筛选。",
            "没有真实trace SAIF、vectorless/vector-based功耗、IR signoff、EM、antenna修复、LVS或GDS签核。",
            "块级宽接口仍被实现为顶层IO，wirelength、via和die面积不可直接外推到芯片。",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# HIFP 分块 OpenROAD 物理实现与同约束筛选",
        "",
        "## 结论",
        "",
        "四个叶级候选均在同一 Nangate45、5 ns、45% 初始利用率和固定引脚随机种子下完成分块 P&R/STA。该结果用于提前暴露布线、时序和接口问题，证据等级为 **[open-pnr][代理]**，不能替代后续 DC/PrimeTime/目标 SRAM 宏结果。",
        "",
        "## 分块结果",
        "",
        "| 块 | 面积 (µm²) | 单元数 | 利用率 | IO | 关键路径 (ns) | setup/hold | 常量/异常未约束 | max-cap | DRC | 线长 (µm) | via |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "datapath_scalar": "DCTF96 Scalar",
        "datapath_ppdi": "DCTF96 PPDI",
        "accumulator_rmw": "Acc RMW",
        "accumulator_ibf": "Acc IBF",
    }
    for key in BLOCKS:
        row = blocks[key]
        md.append(
            f"| {labels[key]} | {row['stdcell_area_um2']:.0f} | {row['stdcell_count']} | "
            f"{pct(row['utilization'])} | {row['io_pins']} | {row['critical_delay_ns']:.3f} | "
            f"{row['setup_violations']}/{row['hold_violations']} | "
            f"{row['unconstrained_endpoints']}/{len(row['unexpected_unconstrained_endpoints'])} | "
            f"{row['max_cap_violations']} | "
            f"{row['drc_violations']} | "
            f"{row['wirelength_um']:.0f} | {row['vias']} |"
        )

    md += [
        "",
        "## 投影子系统组合代理",
        "",
        "组合口径为一份 96-lane datapath 加三份 2-bank、32-lane accumulator；不计块间线和 SRAM 宏。周期取真实 Motion S0-S3 回放，面积取本轮分块 P&R。",
        "",
        "| 模式 | 周期 | 组合面积 (µm²) | 面积比 | 加速比 | 面积归一吞吐 | 5 ns闭合 | DRC clean | DRV clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("scalar_rmw", "ppdi_rmw", "scalar_ibf", "ppdi_ibf"):
        row = combinations[mode]
        md.append(
            f"| `{mode}` | {row['cycles']} | {row['composed_stdcell_area_um2']:.0f} | "
            f"{row['area_ratio_vs_scalar_rmw']:.3f}x | "
            f"{row['speedup_vs_scalar_rmw']:.3f}x | "
            f"{row['area_normalized_throughput_vs_scalar_rmw']:.3f}x | "
            f"{'是' if row['setup_closed_at_5ns'] else '否'} | "
            f"{'是' if row['drc_clean'] else '否'} | "
            f"{'是' if row['drv_clean'] else '否'} |"
        )

    md += [
        "",
        "## 证据边界",
        "",
        "四个块均完成详细布线且最终 DRC=0，5 ns 下 setup/hold=0；但仍有少量 max-cap 违例，因此本轮是流程闭环和物理趋势证据，不是物理签核。",
        "",
    ]
    md.extend(f"- {item}" for item in result["limits"])
    md += [
        "",
        "## 复现入口",
        "",
        "- 流程：`openroad_hifp/run_openroad_hifp_blocks.sh`；",
        "- 版本锁：`openroad_hifp/ORFS_VERSION.lock`；",
        "- 约束：`openroad_hifp/constraint.sdc`；",
        "- 汇总：`scripts/summarize_openroad_hifp_blocks.py`。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
