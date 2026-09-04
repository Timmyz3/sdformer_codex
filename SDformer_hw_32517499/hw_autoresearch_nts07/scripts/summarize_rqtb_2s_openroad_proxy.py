#!/usr/bin/env python3
"""汇总Fixed2S与RQTB2S公平强基线的同约束OpenROAD物理代理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import scripts.summarize_rqtb_openroad_proxy as base
except ModuleNotFoundError:
    import summarize_rqtb_openroad_proxy as base


DESIGN = "h67_rqtb_2s_t450_flopmem_proxy"
MODES = ("fixed2", "rqtb2")
RTL_SOURCES = (
    "rtl_ttx/ttx_ceil_log2_u32.sv",
    "rtl_ttx/ttx_exp2_lut_q8.sv",
    "rtl_ttx/ttx_gate_quant_q17.sv",
    "rtl_h67/h67_motionxor_score_q7.sv",
    "rtl_h67/h67_temporal_slot_encoder.sv",
    "rtl_h67/h67_temporal_slot_fifo_2s.sv",
    "rtl_h67/h67_sync_dual_bank_k_store.sv",
    "rtl_h67/h67_temporal_weighted_scs_directory_2s.sv",
    "rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv",
)
EXPECTED_ENDPOINTS = {
    "fixed2": {*(f"perf_k_read_bits[{index}]" for index in range(5)), "perf_slots[0]"},
    "rqtb2": {*(f"perf_k_read_bits[{index}]" for index in range(5))},
}


def collect_mode(root: Path, mode: str) -> tuple[dict[str, Any], list[Path]]:
    old_design = base.DESIGN
    old_expected = base.EXPECTED_ENDPOINTS
    try:
        base.DESIGN = DESIGN
        base.EXPECTED_ENDPOINTS = EXPECTED_ENDPOINTS
        return base.collect_mode(root / "openroad_hifp/work/logs", mode)
    finally:
        base.DESIGN = old_design
        base.EXPECTED_ENDPOINTS = old_expected


def build_report(root: Path, rtl_report_path: Path) -> dict[str, Any]:
    fixed, fixed_paths = collect_mode(root, "fixed2")
    rqtb, rqtb_paths = collect_mode(root, "rqtb2")
    rtl = base.load_json(rtl_report_path)
    if rtl.get("status") != "PASS" or rtl.get("schema") != "h67_rqtb_strong_baseline_v1":
        raise ValueError("RTL报告不是已通过的双-slot公平强基线")
    if not rtl.get("coverage", {}).get("rejected_restart_fail_closed"):
        raise ValueError("RTL报告缺少非法restart fail-closed receipt")
    if not rtl.get("coverage", {}).get("build_stage_rejected_restart_mutation_killed"):
        raise ValueError("RTL报告缺少构建阶段非法restart mutation-kill receipt")

    physical_inputs = [root / relative for relative in RTL_SOURCES] + [
        root / "openroad_hifp/config_rqtb_2s_t450.mk",
        root / "openroad_hifp/constraint_rqtb.sdc",
    ]
    latest_input_mtime = max(path.stat().st_mtime_ns for path in physical_inputs)
    synth_logs: list[Path] = []
    netlists: list[Path] = []
    for mode in MODES:
        synth_log = (
            root / "openroad_hifp/work/logs/nangate45" / DESIGN / mode / "1_1_yosys.log"
        )
        netlist = (
            root / "openroad_hifp/work/results/nangate45" / DESIGN / mode / "1_synth.v"
        )
        if not synth_log.exists() or not netlist.exists():
            raise FileNotFoundError(f"缺少综合证据: {synth_log} / {netlist}")
        if synth_log.stat().st_mtime_ns < latest_input_mtime:
            raise ValueError(f"OpenROAD综合结果早于当前RTL/约束，必须重跑: {mode}")
        text = synth_log.read_text(encoding="utf-8", errors="replace")
        missing = [
            str(path.resolve()) for path in physical_inputs[: len(RTL_SOURCES)]
            if str(path.resolve()) not in text
        ]
        if missing:
            raise ValueError(f"综合日志缺少RTL源文件({mode}): {missing}")
        synth_logs.append(synth_log)
        netlists.append(netlist)

    speedup = rtl["cycles"]["rqtb_vs_fixed_2s_primary"]["speedup"]
    area_ratio = rqtb["stdcell_area_um2"] / fixed["stdcell_area_um2"]
    result = {
        "schema": "h67_rqtb_2s_openroad_flopmem_proxy_v1",
        "status": "OPEN_PNR_ROUTE_COMPLETE_NOT_SIGNOFF",
        "evidence_level": "[open-pnr代理]",
        "flow": {
            "platform": "Nangate45",
            "clock_period_ns": 5.0,
            "io_budget_each_side_ns": 0.5,
            "effective_register_path_budget_ns": 4.5,
            "slot_width_per_cycle": 2,
            "slot_fifo_depth": 32,
            "temporal_tokens": 450,
            "memory_model": "全部行为memory映射为flop，macro_count=0",
            "openroad_commit": "547465ccf8979379098216194f5837c413c7e2e9",
        },
        "fixed2": fixed,
        "rqtb2": rqtb,
        "comparison": {
            "rtl_cycle_speedup": speedup,
            "rqtb_area_ratio": area_ratio,
            "area_normalized_throughput": speedup / area_ratio,
            "stdcell_count_change_ratio": rqtb["stdcell_count"] / fixed["stdcell_count"] - 1.0,
            "wirelength_change_ratio": rqtb["wirelength_um"] / fixed["wirelength_um"] - 1.0,
            "via_change_ratio": rqtb["vias"] / fixed["vias"] - 1.0,
        },
        "timing_count_contract": {
            "setup_hold_source": (
                "ORFS 6_report.json的finish__timing__drv__"
                "setup/hold_violation_count flow metric"
            ),
            "text_log_note": (
                "6_report.log中未指定多路径数量的find_timing_paths"
                "默认只列一条最差路径，不是flow metric总数"
            ),
        },
        "signoff_gates": {
            "both_route_drc_clean": fixed["routing_drc_clean"] and rqtb["routing_drc_clean"],
            "both_setup_hold_closed": fixed["setup_hold_closed"] and rqtb["setup_hold_closed"],
            "both_electrical_drv_clean": fixed["electrical_drv_clean"] and rqtb["electrical_drv_clean"],
            "asic_signoff": False,
        },
        "negative_results": [
            (
                f"Fixed2S/RQTB2S的post-route WNS为{fixed['setup_slack_ns']:+.4f}/"
                f"{rqtb['setup_slack_ns']:+.4f} ns，setup/hold违例为"
                f"{fixed['setup_violations']}/{fixed['hold_violations']}和"
                f"{rqtb['setup_violations']}/{rqtb['hold_violations']}。"
            ),
            (
                f"Fixed2S/RQTB2S仍有{fixed['max_cap_violations']}/"
                f"{rqtb['max_cap_violations']}个max-cap违例。"
            ),
            "2S FIFO、directory和K store均按多端口flop阵列实现，不等价于目标SRAM宏。",
            "当前只绑定sample0/window0的138个head-row，缺多样本与功耗证据。",
        ],
        "claim_boundary": [
            "可声称：公平双-slot主基线在同一开放工艺、SDC和全flop-memory模型下的详细布线结果。",
            "可声称：只在所有物理门槛逐项报告的前提下使用面积归一吞吐代理。",
            "不可声称：目标ASIC PPA、SRAM宏代价、SAIF/PTPX功耗、完整encoder吞吐或GDS签核。",
        ],
        "rtl_report": {
            "path": str(rtl_report_path.resolve()),
            "sha256": base.sha256(rtl_report_path),
        },
        "source_receipts": base.source_receipt(
            fixed_paths + rqtb_paths + [
                rtl_report_path,
                root / "openroad_hifp/config_rqtb_2s_t450.mk",
                root / "openroad_hifp/constraint_rqtb.sdc",
                root / "openroad_hifp/run_openroad_rqtb_2s_t450.sh",
                root / "openroad_hifp/run_check_setup_rqtb_2s_verbose.sh",
                Path(__file__),
            ] + physical_inputs + synth_logs + netlists
        ),
    }
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    fixed = result["fixed2"]
    rqtb = result["rqtb2"]
    comparison = result["comparison"]
    lines = [
        "# Motion RQTB双-slot公平强基线OpenROAD物理代理",
        "",
        "## 结论",
        "",
        "- 状态：**详细布线与post-route RC/STA完成，但不是ASIC签核**。",
        f"- 公平RTL周期加速：{comparison['rtl_cycle_speedup']:.3f}x；面积比：{comparison['rqtb_area_ratio']:.4f}x；面积归一吞吐代理：{comparison['area_normalized_throughput']:.3f}x。",
        f"- Fixed2S/RQTB2S WNS：{fixed['setup_slack_ns']:+.4f}/{rqtb['setup_slack_ns']:+.4f} ns；详细布线DRC：{fixed['routing_drc_violations']}/{rqtb['routing_drc_violations']}。",
        "",
        "## 同约束结果",
        "",
        "| 候选 | stdcell面积(um2) | 单元数 | WNS(ns) | setup/hold flow metric | max-cap | DRC | 线长(um) | via |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Fixed2S", fixed), ("RQTB2S", rqtb)):
        lines.append(
            f"| {label} | {row['stdcell_area_um2']:.0f} | {row['stdcell_count']} | "
            f"{row['setup_slack_ns']:+.4f} | {row['setup_violations']}/{row['hold_violations']} | "
            f"{row['max_cap_violations']} | {row['routing_drc_violations']} | "
            f"{row['wirelength_um']:.0f} | {row['vias']} |"
        )
    lines += ["", "## 负结果与边界", ""]
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines.append(
        "- setup/hold计数来自`6_report.json`的ORFS flow metric；"
        "文本日志默认只列一条最差负裕量路径，不是总数。"
    )
    lines += ["", "## 论文口径", ""]
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rtl-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rtl_report = args.rtl_report if args.rtl_report.is_absolute() else args.root / args.rtl_report
    result = build_report(args.root, rtl_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
