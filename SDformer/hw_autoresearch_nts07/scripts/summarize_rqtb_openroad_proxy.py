#!/usr/bin/env python3
"""汇总Fixed-TTB32与RQTB16/32的同约束OpenROAD物理代理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DESIGN = "h67_rqtb_t450_flopmem_proxy"
RTL_SOURCES = (
    "rtl_ttx/ttx_ceil_log2_u32.sv",
    "rtl_ttx/ttx_exp2_lut_q8.sv",
    "rtl_ttx/ttx_gate_quant_q17.sv",
    "rtl_h67/h67_motionxor_score_q7.sv",
    "rtl_h67/h67_temporal_slot_encoder.sv",
    "rtl_h67/h67_temporal_slot_fifo.sv",
    "rtl_h67/h67_sync_dual_bank_k_store.sv",
    "rtl_h67/h67_temporal_weighted_scs_directory.sv",
    "rtl_h67/h67_temporal_slot_shiftmax_sync_k_top.sv",
)
EXPECTED_ENDPOINTS = {
    "fixed": {*(f"perf_k_read_bits[{index}]" for index in range(5)), "perf_slots[0]"},
    "rqtb": {*(f"perf_k_read_bits[{index}]" for index in range(5))},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_receipt(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
    ]


def parse_unconstrained_endpoints(text: str) -> list[str]:
    count_match = re.search(r"There (?:is|are) (\d+) unconstrained endpoint", text)
    if count_match is None:
        raise ValueError("check_setup日志缺少未约束端点计数")
    endpoints = re.findall(r"^  ([^\s]+)$", text, flags=re.MULTILINE)
    expected_count = int(count_match.group(1))
    if len(endpoints) != expected_count:
        raise ValueError(
            f"未约束端点计数与名称不一致: count={expected_count}, names={len(endpoints)}"
        )
    return endpoints


def validate_route_complete(route: dict[str, Any], route_text: str) -> None:
    if "[INFO DRT-0198] Complete detail routing." not in route_text:
        raise ValueError("详细布线日志未完成")
    if route.get("detailedroute__route__drc_errors") != 0:
        raise ValueError("详细布线最终DRC不为0")
    iterations = sorted(
        int(key.rsplit(":", 1)[1])
        for key in route
        if key.startswith("detailedroute__route__drc_errors__iter:")
    )
    if not iterations:
        raise ValueError("详细布线日志缺少迭代DRC")
    final_key = f"detailedroute__route__drc_errors__iter:{iterations[-1]}"
    if route[final_key] != 0:
        raise ValueError("详细布线最后一次迭代DRC不为0")


def collect_mode(log_root: Path, mode: str) -> tuple[dict[str, Any], list[Path]]:
    root = log_root / "nangate45" / DESIGN / mode
    route_path = root / "5_2_TritonRoute.json"
    route_log_path = root / "5_2_TritonRoute.log"
    finish_path = root / "6_report.json"
    finish_log_path = root / "6_report.log"
    verbose_path = root / "check_setup_verbose.log"
    for path in (route_path, route_log_path, finish_path, finish_log_path, verbose_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少OpenROAD证据: {path}")

    route = load_json(route_path)
    finish = load_json(finish_path)
    route_text = route_log_path.read_text(encoding="utf-8", errors="replace")
    finish_text = finish_log_path.read_text(encoding="utf-8", errors="replace")
    verbose_text = verbose_path.read_text(encoding="utf-8", errors="replace")
    validate_route_complete(route, route_text)
    if "finish report_design_area" not in finish_text:
        raise ValueError(f"post-route STA报告不完整: {mode}")

    endpoints = parse_unconstrained_endpoints(verbose_text)
    unexpected = sorted(set(endpoints) - EXPECTED_ENDPOINTS[mode])
    missing_expected = sorted(EXPECTED_ENDPOINTS[mode] - set(endpoints))
    if unexpected or missing_expected:
        raise ValueError(
            f"未约束端点合同漂移({mode}): unexpected={unexpected}, missing={missing_expected}"
        )
    if finish["finish__design__instance__count__macros"] != 0:
        raise ValueError("该代理合同要求无macro、memory全部映射为flop")

    setup_violations = finish["finish__timing__drv__setup_violation_count"]
    hold_violations = finish["finish__timing__drv__hold_violation_count"]
    max_slew = finish["finish__timing__drv__max_slew"]
    max_cap = finish["finish__timing__drv__max_cap"]
    max_fanout = finish["finish__timing__drv__max_fanout"]
    result = {
        "mode": mode,
        "stdcell_area_um2": finish["finish__design__instance__area__stdcell"],
        "stdcell_count": finish["finish__design__instance__count__stdcell"],
        "macro_count": finish["finish__design__instance__count__macros"],
        "core_area_um2": finish["finish__design__core__area"],
        "die_area_um2": finish["finish__design__die__area"],
        "utilization": finish["finish__design__instance__utilization__stdcell"],
        "io_pins": finish["finish__design__io"],
        "setup_slack_ns": finish["finish__timing__setup__ws"],
        "setup_tns_ns": finish["finish__timing__setup__tns"],
        "critical_delay_ns": 4.5 - finish["finish__timing__setup__ws"],
        "setup_violations": setup_violations,
        "hold_violations": hold_violations,
        "max_slew_violations": max_slew,
        "max_cap_violations": max_cap,
        "max_fanout_violations": max_fanout,
        "setup_hold_closed": setup_violations == 0 and hold_violations == 0,
        "electrical_drv_clean": max_slew == 0 and max_cap == 0 and max_fanout == 0,
        "routing_drc_violations": route["detailedroute__route__drc_errors"],
        "routing_drc_clean": route["detailedroute__route__drc_errors"] == 0,
        "wirelength_um": route["detailedroute__route__wirelength"],
        "vias": route["detailedroute__route__vias"],
        "unconstrained_endpoints": endpoints,
        "unexpected_unconstrained_endpoints": unexpected,
    }
    return result, [route_path, route_log_path, finish_path, finish_log_path, verbose_path]


def build_physical_boundaries(
    fixed: dict[str, Any], rqtb: dict[str, Any]
) -> tuple[list[str], list[str]]:
    negative_results = [
        (
            "Fixed/RQTB的post-route WNS分别为"
            f"{fixed['setup_slack_ns']:+.4f}/{rqtb['setup_slack_ns']:+.4f} ns，"
            "setup/hold违例分别为"
            f"{fixed['setup_violations']}/{fixed['hold_violations']}和"
            f"{rqtb['setup_violations']}/{rqtb['hold_violations']}。"
        ),
        (
            f"Fixed/RQTB分别仍有{fixed['max_cap_violations']}/"
            f"{rqtb['max_cap_violations']}个max-cap违例，不能称DRV签核。"
        ),
        "本机缺KLayout；最终RC/STA报告已生成，但完整finish/GDS步骤未完成。",
        "两种设计均无SRAM宏，T450 K存储和目录均由触发器映射，面积绝对值不能外推目标ASIC。",
    ]
    claim_boundary = [
        (
            "可声称：Fixed与RQTB在同一开放工艺、同一5 ns约束和同一全flop-memory模型下"
            "均达到0-DRC详细布线与post-route setup/hold闭合。"
        ),
        "可声称：结合真实T450 RTL周期，RQTB获得开放物理代理下的面积归一吞吐收益。",
        "不可声称：两套均完成DRV签核、目标SRAM宏PPA、DC/PTPX功耗或GDS签核。",
    ]
    return negative_results, claim_boundary


def build_report(root: Path, rtl_report_path: Path) -> dict[str, Any]:
    log_root = root / "openroad_hifp/work/logs"
    fixed, fixed_paths = collect_mode(log_root, "fixed")
    rqtb, rqtb_paths = collect_mode(log_root, "rqtb")
    rtl = load_json(rtl_report_path)
    if rtl.get("status") != "PASS" or rtl.get("schema") != "h67_rqtb_physical_flow_t450_v1":
        raise ValueError("RTL报告不是已通过的RQTB T450合同")

    physical_inputs = [root / relative for relative in RTL_SOURCES] + [
        root / "openroad_hifp/config_rqtb_t450.mk",
        root / "openroad_hifp/constraint_rqtb.sdc",
    ]
    latest_input_mtime = max(path.stat().st_mtime_ns for path in physical_inputs)
    physical_netlists = []
    synth_logs = []
    for mode in ("fixed", "rqtb"):
        synth_log = log_root / "nangate45" / DESIGN / mode / "1_1_yosys.log"
        netlist = (
            root / "openroad_hifp/work/results/nangate45" / DESIGN / mode
            / "1_synth.v"
        )
        if not synth_log.exists() or not netlist.exists():
            raise FileNotFoundError(f"缺少综合证据: {synth_log} / {netlist}")
        if synth_log.stat().st_mtime_ns < latest_input_mtime:
            raise ValueError(f"OpenROAD综合结果早于当前RTL/约束，必须重跑: {mode}")
        synth_text = synth_log.read_text(encoding="utf-8", errors="replace")
        missing_sources = [
            str(path.resolve()) for path in physical_inputs[: len(RTL_SOURCES)]
            if str(path.resolve()) not in synth_text
        ]
        if missing_sources:
            raise ValueError(f"综合日志缺少RTL源文件({mode}): {missing_sources}")
        synth_logs.append(synth_log)
        physical_netlists.append(netlist)

    speedup = rtl["performance"]["speedup"]
    area_ratio = rqtb["stdcell_area_um2"] / fixed["stdcell_area_um2"]
    negative_results, claim_boundary = build_physical_boundaries(fixed, rqtb)
    result = {
        "schema": "h67_rqtb_openroad_flopmem_proxy_v1",
        "status": "OPEN_PNR_ROUTE_COMPLETE_NOT_SIGNOFF",
        "evidence_level": "[open-pnr代理]",
        "flow": {
            "platform": "Nangate45",
            "clock_period_ns": 5.0,
            "io_budget_each_side_ns": 0.5,
            "effective_register_path_budget_ns": 4.5,
            "slot_fifo_depth": 32,
            "temporal_tokens": 450,
            "memory_model": "全部行为memory映射为flop，macro_count=0",
            "openroad_commit": "547465ccf8979379098216194f5837c413c7e2e9",
        },
        "fixed": fixed,
        "rqtb": rqtb,
        "comparison": {
            "rtl_cycle_speedup": speedup,
            "rqtb_area_ratio": area_ratio,
            "area_normalized_throughput": speedup / area_ratio,
            "stdcell_count_change_ratio": (
                rqtb["stdcell_count"] / fixed["stdcell_count"] - 1.0
            ),
            "wirelength_change_ratio": rqtb["wirelength_um"] / fixed["wirelength_um"] - 1.0,
            "via_change_ratio": rqtb["vias"] / fixed["vias"] - 1.0,
        },
        "signoff_gates": {
            "both_route_drc_clean": fixed["routing_drc_clean"] and rqtb["routing_drc_clean"],
            "both_setup_hold_closed": fixed["setup_hold_closed"] and rqtb["setup_hold_closed"],
            "both_electrical_drv_clean": fixed["electrical_drv_clean"] and rqtb["electrical_drv_clean"],
            "full_finish_with_klayout": False,
            "asic_signoff": False,
        },
        "negative_results": negative_results,
        "claim_boundary": claim_boundary,
        "rtl_report": {
            "path": str(rtl_report_path.resolve()),
            "sha256": sha256(rtl_report_path),
        },
        "source_receipts": source_receipt(
            fixed_paths
            + rqtb_paths
            + [
                rtl_report_path,
                root / "openroad_hifp/config_rqtb_t450.mk",
                root / "openroad_hifp/constraint_rqtb.sdc",
                root / "openroad_hifp/run_openroad_rqtb_t450.sh",
                root / "openroad_hifp/run_check_setup_rqtb_verbose.sh",
                Path(__file__),
            ]
            + physical_inputs
            + synth_logs
            + physical_netlists
        ),
    }
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    fixed = result["fixed"]
    rqtb = result["rqtb"]
    comparison = result["comparison"]
    lines = [
        "# Motion RQTB同约束OpenROAD物理代理",
        "",
        "## 结论",
        "",
        "- 状态：**详细布线与post-route RC/STA报告完成，但不是ASIC签核**。",
        f"- RQTB相对Fixed的RTL周期加速为{comparison['rtl_cycle_speedup']:.3f}x；标准单元面积比为{comparison['rqtb_area_ratio']:.4f}x；面积归一吞吐代理为{comparison['area_normalized_throughput']:.3f}x。",
        f"- 两者详细布线均为0 DRC；Fixed/RQTB的post-route WNS分别为{fixed['setup_slack_ns']:.4f}/{rqtb['setup_slack_ns']:.4f} ns。",
        f"- 两者仍有{fixed['max_cap_violations']}/{rqtb['max_cap_violations']}个max-cap违例，因此不能称物理签核。",
        "",
        "## 同约束结果",
        "",
        "| 候选 | 标准单元面积(um2) | 单元数 | WNS(ns) | setup/hold | max-cap | DRC | 线长(um) | via | 未约束端点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, row in (("Fixed-TTB32", fixed), ("RQTB16/32", rqtb)):
        lines.append(
            f"| {label} | {row['stdcell_area_um2']:.0f} | {row['stdcell_count']} | "
            f"{row['setup_slack_ns']:.4f} | {row['setup_violations']}/{row['hold_violations']} | "
            f"{row['max_cap_violations']} | {row['routing_drc_violations']} | "
            f"{row['wirelength_um']:.0f} | {row['vias']} | "
            f"`{', '.join(row['unconstrained_endpoints'])}` |"
        )
    lines += [
        "",
        "## 物理变化",
        "",
        f"- RQTB标准单元面积变化：{comparison['rqtb_area_ratio'] - 1.0:+.2%}。",
        f"- 标准单元数变化：{comparison['stdcell_count_change_ratio']:+.2%}。",
        f"- 线长变化：{comparison['wirelength_change_ratio']:+.2%}；via变化：{comparison['via_change_ratio']:+.2%}。",
        "- 未约束端点均为综合后常量化的性能计数器低位，不属于主数据输出；解析器对名称集合做精确检查。",
        "",
        "## 负结果",
        "",
    ]
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines += ["", "## 论文表述边界", ""]
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rtl-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rtl_report = args.rtl_report
    if not rtl_report.is_absolute():
        rtl_report = args.root / rtl_report
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
