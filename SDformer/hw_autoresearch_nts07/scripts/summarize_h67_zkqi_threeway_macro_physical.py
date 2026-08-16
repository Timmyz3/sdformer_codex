#!/usr/bin/env python3
"""汇总Motion ZKQI三方同SRAM宏OpenROAD物理代理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


UNCONSTRAINED_RE = re.compile(r"Warning: There are (\d+) unconstrained endpoints\.")
CRITICAL_DELAY_RE = re.compile(r"finish critical path delay\s*[-=]+\s*([0-9.]+)", re.S)
SETUP_GROUP_RE = re.compile(r"finish setup_violation_count\s*[-=]+\s*setup violation count (\d+)", re.S)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_receipt(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
    ]


def parse_unconstrained(path: Path) -> int:
    matches = UNCONSTRAINED_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"OpenROAD日志缺少unconstrained endpoint统计: {path}")
    return int(matches[-1])


def parse_critical_delay(path: Path) -> float:
    matches = CRITICAL_DELAY_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"OpenROAD日志缺少critical path delay: {path}")
    return float(matches[-1])


def parse_setup_groups(path: Path) -> int:
    matches = SETUP_GROUP_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"OpenROAD日志缺少setup violation group统计: {path}")
    return int(matches[-1])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def physical_mode(log_dir: Path) -> dict[str, Any]:
    final_path = log_dir / "6_report.json"
    route_path = log_dir / "5_2_TritonRoute.json"
    log_path = log_dir / "6_report.log"
    final = read_json(final_path)
    route = read_json(route_path)
    result = {
        "instance_area_um2": final["finish__design__instance__area"],
        "stdcell_area_um2": final["finish__design__instance__area__stdcell"],
        "macro_area_um2": final["finish__design__instance__area__macros"],
        "instance_count": final["finish__design__instance__count"],
        "stdcell_count": final["finish__design__instance__count__stdcell"],
        "macro_count": final["finish__design__instance__count__macros"],
        "utilization": final["finish__design__instance__utilization"],
        "worst_setup_slack_ns": final["finish__timing__setup__ws"],
        "setup_tns_ns": final["finish__timing__setup__tns"],
        # ORFS JSON和文本摘要给出不同粒度：前者是负slack endpoint指标，
        # 后者是setup violation group数。两者均保留，避免混用。
        "setup_negative_endpoint_metric": final[
            "finish__timing__drv__setup_violation_count"
        ],
        "setup_violation_group_count": parse_setup_groups(log_path),
        "hold_violation_count": final[
            "finish__timing__drv__hold_violation_count"
        ],
        "max_cap_violation_count": final["finish__timing__drv__max_cap"],
        "max_slew_violation_count": final["finish__timing__drv__max_slew"],
        "drc_error_count": route["detailedroute__route__drc_errors"],
        "wirelength_um": route["detailedroute__route__wirelength"],
        "via_count": route["detailedroute__route__vias"],
        "unconstrained_endpoint_count": parse_unconstrained(log_path),
        "critical_path_delay_ns": parse_critical_delay(log_path),
    }
    result["timing_closed"] = (
        result["worst_setup_slack_ns"] >= 0
        and result["setup_negative_endpoint_metric"] == 0
        and result["setup_violation_group_count"] == 0
        and result["hold_violation_count"] == 0
        and result["max_cap_violation_count"] == 0
        and result["max_slew_violation_count"] == 0
        and result["unconstrained_endpoint_count"] == 0
    )
    result["source_receipts"] = source_receipt([final_path, route_path, log_path])
    return result


def throughput_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_cycles: int,
    candidate_cycles: int,
) -> dict[str, float]:
    cycle_speedup = baseline_cycles / candidate_cycles
    frequency_ratio_proxy = (
        baseline["critical_path_delay_ns"] / candidate["critical_path_delay_ns"]
    )
    throughput_proxy = cycle_speedup * frequency_ratio_proxy
    area_ratio = candidate["instance_area_um2"] / baseline["instance_area_um2"]
    return {
        "cycle_speedup": cycle_speedup,
        "frequency_ratio_proxy": frequency_ratio_proxy,
        "frequency_adjusted_throughput_proxy": throughput_proxy,
        "area_ratio": area_ratio,
        "area_normalized_throughput_proxy": throughput_proxy / area_ratio,
    }


def summarize(root: Path) -> dict[str, Any]:
    cycle_path = root / "results/h67_zkqi_threeway_20260809/report.json"
    cycle_report = read_json(cycle_path)
    mode0 = cycle_report["cycles"]["modes"]["0"]
    log_root = (
        root
        / "openroad_hifp/work/logs/nangate45"
        / "h67_zkqi_threeway_manualmacro_proxy"
    )
    physical = {
        mode: physical_mode(log_root / mode)
        for mode in ("baseline", "pairbitmap", "ttb8")
    }
    if any(value["macro_count"] != 6 for value in physical.values()):
        raise ValueError("三方必须各含6个SRAM宏")
    macro_areas = {value["macro_area_um2"] for value in physical.values()}
    if len(macro_areas) != 1:
        raise ValueError("三方SRAM宏面积不一致")
    if any(value["drc_error_count"] != 0 for value in physical.values()):
        raise ValueError("详细布线仍有DRC错误")

    cycles = {
        "baseline": mode0["baseline_e2e_cycles"],
        "pairbitmap": mode0["pair_bitmap_e2e_cycles"],
        "ttb8": mode0["ttb8_zkqi_e2e_cycles"],
    }
    comparisons = {
        mode: throughput_metrics(
            physical["baseline"],
            physical[mode],
            cycles["baseline"],
            cycles[mode],
        )
        for mode in ("pairbitmap", "ttb8")
    }
    all_closed = all(value["timing_closed"] for value in physical.values())
    source_paths = [
        cycle_path,
        root / "openroad_hifp/config_h67_zkqi_threeway_macro.mk",
        root / "openroad_hifp/constraint_h67_zkqi_macro.sdc",
        root / "openroad_hifp/h67_zkqi_macros.cfg",
        root / "openroad_hifp/run_openroad_h67_zkqi_threeway_macro.sh",
        Path(__file__),
    ]
    return {
        "schema": "h67_zkqi_threeway_macro_physical_v1",
        "status": "PASS" if all_closed else "PARTIAL",
        "evidence_level": "[rtl]+[open-pnr代理]",
        "scope": (
            "Motion H67 sample0/window0、138条真实T450 head-row；三方同一5ns SDC、"
            "900x700um outline、固定宏位置和6个fakeram45_256x32 SRAM宏"
        ),
        "cycles_preload_inclusive": cycles,
        "physical": physical,
        "comparisons_vs_baseline": comparisons,
        "negative_results": [
            "OpenROAD RTLMP在浅层6宏层次触发内部assert，改用三方相同的固定宏位置后完成布线。",
            "三方详细布线均为0 DRC，但5ns下均存在setup和max-cap违例，不能称为时序闭合。",
            "最终报告仍有101至125个unconstrained endpoints；开放宏lib/时序合同需在DC/PT中复核。",
            "本轮没有门级SAIF、功耗或EDP，不以翻转代理替代功耗。",
        ],
        "claim_boundary": [
            "可声称：同SRAM宏、同outline、同SDC的开放布局布线代理支持TTB8的周期和面积收益趋势。",
            "不可声称：目标工艺ASIC PPA、STA签核、功耗/EDP、full encoder或多样本部署收益。",
            "频率折算使用各自最终critical path delay，只是存在违例和未约束端点条件下的开放代理。",
        ],
        "source_receipts": source_receipt(source_paths),
    }


def render_markdown(result: dict[str, Any]) -> str:
    p = result["physical"]
    c = result["comparisons_vs_baseline"]
    lines = [
        "# Motion ZKQI三方同SRAM宏物理代理",
        "",
        "## 结论",
        "",
        f"- 状态：**{result['status']}**。三方均完成详细布线且DRC=0，但均未完成5ns时序签核。",
        f"- [rtl] 含每行225拍预载周期：基线 `{result['cycles_preload_inclusive']['baseline']}`、pair-bitmap `{result['cycles_preload_inclusive']['pairbitmap']}`、TTB8-ZKQI `{result['cycles_preload_inclusive']['ttb8']}`。",
        f"- [open-pnr代理] TTB8周期加速 `{c['ttb8']['cycle_speedup']:.3f}x`，频率折算吞吐 `{c['ttb8']['frequency_adjusted_throughput_proxy']:.3f}x`，面积归一吞吐 `{c['ttb8']['area_normalized_throughput_proxy']:.3f}x`。",
        f"- [open-pnr代理] pair-bitmap面积归一吞吐 `{c['pairbitmap']['area_normalized_throughput_proxy']:.3f}x`，仍是负结果。",
        "",
        "## 同约束物理结果",
        "",
        "| 候选 | 总面积(um^2) | 标准单元面积 | SRAM宏 | WNS(ns) | TNS(ns) | setup负slack端点指标/违例组 | max-cap数 | DRC | 未约束端点 | 关键路径(ns) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"baseline": "RQTB2S", "pairbitmap": "pair-bitmap", "ttb8": "TTB8-ZKQI"}
    for mode in ("baseline", "pairbitmap", "ttb8"):
        value = p[mode]
        lines.append(
            f"| {labels[mode]} | {value['instance_area_um2']:.0f} | "
            f"{value['stdcell_area_um2']:.1f} | {value['macro_count']} | "
            f"{value['worst_setup_slack_ns']:.4f} | {value['setup_tns_ns']:.4f} | "
            f"{value['setup_negative_endpoint_metric']}/{value['setup_violation_group_count']} | "
            f"{value['max_cap_violation_count']} | "
            f"{value['drc_error_count']} | {value['unconstrained_endpoint_count']} | "
            f"{value['critical_path_delay_ns']:.4f} |"
        )
    lines.extend([
        "",
        "## 公平对照",
        "",
        "| 候选 | 周期加速 | 频率比代理 | 频率折算吞吐 | 面积比 | 面积归一吞吐 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("pairbitmap", "ttb8"):
        value = c[mode]
        lines.append(
            f"| {labels[mode]} | {value['cycle_speedup']:.3f}x | "
            f"{value['frequency_ratio_proxy']:.3f}x | "
            f"{value['frequency_adjusted_throughput_proxy']:.3f}x | "
            f"{value['area_ratio']:.3f}x | "
            f"{value['area_normalized_throughput_proxy']:.3f}x |"
        )
    lines.extend(["", "## 负结果", ""])
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines.extend(["", "## 证据边界", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.extend(["", "## 复现入口", "", "- OpenROAD：`openroad_hifp/run_openroad_h67_zkqi_threeway_macro.sh all`；", "- 汇总：`scripts/summarize_h67_zkqi_threeway_macro_physical.py`。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
