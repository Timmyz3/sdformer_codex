#!/usr/bin/env python3
"""汇总Motion ZKQI三方生产边界OpenROAD物理代理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DESIGN = "h67_zkqi_threeway_production_cap40_setup50ps_macro_proxy"
MODES = ("baseline", "pairbitmap", "ttb8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipts(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def require_last_int(pattern: str, text: str, label: str) -> int:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"最终报告缺少{label}")
    return int(matches[-1])


def require_last_float(pattern: str, text: str, label: str) -> float:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"最终报告缺少{label}")
    return float(matches[-1])


def collect_mode(log_root: Path, mode: str) -> tuple[dict[str, Any], list[Path]]:
    root = log_root / mode
    final_path = root / "6_report.json"
    final_log_path = root / "6_report.log"
    route_path = root / "5_2_TritonRoute.json"
    route_log_path = root / "5_2_TritonRoute.log"
    paths = [final_path, final_log_path, route_path, route_log_path]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"缺少OpenROAD证据: {path}")

    final = load_json(final_path)
    route = load_json(route_path)
    final_text = final_log_path.read_text(encoding="utf-8", errors="replace")
    route_text = route_log_path.read_text(encoding="utf-8", errors="replace")
    if "finish check_setup" not in final_text:
        raise ValueError(f"{mode}缺少最终check_setup")
    if "[INFO DRT-0198] Complete detail routing." not in route_text:
        raise ValueError(f"{mode}详细布线未完成")

    unconstrained_matches = re.findall(
        r"Warning: There (?:is|are) (\d+) unconstrained endpoint", final_text
    )
    unconstrained = max((int(value) for value in unconstrained_matches), default=0)
    setup_groups = require_last_int(
        r"setup violation count\s+(\d+)", final_text, "setup违例组数"
    )
    critical_delay = require_last_float(
        r"finish critical path delay\s*-+\s*([0-9.]+)",
        final_text,
        "关键路径延迟",
    )
    result = {
        "instance_area_um2": final["finish__design__instance__area"],
        "stdcell_area_um2": final["finish__design__instance__area__stdcell"],
        "macro_area_um2": final["finish__design__instance__area__macros"],
        "instance_count": final["finish__design__instance__count"],
        "macro_count": final["finish__design__instance__count__macros"],
        "worst_setup_slack_ns": final["finish__timing__setup__ws"],
        "setup_tns_ns": final["finish__timing__setup__tns"],
        "setup_negative_endpoint_metric": final[
            "finish__timing__drv__setup_violation_count"
        ],
        "setup_violation_group_count": setup_groups,
        "hold_violation_count": final["finish__timing__drv__hold_violation_count"],
        "max_cap_violation_count": final["finish__timing__drv__max_cap"],
        "max_slew_violation_count": final["finish__timing__drv__max_slew"],
        "max_fanout_violation_count": final["finish__timing__drv__max_fanout"],
        "drc_error_count": route["detailedroute__route__drc_errors"],
        "wirelength_um": route["detailedroute__route__wirelength"],
        "via_count": route["detailedroute__route__vias"],
        "unconstrained_endpoint_count": unconstrained,
        "critical_path_delay_ns": critical_delay,
    }
    result["open_proxy_clean"] = (
        result["worst_setup_slack_ns"] >= 0
        and result["setup_tns_ns"] == 0
        and result["setup_negative_endpoint_metric"] == 0
        and result["setup_violation_group_count"] == 0
        and result["hold_violation_count"] == 0
        and result["max_cap_violation_count"] == 0
        and result["max_slew_violation_count"] == 0
        and result["max_fanout_violation_count"] == 0
        and result["drc_error_count"] == 0
        and result["unconstrained_endpoint_count"] == 0
    )
    return result, paths


def comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_cycles: int,
    candidate_cycles: int,
) -> dict[str, float]:
    cycle_speedup = baseline_cycles / candidate_cycles
    area_ratio = candidate["instance_area_um2"] / baseline["instance_area_um2"]
    return {
        "cycle_speedup": cycle_speedup,
        "fixed_5ns_throughput_ratio": cycle_speedup,
        "area_ratio": area_ratio,
        "fixed_5ns_area_normalized_throughput": cycle_speedup / area_ratio,
    }


def summarize(root: Path) -> dict[str, Any]:
    cycle_path = root / "results/h67_zkqi_threeway_20260809/report.json"
    cycle_report = load_json(cycle_path)
    if cycle_report.get("status") != "PASS":
        raise ValueError("三方RTL报告未PASS")
    cycle_mode = cycle_report["cycles"]["modes"]["0"]
    cycles = {
        "baseline": cycle_mode["baseline_e2e_cycles"],
        "pairbitmap": cycle_mode["pair_bitmap_e2e_cycles"],
        "ttb8": cycle_mode["ttb8_zkqi_e2e_cycles"],
    }
    log_root = root / "openroad_hifp/work/logs/nangate45" / DESIGN
    physical: dict[str, Any] = {}
    evidence_paths: list[Path] = []
    for mode in MODES:
        physical[mode], mode_paths = collect_mode(log_root, mode)
        evidence_paths.extend(mode_paths)

    if any(value["macro_count"] != 6 for value in physical.values()):
        raise ValueError("三方必须各含6个SRAM宏")
    if len({value["macro_area_um2"] for value in physical.values()}) != 1:
        raise ValueError("三方SRAM宏面积必须一致")

    comparisons = {
        mode: comparison(physical["baseline"], physical[mode], cycles["baseline"], cycles[mode])
        for mode in ("pairbitmap", "ttb8")
    }
    all_clean = all(value["open_proxy_clean"] for value in physical.values())
    architecture_gate = (
        comparisons["ttb8"]["cycle_speedup"] >= 1.10
        and comparisons["ttb8"]["fixed_5ns_area_normalized_throughput"] > 1.0
    )
    source_paths = [
        cycle_path,
        root / "rtl_h67/h67_banked_active_descriptor_store.sv",
        root / "rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv",
        root / "rtl_h67/h67_zkqi_row_shiftmax_physical_top.sv",
        root / "openroad_hifp/config_h67_zkqi_threeway_production_macro.mk",
        root / "openroad_hifp/constraint_h67_zkqi_production_macro.sdc",
        root / "openroad_hifp/h67_zkqi_production_macros.cfg",
        root / "openroad_hifp/run_openroad_h67_zkqi_threeway_production_macro.sh",
        Path(__file__),
    ]
    return {
        "schema": "h67_zkqi_threeway_production_physical_v2",
        "status": "PASS" if all_clean and architecture_gate else "FAIL",
        "evidence_level": "[rtl]+[open-pnr代理]",
        "scope": "Motion H67 sample0/window0、138条真实T450 head-row；5ns、CAP_MARGIN=40、SETUP_SLACK_MARGIN=0.05ns、同6个SRAM宏",
        "cycles_preload_inclusive": cycles,
        "physical": physical,
        "comparisons_vs_baseline": comparisons,
        "architecture_gate": {
            "all_three_open_proxy_clean": all_clean,
            "ttb8_cycle_speedup_ge_1p10": comparisons["ttb8"]["cycle_speedup"] >= 1.10,
            "ttb8_fixed_5ns_area_normalized_throughput_gt_1": comparisons["ttb8"]["fixed_5ns_area_normalized_throughput"] > 1.0,
        },
        "negative_results": [
            "pair-bitmap在无反压时不减少执行周期；其价值主要是读活动门控，而不是独立吞吐贡献。",
            "三方仅在共同5ns约束下闭合，未扫描各自Fmax；关键路径裸延迟不用于推导频率比。",
            "本报告没有门级SAIF、库功耗或EDP，不能回答功耗收益。",
            "当前真实trace仍只有sample0/window0，不能外推多场景p95/p99。",
        ],
        "claim_boundary": [
            "可声称：三方在相同开放工艺代理、SDC、宏、floorplan和优化margin下的物理趋势。",
            "可声称：在共同200MHz下，由周期数得到固定频率吞吐比；不得声称候选提高了Fmax。",
            "不可声称：DC/PT签核PPA、目标SRAM编译器结果、ASIC功耗/EDP或full encoder FPS。",
            "32-bit开放目录宏写入padding位，功耗方向保守；目标DC应替换为20-bit SRAM宏。",
        ],
        "source_receipts": receipts(source_paths + evidence_paths),
    }


def render_markdown(result: dict[str, Any]) -> str:
    labels = {"baseline": "RQTB2S", "pairbitmap": "pair-bitmap", "ttb8": "TTB8-ZKQI"}
    lines = [
        "# Motion ZKQI三方生产边界开放物理签核",
        "",
        "## 结论",
        "",
        f"- 状态：**{result['status']}**；证据等级：{result['evidence_level']}。",
        f"- 范围：{result['scope']}。",
        "",
        "## 物理结果",
        "",
        "| 候选 | 面积(um²) | 宏数 | WNS(ns) | TNS(ns) | setup端点/组 | hold | cap | slew | DRC | 未约束 | 关键路径(ns) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        value = result["physical"][mode]
        lines.append(
            f"| {labels[mode]} | {value['instance_area_um2']:.0f} | {value['macro_count']} | "
            f"{value['worst_setup_slack_ns']:+.4f} | {value['setup_tns_ns']:.4f} | "
            f"{value['setup_negative_endpoint_metric']}/{value['setup_violation_group_count']} | "
            f"{value['hold_violation_count']} | {value['max_cap_violation_count']} | "
            f"{value['max_slew_violation_count']} | {value['drc_error_count']} | "
            f"{value['unconstrained_endpoint_count']} | {value['critical_path_delay_ns']:.4f} |"
        )
    lines.extend([
        "",
        "## 公平收益",
        "",
        "| 候选 | 含预载周期 | 固定200MHz吞吐比 | 面积比 | 固定200MHz面积归一吞吐 |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in ("pairbitmap", "ttb8"):
        value = result["comparisons_vs_baseline"][mode]
        lines.append(
            f"| {labels[mode]} | {result['cycles_preload_inclusive'][mode]} | "
            f"{value['fixed_5ns_throughput_ratio']:.3f}x | {value['area_ratio']:.3f}x | "
            f"{value['fixed_5ns_area_normalized_throughput']:.3f}x |"
        )
    lines.extend(["", "## 负结果", ""])
    lines.extend(f"- {item}" for item in result["negative_results"])
    lines.extend(["", "## 证据边界", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.extend([
        "",
        "## 复现入口",
        "",
        "- PnR：`openroad_hifp/run_openroad_h67_zkqi_threeway_production_macro.sh all`；",
        "- 汇总：`scripts/summarize_h67_zkqi_threeway_production_physical.py --output-dir results/h67_zkqi_threeway_production_physical_20260809`。",
        "",
    ])
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
    (args.output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
