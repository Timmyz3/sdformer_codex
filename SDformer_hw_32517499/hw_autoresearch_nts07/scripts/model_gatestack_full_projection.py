#!/usr/bin/env python3
"""用H67 ordered trace评估完整窗口GateStack投影生命周期。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/gatestack_full_projection_model_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_full_projection_model_20260715.md"


def _ceil_efficiency(cycles: int, efficiency: float) -> int:
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("delivery_efficiency必须位于(0,1]")
    return math.ceil(cycles / efficiency)


def head_backend_cycles(
    *,
    active_lanes: int,
    class_terms: int,
    active_classes: int,
    delivery_transactions: int,
    class_slots: int,
    head_dim: int,
    product_engines: int,
    multicast_width: int,
    obi_issue_width: int,
    delivery_efficiency: float,
    pipeline_fill: int,
) -> dict[str, int | bool]:
    if min(active_lanes, class_terms, active_classes, delivery_transactions) < 0:
        raise ValueError("trace计数不能为负")
    if min(
        class_slots,
        head_dim,
        product_engines,
        multicast_width,
        obi_issue_width,
    ) <= 0:
        raise ValueError("硬件参数必须为正")
    if pipeline_fill < 0:
        raise ValueError("pipeline_fill不能为负")

    direct_issue = math.ceil(active_lanes / product_engines)
    direct_delivery = _ceil_efficiency(
        math.ceil(active_lanes / multicast_width), delivery_efficiency
    )
    direct = max(direct_issue, direct_delivery)
    if active_lanes:
        direct += pipeline_fill

    overflow = active_classes > class_slots
    if overflow:
        return {
            "overflow": True,
            "direct": direct,
            "fixed_scan": direct,
            "obi_replay": direct,
            "product": direct_issue,
            "delivery": direct_delivery,
            "obi_issue": direct_issue,
        }

    product = math.ceil(class_terms / product_engines)
    delivery = _ceil_efficiency(delivery_transactions, delivery_efficiency)
    # 每个活动slot最多增加一次切换开销；保守地不与首条term重叠。
    obi_issue = math.ceil(class_terms / obi_issue_width) + active_classes
    fixed_scan = max(class_slots * head_dim, product, delivery)
    obi_replay = max(obi_issue, product, delivery)
    if class_terms or delivery_transactions:
        fixed_scan += pipeline_fill
        obi_replay += pipeline_fill
    return {
        "overflow": False,
        "direct": direct,
        "fixed_scan": fixed_scan,
        "obi_replay": obi_replay,
        "product": product,
        "delivery": delivery,
        "obi_issue": obi_issue,
    }


def overlap_two_contexts(builds: list[int], executes: list[int]) -> int:
    """两个context的build/replay流水，序列首尾不免费。"""

    if len(builds) != len(executes):
        raise ValueError("build和execute序列长度不一致")
    if not builds:
        return 0
    total = builds[0]
    for index in range(len(builds) - 1):
        total += max(executes[index], builds[index + 1])
    return total + executes[-1]


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate(
    profile: dict[str, Any],
    *,
    class_slots: int = 4,
    tokens: int = 162,
    head_dim: int = 32,
    output_lanes: int = 32,
    product_engines: int = 1,
    multicast_width: int = 4,
    obi_issue_width: int = 1,
    accumulator_banks: int = 4,
    delivery_efficiency: float = 0.85,
    pipeline_fill: int = 4,
) -> dict[str, Any]:
    totals = {
        "windows": 0,
        "head_rows": 0,
        "overflow_rows": 0,
        "build_cycles": 0,
        "direct_single": 0,
        "fixed_single": 0,
        "obi_single": 0,
        "direct_dual": 0,
        "fixed_dual": 0,
        "obi_dual": 0,
        "direct_terms_all_tiles": 0,
        "selected_terms_all_tiles": 0,
        "fixed_scan_cells_all_tiles": 0,
        "obi_steps_all_tiles": 0,
    }
    stage_totals: dict[int, dict[str, int]] = {}
    direct_window_cycles: list[int] = []
    fixed_window_cycles: list[int] = []
    obi_window_cycles: list[int] = []

    for record in profile["summary"]["h60_records"]:
        heads = int(record["num_heads"])
        output_channels = heads * int(record["head_dim"])
        output_tiles = math.ceil(output_channels / output_lanes)
        active = decode_count_trace(
            record["projection_baseline_active_lanes_ordered_trace"]
        )
        terms = decode_count_trace(
            record["projection_gate_class_channel_terms_deploy_ordered_trace"]
        )
        classes = decode_count_trace(
            record["projection_active_gate_classes_deploy_ordered_trace"]
        )
        delivery = decode_count_trace(
            record[f"projection_gate_multicast_delivery_m{multicast_width}_ordered_trace"]
        )
        if not (len(active) == len(terms) == len(classes) == len(delivery)):
            raise ValueError("ordered trace长度不一致")
        if len(active) % heads:
            raise ValueError("ordered trace不能按窗口head数整组")

        stage = int(record["stage"])
        stage_row = stage_totals.setdefault(
            stage,
            {
                "windows": 0,
                "head_rows": 0,
                "overflow_rows": 0,
                "direct_dual": 0,
                "fixed_dual": 0,
                "obi_dual": 0,
            },
        )
        record_builds: list[int] = []
        record_direct: list[int] = []
        record_fixed: list[int] = []
        record_obi: list[int] = []
        tile_tail = math.ceil(tokens / accumulator_banks) * 2 + 2

        for base in range(0, len(active), heads):
            head_direct = 0
            head_fixed = 0
            head_obi = 0
            window_overflow = 0
            window_direct_terms = 0
            window_selected_terms = 0
            window_obi_steps = 0
            for offset in range(heads):
                index = base + offset
                row = head_backend_cycles(
                    active_lanes=active[index],
                    class_terms=terms[index],
                    active_classes=classes[index],
                    delivery_transactions=delivery[index],
                    class_slots=class_slots,
                    head_dim=head_dim,
                    product_engines=product_engines,
                    multicast_width=multicast_width,
                    obi_issue_width=obi_issue_width,
                    delivery_efficiency=delivery_efficiency,
                    pipeline_fill=pipeline_fill,
                )
                head_direct += int(row["direct"])
                head_fixed += int(row["fixed_scan"])
                head_obi += int(row["obi_replay"])
                overflow = int(bool(row["overflow"]))
                window_overflow += overflow
                window_direct_terms += active[index]
                window_selected_terms += active[index] if overflow else terms[index]
                window_obi_steps += (
                    active[index]
                    if overflow
                    else math.ceil(terms[index] / obi_issue_width) + classes[index]
                )

            build = heads * tokens
            execute_direct = output_tiles * (head_direct + tile_tail)
            execute_fixed = output_tiles * (head_fixed + tile_tail)
            execute_obi = output_tiles * (head_obi + tile_tail)
            record_builds.append(build)
            record_direct.append(execute_direct)
            record_fixed.append(execute_fixed)
            record_obi.append(execute_obi)
            direct_window_cycles.append(execute_direct)
            fixed_window_cycles.append(execute_fixed)
            obi_window_cycles.append(execute_obi)

            totals["windows"] += 1
            totals["head_rows"] += heads
            totals["overflow_rows"] += window_overflow
            totals["build_cycles"] += build
            totals["direct_terms_all_tiles"] += window_direct_terms * output_tiles
            totals["selected_terms_all_tiles"] += window_selected_terms * output_tiles
            totals["fixed_scan_cells_all_tiles"] += (
                class_slots * head_dim * heads * output_tiles
            )
            totals["obi_steps_all_tiles"] += window_obi_steps * output_tiles
            stage_row["windows"] += 1
            stage_row["head_rows"] += heads
            stage_row["overflow_rows"] += window_overflow

        direct_single = sum(b + e for b, e in zip(record_builds, record_direct))
        fixed_single = sum(b + e for b, e in zip(record_builds, record_fixed))
        obi_single = sum(b + e for b, e in zip(record_builds, record_obi))
        direct_dual = overlap_two_contexts(record_builds, record_direct)
        fixed_dual = overlap_two_contexts(record_builds, record_fixed)
        obi_dual = overlap_two_contexts(record_builds, record_obi)
        totals["direct_single"] += direct_single
        totals["fixed_single"] += fixed_single
        totals["obi_single"] += obi_single
        totals["direct_dual"] += direct_dual
        totals["fixed_dual"] += fixed_dual
        totals["obi_dual"] += obi_dual
        stage_row["direct_dual"] += direct_dual
        stage_row["fixed_dual"] += fixed_dual
        stage_row["obi_dual"] += obi_dual

    head_rows = totals["head_rows"]
    return {
        "parameters": {
            "class_slots": class_slots,
            "tokens": tokens,
            "head_dim": head_dim,
            "output_lanes": output_lanes,
            "product_engines": product_engines,
            "multicast_width": multicast_width,
            "obi_issue_width": obi_issue_width,
            "accumulator_banks": accumulator_banks,
            "delivery_efficiency": delivery_efficiency,
            "pipeline_fill": pipeline_fill,
        },
        "totals": totals,
        "speedups": {
            "fixed_dual_vs_direct_dual": totals["direct_dual"] / totals["fixed_dual"],
            "obi_single_vs_direct_single": totals["direct_single"] / totals["obi_single"],
            "obi_dual_vs_direct_dual": totals["direct_dual"] / totals["obi_dual"],
            "obi_dual_vs_fixed_dual": totals["fixed_dual"] / totals["obi_dual"],
        },
        "ratios": {
            "overflow": totals["overflow_rows"] / head_rows,
            "selected_term_reduction": 1.0
            - totals["selected_terms_all_tiles"] / totals["direct_terms_all_tiles"],
            "obi_step_reduction_vs_fixed": 1.0
            - totals["obi_steps_all_tiles"] / totals["fixed_scan_cells_all_tiles"],
        },
        "window_execute_percentiles": {
            "direct_p50": _percentile(direct_window_cycles, 0.50),
            "direct_p99": _percentile(direct_window_cycles, 0.99),
            "obi_p50": _percentile(obi_window_cycles, 0.50),
            "obi_p99": _percentile(obi_window_cycles, 0.99),
            "fixed_p99": _percentile(fixed_window_cycles, 0.99),
        },
        "stage_totals": stage_totals,
        "model_limits": [
            "按[B_windows,heads]行主序把连续head聚合为完整窗口",
            "direct和GateStack均允许两个输入context，基线不故意串行化",
            "delivery_efficiency是bank冲突和反压的敏感性参数，不是实测",
            "权重SRAM按吞吐1项/周期/engine抽象，未计容量和物理端口",
            "bias/requant按相同bank尾相计入，未计真实输出stall",
            "所有结果均为[prof]+[模型]，不是RTL或DC结果",
        ],
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack完整窗口Projection周期模型",
        "",
        f"输入：`{result['profile']}`。所有结果为 `[prof]+[模型]`。",
        "",
        "## 配置敏感性",
        "",
        "| delivery效率 | 窗口数 | overflow | fixed双context | OBI单context | OBI双context | OBI相对fixed |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        p = model["parameters"]
        t = model["totals"]
        s = model["speedups"]
        lines.append(
            f"| {p['delivery_efficiency']:.0%} | {t['windows']} | "
            f"{model['ratios']['overflow']:.4%} | {s['fixed_dual_vs_direct_dual']:.3f}x | "
            f"{s['obi_single_vs_direct_single']:.3f}x | {s['obi_dual_vs_direct_dual']:.3f}x | "
            f"{s['obi_dual_vs_fixed_dual']:.3f}x |"
        )
    chosen = result["models"][1]
    lines += [
        "",
        "## 默认85% delivery效率",
        "",
        f"- 完整窗口数：{chosen['totals']['windows']}；head row：{chosen['totals']['head_rows']}；",
        f"- exact fallback：{chosen['totals']['overflow_rows']} row，比例 {chosen['ratios']['overflow']:.6%}；",
        f"- 选择后term减少：{chosen['ratios']['selected_term_reduction']:.4%}；",
        f"- OBI访问相对固定128项扫描减少：{chosen['ratios']['obi_step_reduction_vs_fixed']:.4%}；",
        f"- OBI双context相对公平双context direct：{chosen['speedups']['obi_dual_vs_direct_dual']:.3f}x；",
        f"- OBI双context相对fixed-scan双context：{chosen['speedups']['obi_dual_vs_fixed_dual']:.3f}x；",
        "",
        "## 使用边界",
        "",
    ]
    lines.extend(f"- {item}；" for item in chosen["model_limits"])
    lines += [
        "",
        "OBI只有在完整RTL的priority iterator、SRAM端口和时序成本计入后仍通过1.20x淘汰线，才能晋级为架构贡献。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    models = [
        evaluate(profile, delivery_efficiency=efficiency)
        for efficiency in (1.0, 0.85, 0.70)
    ]
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "models": models,
        "evidence": "[prof ordered trace]+[完整窗口周期模型]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
