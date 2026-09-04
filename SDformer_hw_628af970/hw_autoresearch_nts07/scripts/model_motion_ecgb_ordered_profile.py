#!/usr/bin/env python3
"""用H67 profile100 ordered数组评估Motion跨行ECGB有限流水。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .analyze_hit_flow_ordered_profiles import decode_count_trace, percentile
except ImportError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace, percentile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_OUT = ROOT / "results/motion_ecgb_ordered_profile100_20260801"


def clog2(value: int) -> int:
    return max(1, math.ceil(math.log2(max(2, value))))


def pingpong_cycles(build: list[int], execute: list[int]) -> int:
    if len(build) != len(execute) or not build:
        raise ValueError("build/execute列表不合法")
    build_finish: list[int] = []
    execute_finish: list[int] = []
    for index, (build_cycles, execute_cycles) in enumerate(zip(build, execute)):
        prior_build = build_finish[-1] if build_finish else 0
        reused_buffer_free = execute_finish[index - 2] if index >= 2 else 0
        build_start = max(prior_build, reused_buffer_free)
        build_finish.append(build_start + build_cycles)
        execute_start = max(
            build_finish[-1], execute_finish[-1] if execute_finish else 0
        )
        execute_finish.append(execute_start + execute_cycles)
    return execute_finish[-1]


def group_payload_bits(
    *, terms: int, active_lanes: int, dim: int, tokens: int, windows: int
) -> int:
    """term目录+destination token list的窄表示下界，不含SRAM宏对齐。"""

    pointer_bits = clog2(active_lanes + 1)
    token_bits = clog2(tokens * windows)
    term_bits = clog2(dim) + 9 + 2 * pointer_bits
    return terms * term_bits + active_lanes * token_bits


def finalization_cycles(tokens: int, windows: int, mode: str) -> int:
    """返回每组窗口的bias/final尾部周期。

    commit_rmw与现有RTL一致：两个bank交替接收token，每个token形成一次
    accumulator读改写。ibf_pipelined保留逐token读出，但两个bank各自可每拍
    退休一个token，并计入3拍启动/收尾。lower_bound只用于标出理论上界。
    """

    if mode == "commit_rmw":
        per_window = tokens + 2
    elif mode == "ibf_pipelined":
        per_window = math.ceil(tokens / 2) + 3
    elif mode == "bias_free_lower_bound":
        per_window = 2
    else:
        raise ValueError(f"未知finalization模式: {mode}")
    return per_window * windows


def evaluate(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("samples") != 100 or not profile.get("ordered_trace"):
        raise ValueError("需要H67 ordered profile100")
    records = profile["summary"]["h60_records"]
    if len(records) != 1200:
        raise ValueError("H67 profile应有1200条attention记录")
    results = []
    for group_windows in (1, 2, 4, 8, 16):
        stage_cycles_by_finalizer = {
            mode: [0, 0, 0, 0]
            for mode in (
                "commit_rmw",
                "ibf_pipelined",
                "bias_free_lower_bound",
            )
        }
        stage_payloads: list[list[int]] = [[], [], [], []]
        terms_total = 0
        delivery_total = 0
        active_lanes_total = 0
        windows_total = 0
        for record in records:
            prefix = f"projection_gate_group_"
            terms = decode_count_trace(
                record[f"{prefix}terms_g{group_windows}_ordered_trace"]
            )
            active = decode_count_trace(
                record[f"{prefix}active_lanes_g{group_windows}_ordered_trace"]
            )
            windows = decode_count_trace(
                record[f"{prefix}window_count_g{group_windows}_ordered_trace"]
            )
            delivery = decode_count_trace(
                record[
                    f"{prefix}delivery_g{group_windows}_m4_ordered_trace"
                ]
            )
            if not (len(terms) == len(active) == len(windows) == len(delivery)):
                raise ValueError("Motion group ordered数组长度不一致")
            tokens = int(record["tokens"])
            heads = int(record["num_heads"])
            dim = heads * int(record["head_dim"])
            build = [tokens * count for count in windows]
            stage = int(record["stage"])
            for mode, stage_cycles in stage_cycles_by_finalizer.items():
                execute = [
                    heads * max(term, deliver)
                    + finalization_cycles(tokens, count, mode)
                    for term, deliver, count in zip(terms, delivery, windows)
                ]
                stage_cycles[stage] += pingpong_cycles(build, execute)
            stage_payloads[stage].extend(
                group_payload_bits(
                    terms=term,
                    active_lanes=lane_count,
                    dim=dim,
                    tokens=tokens,
                    windows=count,
                )
                for term, lane_count, count in zip(terms, active, windows)
            )
            terms_total += sum(terms)
            delivery_total += sum(delivery)
            active_lanes_total += sum(active)
            windows_total += sum(windows)
        results.append(
            {
                "group_windows": group_windows,
                "terms": terms_total,
                "delivery_m4": delivery_total,
                "active_lanes": active_lanes_total,
                "windows": windows_total,
                "finite_pingpong_cycles": sum(
                    stage_cycles_by_finalizer["commit_rmw"]
                ),
                "stage_cycles": stage_cycles_by_finalizer["commit_rmw"],
                "finite_cycles_by_finalizer": {
                    mode: sum(stage_cycles)
                    for mode, stage_cycles in stage_cycles_by_finalizer.items()
                },
                "stage_cycles_by_finalizer": stage_cycles_by_finalizer,
                "stage_payload_bits": [
                    {
                        "stage": stage,
                        "groups": len(values),
                        "mean": sum(values) / len(values),
                        "p95": percentile(values, 0.95),
                        "p99": percentile(values, 0.99),
                        "max": max(values),
                    }
                    for stage, values in enumerate(stage_payloads)
                ],
            }
        )
    baseline = results[0]
    baseline_current = baseline["finite_cycles_by_finalizer"]["commit_rmw"]
    for row in results:
        row["term_reduction_vs_g1"] = 1.0 - row["terms"] / baseline["terms"]
        row["delivery_reduction_vs_g1"] = (
            1.0 - row["delivery_m4"] / baseline["delivery_m4"]
        )
        row["cycle_reduction_vs_g1"] = (
            1.0
            - row["finite_pingpong_cycles"]
            / baseline["finite_pingpong_cycles"]
        )
        row["speedup_vs_g1"] = (
            baseline["finite_pingpong_cycles"]
            / row["finite_pingpong_cycles"]
        )
        row["ibf_speedup_vs_current_g1"] = (
            baseline_current
            / row["finite_cycles_by_finalizer"]["ibf_pipelined"]
        )
        row["ibf_speedup_vs_ibf_g1"] = (
            baseline["finite_cycles_by_finalizer"]["ibf_pipelined"]
            / row["finite_cycles_by_finalizer"]["ibf_pipelined"]
        )
    best = min(results, key=lambda row: row["finite_pingpong_cycles"])
    ibf_best = min(
        results,
        key=lambda row: row["finite_cycles_by_finalizer"]["ibf_pipelined"],
    )
    return {
        "schema": "motion_ecgb_ordered_profile100_v1",
        "profile": str(profile_path.resolve()),
        "evidence": "H67 crop/W9 ordered profile100 + bounded finite pipeline",
        "groups": results,
        "best_group_windows": best["group_windows"],
        "best_speedup_vs_g1": best["speedup_vs_g1"],
        "best_cycle_reduction_vs_g1": best["cycle_reduction_vs_g1"],
        "ibf_best_group_windows": ibf_best["group_windows"],
        "ibf_best_speedup_vs_current_g1": ibf_best[
            "ibf_speedup_vs_current_g1"
        ],
        "model_contract": {
            "builder": "T*valid_windows",
            "executor": "heads*max(terms,delivery_m4)+(T+2)*valid_windows",
            "ibf_executor": (
                "heads*max(terms,delivery_m4)+(ceil(T/2)+3)*valid_windows"
            ),
            "buffers": 2,
            "output_tile": 32,
            "storage": "term directory + explicit destination token list lower bound",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Motion ECGB Ordered Profile100 有限流水评估",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：`[prof-ordered] + [bounded-model]`；输入为旧crop/W9 T=162，"
        "不是fullres/W15、RTL或PPA。",
        "",
        "## 结论",
        "",
        f"现有bias读改写下，G={report['best_group_windows']} 的有限双buffer周期最小，相对G=1为 "
        f"{report['best_speedup_vs_g1']:.4f}x，周期下降 "
        f"{report['best_cycle_reduction_vs_g1']:.2%}。虽然term减少更大，但build、"
        "bias/final和destination delivery限制了实际收益。",
        f"若采用双bank流水IBF，G={report['ibf_best_group_windows']} 的模型周期最小，"
        f"相对现有G1为 {report['ibf_best_speedup_vs_current_g1']:.4f}x。该值仍包含逐token"
        "final drain，不是删除bias阶段的理想估计。",
        "",
        "| G | terms | term减少 | M4 delivery | delivery减少 | 有限周期 | speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['group_windows']} | {row['terms']} | "
            f"{row['term_reduction_vs_g1']:.2%} | {row['delivery_m4']} | "
            f"{row['delivery_reduction_vs_g1']:.2%} | "
            f"{row['finite_pingpong_cycles']} | {row['speedup_vs_g1']:.4f}x |"
        )
    lines += [
        "",
        "## IBF尾部解锁敏感性",
        "",
        "`commit_rmw`是现有读加写；`ibf_pipelined`是两个bank各自每拍退休一个token，"
        "尾部为`ceil(T/2)+3`；lower bound仅标上界。",
        "",
        "| G | commit_rmw | ibf_pipelined | lower bound | IBF相对现有G1 | IBF内相对G1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        variants = row["finite_cycles_by_finalizer"]
        lines.append(
            f"| {row['group_windows']} | {variants['commit_rmw']} | "
            f"{variants['ibf_pipelined']} | {variants['bias_free_lower_bound']} | "
            f"{row['ibf_speedup_vs_current_g1']:.4f}x | "
            f"{row['ibf_speedup_vs_ibf_g1']:.4f}x |"
        )
    lines += [
        "",
        "## Stage 周期",
        "",
        "| G | S0 | S1 | S2 | S3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        values = row["stage_cycles"]
        lines.append(
            f"| {row['group_windows']} | {values[0]} | {values[1]} | "
            f"{values[2]} | {values[3]} |"
        )
    lines += [
        "",
        "## 窄目录存储下界",
        "",
        "以下是单buffer payload；双buffer需乘2，且尚未计SRAM宏对齐和控制。",
        "",
        "| G | stage | mean bit | p95 bit | p99 bit | max bit |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        for stage in row["stage_payload_bits"]:
            lines.append(
                f"| {row['group_windows']} | {stage['stage']} | "
                f"{stage['mean']:.0f} | {stage['p95']:.0f} | "
                f"{stage['p99']:.0f} | {stage['max']:.0f} |"
            )
    lines += [
        "",
        "## 架构决定",
        "",
        "1. 现有尾部下Motion不把G>1 ECGB做成主配置；G=8只有约6.3%周期下降，"
        "S0 p99 payload却从G1约11.5 Kbit增至约95.7 Kbit；",
        "2. IBF使G4/G8重新具备复审价值；先以G4作为容量保守点、G8作为性能点，"
        "但只有IBF叶RTL证明双bank 2 token/cycle后才能晋级；",
        "3. Motion当前已实现主配置仍是G=1 + DVCO，模型候选不能替代RTL结论；",
        "4. 本结果不能外推到fullres T=450，需新的W15 ordered profile；",
        "5. ECGB优先在Local5推进，Motion作为共享后端兼容模式。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = evaluate(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
