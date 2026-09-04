#!/usr/bin/env python3
"""回放Local5 v2 ordered source-frontier trace，评估FCSR FIFO与周期。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.et3_ordered_trace_replay import file_sha256, load_trace
    from scripts.model_local5_frontier_retirement import (
        simulate_plane_serial_frontier,
        simulate_plane_serial_stripe,
        simulate_plane_serial_two_phase,
        stripe_retirement_events,
    )
except ModuleNotFoundError:
    from et3_ordered_trace_replay import file_sha256, load_trace
    from model_local5_frontier_retirement import (
        simulate_plane_serial_frontier,
        simulate_plane_serial_stripe,
        simulate_plane_serial_two_phase,
        stripe_retirement_events,
    )


ROOT = Path(__file__).resolve().parents[1]


def quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * probability) - 1)
    return float(ordered[index])


def trace_group_source_work(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    group_index: int,
) -> tuple[list[list[int]], list[int]]:
    required = {
        "source_group_offsets",
        "source_delivery_count",
        "source_retire_destination",
        "source_service_cycles_pipelined",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(
            "trace缺少FCSR v2数组: " + ", ".join(sorted(missing))
        )
    source_offsets = arrays["source_group_offsets"]
    if len(source_offsets) != len(manifest["groups"]) + 1:
        raise ValueError("source_group_offsets长度与group数量不一致")
    start = int(source_offsets[group_index])
    end = int(source_offsets[group_index + 1])
    tokens = int(manifest["groups"][group_index]["tokens"])
    if end - start != tokens:
        raise ValueError("source trace长度与group token数不一致")

    work = (
        arrays["source_service_cycles_pipelined"][start:end]
        .astype(np.int64)
        .tolist()
    )
    retire = (
        arrays["source_retire_destination"][start:end]
        .astype(np.int64)
        .tolist()
    )
    events: list[list[int]] = [[] for _ in range(tokens)]
    for source, destination in enumerate(retire):
        if not 0 <= destination < tokens:
            raise ValueError("source retire destination越界")
        events[destination].append(source)
    if sorted(source for event in events for source in event) != list(
        range(tokens)
    ):
        raise ValueError("source retirement不守恒")
    return events, work


def trace_group_destination_cycles(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    group_index: int,
    *,
    mode: str,
) -> list[int]:
    if mode == "direct":
        array_name = "destination_direct_score_cycles"
    elif mode == "independent_w1x4":
        array_name = "destination_independent_w1x4_score_cycles"
    elif mode in {
        "qfsa_w2",
        "qfsa_w4",
        "qfsa_w8",
        "qfsa_xb4",
        "qfsa_xb4_t4",
        "qfsa_xb4_t8",
        "qfsa_xb4_t8b2",
        "qfsa_xb4_t12",
    }:
        array_name = f"destination_{mode}_score_cycles"
    else:
        raise ValueError(f"未知score mode: {mode}")
    if array_name not in arrays:
        raise ValueError(f"trace缺少{array_name}")
    offsets = arrays["source_group_offsets"]
    start = int(offsets[group_index])
    end = int(offsets[group_index + 1])
    values = arrays[array_name][start:end].astype(np.int64).tolist()
    tokens = int(manifest["groups"][group_index]["tokens"])
    if len(values) != tokens or any(value <= 0 for value in values):
        raise ValueError(f"{array_name}长度或数值非法")
    return values


def summarize_rows(rows: list[dict[str, int]]) -> dict[str, float | int]:
    if not rows:
        return {
            "groups": 0,
            "cycles_mean": 0.0,
            "cycles_p95": 0.0,
            "cycles_p99": 0.0,
        }
    return {
        "groups": len(rows),
        "cycles_mean": sum(row["cycles"] for row in rows) / len(rows),
        "cycles_p95": quantile(
            [row["cycles"] for row in rows],
            0.95,
        ),
        "cycles_p99": quantile(
            [row["cycles"] for row in rows],
            0.99,
        ),
        "terms_mean": sum(row["terms"] for row in rows) / len(rows),
        "stalls_mean": sum(
            row.get("producer_stalls", 0) for row in rows
        )
        / len(rows),
        "stalls_p99": quantile(
            [row.get("producer_stalls", 0) for row in rows],
            0.99,
        ),
        "max_fifo_sources": max(
            row.get("max_fifo_sources", 0) for row in rows
        ),
        "max_fifo_terms": max(
            row.get("max_fifo_terms", 0) for row in rows
        ),
        "max_stripe_owned_rows": max(
            row.get("max_stripe_owned_rows", 0) for row in rows
        ),
    }


def replay(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    fifo_depths: tuple[int, ...] = (3, 4, 8, 16),
    ready_percents: tuple[int, ...] = (100, 90, 75),
) -> dict[str, Any]:
    groups = len(manifest["groups"])
    output: dict[str, Any] = {
        "schema": "local5_frontier_ordered_replay_v1",
        "source_trace_schema": manifest["schema"],
        "evidence_level": manifest["evidence_level"],
        "groups": groups,
        "configs": {},
    }
    for ready in ready_percents:
        baseline_rows = []
        group_work = []
        for index in range(groups):
            events, work = trace_group_source_work(
                manifest,
                arrays,
                index,
            )
            direct_cycles = trace_group_destination_cycles(
                manifest,
                arrays,
                index,
                mode="direct",
            )
            qfsa_cycles = trace_group_destination_cycles(
                manifest,
                arrays,
                index,
                mode="qfsa_w4",
            )
            xorbank_cycles = trace_group_destination_cycles(
                manifest,
                arrays,
                index,
                mode="qfsa_xb4_t8b2",
            )
            independent_cycles = trace_group_destination_cycles(
                manifest,
                arrays,
                index,
                mode="independent_w1x4",
            )
            tokens = len(events)
            plane_tokens = tokens // 2
            side = math.isqrt(plane_tokens)
            if 2 * side * side != tokens:
                raise ValueError("Local5 stripe replay要求T=2方形空间窗口")
            stripe_events = stripe_retirement_events(side, side)
            group_work.append(
                (
                    events,
                    stripe_events,
                    work,
                    direct_cycles,
                    independent_cycles,
                    qfsa_cycles,
                    xorbank_cycles,
                    side,
                )
            )
            baseline_rows.append(
                simulate_plane_serial_two_phase(
                    source_work=work,
                    plane_tokens=plane_tokens,
                    ready_percent=ready,
                    destination_cycles=direct_cycles,
                )
            )
        baseline = summarize_rows(baseline_rows)
        for fifo_depth in fifo_depths:
            independent_two_phase_rows = [
                simulate_plane_serial_two_phase(
                    source_work=work,
                    plane_tokens=len(events) // 2,
                    ready_percent=ready,
                    destination_cycles=independent_cycles,
                )
                for (
                    events,
                    _,
                    work,
                    _,
                    independent_cycles,
                    _,
                    _,
                    _,
                ) in group_work
            ]
            qfsa_two_phase_rows = [
                simulate_plane_serial_two_phase(
                    source_work=work,
                    plane_tokens=len(events) // 2,
                    ready_percent=ready,
                    destination_cycles=qfsa_cycles,
                )
                for events, _, work, _, _, qfsa_cycles, _, _ in group_work
            ]
            xorbank_two_phase_rows = [
                simulate_plane_serial_two_phase(
                    source_work=work,
                    plane_tokens=len(events) // 2,
                    ready_percent=ready,
                    destination_cycles=xorbank_cycles,
                )
                for events, _, work, _, _, _, xorbank_cycles, _ in group_work
            ]
            direct_frontier_rows = [
                simulate_plane_serial_frontier(
                    events,
                    work,
                    plane_tokens=len(events) // 2,
                    fifo_depth=fifo_depth,
                    ready_percent=ready,
                    destination_cycles=direct_cycles,
                )
                for events, _, work, direct_cycles, _, _, _, _ in group_work
            ]
            independent_stripe_rows = [
                simulate_plane_serial_stripe(
                    work,
                    height=side,
                    width=side,
                    ready_percent=ready,
                    destination_cycles=independent_cycles,
                )
                for (
                    _,
                    stripe_events,
                    work,
                    _,
                    independent_cycles,
                    _,
                    _,
                    side,
                ) in group_work
            ]
            qfsa_stripe_rows = [
                simulate_plane_serial_stripe(
                    work,
                    height=side,
                    width=side,
                    ready_percent=ready,
                    destination_cycles=qfsa_cycles,
                )
                for (
                    _,
                    stripe_events,
                    work,
                    _,
                    _,
                    qfsa_cycles,
                    _,
                    side,
                ) in group_work
            ]
            xorbank_stripe_rows = [
                simulate_plane_serial_stripe(
                    work,
                    height=side,
                    width=side,
                    ready_percent=ready,
                    destination_cycles=xorbank_cycles,
                )
                for (
                    _,
                    _,
                    work,
                    _,
                    _,
                    _,
                    xorbank_cycles,
                    side,
                ) in group_work
            ]
            independent_frontier_rows = [
                simulate_plane_serial_frontier(
                    events,
                    work,
                    plane_tokens=len(events) // 2,
                    fifo_depth=fifo_depth,
                    ready_percent=ready,
                    destination_cycles=independent_cycles,
                )
                for (
                    events,
                    _,
                    work,
                    _,
                    independent_cycles,
                    _,
                    _,
                    _,
                ) in group_work
            ]
            combined_rows = [
                simulate_plane_serial_frontier(
                    events,
                    work,
                    plane_tokens=len(events) // 2,
                    fifo_depth=fifo_depth,
                    ready_percent=ready,
                    destination_cycles=qfsa_cycles,
                )
                for events, _, work, _, _, qfsa_cycles, _, _ in group_work
            ]
            xorbank_frontier_rows = [
                simulate_plane_serial_frontier(
                    events,
                    work,
                    plane_tokens=len(events) // 2,
                    fifo_depth=fifo_depth,
                    ready_percent=ready,
                    destination_cycles=xorbank_cycles,
                )
                for events, _, work, _, _, _, xorbank_cycles, _ in group_work
            ]
            independent_two_phase = summarize_rows(
                independent_two_phase_rows
            )
            qfsa_two_phase = summarize_rows(qfsa_two_phase_rows)
            xorbank_two_phase = summarize_rows(xorbank_two_phase_rows)
            direct_frontier = summarize_rows(direct_frontier_rows)
            independent_stripe = summarize_rows(independent_stripe_rows)
            qfsa_stripe = summarize_rows(qfsa_stripe_rows)
            xorbank_stripe = summarize_rows(xorbank_stripe_rows)
            independent_frontier = summarize_rows(
                independent_frontier_rows
            )
            combined = summarize_rows(combined_rows)
            xorbank_combined = summarize_rows(xorbank_frontier_rows)
            output["configs"][f"fifo{fifo_depth}_ready{ready}"] = {
                "direct_two_phase": baseline,
                "independent_w1x4_two_phase": independent_two_phase,
                "qfsa_two_phase": qfsa_two_phase,
                "qfsa_xb4_two_phase": xorbank_two_phase,
                "direct_frontier": direct_frontier,
                "independent_w1x4_stripe": independent_stripe,
                "qfsa_stripe": qfsa_stripe,
                "qfsa_xb4_stripe": xorbank_stripe,
                "independent_w1x4_frontier": independent_frontier,
                "qfsa_frontier": combined,
                "qfsa_xb4_frontier": xorbank_combined,
                "xorbank_vs_independent_speedup_mean": (
                    independent_two_phase["cycles_mean"]
                    / xorbank_two_phase["cycles_mean"]
                    if xorbank_two_phase["cycles_mean"]
                    else 0.0
                ),
                "xorbank_combined_vs_strong_speedup_mean": (
                    independent_stripe["cycles_mean"]
                    / xorbank_combined["cycles_mean"]
                    if xorbank_combined["cycles_mean"]
                    else 0.0
                ),
                "cdrp_vs_independent_speedup_mean": (
                    independent_two_phase["cycles_mean"]
                    / qfsa_two_phase["cycles_mean"]
                    if qfsa_two_phase["cycles_mean"]
                    else 0.0
                ),
                "fcsr_vs_stripe_speedup_mean": (
                    qfsa_stripe["cycles_mean"] / combined["cycles_mean"]
                    if combined["cycles_mean"]
                    else 0.0
                ),
                "combined_vs_strong_speedup_mean": (
                    independent_stripe["cycles_mean"]
                    / combined["cycles_mean"]
                    if combined["cycles_mean"]
                    else 0.0
                ),
                "combined_speedup_p95_ratio": (
                    baseline["cycles_p95"] / combined["cycles_p95"]
                    if combined["cycles_p95"]
                    else 0.0
                ),
            }
    return output


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local5 FCSR Ordered Trace 回放",
        "",
        f"- trace evidence：`{report['evidence_level']}`",
        f"- sampled window-head groups：{report['groups']}",
        "- 服务量下界：逐 source 的 `max(product term, destination delivery)`，而非只数产品 term。",
        "- T0/T1 强制平面串行：T0 的 score、写回和 FIFO 全部排空后才进入 T1。",
        "- Stripe 强基线在行内增量建表，采用双row ping-pong；行末只交换ownership，consumer与producer可并行。",
        "- Stripe与FCSR下游均为单service/cycle，不把同步snapshot开销计入Stripe。",
        "",
        "| 配置 | Direct两阶段 | 4xW1两阶段 | 全局QFSA两阶段 | XBF-T8两阶段 | 4xW1+Stripe | 全局QFSA+Stripe | XBF-T8+Stripe | 4xW1+FCSR | 全局QFSA+FCSR | XBF-T8+FCSR | XBF-T8/4xW1 | XBF-T8联合/强基线 | stall p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["configs"].items():
        baseline = row["direct_two_phase"]
        independent = row["independent_w1x4_two_phase"]
        qfsa = row["qfsa_two_phase"]
        xorbank = row["qfsa_xb4_two_phase"]
        independent_stripe = row["independent_w1x4_stripe"]
        qfsa_stripe = row["qfsa_stripe"]
        xorbank_stripe = row["qfsa_xb4_stripe"]
        independent_frontier = row["independent_w1x4_frontier"]
        combined = row["qfsa_frontier"]
        xorbank_combined = row["qfsa_xb4_frontier"]
        lines.append(
            f"| {name} | {baseline['cycles_mean']:.2f} | "
            f"{independent['cycles_mean']:.2f} | {qfsa['cycles_mean']:.2f} | "
            f"{xorbank['cycles_mean']:.2f} | "
            f"{independent_stripe['cycles_mean']:.2f} | "
            f"{qfsa_stripe['cycles_mean']:.2f} | "
            f"{xorbank_stripe['cycles_mean']:.2f} | "
            f"{independent_frontier['cycles_mean']:.2f} | "
            f"{combined['cycles_mean']:.2f} | "
            f"{xorbank_combined['cycles_mean']:.2f} | "
            f"{row['xorbank_vs_independent_speedup_mean']:.3f}× | "
            f"{row['xorbank_combined_vs_strong_speedup_mean']:.3f}× | "
            f"{xorbank_combined['stalls_p99']:.0f} |"
        )
    lines.extend(
        [
            "",
            "该回放仍是周期模型，不包含 SRAM macro PPA、MFEP 跨 source 产品目录命中、",
            "Acc bank conflict 或完整 encoder 调度。只有 post-G0/fullres trace 可进入论文主表。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, arrays = load_trace(args.manifest)
    report = replay(manifest, arrays)
    report["manifest"] = str(args.manifest.resolve())
    report["manifest_sha256"] = file_sha256(args.manifest)
    report["run_identity_file_sha256"] = manifest.get(
        "run_identity_file_sha256"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
