#!/usr/bin/env python3
"""All-Class Relation Transduction的SCS到term完整周期模型。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.model_fcip_equal_bandwidth import (
        paired_comparison,
        sink_ready,
        summarize,
    )
except ModuleNotFoundError:
    from model_fcip_equal_bandwidth import (
        paired_comparison,
        sink_ready,
        summarize,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "results/h67_fcip_real_trace_profile_20260730/report.json"
DEFAULT_OUT = ROOT / "results/acrt_full_pipeline_model_20260730"


def _gate_words(row: dict[str, Any], classes: list[int]) -> list[int]:
    words = [0] * int(row["segments"])
    for class_id in classes:
        for segment, word in enumerate(row["class_words"][str(class_id)]):
            words[segment] |= int(word)
    return words


def emit_cycles_from(
    start_cycle: int,
    terms: int,
    ready_percent: int,
) -> int:
    cycle = start_cycle
    emitted = 0
    while emitted < terms:
        if sink_ready(cycle, ready_percent):
            emitted += 1
        cycle += 1
    return cycle - start_cycle


def segment_major_intersection(
    row: dict[str, Any],
    *,
    lane_width: int,
    read_latency: int,
    ready_percent: int,
) -> dict[str, int]:
    """1个G segment广播到lane_width个K bank，按lane group装配term。"""

    cycle = 0
    gate_reads = 0
    k_reads = 0
    emitted_terms = 0
    groups = 0
    for classes in row["final_gate_groups"].values():
        gate_words = _gate_words(row, classes)
        for lane_start in range(0, int(row["lanes"]), lane_width):
            lane_words = row["k_words"][lane_start : lane_start + lane_width]
            segments = [
                segment
                for segment, gate_word in enumerate(gate_words)
                if int(gate_word) != 0
                and any(int(words[segment]) != 0 for words in lane_words)
            ]
            if not segments:
                continue
            groups += 1
            cycle += len(segments)
            gate_reads += len(segments)
            k_reads += len(segments) * len(lane_words)
            cycle += read_latency
            terms = sum(
                any(
                    int(gate_words[segment]) & int(words[segment])
                    for segment in segments
                )
                for words in lane_words
            )
            cycle += emit_cycles_from(cycle, terms, ready_percent)
            emitted_terms += terms
    if emitted_terms != int(row["final_gate_lane_terms"]):
        raise AssertionError(
            "segment-major intersection的final term不守恒："
            f"{emitted_terms} vs {row['final_gate_lane_terms']}"
        )
    return {
        "cycles": cycle,
        "gate_reads": gate_reads,
        "k_reads": k_reads,
        "emitted_terms": emitted_terms,
        "lane_groups": groups,
    }


def current_scs_g1_cycles(
    row: dict[str, Any],
    *,
    ready_percent: int,
    fold_pipeline_cycles: int = 2,
) -> dict[str, int]:
    """按当前h67_score_class_row_engine形状计算SCS到grouped term。"""

    active = int(row["active_tokens"])
    folded = fold_pipeline_cycles * int(row["kzero_score_classes"])
    term_emit = emit_cycles_from(
        0,
        int(row["final_gate_lane_terms"]),
        ready_percent,
    )
    return {
        "cycles": active + folded + active + term_emit,
        "sum_active": active,
        "sum_fold": folded,
        "emit_active_to_g1": active,
        "emit_terms": term_emit,
    }


def allclass_replay_cycles(
    row: dict[str, Any],
    *,
    replay_width: int,
    ready_percent: int,
    fold_pipeline_cycles: int = 2,
) -> dict[str, int]:
    """强中间基线：all-class denominator后仍按token构建常规G1。"""

    denominator = fold_pipeline_cycles * int(row["all_score_classes"])
    replay = math.ceil(int(row["active_tokens"]) / replay_width)
    term_emit = emit_cycles_from(
        denominator + replay,
        int(row["final_gate_lane_terms"]),
        ready_percent,
    )
    return {
        "cycles": denominator + replay + term_emit,
        "denominator": denominator,
        "replay": replay,
        "term_emit": term_emit,
    }


def sparse_prefix_cycles(
    row: dict[str, Any],
    *,
    ready_percent: int,
    fold_pipeline_cycles: int = 2,
) -> dict[str, int]:
    """有界prefix模式：当前式denominator后直接发singleton term。"""

    denominator = (
        int(row["active_tokens"])
        + fold_pipeline_cycles * int(row["kzero_score_classes"])
    )
    gatezero_scan = int(row["active_gatezero_tokens"])
    singleton_emit = emit_cycles_from(
        denominator,
        int(row["active_nonzero_gate_lane_events"]),
        ready_percent,
    )
    return {
        "cycles": denominator + gatezero_scan + singleton_emit,
        "denominator": denominator,
        "gatezero_scan": gatezero_scan,
        "singleton_emit": singleton_emit,
    }


def acrt_class_cycles(
    row: dict[str, Any],
    *,
    lane_width: int,
    read_latency: int,
    ready_percent: int,
    fold_pipeline_cycles: int = 2,
    gate_fold_drain: int = 1,
) -> dict[str, int]:
    """全class denominator + segment-distributed gate fold + G∩K。"""

    denominator = fold_pipeline_cycles * int(row["all_score_classes"])
    active_classes = int(row["active_score_classes"])
    gate_fold = active_classes + (gate_fold_drain if active_classes else 0)
    intersection = segment_major_intersection(
        row,
        lane_width=lane_width,
        read_latency=read_latency,
        ready_percent=ready_percent,
    )
    return {
        "cycles": denominator + gate_fold + intersection["cycles"],
        "denominator": denominator,
        "gate_fold": gate_fold,
        "intersection": intersection["cycles"],
        "gate_reads": intersection["gate_reads"],
        "k_reads": intersection["k_reads"],
    }


def aenr_cycles(
    row: dict[str, Any],
    *,
    event_threshold: int,
    lane_width: int,
    read_latency: int,
    ready_percent: int,
) -> dict[str, int | str]:
    if int(row["active_lane_events"]) <= event_threshold:
        sparse = sparse_prefix_cycles(
            row,
            ready_percent=ready_percent,
        )
        return {
            "cycles": sparse["cycles"],
            "mode": "singleton",
        }
    dense = acrt_class_cycles(
        row,
        lane_width=lane_width,
        read_latency=read_latency,
        ready_percent=ready_percent,
    )
    return {
        "cycles": dense["cycles"],
        "mode": "class",
    }


def payload_bits(
    tokens: int,
    *,
    prefix_events: int,
    class_slots: int = 16,
    gate_slots: int = 4,
    lanes: int = 32,
    contexts: int = 4,
) -> dict[str, int]:
    prefix_entry_bits = 8 + 9 + lanes
    return {
        "b1_g4_lane_directory_only": gate_slots * lanes * tokens,
        "acrt_relation_peak": (
            (class_slots + gate_slots + lanes + contexts) * tokens
        ),
        "aenr_prefix": prefix_events * prefix_entry_bits,
        "aenr_relation_plus_prefix": (
            (class_slots + gate_slots + lanes + contexts) * tokens
            + prefix_events * prefix_entry_bits
        ),
    }


def model(profile: dict[str, Any]) -> dict[str, Any]:
    rows = profile["rows"]
    configurations = []
    for ready_percent in (100, 90, 75):
        baseline = [
            current_scs_g1_cycles(
                row,
                ready_percent=ready_percent,
            )["cycles"]
            for row in rows
        ]
        allclass_replay_w1 = [
            allclass_replay_cycles(
                row,
                replay_width=1,
                ready_percent=ready_percent,
            )["cycles"]
            for row in rows
        ]
        allclass_replay_w4 = [
            allclass_replay_cycles(
                row,
                replay_width=4,
                ready_percent=ready_percent,
            )["cycles"]
            for row in rows
        ]
        class_records = [
            acrt_class_cycles(
                row,
                lane_width=4,
                read_latency=1,
                ready_percent=ready_percent,
            )
            for row in rows
        ]
        class_cycles = [int(record["cycles"]) for record in class_records]
        threshold_rows = []
        for threshold in (0, 4, 8, 12, 16, 20, 24, 32, 48):
            records = [
                aenr_cycles(
                    row,
                    event_threshold=threshold,
                    lane_width=4,
                    read_latency=1,
                    ready_percent=ready_percent,
                )
                for row in rows
            ]
            cycles = [int(record["cycles"]) for record in records]
            threshold_rows.append(
                {
                    "event_threshold": threshold,
                    "cycles": summarize(cycles),
                    "vs_current": paired_comparison(baseline, cycles),
                    "vs_allclass_replay_w4": paired_comparison(
                        allclass_replay_w4,
                        cycles,
                    ),
                    "singleton_rows": sum(
                        record["mode"] == "singleton" for record in records
                    ),
                }
            )
        configurations.append(
            {
                "ready_percent": ready_percent,
                "current_cycles": summarize(baseline),
                "allclass_replay_w1_cycles": summarize(allclass_replay_w1),
                "allclass_replay_w4_cycles": summarize(allclass_replay_w4),
                "acrt_class_cycles": summarize(class_cycles),
                "acrt_class_vs_current": paired_comparison(
                    baseline,
                    class_cycles,
                ),
                "acrt_class_vs_allclass_replay_w1": paired_comparison(
                    allclass_replay_w1,
                    class_cycles,
                ),
                "acrt_class_vs_allclass_replay_w4": paired_comparison(
                    allclass_replay_w4,
                    class_cycles,
                ),
                "threshold_sweep": threshold_rows,
                "acrt_gate_reads": sum(
                    int(record["gate_reads"]) for record in class_records
                ),
                "acrt_k_reads": sum(
                    int(record["k_reads"]) for record in class_records
                ),
            }
        )
    tokens = max(int(row["tokens"]) for row in rows)
    return {
        "schema": "acrt_full_pipeline_cycle_model_v1",
        "evidence": (
            "[真实网络bit trace]+[现有SCS FSM形状]+[保守segment-major模型]；"
            "不是RTL、DC、SAIF或fullres统计"
        ),
        "rows": len(rows),
        "tokens": tokens,
        "configurations": configurations,
        "payload_bits": {
            f"e{threshold}": payload_bits(
                tokens,
                prefix_events=threshold,
            )
            for threshold in (4, 20, 32)
        },
        "physical_contract": [
            "当前基线按active-token denominator、2-cycle K-zero class fold、"
            "active-token emit/G1 build和单term sink分账。",
            "强B1/B2先采用相同all-class denominator，再分别以W1/W4 active-token"
            "replay构建常规G1；W4是周期上界，端口/面积尚未综合。",
            "ACRT把active与K-zero统一为all-class denominator；第二遍每拍一个"
            "active class，并由3个T162 segment bank本地并行折叠到G4。",
            "projection严格segment-major：1个G word广播、4个独立K-lane bank读、"
            "4个T-bit context；一组term发完后才处理下一lane group。",
            "AENR只按LOAD期可知的原始K lane-event计数缓存有界prefix；超过阈值"
            "立即丢弃prefix并进入class模式，"
            "不实例化完整B1目录。",
        ],
        "limits": [
            "一个样本、45行、T162；threshold sweep只能发现敏感性，不能冻结阈值。",
            "现有SCS RTL尚未实现all-class histogram、第二遍gate fold或AENR。",
            "segment bank按寄存器或SRAM实现后的端口、Fmax和能耗未知。",
            "回压为周期模式，尚无跨row burst FIFO仿真。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ACRT 全链周期模型",
        "",
        f"- rows：{report['rows']}",
        f"- 证据：{report['evidence']}",
        "",
        "## ACRT class-only",
        "",
        "| ready | current mean | ACRT mean | aggregate speedup | "
        "paired p99 slowdown | >10%慢行 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        comparison = config["acrt_class_vs_current"]
        lines.append(
            f"| {config['ready_percent']}% | "
            f"{config['current_cycles']['mean']:.2f} | "
            f"{config['acrt_class_cycles']['mean']:.2f} | "
            f"{comparison['aggregate_speedup']:.3f}x | "
            f"{comparison['paired_slowdown']['p99']:.3f}x | "
            f"{comparison['rows_over_10pct_slower_ratio']:.1%} |"
        )
    lines += [
        "",
        "## 强中间基线",
        "",
        "| ready | all-class replay W1 | all-class replay W4 | ACRT | "
        "ACRT/W1 | ACRT/W4 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        versus_w1 = config["acrt_class_vs_allclass_replay_w1"]
        versus_w4 = config["acrt_class_vs_allclass_replay_w4"]
        lines.append(
            f"| {config['ready_percent']}% | "
            f"{config['allclass_replay_w1_cycles']['mean']:.2f} | "
            f"{config['allclass_replay_w4_cycles']['mean']:.2f} | "
            f"{config['acrt_class_cycles']['mean']:.2f} | "
            f"{versus_w1['aggregate_speedup']:.3f}x | "
            f"{versus_w4['aggregate_speedup']:.3f}x |"
        )
    lines += [
        "",
        "## AENR event-prefix阈值扫描",
        "",
        "| ready | E | singleton rows | vs current | vs strong W4 | "
        "paired p99 slowdown(vs W4) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        for row in config["threshold_sweep"]:
            comparison = row["vs_current"]
            strong = row["vs_allclass_replay_w4"]
            lines.append(
                f"| {config['ready_percent']}% | "
                f"{row['event_threshold']} | {row['singleton_rows']} | "
                f"{comparison['aggregate_speedup']:.3f}x | "
                f"{strong['aggregate_speedup']:.3f}x | "
                f"{strong['paired_slowdown']['p99']:.3f}x |"
            )
    lines += [
        "",
        "## Payload",
        "",
        "| prefix | B1 G4×L×T | ACRT relation | prefix | AENR total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in report["payload_bits"].items():
        lines.append(
            f"| {name} | {payload['b1_g4_lane_directory_only']} | "
            f"{payload['acrt_relation_peak']} | {payload['aenr_prefix']} | "
            f"{payload['aenr_relation_plus_prefix']} |"
        )
    lines += ["", "## 物理合同", ""]
    lines.extend(f"- {item}" for item in report["physical_contract"])
    lines += ["", "## 边界", ""]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = model(profile)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
