#!/usr/bin/env python3
"""FCIP与强基线的同带宽、有限上下文逐拍模型。"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "results/h67_fcip_real_trace_profile_20260730/report.json"
DEFAULT_OUT = ROOT / "results/fcip_equal_bandwidth_model_20260730"


def ceil_div(value: int, width: int) -> int:
    return (value + width - 1) // width


def sink_ready(cycle: int, ready_percent: int) -> bool:
    if ready_percent == 100:
        return True
    if ready_percent == 90:
        return cycle % 10 != 9
    if ready_percent == 75:
        return cycle % 4 != 3
    raise ValueError("仅支持100/90/75百分比ready")


def sink_service_cycles(terms: int, ready_percent: int) -> int:
    emitted = 0
    cycles = 0
    while emitted < terms:
        if sink_ready(cycles, ready_percent):
            emitted += 1
        cycles += 1
    return cycles


def simulate_bounded_assembler(
    tasks: list[dict[str, int | bool]],
    *,
    contexts: int,
    read_width: int,
    read_latency: int,
    ready_percent: int,
) -> dict[str, int]:
    """模拟有限term上下文、共享读端口和单term ready/valid出口。"""

    if contexts <= 0 or read_width <= 0 or read_latency <= 0:
        raise ValueError("context/read width/read latency必须为正")
    pending = deque(dict(task) for task in tasks)
    slots: list[dict[str, int | bool] | None] = [None] * contexts
    cycle = 0
    emitted = 0
    empty_retired = 0
    reads = 0
    sink_stalls = 0
    max_occupancy = 0

    while pending or any(slot is not None for slot in slots):
        # 读响应到达后，空term不占出口即可释放。
        for index, slot in enumerate(slots):
            if (
                slot is not None
                and int(slot["remaining"]) == 0
                and int(slot["ready_cycle"]) <= cycle
                and not bool(slot["produces_term"])
            ):
                slots[index] = None
                empty_retired += 1

        completed = [
            index
            for index, slot in enumerate(slots)
            if slot is not None
            and int(slot["remaining"]) == 0
            and int(slot["ready_cycle"]) <= cycle
            and bool(slot["produces_term"])
        ]
        if completed:
            if sink_ready(cycle, ready_percent):
                slots[completed[0]] = None
                emitted += 1
            else:
                sink_stalls += 1

        for index, slot in enumerate(slots):
            if slot is None and pending:
                task = pending.popleft()
                work = int(task["read_work"])
                if work <= 0:
                    raise ValueError("assembler task必须至少读取一个fragment")
                slots[index] = {
                    "remaining": work,
                    "ready_cycle": -1,
                    "produces_term": bool(task["produces_term"]),
                }

        issued = 0
        for slot in slots:
            if (
                slot is not None
                and int(slot["remaining"]) > 0
                and issued < read_width
            ):
                slot["remaining"] = int(slot["remaining"]) - 1
                reads += 1
                issued += 1
                if int(slot["remaining"]) == 0:
                    slot["ready_cycle"] = cycle + read_latency

        max_occupancy = max(
            max_occupancy,
            sum(slot is not None for slot in slots),
        )
        cycle += 1
        if cycle > 10_000_000:
            raise RuntimeError("assembler模型疑似未收敛")

    expected_terms = sum(bool(task["produces_term"]) for task in tasks)
    if emitted != expected_terms:
        raise AssertionError("assembler term发射不守恒")
    return {
        "cycles": cycle,
        "reads": reads,
        "emitted_terms": emitted,
        "empty_retired": empty_retired,
        "sink_stalls": sink_stalls,
        "max_context_occupancy": max_occupancy,
    }


def _gate_words(row: dict[str, Any], classes: list[int]) -> list[int]:
    segments = int(row["segments"])
    words = [0] * segments
    for class_id in classes:
        for segment, word in enumerate(row["class_words"][str(class_id)]):
            words[segment] |= int(word)
    return words


def _fcip_gate_tasks(
    row: dict[str, Any],
    classes: list[int],
) -> tuple[list[dict[str, int | bool]], int]:
    gate_words = _gate_words(row, classes)
    tasks: list[dict[str, int | bool]] = []
    false_candidates = 0
    for lane_words in row["k_words"]:
        candidate_segments = [
            segment
            for segment, (gate_word, lane_word) in enumerate(
                zip(gate_words, lane_words)
            )
            if int(gate_word) != 0 and int(lane_word) != 0
        ]
        if not candidate_segments:
            continue
        produces = any(
            int(gate_words[segment]) & int(lane_words[segment])
            for segment in candidate_segments
        )
        tasks.append(
            {
                "read_work": len(candidate_segments),
                "produces_term": produces,
            }
        )
        if not produces:
            false_candidates += 1
    return tasks, false_candidates


def _b2_gate_tasks(
    row: dict[str, Any],
    classes: list[int],
) -> list[dict[str, int | bool]]:
    tasks: list[dict[str, int | bool]] = []
    for lane_words in row["k_words"]:
        reads = 0
        produces = False
        for class_id in classes:
            class_words = row["class_words"][str(class_id)]
            for class_word, lane_word in zip(class_words, lane_words):
                relation = int(class_word) & int(lane_word)
                if relation:
                    reads += 1
                    produces = True
        if produces:
            tasks.append(
                {
                    "read_work": reads,
                    "produces_term": True,
                }
            )
    return tasks


def b1_cycles(
    row: dict[str, Any],
    *,
    ingress_width: int,
    ready_percent: int,
) -> dict[str, int]:
    build = ceil_div(int(row["active_tokens"]), ingress_width)
    emit = sink_service_cycles(
        int(row["final_gate_lane_terms"]),
        ready_percent,
    )
    return {"cycles": build + emit, "build": build, "emit": emit}


def singleton_event_cycles(
    row: dict[str, Any],
    *,
    ready_percent: int,
) -> int:
    """将每个非零gated-K lane event作为单目的位图term直接发射。"""

    return sink_service_cycles(
        int(row["active_nonzero_gate_lane_events"]),
        ready_percent,
    )


def relation_plane_cycles(
    row: dict[str, Any],
    *,
    architecture: str,
    ingress_width: int,
    read_width: int,
    contexts: int,
    read_latency: int,
    ready_percent: int,
    active_class_slots: int = 16,
) -> dict[str, int | bool]:
    build = ceil_div(int(row["active_tokens"]), ingress_width)
    if int(row["active_score_classes"]) > active_class_slots:
        replay = b1_cycles(
            row,
            ingress_width=ingress_width,
            ready_percent=ready_percent,
        )
        return {
            "cycles": build + 1 + replay["cycles"],
            "build": build,
            "fallback": True,
            "fallback_replay": replay["cycles"],
            "union_cycles": 0,
            "assembler_cycles": 0,
            "reads": 0,
            "false_candidates": 0,
            "emitted_terms": int(row["final_gate_lane_terms"]),
        }

    union_cycles = 0
    assembler_cycles = 0
    reads = 0
    false_candidates = 0
    emitted_terms = 0
    for classes in row["final_gate_groups"].values():
        if architecture == "fcip":
            class_fragment_reads = sum(
                int(word) != 0
                for class_id in classes
                for word in row["class_words"][str(class_id)]
            )
            union_cycles += ceil_div(class_fragment_reads, read_width)
            tasks, false_count = _fcip_gate_tasks(row, classes)
            false_candidates += false_count
        elif architecture == "b2":
            tasks = _b2_gate_tasks(row, classes)
        else:
            raise ValueError("architecture必须为fcip或b2")
        service = simulate_bounded_assembler(
            tasks,
            contexts=contexts,
            read_width=read_width,
            read_latency=read_latency,
            ready_percent=ready_percent,
        )
        assembler_cycles += service["cycles"]
        reads += service["reads"]
        emitted_terms += service["emitted_terms"]

    if emitted_terms != int(row["final_gate_lane_terms"]):
        raise AssertionError(
            f"{architecture} final term不守恒："
            f"{emitted_terms} vs {row['final_gate_lane_terms']}"
        )
    return {
        "cycles": build + union_cycles + assembler_cycles,
        "build": build,
        "fallback": False,
        "fallback_replay": 0,
        "union_cycles": union_cycles,
        "assembler_cycles": assembler_cycles,
        "reads": reads,
        "false_candidates": false_candidates,
        "emitted_terms": emitted_terms,
    }


def ncfip_cycles(
    row: dict[str, Any],
    *,
    ingress_width: int,
    read_width: int,
    contexts: int,
    read_latency: int,
    ready_percent: int,
    transduction_overlapped: bool,
    active_class_slots: int = 16,
) -> dict[str, int | bool]:
    """归一化耦合FIP：SCS扫描class时折叠出final-gate平面。"""

    if int(row["active_score_classes"]) > active_class_slots:
        replay = b1_cycles(
            row,
            ingress_width=ingress_width,
            ready_percent=ready_percent,
        )
        return {
            "cycles": 1 + replay["cycles"],
            "fallback": True,
            "transduction_cycles": 0,
            "assembler_cycles": 0,
            "reads": 0,
            "false_candidates": 0,
            "emitted_terms": int(row["final_gate_lane_terms"]),
        }

    transduction_cycles = 0
    assembler_cycles = 0
    reads = 0
    false_candidates = 0
    emitted_terms = 0
    for classes in row["final_gate_groups"].values():
        class_fragment_reads = sum(
            int(word) != 0
            for class_id in classes
            for word in row["class_words"][str(class_id)]
        )
        transduction_cycles += ceil_div(class_fragment_reads, read_width)
        tasks, false_count = _fcip_gate_tasks(row, classes)
        false_candidates += false_count
        service = simulate_bounded_assembler(
            tasks,
            contexts=contexts,
            read_width=read_width,
            read_latency=read_latency,
            ready_percent=ready_percent,
        )
        assembler_cycles += service["cycles"]
        reads += service["reads"]
        emitted_terms += service["emitted_terms"]
    if emitted_terms != int(row["final_gate_lane_terms"]):
        raise AssertionError("NC-FIP final term不守恒")
    paid_transduction = 0 if transduction_overlapped else transduction_cycles
    return {
        "cycles": paid_transduction + assembler_cycles,
        "fallback": False,
        "transduction_cycles": transduction_cycles,
        "assembler_cycles": assembler_cycles,
        "reads": reads,
        "false_candidates": false_candidates,
        "emitted_terms": emitted_terms,
    }


def summarize(values: list[float | int]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def paired_comparison(
    baseline: list[int],
    candidate: list[int],
) -> dict[str, Any]:
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ratio = np.where(
        base > 0,
        cand / np.maximum(base, 1.0),
        np.where(cand == 0, 1.0, cand),
    )
    delta = cand - base
    aggregate = (
        1.0
        if base.sum() == 0 and cand.sum() == 0
        else float(base.sum() / max(cand.sum(), 1.0))
    )
    return {
        "aggregate_speedup": aggregate,
        "paired_slowdown": summarize(ratio.tolist()),
        "paired_delta_cycles": summarize(delta.tolist()),
        "rows_faster_ratio": float(np.mean(cand < base)),
        "rows_over_10pct_slower_ratio": float(np.mean(ratio > 1.10)),
    }


def relation_payload_bits(
    tokens: int,
    *,
    class_slots: int,
    lanes: int,
    contexts: int,
) -> dict[str, int]:
    return {
        "b1_final_gate_directory_g4": 4 * lanes * tokens,
        "b2_materialized_class_lane": class_slots * lanes * tokens,
        "fcip_factor_planes_plus_gate_buffer_and_contexts": (
            (class_slots + lanes + 1 + contexts) * tokens
        ),
        "ncfip_peak_class_k_gate_contexts": (
            (class_slots + lanes + 4 + contexts) * tokens
        ),
        "ncfip_postnorm_k_gate_contexts": (
            (lanes + 4 + contexts) * tokens
        ),
    }


def _stage_summary(
    rows: list[dict[str, Any]],
    name: str,
    *,
    active_class_slots: int,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["name"]) == name]
    baseline = [
        b1_cycles(
            row,
            ingress_width=4,
            ready_percent=100,
        )["cycles"]
        for row in selected
    ]
    explicit = [
        int(
            ncfip_cycles(
                row,
                ingress_width=4,
                read_width=4,
                contexts=4,
                read_latency=1,
                ready_percent=100,
                transduction_overlapped=False,
                active_class_slots=active_class_slots,
            )["cycles"]
        )
        for row in selected
    ]
    fused = [
        int(
            ncfip_cycles(
                row,
                ingress_width=4,
                read_width=4,
                contexts=4,
                read_latency=1,
                ready_percent=100,
                transduction_overlapped=True,
                active_class_slots=active_class_slots,
            )["cycles"]
        )
        for row in selected
    ]
    drms_e4_explicit = [
        singleton_event_cycles(row, ready_percent=100)
        if int(row["active_tokens"]) <= 4
        else candidate
        for row, candidate in zip(selected, explicit)
    ]
    drms_e4_fused = [
        singleton_event_cycles(row, ready_percent=100)
        if int(row["active_tokens"]) <= 4
        else candidate
        for row, candidate in zip(selected, fused)
    ]
    return {
        "rows": len(selected),
        "b1": summarize(baseline),
        "ncfip_explicit": summarize(explicit),
        "ncfip_fused": summarize(fused),
        "drms_e4_explicit_vs_b1": paired_comparison(
            baseline,
            drms_e4_explicit,
        ),
        "drms_e4_fused_vs_b1": paired_comparison(
            baseline,
            drms_e4_fused,
        ),
    }


def model(
    profile: dict[str, Any],
    *,
    active_class_slots: int = 16,
) -> dict[str, Any]:
    rows = profile["rows"]
    configurations = []
    for ready_percent in (100, 90, 75):
        for width, contexts in ((1, 2), (4, 4)):
            b1 = [
                b1_cycles(
                    row,
                    ingress_width=width,
                    ready_percent=ready_percent,
                )["cycles"]
                for row in rows
            ]
            b2_records = [
                relation_plane_cycles(
                    row,
                    architecture="b2",
                    ingress_width=width,
                    read_width=width,
                    contexts=contexts,
                    read_latency=1,
                    ready_percent=ready_percent,
                    active_class_slots=active_class_slots,
                )
                for row in rows
            ]
            fcip_records = [
                relation_plane_cycles(
                    row,
                    architecture="fcip",
                    ingress_width=width,
                    read_width=width,
                    contexts=contexts,
                    read_latency=1,
                    ready_percent=ready_percent,
                    active_class_slots=active_class_slots,
                )
                for row in rows
            ]
            ncfip_explicit_records = [
                ncfip_cycles(
                    row,
                    ingress_width=width,
                    read_width=width,
                    contexts=contexts,
                    read_latency=1,
                    ready_percent=ready_percent,
                    transduction_overlapped=False,
                    active_class_slots=active_class_slots,
                )
                for row in rows
            ]
            ncfip_fused_records = [
                ncfip_cycles(
                    row,
                    ingress_width=width,
                    read_width=width,
                    contexts=contexts,
                    read_latency=1,
                    ready_percent=ready_percent,
                    transduction_overlapped=True,
                    active_class_slots=active_class_slots,
                )
                for row in rows
            ]
            b2 = [int(record["cycles"]) for record in b2_records]
            fcip = [int(record["cycles"]) for record in fcip_records]
            ncfip_explicit = [
                int(record["cycles"]) for record in ncfip_explicit_records
            ]
            ncfip_fused = [
                int(record["cycles"]) for record in ncfip_fused_records
            ]
            # active-token计数在K metadata阶段已精确可知，不使用未来term数。
            # 极稀疏行直接把lane event发为singleton term，不实例化B1目录。
            drms_e4_explicit = [
                singleton_event_cycles(row, ready_percent=ready_percent)
                if int(row["active_tokens"]) <= 4
                else ncfip_cycle
                for row, ncfip_cycle in zip(
                    rows,
                    ncfip_explicit,
                )
            ]
            drms_e4_fused = [
                singleton_event_cycles(row, ready_percent=ready_percent)
                if int(row["active_tokens"]) <= 4
                else ncfip_cycle
                for row, ncfip_cycle in zip(
                    rows,
                    ncfip_fused,
                )
            ]
            configurations.append(
                {
                    "width": width,
                    "contexts": contexts,
                    "ready_percent": ready_percent,
                    "b1_cycles": summarize(b1),
                    "b2_cycles": summarize(b2),
                    "fcip_cycles": summarize(fcip),
                    "ncfip_explicit_cycles": summarize(ncfip_explicit),
                    "ncfip_fused_cycles": summarize(ncfip_fused),
                    "drms_e4_explicit_cycles": summarize(drms_e4_explicit),
                    "drms_e4_fused_cycles": summarize(drms_e4_fused),
                    "b2_vs_b1": paired_comparison(b1, b2),
                    "fcip_vs_b1": paired_comparison(b1, fcip),
                    "fcip_vs_b2": paired_comparison(b2, fcip),
                    "ncfip_explicit_vs_b1": paired_comparison(
                        b1,
                        ncfip_explicit,
                    ),
                    "ncfip_fused_vs_b1": paired_comparison(
                        b1,
                        ncfip_fused,
                    ),
                    "drms_e4_explicit_vs_b1": paired_comparison(
                        b1,
                        drms_e4_explicit,
                    ),
                    "drms_e4_fused_vs_b1": paired_comparison(
                        b1,
                        drms_e4_fused,
                    ),
                    "fcip_false_candidates": int(
                        sum(
                            int(record["false_candidates"])
                            for record in fcip_records
                        )
                    ),
                    "fcip_reads": int(
                        sum(int(record["reads"]) for record in fcip_records)
                    ),
                    "b2_reads": int(
                        sum(int(record["reads"]) for record in b2_records)
                    ),
                    "fallback_rows": int(
                        sum(bool(record["fallback"]) for record in fcip_records)
                    ),
                }
            )
    tokens = max(int(row["tokens"]) for row in rows)
    source_scope = dict(profile.get("source_scope", {}))
    scope_text = str(profile.get("evidence", "[真实网络bit trace]"))
    return {
        "schema": "fcip_equal_bandwidth_cycle_model_v1",
        "evidence": (
            f"{scope_text}+[CPU逐拍有限资源模型]；"
            "不是RTL、DC、STA、SAIF或PTPX"
        ),
        "source_profile": str(DEFAULT_PROFILE),
        "source_scope": source_scope,
        "active_class_slots": active_class_slots,
        "rows": len(rows),
        "tokens": tokens,
        "configurations": configurations,
        "per_stage_w4_ready100": {
            name: _stage_summary(
                rows,
                name,
                active_class_slots=active_class_slots,
            )
            for name in sorted({str(row["name"]) for row in rows})
        },
        "relation_payload_bits": {
            "contexts2": relation_payload_bits(
                tokens,
                class_slots=active_class_slots,
                lanes=32,
                contexts=2,
            ),
            "contexts4": relation_payload_bits(
                tokens,
                class_slots=active_class_slots,
                lanes=32,
                contexts=4,
            ),
        },
        "contracts": [
            "B1-Wx与FCIP-Wx使用相同x-token ingress和单term出口。",
            f"B2物化S{active_class_slots} score-class×lane×destination，"
            "FCIP仅物化两个因子平面。",
            "FCIP先构建单个final-gate bitmap，再以2/4个上下文逐lane求交。",
            "上下文持有完整T-bit term；无完整gate×lane目录。",
            "不同final gate串行；未假设alias、intersection、emit理想max重叠。",
            "FCIP只用gate-segment与K-lane-segment occupancy跳过确定为空的读，"
            "不使用class×lane oracle。",
            f"超过S{active_class_slots}时支付factor build、abort和B1整行replay。",
            "NC-FIP在score/classify阶段构建C与K平面；SCS class扫描把C折叠为"
            "G4 final-gate平面，投影阶段不再active-token replay。",
            "NC-FIP explicit支付class-fragment转导周期；fused仅表示这些读被"
            "SCS occupied-class扫描完全覆盖的上界，二者共同给出实现窗口。",
            "DRMS-E4只使用行开始前已知的active-token计数：不超过4时从小型"
            "{class,token,K-mask}寄存器枚举singleton term，其余走NC-FIP；"
            "不实例化B1 G4目录，也不是事后按真实周期取min的oracle。",
        ],
        "limits": [
            f"输入范围严格继承source profile：{scope_text}",
            "B1-W4需要四token合并写能力，其面积/端口代价尚未综合。",
            "segment builder、SRAM宏、地址译码和clock gating能耗尚未建模。",
            "若source profile仅含单样本，则不能外推多样本p95/p99。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FCIP 同带宽有限上下文模型",
        "",
        f"- 行数：{report['rows']}",
        f"- 证据：{report['evidence']}",
        "",
        "| W | ctx | ready | B1 mean | FCIP mean | NC explicit | NC fused | "
        "NC explicit/B1 | NC fused/B1 | fused p99 slowdown |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        explicit = config["ncfip_explicit_vs_b1"]
        fused = config["ncfip_fused_vs_b1"]
        lines.append(
            f"| {config['width']} | {config['contexts']} | "
            f"{config['ready_percent']}% | "
            f"{config['b1_cycles']['mean']:.2f} | "
            f"{config['fcip_cycles']['mean']:.2f} | "
            f"{config['ncfip_explicit_cycles']['mean']:.2f} | "
            f"{config['ncfip_fused_cycles']['mean']:.2f} | "
            f"{explicit['aggregate_speedup']:.3f}x | "
            f"{fused['aggregate_speedup']:.3f}x | "
            f"{fused['paired_slowdown']['p99']:.3f}x |"
        )
    lines += [
        "",
        "## DRMS-E4 精确模式切换",
        "",
        "| W | ready | explicit aggregate | explicit p99 slowdown | "
        "fused aggregate | fused p99 slowdown | fused >10%慢行 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        explicit = config["drms_e4_explicit_vs_b1"]
        fused = config["drms_e4_fused_vs_b1"]
        lines.append(
            f"| {config['width']} | {config['ready_percent']}% | "
            f"{explicit['aggregate_speedup']:.3f}x | "
            f"{explicit['paired_slowdown']['p99']:.3f}x | "
            f"{fused['aggregate_speedup']:.3f}x | "
            f"{fused['paired_slowdown']['p99']:.3f}x | "
            f"{fused['rows_over_10pct_slower_ratio']:.1%} |"
        )
    lines += [
        "",
        "## W4/ready100 分stage",
        "",
        "| stage | rows | B1 mean | NC explicit mean | NC fused mean | "
        "DRMS explicit/B1 | DRMS fused/B1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stage in report["per_stage_w4_ready100"].items():
        lines.append(
            f"| {name} | {stage['rows']} | {stage['b1']['mean']:.2f} | "
            f"{stage['ncfip_explicit']['mean']:.2f} | "
            f"{stage['ncfip_fused']['mean']:.2f} | "
            f"{stage['drms_e4_explicit_vs_b1']['aggregate_speedup']:.3f}x | "
            f"{stage['drms_e4_fused_vs_b1']['aggregate_speedup']:.3f}x |"
        )
    lines += [
        "",
        "## 关系payload",
        "",
        "仅比较关系位图与有限上下文，不等同于面积。",
        "",
        f"| 配置 | B1 G4目录 | B2 S{report['active_class_slots']}联合平面 | FCIP | "
        "NC-FIP peak | NC-FIP postnorm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, payload in report["relation_payload_bits"].items():
        lines.append(
            f"| {name} | {payload['b1_final_gate_directory_g4']} | "
            f"{payload['b2_materialized_class_lane']} | "
            f"{payload['fcip_factor_planes_plus_gate_buffer_and_contexts']} | "
            f"{payload['ncfip_peak_class_k_gate_contexts']} | "
            f"{payload['ncfip_postnorm_k_gate_contexts']} |"
        )
    lines += ["", "## 模型合同", ""]
    lines.extend(f"- {item}" for item in report["contracts"])
    lines += ["", "## 边界", ""]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--active-class-slots", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if args.active_class_slots <= 0:
        raise SystemExit("--active-class-slots must be positive")
    report = model(profile, active_class_slots=args.active_class_slots)
    report["source_profile"] = str(args.profile)
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
