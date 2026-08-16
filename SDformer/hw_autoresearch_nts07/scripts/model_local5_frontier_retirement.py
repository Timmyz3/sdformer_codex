#!/usr/bin/env python3
"""Local5固定拓扑的frontier-complete源退休离散事件模型。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT
    / "results/local5_hardware_profile_preG0_profile100_20260726"
    / "local5_hardware_features.json"
)
DEFAULT_OUT = ROOT / "results/local5_frontier_retirement_model_20260730"


def sink_ready(cycle: int, ready_percent: int) -> bool:
    if ready_percent <= 0 or ready_percent > 100:
        raise ValueError("ready_percent必须位于1..100")
    return (cycle % 100) < ready_percent


def source_consumers(height: int, width: int, y: int, x: int) -> list[int]:
    """返回会读取源(y,x)的Local5 destination raster索引。"""

    consumers = [(y, x)]
    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= yy < height and 0 <= xx < width:
            consumers.append((yy, xx))
    return sorted(yy * width + xx for yy, xx in consumers)


def retirement_events(
    height: int,
    width: int,
    time_planes: int = 2,
) -> list[list[int]]:
    """每个destination完成后可退休的全局source ID列表。"""

    plane_tokens = height * width
    events: list[list[int]] = [[] for _ in range(plane_tokens * time_planes)]
    for time_id in range(time_planes):
        base = time_id * plane_tokens
        for y in range(height):
            for x in range(width):
                source = base + y * width + x
                latest = base + max(source_consumers(height, width, y, x))
                events[latest].append(source)
    return events


def closed_form_retirement_events(
    height: int,
    width: int,
    time_planes: int = 2,
) -> list[list[int]]:
    """不用引用计数或查表，由raster坐标直接生成精确source退休。"""

    plane_tokens = height * width
    events: list[list[int]] = [[] for _ in range(plane_tokens * time_planes)]
    for time_id in range(time_planes):
        base = time_id * plane_tokens
        for destination_y in range(height):
            for destination_x in range(width):
                destination = (
                    base + destination_y * width + destination_x
                )
                if destination_y > 0:
                    events[destination].append(
                        base
                        + (destination_y - 1) * width
                        + destination_x
                    )
                if destination_y == height - 1 and destination_x > 0:
                    events[destination].append(
                        base
                        + destination_y * width
                        + destination_x
                        - 1
                    )
                if (
                    destination_y == height - 1
                    and destination_x == width - 1
                ):
                    events[destination].append(
                        base + destination_y * width + destination_x
                    )
    if events != retirement_events(height, width, time_planes):
        raise AssertionError("闭式退休公式与消费者枚举不一致")
    return events


def stripe_retirement_events(
    height: int,
    width: int,
    time_planes: int = 2,
) -> list[list[int]]:
    """强R1基线：整source row在下一consumer row结束后成批可见。"""

    plane_tokens = height * width
    events: list[list[int]] = [[] for _ in range(plane_tokens * time_planes)]
    for time_id in range(time_planes):
        base = time_id * plane_tokens
        for source_y in range(height):
            release_y = min(height - 1, source_y + 1)
            release_destination = base + (release_y + 1) * width - 1
            events[release_destination].extend(
                base + source_y * width + x
                for x in range(width)
            )
    retired = [source for event in events for source in event]
    if sorted(retired) != list(range(plane_tokens * time_planes)):
        raise AssertionError("stripe退休必须覆盖每个source且恰好一次")
    return events


def frontier_geometry(
    height: int,
    width: int,
    time_planes: int = 2,
) -> dict[str, int | list[int]]:
    events = closed_form_retirement_events(
        height,
        width,
        time_planes,
    )
    bursts = [len(event) for event in events]
    retired = [source for event in events for source in event]
    tokens = height * width * time_planes
    if sorted(retired) != list(range(tokens)):
        raise AssertionError("frontier退休必须覆盖每个source且恰好一次")
    return {
        "tokens": tokens,
        "event_cycles": sum(bool(event) for event in events),
        "max_retire_burst": max(bursts, default=0),
        "retire_burst_histogram": [
            sum(burst == size for burst in bursts)
            for size in range(max(bursts, default=0) + 1)
        ],
        "warmup_destinations": next(
            (index for index, event in enumerate(events) if event),
            tokens,
        ),
        "gate_ring_rows": 3,
    }


def allocate_terms(
    total_terms: int,
    events: list[list[int]],
    scenario: str,
    max_terms_per_source: int = 160,
) -> list[int]:
    """仅在缺少ordered profile时构造有明确边界含义的source work。"""

    tokens = sum(len(event) for event in events)
    work = [0] * tokens
    retirement_order = [source for event in events for source in event]
    if total_terms > tokens * max_terms_per_source:
        raise ValueError("term数量超过Local5 source容量上界")
    if total_terms == 0:
        return work

    if scenario == "uniform":
        # 令非零source尽量均匀分布于source ID空间，避免人为前置或后置。
        for term in range(total_terms):
            source = min(tokens - 1, (term * tokens) // total_terms)
            if work[source] >= max_terms_per_source:
                source = next(
                    index
                    for index in range(tokens)
                    if work[index] < max_terms_per_source
                )
            work[source] += 1
        return work

    if scenario not in {"front_loaded", "tail_loaded"}:
        raise ValueError(f"未知scenario: {scenario}")
    order = (
        retirement_order
        if scenario == "front_loaded"
        else list(reversed(retirement_order))
    )
    remaining = total_terms
    for source in order:
        assigned = min(max_terms_per_source, remaining)
        work[source] = assigned
        remaining -= assigned
        if remaining == 0:
            break
    if remaining:
        raise AssertionError("term分配未守恒")
    return work


def simulate_two_phase(
    *,
    tokens: int,
    total_terms: int,
    ready_percent: int,
    destination_cycles: list[int] | None = None,
    start_cycle: int = 0,
) -> dict[str, int]:
    """强基线：同样II=1 score producer，窗口结束后同后端发term。"""

    if destination_cycles is None:
        destination_cycles = [1] * tokens
    if len(destination_cycles) != tokens or any(
        value <= 0 for value in destination_cycles
    ):
        raise ValueError("destination_cycles必须为每token一个正整数")
    score_cycles = sum(destination_cycles)
    cycle = start_cycle + score_cycles
    terms = 0
    while terms < total_terms:
        if sink_ready(cycle, ready_percent):
            terms += 1
        cycle += 1
    return {
        "cycles": cycle - start_cycle,
        "score_cycles": score_cycles,
        "projection_cycles": cycle - start_cycle - score_cycles,
        "terms": total_terms,
    }


def simulate_frontier(
    events: list[list[int]],
    source_work: list[int],
    *,
    fifo_depth: int,
    ready_percent: int,
    retire_width: int = 1,
    destination_cycles: list[int] | None = None,
    start_cycle: int = 0,
) -> dict[str, int]:
    """score producer与单term/cycle投影后端重叠，FIFO按source descriptor计数。"""

    if fifo_depth < 3:
        raise ValueError("FIFO深度至少覆盖固定拓扑最大退休突发3")
    if retire_width <= 0:
        raise ValueError("retire_width必须为正")
    tokens = len(events)
    if len(source_work) != tokens:
        raise ValueError("source_work长度必须等于token数")
    if destination_cycles is None:
        destination_cycles = [1] * tokens
    if len(destination_cycles) != tokens or any(
        value <= 0 for value in destination_cycles
    ):
        raise ValueError("destination_cycles必须为每token一个正整数")

    fifo: list[int] = []
    destination = 0
    cycle = start_cycle
    emitted = 0
    producer_stalls = 0
    max_fifo_sources = 0
    max_fifo_terms = 0
    total_terms = sum(source_work)
    pending_arrivals: list[int] = []
    producer_remaining = destination_cycles[0] if tokens else 0

    while destination < tokens or pending_arrivals or fifo:
        # 先消费可使本周期同拍退休获得空间；新退休项下周期才可执行。
        if fifo and sink_ready(cycle, ready_percent):
            fifo[0] -= 1
            emitted += 1
            if fifo[0] == 0:
                fifo.pop(0)

        if pending_arrivals:
            issue = min(
                retire_width,
                len(pending_arrivals),
                fifo_depth - len(fifo),
            )
            if issue:
                fifo.extend(pending_arrivals[:issue])
                del pending_arrivals[:issue]
            if pending_arrivals:
                producer_stalls += 1
            else:
                destination += 1
                if destination < tokens:
                    producer_remaining = destination_cycles[destination]
        elif destination < tokens:
            producer_remaining -= 1
            if producer_remaining == 0:
                pending_arrivals = [
                    source_work[source]
                    for source in events[destination]
                    if source_work[source] > 0
                ]
                issue = min(
                    retire_width,
                    len(pending_arrivals),
                    fifo_depth - len(fifo),
                )
                if issue:
                    fifo.extend(pending_arrivals[:issue])
                    del pending_arrivals[:issue]
                if pending_arrivals:
                    producer_stalls += 1
                else:
                    destination += 1
                    if destination < tokens:
                        producer_remaining = destination_cycles[destination]

        max_fifo_sources = max(max_fifo_sources, len(fifo))
        max_fifo_terms = max(max_fifo_terms, sum(fifo))
        cycle += 1
        if cycle - start_cycle > 10_000_000:
            raise RuntimeError("frontier模型未收敛")

    if emitted != total_terms:
        raise AssertionError("frontier term不守恒")
    return {
        "cycles": cycle - start_cycle,
        "terms": emitted,
        "producer_stalls": producer_stalls,
        "max_fifo_sources": max_fifo_sources,
        "max_fifo_terms": max_fifo_terms,
        "producer_work_cycles": sum(destination_cycles),
    }


def _split_plane_events(
    events: list[list[int]],
    plane_tokens: int,
) -> list[list[list[int]]]:
    if plane_tokens <= 0 or len(events) % plane_tokens:
        raise ValueError("plane_tokens必须正好划分event序列")
    planes: list[list[list[int]]] = []
    for plane_start in range(0, len(events), plane_tokens):
        plane_events: list[list[int]] = []
        for destination in range(plane_start, plane_start + plane_tokens):
            local_sources = []
            for source in events[destination]:
                if not plane_start <= source < plane_start + plane_tokens:
                    raise ValueError("Local5 plane-serial事件包含跨平面source")
                local_sources.append(source - plane_start)
            plane_events.append(local_sources)
        retired = sorted(
            source for destination in plane_events for source in destination
        )
        if retired != list(range(plane_tokens)):
            raise ValueError("每个时间平面必须恰好退休所有source")
        planes.append(plane_events)
    return planes


def simulate_plane_serial_two_phase(
    *,
    source_work: list[int],
    plane_tokens: int,
    ready_percent: int,
    destination_cycles: list[int],
) -> dict[str, int]:
    """T0完成score和写回并排空后，才允许T1进入同一物理资源。"""

    if len(source_work) != len(destination_cycles):
        raise ValueError("source_work与destination_cycles长度不一致")
    if plane_tokens <= 0 or len(source_work) % plane_tokens:
        raise ValueError("plane_tokens必须正好划分source_work")
    totals = {
        "cycles": 0,
        "score_cycles": 0,
        "projection_cycles": 0,
        "terms": 0,
    }
    absolute_cycle = 0
    for start in range(0, len(source_work), plane_tokens):
        result = simulate_two_phase(
            tokens=plane_tokens,
            total_terms=sum(source_work[start : start + plane_tokens]),
            ready_percent=ready_percent,
            destination_cycles=destination_cycles[
                start : start + plane_tokens
            ],
            start_cycle=absolute_cycle,
        )
        absolute_cycle += result["cycles"]
        for key in totals:
            totals[key] += result[key]
    return totals


def simulate_plane_serial_frontier(
    events: list[list[int]],
    source_work: list[int],
    *,
    plane_tokens: int,
    fifo_depth: int,
    ready_percent: int,
    retire_width: int = 1,
    destination_cycles: list[int] | None = None,
) -> dict[str, int]:
    """逐时间平面运行frontier；边界处FIFO必须完全排空。"""

    if len(source_work) != len(events):
        raise ValueError("source_work长度必须等于event数")
    if destination_cycles is None:
        destination_cycles = [1] * len(events)
    if len(destination_cycles) != len(events):
        raise ValueError("destination_cycles长度必须等于event数")
    plane_events = _split_plane_events(events, plane_tokens)
    totals = {
        "cycles": 0,
        "terms": 0,
        "producer_stalls": 0,
        "max_fifo_sources": 0,
        "max_fifo_terms": 0,
        "producer_work_cycles": 0,
    }
    absolute_cycle = 0
    for plane_index, local_events in enumerate(plane_events):
        start = plane_index * plane_tokens
        result = simulate_frontier(
            local_events,
            source_work[start : start + plane_tokens],
            fifo_depth=fifo_depth,
            ready_percent=ready_percent,
            retire_width=retire_width,
            destination_cycles=destination_cycles[
                start : start + plane_tokens
            ],
            start_cycle=absolute_cycle,
        )
        absolute_cycle += result["cycles"]
        totals["cycles"] += result["cycles"]
        totals["terms"] += result["terms"]
        totals["producer_stalls"] += result["producer_stalls"]
        totals["max_fifo_sources"] = max(
            totals["max_fifo_sources"],
            result["max_fifo_sources"],
        )
        totals["max_fifo_terms"] = max(
            totals["max_fifo_terms"],
            result["max_fifo_terms"],
        )
        totals["producer_work_cycles"] += result[
            "producer_work_cycles"
        ]
    return totals


def simulate_plane_serial_stripe(
    source_work: list[int],
    *,
    height: int,
    width: int,
    ready_percent: int,
    destination_cycles: list[int],
    row_buffer_slots: int = 2,
) -> dict[str, int]:
    """强R1：行内增量建表，行末仅交换ping-pong row ownership。"""

    plane_tokens = height * width
    if plane_tokens <= 0 or len(source_work) % plane_tokens:
        raise ValueError("source_work无法按指定stripe几何划分")
    if len(destination_cycles) != len(source_work):
        raise ValueError("destination_cycles与source_work长度不一致")
    if row_buffer_slots < 2:
        raise ValueError("nonblocking stripe至少需要两个row buffer")

    totals = {
        "cycles": 0,
        "terms": 0,
        "producer_stalls": 0,
        "max_fifo_sources": 0,
        "max_fifo_terms": 0,
        "max_stripe_owned_rows": 0,
        "producer_work_cycles": sum(destination_cycles),
    }
    absolute_cycle = 0
    for plane_start in range(0, len(source_work), plane_tokens):
        plane_work = source_work[
            plane_start : plane_start + plane_tokens
        ]
        plane_cycles = destination_cycles[
            plane_start : plane_start + plane_tokens
        ]
        row_queue: list[list[int]] = []
        pending_rows: list[list[int]] = []
        destination = 0
        producer_remaining = plane_cycles[0]
        emitted = 0
        build_slots_held = 0

        while destination < plane_tokens or pending_rows or row_queue:
            if row_queue and sink_ready(absolute_cycle, ready_percent):
                row_queue[0][0] -= 1
                emitted += 1
                if row_queue[0][0] == 0:
                    row_queue[0].pop(0)
                if not row_queue[0]:
                    row_queue.pop(0)

            if pending_rows:
                if len(row_queue) + len(pending_rows) <= row_buffer_slots:
                    row_queue.extend(pending_rows)
                    pending_rows = []
                    build_slots_held = 0
                    destination += 1
                    if destination < plane_tokens:
                        producer_remaining = plane_cycles[destination]
                else:
                    totals["producer_stalls"] += 1
            elif destination < plane_tokens:
                destination_x = destination % width
                at_row_start = (
                    destination_x == 0
                    and producer_remaining == plane_cycles[destination]
                )
                destination_y = destination // width
                required_build_slots = (
                    2 if destination_y == height - 1 else 1
                )
                if (
                    at_row_start
                    and not build_slots_held
                    and len(row_queue) + required_build_slots
                    > row_buffer_slots
                ):
                    totals["producer_stalls"] += 1
                else:
                    if at_row_start:
                        build_slots_held = required_build_slots
                    producer_remaining -= 1
                if producer_remaining == 0:
                    if destination_x == width - 1:
                        release_rows = []
                        if destination_y > 0:
                            release_rows.append(destination_y - 1)
                        if destination_y == height - 1:
                            release_rows.append(height - 1)
                        pending_rows = [
                            [
                                work
                                for work in plane_work[
                                    source_y
                                    * width : (source_y + 1)
                                    * width
                                ]
                                if work > 0
                            ]
                            for source_y in release_rows
                        ]
                        pending_rows = [
                            row for row in pending_rows if row
                        ]
                        if (
                            pending_rows
                            and len(row_queue) + len(pending_rows)
                            <= row_buffer_slots
                        ):
                            row_queue.extend(pending_rows)
                            pending_rows = []
                            build_slots_held = 0
                        if not pending_rows:
                            build_slots_held = 0
                            destination += 1
                            if destination < plane_tokens:
                                producer_remaining = plane_cycles[
                                    destination
                                ]
                    else:
                        destination += 1
                        if destination < plane_tokens:
                            producer_remaining = plane_cycles[destination]

            totals["max_fifo_sources"] = max(
                totals["max_fifo_sources"],
                sum(len(row) for row in row_queue),
            )
            totals["max_fifo_terms"] = max(
                totals["max_fifo_terms"],
                sum(sum(row) for row in row_queue),
            )
            totals["max_stripe_owned_rows"] = max(
                totals["max_stripe_owned_rows"],
                len(row_queue) + build_slots_held,
            )
            absolute_cycle += 1
            totals["cycles"] += 1
            if totals["cycles"] > 10_000_000:
                raise RuntimeError("stripe模型未收敛")

        if emitted != sum(plane_work):
            raise AssertionError("stripe service work不守恒")
        totals["terms"] += emitted
    return totals


def storage_bits(
    height: int,
    width: int,
    *,
    time_planes: int = 2,
    head_dim: int = 32,
    gate_width: int = 9,
    candidates: int = 5,
    fifo_depth: int = 8,
) -> dict[str, int | float]:
    tokens = height * width * time_planes
    token_id_width = max(1, math.ceil(math.log2(tokens)))
    line_buffer = 3 * width * 2 * head_dim
    # 两个时间平面无跨时刻邻接，B1和FCSR都允许逐平面复用状态。
    plane_tokens = height * width
    full_k_plane = plane_tokens * head_dim
    full_gate_plane = plane_tokens * candidates * gate_width
    gate_ring = 3 * width * candidates * gate_width
    descriptor = (
        token_id_width + head_dim + candidates * gate_width + candidates
    )
    two_phase = line_buffer + full_k_plane + full_gate_plane
    frontier = line_buffer + gate_ring + fifo_depth * descriptor
    stripe = line_buffer + gate_ring + (2 * width) * descriptor
    return {
        "line_buffer": line_buffer,
        "full_k_plane": full_k_plane,
        "full_gate_plane": full_gate_plane,
        "gate_ring": gate_ring,
        "descriptor_bits": descriptor,
        "fifo_bits": fifo_depth * descriptor,
        "two_phase_peak_bits": two_phase,
        "frontier_peak_bits": frontier,
        "stripe_peak_bits": stripe,
        "reduction": 1.0 - frontier / two_phase,
    }


def weighted_quantile(histogram: list[int], quantile: float) -> int:
    total = sum(histogram)
    if total == 0:
        return 0
    target = math.ceil(total * quantile)
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return value
    return len(histogram) - 1


def weighted_stage_model(
    histogram: list[int],
    events: list[list[int]],
    *,
    scenario: str,
    fifo_depth: int,
    ready_percent: int,
) -> dict[str, float | int]:
    samples = sum(histogram)
    if not samples:
        return {
            "samples": 0,
            "two_phase_mean": 0.0,
            "frontier_mean": 0.0,
            "speedup": 0.0,
            "mean_stalls": 0.0,
            "max_fifo_sources": 0,
        }

    two_sum = 0
    frontier_sum = 0
    stall_sum = 0
    max_fifo_sources = 0
    for terms, count in enumerate(histogram):
        if not count:
            continue
        work = allocate_terms(terms, events, scenario)
        plane_tokens = len(events) // 2
        baseline = simulate_plane_serial_two_phase(
            source_work=work,
            plane_tokens=plane_tokens,
            ready_percent=ready_percent,
            destination_cycles=[1] * len(events),
        )
        frontier = simulate_plane_serial_frontier(
            events,
            work,
            plane_tokens=plane_tokens,
            fifo_depth=fifo_depth,
            ready_percent=ready_percent,
        )
        two_sum += count * baseline["cycles"]
        frontier_sum += count * frontier["cycles"]
        stall_sum += count * frontier["producer_stalls"]
        max_fifo_sources = max(
            max_fifo_sources,
            frontier["max_fifo_sources"],
        )
    two_mean = two_sum / samples
    frontier_mean = frontier_sum / samples
    return {
        "samples": samples,
        "two_phase_mean": two_mean,
        "frontier_mean": frontier_mean,
        "speedup": two_mean / frontier_mean,
        "mean_stalls": stall_sum / samples,
        "max_fifo_sources": max_fifo_sources,
        "terms_mean": sum(
            value * count for value, count in enumerate(histogram)
        )
        / samples,
        "terms_p95": weighted_quantile(histogram, 0.95),
        "terms_p99": weighted_quantile(histogram, 0.99),
    }


def scaled_histogram(histogram: list[int], scale: float) -> list[int]:
    """仅用于W15敏感性：保持样本权重，按token数比例缩放term。"""

    result: defaultdict[int, int] = defaultdict(int)
    for terms, count in enumerate(histogram):
        if count:
            result[int(round(terms * scale))] += count
    output = [0] * (max(result, default=0) + 1)
    for terms, count in result.items():
        output[terms] = count
    return output


def build_report(
    profile: dict[str, Any],
    *,
    fifo_depth: int,
) -> dict[str, Any]:
    topologies = {}
    for size in (9, 15):
        topologies[str(size)] = {
            "geometry": frontier_geometry(size, size),
            "storage": storage_bits(size, size, fifo_depth=fifo_depth),
        }

    events_w9 = retirement_events(9, 9)
    events_w15 = retirement_events(15, 15)
    stages = {}
    for stage, summary in profile["by_stage"].items():
        histogram = summary["mfep_terms_per_window_head_histogram"]
        stage_result: dict[str, Any] = {
            "w9_pre_g0": {},
            "w15_density_scaled_sensitivity": {},
        }
        for ready in (100, 90, 75):
            for scenario in ("uniform", "front_loaded", "tail_loaded"):
                key = f"ready{ready}_{scenario}"
                stage_result["w9_pre_g0"][key] = weighted_stage_model(
                    histogram,
                    events_w9,
                    scenario=scenario,
                    fifo_depth=fifo_depth,
                    ready_percent=ready,
                )
                stage_result["w15_density_scaled_sensitivity"][key] = (
                    weighted_stage_model(
                        scaled_histogram(histogram, 450 / 162),
                        events_w15,
                        scenario=scenario,
                        fifo_depth=fifo_depth,
                        ready_percent=ready,
                    )
                )
        stages[stage] = stage_result

    return {
        "schema": "local5_frontier_retirement_model_v1",
        "evidence": {
            "geometry": "[解析] Local5固定五点拓扑",
            "workload": "[模型/pre-G0] 旧crop profile100的MFEP term直方图",
            "w15": "[敏感性] 按450/162缩放term，不是fullres实测",
        },
        "fifo_depth": fifo_depth,
        "topologies": topologies,
        "stages": stages,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local5 FCSR 前沿完整源退休架构模型",
        "",
        "## 结论",
        "",
        "- `[解析]` 固定五点 stencil 使每个源 token 的最后消费者可由坐标静态确定；9×9 与 15×15 的单周期最大退休突发均为 3。",
        "- 首版物理相序冻结为 T0/T1 平面串行；底边 2–3 source 退休由单snapshot通路序列化，额外stall已计入模型。",
        "- FCSR 在最后一个消费者完成时捕获仍驻留于三行缓冲的 K，并立即将五个入边 gate 送入源主序投影；它改变的是跨阶段存储生命周期，而不是依赖 gate-code 复用的算术技巧。",
        "- `[模型/pre-G0]` 下面的吞吐仅由旧 profile 的 window-head term 总量和三种到达边界构成，不能替代 fullres post-G0 ordered trace。",
        "- 旧 profile 只有全窗口 MFEP 目录 term；FCSR 的真实 source-local 服务量可能更大，因此周期表只是抽象敏感性，不是加速证据。",
        "",
        "## 拓扑与存储",
        "",
        "| 窗口 | token | 首次退休位置 | 最大退休突发 | 两阶段峰值状态(bit) | FCSR峰值状态(bit) | 降低 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in ("9", "15"):
        topo = report["topologies"][size]
        geometry = topo["geometry"]
        storage = topo["storage"]
        lines.append(
            f"| {size}×{size}×2 | {geometry['tokens']} | "
            f"{geometry['warmup_destinations']} | {geometry['max_retire_burst']} | "
            f"{storage['two_phase_peak_bits']} | {storage['frontier_peak_bits']} | "
            f"{100 * storage['reduction']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "状态口径包含 Q/K 三行缓冲；两阶段强基线另含完整 K plane 与五方向 gate plane，FCSR 改为三行 gate ring 与 8-entry source descriptor FIFO。两者均不计共享权重 SRAM 和输出累加器。",
            "",
            "## W9旧Profile周期边界",
            "",
            "抽象口径：score producer 均为 1 destination/cycle，投影后端均为 1 term/cycle，区别仅为窗口结束后启动或按 frontier 重叠。旧 profile 的全窗口 MFEP term 被临时分配到 source；这不等价于真实 source term。表中为 ready=100%、8-entry FIFO。",
            "",
            "| stage | term均值 | term p95 | 两阶段周期 | FCSR均匀周期 | 均匀加速 | 最早聚集加速 | 最晚聚集加速 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, data in report["stages"].items():
        uniform = data["w9_pre_g0"]["ready100_uniform"]
        early = data["w9_pre_g0"]["ready100_front_loaded"]
        late = data["w9_pre_g0"]["ready100_tail_loaded"]
        lines.append(
            f"| S{stage} | {uniform['terms_mean']:.2f} | {uniform['terms_p95']} | "
            f"{uniform['two_phase_mean']:.2f} | {uniform['frontier_mean']:.2f} | "
            f"{uniform['speedup']:.3f}× | {early['speedup']:.3f}× | "
            f"{late['speedup']:.3f}× |"
        )

    lines.extend(
        [
            "",
            "“最早/最晚聚集”将全部 term 在每源容量上界内分别压向退休序列首/尾，只用于暴露缺少 ordered trace 时的上下界，不代表真实网络。",
            "",
            "## 架构合同",
            "",
            "1. T0/T1 两个独立时间平面串行执行并复用同一套行状态；每个 destination 的五个 gate 写入 `{plane_epoch,row_mod3,x,direction}` gate ring。",
            "2. 静态 frontier 表只由 `(y,x)` 和边界决定；不设置动态引用计数器。",
            "3. source 最后消费者完成时，读取五方向 gate 与 source K，形成 source descriptor；同拍多source按一项/拍序列化。",
            "4. descriptor FIFO 满时反压 destination score 流；当前行 K 必须在所有待退休 source 已捕获后才允许旋转。",
            "5. 下游 MFEP/DCTF 保持整数加法和 destination 集合不变，因此该调度应可 bit-exact。",
            "",
            "## 证据边界与RTL门槛",
            "",
            "- 当前存储收益是拓扑解析模型，不是综合面积。",
            "- 当前 term 到达顺序不是实测；fullres Local5 hardware-order follower 必须新增 per-source ordered term trace。",
            "- `ordered_term_trace_v2` 已定义 product term、destination delivery、四方向 delta、route/wave、source last-consumer 与 plane-serial 合同；post-G0 产物必须使用 `replay_local5_frontier_trace.py`，不得继续引用本表作为性能结果。",
            "- 第一版 RTL 只做 W1 destination producer、三行 gate ring、8-entry source FIFO、边界 drain 和随机反压，不引入 term 压缩新语义。",
            "- 只有在 post-G0 fullres trace 上同时满足：状态减少≥50%、均值周期收益≥15%、p99无溢出、同约束综合 PPA 为正，才把 FCSR 升为 DATE 主贡献。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fifo-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.profile.open() as handle:
        profile = json.load(handle)
    report = build_report(profile, fifo_depth=args.fifo_depth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    md_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
