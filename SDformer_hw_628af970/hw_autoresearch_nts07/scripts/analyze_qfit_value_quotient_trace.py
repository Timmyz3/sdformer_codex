#!/usr/bin/env python3
"""分析有序term流中的精确值商复用与有限frontier重排。"""

from __future__ import annotations

import csv
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/qfit_local5_projection_tile_yosys_20260731"
OUT = ROOT / "results/qfit_value_quotient_trace_20260731"


def load_rows() -> list[dict[str, int]]:
    rows = []
    with (SOURCE / "ordered_term_trace.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: int(value) for key, value in row.items()})
    if not rows:
        raise RuntimeError("ordered term trace为空")
    if [row["seq"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("ordered term trace序号不连续")
    return rows


def value_key(row: dict[str, int]) -> tuple[int, int]:
    return row["lane"], row["gate"]


def grouped_unique(
    rows: list[dict[str, int]],
    group_key,
) -> tuple[int, int, int]:
    groups: dict[object, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    computes = sum(len({value_key(row) for row in group}) for group in groups.values())
    max_terms = max(len(group) for group in groups.values())
    return computes, max_terms, len(groups)


def chunk_unique(
    rows: list[dict[str, int]], capacity: int
) -> tuple[int, int, int]:
    groups = [rows[start : start + capacity] for start in range(0, len(rows), capacity)]
    computes = sum(len({value_key(row) for row in group}) for group in groups)
    return computes, max(map(len, groups)), len(groups)


def lru_misses(rows: list[dict[str, int]], capacity: int) -> tuple[int, int]:
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = 0
    misses = 0
    for row in rows:
        key = value_key(row)
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            if len(cache) == capacity:
                cache.popitem(last=False)
            cache[key] = None
    return hits, misses


def lane_lru_misses(
    rows: list[dict[str, int]], ways: int
) -> tuple[int, int]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    hits = 0
    misses = 0
    for row in rows:
        lane = row["lane"]
        gate = row["gate"]
        cache = caches[lane]
        if gate in cache:
            hits += 1
            cache.move_to_end(gate)
        else:
            misses += 1
            if len(cache) == ways:
                cache.popitem(last=False)
            cache[gate] = None
    return hits, misses


def segmented_frontier(
    rows: list[dict[str, int]],
    *,
    term_capacity: int,
    lane_ways: int,
) -> dict[str, int | float]:
    if term_capacity <= 0 or lane_ways <= 0:
        raise ValueError("segment参数必须为正")
    product_computes = 0
    segment_count = 0
    capacity_seals = 0
    directory_seals = 0
    row_seals = 0
    max_terms = 0
    keys: set[tuple[int, int]] = set()
    lane_gates: dict[int, set[int]] = defaultdict(set)
    segment_terms = 0
    current_row: tuple[int, int] | None = None

    def seal(reason: str) -> None:
        nonlocal product_computes
        nonlocal segment_count
        nonlocal capacity_seals
        nonlocal directory_seals
        nonlocal row_seals
        nonlocal max_terms
        nonlocal keys
        nonlocal lane_gates
        nonlocal segment_terms
        if segment_terms == 0:
            return
        product_computes += len(keys)
        segment_count += 1
        max_terms = max(max_terms, segment_terms)
        if reason == "capacity":
            capacity_seals += 1
        elif reason == "directory":
            directory_seals += 1
        elif reason == "row":
            row_seals += 1
        keys = set()
        lane_gates = defaultdict(set)
        segment_terms = 0

    for row in rows:
        row_id = (row["plane"], row["y"])
        if current_row is not None and row_id != current_row:
            seal("row")
        current_row = row_id
        lane = row["lane"]
        gate = row["gate"]
        key = (lane, gate)
        new_gate = gate not in lane_gates[lane]
        if segment_terms == term_capacity:
            seal("capacity")
        elif new_gate and len(lane_gates[lane]) == lane_ways:
            seal("directory")
        keys.add(key)
        lane_gates[lane].add(gate)
        segment_terms += 1
    seal("row")
    return {
        "product_computes": product_computes,
        "reuse_ratio": 1.0 - product_computes / len(rows),
        "segments": segment_count,
        "max_buffered_terms": max_terms,
        "capacity_seals": capacity_seals,
        "directory_seals": directory_seals,
        "row_seals": row_seals,
    }


def row_owned_segmented_frontier(
    rows: list[dict[str, int]],
    *,
    term_capacity: int,
    lane_ways: int,
) -> dict[str, int | float]:
    groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    first: dict[tuple[int, int], int] = {}
    last: dict[tuple[int, int], int] = {}
    for index, row in enumerate(rows):
        row_id = (row["plane"], row["y"])
        groups[row_id].append(row)
        first.setdefault(row_id, index)
        last[row_id] = index
    max_live_rows = max(
        sum(
            first[row_id] <= index <= last[row_id]
            for row_id in groups
        )
        for index in range(len(rows))
    )
    totals = {
        "product_computes": 0,
        "segments": 0,
        "capacity_seals": 0,
        "directory_seals": 0,
        "row_seals": 0,
    }
    max_buffered_terms = 0
    for group in groups.values():
        stats = segmented_frontier(
            group,
            term_capacity=term_capacity,
            lane_ways=lane_ways,
        )
        for key in totals:
            totals[key] += int(stats[key])
        max_buffered_terms = max(
            max_buffered_terms,
            int(stats["max_buffered_terms"]),
        )
    return {
        **totals,
        "reuse_ratio": 1.0 - totals["product_computes"] / len(rows),
        "max_buffered_terms": max_buffered_terms,
        "max_live_rows": max_live_rows,
    }


def main() -> None:
    rows = load_rows()
    terms = len(rows)
    unique = len({value_key(row) for row in rows})

    grouped = {}
    group_specs = {
        "source": lambda row: (
            row["plane"], row["y"], row["x"]
        ),
        "row_frontier": lambda row: (row["plane"], row["y"]),
        "plane_frontier": lambda row: row["plane"],
        "window": lambda row: 0,
    }
    for name, function in group_specs.items():
        computes, max_terms, groups = grouped_unique(rows, function)
        grouped[name] = {
            "product_computes": computes,
            "reuse_ratio": 1.0 - computes / terms,
            "max_buffered_terms": max_terms,
            "groups": groups,
        }
    for capacity in (16, 32, 64, 128):
        computes, max_terms, groups = chunk_unique(rows, capacity)
        grouped[f"chunk_{capacity}"] = {
            "product_computes": computes,
            "reuse_ratio": 1.0 - computes / terms,
            "max_buffered_terms": max_terms,
            "groups": groups,
        }

    lru = {}
    for capacity in (16, 32, 64, 128):
        hits, misses = lru_misses(rows, capacity)
        lru[f"fully_associative_{capacity}"] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / terms,
        }
    for ways in (1, 2, 4, 8):
        hits, misses = lane_lru_misses(rows, ways)
        lru[f"lane_local_{ways}way"] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / terms,
            "entries": ways * 32,
        }

    row_groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        row_groups[(row["plane"], row["y"])].append(row)
    row_unique_keys = []
    row_lane_gate_cardinality = []
    key_multiplicities = []
    for group in row_groups.values():
        counts = Counter(value_key(row) for row in group)
        row_unique_keys.append(len(counts))
        key_multiplicities.extend(counts.values())
        by_lane: dict[int, set[int]] = defaultdict(set)
        for lane, gate in counts:
            by_lane[lane].add(gate)
        row_lane_gate_cardinality.extend(
            len(gates) for gates in by_lane.values()
        )

    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
        return ordered[index]

    row_directory = {
        "row_unique_keys_mean": sum(row_unique_keys) / len(row_unique_keys),
        "row_unique_keys_max": max(row_unique_keys),
        "lane_gate_cardinality_mean": (
            sum(row_lane_gate_cardinality) / len(row_lane_gate_cardinality)
        ),
        "lane_gate_cardinality_p95": percentile(
            row_lane_gate_cardinality, 0.95
        ),
        "lane_gate_cardinality_max": max(row_lane_gate_cardinality),
        "terms_per_value_key_mean": (
            sum(key_multiplicities) / len(key_multiplicities)
        ),
        "terms_per_value_key_p95": percentile(key_multiplicities, 0.95),
        "terms_per_value_key_max": max(key_multiplicities),
    }
    segmented = {}
    row_owned_segmented = {}
    for term_capacity in (64, 128, 256, 384):
        for lane_ways in (2, 4, 6, 8):
            name = f"c{term_capacity}_w{lane_ways}"
            segmented[name] = segmented_frontier(
                rows,
                term_capacity=term_capacity,
                lane_ways=lane_ways,
            )
            row_owned_segmented[name] = row_owned_segmented_frontier(
                rows,
                term_capacity=term_capacity,
                lane_ways=lane_ways,
            )

    result = {
        "evidence": "W6定向有序term trace；TB观测模型；非post-G0",
        "terms": terms,
        "unique_lane_gate_keys": unique,
        "global_reuse_upper_bound": 1.0 - unique / terms,
        "bounded_reordering": grouped,
        "ordered_memo": lru,
        "row_directory": row_directory,
        "segmented_frontier": segmented,
        "row_owned_segmented_frontier": row_owned_segmented,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "value_quotient_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Local5 值商复用与有限Frontier重排分析",
        "",
        "## 结论",
        "",
        f"- 有序term数：{terms}；全窗口唯一`(lane,gate)`值键：{unique}；"
        f"全窗口理论复用上界：{1 - unique / terms:.2%}。",
        "- 简单在线memo与有限frontier重排不是同一机制：前者依赖时间局部性，"
        "后者用有界缓冲显式制造值键局部性。",
        f"- row frontier平均含 {row_directory['row_unique_keys_mean']:.2f} 个"
        f"值键、最大 {row_directory['row_unique_keys_max']} 个；每lane每row "
        f"gate基数p95={row_directory['lane_gate_cardinality_p95']}、"
        f"max={row_directory['lane_gate_cardinality_max']}。",
        f"- 每个值键平均关联 {row_directory['terms_per_value_key_mean']:.2f} "
        f"条term，p95={row_directory['terms_per_value_key_p95']}、"
        f"max={row_directory['terms_per_value_key_max']}。",
        "",
        "## 有序Memo",
        "",
        "| 结构 | 项数 | 命中 | 命中率 |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in lru.items():
        if "entries" in stats:
            entries = stats["entries"]
        else:
            entries = int(name.rsplit("_", 1)[-1])
        lines.append(
            f"| {name} | {entries} | {stats['hits']} | "
            f"{stats['hit_rate']:.2%} |"
        )
    lines.extend([
        "",
        "## 有限Frontier值键重排",
        "",
        "| 范围 | product计算次数 | 减少比例 | 最大缓冲term | 分组数 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, stats in grouped.items():
        lines.append(
            f"| {name} | {stats['product_computes']} | "
            f"{stats['reuse_ratio']:.2%} | {stats['max_buffered_terms']} | "
            f"{stats['groups']} |"
        )
    lines.extend([
        "",
        "## 可提前封口的分段DQFS",
        "",
        "| term容量/每lane way | product计算 | 减少 | segment | cap封口 | "
        "way封口 | row封口 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stats in segmented.items():
        lines.append(
            f"| {name} | {stats['product_computes']} | "
            f"{stats['reuse_ratio']:.2%} | {stats['segments']} | "
            f"{stats['capacity_seals']} | {stats['directory_seals']} | "
            f"{stats['row_seals']} |"
        )
    lines.extend([
        "",
        "## Row-Owned交错收集DQFS",
        "",
        "| term容量/每lane way | product计算 | 减少 | segment | cap封口 | "
        "way封口 | 最大活跃row |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stats in row_owned_segmented.items():
        lines.append(
            f"| {name} | {stats['product_computes']} | "
            f"{stats['reuse_ratio']:.2%} | {stats['segments']} | "
            f"{stats['capacity_seals']} | {stats['directory_seals']} | "
            f"{stats['max_live_rows']} |"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- 所有数字来自W6定向向量，不代表full-resolution部署分布；",
        "- product计算次数减少只代表乘法器激活机会，不自动等于周期或能耗下降；",
        "- frontier重排必须增加descriptor缓冲、值键目录和有序提交控制；",
        "- post-G0真实trace、RTL、同宏DC/STA/SAIF完成前，不列为论文已实现贡献。",
        "",
    ])
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
