#!/usr/bin/env python3
"""用真实Local5 source-major trace评估颜色bank前的驻留累加缓存。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEIGHT = 15
WIDTH = 15
PLANES = 2
ROLES = 5
HEAD_DIM = 32
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)
BANKS = 5
BANK_DEPTH = 90


@dataclass
class CacheCounters:
    updates: int = 0
    hits: int = 0
    misses: int = 0
    reads: int = 0
    writes: int = 0
    stall_cycles: int = 0
    flush_cycles: int = 0


def source_targets(
    source_terms: list[list[tuple[int, int]]]
) -> list[int | None]:
    targets: list[int | None] = [None] * BANKS
    for term in source_terms:
        for bank, address in term:
            if targets[bank] is not None and targets[bank] != address:
                raise AssertionError("同一source在一个颜色bank访问多个地址")
            targets[bank] = address
    return targets


def geometry_targets(plane: int, source_y: int, source_x: int) -> list[int | None]:
    targets: list[int | None] = [None] * BANKS
    for role in range(ROLES):
        y = source_y + ROLE_DY[role]
        x = source_x + ROLE_DX[role]
        if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
            continue
        bank, address = destination_bank_address(plane, y, x)
        if targets[bank] is not None and targets[bank] != address:
            raise AssertionError("Local5几何前瞻发生颜色bank冲突")
        targets[bank] = address
    return targets


def simulate_dual_context_prefetch(
    source_stream: list[list[list[tuple[int, int]]]],
    descriptor_latency: int,
    prepare_targets: list[list[int | None]] | None = None,
) -> CacheCounters:
    slots: list[list[dict[str, object] | None]] = [
        [None, None] for _ in range(BANKS)
    ]
    active_slot: list[int | None] = [None] * BANKS
    materialized: list[set[int]] = [set() for _ in range(BANKS)]
    counters = CacheCounters()
    update_targets = [source_targets(source) for source in source_stream]
    targets = prepare_targets if prepare_targets is not None else update_targets
    if len(targets) != len(source_stream):
        raise ValueError("prepare target数量与source stream不一致")

    def prepare(bank: int, address: int) -> tuple[int, int]:
        for slot_index, slot in enumerate(slots[bank]):
            if slot is not None and int(slot["address"]) == address:
                return slot_index, 0
        victim = 0 if active_slot[bank] != 0 else 1
        slot = slots[bank][victim]
        operations = 0
        if slot is not None and bool(slot["dirty"]):
            counters.writes += 1
            operations += 1
            materialized[bank].add(int(slot["address"]))
        if address in materialized[bank]:
            counters.reads += 1
            operations += 1
        slots[bank][victim] = {"address": address, "dirty": False}
        counters.misses += 1
        return victim, operations

    if source_stream:
        for bank, address in enumerate(targets[0]):
            if address is not None:
                active_slot[bank], operations = prepare(bank, address)
                counters.stall_cycles += operations

    for source_index, source_terms in enumerate(source_stream):
        duration = len(source_terms)
        counters.updates += sum(len(term) for term in source_terms)
        current_targets = update_targets[source_index]
        for bank, address in enumerate(current_targets):
            if address is None:
                continue
            slot_index = active_slot[bank]
            if slot_index is None:
                raise AssertionError("active slot未准备")
            slot = slots[bank][slot_index]
            if slot is None or int(slot["address"]) != address:
                raise AssertionError("active slot地址错误")
            slot["dirty"] = True

        if source_index + 1 == len(source_stream):
            continue
        next_active: list[int | None] = [None] * BANKS
        operations_per_bank = [0] * BANKS
        for bank, address in enumerate(targets[source_index + 1]):
            if address is None:
                continue
            next_active[bank], operations_per_bank[bank] = prepare(bank, address)
            if operations_per_bank[bank] == 0:
                counters.hits += 1
        counters.stall_cycles += max(
            (
                max(0, descriptor_latency + operations - duration)
                for operations in operations_per_bank
            ),
            default=0,
        )
        active_slot = next_active

    dirty_per_bank = []
    for bank in range(BANKS):
        dirty = 0
        for slot in slots[bank]:
            if slot is not None and bool(slot["dirty"]):
                dirty += 1
                counters.writes += 1
        dirty_per_bank.append(dirty)
    counters.flush_cycles = max(dirty_per_bank, default=0)
    return counters


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def destination_bank_address(plane: int, y: int, x: int) -> tuple[int, int]:
    bank = (x + 2 * y) % 5
    address = plane * 45 + y * 3 + x // 5
    return bank, address


def descriptor_terms(
    plane: int,
    source_y: int,
    source_x: int,
    k_bitmap: int,
    gates: np.ndarray,
    valid_mask: int,
) -> list[list[tuple[int, int]]]:
    unique_gates: list[int] = []
    gate_roles: list[list[int]] = []
    for role in range(ROLES):
        gate = int(gates[role])
        if not ((valid_mask >> role) & 1) or gate == 0:
            continue
        if gate in unique_gates:
            gate_roles[unique_gates.index(gate)].append(role)
        else:
            unique_gates.append(gate)
            gate_roles.append([role])

    terms: list[list[tuple[int, int]]] = []
    for lane in range(HEAD_DIM):
        if not ((k_bitmap >> lane) & 1):
            continue
        for roles in gate_roles:
            destinations: list[tuple[int, int]] = []
            occupied_banks: set[int] = set()
            for role in roles:
                y = source_y + ROLE_DY[role]
                x = source_x + ROLE_DX[role]
                if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
                    raise ValueError("有效Local5 destination越界")
                bank, address = destination_bank_address(plane, y, x)
                if bank in occupied_banks:
                    raise AssertionError("五色映射在单term内发生bank冲突")
                occupied_banks.add(bank)
                destinations.append((bank, address))
            terms.append(destinations)
    return terms


def simulate_cache(
    term_stream: list[list[tuple[int, int]]], capacity: int
) -> CacheCounters:
    caches: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(BANKS)]
    materialized: list[set[int]] = [set() for _ in range(BANKS)]
    counters = CacheCounters()

    for destinations in term_stream:
        bank_operations = [0] * BANKS
        for bank, address in destinations:
            counters.updates += 1
            cache = caches[bank]
            if address in cache:
                counters.hits += 1
                cache.move_to_end(address)
                continue

            counters.misses += 1
            if len(cache) >= capacity:
                evicted, _ = cache.popitem(last=False)
                counters.writes += 1
                bank_operations[bank] += 1
                materialized[bank].add(evicted)
            if address in materialized[bank]:
                counters.reads += 1
                bank_operations[bank] += 1
            cache[address] = None
        counters.stall_cycles += max(bank_operations, default=0)

    # Banks flush in parallel; each single-port bank writes one entry per cycle.
    for cache in caches:
        counters.writes += len(cache)
    counters.flush_cycles = max((len(cache) for cache in caches), default=0)
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-manifest",
        type=Path,
        default=Path(
            "tb_qfit/vectors/local5_active_projection_postg0_100/manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_source_stationary_acc_cache_20260803"),
    )
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--descriptor-latencies", type=int, nargs="+", default=[0, 2, 3, 4, 6]
    )
    parser.add_argument(
        "--measured-top-log",
        type=Path,
        default=Path(
            "results/local5_active_projection_sync_sram_postg0_rtl_20260803/"
            "tcfm5_l1_verilator.log"
        ),
    )
    args = parser.parse_args()

    vector_manifest = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    source_payload = Path(vector_manifest["source_payload"])
    source_manifest = Path(vector_manifest["source_manifest"])
    ordered_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload = np.load(source_payload, mmap_mode="r")
    offsets = np.asarray(payload["descriptor_group_offsets"])
    descriptor_k_bitmap = np.asarray(payload["descriptor_k_bitmap"])
    descriptor_gates = np.asarray(payload["descriptor_incoming_gates"])
    descriptor_valid_mask = np.asarray(payload["descriptor_valid_mask"])
    descriptor_plane = np.asarray(payload["descriptor_source_plane"])
    descriptor_y = np.asarray(payload["descriptor_source_y"])
    descriptor_x = np.asarray(payload["descriptor_source_x"])
    selection = vector_manifest["selection"]["rows"]
    measured_top_cycles: dict[int, int] = {}
    group_pattern = re.compile(r"GROUP backend=0 latency=1 group=(\d+) cycles=(\d+)")
    for line in args.measured_top_log.read_text(encoding="utf-8").splitlines():
        match = group_pattern.search(line)
        if match:
            measured_top_cycles[int(match.group(1))] = int(match.group(2))
    if set(measured_top_cycles) != set(range(len(selection))):
        raise ValueError("真实TCFM5 L1日志未覆盖全部100组")

    rows: list[dict[str, object]] = []
    aggregate: dict[int, CacheCounters] = {
        capacity: CacheCounters() for capacity in args.capacities
    }
    speedups: dict[int, list[float]] = {capacity: [] for capacity in args.capacities}
    prefetch_aggregate: dict[int, CacheCounters] = {
        latency: CacheCounters() for latency in args.descriptor_latencies
    }
    prefetch_speedups: dict[int, list[float]] = {
        latency: [] for latency in args.descriptor_latencies
    }
    geometry_aggregate: dict[int, CacheCounters] = {
        latency: CacheCounters() for latency in (0, 1)
    }
    actual_target_slots = 0
    geometry_target_slots = 0

    for vector_group, selected in enumerate(selection):
        input_group = int(selected["input_group_index"])
        start = int(offsets[input_group])
        stop = int(offsets[input_group + 1])
        term_stream: list[list[tuple[int, int]]] = []
        source_stream: list[list[list[tuple[int, int]]]] = []
        source_geometry_stream: list[list[int | None]] = []
        active_sources = 0
        for index in range(start, stop):
            k_bitmap = int(descriptor_k_bitmap[index])
            gates = descriptor_gates[index]
            valid_mask = int(descriptor_valid_mask[index])
            source_terms = descriptor_terms(
                int(descriptor_plane[index]),
                int(descriptor_y[index]),
                int(descriptor_x[index]),
                k_bitmap,
                gates,
                valid_mask,
            )
            if source_terms:
                active_sources += 1
                term_stream.extend(source_terms)
                source_stream.append(source_terms)
                actual = source_targets(source_terms)
                geometry = geometry_targets(
                    int(descriptor_plane[index]),
                    int(descriptor_y[index]),
                    int(descriptor_x[index]),
                )
                actual_target_slots += sum(item is not None for item in actual)
                geometry_target_slots += sum(item is not None for item in geometry)
                source_geometry_stream.append(geometry)

        terms = len(term_stream)
        updates = sum(len(term) for term in term_stream)
        if terms != int(selected["terms"]) or updates != int(selected["updates"]):
            raise AssertionError("重建term/update计数与qualified trace不一致")

        # Current 1R1W design clears 90 addresses in parallel across five banks,
        # then accepts one term/cycle. Two cycles cover final pipeline drain.
        direct_cycles = BANK_DEPTH + terms + 2
        direct_transactions = BANKS * BANK_DEPTH + 2 * updates
        row: dict[str, object] = {
            "vector_group": vector_group,
            "input_group": input_group,
            "stage": int(selected["stage"]),
            "active_sources": active_sources,
            "terms": terms,
            "updates": updates,
            "direct_1r1w_cycles": direct_cycles,
            "direct_sram_transactions": direct_transactions,
            "measured_1r1w_top_cycles": measured_top_cycles[vector_group],
            "modeled_direct_1rw_top_cycles": (
                measured_top_cycles[vector_group] + terms
            ),
        }

        for capacity in args.capacities:
            counters = simulate_cache(term_stream, capacity)
            if counters.updates != updates:
                raise AssertionError("cache update计数错误")
            # One metadata epoch-start cycle replaces data-SRAM clearing.
            cache_cycles = 1 + terms + counters.stall_cycles + counters.flush_cycles
            cache_transactions = counters.reads + counters.writes
            speedup = direct_cycles / cache_cycles if cache_cycles else 1.0
            row.update(
                {
                    f"c{capacity}_hits": counters.hits,
                    f"c{capacity}_misses": counters.misses,
                    f"c{capacity}_reads": counters.reads,
                    f"c{capacity}_writes": counters.writes,
                    f"c{capacity}_stall_cycles": counters.stall_cycles,
                    f"c{capacity}_cycles": cache_cycles,
                    f"c{capacity}_speedup_vs_1r1w": speedup,
                    f"c{capacity}_transaction_reduction": (
                        1 - cache_transactions / direct_transactions
                        if direct_transactions
                        else 0.0
                    ),
                    f"c{capacity}_modeled_top_cycles": (
                        measured_top_cycles[vector_group]
                        + counters.stall_cycles
                        + max(0, counters.flush_cycles - 1)
                    ),
                }
            )
            total = aggregate[capacity]
            for field in CacheCounters.__dataclass_fields__:
                setattr(total, field, getattr(total, field) + getattr(counters, field))
            speedups[capacity].append(speedup)
        for latency in args.descriptor_latencies:
            prefetch = simulate_dual_context_prefetch(source_stream, latency)
            prefetch_cycles = 1 + terms + prefetch.stall_cycles + prefetch.flush_cycles
            prefetch_transactions = prefetch.reads + prefetch.writes
            prefetch_speedup = direct_cycles / prefetch_cycles if prefetch_cycles else 1.0
            prefix = f"dc_l{latency}"
            row.update(
                {
                    f"{prefix}_reads": prefetch.reads,
                    f"{prefix}_writes": prefetch.writes,
                    f"{prefix}_stall_cycles": prefetch.stall_cycles,
                    f"{prefix}_cycles": prefetch_cycles,
                    f"{prefix}_speedup_vs_1r1w": prefetch_speedup,
                    f"{prefix}_transaction_reduction": (
                        1 - prefetch_transactions / direct_transactions
                        if direct_transactions
                        else 0.0
                    ),
                    f"{prefix}_modeled_top_cycles": (
                        measured_top_cycles[vector_group]
                        + prefetch.stall_cycles
                        + max(0, prefetch.flush_cycles - 1)
                    ),
                }
            )
            total = prefetch_aggregate[latency]
            for field in CacheCounters.__dataclass_fields__:
                setattr(total, field, getattr(total, field) + getattr(prefetch, field))
            prefetch_speedups[latency].append(prefetch_speedup)
        for latency in (0, 1):
            geometry = simulate_dual_context_prefetch(
                source_stream, latency, source_geometry_stream
            )
            prefix = f"gap_l{latency}"
            row.update(
                {
                    f"{prefix}_reads": geometry.reads,
                    f"{prefix}_writes": geometry.writes,
                    f"{prefix}_stall_cycles": geometry.stall_cycles,
                    f"{prefix}_modeled_top_cycles": (
                        measured_top_cycles[vector_group]
                        + geometry.stall_cycles
                        + max(0, geometry.flush_cycles - 1)
                    ),
                }
            )
            total = geometry_aggregate[latency]
            for field in CacheCounters.__dataclass_fields__:
                setattr(total, field, getattr(total, field) + getattr(geometry, field))
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_group.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_terms = sum(int(row["terms"]) for row in rows)
    total_updates = sum(int(row["updates"]) for row in rows)
    total_direct_cycles = sum(int(row["direct_1r1w_cycles"]) for row in rows)
    total_direct_transactions = sum(
        int(row["direct_sram_transactions"]) for row in rows
    )
    total_measured_top_cycles = sum(
        int(row["measured_1r1w_top_cycles"]) for row in rows
    )
    total_direct_1rw_top_cycles = sum(
        int(row["modeled_direct_1rw_top_cycles"]) for row in rows
    )
    summary_rows: list[dict[str, object]] = []
    for capacity in args.capacities:
        counters = aggregate[capacity]
        total_cache_cycles = sum(int(row[f"c{capacity}_cycles"]) for row in rows)
        transactions = counters.reads + counters.writes
        modeled_top_cycles = sum(
            int(row[f"c{capacity}_modeled_top_cycles"]) for row in rows
        )
        summary_rows.append(
            {
                "capacity_per_bank": capacity,
                "cache_bits_outdim2": capacity * BANKS * (64 + 7 + 2),
                "hit_rate": counters.hits / counters.updates if counters.updates else 1.0,
                "reads": counters.reads,
                "writes": counters.writes,
                "sram_transactions": transactions,
                "transaction_reduction": 1 - transactions / total_direct_transactions,
                "estimated_cycles": total_cache_cycles,
                "aggregate_speedup_vs_direct_1r1w": (
                    total_direct_cycles / total_cache_cycles
                ),
                "per_group_speedup_mean": float(np.mean(speedups[capacity])),
                "per_group_speedup_p50": percentile(speedups[capacity], 50),
                "per_group_speedup_p95": percentile(speedups[capacity], 95),
                "stall_cycles": counters.stall_cycles,
                "flush_cycles": counters.flush_cycles,
                "modeled_top_cycles": modeled_top_cycles,
                "top_speedup_vs_measured_1r1w": (
                    total_measured_top_cycles / modeled_top_cycles
                ),
            }
        )

    prefetch_sweep: list[dict[str, object]] = []
    for latency in args.descriptor_latencies:
        counters = prefetch_aggregate[latency]
        prefix = f"dc_l{latency}"
        backend_cycles = sum(int(row[f"{prefix}_cycles"]) for row in rows)
        transactions = counters.reads + counters.writes
        modeled_top_cycles = sum(
            int(row[f"{prefix}_modeled_top_cycles"]) for row in rows
        )
        prefetch_sweep.append(
            {
                "descriptor_latency": latency,
                "contexts_per_bank": 2,
                "cache_bits_outdim2": 2 * BANKS * (64 + 7 + 2),
                "reads": counters.reads,
                "writes": counters.writes,
                "sram_transactions": transactions,
                "transaction_reduction": 1 - transactions / total_direct_transactions,
                "backend_cycles": backend_cycles,
                "fair_lazy_zero_1r1w_backend_speedup": (
                    (total_terms + 3 * len(rows)) / backend_cycles
                ),
                "modeled_top_cycles": modeled_top_cycles,
                "top_speedup_vs_measured_1r1w": (
                    total_measured_top_cycles / modeled_top_cycles
                ),
                "top_speedup_vs_direct_1rw": (
                    total_direct_1rw_top_cycles / modeled_top_cycles
                ),
                "stall_cycles": counters.stall_cycles,
                "flush_cycles": counters.flush_cycles,
                "per_group_speedup_p50": percentile(
                    prefetch_speedups[latency], 50
                ),
                "per_group_speedup_p95": percentile(
                    prefetch_speedups[latency], 95
                ),
            }
        )
    main_latency = 3 if 3 in args.descriptor_latencies else args.descriptor_latencies[0]
    prefetch_summary = next(
        item for item in prefetch_sweep if item["descriptor_latency"] == main_latency
    )
    geometry_sweep: list[dict[str, object]] = []
    for latency in (0, 1):
        counters = geometry_aggregate[latency]
        prefix = f"gap_l{latency}"
        transactions = counters.reads + counters.writes
        modeled_top_cycles = sum(
            int(row[f"{prefix}_modeled_top_cycles"]) for row in rows
        )
        geometry_sweep.append(
            {
                "index_to_target_latency": latency,
                "reads": counters.reads,
                "writes": counters.writes,
                "sram_transactions": transactions,
                "transaction_reduction": 1 - transactions / total_direct_transactions,
                "stall_cycles": counters.stall_cycles,
                "flush_cycles": counters.flush_cycles,
                "modeled_top_cycles": modeled_top_cycles,
                "top_speedup_vs_measured_1r1w": (
                    total_measured_top_cycles / modeled_top_cycles
                ),
                "top_speedup_vs_direct_1rw": (
                    total_direct_1rw_top_cycles / modeled_top_cycles
                ),
            }
        )

    summary = {
        "schema": "local5_source_stationary_acc_cache_profile_v1",
        "evidence": "qualified Local5 post-G0 source-major trace, 100 stratified groups",
        "source_vector_manifest": str(args.vector_manifest.resolve()),
        "groups": len(rows),
        "terms": total_terms,
        "updates": total_updates,
        "source_target_slots": {
            "payload_active": actual_target_slots,
            "geometry_ahead": geometry_target_slots,
            "extra_clean_prefetches": geometry_target_slots - actual_target_slots,
        },
        "direct_1r1w": {
            "cycles": total_direct_cycles,
            "sram_transactions": total_direct_transactions,
            "clear_transactions": len(rows) * BANKS * BANK_DEPTH,
            "measured_postscore_top_cycles": total_measured_top_cycles,
        },
        "direct_1rw_model": {
            "postscore_top_cycles": total_direct_1rw_top_cycles,
            "assumption": "one extra SRAM write cycle per nonempty term",
        },
        "cache_candidates": summary_rows,
        "dual_context_lookahead1": prefetch_summary,
        "descriptor_latency_sweep": prefetch_sweep,
        "geometry_ahead_prefetch_sweep": geometry_sweep,
        "model_boundary": {
            "term_issue": "one term per cycle before cache miss stalls",
            "bank_parallelism": "five SRAM banks service misses/flushes in parallel",
            "first_touch": "zero base via per-address valid metadata; no data-SRAM read",
            "replacement": "per-bank true LRU",
            "not_measured": ["RTL critical path", "macro power", "routing", "backpressure"],
            "prefetch_requirement": "next source descriptor available while current source terms execute",
            "descriptor_latency_definition": "cycles from next-slot availability to decoded next-source targets",
            "geometry_ahead": "prefetch all in-bound stencil destinations from word-skipper source coordinates before gate/K payload returns",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Local5 源驻留累加缓存真实 trace 评估",
        "",
        "## 证据边界",
        "",
        "本报告使用 qualified Local5 post-G0 的 100 个分层抽样组，按 RTL 的 source-major、lane-major、gate-major 顺序重建 term。结果属于 `[prof+模型]`，尚不是 RTL 或 PPA。",
        "",
        "## 汇总结论",
        "",
        f"- 总 term：{total_terms:,}；总 destination update：{total_updates:,}。",
        f"- 当前双端口 RMW 口径的数据 SRAM 事务：{total_direct_transactions:,}，其中窗口清零写 {len(rows) * BANKS * BANK_DEPTH:,}。",
        f"- 真实 TCFM5 L1 post-score 顶层周期：{total_measured_top_cycles:,}；同端口直接 1RW-RMW 模型：{total_direct_1rw_top_cycles:,}。",
        f"- Geometry-ahead 目标槽：{geometry_target_slots:,}，比 payload-active 的 {actual_target_slots:,} 多 {geometry_target_slots - actual_target_slots:,} 个保守预取目标。",
        "",
        "| 每 bank 驻留项 | 命中率 | SRAM 事务 | 事务下降 | 顶层估算周期 | 相对实测 1R1W 顶层 | 缓存位数(OUT_DIM=2) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in summary_rows:
        lines.append(
            "| {capacity_per_bank} | {hit_rate:.2%} | {sram_transactions:,} | "
            "{transaction_reduction:.2%} | {modeled_top_cycles:,} | "
            "{top_speedup_vs_measured_1r1w:.3f}x | {cache_bits_outdim2:,} |".format(
                **candidate
            )
        )
    lines.extend(
        [
            "",
            "主口径采用 descriptor latency={descriptor_latency}：SRAM 事务 {sram_transactions:,}，下降 {transaction_reduction:.2%}；估算 post-score 顶层周期 {modeled_top_cycles:,}，相对实测理想 1R1W 顶层为 {top_speedup_vs_measured_1r1w:.3f}x，相对同端口直接 1RW-RMW 模型为 {top_speedup_vs_direct_1rw:.3f}x；未隐藏停顿 {stall_cycles:,} 周期。".format(
                **prefetch_summary
            ),
            "",
            "| Descriptor latency | 残余停顿 | 顶层估算周期 | 对实测1R1W | 对直接1RW | 公平lazy-zero backend |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in prefetch_sweep:
        lines.append(
            "| {descriptor_latency} | {stall_cycles:,} | {modeled_top_cycles:,} | {top_speedup_vs_measured_1r1w:.3f}x | {top_speedup_vs_direct_1rw:.3f}x | {fair_lazy_zero_1r1w_backend_speedup:.3f}x |".format(
                **candidate
            )
        )
    lines.extend(
        [
            "",
            "Geometry-ahead 会预取全部合法十字邻居，不等待 gate/K payload：",
            "",
            "| Index到目标延迟 | SRAM事务 | 事务下降 | 残余停顿 | 顶层周期 | 对实测1R1W | 对直接1RW |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in geometry_sweep:
        lines.append(
            "| {index_to_target_latency} | {sram_transactions:,} | {transaction_reduction:.2%} | {stall_cycles:,} | {modeled_top_cycles:,} | {top_speedup_vs_measured_1r1w:.3f}x | {top_speedup_vs_direct_1rw:.3f}x |".format(
                **candidate
            )
        )
    lines.extend(
        [
            "",
            "## 架构含义",
            "",
            "同一 source 的 term 连续，且五色映射使每个 bank 在该 source 内只有一个固定 destination 地址。驻留槽因此可在寄存器中合并多个精确整数增量，并把每-term SRAM RMW 改为 miss/evict 时的单端口访问。该机制不丢 term、不近似 gate，也不改变 Acc32 加法次序以外的整数和。",
            "双上下文模型要求 relation frontier/term builder 至少提前保存一个 source descriptor；若下一 source 不能在当前 source 执行期间可见，906 个残余停顿和 1RW 对照都会失去意义。",
            "",
            "## 尚未证明",
            "",
            "尚未实现驻留槽 RTL、valid metadata、单端口 miss FSM、反压和 flush；估算周期不能直接写作实测加速，必须以 bit-exact RTL 回放和同宏 OpenROAD 对照替换。",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
