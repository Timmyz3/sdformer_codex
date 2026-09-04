#!/usr/bin/env python3
"""按真实OUT_DIM计算Local5值复用结构的存储-计算Pareto。"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = (
    ROOT
    / "results/qfit_local5_projection_tile_yosys_20260731"
    / "ordered_term_trace.csv"
)
VALUE_STATS = (
    ROOT
    / "results/qfit_value_quotient_trace_20260731"
    / "value_quotient_stats.json"
)
OUT = ROOT / "results/qfit_product_storage_pareto_20260731"


def clog2(value: int) -> int:
    if value <= 1:
        return 1
    return math.ceil(math.log2(value))


def load_trace() -> list[dict[str, int]]:
    with TRACE.open(newline="") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError("ordered term trace为空")
    return rows


def lane_lru(rows: list[dict[str, int]], ways: int) -> tuple[int, int]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    hits = 0
    misses = 0
    for row in rows:
        cache = caches[row["lane"]]
        gate = row["gate"]
        if gate in cache:
            hits += 1
            cache.move_to_end(gate)
        else:
            misses += 1
            if len(cache) == ways:
                cache.popitem(last=False)
            cache[gate] = None
    return hits, misses


def lane_no_replace_slots(
    rows: list[dict[str, int]], slots: int
) -> tuple[int, int, int]:
    tables: dict[int, list[int]] = defaultdict(list)
    reused = 0
    computes = 0
    overflow_computes = 0
    for row in rows:
        table = tables[row["lane"]]
        gate = row["gate"]
        if gate in table:
            reused += 1
        else:
            computes += 1
            if len(table) < slots:
                table.append(gate)
            else:
                overflow_computes += 1
    return reused, computes, overflow_computes


def cache_bits(
    *,
    lanes: int,
    ways: int,
    product_bits: int,
    gate_bits: int,
) -> dict[str, int]:
    entries = lanes * ways
    replacement_bits = clog2(ways) if ways > 1 else 0
    data = entries * product_bits
    tags = entries * (gate_bits + 1)
    replacement = entries * replacement_bits
    output_register = product_bits
    return {
        "data_bits": data,
        "metadata_bits": tags + replacement,
        "register_bits": output_register,
        "total_bits": data + tags + replacement + output_register,
    }


def slot_table_bits(
    *,
    lanes: int,
    slots: int,
    product_bits: int,
    gate_bits: int,
) -> dict[str, int]:
    entries = lanes * slots
    data = entries * product_bits
    tags = entries * (gate_bits + 1)
    lane_counts = lanes * clog2(slots + 1)
    output_register = product_bits
    return {
        "data_bits": data,
        "metadata_bits": tags + lane_counts,
        "register_bits": output_register,
        "total_bits": data + tags + lane_counts + output_register,
    }


def frozen_codebook_bits(
    *,
    lanes: int,
    codes: int,
    product_bits: int,
) -> dict[str, int]:
    entries = lanes * codes
    data = entries * product_bits
    valid = entries
    output_register = product_bits
    return {
        "data_bits": data,
        "metadata_bits": valid,
        "register_bits": output_register,
        "total_bits": data + valid + output_register,
    }


def dqfs_bits(
    *,
    capacity: int,
    ways: int,
    product_bits: int,
    contexts: int = 2,
    lanes: int = 32,
    gate_bits: int = 9,
    plane_bits: int = 1,
    y_bits: int = 4,
    x_bits: int = 4,
    mask_bits: int = 5,
    row_bits: int = 5,
    epoch_bits: int = 4,
    tile_bits: int = 4,
) -> dict[str, int]:
    ptr_bits = clog2(capacity)
    count_bits = clog2(capacity + 1)
    term_entry = (
        plane_bits
        + y_bits
        + x_bits
        + mask_bits
        + ptr_bits
        + 1
    )
    directory_entry = 1 + gate_bits + ptr_bits + count_bits
    context_metadata = (
        3
        + row_bits
        + epoch_bits
        + tile_bits
        + 2
        + 2 * count_bits
        + 32
    )
    term = contexts * capacity * term_entry
    directory = contexts * lanes * ways * directory_entry
    metadata = contexts * context_metadata
    active_product = product_bits
    return {
        "term_bits": term,
        "directory_bits": directory,
        "metadata_bits": metadata,
        "register_bits": active_product,
        "total_bits": term + directory + metadata + active_product,
    }


def mark_dominated(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["dominated"] = any(
            other is not row
            and int(other["total_bits"]) <= int(row["total_bits"])
            and float(other["reuse_ratio"]) >= float(row["reuse_ratio"])
            and (
                int(other["total_bits"]) < int(row["total_bits"])
                or float(other["reuse_ratio"]) > float(row["reuse_ratio"])
            )
            for other in rows
        )


def evaluate(
    out_dim: int,
    product_bits: int | None = None,
    storage_format: str = "exact_compact",
) -> dict[str, object]:
    rows = load_trace()
    with VALUE_STATS.open() as handle:
        stats = json.load(handle)
    terms = len(rows)
    lanes = 32
    gate_bits = 9
    if product_bits is None:
        product_bits = out_dim * (gate_bits + 8)
    candidates: list[dict[str, object]] = []

    for ways in (1, 2, 4, 6, 8):
        hits, misses = lane_lru(rows, ways)
        storage = cache_bits(
            lanes=lanes,
            ways=ways,
            product_bits=product_bits,
            gate_bits=gate_bits,
        )
        candidates.append(
            {
                "name": f"lane_lru_{ways}way",
                "kind": "wide_product_cache",
                "product_computes": misses,
                "reuse_ratio": hits / terms,
                **storage,
            }
        )

    lane_gates: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        lane_gates[row["lane"]].add(row["gate"])
    max_lane_gates = max(map(len, lane_gates.values()))
    unique_keys = sum(map(len, lane_gates.values()))
    for slots in (1, 2, 3, 4, 5, 6, 7):
        reused, computes, overflow = lane_no_replace_slots(rows, slots)
        slot_storage = slot_table_bits(
            lanes=lanes,
            slots=slots,
            product_bits=product_bits,
            gate_bits=gate_bits,
        )
        candidates.append(
            {
                "name": f"cross_stage_gate_slot_{slots}",
                "kind": "direct_slot_table",
                "product_computes": computes,
                "reuse_ratio": reused / terms,
                "overflow_computes": overflow,
                **slot_storage,
            }
        )

    gate_frequency: dict[int, int] = defaultdict(int)
    for row in rows:
        gate_frequency[row["gate"]] += 1
    ranked_gates = sorted(
        gate_frequency,
        key=lambda gate: (-gate_frequency[gate], gate),
    )
    for codes in range(1, min(7, len(ranked_gates)) + 1):
        codebook = set(ranked_gates[:codes])
        cached_keys = {
            (row["lane"], row["gate"])
            for row in rows
            if row["gate"] in codebook
        }
        bypass_computes = sum(
            row["gate"] not in codebook for row in rows
        )
        computes = len(cached_keys) + bypass_computes
        storage = frozen_codebook_bits(
            lanes=lanes,
            codes=codes,
            product_bits=product_bits,
        )
        candidates.append(
            {
                "name": f"profile_frozen_gate_codebook_{codes}",
                "kind": "profile_frozen_codebook",
                "codebook": sorted(codebook),
                "product_computes": computes,
                "reuse_ratio": 1.0 - computes / terms,
                "overflow_computes": bypass_computes,
                **storage,
            }
        )

    frontier = stats["row_owned_segmented_frontier"]
    for name, frontier_row in frontier.items():
        capacity_text, ways_text = name.split("_")
        capacity = int(capacity_text[1:])
        ways = int(ways_text[1:])
        storage = dqfs_bits(
            capacity=capacity,
            ways=ways,
            product_bits=product_bits,
        )
        candidates.append(
            {
                "name": f"dqfs_{name}",
                "kind": "narrow_term_reorder",
                "product_computes": frontier_row["product_computes"],
                "reuse_ratio": frontier_row["reuse_ratio"],
                "segments": frontier_row["segments"],
                **storage,
            }
        )

    mark_dominated(candidates)
    return {
        "out_dim": out_dim,
        "product_bits": product_bits,
        "storage_format": storage_format,
        "terms": terms,
        "max_distinct_gates_per_lane": max_lane_gates,
        "unique_lane_gate_keys": unique_keys,
        "candidates": candidates,
    }


def write_report(payload: dict[str, object]) -> None:
    actual = payload["exact_compact_out_dim_4"]
    macro = payload["macro_aligned_out_dim_4"]
    unpacked = payload["unpacked_acc_out_dim_4"]
    sensitivity = payload["sensitivity_out_dim_32"]
    rows = sorted(
        actual["candidates"],
        key=lambda row: (int(row["total_bits"]), -float(row["reuse_ratio"])),
    )
    lines = [
        "# Local5 Product复用结构存储-计算Pareto",
        "",
        "## 1. 口径",
        "",
        "- ordered term来自W6定向RTL trace，共1494项；",
        "- 当前Local5/TCFM为`OUT_DIM=4`、9-bit无符号gate、8-bit有符号"
        "weight、32-bit Acc接口；",
        "- 每路精确product只需17 bit，cache逻辑格式为68 bit；同时报告"
        "72-bit宏对齐与128-bit未压缩接口格式；",
        "- cache计入data、gate/valid tag、LRU age和一个输出寄存器；",
        "- DQFS计入双context term、directory、context metadata和一个活动product；",
        "- 位数是逻辑存储下界，不含SRAM宏拼接、比较器、控制和布线。",
        "",
        "## 2. 真实OUT_DIM=4精确68-bit格式",
        "",
        "| 候选 | 类型 | 存储bit | product计算 | 减少 | 被支配 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['total_bits']} | "
            f"{row['product_computes']} | {row['reuse_ratio']:.2%} | "
            f"{'是' if row['dominated'] else '否'} |"
        )
    dqfs = [
        row
        for row in actual["candidates"]
        if row["kind"] == "narrow_term_reorder"
    ]
    cache4 = next(
        row for row in actual["candidates"] if row["name"] == "lane_lru_4way"
    )
    slot4 = next(
        row
        for row in actual["candidates"]
        if row["name"] == "cross_stage_gate_slot_4"
    )
    slot6 = next(
        row
        for row in actual["candidates"]
        if row["name"] == "cross_stage_gate_slot_6"
    )
    frozen4 = next(
        row
        for row in actual["candidates"]
        if row["name"] == "profile_frozen_gate_codebook_4"
    )
    format_rows = []
    for candidate_payload in (actual, macro, unpacked):
        candidate_cache = next(
            row
            for row in candidate_payload["candidates"]
            if row["name"] == "lane_lru_4way"
        )
        candidate_slot = next(
            row
            for row in candidate_payload["candidates"]
            if row["name"] == "cross_stage_gate_slot_4"
        )
        candidate_frozen = next(
            row
            for row in candidate_payload["candidates"]
            if row["name"] == "profile_frozen_gate_codebook_4"
        )
        format_rows.append(
            (
                candidate_payload["storage_format"],
                candidate_payload["product_bits"],
                candidate_cache["total_bits"],
                candidate_slot["total_bits"],
                candidate_frozen["total_bits"],
            )
        )
    lines += [
        "",
        "## 3. 判定",
        "",
        f"- lane-local 4-way cache：{cache4['total_bits']} bit，"
        f"product减少{cache4['reuse_ratio']:.2%}；",
        f"- 4-slot跨阶段gate表：{slot4['total_bits']} bit，"
        f"product减少{slot4['reuse_ratio']:.2%}，overflow直算"
        f"{slot4['overflow_computes']}次；",
        f"- 6-slot跨阶段gate表：{slot6['total_bits']} bit，"
        f"product减少{slot6['reuse_ratio']:.2%}，overflow直算"
        f"{slot6['overflow_computes']}次；",
        f"- profile-frozen 4-code表：{frozen4['total_bits']} bit，"
        f"product减少{frozen4['reuse_ratio']:.2%}，codebook="
        f"`{frozen4['codebook']}`；",
        f"- {sum(bool(row['dominated']) for row in dqfs)}/{len(dqfs)}个DQFS点"
        "在当前二维Pareto中被其他候选支配；",
        "- 128-bit是Acc接口格式而不是cache逻辑下界，不能用它人为放大"
        "wide-product结构成本；",
        "- 当前更有价值的候选是由上游首次出现时分配gate-slot、下游直接"
        "索引的无替换跨阶段值表。满表后的新gate走精确直算fallback，"
        "不做有损丢弃；",
        "- profile-frozen 4-code表在本trace上进一步优于4-slot首次绑定，"
        "但它是训练/验证profile驱动的专用化，必须用独立样本防止过拟合。",
        "",
        "## 4. 存储格式敏感性",
        "",
        "| 格式 | product bit | W4 LRU bit | 4-slot bit | PF4 bit |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, product_bits, cache_bits_value, slot_bits_value, frozen_bits in (
        format_rows
    ):
        lines.append(
            f"| {name} | {product_bits} | {cache_bits_value} | "
            f"{slot_bits_value} | {frozen_bits} |"
        )
    lines += [
        "",
        "68 bit是数学精确逻辑格式；72 bit是假设SRAM按常见字宽向上对齐的"
        "敏感性；128 bit仅保留为不压缩接口基线。实际面积必须由目标memory "
        "compiler或统一macro拼接得到。",
        "",
        "## 5. OUT_DIM=32敏感性",
        "",
        f"若product扩为{sensitivity['product_bits']} bit，wide cache数据阵列"
        "会线性增长，而DQFS term阵列基本不变。该敏感性只能说明DQFS适合宽"
        "output tile，不能替代当前4-wide硬件的真实比较。",
        "",
        "## 6. 下一步",
        "",
        "1. 实现lane-local 4/6/8-way cache周期与活动基线；",
        "2. 实现4/6-slot跨阶段gate表和4/5-code frozen codebook；",
        "3. 统计compare、tag/LRU写、product写读、weight读和TCFM stall；",
        "4. post-G0验证每lane gate cardinality的mean/p95/max及overflow；",
        "5. DQFS暂停product/TCFM集成，除非宽tile或跨tile复用重新使其进入Pareto。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence": "W6定向ordered RTL term trace；存储位模型；非PPA",
        "exact_compact_out_dim_4": evaluate(
            4, product_bits=68, storage_format="exact_4x17"
        ),
        "macro_aligned_out_dim_4": evaluate(
            4, product_bits=72, storage_format="macro_aligned_72"
        ),
        "unpacked_acc_out_dim_4": evaluate(
            4, product_bits=128, storage_format="unpacked_4x32"
        ),
        "sensitivity_out_dim_32": evaluate(
            32, product_bits=544, storage_format="exact_32x17"
        ),
    }
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    write_report(payload)
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
