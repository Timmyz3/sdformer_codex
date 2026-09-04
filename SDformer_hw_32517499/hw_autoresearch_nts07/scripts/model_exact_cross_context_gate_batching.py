#!/usr/bin/env python3
"""Local5精确跨context门码批处理（ECGB）的有界Pareto模型。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = (
    ROOT
    / "results/qfit_local5_projection_tile_yosys_20260731"
    / "ordered_term_trace.csv"
)
DEFAULT_OUT = ROOT / "results/exact_cross_context_gate_batching_20260801"


def clog2(value: int) -> int:
    return max(1, math.ceil(math.log2(max(2, value))))


def load_contexts(path: Path) -> list[list[dict[str, int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError("term trace为空")
    contexts: list[list[dict[str, int]]] = []
    current_key = None
    current: list[dict[str, int]] = []
    for row in rows:
        key = (row["plane"], row["y"], row["x"])
        if current and key != current_key:
            contexts.append(current)
            current = []
        current_key = key
        current.append(row)
    contexts.append(current)
    return contexts


def flatten(contexts: list[list[dict[str, int]]]) -> list[dict[str, int]]:
    return [row for context in contexts for row in context]


def lru_misses(rows: list[dict[str, int]], ways: int) -> int:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    misses = 0
    for row in rows:
        cache = caches[row["lane"]]
        gate = row["gate"]
        if gate in cache:
            cache.move_to_end(gate)
            continue
        misses += 1
        if len(cache) == ways:
            cache.popitem(last=False)
        cache[gate] = None
    return misses


def reorder_batches(
    contexts: list[list[dict[str, int]]], batch: int
) -> tuple[list[dict[str, int]], int, int]:
    groups, max_terms, max_gate_slots_per_lane = make_batches(contexts, batch)
    return flatten(groups), max_terms, max_gate_slots_per_lane


def make_batches(
    contexts: list[list[dict[str, int]]], batch: int
) -> tuple[list[list[dict[str, int]]], int, int]:
    output: list[list[dict[str, int]]] = []
    max_terms = 0
    max_gate_slots_per_lane = 0
    for offset in range(0, len(contexts), batch):
        group = contexts[offset : offset + batch]
        tagged = []
        lane_gates: dict[int, set[int]] = defaultdict(set)
        for context_index, context in enumerate(group):
            for row in context:
                item = dict(row)
                item["context_index"] = context_index
                tagged.append(item)
                lane_gates[row["lane"]].add(row["gate"])
        max_terms = max(max_terms, len(tagged))
        max_gate_slots_per_lane = max(
            max_gate_slots_per_lane,
            max((len(values) for values in lane_gates.values()), default=0),
        )
        # B=1是原序双buffer基线；跨context时才启用gate-key聚类。
        output.append(
            sorted(
                tagged,
                key=lambda row: (
                    row["lane"],
                    row["gate"],
                    row["context_index"],
                    row["seq"],
                ),
            ) if batch > 1 else tagged
        )
    return output, max_terms, max_gate_slots_per_lane


def lru_misses_by_group(
    groups: list[list[dict[str, int]]], ways: int
) -> list[int]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    result = []
    for rows in groups:
        misses = 0
        for row in rows:
            cache = caches[row["lane"]]
            gate = row["gate"]
            if gate in cache:
                cache.move_to_end(gate)
                continue
            misses += 1
            if len(cache) == ways:
                cache.popitem(last=False)
            cache[gate] = None
        result.append(misses)
    return result


def pingpong_cycles(
    groups: list[list[dict[str, int]]],
    misses: list[int],
    miss_penalty: int,
) -> int:
    """双buffer有限调度：builder最多领先executor一个完整batch。"""

    if len(groups) != len(misses) or not groups:
        raise ValueError("group/miss列表不合法")
    build_finish: list[int] = []
    execute_finish: list[int] = []
    for index, rows in enumerate(groups):
        prior_build = build_finish[-1] if build_finish else 0
        reused_buffer_free = execute_finish[index - 2] if index >= 2 else 0
        build_start = max(prior_build, reused_buffer_free)
        build_finish.append(build_start + len(rows))
        execute_start = max(
            build_finish[-1], execute_finish[-1] if execute_finish else 0
        )
        execute_finish.append(
            execute_start + len(rows) + miss_penalty * misses[index]
        )
    return execute_finish[-1]


def product_cache_bits(*, out_dim: int, ways: int) -> int:
    lanes = 32
    gate_bits = 9
    product_bits = out_dim * 17
    entries = lanes * ways
    replacement = entries * (clog2(ways) if ways > 1 else 0)
    return (
        entries * product_bits
        + entries * (gate_bits + 1)
        + replacement
        + product_bits
    )


def ecgb_bits(
    *, batch: int, capacity: int, slots: int, out_dim: int, contexts: int = 2
) -> dict[str, int]:
    ptr_bits = clog2(capacity + 1)
    context_bits = clog2(batch)
    slot_bits = clog2(slots)
    product_bits = out_dim * 17
    # 每term只保存context、lane、gate-slot、5-bit mask和同bucket next pointer。
    term_entry_bits = context_bits + 5 + slot_bits + 5 + ptr_bits + 1
    term_array = contexts * capacity * term_entry_bits
    # 每lane/slot一对head/tail和valid；gate表保存slot对应的精确9-bit gate。
    directory = contexts * 32 * slots * (2 * ptr_bits + 1)
    vocabulary = contexts * 32 * slots * 9
    context_table = contexts * batch * 10  # plane/y/x + complete
    product_register = product_bits
    total = term_array + directory + vocabulary + context_table + product_register
    return {
        "term_entry_bits": term_entry_bits,
        "term_array_bits": term_array,
        "directory_bits": directory,
        "vocabulary_bits": vocabulary,
        "context_table_bits": context_table,
        "product_register_bits": product_register,
        "total_bits": total,
    }


def evaluate(path: Path) -> dict[str, object]:
    contexts = load_contexts(path)
    original = flatten(contexts)
    terms = len(original)
    baseline = {
        str(ways): {
            "ways": ways,
            "product_computes": lru_misses(original, ways),
            "storage_out4_bits": product_cache_bits(out_dim=4, ways=ways),
            "storage_out32_bits": product_cache_bits(out_dim=32, ways=ways),
        }
        for ways in (1, 2, 4, 6)
    }
    baseline_groups, _, _ = make_batches(contexts, 1)
    baseline_group_misses = lru_misses_by_group(baseline_groups, 1)
    baseline_pingpong = {
        str(penalty): pingpong_cycles(
            baseline_groups, baseline_group_misses, penalty
        )
        for penalty in (1, 2, 4)
    }
    rows = []
    for batch in (1, 2, 4, 8, 16, len(contexts)):
        batch_groups, capacity, slots = make_batches(contexts, batch)
        reordered = flatten(batch_groups)
        misses = lru_misses(reordered, 1)
        group_misses = lru_misses_by_group(batch_groups, 1)
        row: dict[str, object] = {
            "batch_contexts": batch,
            "capacity_terms": capacity,
            "max_gate_slots_per_lane": slots,
            "product_computes": misses,
            "product_compute_reduction_vs_original_1way": 1.0
            - misses / baseline["1"]["product_computes"],
            "storage_out4": ecgb_bits(
                batch=batch, capacity=capacity, slots=slots, out_dim=4
            ),
            "storage_out32": ecgb_bits(
                batch=batch, capacity=capacity, slots=slots, out_dim=32
            ),
            "cycle_sensitivity": {},
            "finite_pingpong": {},
        }
        for penalty in (1, 2, 4):
            base_cycles = terms + penalty * baseline["1"]["product_computes"]
            cycles = terms + penalty * misses
            row["cycle_sensitivity"][str(penalty)] = {
                "miss_penalty": penalty,
                "cycles": cycles,
                "speedup_vs_original_1way": base_cycles / cycles,
            }
            finite_cycles = pingpong_cycles(
                batch_groups, group_misses, penalty
            )
            row["finite_pingpong"][str(penalty)] = {
                "miss_penalty": penalty,
                "cycles": finite_cycles,
                "speedup_vs_b1_pingpong": (
                    baseline_pingpong[str(penalty)] / finite_cycles
                ),
            }
        rows.append(row)
    candidates = []
    for ways, value in baseline.items():
        candidates.append(
            {
                "name": f"wide_cache_w{ways}",
                "kind": "wide_product_cache",
                "storage_out4_bits": value["storage_out4_bits"],
                "storage_out32_bits": value["storage_out32_bits"],
                "product_computes": value["product_computes"],
            }
        )
    for row in rows:
        candidates.append(
            {
                "name": f"ecgb_b{row['batch_contexts']}",
                "kind": "narrow_term_reorder",
                "storage_out4_bits": row["storage_out4"]["total_bits"],
                "storage_out32_bits": row["storage_out32"]["total_bits"],
                "product_computes": row["product_computes"],
            }
        )
    for width in (4, 32):
        storage_key = f"storage_out{width}_bits"
        for candidate in candidates:
            candidate[f"dominated_out{width}"] = any(
                other is not candidate
                and other[storage_key] <= candidate[storage_key]
                and other["product_computes"] <= candidate["product_computes"]
                and (
                    other[storage_key] < candidate[storage_key]
                    or other["product_computes"] < candidate["product_computes"]
                )
                for other in candidates
            )
    return {
        "schema": "exact_cross_context_gate_batching_v1",
        "evidence": "Local5 W6 directed ordered RTL trace + bounded storage/cycle model",
        "trace": str(path.resolve()),
        "terms": terms,
        "contexts": len(contexts),
        "context_terms_min": min(map(len, contexts)),
        "context_terms_max": max(map(len, contexts)),
        "unique_lane_gate_keys": len(
            {(row["lane"], row["gate"]) for row in original}
        ),
        "baseline_lru": baseline,
        "baseline_pingpong_cycles": baseline_pingpong,
        "ecgb": rows,
        "storage_compute_candidates": candidates,
        "exactness_contract": (
            "仅在同weight/theta epoch和不可见final barrier内重排；不删除term；"
            "INT32无溢出时整数累加结果不变"
        ),
    }


def markdown(report: dict[str, object]) -> str:
    base = report["baseline_lru"]
    lines = [
        "# ECGB 精确跨 Context 门码批处理模型",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：`[rtl-directed-trace] + [bounded-model]`；不是post-G0 profile或PPA。",
        "",
        "## 结论",
        "",
        f"真实trace含 {report['terms']} 个term、{report['contexts']} 个context、"
        f"{report['unique_lane_gate_keys']} 个唯一 `(lane,gate)` 键。ECGB不做稀疏"
        "预测或删除，而是在有限context组内按键重排，使一个窄term buffer替代多路宽"
        "product cache。",
        "",
        "| B | 最大term容量 | 每lane最大slot | product计算 | 相对原序1-way减少 | OUT4双buffer bit | OUT32双buffer bit |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ecgb"]:
        lines.append(
            f"| {row['batch_contexts']} | {row['capacity_terms']} | "
            f"{row['max_gate_slots_per_lane']} | {row['product_computes']} | "
            f"{row['product_compute_reduction_vs_original_1way']:.2%} | "
            f"{row['storage_out4']['total_bits']} | "
            f"{row['storage_out32']['total_bits']} |"
        )
    lines += [
        "",
        "## 宽 Product Cache 对照",
        "",
        "| ways | product计算 | OUT4 bit | OUT32 bit |",
        "|---:|---:|---:|---:|",
    ]
    for ways in ("1", "2", "4", "6"):
        row = base[ways]
        lines.append(
            f"| {ways} | {row['product_computes']} | "
            f"{row['storage_out4_bits']} | {row['storage_out32_bits']} |"
        )
    lines += [
        "",
        "## 周期敏感性",
        "",
        "先给完全隐藏建表的理想稳态，再给双buffer有限调度。两者都不包含真实SRAM "
        "latency或bank ready。",
        "",
        "| B | penalty=1 | penalty=2 | penalty=4 |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["ecgb"]:
        values = row["cycle_sensitivity"]
        lines.append(
            f"| {row['batch_contexts']} | "
            f"{values['1']['speedup_vs_original_1way']:.3f}x | "
            f"{values['2']['speedup_vs_original_1way']:.3f}x | "
            f"{values['4']['speedup_vs_original_1way']:.3f}x |"
        )
    lines += [
        "",
        "### 双 Buffer 有限 Trace",
        "",
        "builder每拍接收一个term，executor按term与miss服务；builder最多领先一个完整"
        "batch。B=1原序作为同结构基线。",
        "",
        "| B | penalty=1 | penalty=2 | penalty=4 |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["ecgb"]:
        values = row["finite_pingpong"]
        lines.append(
            f"| {row['batch_contexts']} | "
            f"{values['1']['cycles']} / {values['1']['speedup_vs_b1_pingpong']:.3f}x | "
            f"{values['2']['cycles']} / {values['2']['speedup_vs_b1_pingpong']:.3f}x | "
            f"{values['4']['cycles']} / {values['4']['speedup_vs_b1_pingpong']:.3f}x |"
        )
    lines += [
        "",
        "## 存储-计算 Pareto",
        "",
        "| 候选 | product计算 | OUT4 bit | OUT4被支配 | OUT32 bit | OUT32被支配 |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in report["storage_compute_candidates"]:
        lines.append(
            f"| {row['name']} | {row['product_computes']} | "
            f"{row['storage_out4_bits']} | {'是' if row['dominated_out4'] else '否'} | "
            f"{row['storage_out32_bits']} | {'是' if row['dominated_out32'] else '否'} |"
        )
    lines += [
        "",
        "## 精确性与限制",
        "",
        "- 只允许在同weight/theta epoch内重排；",
        "- context final在全部term完成前不可见；",
        "- 不允许跨ATLIF/skip状态边界合并；",
        "- 必须证明INT32 accumulator不溢出；",
        "- 当前trace是W6定向RTL trace，不是fullres post-G0多样本；",
        "- OUT32结果是位宽敏感性，不是已经实现的32-wide Local5 backend。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = evaluate(args.trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
