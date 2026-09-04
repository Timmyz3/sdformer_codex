#!/usr/bin/env python3
"""Screen exact dyadic gate recoding against the Local5 W4 product cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = (
    ROOT
    / "tb_qfit/vectors/"
    "local5_joint_ep29_active_projection_realw_sample100_population_v3_20260813"
)
DEFAULT_OUT = ROOT / "results/local5_dyadic_gate_projection_screen_20260814"
HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES
LANES = 32
ROLES = 5
GATE_W = 9
PRODUCT_W = 17
COMMON_PRODUCT_W = 13
WEIGHT_W = 8
WAYS = 4
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)
SHIFT_GATES = frozenset((16, 32))
SHIFT_SUB_GATES = frozenset((15, 31))
COMMON_GATES = SHIFT_GATES | SHIFT_SUB_GATES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_memh(path: Path) -> list[int]:
    values: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        try:
            values.append(int(text, 16))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid hex") from exc
    return values


def destination_index(source: int, role: int) -> int | None:
    plane, spatial = divmod(source, HEIGHT * WIDTH)
    y, x = divmod(spatial, WIDTH)
    dy = y + ROLE_DY[role]
    dx = x + ROLE_DX[role]
    if not (0 <= dy < HEIGHT and 0 <= dx < WIDTH):
        return None
    return plane * HEIGHT * WIDTH + dy * WIDTH + dx


def unique_source_gates(
    source: int, valid_words: list[int], gate_words: list[int]
) -> list[int]:
    unique: list[int] = []
    for role in range(ROLES):
        destination = destination_index(source, role)
        if destination is None or ((valid_words[destination] >> role) & 1) == 0:
            continue
        gate = (gate_words[destination] >> (role * GATE_W)) & 0x1FF
        if gate and gate not in unique:
            unique.append(gate)
    return unique


def reconstruct_contexts(
    k_words: list[int], valid_words: list[int], gate_words: list[int]
) -> list[list[tuple[int, int]]]:
    contexts: list[list[tuple[int, int]]] = []
    for source in range(SOURCES):
        gates = unique_source_gates(source, valid_words, gate_words)
        lanes = [lane for lane in range(LANES) if (k_words[source] >> lane) & 1]
        terms = [(lane, gate) for lane in lanes for gate in gates]
        if terms:
            contexts.append(terms)
    return contexts


def lru_stats(terms: Iterable[tuple[int, int]], ways: int = WAYS) -> tuple[int, int]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    hits = 0
    misses = 0
    for lane, gate in terms:
        cache = caches[lane]
        if gate in cache:
            hits += 1
            cache.move_to_end(gate)
            continue
        misses += 1
        if len(cache) == ways:
            cache.popitem(last=False)
        cache[gate] = None
    return hits, misses


def classify_gate(gate: int) -> str:
    if gate in SHIFT_GATES:
        return "shift"
    if gate in SHIFT_SUB_GATES:
        return "shift_sub"
    return "escape"


def dyadic_product(gate: int, weight: int) -> int:
    if not (0 <= gate < (1 << GATE_W)):
        raise ValueError("gate outside 9-bit range")
    if not (-(1 << (WEIGHT_W - 1)) <= weight < (1 << (WEIGHT_W - 1))):
        raise ValueError("weight outside signed INT8 range")
    if gate == 15:
        return (weight << 4) - weight
    if gate == 16:
        return weight << 4
    if gate == 31:
        return (weight << 5) - weight
    if gate == 32:
        return weight << 5
    return gate * weight


def exhaustive_numeric_check() -> dict[str, int]:
    mismatches = 0
    minimum = 0
    maximum = 0
    for gate in range(1 << GATE_W):
        for weight in range(-(1 << (WEIGHT_W - 1)), 1 << (WEIGHT_W - 1)):
            candidate = dyadic_product(gate, weight)
            reference = gate * weight
            mismatches += int(candidate != reference)
            minimum = min(minimum, candidate)
            maximum = max(maximum, candidate)
    if mismatches:
        raise AssertionError("dyadic product is not integer exact")
    if minimum < -(1 << (PRODUCT_W - 1)) or maximum >= (1 << (PRODUCT_W - 1)):
        raise AssertionError("17-bit product contract is insufficient")
    return {
        "vectors": (1 << GATE_W) * (1 << WEIGHT_W),
        "mismatches": mismatches,
        "minimum": minimum,
        "maximum": maximum,
        "signed_product_width": PRODUCT_W,
    }


def frontier_escape_max(valid_words: list[int], gate_words: list[int]) -> int:
    """Maximum raw-gate escapes live in one three-row, one-plane frontier."""
    maximum = 0
    for plane in range(PLANES):
        base = plane * HEIGHT * WIDTH
        for center_y in range(HEIGHT):
            first_y = max(0, center_y - 1)
            last_y = min(HEIGHT - 1, center_y + 1)
            escapes = 0
            for y in range(first_y, last_y + 1):
                for x in range(WIDTH):
                    destination = base + y * WIDTH + x
                    for role in range(ROLES):
                        if ((valid_words[destination] >> role) & 1) == 0:
                            continue
                        gate = (gate_words[destination] >> (role * GATE_W)) & 0x1FF
                        escapes += int(gate not in COMMON_GATES)
            maximum = max(maximum, escapes)
    return maximum


def percentile_ceiling(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    return int(math.ceil(float(np.percentile(values, quantile))))


def output_width_activity(
    *, out_dim: int, terms: int, hits: int, misses: int,
    weight_row_reads: int, pinned_lane_fills: int,
    term_class: dict[str, int],
    raw_relation_storage_bits: int, symbol_relation_storage_bits: int,
) -> dict[str, Any]:
    product_bits = out_dim * PRODUCT_W
    weight_bits = out_dim * WEIGHT_W
    cache_storage_bits = (
        LANES * WAYS * (product_bits + GATE_W + 1 + math.ceil(math.log2(WAYS)))
        + product_bits
    )
    cache_activity = {
        "tag_compare_bits": terms * WAYS * GATE_W,
        "product_read_bits": hits * product_bits,
        "product_write_bits": misses * product_bits,
        "weight_read_bits": misses * weight_bits,
        "vector_multiplier_starts": misses,
    }
    cache_activity["data_array_bits"] = (
        cache_activity["product_read_bits"]
        + cache_activity["product_write_bits"]
        + cache_activity["weight_read_bits"]
    )
    arithmetic_activity = {
        "weight_row_reads": weight_row_reads,
        "weight_read_bits": weight_row_reads * weight_bits,
        "shift_only_terms": term_class["shift"],
        "vector_subtractor_terms": term_class["shift_sub"],
        "vector_multiplier_escape_terms": term_class["escape"],
        "product_sram_read_bits": 0,
        "product_sram_write_bits": 0,
    }
    arithmetic_activity["data_array_bits"] = arithmetic_activity["weight_read_bits"]
    w4_system_storage_bits = raw_relation_storage_bits + cache_storage_bits
    common_product_bits = out_dim * COMMON_PRODUCT_W
    pinned_storage_bits = LANES * len(COMMON_GATES) * common_product_bits + common_product_bits
    pinned_activity = {
        "lane_fills": pinned_lane_fills,
        "weight_read_bits": (
            pinned_lane_fills + term_class["escape"]
        ) * weight_bits,
        "product_write_bits": (
            pinned_lane_fills * len(COMMON_GATES) * common_product_bits
        ),
        "product_read_bits": (
            (term_class["shift"] + term_class["shift_sub"]) * common_product_bits
        ),
        "precompute_vector_subtractor_terms": pinned_lane_fills * 2,
        "vector_multiplier_escape_terms": term_class["escape"],
    }
    pinned_activity["data_array_bits"] = (
        pinned_activity["weight_read_bits"]
        + pinned_activity["product_write_bits"]
        + pinned_activity["product_read_bits"]
    )
    return {
        "out_dim": out_dim,
        "w4_product_cache": {
            "ways": WAYS,
            "hits": hits,
            "misses": misses,
            "storage_bits": cache_storage_bits,
            "activity": cache_activity,
        },
        "raw_relation_dyadic_arithmetic": {
            "relation_encoding": "raw 9-bit gate + valid",
            "activity": arithmetic_activity,
        },
        "symbol_relation_dyadic_arithmetic": {
            "relation_encoding": "3-bit common/escape symbol",
            "activity": arithmetic_activity,
        },
        "symbol_relation_pinned_four_product": {
            "relation_encoding": "3-bit common/escape symbol",
            "common_product_width": COMMON_PRODUCT_W,
            "product_table_storage_bits": pinned_storage_bits,
            "activity": pinned_activity,
        },
        "system_storage_bits": {
            "raw_relation_plus_w4_cache": w4_system_storage_bits,
            "raw_relation_plus_dyadic_arithmetic": raw_relation_storage_bits,
            "symbol_relation_plus_dyadic_arithmetic": symbol_relation_storage_bits,
            "symbol_relation_plus_pinned_four_product": (
                symbol_relation_storage_bits + pinned_storage_bits
            ),
            "raw_dyadic_ratio_vs_w4": (
                raw_relation_storage_bits / w4_system_storage_bits
            ),
            "symbol_dyadic_ratio_vs_w4": (
                symbol_relation_storage_bits / w4_system_storage_bits
            ),
            "symbol_ratio_vs_raw_dyadic": (
                symbol_relation_storage_bits / raw_relation_storage_bits
            ),
            "pinned_four_ratio_vs_symbol_dyadic": (
                (symbol_relation_storage_bits + pinned_storage_bits)
                / symbol_relation_storage_bits
            ),
        },
        "data_array_bit_activity_ratio_vs_w4": (
            arithmetic_activity["data_array_bits"] / cache_activity["data_array_bits"]
        ),
        "weight_read_bit_ratio_vs_w4": (
            arithmetic_activity["weight_read_bits"] / cache_activity["weight_read_bits"]
        ),
    }


def evaluate(vector_dir: Path, expected_groups: int | None = 100) -> dict[str, Any]:
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "local5_active_projection_postg0_vectors_v1":
        raise ValueError("unexpected Local5 vector schema")
    rows = manifest.get("selection", {}).get("rows", [])
    groups = len(rows)
    if expected_groups is not None and groups != expected_groups:
        raise ValueError(f"expected {expected_groups} population groups, got {groups}")

    k_flat = read_memh(vector_dir / "input_k.memh")
    valid_flat = read_memh(vector_dir / "input_valid.memh")
    gates_flat = read_memh(vector_dir / "input_gates.memh")
    expected_terms = read_memh(vector_dir / "expected_terms.memh")
    expected_size = groups * SOURCES
    for name, values in (("K", k_flat), ("valid", valid_flat), ("gate", gates_flat)):
        if len(values) != expected_size:
            raise ValueError(f"{name} entries {len(values)} != {expected_size}")
    if len(expected_terms) != groups:
        raise ValueError("expected term count length mismatch")

    term_class = {"shift": 0, "shift_sub": 0, "escape": 0}
    relation_class = {"common": 0, "escape": 0}
    term_gate_histogram: Counter[int] = Counter()
    relation_gate_histogram: Counter[int] = Counter()
    total_terms = 0
    total_contexts = 0
    weight_row_reads = 0
    pinned_lane_fills = 0
    cache_hits = 0
    cache_misses = 0
    frontier_escapes: list[int] = []
    group_escapes: list[int] = []
    stage_terms: dict[str, int] = defaultdict(int)

    for group, row in enumerate(rows):
        start = group * SOURCES
        stop = start + SOURCES
        k_words = k_flat[start:stop]
        valid_words = valid_flat[start:stop]
        gate_words = gates_flat[start:stop]
        contexts = reconstruct_contexts(k_words, valid_words, gate_words)
        terms = [term for context in contexts for term in context]
        if len(terms) != expected_terms[group]:
            raise AssertionError(
                f"group {group}: reconstructed {len(terms)} != {expected_terms[group]}"
            )
        hits, misses = lru_stats(terms)
        cache_hits += hits
        cache_misses += misses
        total_terms += len(terms)
        total_contexts += len(contexts)
        pinned_lane_fills += len({lane for lane, _gate in terms})
        stage_terms[str(int(row["stage"]))] += len(terms)
        for context in contexts:
            weight_row_reads += len({lane for lane, _gate in context})
            for _lane, gate in context:
                term_gate_histogram[gate] += 1
                term_class[classify_gate(gate)] += 1
        group_escape_count = 0
        for valid_word, gate_word in zip(valid_words, gate_words, strict=True):
            for role in range(ROLES):
                if ((valid_word >> role) & 1) == 0:
                    continue
                gate = (gate_word >> (role * GATE_W)) & 0x1FF
                relation_gate_histogram[gate] += 1
                gate_class = "common" if gate in COMMON_GATES else "escape"
                relation_class[gate_class] += 1
                group_escape_count += int(gate_class == "escape")
        frontier_escapes.append(frontier_escape_max(valid_words, gate_words))
        group_escapes.append(group_escape_count)

    if total_terms != cache_hits + cache_misses or total_terms != sum(term_class.values()):
        raise AssertionError("term accounting is not conservative")

    frontier_exception_entry_bits = 6 + 3 + GATE_W + 1
    t450_exception_entry_bits = 9 + 3 + GATE_W + 1
    frontier_p99 = percentile_ceiling(frontier_escapes, 99)
    frontier_max = max(frontier_escapes, default=0)
    group_escape_max = max(group_escapes, default=0)
    compressed_frontier_bits = (
        3 * WIDTH * ROLES * 3
        + frontier_max * frontier_exception_entry_bits
    )
    raw_frontier_gate_bits = 3 * WIDTH * ROLES * (GATE_W + 1)
    frontier_k_bits = 3 * WIDTH * LANES
    raw_frontier_total_bits = raw_frontier_gate_bits + frontier_k_bits
    compressed_frontier_total_bits = compressed_frontier_bits + frontier_k_bits
    raw_t450_gate_bits = SOURCES * ROLES * (GATE_W + 1)
    compressed_t450_gate_bits = (
        SOURCES * ROLES * 3 + group_escape_max * t450_exception_entry_bits
    )
    t450_k_bits = SOURCES * LANES
    raw_t450_total_bits = raw_t450_gate_bits + t450_k_bits
    compressed_t450_total_bits = compressed_t450_gate_bits + t450_k_bits
    term_common_fraction = (term_class["shift"] + term_class["shift_sub"]) / total_terms
    relation_total = sum(relation_class.values())
    relation_common_fraction = relation_class["common"] / relation_total
    out2 = output_width_activity(
        out_dim=2, terms=total_terms, hits=cache_hits, misses=cache_misses,
        weight_row_reads=weight_row_reads, pinned_lane_fills=pinned_lane_fills,
        term_class=term_class,
        raw_relation_storage_bits=raw_t450_total_bits,
        symbol_relation_storage_bits=compressed_t450_total_bits,
    )
    out32 = output_width_activity(
        out_dim=32, terms=total_terms, hits=cache_hits, misses=cache_misses,
        weight_row_reads=weight_row_reads, pinned_lane_fills=pinned_lane_fills,
        term_class=term_class,
        raw_relation_storage_bits=raw_t450_total_bits,
        symbol_relation_storage_bits=compressed_t450_total_bits,
    )
    frontier_gate_ratio = compressed_frontier_bits / raw_frontier_gate_bits
    frontier_total_ratio = compressed_frontier_total_bits / raw_frontier_total_bits
    t450_gate_ratio = compressed_t450_gate_bits / raw_t450_gate_bits
    t450_total_ratio = compressed_t450_total_bits / raw_t450_total_bits
    passes = {
        "term_common_fraction_ge_0p999": term_common_fraction >= 0.999,
        "relation_common_fraction_ge_0p999": relation_common_fraction >= 0.999,
        "current_t450_total_storage_ratio_le_0p65": t450_total_ratio <= 0.65,
        "out2_data_array_bit_activity_ratio_le_0p5": (
            out2["data_array_bit_activity_ratio_vs_w4"] <= 0.5
        ),
    }
    status = "CONDITIONAL_RTL_PPA" if all(passes.values()) else "NO_GO"
    return {
        "schema": "local5_dyadic_gate_projection_screen_v2",
        "status": status,
        "evidence": "[profile-qualified-trace]+[bounded-storage-activity-model]",
        "scope": (
            f"{groups} profile groups; OUT_DIM=2 primary plus OUT_DIM=32 sensitivity; "
            "not encoder"
        ),
        "source": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "groups": groups,
            "stage_counts": manifest["selection"].get("stage_counts", {}),
            "stage_terms": dict(stage_terms),
        },
        "integer_reference": exhaustive_numeric_check(),
        "workload": {
            "terms": total_terms,
            "active_source_contexts": total_contexts,
            "pinned_lane_fills": pinned_lane_fills,
            "term_gate_classes": term_class,
            "term_gate_histogram": {
                str(gate): count for gate, count in term_gate_histogram.most_common()
            },
            "term_common_fraction": term_common_fraction,
            "relation_gate_classes": relation_class,
            "relation_gate_histogram": {
                str(gate): count for gate, count in relation_gate_histogram.most_common()
            },
            "relation_common_fraction": relation_common_fraction,
            "frontier_escape_entries": {
                "mean": float(np.mean(frontier_escapes)),
                "p95": percentile_ceiling(frontier_escapes, 95),
                "p99": frontier_p99,
                "max": max(frontier_escapes, default=0),
            },
            "group_escape_entries": {
                "mean": float(np.mean(group_escapes)),
                "p95": percentile_ceiling(group_escapes, 95),
                "p99": percentile_ceiling(group_escapes, 99),
                "max": group_escape_max,
            },
        },
        "dyadic_candidate": {
            "exact_codes": {
                "15": "(W << 4) - W",
                "16": "W << 4",
                "31": "(W << 5) - W",
                "32": "W << 5",
                "other": "exact multiplier fallback",
            },
            "current_t450_storage": {
                "raw_gate_bits": raw_t450_gate_bits,
                "compressed_gate_bits_exact_max": compressed_t450_gate_bits,
                "gate_ratio": t450_gate_ratio,
                "common_k_bits": t450_k_bits,
                "raw_total_bits": raw_t450_total_bits,
                "compressed_total_bits_exact_max": compressed_t450_total_bits,
                "total_ratio": t450_total_ratio,
                "exception_entry_bits": t450_exception_entry_bits,
            },
            "three_row_lower_bound": {
                "raw_gate_bits": raw_frontier_gate_bits,
                "compressed_gate_bits_exact_max": compressed_frontier_bits,
                "gate_ratio": frontier_gate_ratio,
                "common_k_bits": frontier_k_bits,
                "raw_total_bits": raw_frontier_total_bits,
                "compressed_total_bits_exact_max": compressed_frontier_total_bits,
                "total_ratio": frontier_total_ratio,
                "exception_entry_bits": frontier_exception_entry_bits,
                "p99_escape_entries": frontier_p99,
                "max_escape_entries": frontier_max,
            },
            "output_width_comparison": {
                "2": out2,
                "32": out32,
            },
            "baseline_isolation": {
                "arithmetic_only": (
                    "raw relation banks plus the same dyadic arithmetic; isolates multiplier specialization"
                ),
                "full_candidate": (
                    "3-bit relation symbols plus dyadic arithmetic; incremental architecture effect is relation storage/access"
                ),
            },
        },
        "admission": {
            "gates": passes,
            "result": status,
            "next": (
                "Only an exact same-throughput leaf with matched SRAM/SDC and Acc32 miter may advance."
                if status == "CONDITIONAL_RTL_PPA"
                else "Stop before RTL."
            ),
        },
        "claim_boundary": [
            "The four-code recoding is exact in a sufficiently wide signed product path; RTL still must prove finite-width equivalence.",
            "Bit counts are activity proxies without SRAM macro energy coefficients.",
            "No cycle speedup is claimed: both designs may deliver one term per cycle.",
            "The exception sidecar is a bounded logical model, not a synthesized memory implementation.",
            "OUT_DIM=2 matches the current tile width; OUT_DIM=32 is sensitivity only. Neither is an encoder result.",
        ],
    }


def render(report: dict[str, Any]) -> str:
    workload = report["workload"]
    candidate = report["dyadic_candidate"]
    out2 = candidate["output_width_comparison"]["2"]
    out32 = candidate["output_width_comparison"]["32"]
    cache = out2["w4_product_cache"]
    t450 = candidate["current_t450_storage"]
    frontier = candidate["three_row_lower_bound"]
    ca = cache["activity"]
    da = out2["symbol_relation_dyadic_arithmetic"]["activity"]
    pa = out2["symbol_relation_pinned_four_product"]["activity"]
    ss = out2["system_storage_bits"]
    escapes = workload["frontier_escape_entries"]
    return f"""# Local5 精确 Dyadic Gate 投影筛选

## 裁决

`{report['status']}`。这是 `[模型]`：`{report['source']['groups']}` 个真实 profile group；
`OUT_DIM=2` 对齐当前 tile，`OUT_DIM=32` 仅为敏感性，均不是 encoder。

## 真实 Gate 结构

| term 类别 | 数量 | 比例 |
|---|---:|---:|
| `16/32`: shift-only | {workload['term_gate_classes']['shift']} | {workload['term_gate_classes']['shift']/workload['terms']:.4%} |
| `15/31`: shift-minus-W | {workload['term_gate_classes']['shift_sub']} | {workload['term_gate_classes']['shift_sub']/workload['terms']:.4%} |
| exact multiplier escape | {workload['term_gate_classes']['escape']} | {workload['term_gate_classes']['escape']/workload['terms']:.4%} |

四个 dyadic/dyadic-minus-one gate 覆盖 `{workload['term_common_fraction']:.4%}`
真实投影 term；relation entry 覆盖 `{workload['relation_common_fraction']:.4%}`。
三行 frontier 的 raw escape 数为 mean `{escapes['mean']:.3f}`、p95 `{escapes['p95']}`、
p99 `{escapes['p99']}`、max `{escapes['max']}`。

## W4 Product Cache 强基线

| 指标 | W4 cache | pinned-four | Dyadic 候选 |
|---|---:|---:|---:|
| term | {workload['terms']} | {workload['terms']} | {workload['terms']} |
| product hit/miss | {cache['hits']}/{cache['misses']} | 固定四项 | 不存 product |
| OUT2 product/cache storage bit | {cache['storage_bits']} | {out2['symbol_relation_pinned_four_product']['product_table_storage_bits']} | 0 |
| product SRAM read bit | {ca['product_read_bits']} | {pa['product_read_bits']} | 0 |
| product SRAM write bit | {ca['product_write_bits']} | {pa['product_write_bits']} | 0 |
| weight read bit | {ca['weight_read_bits']} | {pa['weight_read_bits']} | {da['weight_read_bits']} |
| data-array bit activity | {ca['data_array_bits']} | {pa['data_array_bits']} | {da['data_array_bits']} |
| vector multiplier start/escape | {ca['vector_multiplier_starts']} | {pa['vector_multiplier_escape_terms']} | {da['vector_multiplier_escape_terms']} |
| vector subtractor active term | 0 | {pa['precompute_vector_subtractor_terms']} | {da['vector_subtractor_terms']} |

OUT2 候选数据阵列 bit 活动为 W4 的 `{out2['data_array_bit_activity_ratio_vs_w4']:.4f}x`，
但 weight read bit 是 W4 的 `{out2['weight_read_bit_ratio_vs_w4']:.4f}x`。
该比值不是能量：weight SRAM 与 product SRAM 的每 bit 能量必须由同一 macro/SAIF
复核，当前 `OUT_DIM=2` 的两路 subtractor 逻辑活动也尚未计入。

## Relation 存储对象

当前生产 tile 实际例化深度 450 的 K/五 role 同步 bank，不是理想 3 行环。因此分开
报告现行实体状态和理论下界；两者的 exact sidecar 均按观测最大 escape 容量，而非
p99 容量计：

| 存储边界 | raw gate | symbol+exact sidecar | gate 比例 | 含公共 K32 总比例 |
|---|---:|---:|---:|---:|
| 当前 T450 bank | {t450['raw_gate_bits']} | {t450['compressed_gate_bits_exact_max']} | {t450['gate_ratio']:.4f}x | {t450['total_ratio']:.4f}x |
| 3 行理论下界 | {frontier['raw_gate_bits']} | {frontier['compressed_gate_bits_exact_max']} | {frontier['gate_ratio']:.4f}x | {frontier['total_ratio']:.4f}x |

这改变的是 role relation bank 的存储对象；公共 K32 bank 不变，所以总存储降幅小于
只看 gate payload 的降幅。exception table 需要按地址/role 查找，其 CAM/SRAM 端口
和比较活动尚未物理化。

## 必须保留的算术隔离基线

`raw relation + dyadic arithmetic` 与 `3-bit symbol relation + dyadic arithmetic`
具有完全相同的投影活动；后者相对前者的新增作用只在 relation 存储/读取。也就是说，
shift/shift-sub 替代 multiplier 的收益属于算术特化，不能归功于符号 relation 数据流。

现行 `T450 + OUT_DIM=2` 同边界存储账本为：

| 对照 | relation/K + product cache bit | 相对 W4 | 含义 |
|---|---:|---:|---|
| raw relation + W4 | {ss['raw_relation_plus_w4_cache']} | 1.0000x | 强缓存基线 |
| raw relation + dyadic arithmetic | {ss['raw_relation_plus_dyadic_arithmetic']} | {ss['raw_dyadic_ratio_vs_w4']:.4f}x | 只隔离去 product cache/算术特化 |
| symbol relation + dyadic arithmetic | {ss['symbol_relation_plus_dyadic_arithmetic']} | {ss['symbol_dyadic_ratio_vs_w4']:.4f}x | 再加入 relation 存储对象变窄 |
| symbol relation + pinned-four product | {ss['symbol_relation_plus_pinned_four_product']} | {ss['symbol_relation_plus_pinned_four_product']/ss['raw_relation_plus_w4_cache']:.4f}x | 针对四码的最危险强基线 |

因此符号 relation 相对纯算术基线的独立存储比例只有
`{ss['symbol_ratio_vs_raw_dyadic']:.4f}x`；是否值得付出 decoder/escape 端口代价，仍需
同 SRAM 规则的物理代理决定。
Pinned-four 只多占 dyadic 候选 `{ss['pinned_four_ratio_vs_symbol_dyadic']-1:.2%}`
系统存储，却把 term-time shift/sub 改成每个 epoch/lane 一次预生成；因此在它被物理
结果击败前，dyadic 候选不能作为架构贡献。

OUT32 敏感性中，W4 storage 为
`{out32['w4_product_cache']['storage_bits']}` bit，候选/W4 data-array bit 比仍为
`{out32['data_array_bit_activity_ratio_vs_w4']:.4f}x`；该列只可放附录。

## 下一闸门

只有同时满足以下条件才可晋级：

1. 同吞吐 exact leaf，`15/16/31/32` 与 raw multiplier 逐项 miter；
2. rare escape 在任意反压下保序，Acc32 真实权重 0 mismatch；
3. W4 cache、symbol+pinned-four、raw-relation dyadic arithmetic、
   symbol-relation dyadic arithmetic使用相同 weight/product SRAM 规则和 SDC；
4. 计入 32 路 subtractor、decoder、exception table 后仍有 area/energy Pareto；
5. 不把该模型写成周期加速、ASIC PPA 或 encoder 收益。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-groups", type=int, default=100)
    args = parser.parse_args()
    report = evaluate(args.vectors, expected_groups=args.expected_groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(render(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
