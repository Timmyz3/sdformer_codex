#!/usr/bin/env python3
"""Screen Local5 cross-context gate batching on the frozen 100-sample vectors.

This is a bounded storage/work model. It reconstructs the exact source-major
term order consumed by the RTL term builder and compares narrow-term batching
against persistent per-lane wide-product LRU caches.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = (
    ROOT
    / "tb_qfit/vectors/"
    "local5_joint_ep29_active_projection_realw_sample100_population_v3_20260813"
)
DEFAULT_OUT = ROOT / "results/local5_ecgb_population_screen_20260813"
HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES
LANES = 32
ROLES = 5
GATE_W = 9
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)


def clog2(value: int) -> int:
    return max(1, math.ceil(math.log2(max(2, value))))


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    return int(math.ceil(float(np.percentile(values, quantile))))


def read_memh(path: Path) -> list[int]:
    values = []
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        try:
            values.append(int(text, 16))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid hex value") from exc
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
    *,
    source: int,
    valid_words: list[int],
    gate_words: list[int],
) -> list[int]:
    unique: list[int] = []
    for role in range(ROLES):
        destination = destination_index(source, role)
        if destination is None:
            continue
        if ((valid_words[destination] >> role) & 1) == 0:
            continue
        gate = (gate_words[destination] >> (role * GATE_W)) & ((1 << GATE_W) - 1)
        if gate and gate not in unique:
            unique.append(gate)
    return unique


def reconstruct_group_contexts(
    *,
    k_words: list[int],
    valid_words: list[int],
    gate_words: list[int],
) -> list[list[tuple[int, int]]]:
    contexts: list[list[tuple[int, int]]] = []
    for source in range(SOURCES):
        gates = unique_source_gates(
            source=source,
            valid_words=valid_words,
            gate_words=gate_words,
        )
        lanes = [lane for lane in range(LANES) if (k_words[source] >> lane) & 1]
        terms = [(lane, gate) for lane in lanes for gate in gates]
        if terms:
            contexts.append(terms)
    return contexts


def lru_misses(terms: Iterable[tuple[int, int]], ways: int) -> int:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    misses = 0
    for lane, gate in terms:
        cache = caches[lane]
        if gate in cache:
            cache.move_to_end(gate)
            continue
        misses += 1
        if len(cache) == ways:
            cache.popitem(last=False)
        cache[gate] = None
    return misses


def make_batches(
    contexts: list[list[tuple[int, int]]], batch_size: int
) -> list[dict[str, object]]:
    batches: list[dict[str, object]] = []
    for offset in range(0, len(contexts), batch_size):
        group = contexts[offset : offset + batch_size]
        original = [term for context in group for term in context]
        reordered = sorted(original)
        lane_gates: dict[int, set[int]] = defaultdict(set)
        for lane, gate in original:
            lane_gates[lane].add(gate)
        batches.append(
            {
                "terms": len(original),
                "slots": max((len(gates) for gates in lane_gates.values()), default=0),
                "original_w1_misses": lru_misses(original, 1),
                "reordered_w1_misses": lru_misses(reordered, 1),
            }
        )
    return batches


def product_cache_bits(*, out_dim: int, ways: int) -> int:
    gate_bits = 9
    product_bits = out_dim * 17
    entries = LANES * ways
    replacement = entries * (clog2(ways) if ways > 1 else 0)
    return entries * (product_bits + gate_bits + 1) + replacement + product_bits


def ecgb_bits(*, batch: int, capacity: int, slots: int, out_dim: int) -> int:
    contexts = 2
    ptr_bits = clog2(capacity + 1)
    context_bits = clog2(batch)
    slot_bits = clog2(slots)
    product_bits = out_dim * 17
    term_entry_bits = context_bits + 5 + slot_bits + 5 + ptr_bits + 1
    term_array = contexts * capacity * term_entry_bits
    directory = contexts * LANES * slots * (2 * ptr_bits + 1)
    vocabulary = contexts * LANES * slots * GATE_W
    context_table = contexts * batch * 10
    return term_array + directory + vocabulary + context_table + product_bits


def summarize_int(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(array.max()),
    }


def evaluate(vector_dir: Path) -> dict[str, object]:
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["selection"]["rows"]
    groups = len(rows)
    flat_k = read_memh(vector_dir / "input_k.memh")
    flat_valid = read_memh(vector_dir / "input_valid.memh")
    flat_gates = read_memh(vector_dir / "input_gates.memh")
    expected_terms = read_memh(vector_dir / "expected_terms.memh")
    expected_size = groups * SOURCES
    for name, values in (
        ("input_k", flat_k),
        ("input_valid", flat_valid),
        ("input_gates", flat_gates),
    ):
        if len(values) != expected_size:
            raise ValueError(f"{name}: {len(values)} entries, expected {expected_size}")
    if len(expected_terms) != groups:
        raise ValueError("expected_terms group count mismatch")

    group_contexts: list[list[list[tuple[int, int]]]] = []
    group_terms: list[list[tuple[int, int]]] = []
    stages: list[int] = []
    for group, row in enumerate(rows):
        start = group * SOURCES
        stop = start + SOURCES
        contexts = reconstruct_group_contexts(
            k_words=flat_k[start:stop],
            valid_words=flat_valid[start:stop],
            gate_words=flat_gates[start:stop],
        )
        terms = [term for context in contexts for term in context]
        if len(terms) != expected_terms[group]:
            raise AssertionError(
                f"group {group}: reconstructed terms {len(terms)} != {expected_terms[group]}"
            )
        group_contexts.append(contexts)
        group_terms.append(terms)
        stages.append(int(row["stage"]))

    baseline = {}
    total_terms = sum(map(len, group_terms))
    for ways in (1, 2, 4, 6):
        misses_by_group = [lru_misses(terms, ways) for terms in group_terms]
        baseline[str(ways)] = {
            "ways": ways,
            "product_computes": sum(misses_by_group),
            "product_compute_reduction_vs_terms": 1.0 - sum(misses_by_group) / total_terms,
            "storage_out32_bits": product_cache_bits(out_dim=32, ways=ways),
            "per_group": summarize_int(misses_by_group),
        }

    candidates = []
    for batch_size in (2, 4, 8, 16):
        all_batches = [
            batch
            for contexts in group_contexts
            for batch in make_batches(contexts, batch_size)
        ]
        capacities = [int(batch["terms"]) for batch in all_batches]
        slots = [int(batch["slots"]) for batch in all_batches]
        cap_p99 = max(1, percentile(capacities, 99))
        slots_p99 = max(1, percentile(slots, 99))
        overflow = [
            int(batch["terms"]) > cap_p99 or int(batch["slots"]) > slots_p99
            for batch in all_batches
        ]
        ideal_misses = sum(int(batch["reordered_w1_misses"]) for batch in all_batches)
        fallback_misses = sum(
            int(batch["original_w1_misses"] if over else batch["reordered_w1_misses"])
            for batch, over in zip(all_batches, overflow, strict=True)
        )
        record: dict[str, object] = {
            "batch_contexts": batch_size,
            "batches": len(all_batches),
            "capacity_terms": summarize_int(capacities),
            "gate_slots_per_lane": summarize_int(slots),
            "p99_design": {
                "capacity_terms": cap_p99,
                "gate_slots_per_lane": slots_p99,
                "overflow_batches": sum(overflow),
                "overflow_fraction": sum(overflow) / len(all_batches),
                "storage_out32_bits": ecgb_bits(
                    batch=batch_size,
                    capacity=cap_p99,
                    slots=slots_p99,
                    out_dim=32,
                ),
                "product_computes_with_fallback": fallback_misses,
                "reduction_vs_terms": 1.0 - fallback_misses / total_terms,
                "reduction_vs_w4": 1.0 - fallback_misses / int(baseline["4"]["product_computes"]),
            },
            "max_design_storage_out32_bits": ecgb_bits(
                batch=batch_size,
                capacity=max(capacities, default=1),
                slots=max(slots, default=1),
                out_dim=32,
            ),
            "ideal_product_computes": ideal_misses,
            "ideal_reduction_vs_terms": 1.0 - ideal_misses / total_terms,
        }
        cycle_sensitivity = {}
        for penalty in (1, 2, 4):
            proposed = total_terms + penalty * fallback_misses
            w4 = total_terms + penalty * int(baseline["4"]["product_computes"])
            cycle_sensitivity[str(penalty)] = {
                "miss_penalty": penalty,
                "proposed_cycles": proposed,
                "w4_cycles": w4,
                "speedup_vs_w4": w4 / proposed,
            }
        record["cycle_sensitivity"] = cycle_sensitivity
        candidates.append(record)

    stage_terms = {
        str(stage): sum(len(group_terms[index]) for index in range(groups) if stages[index] == stage)
        for stage in range(4)
    }
    return {
        "schema": "local5_ecgb_population_screen_v1",
        "status": "PASS_SCREEN_ONLY",
        "evidence": "[profile-qualified-trace]+[bounded-storage-work-model]",
        "source_vectors": str(vector_dir.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "groups": groups,
        "stage_groups": {str(stage): stages.count(stage) for stage in range(4)},
        "stage_terms": stage_terms,
        "active_source_contexts": sum(map(len, group_contexts)),
        "terms": total_terms,
        "baseline_wide_product_cache": baseline,
        "ecgb_candidates": candidates,
        "exactness_contract": [
            "Only contexts within one group share the same checkpoint weight slice.",
            "Terms are reordered but never deleted; destination masks remain attached.",
            "A context final is invisible until every original term has committed.",
            "Accumulator arithmetic must not overflow before the frozen Acc32 boundary.",
        ],
        "claim_boundary": [
            "This is not RTL, cycle measurement, PPA, energy, or full encoder evidence.",
            "The miss-penalty table is sensitivity only; cache lookup/control cost is absent.",
            "A candidate advances only if it beats the W4 wide-product cache under a comparable bit budget.",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Local5 ECGB 百样本强基线筛选",
        "",
        "> 证据：`[profile-qualified-trace]+[bounded-storage-work-model]`。不是 RTL、PPA、能量或 full encoder。",
        "",
        f"- groups: {report['groups']}，active source contexts: {report['active_source_contexts']}，terms: {report['terms']}",
        f"- stage groups: `{report['stage_groups']}`",
        "",
        "## 宽 product cache 强基线",
        "",
        "| ways | product compute | vs term reduction | OUT32 storage bit |",
        "|---:|---:|---:|---:|",
    ]
    for ways in ("1", "2", "4", "6"):
        row = report["baseline_wide_product_cache"][ways]
        lines.append(
            f"| {ways} | {row['product_computes']} | "
            f"{row['product_compute_reduction_vs_terms']:.2%} | {row['storage_out32_bits']} |"
        )
    lines += [
        "",
        "## 窄 term batch 候选",
        "",
        "固定容量取全体 batch 的 p99；超出者 fail-closed 回到原序 W1。",
        "",
        "| B | term cap p99/max | gate slot p99/max | overflow | OUT32 bit | product compute | vs W4 compute | miss=2 vs W4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ecgb_candidates"]:
        design = row["p99_design"]
        lines.append(
            f"| {row['batch_contexts']} | {design['capacity_terms']}/{row['capacity_terms']['max']} | "
            f"{design['gate_slots_per_lane']}/{row['gate_slots_per_lane']['max']} | "
            f"{design['overflow_fraction']:.2%} | {design['storage_out32_bits']} | "
            f"{design['product_computes_with_fallback']} | {design['reduction_vs_w4']:.2%} | "
            f"{row['cycle_sensitivity']['2']['speedup_vs_w4']:.4f}x |"
        )
    lines += ["", "## Exact 合同", ""]
    lines.extend(f"- {item}" for item in report["exactness_contract"])
    lines += ["", "## 边界", ""]
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args.vectors)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
