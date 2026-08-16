#!/usr/bin/env python3
"""Compare Local5 source-owned multicast with lane-local product caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
)
DEFAULT_COHORT = (
    ROOT
    / "results/local5_joint_ep29_tcfm5_linear5_realw_sample100_population_rtl_v5_final_20260813"
    / "source/bound/manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_unique_nonzero_gates(
    gates: Sequence[int], valid_mask: int
) -> list[int]:
    unique: list[int] = []
    for role, gate_value in enumerate(gates):
        gate = int(gate_value)
        if not ((valid_mask >> role) & 1) or gate == 0:
            continue
        if gate not in unique:
            unique.append(gate)
    return unique


def active_lanes(k_bitmap: int, lanes: int = 32) -> Iterable[int]:
    bitmap = int(k_bitmap)
    for lane in range(lanes):
        if (bitmap >> lane) & 1:
            yield lane


def source_owned_sequence(
    gates: np.ndarray,
    valid_masks: np.ndarray,
    k_bitmaps: np.ndarray,
) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for descriptor_gates, valid_mask, k_bitmap in zip(
        gates, valid_masks, k_bitmaps, strict=True
    ):
        unique = ordered_unique_nonzero_gates(
            descriptor_gates, int(valid_mask)
        )
        for lane in active_lanes(int(k_bitmap)):
            rows.extend((lane, gate) for gate in unique)
    return rows


def lru_product_starts(
    rows: Sequence[tuple[int, int]], ways: int, lanes: int = 32
) -> int:
    if ways <= 0:
        raise ValueError("ways must be positive")
    caches = [OrderedDict() for _ in range(lanes)]
    misses = 0
    for lane, gate in rows:
        if not 0 <= lane < lanes:
            raise ValueError(f"lane {lane} is outside [0,{lanes})")
        cache = caches[lane]
        if gate in cache:
            cache.move_to_end(gate)
            continue
        misses += 1
        if len(cache) == ways:
            cache.popitem(last=False)
        cache[gate] = None
    return misses


def cache_state_bits(
    *, lanes: int, ways: int, out_dim: int, gate_bits: int = 9, weight_bits: int = 8
) -> dict[str, int]:
    product_bits = out_dim * (gate_bits + weight_bits)
    age_bits = max(1, math.ceil(math.log2(ways)))
    entries = lanes * ways
    data_bits = entries * product_bits
    metadata_bits = entries * (gate_bits + 1 + age_bits)
    return {
        "product_bits_per_entry": product_bits,
        "data_bits": data_bits,
        "metadata_bits": metadata_bits,
        "output_register_bits": product_bits,
        "total_logic_bits": data_bits + metadata_bits + product_bits,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def analyze_group(
    *,
    group_index: int,
    item_offsets: np.ndarray,
    item_lanes: np.ndarray,
    item_gates: np.ndarray,
    item_multiplicity: np.ndarray,
    descriptor_offsets: np.ndarray,
    descriptor_gates: np.ndarray,
    descriptor_valid_masks: np.ndarray,
    descriptor_k_bitmaps: np.ndarray,
    ways: Sequence[int],
) -> dict[str, object]:
    item_lo = int(item_offsets[group_index])
    item_hi = int(item_offsets[group_index + 1])
    descriptor_lo = int(descriptor_offsets[group_index])
    descriptor_hi = int(descriptor_offsets[group_index + 1])

    destination_rows = [
        (int(lane), int(gate))
        for lane, gate in zip(
            item_lanes[item_lo:item_hi],
            item_gates[item_lo:item_hi],
            strict=True,
        )
    ]
    source_rows = source_owned_sequence(
        descriptor_gates[descriptor_lo:descriptor_hi],
        descriptor_valid_masks[descriptor_lo:descriptor_hi],
        descriptor_k_bitmaps[descriptor_lo:descriptor_hi],
    )
    raw_delivery = int(item_multiplicity[item_lo:item_hi].sum())
    destination_keys = set(destination_rows)
    source_keys = set(source_rows)
    if destination_keys != source_keys:
        raise ValueError(
            f"group {group_index}: destination/source product-key sets differ"
        )
    cache = {}
    for way_count in ways:
        destination_misses = lru_product_starts(destination_rows, way_count)
        source_misses = lru_product_starts(source_rows, way_count)
        cache[str(way_count)] = {
            "destination_product_starts": destination_misses,
            "source_product_starts": source_misses,
            "destination_hits": len(destination_rows) - destination_misses,
            "source_hits": len(source_rows) - source_misses,
        }
    return {
        "group_index": group_index,
        "descriptors": descriptor_hi - descriptor_lo,
        "raw_relation_lane_issues": raw_delivery,
        "destination_mfep_issues": len(destination_rows),
        "source_owned_issues": len(source_rows),
        "descriptor_local_issue_lower_bound": len(source_rows),
        "epoch_unique_product_keys": len(source_keys),
        "cache": cache,
    }


def build_report(
    source_dir: Path,
    cohort_manifest: Path,
    *,
    ways: Sequence[int] = (4, 6),
    out_dim: int = 2,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    cohort_manifest = cohort_manifest.resolve()
    source_manifest_path = source_dir / "ordered_term_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    payload_path = source_dir / source_manifest["payload_file"]
    if sha256(payload_path) != source_manifest["payload_sha256"]:
        raise ValueError("source payload SHA256 mismatch")

    cohort = json.loads(cohort_manifest.read_text(encoding="utf-8"))
    selection = cohort.get("selection", {})
    rows = selection.get("rows", [])
    represented_samples = len({int(row["sample"]) for row in rows})
    if (
        len(rows) != 100
        or represented_samples != 100
        or selection.get("method")
        != "sample-disjoint population-stage-weighted deterministic groups"
    ):
        raise ValueError("cohort is not the qualified population100 selection")
    group_indices = [int(row["input_group_index"]) for row in rows]
    if len(set(group_indices)) != len(group_indices):
        raise ValueError("cohort contains duplicate input_group_index")

    with np.load(payload_path, allow_pickle=False) as payload:
        required = {
            "group_offsets",
            "item_lane_id",
            "item_gate_code",
            "item_multiplicity",
            "descriptor_group_offsets",
            "descriptor_incoming_gates",
            "descriptor_valid_mask",
            "descriptor_k_bitmap",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"payload missing arrays: {missing}")
        group_reports = [
            analyze_group(
                group_index=group_index,
                item_offsets=payload["group_offsets"],
                item_lanes=payload["item_lane_id"],
                item_gates=payload["item_gate_code"],
                item_multiplicity=payload["item_multiplicity"],
                descriptor_offsets=payload["descriptor_group_offsets"],
                descriptor_gates=payload["descriptor_incoming_gates"],
                descriptor_valid_masks=payload["descriptor_valid_mask"],
                descriptor_k_bitmaps=payload["descriptor_k_bitmap"],
                ways=ways,
            )
            for group_index in group_indices
        ]

    totals: dict[str, object] = {
        field: sum(int(group[field]) for group in group_reports)
        for field in (
            "descriptors",
            "raw_relation_lane_issues",
            "destination_mfep_issues",
            "source_owned_issues",
            "descriptor_local_issue_lower_bound",
            "epoch_unique_product_keys",
        )
    }
    cache_totals = {}
    for way_count in ways:
        key = str(way_count)
        cache_totals[key] = {
            field: sum(int(group["cache"][key][field]) for group in group_reports)
            for field in (
                "destination_product_starts",
                "source_product_starts",
                "destination_hits",
                "source_hits",
            )
        }
        cache_totals[key]["destination_tag_compares"] = (
            int(totals["destination_mfep_issues"]) * way_count
        )
        cache_totals[key]["source_tag_compares"] = (
            int(totals["source_owned_issues"]) * way_count
        )
        cache_totals[key]["out2_state"] = cache_state_bits(
            lanes=32, ways=way_count, out_dim=out_dim
        )
    totals["cache"] = cache_totals
    totals["source_issue_reduction_vs_destination_mfep"] = 1.0 - _ratio(
        int(totals["source_owned_issues"]),
        int(totals["destination_mfep_issues"]),
    )
    totals["source_issue_reduction_vs_relation_lane"] = 1.0 - _ratio(
        int(totals["source_owned_issues"]),
        int(totals["raw_relation_lane_issues"]),
    )
    totals["descriptor_lower_bound_attained"] = (
        totals["source_owned_issues"]
        == totals["descriptor_local_issue_lower_bound"]
    )

    return {
        "schema": "local5_source_owned_cache_baselines_v1",
        "status": "PROFILE_STRONG_BASELINE_AUDIT",
        "evidence": "[prof]",
        "scope": (
            "100 sample-disjoint population-stage-weighted groups; OUT_DIM=2 "
            "logical cache model; cache reset at each weight/head group"
        ),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_payload": str(payload_path),
        "source_payload_sha256": sha256(payload_path),
        "cohort_manifest": str(cohort_manifest),
        "cohort_manifest_sha256": sha256(cohort_manifest),
        "contract": {
            "issue_shape": "one source, one active K lane, one nonzero gate, arbitrary five-destination mask",
            "local_minimum": (
                "for a fixed source/lane, exact one-gate issues require at least "
                "the number of distinct nonzero destination gates"
            ),
            "local_minimum_scope": (
                "does not bound cross-descriptor product caches, multi-source "
                "commands, or algebraic combinations of different gates"
            ),
            "w4_w6_scope": (
                "lane-local true-LRU product reuse; no cycle, SRAM energy, or PPA claim"
            ),
        },
        "totals": totals,
        "per_group": group_reports,
        "claim_boundary": {
            "term_issue_is_not_product_start": True,
            "product_start_is_not_cycle_or_energy": True,
            "logical_state_bits_are_not_physical_area": True,
            "not_encoder": True,
            "not_asic_ppa": True,
            "does_not_modify_docs359": True,
        },
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    totals = report["totals"]
    cache = totals["cache"]
    lines = [
        "# Local5 source-owned 与 product-cache 强基线",
        "",
        "## 裁决",
        "",
        "- 状态：`PROFILE_STRONG_BASELINE_AUDIT`。",
        "- source-owned 达到的是 descriptor-local、one-gate-per-issue 合同下的最少 issue 数，不是全局 product-compute 下界。",
        "- W4/W6 可以跨 descriptor 继续复用 `(lane,gate)` product；因此 generic cache 与 source-owned multicast 正交，不能省略任一方成本。",
        "",
        "## 同一 100-group 分账 `[prof]`",
        "",
        "| 执行流 | issue | 无 cache product start | W4 start | W6 start |",
        "|---|---:|---:|---:|---:|",
        f"| raw relation-lane | {totals['raw_relation_lane_issues']} | 不单列 | 与MFEP相同key首次 | 与MFEP相同key首次 |",
        f"| destination-local MFEP | {totals['destination_mfep_issues']} | {totals['destination_mfep_issues']} | {cache['4']['destination_product_starts']} | {cache['6']['destination_product_starts']} |",
        f"| source-owned multicast | {totals['source_owned_issues']} | {totals['source_owned_issues']} | {cache['4']['source_product_starts']} | {cache['6']['source_product_starts']} |",
        "",
        f"source-owned 相对 MFEP 的 issue 减少为 `{100*totals['source_issue_reduction_vs_destination_mfep']:.3f}%`；",
        "这项收益来自一个 term 携带五位 destination mask，不等同于缓存命中。",
        "",
        "## 成本边界",
        "",
        f"- W4 OUT2 logical state：`{cache['4']['out2_state']['total_logic_bits']}` bit；tag compare：`{cache['4']['source_tag_compares']}`（source流）或 `{cache['4']['destination_tag_compares']}`（MFEP流）。",
        f"- W6 OUT2 logical state：`{cache['6']['out2_state']['total_logic_bits']}` bit；tag compare：`{cache['6']['source_tag_compares']}`（source流）或 `{cache['6']['destination_tag_compares']}`（MFEP流）。",
        "- source builder 的五 gate 比较和 mask 形成是 descriptor-local 组合逻辑；这里只证明功能/工作量边界，没有 SAIF、目标 SRAM 或 DC 面积。",
        "",
        "## 可写与不可写",
        "",
        "可写：固定五邻域下，生产 builder 对每个 source/lane 的相同 gate consumer 建立精确等价类，以一个 mask term 达到该 issue 合同的局部最小值。",
        "",
        "不可写：全局最少乘法、首次 source-stationary、打赢 Prosperity PPA、term 减少等于周期或能量。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.source_dir, args.cohort_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "report.md", report)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
