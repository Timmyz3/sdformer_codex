#!/usr/bin/env python3
"""评估GateStack双context有界descriptor residency的命中率与容量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_gatestack_csr_storage import (
    classify_head_slot,
    physical_storage_by_stage,
)
from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/gatestack_descriptor_residency_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_descriptor_residency_20260715.md"
DEPTHS = (32, 48, 64, 80, 96, 128)
HEADS_BY_STAGE = {0: 3, 1: 6, 2: 12, 3: 24}


def collect_rows(profile: dict[str, Any]) -> dict[int, list[tuple[int, int, int]]]:
    rows = {stage: [] for stage in range(4)}
    for record in profile["summary"]["h60_records"]:
        active = decode_count_trace(
            record["projection_baseline_active_lanes_ordered_trace"]
        )
        terms = decode_count_trace(
            record["projection_gate_class_channel_terms_deploy_ordered_trace"]
        )
        classes = decode_count_trace(
            record["projection_active_gate_classes_deploy_ordered_trace"]
        )
        rows[int(record["stage"])].extend(zip(active, terms, classes))
    return rows


def summarize_depth(
    rows_by_stage: dict[int, list[tuple[int, int, int]]], depth: int
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    total_rows = 0
    csr_rows = 0
    cached_rows = 0
    weighted_frontend_all = 0
    weighted_frontend_cached = 0
    storage = physical_storage_by_stage()
    for stage, rows in rows_by_stage.items():
        stage_csr = 0
        stage_cached = 0
        stage_front_all = 0
        stage_front_cached = 0
        output_tiles = HEADS_BY_STAGE[stage]
        for active, terms, classes in rows:
            mode = classify_head_slot(
                active_lanes=active,
                class_terms=terms,
                active_classes=classes,
            )["mode"]
            if mode != "TERM_CSR":
                continue
            stage_csr += 1
            frontend = 2 + (terms + 1) // 2
            stage_front_all += frontend * output_tiles
            if terms <= depth:
                stage_cached += 1
            else:
                stage_front_cached += frontend * output_tiles
        cache_bits = 2 * HEADS_BY_STAGE[stage] * (depth * 24 + 8)
        cache_kib = cache_bits / 8 / 1024
        stages[str(stage)] = {
            "rows": len(rows),
            "csr_rows": stage_csr,
            "cached_rows": stage_cached,
            "cached_ratio_within_csr": stage_cached / stage_csr,
            "frontend_cycles_without_cache": stage_front_all,
            "frontend_cycles_with_cache": stage_front_cached,
            "frontend_cycle_reduction": 1.0 - stage_front_cached / stage_front_all,
            "dual_context_cache_kib": cache_kib,
            "nonweight_total_kib": storage[stage]["total_kib"] + cache_kib,
        }
        total_rows += len(rows)
        csr_rows += stage_csr
        cached_rows += stage_cached
        weighted_frontend_all += stage_front_all
        weighted_frontend_cached += stage_front_cached
    return {
        "depth": depth,
        "rows": total_rows,
        "csr_rows": csr_rows,
        "cached_rows": cached_rows,
        "cached_ratio_within_csr": cached_rows / csr_rows,
        "cached_ratio_all_heads": cached_rows / total_rows,
        "weighted_frontend_cycle_reduction":
            1.0 - weighted_frontend_cached / weighted_frontend_all,
        "stages": stages,
    }


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    rows = collect_rows(profile)
    return {str(depth): summarize_depth(rows, depth) for depth in DEPTHS}


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack 有界 Descriptor Residency DSE（2026-07-15）",
        "",
        f"输入：`{result['profile']}`。证据为 `[prof]+[存储/周期计数模型]`。",
        "",
        "每个cache entry内部使用24 bit `{reserved2,count8,lane5,gate9}`；",
        "head slot仍使用IPD32W。cache在compaction阶段旁路写入，命中head在所有output tile上不再读取header/descriptor；",
        "超过深度的CSR head保持顺序前端，RAW保持RAW41，均不改变数值。",
        "",
        "| 深度/Head | CSR内命中 | 全head命中 | 加权前端周期减少 | Stage3双context cache | Stage3非weight合计 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        row = result["analysis"][str(depth)]
        stage3 = row["stages"]["3"]
        lines.append(
            f"| {depth} | {row['cached_ratio_within_csr']:.4%} | "
            f"{row['cached_ratio_all_heads']:.4%} | "
            f"{row['weighted_frontend_cycle_reduction']:.4%} | "
            f"{stage3['dual_context_cache_kib']:.2f} KiB | "
            f"{stage3['nonweight_total_kib']:.2f} KiB |"
        )
    chosen = result["analysis"]["80"]
    lines += [
        "",
        "## Depth=80 分Stage",
        "",
        "| Stage | CSR内命中 | 前端周期减少 | cache | 非weight合计 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, row in chosen["stages"].items():
        lines.append(
            f"| {stage} | {row['cached_ratio_within_csr']:.4%} | "
            f"{row['frontend_cycle_reduction']:.4%} | "
            f"{row['dual_context_cache_kib']:.2f} KiB | "
            f"{row['nonweight_total_kib']:.2f} KiB |"
        )
    lines += [
        "",
        "上述容量是逻辑bit，不含macro rounding、ECC、BIST和布线通道。",
        "若Depth=80在目标库macro padding后超过80 KiB门槛，应降到64，而不是缩小RAW exact slot。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "analysis": analyze(profile),
        "evidence": "[prof ordered trace]+[存储/周期计数模型]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
