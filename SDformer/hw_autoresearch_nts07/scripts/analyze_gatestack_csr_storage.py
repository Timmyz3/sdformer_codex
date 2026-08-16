#!/usr/bin/env python3
"""评估GateStack容量安全TERM-CSR/RAW双格式head slot。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace, percentile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_PROFILE = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_JSON = ROOT / "results/gatestack_csr_storage_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_csr_storage_20260715.md"


def classify_head_slot(
    *,
    active_lanes: int,
    class_terms: int,
    active_classes: int,
    tokens: int = 162,
    head_dim: int = 32,
    gate_bits: int = 9,
    class_slots: int = 4,
    header_bits: int = 128,
    descriptor_bits: int = 32,
    descriptor_word_padded: bool = True,
) -> dict[str, int | str]:
    if min(active_lanes, class_terms, active_classes) < 0:
        raise ValueError("计数不能为负")
    raw_bits = tokens * (head_dim + gate_bits)
    token_id_bits = max(1, math.ceil(math.log2(tokens)))
    descriptor_payload_bits = class_terms * descriptor_bits
    if descriptor_word_padded:
        descriptor_payload_bits = (
            (descriptor_payload_bits + 63) // 64
        ) * 64
    csr_bits = header_bits + descriptor_payload_bits + active_lanes * token_id_bits
    if active_classes > class_slots:
        mode = "RAW_CLASS_OVERFLOW"
    elif csr_bits > raw_bits:
        mode = "RAW_CAPACITY_OVERFLOW"
    else:
        mode = "TERM_CSR"
    stored_bits = csr_bits if mode == "TERM_CSR" else raw_bits
    return {
        "mode": mode,
        "raw_bits": raw_bits,
        "csr_bits": csr_bits,
        "stored_bits": stored_bits,
        "descriptor_bits": descriptor_bits,
        "token_id_bits": token_id_bits,
    }


def summarize_rows(rows: list[tuple[int, int, int]]) -> dict[str, Any]:
    classified = [
        classify_head_slot(
            active_lanes=active,
            class_terms=terms,
            active_classes=classes,
        )
        for active, terms, classes in rows
    ]
    stored = [int(row["stored_bits"]) for row in classified]
    csr = [int(row["csr_bits"]) for row in classified]
    raw_bits = int(classified[0]["raw_bits"]) if classified else 0
    bitmap_bits = 4 * 32 * 162 + 128 + 36
    modes = {
        mode: sum(row["mode"] == mode for row in classified)
        for mode in ("TERM_CSR", "RAW_CLASS_OVERFLOW", "RAW_CAPACITY_OVERFLOW")
    }
    return {
        "rows": len(rows),
        "modes": modes,
        "mode_ratios": {
            key: value / len(rows) if rows else 0.0 for key, value in modes.items()
        },
        "raw_slot_bits": raw_bits,
        "bitmap_slot_bits": bitmap_bits,
        "descriptor_bits": int(classified[0]["descriptor_bits"]) if classified else 0,
        "stored_bits_mean": sum(stored) / len(stored) if stored else 0.0,
        "stored_bits_p50": percentile(stored, 0.50),
        "stored_bits_p95": percentile(stored, 0.95),
        "stored_bits_p99": percentile(stored, 0.99),
        "stored_bits_max": max(stored, default=0),
        "csr_bits_max_before_fallback": max(csr, default=0),
        "effective_payload_saving_vs_raw": (
            1.0 - sum(stored) / (len(stored) * raw_bits)
            if stored and raw_bits
            else 0.0
        ),
        "effective_payload_saving_vs_bitmap": (
            1.0 - sum(stored) / (len(stored) * bitmap_bits)
            if stored
            else 0.0
        ),
        "active_lanes_mean": sum(row[0] for row in rows) / len(rows) if rows else 0.0,
        "active_lanes_p99": percentile([row[0] for row in rows], 0.99),
        "class_terms_mean": sum(row[1] for row in rows) / len(rows) if rows else 0.0,
        "class_terms_p99": percentile([row[1] for row in rows], 0.99),
    }


def physical_storage_by_stage() -> dict[int, dict[str, float]]:
    raw_slot_bits = 162 * (32 + 9)
    accumulator_bits = 162 * 32 * 32
    scratch_bits = 2 * raw_slot_bits
    result = {}
    for stage, heads in ((0, 3), (1, 6), (2, 12), (3, 24)):
        dual_head_slots = 2 * heads * raw_slot_bits
        metadata = 2 * heads * (128 + 64)
        total = dual_head_slots + accumulator_bits + scratch_bits + metadata
        bitmap_total = (
            2 * heads * (4 * 32 * 162 + 128 + 36)
            + accumulator_bits
            + raw_slot_bits
        )
        result[stage] = {
            "heads": heads,
            "dual_head_slots_kib": dual_head_slots / 8 / 1024,
            "accumulator_kib": accumulator_bits / 8 / 1024,
            "two_scratch_kib": scratch_bits / 8 / 1024,
            "metadata_kib": metadata / 8 / 1024,
            "total_kib": total / 8 / 1024,
            "bitmap_design_kib": bitmap_total / 8 / 1024,
            "physical_reduction_vs_bitmap": 1.0 - total / bitmap_total,
        }
    return result


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    rows_by_stage: dict[int, list[tuple[int, int, int]]] = {0: [], 1: [], 2: [], 3: []}
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
        rows_by_stage[int(record["stage"])].extend(zip(active, terms, classes))
    all_rows = [row for stage in sorted(rows_by_stage) for row in rows_by_stage[stage]]
    return {
        "all": summarize_rows(all_rows),
        "stages": {stage: summarize_rows(rows) for stage, rows in rows_by_stage.items()},
        "physical_storage": physical_storage_by_stage(),
        "missing_profile_for_compaction": [
            "每head活动token数ordered trace",
            "每token活动K lane数ordered trace",
            "每head最大单lane事件数",
            "R路event compactor bank冲突与stall",
        ],
    }


def render_md(result: dict[str, Any]) -> str:
    overall = result["analysis"]["all"]
    lines = [
        "# GateStack TERM-CSR/RAW双格式存储分析",
        "",
        f"输入：`{result['profile']}`，证据为 `[prof]+[存储模型]`。",
        "",
        "## 1. 表示合同",
        "",
        "每个head物理slot固定为原始token流容量：",
        "",
        "```text",
        "RAW = 162 × (K32 + gate9) = 6642 bit",
        "TERM-CSR = header128 + ceil(term_count/2)×64-bit IPD32W + active_lane_count×token_id8",
        "```",
        "",
        "若gate class超过S=4，或TERM-CSR超过6642 bit，则该head无损保留RAW；否则提交TERM-CSR。物理slot永不溢出，也不需要固定20.9 Kbit bitmap。",
        "",
        "## 2. 全量profile100",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| head rows | {overall['rows']} |",
        f"| TERM-CSR | {overall['mode_ratios']['TERM_CSR']:.4%} |",
        f"| RAW class overflow | {overall['mode_ratios']['RAW_CLASS_OVERFLOW']:.4%} |",
        f"| RAW capacity overflow | {overall['mode_ratios']['RAW_CAPACITY_OVERFLOW']:.4%} |",
        f"| 平均有效payload | {overall['stored_bits_mean']:.1f} bit |",
        f"| p99有效payload | {overall['stored_bits_p99']:.1f} bit |",
        f"| 相对RAW平均有效位减少 | {overall['effective_payload_saving_vs_raw']:.4%} |",
        f"| 相对bitmap平均有效位减少 | {overall['effective_payload_saving_vs_bitmap']:.4%} |",
        "",
        "## 3. 分stage",
        "",
        "| Stage | TERM-CSR | RAW总比例 | active mean/p99 | term mean/p99 | 平均payload |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, row in result["analysis"]["stages"].items():
        raw_ratio = 1.0 - row["mode_ratios"]["TERM_CSR"]
        lines.append(
            f"| {stage} | {row['mode_ratios']['TERM_CSR']:.3%} | {raw_ratio:.3%} | "
            f"{row['active_lanes_mean']:.2f}/{row['active_lanes_p99']:.0f} | "
            f"{row['class_terms_mean']:.2f}/{row['class_terms_p99']:.0f} | "
            f"{row['stored_bits_mean']:.1f} bit |"
        )
    lines += [
        "",
        "## 4. 双context物理容量",
        "",
        "固定RAW-sized slot保证任意head可回退，平均payload减少不等于进一步缩小物理slot。",
        "",
        "| Stage | 双context head slots | AccTile | 双scratch | metadata | 合计 | 相对bitmap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in result["analysis"]["physical_storage"].items():
        lines.append(
            f"| {stage} | {row['dual_head_slots_kib']:.2f} KiB | "
            f"{row['accumulator_kib']:.2f} KiB | {row['two_scratch_kib']:.2f} KiB | "
            f"{row['metadata_kib']:.2f} KiB | **{row['total_kib']:.2f} KiB** | "
            f"-{row['physical_reduction_vs_bitmap']:.2%} |"
        )
    lines += [
        "",
        "## 5. 端口意义",
        "",
        "- RAW scratch按token写41 bit，天然匹配SCS输出；",
        "- TERM-CSR按term descriptor和连续token-id list读取，天然匹配product复用和multicast；",
        "- 两遍compaction把token-major写与term-major读解耦，避免bitmap SRAM的32路bit-write/162-bit-read转置冲突；",
        "- OBI在compaction阶段枚举128-bit有效term mask，packed descriptor在每个output tile直接顺序重放；",
        "- RAW head复用同一direct/product/multicast/accumulator后端，不复制第二套核。",
        "",
        "## 6. 尚缺profile",
        "",
    ]
    lines.extend(
        f"- {item}；" for item in result["analysis"]["missing_profile_for_compaction"]
    )
    lines += [
        "",
        "因此本表能冻结存储容量和表示选择，不能冻结compactor并行度R或build/replay重叠周期。下一次profile必须补活动token与逐token K-count trace。",
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
        "evidence": "[prof ordered trace]+[存储模型]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
