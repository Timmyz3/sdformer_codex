#!/usr/bin/env python3
"""比较GateStack descriptor编码，并验证隐式前缀格式可逆。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
DEFAULT_JSON = ROOT / "results/gatestack_descriptor_format_dse_20260715.json"
DEFAULT_MD = ROOT / "results/gatestack_descriptor_format_dse_20260715.md"


@dataclass(frozen=True)
class Format:
    name: str
    header_bits: int
    descriptor_bits: int
    byte_aligned: bool
    implicit_prefix: bool
    descriptor_word_padded: bool = False


FORMATS = (
    Format("packed35_h192", 192, 35, False, False),
    Format("byte40_h192", 192, 40, True, False),
    Format("ipd24_h128", 128, 24, True, True),
    Format("ipd32w_h128", 128, 32, True, True, True),
    Format("ipd22_h128", 128, 22, False, True),
)


def classify(
    active_events: int,
    class_terms: int,
    active_classes: int,
    fmt: Format,
    *,
    raw_bits: int = 6642,
    class_slots: int = 4,
) -> tuple[str, int]:
    descriptor_payload = fmt.descriptor_bits * class_terms
    if fmt.descriptor_word_padded:
        descriptor_payload = ((descriptor_payload + 63) // 64) * 64
    payload = fmt.header_bits + descriptor_payload + 8 * active_events
    if active_classes > class_slots:
        return "RAW_CLASS", raw_bits
    if payload > raw_bits:
        return "RAW_CAPACITY", raw_bits
    return "CSR", payload


def collect_rows(profile: dict[str, Any]) -> dict[int, list[tuple[int, int, int]]]:
    rows: dict[int, list[tuple[int, int, int]]] = {stage: [] for stage in range(4)}
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
        if not (len(active) == len(terms) == len(classes)):
            raise ValueError("ordered trace长度不一致")
        rows[int(record["stage"])].extend(zip(active, terms, classes))
    return rows


def summarize(rows: list[tuple[int, int, int]], fmt: Format) -> dict[str, Any]:
    modes: dict[str, int] = {"CSR": 0, "RAW_CLASS": 0, "RAW_CAPACITY": 0}
    stored: list[int] = []
    csr_payload: list[int] = []
    for active, terms, classes in rows:
        mode, bits = classify(active, terms, classes, fmt)
        modes[mode] += 1
        stored.append(bits)
        if mode == "CSR":
            csr_payload.append(bits)
    count = len(rows)
    return {
        "rows": count,
        "mode_counts": modes,
        "mode_ratios": {key: value / count for key, value in modes.items()},
        "stored_bits_mean": sum(stored) / count,
        "stored_bits_p99": percentile(stored, 0.99),
        "csr_payload_mean": sum(csr_payload) / len(csr_payload),
        "saving_vs_raw": 1.0 - sum(stored) / (count * 6642),
    }


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    rows_by_stage = collect_rows(profile)
    all_rows = [row for stage in range(4) for row in rows_by_stage[stage]]
    return {
        fmt.name: {
            "contract": {
                "header_bits": fmt.header_bits,
                "descriptor_bits": fmt.descriptor_bits,
                "byte_aligned": fmt.byte_aligned,
                "implicit_prefix": fmt.implicit_prefix,
                "descriptor_word_padded": fmt.descriptor_word_padded,
            },
            "all": summarize(all_rows, fmt),
            "stages": {
                str(stage): summarize(rows, fmt)
                for stage, rows in rows_by_stage.items()
            },
        }
        for fmt in FORMATS
    }


def render_md(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    lines = [
        "# GateStack Descriptor 格式 DSE（2026-07-15）",
        "",
        f"输入：`{result['profile']}`。证据等级为 `[prof]+[格式模型]`。",
        "",
        "## 1. 候选",
        "",
        "| 格式 | header | descriptor | 对齐 | event_base |",
        "|---|---:|---:|---|---|",
        "| packed35_h192 | 192 bit | 35 bit | bit-packed | 显式13 bit |",
        "| byte40_h192 | 192 bit | 40 bit | byte | 显式13 bit |",
        "| **ipd24_h128** | **128 bit** | **24 bit** | **byte** | **隐式前缀** |",
        "| ipd32w_h128 | 128 bit | 32 bit，2项/word | 64-bit word | 隐式前缀 |",
        "| ipd22_h128 | 128 bit | 22 bit | bit-packed | 隐式前缀 |",
        "",
        "`ipd24` 的 descriptor 为 `{reserved2,event_count8,lane_id5,gate_code9}`。",
        "token ID 列表严格按 descriptor 顺序连续保存，replay 只维护滚动 event 指针，",
        "所以无需每个 term 保存 13-bit `event_base`。",
        "",
        "## 2. Profile100 总体结果",
        "",
        "| 格式 | CSR比例 | RAW-class | RAW-capacity | 平均有效位 | p99 | 相对RAW平均减少 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "packed35_h192",
        "byte40_h192",
        "ipd24_h128",
        "ipd32w_h128",
        "ipd22_h128",
    ):
        row = analysis[name]["all"]
        lines.append(
            f"| {name} | {row['mode_ratios']['CSR']:.4%} | "
            f"{row['mode_ratios']['RAW_CLASS']:.4%} | "
            f"{row['mode_ratios']['RAW_CAPACITY']:.4%} | "
            f"{row['stored_bits_mean']:.1f} | {row['stored_bits_p99']:.0f} | "
            f"{row['saving_vs_raw']:.4%} |"
        )
    lines += [
        "",
        "## 3. ipd24 分 Stage",
        "",
        "| Stage | CSR比例 | RAW-capacity | 平均有效位 |",
        "|---|---:|---:|---:|",
    ]
    for stage, row in analysis["ipd24_h128"]["stages"].items():
        lines.append(
            f"| {stage} | {row['mode_ratios']['CSR']:.4%} | "
            f"{row['mode_ratios']['RAW_CAPACITY']:.4%} | "
            f"{row['stored_bits_mean']:.1f} |"
        )
    lines += [
        "",
        "## 4. 决策规则",
        "",
        "- `ipd24` 必须先通过 byte-exact serialize/parse reference，才能替换旧35-bit格式；",
        "- `ipd32w` 每个64-bit word固定容纳两个descriptor，奇数term补32 bit，前端可稳定提供至少1 term/cycle；",
        "- `ipd22` 只作为理论容量下界，因为跨64-bit边界提取增加 barrel/拼接控制；",
        "- RAW仍使用原始162×41-bit bitstream，物理slot不缩小；",
        "- 格式收益只影响容量回退率和元数据流量，不能直接写成ASIC面积或功耗收益。",
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
        "evidence": "[prof ordered trace]+[格式模型]",
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
