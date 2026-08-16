#!/usr/bin/env python3
"""分析GateStack容量优先RAW/IPD/FADC片上格式策略。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from gatestack_fadc24_reference import _record_rows, serialize_fadc24


SLOT_BYTES = 832
RAW_PAYLOAD_BITS = 6642
IPD_CLASS_SLOTS = 4


def decide_format(
    *, active_classes: int, term_count: int, event_count: int,
    fadc_destination_bytes: int, metadata_overflow: bool = False,
) -> dict[str, Any]:
    ipd_bytes = 16 + 8 * math.ceil(term_count / 2) + event_count
    fadc_bytes = 16 + 3 * term_count + fadc_destination_bytes
    if metadata_overflow:
        selected, reason, payload_bits = "RAW41", "metadata_overflow", RAW_PAYLOAD_BITS
    elif active_classes <= IPD_CLASS_SLOTS and ipd_bytes <= SLOT_BYTES:
        selected, reason, payload_bits = "IPD32W", "ipd_fit", ipd_bytes * 8
    elif fadc_bytes <= SLOT_BYTES:
        reason = "ipd_capacity" if active_classes <= IPD_CLASS_SLOTS else "ipd_class"
        selected, payload_bits = "FADC24", fadc_bytes * 8
    else:
        selected, reason, payload_bits = "RAW41", "fadc_capacity", RAW_PAYLOAD_BITS
    return {
        "format": selected,
        "reason": reason,
        "payload_bits": payload_bits,
        "word_count": math.ceil(payload_bits / 64),
        "ipd_bytes": ipd_bytes,
        "fadc_bytes": fadc_bytes,
    }


def analyze_exact_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in manifest["records"]:
        source_record = {"name": record["name"], "file": record["source"]}
        rows = []
        for head, terms in enumerate(_record_rows(source_record, int(record["window"]))):
            event_count = sum(len(term["tokens"]) for term in terms)
            active_classes = len({int(term["gate"]) for term in terms})
            fadc_destination_bytes = sum(min(len(term["tokens"]), 21) for term in terms)
            decision = decide_format(
                active_classes=active_classes,
                term_count=len(terms),
                event_count=event_count,
                fadc_destination_bytes=fadc_destination_bytes,
            )
            encoded = serialize_fadc24(terms, tag=(int(record["stage"]) << 16) | head)
            if len(encoded) != decision["fadc_bytes"]:
                raise AssertionError("FADC容量公式与金参考不一致")
            rows.append(
                {
                    "head": head,
                    "active_classes": active_classes,
                    "terms": len(terms),
                    "events": event_count,
                    "bitmap_terms": sum(len(term["tokens"]) > 21 for term in terms),
                    **decision,
                }
            )
        counts = {
            name: sum(row["format"] == name for row in rows)
            for name in ("IPD32W", "FADC24", "RAW41")
        }
        records.append(
            {
                "name": record["name"],
                "stage": int(record["stage"]),
                "heads": len(rows),
                "format_counts": counts,
                "selected_words": sum(row["word_count"] for row in rows),
                "rows": rows,
            }
        )
    return records


def analyze_profile100_bounds(profile: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in profile["records"]:
        heads = int(row["head_instances"])
        ipd_fallback = int(row["ipd32w_raw_fallbacks"])
        ipd = heads - ipd_fallback
        guaranteed = int(row["fadc24_guaranteed_fit"])
        ambiguous = int(row["fadc24_ambiguous"])
        impossible = int(row["fadc24_impossible_fit"])
        fadc_lower = max(0, ipd_fallback - ambiguous - impossible)
        fadc_upper = min(ipd_fallback, guaranteed + ambiguous)
        raw_lower = max(0, ipd_fallback - guaranteed - ambiguous)
        raw_upper = min(ipd_fallback, ambiguous + impossible)
        output.append(
            {
                "stage": int(row["stage"]),
                "heads": heads,
                "ipd_exact": ipd,
                "fadc_lower": fadc_lower,
                "fadc_upper": fadc_upper,
                "raw_lower": raw_lower,
                "raw_upper": raw_upper,
                "ambiguous_due_to_missing_term_fanout": ambiguous,
            }
        )
    return output


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack片上三格式策略真实负载分析",
        "",
        "## 策略合同",
        "",
        "采用容量优先且不依赖stage ID的精确策略：IPD32W在class与容量均满足时优先；否则尝试FADC24；两者均失败或元数据计数溢出时回退RAW41。该策略不按最小字节盲选FADC，因为现有真实RTL已显示FADC在S0/S2即使更短也可能更慢。",
        "",
        "## 单样本四Stage位级Trace",
        "",
        "| Stage | Head | IPD32W | FADC24 | RAW41 | 选中payload word |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["exact_records"]:
        counts = record["format_counts"]
        lines.append(
            f"| S{record['stage']} | {record['heads']} | {counts['IPD32W']} | "
            f"{counts['FADC24']} | {counts['RAW41']} | {record['selected_words']} |"
        )
    lines.extend(
        [
            "",
            "### 非IPD Head明细",
            "",
            "| Stage | Head | class | term | event | bitmap term | IPD byte | FADC byte | 决策 | 原因 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    non_ipd = 0
    for record in result["exact_records"]:
        for row in record["rows"]:
            if row["format"] == "IPD32W":
                continue
            non_ipd += 1
            lines.append(
                f"| S{record['stage']} | {row['head']} | {row['active_classes']} | "
                f"{row['terms']} | {row['events']} | {row['bitmap_terms']} | "
                f"{row['ipd_bytes']} | {row['fadc_bytes']} | {row['format']} | "
                f"{row['reason']} |"
            )
    if non_ipd == 0:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Profile100容量边界",
            "",
            "ordered profile100缺少逐term fanout，因此FADC/RAW只能给严格上下界，不能伪装成精确格式分布。",
            "",
            "| Stage | Head实例 | IPD精确数 | FADC下界 | FADC上界 | RAW下界 | RAW上界 | fanout歧义项 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["profile100_bounds"]:
        lines.append(
            f"| S{row['stage']} | {row['heads']} | {row['ipd_exact']} | "
            f"{row['fadc_lower']} | {row['fadc_upper']} | "
            f"{row['raw_lower']} | {row['raw_upper']} | "
            f"{row['ambiguous_due_to_missing_term_fanout']} |"
        )
    lines.extend(
        [
            "",
            "## 架构指导",
            "",
            "1. 真实首window中仅S3的一个高扇出head需要FADC，其他head继续走IPD，证明运行时策略可以替代stage硬编码。",
            "2. Policy只需要term数、event数、active class数与`sum(min(fanout,21))`，均可在OBI/term枚举期间顺序累加，不需要读取完整destination payload做组合决策。",
            "3. 完整片上builder宜采用RAW scratch上的metadata-first两遍构建：第一遍统计并选格式，第二遍只序列化被选格式，避免并行维护IPD/FADC两套832-byte候选buffer。",
            "4. 当前结果完成的是策略与元数据前端，不等于payload serializer、目标库PPA或全encoder吞吐闭环。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--profile100", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "policy": "capacity_first_ipd_then_fadc_then_raw",
        "exact_records": analyze_exact_manifest(
            json.loads(args.trace_manifest.read_text(encoding="utf-8"))
        ),
        "profile100_bounds": analyze_profile100_bounds(
            json.loads(args.profile100.read_text(encoding="utf-8"))
        ),
        "evidence_boundary": {
            "exact_trace": "一个样本、四stage首block首window位级trace",
            "profile100": "ordered统计，无逐term fanout，仅给容量上下界",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
