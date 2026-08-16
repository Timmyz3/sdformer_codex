#!/usr/bin/env python3
"""用H67 profile100 ordered trace审计FADC24容量上下界。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


HEADER_BYTES = 16
DESCRIPTOR_BYTES = 3
BITMAP_BYTES = 21
RAW_HEAD_BITS = 162 * 41
SLOT_BYTES = math.ceil(RAW_HEAD_BITS / 64) * 8


def destination_byte_bounds(terms: int, events: int, max_fanout: int) -> tuple[int, int]:
    """返回sum(min(fanout, 21))的可实现下界和上界。"""

    if terms == 0:
        if events != 0 or max_fanout != 0:
            raise ValueError("空term统计不一致")
        return 0, 0
    if not (1 <= max_fanout <= 162):
        raise ValueError("max_fanout越界")
    if not (max_fanout + terms - 1 <= events <= terms * max_fanout):
        raise ValueError("term/event/max_fanout统计不一致")

    counts = [1] * terms
    counts[0] = max_fanout
    remaining = events - sum(counts)
    for index in range(1, terms):
        add = min(remaining, max_fanout - 1)
        counts[index] += add
        remaining -= add
    if remaining:
        raise AssertionError("fanout下界构造失败")
    lower = sum(min(count, BITMAP_BYTES) for count in counts)
    upper = min(events, BITMAP_BYTES * terms)
    return lower, upper


def classify(terms: int, events: int, max_fanout: int) -> dict[str, Any]:
    lower_dest, upper_dest = destination_byte_bounds(terms, events, max_fanout)
    fixed = HEADER_BYTES + DESCRIPTOR_BYTES * terms
    lower = fixed + lower_dest
    upper = fixed + upper_dest
    ipd32w_bytes = HEADER_BYTES + 8 * math.ceil(terms / 2) + events
    return {
        "fadc24_lower_bytes": lower,
        "fadc24_upper_bytes": upper,
        "ipd32w_bytes": ipd32w_bytes,
        "ipd32w_fits_raw_bits": ipd32w_bytes * 8 <= RAW_HEAD_BITS,
        "fadc24_guaranteed_fit": upper <= SLOT_BYTES,
        "fadc24_impossible_fit": lower > SLOT_BYTES,
        "fadc24_ambiguous": lower <= SLOT_BYTES < upper,
    }


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    stage_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in profile["summary"]["h60_records"]:
        stage = int(record["stage"])
        terms = decode_count_trace(
            record["projection_gate_class_channel_terms_deploy_ordered_trace"]
        )
        events = decode_count_trace(record["projection_baseline_active_lanes_ordered_trace"])
        max_fanout = decode_count_trace(
            record["projection_gate_class_channel_max_fanout_deploy_ordered_trace"]
        )
        if not (len(terms) == len(events) == len(max_fanout)):
            raise ValueError(f"{record['name']} ordered trace长度不一致")
        for term_count, event_count, maximum in zip(terms, events, max_fanout):
            row = {
                "terms": int(term_count),
                "events": int(event_count),
                "max_fanout": int(maximum),
            }
            row.update(classify(**row))
            stage_rows[stage].append(row)

    records = []
    for stage, rows in sorted(stage_rows.items()):
        total = len(rows)
        current_fallbacks = sum(not row["ipd32w_fits_raw_bits"] for row in rows)
        guaranteed = sum(row["fadc24_guaranteed_fit"] for row in rows)
        impossible = sum(row["fadc24_impossible_fit"] for row in rows)
        ambiguous = sum(row["fadc24_ambiguous"] for row in rows)
        current_work = sum(
            row["terms"] if row["ipd32w_fits_raw_bits"] else row["events"]
            for row in rows
        )
        fadc_best_work = sum(
            row["events"] if row["fadc24_impossible_fit"] else row["terms"]
            for row in rows
        )
        fadc_worst_work = sum(
            row["terms"] if row["fadc24_guaranteed_fit"] else row["events"]
            for row in rows
        )
        if guaranteed + impossible + ambiguous != total:
            raise AssertionError("FADC24分类未穷尽")
        records.append(
            {
                "stage": stage,
                "head_instances": total,
                "ipd32w_raw_fallbacks": current_fallbacks,
                "ipd32w_raw_fallback_rate": current_fallbacks / total if total else 0.0,
                "fadc24_guaranteed_fit": guaranteed,
                "fadc24_guaranteed_fit_rate": guaranteed / total if total else 0.0,
                "fadc24_ambiguous": ambiguous,
                "fadc24_ambiguous_rate": ambiguous / total if total else 0.0,
                "fadc24_impossible_fit": impossible,
                "fadc24_impossible_fit_rate": impossible / total if total else 0.0,
                "current_executed_terms": current_work,
                "fadc24_best_case_executed_terms": fadc_best_work,
                "fadc24_worst_case_executed_terms": fadc_worst_work,
                "fadc24_best_case_term_reduction": (
                    0.0 if current_work == 0 else 1.0 - fadc_best_work / current_work
                ),
                "fadc24_worst_case_term_reduction": (
                    0.0 if current_work == 0 else 1.0 - fadc_worst_work / current_work
                ),
                "max_fadc24_lower_bytes": max(
                    (row["fadc24_lower_bytes"] for row in rows), default=0
                ),
                "max_fadc24_upper_bytes": max(
                    (row["fadc24_upper_bytes"] for row in rows), default=0
                ),
            }
        )
    return {
        "schema_version": 1,
        "source_kind": "H67 profile100 ordered trace容量上下界",
        "contract": {
            "slot_bytes": SLOT_BYTES,
            "raw_head_bits": RAW_HEAD_BITS,
            "rtl_contract_change_required": (
                "将RAW_PAYLOAD_BITS=6642与SLOT_CAPACITY_BITS=6656拆分；"
                "当前head-slot adapter将两者混用"
            ),
            "fadc24_bytes": "16 + 3*T + sum(min(fanout_i, 21))",
            "limitation": "ordered trace没有逐term fanout，因此ambiguous项必须由位级trace消歧",
        },
        "records": records,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GateStack FADC24 Profile100容量上下界审计",
        "",
        "## 结论表",
        "",
        "| Stage | head实例 | 当前IPD32W fallback | FADC24 guaranteed-fit | FADC24 ambiguous | FADC24 impossible | 最大下界byte | 最大上界byte |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["records"]:
        lines.append(
            f"| S{record['stage']} | {record['head_instances']} | "
            f"{record['ipd32w_raw_fallbacks']} ({record['ipd32w_raw_fallback_rate']:.3%}) | "
            f"{record['fadc24_guaranteed_fit']} ({record['fadc24_guaranteed_fit_rate']:.3%}) | "
            f"{record['fadc24_ambiguous']} ({record['fadc24_ambiguous_rate']:.3%}) | "
            f"{record['fadc24_impossible_fit']} ({record['fadc24_impossible_fit_rate']:.3%}) | "
            f"{record['max_fadc24_lower_bytes']} | {record['max_fadc24_upper_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Projection term工作量边界",
            "",
            "| Stage | 当前执行term | FADC24最好界 | FADC24最坏界 | 相对当前减少范围 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for record in result["records"]:
        low = record["fadc24_worst_case_term_reduction"]
        high = record["fadc24_best_case_term_reduction"]
        lines.append(
            f"| S{record['stage']} | {record['current_executed_terms']} | "
            f"{record['fadc24_best_case_executed_terms']} | "
            f"{record['fadc24_worst_case_executed_terms']} | "
            f"{low:.2%} 到 {high:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 判定定义",
            "",
            f"- 物理head槽容量为{result['contract']['slot_bytes']} byte。",
            "- guaranteed-fit：给定term/event/max_fanout后，任何合法逐term fanout分布都能装入。",
            "- impossible：即使采用最有利的fanout集中分布仍装不入。",
            "- ambiguous：是否装入取决于真实逐term fanout，必须由位级trace消歧。",
            "",
            "## 证据边界",
            "",
            "- 本结果覆盖profile100中所有ordered head实例，但不是FADC24 RTL结果。",
            "- 上下界只判断容量，不包含24-bit解包、bitmap扫描和回压的周期/面积代价。",
            "- 若ambiguous占比较高，不能凭单样本结果冻结FADC24。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = analyze(profile)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
