#!/usr/bin/env python3
"""汇总45-head RTL latency并评估C1双workspace的有序重叠上界。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LATENCY_RE = re.compile(
    r"LATENCY index=(?P<index>\d+) stage=(?P<stage>\d+) "
    r"head=(?P<head>\d+) format=(?P<format>\d+) terms=(?P<terms>\d+) "
    r"events=(?P<events>\d+) words=(?P<words>\d+) cycles=(?P<cycles>\d+)"
)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def schedule(rows: list[dict[str, Any]], workspaces: int) -> dict[str, Any]:
    bank_free = [0] * workspaces
    capture_free = 0
    serializer_free = 0
    capture_stall = 0
    service_cycles = 0
    timeline = []
    for row in rows:
        bank = min(range(workspaces), key=lambda item: bank_free[item])
        capture_start = max(capture_free, bank_free[bank])
        capture_stall += capture_start - capture_free
        capture_end = capture_start + row["capture_cycles_model"]
        ready = capture_end + row["analyze_cycles_model"]
        service_start = max(serializer_free, ready)
        done = service_start + row["service_cycles_model"]
        capture_free = capture_end
        serializer_free = done
        bank_free[bank] = done
        service_cycles += row["service_cycles_model"]
        timeline.append(
            {
                "index": row["index"],
                "bank": bank,
                "capture_start": capture_start,
                "ready": ready,
                "service_start": service_start,
                "done": done,
            }
        )
    makespan = max(bank_free, default=0)
    return {
        "workspaces": workspaces,
        "makespan_cycles": makespan,
        "capture_port_stall_cycles": capture_stall,
        "serializer_service_cycles": service_cycles,
        "serializer_utilization": service_cycles / makespan if makespan else 0.0,
        "timeline": timeline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    latency_rows = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in LATENCY_RE.finditer(args.log.read_text(encoding="utf-8"))
    ]
    if len(latency_rows) != manifest["head_count"]:
        raise ValueError("latency记录数与manifest不一致")

    rows = []
    for measured, source in zip(latency_rows, manifest["rows"], strict=True):
        for key in ("index", "stage", "head", "terms", "events"):
            if measured[key] != int(source[key]):
                raise ValueError(f"latency与manifest字段不一致: {key}")
        capture_cycles = 163
        analyze_cycles = 1 + 33 * int(source["active_classes"])
        service_cycles = measured["cycles"] - capture_cycles - analyze_cycles
        if service_cycles <= 0:
            raise ValueError("分解得到非正service cycle")
        rows.append(
            {
                **source,
                "rtl_latency_cycles": measured["cycles"],
                "capture_cycles_model": capture_cycles,
                "analyze_cycles_model": analyze_cycles,
                "service_cycles_model": service_cycles,
            }
        )

    stage_rows = []
    c0_total = 0
    c1_total = 0
    unlimited_total = 0
    for stage in range(4):
        selected = [row for row in rows if row["stage"] == stage]
        c0 = schedule(selected, 1)
        c1 = schedule(selected, 2)
        unlimited = schedule(selected, len(selected))
        c0_total += c0["makespan_cycles"]
        c1_total += c1["makespan_cycles"]
        unlimited_total += unlimited["makespan_cycles"]
        stage_rows.append(
            {
                "stage": stage,
                "heads": len(selected),
                "rtl_latency_sum": sum(row["rtl_latency_cycles"] for row in selected),
                "rtl_latency_p50": percentile(
                    [row["rtl_latency_cycles"] for row in selected], 0.50
                ),
                "rtl_latency_p95": percentile(
                    [row["rtl_latency_cycles"] for row in selected], 0.95
                ),
                "rtl_latency_max": max(row["rtl_latency_cycles"] for row in selected),
                "c0_model": c0,
                "c1_model": c1,
                "unlimited_workspace_model": unlimited,
                "c1_speedup": c0["makespan_cycles"] / c1["makespan_cycles"],
            }
        )

    c0_rtl_sum = sum(row["rtl_latency_cycles"] for row in rows)
    if c0_total != c0_rtl_sum:
        raise AssertionError("单workspace模型未复现RTL latency总和")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "evidence": {
            "latency": "[rtl]",
            "c1_overlap": "[模型]",
            "source_scope": "sample0/B0/window0四stage全部45 head",
        },
        "head_count": len(rows),
        "rtl": {
            "latency_sum_cycles": c0_rtl_sum,
            "latency_p50": percentile(
                [row["rtl_latency_cycles"] for row in rows], 0.50
            ),
            "latency_p95": percentile(
                [row["rtl_latency_cycles"] for row in rows], 0.95
            ),
            "latency_p99": percentile(
                [row["rtl_latency_cycles"] for row in rows], 0.99
            ),
            "latency_max": max(row["rtl_latency_cycles"] for row in rows),
        },
        "stage_bounded_overlap": {
            "c0_cycles": c0_total,
            "c1_cycles": c1_total,
            "unlimited_workspace_cycles": unlimited_total,
            "c1_speedup": c0_total / c1_total,
            "c1_cycle_reduction": 1.0 - c1_total / c0_total,
            "unlimited_speedup": c0_total / unlimited_total,
        },
        "stages": stage_rows,
        "rows": rows,
        "model_contract": [
            "每head捕获固定163拍：head begin加162 token",
            "canonical分析按1+33×active_class建模，与当前FSM一致",
            "service周期由实测RTL latency扣除capture/analyze得到",
            "C1有两个完整workspace、一个共享capture输入和一个共享Serializer，按输入顺序提交",
            "stage边界清空重叠，不跨stage预取，属于保守于跨stage连续流的模型",
        ],
        "limits": [
            "C1尚未实现RTL，结果只能标为模型",
            "真实trace只覆盖sample0/B0/window0",
            "回放逐word验证不计入head build latency",
            "尚无目标库频率、面积或功耗",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# GateStack四Stage全部45-Head RTL Latency与C1收益判定",
        "",
        "## 1. 结论",
        "",
        f"完整C0对四stage全部{len(rows)}个真实head逐word回放零失配。"
        f"head build latency总计{c0_rtl_sum}拍，p50={result['rtl']['latency_p50']}、"
        f"p95={result['rtl']['latency_p95']}、p99={result['rtl']['latency_p99']}、"
        f"max={result['rtl']['latency_max']}拍。",
        "",
        f"按两个完整workspace、单capture输入、共享单Serializer和严格有序提交建模，"
        f"stage边界清空时C1为{c1_total}拍，相对C0减少"
        f"{result['stage_bounded_overlap']['c1_cycle_reduction']:.2%}，加速"
        f"{result['stage_bounded_overlap']['c1_speedup']:.3f}x。该项是`[模型]`，不是RTL结果。",
        "",
        "## 2. 分Stage",
        "",
        "| Stage | Head | C0 RTL/模型 | C1模型 | C1加速 | p50 | p95 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stage_rows:
        lines.append(
            f"| {stage['stage']} | {stage['heads']} | "
            f"{stage['c0_model']['makespan_cycles']} | "
            f"{stage['c1_model']['makespan_cycles']} | "
            f"{stage['c1_speedup']:.3f}x | {stage['rtl_latency_p50']} | "
            f"{stage['rtl_latency_p95']} | {stage['rtl_latency_max']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 关键长尾",
            "",
            "| Stage/Head | 格式 | term | event | word | RTL cycle | service模型 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["rtl_latency_cycles"], reverse=True)[:8]:
        lines.append(
            f"| S{row['stage']}/H{row['head']} | {row['format']} | "
            f"{row['terms']} | {row['events']} | {row['word_count']} | "
            f"{row['rtl_latency_cycles']} | {row['service_cycles_model']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 决策",
            "",
            "C1只有在上述模型经RTL复现后才能晋级。实现时必须共享一套Serializer和slot写口，"
            "两个workspace只复制RAW scratch、class directory、bitmap和局部控制；不得用双Serializer伪造重叠收益。",
            "",
            "当前最大长尾是S3/H4 FADC。下一步应先分解其workspace output stall、bitmap term和Serializer pack/commit周期；"
            "若直接流式commit能显著缩短service，必须用优化后的service重新评估C1，避免高估双workspace收益。",
            "",
            "## 5. 证据边界",
            "",
            "- 45-head latency与逐word回放是`[rtl]`；",
            "- C1调度结果是`[模型]`，尚无C1 RTL；",
            "- scope仅为sample0/B0/window0，不是全数据集；",
            "- 没有目标库PPA、全encoder FPS或功耗结论。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
