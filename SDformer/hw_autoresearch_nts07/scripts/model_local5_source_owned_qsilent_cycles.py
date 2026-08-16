#!/usr/bin/env python3
"""Bounded cycle/Pareto model for source-owned Local5 Q-silent scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TOKENS = 450
GROUP_RE = re.compile(r"(\w+)=([^ ]+)")


def parse_log(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("GROUP "):
            continue
        fields = {key: value for key, value in GROUP_RE.findall(line)}
        required = {"group", "cycles", "score_rows", "qsilent_rows", "identk_rows"}
        if not required.issubset(fields):
            raise ValueError(f"GROUP行字段不全: {line}")
        rows.append({key: int(fields[key]) for key in fields if fields[key].lstrip("-").isdigit()})
    if not rows:
        raise ValueError("log没有GROUP记录")
    if [row["group"] for row in rows] != list(range(len(rows))):
        raise ValueError("GROUP顺序不是从0连续递增")
    return rows


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = [int(row["current_cycles"]) for row in rows]
    destination = [int(row["destination_pipeline_cycles"]) for row in rows]
    source = [int(row["source_owned_cycles"]) for row in rows]
    current_total = sum(current)
    destination_total = sum(destination)
    source_total = sum(source)
    return {
        "groups": len(rows),
        "cycles": {
            "current": current_total,
            "destination_pipeline": destination_total,
            "popcount_sidecar": destination_total,
            "source_owned": source_total,
        },
        "speedup_vs_current": {
            "destination_pipeline": current_total / destination_total,
            "popcount_sidecar": current_total / destination_total,
            "source_owned": current_total / source_total,
        },
        "source_vs_destination_cycle_ratio": source_total / destination_total,
        "source_selected_groups": sum(bool(row["source_mode"]) for row in rows),
        "distributions": {
            name: {
                "mean": float(np.mean(values)),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "max": max(values),
            }
            for name, values in (
                ("current", current),
                ("destination_pipeline", destination),
                ("source_owned", source),
            )
        },
    }


def model(
    profile_path: Path, log_path: Path, baseline_name: str = "q0_ident_overlap"
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != "local5_source_owned_qsilent_profile_v1":
        raise ValueError("输入profile schema不匹配")
    profile_rows = profile.get("rows")
    if not isinstance(profile_rows, list) or not profile_rows:
        raise ValueError("profile缺少逐group数据")
    log_rows = parse_log(log_path)
    if len(profile_rows) != len(log_rows):
        raise ValueError("profile与RTL log group数量不同")

    rows: list[dict[str, Any]] = []
    for profile_row, log_row in zip(profile_rows, log_rows):
        if int(profile_row["group"]) != int(log_row["group"]):
            raise ValueError("profile与RTL log group顺序不一致")
        qsilent = int(profile_row["qsilent_destinations"])
        if qsilent != int(log_row["qsilent_rows"]):
            raise ValueError(
                f"group {profile_row['group']} Q-silent人口不一致: "
                f"profile={qsilent}, rtl={log_row['qsilent_rows']}"
            )
        current = int(log_row["cycles"])

        # Strong baseline: keep five parallel 32-bit popcount trees and allow
        # one Q-silent destination to enter the score pipeline every cycle.
        destination_saving = qsilent
        destination_cycles = current - destination_saving

        # Source-owned mode uses one 32-bit popcount per source and a fixed
        # one-source/cycle sweep. The affine five-color map gives each of the
        # source's five destinations a distinct single-write bank, so no
        # runtime bank replay is charged. A group uses this mode only when its
        # fixed sweep beats the existing two-cycle-per-silent-row path.
        source_saving = max(0, 2 * qsilent - TOKENS)
        source_cycles = current - source_saving
        if min(destination_cycles, source_cycles) <= 0:
            raise AssertionError("cycle model produced nonpositive cycles")
        rows.append(
            {
                "group": int(profile_row["group"]),
                "sample": int(profile_row["sample"]),
                "stage": int(profile_row["stage"]),
                "block": int(profile_row["block"]),
                "qsilent_destinations": qsilent,
                "current_cycles": current,
                "destination_pipeline_cycles": destination_cycles,
                "source_owned_cycles": source_cycles,
                "destination_pipeline_saving": destination_saving,
                "source_owned_saving": source_saving,
                "source_mode": source_saving > 0,
            }
        )

    by_stage: dict[str, Any] = {}
    for stage in range(4):
        selected = [row for row in rows if int(row["stage"]) == stage]
        if selected:
            by_stage[str(stage)] = aggregate(selected)
    profile_global = profile["global"]
    return {
        "schema": "local5_source_owned_qsilent_cycle_model_v1",
        "status": "FIRST_ORDER_MODEL_ONLY",
        "source": {
            "profile": str(profile_path.resolve()),
            "rtl_log": str(log_path.resolve()),
            "rtl_baseline": baseline_name,
            "groups": len(rows),
            "samples": profile["source"]["samples"],
            "scope": "OUT_DIM=2 score+projection tile; not encoder",
        },
        "global": aggregate(rows),
        "by_stage": by_stage,
        "work": {
            "baseline_qsilent_popcounts": profile_global["totals"][
                "baseline_qsilent_popcounts"
            ],
            "source_owned_all_source_popcounts": profile_global["totals"][
                "source_owned_all_source_popcounts"
            ],
            "popcount_reduction": profile_global["reductions"][
                "popcount_all_source"
            ],
            "baseline_score_k_read_bits": profile_global["totals"][
                "baseline_score_k_read_bits"
            ],
            "source_owned_score_k_read_bits": profile_global["totals"][
                "source_owned_score_k_read_bits"
            ],
            "score_k_read_bit_reduction": profile_global["reductions"][
                "score_k_read_bits"
            ],
        },
        "logical_state_bits": {
            "popcount_sidecar_three_rows_15x6": 3 * 15 * 6,
            "qzero_plane_1bit_x450": 450,
            "three_row_five_role_score_2bit": 3 * 15 * 5 * 2,
            "three_row_role_valid_1bit": 3 * 15 * 5,
            "total": 450 + 3 * 15 * 5 * 2 + 3 * 15 * 5,
            "note": "logic bits only; excludes SRAM periphery, tags and routing",
        },
        "contracts": [
            f"Current cycles are fresh {baseline_name} RTL group cycles; frozen main-table numbers are unchanged.",
            "Destination-pipeline baseline is an optimistic five-popcount-tree baseline: one silent destination/cycle and zero added routing cost.",
            "Popcount-sidecar baseline computes one 6-bit statistic when each K enters the common store, then reads five 6-bit stripe taps per destination; its score-side throughput equals the optimistic destination pipeline.",
            "Source-owned mode uses one source/cycle. bank(x,y)=(x+2y) mod5 maps one source's five consumers to distinct write banks.",
            "The mode decision uses Q-silent population known from the prebuilt Q-zero bitmap; it does not inspect future score/gate results.",
            "The model serially charges a full 450-source sweep and therefore does not claim free overlap with residual QFSA.",
        ],
        "limits": [
            "Source-owned cycle savings are a bounded CPU model, not RTL cycles.",
            "The current RTL baseline and candidate have different score input contracts; a fair RTL must use the same Q/K source store and memory latency.",
            "The destination-pipeline baseline may match candidate throughput with about five popcount trees; only synthesis/activity can resolve the Pareto.",
            "The popcount-sidecar baseline may dominate source-owned routing. Its K-ingress serialization/tree count and stripe read implementation must be charged under the same source-store contract.",
            "All cycles are OUT_DIM=2 tile scope, not a 12-block scheduler or full encoder.",
        ],
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    global_row = report["global"]
    cycles = global_row["cycles"]
    speedup = global_row["speedup_vs_current"]
    work = report["work"]
    lines = [
        "# Local5 source-owned Q-silent 有界周期模型",
        "",
        f"- 输入：{report['source']['groups']} group / "
        f"{report['source']['samples']} sample",
        f"- 当前基线：{report['source']['rtl_baseline']} [rtl]",
        "- 候选：单source popcount + 五色充分统计多播 [CPU有限资源模型]",
        "- 口径：OUT_DIM=2 score+projection tile；不是encoder",
        "",
        "| 架构 | 总周期 | vs当前 | popcount树/并行统计 |",
        "|---|---:|---:|---:|",
        f"| {report['source']['rtl_baseline']} RTL | {cycles['current']} | "
        "1.000x | 5个组合popcount |",
        f"| destination-pipeline强上界 | {cycles['destination_pipeline']} | "
        f"{speedup['destination_pipeline']:.4f}x | 5 |",
        f"| popcount-sidecar强基线 | {cycles['popcount_sidecar']} | "
        f"{speedup['popcount_sidecar']:.4f}x | 1（K写入侧） |",
        f"| source-owned bounded | {cycles['source_owned']} | "
        f"{speedup['source_owned']:.4f}x | 1 |",
        "",
        f"source-owned相对destination-pipeline周期比："
        f"{global_row['source_vs_destination_cycle_ratio']:.4f}；"
        f"启用source模式group：{global_row['source_selected_groups']}/"
        f"{global_row['groups']}。",
        "",
        "## 工作与状态",
        "",
        f"- popcount evaluation：{work['baseline_qsilent_popcounts']} -> "
        f"{work['source_owned_all_source_popcounts']} "
        f"({work['popcount_reduction']:.2%}减少)。",
        f"- score侧K读取bit：{work['baseline_score_k_read_bits']} -> "
        f"{work['source_owned_score_k_read_bits']} "
        f"({work['score_k_read_bit_reduction']:.2%}减少)。",
        f"- 候选逻辑状态：{report['logical_state_bits']['total']} bit；"
        "这不是macro面积。",
        f"- sidecar三行统计状态："
        f"{report['logical_state_bits']['popcount_sidecar_three_rows_15x6']} bit；"
        "未计写入staging、valid和读出寄存器。",
        "",
        "## 分stage",
        "",
        "| stage | groups | current | source-owned | speedup | source/dest-pipe |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in report["by_stage"].items():
        lines.append(
            f"| {stage} | {row['groups']} | {row['cycles']['current']} | "
            f"{row['cycles']['source_owned']} | "
            f"{row['speedup_vs_current']['source_owned']:.4f}x | "
            f"{row['source_vs_destination_cycle_ratio']:.4f} |"
        )
    lines.extend(["", "## 合同", ""])
    lines.extend(f"- {item}" for item in report["contracts"])
    lines.extend(["", "## 边界", ""])
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("results/local5_source_owned_qsilent_profile_20260814/report.json"),
    )
    parser.add_argument(
        "--rtl-log",
        type=Path,
        default=Path(
            "results/local5_qsilent_overlap_ablation_20260813/"
            "q0_ident_overlap.log"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/local5_source_owned_qsilent_model_20260814"),
    )
    parser.add_argument(
        "--baseline-name",
        default="q0_ident_overlap",
        help="Human-readable RTL baseline label recorded in the artifact.",
    )
    args = parser.parse_args()
    report = model(args.profile, args.rtl_log, args.baseline_name)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
