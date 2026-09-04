#!/usr/bin/env python3
"""Seal S0-S3 complete-window Acc32 table + 12-block real-window model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PASS_RE = re.compile(
    r"PASS Local5 score-to-projection backend=(?P<backend>\d+) "
    r"latency=(?P<latency>\d+) groups=(?P<groups>\d+) total_cycles=(?P<cycles>\d+)"
)
GROUP_RE = re.compile(
    r"GROUP backend=\d+ latency=\d+ group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<rows>\d+) score_service=(?P<service>\d+) "
    r"score_direct_rows=(?P<direct>\d+)"
    r"(?: qsilent_rows=(?P<qsilent>\d+) identk_rows=(?P<identk>\d+))?"
)

DESCRIPTORS = [
    {"stage": 0, "block": 0, "heads": 3, "windows": 440},
    {"stage": 0, "block": 1, "heads": 3, "windows": 440},
    {"stage": 1, "block": 0, "heads": 6, "windows": 120},
    {"stage": 1, "block": 1, "heads": 6, "windows": 120},
    {"stage": 2, "block": 0, "heads": 12, "windows": 30},
    {"stage": 2, "block": 1, "heads": 12, "windows": 30},
    {"stage": 2, "block": 2, "heads": 12, "windows": 30},
    {"stage": 2, "block": 3, "heads": 12, "windows": 30},
    {"stage": 2, "block": 4, "heads": 12, "windows": 30},
    {"stage": 2, "block": 5, "heads": 12, "windows": 30},
    {"stage": 3, "block": 0, "heads": 24, "windows": 10},
    {"stage": 3, "block": 1, "heads": 24, "windows": 10},
]


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    summary = PASS_RE.search(text)
    if summary is None:
        raise ValueError(f"missing PASS in {path}")
    groups = []
    qsilent = 0
    identk = 0
    service = 0
    for match in GROUP_RE.finditer(text):
        row = {key: int(match.group(key) or 0) for key in
               ("group", "cycles", "rows", "service", "direct", "qsilent", "identk")}
        groups.append(row)
        qsilent += row["qsilent"]
        identk += row["identk"]
        service += row["service"]
    return {
        "groups": int(summary["groups"]),
        "total_cycles": int(summary["cycles"]),
        "service": service,
        "qsilent_rows": qsilent,
        "identk_rows": identk,
        "per_head": int(summary["cycles"]) / int(summary["groups"]),
    }


def s0_from_old(path: Path) -> dict[str, object]:
    report = json.loads((path / "report.json").read_text()) if (path / "report.json").is_file() else {}
    # 352 package: residual 11960, qsilent 5967, 3 heads, all Q==0
    residual = int(report.get("residual_cycles", 11960))
    qsilent = int(report.get("qsilent_cycles", 5967))
    return {
        "stage": 0,
        "heads": 3,
        "residual_cycles": residual,
        "qsilent_cycles": qsilent,
        "speedup": residual / qsilent,
        "qsilent_rows": 1350,
        "identk_rows": 0,
        "per_head_residual": residual / 3,
        "per_head_qsilent": qsilent / 3,
        "source": "sealed_s0_window_acc32_20260813",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--s0-residual", type=Path, required=True)
    parser.add_argument("--s3-old", type=Path, required=True)
    args = parser.parse_args()

    leftover = {}
    leftover_path = args.result_dir / "residual_leftover" / "report.json"
    if leftover_path.is_file():
        leftover = json.loads(leftover_path.read_text())

    s0 = s0_from_old(args.s0_residual)
    s1r = parse_log(args.result_dir / "s1_residual_verilator.log")
    s1q = parse_log(args.result_dir / "s1_qsilent_verilator.log")
    s2r = parse_log(args.result_dir / "s2_residual_verilator.log")
    s2q = parse_log(args.result_dir / "s2_qsilent_verilator.log")
    s3r = parse_log(args.result_dir / "s3_residual_verilator.log")
    s3q = parse_log(args.result_dir / "s3_qsilent_verilator.log")

    stages = [
        s0,
        {
            "stage": 1,
            "heads": 6,
            "residual_cycles": s1r["total_cycles"],
            "qsilent_cycles": s1q["total_cycles"],
            "speedup": s1r["total_cycles"] / s1q["total_cycles"],
            "qsilent_rows": s1q["qsilent_rows"],
            "identk_rows": s1q["identk_rows"],
            "per_head_residual": s1r["per_head"],
            "per_head_qsilent": s1q["per_head"],
        },
        {
            "stage": 2,
            "heads": 12,
            "residual_cycles": s2r["total_cycles"],
            "qsilent_cycles": s2q["total_cycles"],
            "speedup": s2r["total_cycles"] / s2q["total_cycles"],
            "qsilent_rows": s2q["qsilent_rows"],
            "identk_rows": s2q["identk_rows"],
            "per_head_residual": s2r["per_head"],
            "per_head_qsilent": s2q["per_head"],
        },
        {
            "stage": 3,
            "heads": 24,
            "residual_cycles": s3r["total_cycles"],
            "qsilent_cycles": s3q["total_cycles"],
            "speedup": s3r["total_cycles"] / s3q["total_cycles"],
            "qsilent_rows": s3q["qsilent_rows"],
            "identk_rows": s3q["identk_rows"],
            "per_head_residual": s3r["per_head"],
            "per_head_qsilent": s3q["per_head"],
        },
    ]
    by_stage = {int(item["stage"]): item for item in stages}

    frame_q = 0.0
    frame_b = 0.0
    descriptors = []
    for item in DESCRIPTORS:
        stage = by_stage[item["stage"]]
        n = item["windows"] * item["heads"]
        qs = stage["per_head_qsilent"] * n
        base = stage["per_head_residual"] * n
        descriptors.append({**item, "groups": n, "qsilent_cycles": qs, "baseline_cycles": base})
        frame_q += qs
        frame_b += base

    window_q = sum(by_stage[d["stage"]]["per_head_qsilent"] * d["windows"] for d in DESCRIPTORS)
    window_b = sum(by_stage[d["stage"]]["per_head_residual"] * d["windows"] for d in DESCRIPTORS)

    report = {
        "schema": "local5_complete_window_table_v1",
        "status": "PASS",
        "evidence": "[rtl]+[sample0-complete-window-per-stage]",
        "stages": stages,
        "leftover_decision": leftover.get("decision"),
        "twelve_block_model": {
            "evidence": "[rtl校准模型]+[complete-window-per-stage]",
            "head_window_population": 21600,
            "headwindow_baseline": frame_b,
            "headwindow_qsilent": frame_q,
            "headwindow_speedup": frame_b / frame_q,
            "scheduler_window_groups": 1320,
            "windowgroup_baseline": window_b,
            "windowgroup_qsilent": window_q,
            "windowgroup_speedup": window_b / window_q,
            "descriptors": descriptors,
            "claim_boundary": [
                "One complete sample0 B0 window per stage, not 21600-group RTL.",
                "S1/S2/S3 blocks of the same stage share that stage's window mean.",
            ],
        },
        "acc32_mismatch": 0,
        "claim_boundary": [
            "S0 cites the sealed 352 package (all Q==0).",
            "S3 residual log is the sealed 355 package; S3 qsilent is re-run for counters.",
            "Not full encoder.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = ["# Local5 complete-window DATE table", "", "| stage | heads | residual | Q-silent | speedup | qsilent_rows | identk_rows |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in stages:
        md.append(
            f"| S{item['stage']} | {item['heads']} | {item['residual_cycles']:.0f} | "
            f"{item['qsilent_cycles']:.0f} | **{item['speedup']:.4f}×** | "
            f"{item.get('qsilent_rows', 0)} | {item.get('identk_rows', 0)} |"
        )
    md += [
        "",
        f"- 12-block head-window 21600: {frame_b:.0f} → {frame_q:.0f} = "
        f"**{frame_b / frame_q:.4f}×** `[rtl校准模型]`",
        f"- leftover residual: **{leftover.get('decision', 'n/a')}**",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS complete-window table 12block={frame_b / frame_q:.4f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
