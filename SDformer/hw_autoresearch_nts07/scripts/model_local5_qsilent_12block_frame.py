#!/usr/bin/env python3
"""12-block Local5 frame ledger from Q-silent RTL group cycles."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+)"
)

# Frozen Local5 encoder descriptors (see qfit_local5_encoder_job_scheduler).
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


def parse_groups(path: Path) -> list[dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if "PASS Local5 score-to-projection" not in text:
        raise ValueError(f"{path} missing PASS")
    return [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in GROUP_RE.finditer(text)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsilent-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    selection = manifest["selection"]["rows"]
    qsilent = parse_groups(args.qsilent_log)
    baseline = parse_groups(args.baseline_log)
    if len(qsilent) != 100 or len(baseline) != 100 or len(selection) != 100:
        raise ValueError("need 100 groups")

    by_stage: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"qsilent": [], "baseline": []}
    )
    for meta, qs, base in zip(selection, qsilent, baseline, strict=True):
        stage = int(meta["stage"])
        by_stage[stage]["qsilent"].append(qs["cycles"])
        by_stage[stage]["baseline"].append(base["cycles"])

    descriptors = []
    frame_qsilent = 0
    frame_baseline = 0
    groups = 0
    for item in DESCRIPTORS:
        stage = item["stage"]
        qs_mean = sum(by_stage[stage]["qsilent"]) / len(by_stage[stage]["qsilent"])
        base_mean = sum(by_stage[stage]["baseline"]) / len(by_stage[stage]["baseline"])
        # One encoder group is one window of that descriptor's head count?
        # The 100-group RTL is one T450 group = one head-window.
        n_groups = item["windows"] * item["heads"]
        qs_cycles = qs_mean * n_groups
        base_cycles = base_mean * n_groups
        descriptors.append(
            {
                **item,
                "groups": n_groups,
                "stage_qsilent_mean": qs_mean,
                "stage_baseline_mean": base_mean,
                "qsilent_cycles": qs_cycles,
                "baseline_cycles": base_cycles,
            }
        )
        frame_qsilent += qs_cycles
        frame_baseline += base_cycles
        groups += n_groups

    report = {
        "schema": "local5_qsilent_12block_frame_v1",
        "evidence": "[rtl校准模型]+[12-block-descriptor]",
        "groups": groups,
        "expected_groups": 1320,
        "note": (
            "1320 is the scheduler window-group count, not head-windows. "
            "Head-window population is sum(windows*heads)=21600. "
            "Both ledgers are reported."
        ),
        "head_window_population": groups,
        "scheduler_window_groups": 1320,
        "frame_headwindow_qsilent_cycles": frame_qsilent,
        "frame_headwindow_baseline_cycles": frame_baseline,
        "frame_headwindow_speedup": frame_baseline / frame_qsilent,
        "descriptors": descriptors,
        "claim_boundary": [
            "Uses stage-mean of the 100-group Q-silent RTL, not 21600-group RTL.",
            "Does not include ATLIF, IO, or output-tile replay.",
            "Scheduler 1320 counts windows, not heads; do not mix the two.",
        ],
    }
    # Also scale 1320 window-groups using mixed stage means weighted by windows.
    window_qs = 0.0
    window_base = 0.0
    for item in descriptors:
        window_qs += item["stage_qsilent_mean"] * item["windows"]
        window_base += item["stage_baseline_mean"] * item["windows"]
    report["frame_windowgroup_qsilent_cycles"] = window_qs
    report["frame_windowgroup_baseline_cycles"] = window_base
    report["frame_windowgroup_speedup"] = window_base / window_qs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Local5 Q-silent 12-block frame ledger",
        "",
        "> 证据：`[rtl校准模型]`。100-group 真实 RTL 的 stage 均值外推。",
        "",
        f"- head-window 总体 21600：{frame_baseline:.0f} → {frame_qsilent:.0f}，"
        f"**{frame_baseline/frame_qsilent:.4f}x**",
        f"- scheduler window-group 1320：{window_base:.0f} → {window_qs:.0f}，"
        f"**{window_base/window_qs:.4f}x**",
        "",
        "不是 21600-group RTL，也不是 full encoder。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS Local5 12-block model headwindow={frame_baseline/frame_qsilent:.4f}x "
        f"windowgroup={window_base/window_qs:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
