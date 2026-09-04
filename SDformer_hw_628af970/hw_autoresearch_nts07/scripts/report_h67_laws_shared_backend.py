#!/usr/bin/env python3
"""Seal Motion shared-backend row-pipeline evidence from phase-split RTL."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PHASE_RE = re.compile(
    r"LAWS_PHASE row=(?P<row>\d+) stage=(?P<stage>\d+) block=(?P<block>\d+) "
    r"head=(?P<head>\d+) build=(?P<build>\d+) emit=(?P<emit>\d+) "
    r"total=(?P<total>\d+) slots=(?P<slots>\d+) active=(?P<active>\d+) "
    r"classes=(?P<classes>\d+)"
)
SUM_RE = re.compile(
    r"LAWS_PHASE_SUM build=(?P<build>\d+) emit=(?P<emit>\d+) "
    r"seq=(?P<seq>\d+) rows=(?P<rows>\d+)"
)


def shared_backend_wall(rows: list[dict[str, int]]) -> int:
    """One shared encoder and one shared emit path, two workspaces."""
    wall = 0
    prev_emit = 0
    for row in rows:
        wall += max(prev_emit, row["build"])
        prev_emit = row["emit"]
    return wall + prev_emit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-json", type=Path, default=None)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_phase_split_2s" not in text:
        raise ValueError("phase-split log missing PASS")
    rows = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in PHASE_RE.finditer(text)
    ]
    summary = SUM_RE.search(text)
    if summary is None or len(rows) != 138:
        raise ValueError(f"expected 138 rows, got {len(rows)}")

    seq = sum(row["total"] for row in rows)
    pipe_all = shared_backend_wall(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["block"])].append(row)
    pipe_by_block = 0
    for members in groups.values():
        pipe_by_block += shared_backend_wall(members)

    builds = [row["build"] for row in rows]
    emits = [row["emit"] for row in rows]
    report = {
        "schema": "h67_laws_shared_backend_phase_v1",
        "status": "PASS",
        "evidence": "[rtl]+[phase-split]+[shared-backend-schedule-model]",
        "rows": 138,
        "no_backpressure": True,
        "sequential_cycles": seq,
        "build_cycles": sum(builds),
        "emit_cycles": sum(emits),
        "build_mean": sum(builds) / len(builds),
        "emit_mean": sum(emits) / len(emits),
        "emit_fraction": sum(emits) / seq,
        "shared_backend_all138_cycles": pipe_all,
        "shared_backend_all138_speedup": seq / pipe_all,
        "shared_backend_per_block_cycles": pipe_by_block,
        "shared_backend_per_block_speedup": seq / pipe_by_block,
        "interpretation": (
            "Schedule is encode(i+1) || emit(i) with one encoder and one emit "
            "path. This is NOT two full cores encoding at once."
        ),
        "claim_boundary": [
            "Phase split is RQTB2S RTL with ready=1, not the LFSR fair-baseline TB.",
            "Wall time is a schedule applied to measured build/emit, not a dual-workspace netlist yet.",
            "Not ASIC PPA or energy.",
        ],
        "rows_detail": rows,
    }
    if args.area_json and args.area_json.is_file():
        report["area_proxy"] = json.loads(args.area_json.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion shared-backend row-pipeline phase split",
        "",
        "> 证据：`[rtl]+[shared-backend-schedule-model]`。ready=1，不是 LFSR 公平包。",
        "",
        f"- 顺序：{seq} cycle（build {sum(builds)}, emit {sum(emits)}, emit占比 {sum(emits)/seq:.1%}）",
        f"- 全 138 行一条流水：{pipe_all} cycle，**{seq/pipe_all:.4f}x**",
        f"- 按 stage/block 断开：{pipe_by_block} cycle，**{seq/pipe_by_block:.4f}x**",
        "",
        "这是共享 encoder + 共享 Shiftmax、双 directory/K 的合法上界，不是整核双复制。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS LAWS phase seq={seq} pipe_block={pipe_by_block} "
        f"speedup={seq/pipe_by_block:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
