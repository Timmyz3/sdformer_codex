#!/usr/bin/env python3
"""Seal Motion exact empty-K row skip evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SKIP_RE = re.compile(
    r"EMPTY_SKIP row=(?P<row>\d+) stage=(?P<stage>\d+) block=(?P<block>\d+) cycles=(?P<cycles>\d+)"
)
KEEP_RE = re.compile(
    r"EMPTY_KEEP row=(?P<row>\d+) stage=(?P<stage>\d+) block=(?P<block>\d+) cycles=(?P<cycles>\d+)"
)
SUM_RE = re.compile(
    r"EMPTY_SKIP_SUM empty=(?P<empty>\d+) dense=(?P<dense>\d+) "
    r"skip_cycles=(?P<skip_cycles>\d+) run_cycles=(?P<run_cycles>\d+) total=(?P<total>\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--phase-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    if "PASS tb_h67_empty_row_skip_2s" not in text:
        raise ValueError("empty-skip log missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing EMPTY_SKIP_SUM")
    phase = json.loads(args.phase_report.read_text(encoding="utf-8"))
    empty = int(summary["empty"])
    dense = int(summary["dense"])
    skip_cycles = int(summary["skip_cycles"])
    run_cycles = int(summary["run_cycles"])
    total = int(summary["total"])
    baseline = int(phase["sequential_cycles"])
    report = {
        "schema": "h67_empty_row_skip_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[ep35-sample0-window0]+[occupancy-sideband]",
        "empty_rows": empty,
        "dense_rows": dense,
        "skip_cycles": skip_cycles,
        "dense_run_cycles": run_cycles,
        "total_cycles": total,
        "phase_split_baseline": baseline,
        "speedup_vs_phase_split": baseline / total,
        "cycle_reduction": 1.0 - total / baseline,
        "occupancy_contract": (
            "row_k_present is the OR of the 450 K words, stored when K is written. "
            "Empty rows emit no gated-K tokens, so skipping the 225-pair scan is exact."
        ),
        "claim_boundary": [
            "ready=1 sidecar, not LFSR Fixed2S/RQTB2S fair package.",
            "Does not replace RQTB 1.1865x. It is an orthogonal empty-row cascade.",
            "Not energy or ASIC PPA.",
        ],
        "empty_row_list": [
            {key: int(value) for key, value in match.groupdict().items()}
            for match in SKIP_RE.finditer(text)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion exact empty-K row skip",
        "",
        "> 证据：`[rtl]+[occupancy-sideband]`。",
        "",
        f"- 空行 {empty} / 138，密行 {dense}",
        f"- 无跳过顺序 {baseline} → 跳过 {total}，**{baseline/total:.4f}x**（−{100*(1-total/baseline):.1f}%）",
        "- 空行不再付 226 拍 pair scan；K occupancy 是写 K 时留下的 1-bit 边带。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS empty-row skip {baseline}->{total} {baseline/total:.4f}x empty={empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
