#!/usr/bin/env python3
"""Seal Motion dual-workspace shared-backend RTL wall-time."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(r"LAWS_DW_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+)")
DONE_RE = re.compile(r"LAWS_DW_DONE row=(?P<row>\d+) eng=(?P<eng>\d+) outs=(?P<outs>\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--phase-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_dual_workspace_2s" not in text:
        raise ValueError("dual-workspace log missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing LAWS_DW_SUM")
    phase = json.loads(args.phase_report.read_text(encoding="utf-8"))
    wall = int(summary["wall"])
    seq = int(phase["sequential_cycles"])
    model = int(phase["shared_backend_per_block_cycles"])
    report = {
        "schema": "h67_laws_dual_workspace_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[exclusive-build-emit]+[2x-engine-area]",
        "rows": int(summary["rows"]),
        "completed_rows": len(DONE_RE.findall(text)),
        "wall_cycles": wall,
        "sequential_ready1_cycles": seq,
        "phase_model_per_block_cycles": model,
        "speedup_vs_sequential": seq / wall,
        "model_ratio": model / wall,
        "area_contract": "Two full RQTB2S engines. Schedule is shared-backend; area is not.",
        "claim_boundary": [
            "ready=1, not LFSR fair Fixed2S package.",
            "Do not call this shared-directory netlist or 1.1865x replacement.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion dual-workspace row-pipeline RTL",
        "",
        f"- 顺序 ready=1：{seq}",
        f"- 双实例 exclusive build/emit wall：{wall}，**{seq/wall:.4f}x**",
        f"- 相位模型按 block：{model}，模型/实测={model/wall:.3f}",
        "- 面积是两套完整 RQTB2S；调度合同才是共享 encoder/Shiftmax。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS dual-workspace wall={wall} speedup={seq/wall:.4f}x model={model/wall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
