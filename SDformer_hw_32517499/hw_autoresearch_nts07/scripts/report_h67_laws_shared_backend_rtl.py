#!/usr/bin/env python3
"""Seal Motion true shared-backend RTL + Yosys area."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"SB_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+)(?: skip=(?P<skip>\d+))?"
)
SEQ = 85912
PREV_SHARED = 65695


def area_cells(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("design", data).get("num_cells", data.get("num_cells", 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    text = (args.result_dir / "shared_backend_iverilog.log").read_text(encoding="utf-8")
    if "PASS tb_h67_laws_shared_backend_2s" not in text:
        raise ValueError("shared-backend log missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing SB_SUM")
    wall = int(summary["wall"])
    shared_cells = area_cells(args.result_dir / "yosys_shared_stat.json")
    single_cells = area_cells(args.result_dir / "yosys_single_stat.json")
    speedup = SEQ / wall
    area_ratio = shared_cells / single_cells if single_cells else float("nan")
    ant = speedup / area_ratio if area_ratio else float("nan")
    report = {
        "schema": "h67_laws_shared_backend_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[shared-encoder-shiftmax]+[dual-directory-k]",
        "rows": int(summary["rows"]),
        "wall_cycles": wall,
        "sequential_ready1_cycles": SEQ,
        "speedup_vs_sequential": speedup,
        "yosys_generic_cells": {
            "shared": shared_cells,
            "single_2s": single_cells,
            "area_ratio": area_ratio,
        },
        "area_normalized_throughput": ant,
        "claim_boundary": [
            "ready=1 sidecar, not LFSR Fixed2S 1.1865x package.",
            "Yosys generic cells are structure proxy, not ASIC area.",
        ],
    }
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.result_dir / "report.md").write_text(
        "# Motion shared-backend RTL\n\n"
        f"- wall {wall} vs sequential {SEQ} = **{speedup:.4f}x**\n"
        f"- Yosys cells shared {shared_cells} / single {single_cells} "
        f"(×{area_ratio:.3f})\n"
        f"- area-normalized throughput **{ant:.3f}** (generic cells only)\n"
    )
    print(f"PASS shared-backend {speedup:.4f}x ANT={ant:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
