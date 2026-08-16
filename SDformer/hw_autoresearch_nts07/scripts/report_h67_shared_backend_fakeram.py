#!/usr/bin/env python3
"""Seal fakeram-K shared backend + blackbox logic ANT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"SB_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+)(?: skip=(?P<skip>\d+))?"
)
FLOP_WALL = 58288
SEQ = 85912


def area(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    design = data.get("design", data)
    cells = design.get("num_cells", 0)
    types = design.get("num_cells_by_type", design.get("cells", {}))
    macros = 0
    if isinstance(types, dict):
        for name, count in types.items():
            if "fakeram" in str(name).lower():
                macros += int(count)
    return int(cells), macros


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    text = (args.result_dir / "shared_backend_fakeram_iverilog.log").read_text()
    if "PASS tb_h67_laws_shared_backend_2s" not in text:
        raise ValueError("fakeram sim missing PASS")
    sva = (args.result_dir / "shared_backend_sva_verilator.log").read_text()
    if "PASS tb_h67_laws_shared_backend_2s" not in sva:
        raise ValueError("SVA sim missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing SB_SUM")
    wall = int(summary["wall"])
    shared_cells, shared_macros = area(args.result_dir / "yosys_shared_bb_stat.json")
    single_cells, single_macros = area(args.result_dir / "yosys_single_bb_stat.json")
    # blackbox instances are cells too; subtract macros for logic-only ratio
    shared_logic = shared_cells - shared_macros
    single_logic = single_cells - single_macros
    speedup = SEQ / wall
    logic_ratio = shared_logic / single_logic if single_logic else float("nan")
    ant = speedup / logic_ratio if logic_ratio else float("nan")
    report = {
        "schema": "h67_shared_backend_fakeram_v1",
        "status": "PASS",
        "evidence": "[rtl]+[fakeram45-functional]+[yosys-blackbox-logic]",
        "rows": int(summary["rows"]),
        "skip_rows": int(summary["skip"] or 0),
        "wall_cycles": wall,
        "flop_wall_cycles": FLOP_WALL,
        "wall_delta_vs_flop": wall - FLOP_WALL,
        "sequential_ready1_cycles": SEQ,
        "speedup_vs_sequential": speedup,
        "yosys_blackbox": {
            "shared_cells": shared_cells,
            "shared_macros": shared_macros,
            "shared_logic_cells": shared_logic,
            "single_cells": single_cells,
            "single_macros": single_macros,
            "single_logic_cells": single_logic,
            "logic_area_ratio": logic_ratio,
        },
        "logic_normalized_throughput": ant,
        "claim_boundary": [
            "fakeram45 is an open SRAM proxy, not a foundry compiler result.",
            "ANT divides by logic cells only; macros are counted separately.",
            "ready=1 sidecar, not the 1.1865x LFSR anchor.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.result_dir / "report.md").write_text(
        "# Motion shared-backend fakeram K\n\n"
        f"- wall **{wall}** vs flop {FLOP_WALL} (delta {wall - FLOP_WALL})\n"
        f"- vs sequential {SEQ} = **{speedup:.4f}x**\n"
        f"- macros shared {shared_macros} / single {single_macros}\n"
        f"- logic cells {shared_logic} / {single_logic} (×{logic_ratio:.3f})\n"
        f"- logic-normalized throughput **{ant:.3f}**\n"
        "- SVA skip/emit mutex + retire bound PASS\n",
        encoding="utf-8",
    )
    print(
        f"PASS fakeram wall={wall} macros={shared_macros}/{single_macros} "
        f"ANT_logic={ant:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
