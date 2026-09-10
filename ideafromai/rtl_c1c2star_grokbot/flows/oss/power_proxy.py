#!/usr/bin/env python3
"""GROKBOT NEW FILE -- iscas_ssh
Rough dynamic-energy PROXY: VCD toggle count × gate (cell) count.

*** NOT silicon power. *** No liberty, no capacitance, no voltage.
Label every printed number as PROXY / heuristic until .lib + OpenSTA/OpenROAD exist.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_stat_cells(stat_path: Path) -> int:
    text = stat_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Number of cells:\s+(\d+)", text)
    if not m:
        raise SystemExit(f"no cell count in {stat_path}")
    return int(m.group(1))


def count_vcd_toggles(vcd_path: Path) -> tuple[int, int]:
    """Return (toggle_events, value_change_lines) from a simple VCD.
    Counts lines that look like value changes (0/1/b... + id), not definitions.
    """
    toggles = 0
    changes = 0
    if not vcd_path.is_file():
        return 0, 0
    in_dump = False
    with vcd_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("$dumpvars") or line.startswith("#"):
                in_dump = True
            if not in_dump:
                continue
            if not line or line.startswith("$"):
                continue
            if line.startswith("#"):
                continue
            # scalar: 0!<id> or 1!<id> ; vector: b0101 <id>
            if line[0] in "01xXzZ" or line[0] == "b" or line[0] == "r":
                changes += 1
                # treat every recorded change as one toggle event for proxy
                toggles += 1
    return toggles, changes


def main() -> int:
    ap = argparse.ArgumentParser(description="VCD×gate dynamic energy PROXY (NOT silicon)")
    ap.add_argument("--vcd", required=True, type=Path)
    ap.add_argument("--stat", required=True, type=Path, help="yosys tee -o stat file")
    ap.add_argument("--label", default="module")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="arbitrary scale; PROXY = alpha * toggles * cells")
    args = ap.parse_args()

    cells = parse_stat_cells(args.stat)
    toggles, changes = count_vcd_toggles(args.vcd)
    proxy = args.alpha * float(toggles) * float(cells)

    print("=" * 60)
    print("POWER PROXY — NOT SILICON POWER / NOT PRIMETIME")
    print("No liberty (.lib), no C_load, no Vdd — heuristic only.")
    print("=" * 60)
    print(f"label:          {args.label}")
    print(f"vcd:            {args.vcd}")
    print(f"stat:           {args.stat}")
    print(f"yosys cells:    {cells}")
    print(f"vcd changes:    {changes}")
    print(f"toggle events:  {toggles}")
    print(f"alpha:          {args.alpha}")
    print(f"PROXY score:    {proxy:.3g}  (= alpha * toggles * cells)")
    print("Interpret as relative activity×area score across ablations only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
