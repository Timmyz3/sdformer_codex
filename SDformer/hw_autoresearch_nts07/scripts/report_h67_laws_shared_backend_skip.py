#!/usr/bin/env python3
"""Seal Motion shared-backend + empty-row skip ready=1 wall."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"SB_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+)(?: skip=(?P<skip>\d+))?"
)
SKIP_RE = re.compile(r"SB_SKIP row=(\d+)")
SEQ = 85912
PREV_SHARED = 65695


def area_cells(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("design", data).get("num_cells", data.get("num_cells", 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    log = args.result_dir / "shared_backend_skip_iverilog.log"
    if not log.is_file():
        log = args.result_dir / "shared_backend_iverilog.log"
    text = log.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_shared_backend_2s" not in text:
        raise ValueError("shared-backend+skip log missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing SB_SUM")
    wall = int(summary["wall"])
    skip = int(summary["skip"] or len(SKIP_RE.findall(text)))
    yosys_shared = args.result_dir / "yosys_shared_stat.json"
    yosys_single = args.result_dir / "yosys_single_stat.json"
    shared_cells = area_cells(yosys_shared) if yosys_shared.is_file() else None
    single_cells = area_cells(yosys_single) if yosys_single.is_file() else None
    speedup = SEQ / wall
    vs_prev = PREV_SHARED / wall
    report = {
        "schema": "h67_laws_shared_backend_skip_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[shared-encoder-shiftmax]+[occupancy-skip]",
        "rows": int(summary["rows"]),
        "skip_rows": skip,
        "wall_cycles": wall,
        "sequential_ready1_cycles": SEQ,
        "previous_shared_noskip_cycles": PREV_SHARED,
        "speedup_vs_sequential": speedup,
        "speedup_vs_shared_noskip": vs_prev,
        "claim_boundary": [
            "ready=1 sidecar, not LFSR Fixed2S 1.1865x package.",
            "Does not replace the 1.1865x fair-package anchor.",
            "Yosys generic cells are structure proxy, not ASIC area.",
        ],
    }
    if shared_cells is not None and single_cells:
        area_ratio = shared_cells / single_cells
        report["yosys_generic_cells"] = {
            "shared": shared_cells,
            "single_2s": single_cells,
            "area_ratio": area_ratio,
        }
        report["area_normalized_throughput"] = speedup / area_ratio
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion shared-backend + empty-row skip",
        "",
        f"- wall **{wall}** vs sequential {SEQ} = **{speedup:.4f}x**",
        f"- vs shared without skip {PREV_SHARED} = **{vs_prev:.4f}x**",
        f"- empty rows skipped: **{skip}** / {summary['rows']}",
        "- occupancy skip retires in start order; empty rows do not scan 225 pairs",
        "",
    ]
    if "yosys_generic_cells" in report:
        cells = report["yosys_generic_cells"]
        md.append(
            f"- Yosys cells shared {cells['shared']} / single {cells['single_2s']} "
            f"(×{cells['area_ratio']:.3f})"
        )
        md.append(
            f"- area-normalized throughput **{report['area_normalized_throughput']:.3f}** "
            "(generic cells only)"
        )
    (args.result_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"PASS shared+skip {speedup:.4f}x vs seq, {vs_prev:.4f}x vs noskip skip={skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
