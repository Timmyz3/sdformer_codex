#!/usr/bin/env python3
"""Seal Fixed2S / RQTB2S / Shared+Skip LFSR fair package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"FAIR_SUM rows=(?P<rows>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+)"
)
SEALED_FIXED = 112589
SEALED_RQTB = 94891


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    text = (args.result_dir / "fair_lfsr_threeway_iverilog.log").read_text(
        encoding="utf-8"
    )
    if "PASS tb_h67_laws_fair_lfsr_threeway_2s" not in text:
        raise ValueError("LFSR three-way log missing PASS")
    summary = SUM_RE.search(text)
    if summary is None:
        raise ValueError("missing FAIR_SUM")
    fixed = int(summary["fixed"])
    rqtb = int(summary["rqtb"])
    shared = int(summary["shared"])
    report = {
        "schema": "h67_laws_fair_lfsr_threeway_v1",
        "status": "PASS",
        "evidence": "[rtl]+[lfsr-fair-package]",
        "lfsr": {
            "seed": "16'h1d3f",
            "poly": "x^16+x^14+x^13+x^11+1 (bits 15,13,12,10)",
            "descriptor_issue_enable": "lfsr[0]|lfsr[5]",
            "out_ready": "lfsr[2]|lfsr[9]",
        },
        "rows": int(summary["rows"]),
        "skip_rows": int(summary["skip"]),
        "cycles": {"fixed2s": fixed, "rqtb2s": rqtb, "shared_skip": shared},
        "speedup_rqtb_vs_fixed": fixed / rqtb,
        "speedup_shared_vs_fixed": fixed / shared,
        "speedup_shared_vs_rqtb": rqtb / shared,
        "sealed_anchor": {
            "fixed2s": SEALED_FIXED,
            "rqtb2s": SEALED_RQTB,
            "speedup": SEALED_FIXED / SEALED_RQTB,
            "note": "Do not replace this 1.1865x number unless SHA-identical rerun drifts.",
        },
        "claim_boundary": [
            "Shared+Skip is sequential-row, same LFSR as the sealed 2S TB.",
            "Shared+Skip must not silently replace the Fixed2S→RQTB2S 1.1865x anchor.",
            "ready=1 shared wall is a different column.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion LFSR three-way fair package",
        "",
        f"- Fixed2S **{fixed}**",
        f"- RQTB2S **{rqtb}** = **{fixed / rqtb:.4f}x** vs Fixed",
        f"- Shared+Skip **{shared}** = **{fixed / shared:.4f}x** vs Fixed, "
        f"**{rqtb / shared:.4f}x** vs RQTB",
        f"- empty-row skips: {summary['skip']}",
        f"- sealed 1.1865x anchor remains {SEALED_FIXED}→{SEALED_RQTB}",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS fair LFSR fixed={fixed} rqtb={rqtb} shared={shared} "
        f"{fixed / rqtb:.4f}x/{fixed / shared:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
