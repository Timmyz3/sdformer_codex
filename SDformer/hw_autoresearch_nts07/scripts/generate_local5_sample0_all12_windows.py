#!/usr/bin/env python3
"""Generate sample0 complete-window vectors for all 12 Local5 blocks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = [
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 0), (3, 1),
]


def main() -> int:
    for stage, block in BLOCKS:
        out = ROOT / "tb_qfit" / "vectors" / f"local5_s{stage}b{block}_window_proj_20260813"
        if (out / "manifest.json").is_file():
            print(f"KEEP {out.name}")
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_local5_window_score_projection.py"),
            "--output-dir",
            str(out),
            "--sample",
            "0",
            "--stage",
            str(stage),
            "--block",
            str(block),
        ]
        print("GEN", out.name)
        subprocess.check_call(cmd, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
