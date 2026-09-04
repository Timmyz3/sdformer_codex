#!/usr/bin/env python3
"""Row-wise MotionXOR equal vs FAIR_ROW equal. Must match every row."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_h67_empty_active_equal import load_rows, motionxor_q7

ROW_RE = re.compile(
    r"FAIR_ROW row=(?P<row>\d+) active=(?P<active>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) "
    r"fslots=(?P<fslots>\d+) rslots=(?P<rslots>\d+) equal=(?P<equal>\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--fair-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.vectors)
    matches = list(ROW_RE.finditer(args.fair_log.read_text(encoding="utf-8")))
    if len(matches) != len(rows):
        raise SystemExit(f"row count {len(matches)} vs vectors {len(rows)}")
    row_ids = [int(match["row"]) for match in matches]
    if row_ids != list(range(len(rows))):
        raise SystemExit(
            "FAIR_ROW IDs must be unique, complete, and ordered from zero"
        )
    mismatches = []
    model_total = 0
    rtl_total = 0
    for qs_ks, match in zip(rows, matches):
        qs, ks = qs_ks
        model = 0
        for pair in range(225):
            s0 = motionxor_q7(qs[pair], ks[pair], ks[pair + 225])
            s1 = motionxor_q7(qs[pair + 225], ks[pair + 225], ks[pair])
            model += int(s0 == s1)
        rtl = int(match["equal"])
        model_total += model
        rtl_total += rtl
        if model != rtl:
            mismatches.append(
                {"row": int(match["row"]), "model": model, "rtl": rtl}
            )
    report = {
        "schema": "h67_fair_row_equal_miter_v1",
        "status": "PASS" if not mismatches else "FAIL",
        "rows": len(rows),
        "model_total": model_total,
        "rtl_total": rtl_total,
        "mismatches": mismatches,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        "# Motion 公平包逐行 equal miter\n\n"
        f"- rows {len(rows)}, model {model_total}, RTL {rtl_total}, "
        f"mismatch {len(mismatches)}\n"
        f"- **{report['status']}**\n"
    )
    print(
        f"{report['status']} row equal miter model={model_total} "
        f"rtl={rtl_total} mismatch={len(mismatches)}"
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
