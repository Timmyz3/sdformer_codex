#!/usr/bin/env python3
"""Seal Motion ep35 Acc32 + ep30 held-out shared+skip walls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUM_RE = re.compile(
    r"SB_SUM rows=(?P<rows>\d+) wall=(?P<wall>\d+) skip=(?P<skip>\d+)"
    r"(?: tokens=(?P<tokens>\d+) pairs=(?P<pairs>\d+) empty_pairs=(?P<empty>\d+)"
    r" sequential=(?P<seq>\d+))?"
)


def parse(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    if "PASS tb_h67_laws_shared_backend_2s" not in text:
        raise ValueError(f"missing PASS {path}")
    match = SUM_RE.search(text)
    if match is None:
        raise ValueError(f"missing SB_SUM {path}")
    return {key: int(match.group(key) or 0) for key in match.groupdict()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    ep35_ov = parse(args.result_dir / "ep35_overlap_acc32.log")
    ep35_seq = parse(args.result_dir / "ep35_sequential_acc32.log")
    ep30_ov = parse(args.result_dir / "ep30_overlap.log")
    ep30_seq = parse(args.result_dir / "ep30_sequential.log")
    report = {
        "schema": "h67_date_paper_stack_v1",
        "status": "PASS",
        "evidence": "[rtl]+[ep35-acc32]+[ep30-heldout]",
        "ep35": {
            "overlap": ep35_ov,
            "sequential": ep35_seq,
            "speedup": ep35_seq["wall"] / ep35_ov["wall"],
            "empty_pair_frac": ep35_ov["empty"] / ep35_ov["pairs"] if ep35_ov["pairs"] else 0,
        },
        "ep30_heldout": {
            "overlap": ep30_ov,
            "sequential": ep30_seq,
            "speedup": ep30_seq["wall"] / ep30_ov["wall"],
            "empty_pair_frac": ep30_ov["empty"] / ep30_ov["pairs"] if ep30_ov["pairs"] else 0,
        },
        "claim_boundary": [
            "ep30 is a held-out checkpoint, not a replacement for ep35 1.1865x.",
            "Acc32 uses the sealed 2S lane_weight contract, not full projection.",
            "empty_pairs is activity, not an energy number.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Motion DATE paper stack",
        "",
        f"- ep35 overlap {ep35_ov['wall']} / sequential {ep35_seq['wall']} = "
        f"**{ep35_seq['wall'] / ep35_ov['wall']:.4f}×**, Acc32+gate 0 mismatch, "
        f"skip {ep35_ov['skip']}, empty pairs {ep35_ov['empty']}/{ep35_ov['pairs']}",
        f"- ep30 held-out overlap {ep30_ov['wall']} / sequential {ep30_seq['wall']} = "
        f"**{ep30_seq['wall'] / ep30_ov['wall']:.4f}×**, skip {ep30_ov['skip']}",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"PASS motion stack ep35={ep35_seq['wall']/ep35_ov['wall']:.4f}x "
        f"ep30={ep30_seq['wall']/ep30_ov['wall']:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
