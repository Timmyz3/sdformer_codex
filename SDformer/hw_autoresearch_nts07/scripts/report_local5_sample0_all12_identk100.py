#!/usr/bin/env python3
"""Seal sample0 12-block window table + 100-group ident-K slice."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PASS_RE = re.compile(
    r"PASS Local5 score-to-projection backend=(?P<backend>\d+) "
    r"latency=(?P<latency>\d+) groups=(?P<groups>\d+) total_cycles=(?P<cycles>\d+)"
)
GROUP_RE = re.compile(
    r"GROUP backend=\d+ latency=\d+ group=\d+ cycles=(?P<cycles>\d+) "
    r"score_rows=\d+ score_service=(?P<service>\d+) "
    r"score_direct_rows=\d+"
    r"(?: qsilent_rows=(?P<qsilent>\d+) identk_rows=(?P<identk>\d+))?"
)
BLOCKS = [
    ("s0b0", 0, 0, 3),
    ("s0b1", 0, 1, 3),
    ("s1b0", 1, 0, 6),
    ("s1b1", 1, 1, 6),
    ("s2b0", 2, 0, 12),
    ("s2b1", 2, 1, 12),
    ("s2b2", 2, 2, 12),
    ("s2b3", 2, 3, 12),
    ("s2b4", 2, 4, 12),
    ("s2b5", 2, 5, 12),
    ("s3b0", 3, 0, 24),
    ("s3b1", 3, 1, 24),
]
RESIDUAL100 = 324605


def parse(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    summary = PASS_RE.search(text)
    if summary is None:
        raise ValueError(f"missing PASS {path}")
    qsilent = identk = service = 0
    for match in GROUP_RE.finditer(text):
        qsilent += int(match.group("qsilent") or 0)
        identk += int(match.group("identk") or 0)
        service += int(match.group("service") or 0)
    return {
        "groups": int(summary["groups"]),
        "cycles": int(summary["cycles"]),
        "backend": int(summary["backend"]),
        "qsilent_rows": qsilent,
        "identk_rows": identk,
        "service": service,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for tag, stage, block, heads in BLOCKS:
        residual = parse(args.result_dir / f"{tag}_residual.log")
        qsilent = parse(args.result_dir / f"{tag}_qsilent.log")
        rows.append(
            {
                "tag": tag,
                "stage": stage,
                "block": block,
                "heads": heads,
                "residual": residual["cycles"],
                "qsilent": qsilent["cycles"],
                "speedup": residual["cycles"] / qsilent["cycles"],
                "qsilent_rows": qsilent["qsilent_rows"],
                "identk_rows": qsilent["identk_rows"],
            }
        )
    identk100 = parse(args.result_dir / "identk100_tcfm5_l1.log")
    lin_r = parse(args.result_dir / "s3b0_linear5_residual.log")
    lin_q = parse(args.result_dir / "s3b0_linear5_qsilent.log")
    leftover = {}
    leftover_path = args.result_dir / "leftover_all12" / "report.json"
    if leftover_path.is_file():
        leftover = json.loads(leftover_path.read_text())
    report = {
        "schema": "local5_sample0_all12_identk100_v1",
        "status": "PASS",
        "evidence": "[rtl]+[sample0-all12-complete-windows]+[100-group-identk]",
        "blocks": rows,
        "sample0_total_residual": sum(r["residual"] for r in rows),
        "sample0_total_qsilent": sum(r["qsilent"] for r in rows),
        "sample0_speedup": sum(r["residual"] for r in rows) / sum(r["qsilent"] for r in rows),
        "identk100_tcfm5_l1": {
            "cycles": identk100["cycles"],
            "residual_sealed": RESIDUAL100,
            "speedup_vs_residual": RESIDUAL100 / identk100["cycles"],
            "qsilent_rows": identk100["qsilent_rows"],
            "identk_rows": identk100["identk_rows"],
            "note": "Does not replace sealed 1.6957x Q-silent-only package.",
        },
        "s3b0_linear5": {
            "residual": lin_r["cycles"],
            "qsilent": lin_q["cycles"],
            "speedup": lin_r["cycles"] / lin_q["cycles"],
        },
        "leftover": leftover.get("decision"),
        "claim_boundary": [
            "sample0 one window per block, not 100 samples.",
            "100-group ident-K is a new slice, not a refresh of 1.6957x.",
            "Not full encoder.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Local5 sample0 all-12 + ident-K 100-group",
        "",
        "| block | heads | residual | Q-silent | speedup | qsilent | identk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| S{row['stage']}.B{row['block']} | {row['heads']} | {row['residual']} | "
            f"{row['qsilent']} | **{row['speedup']:.4f}×** | {row['qsilent_rows']} | "
            f"{row['identk_rows']} |"
        )
    md += [
        "",
        f"- sample0 十二块合计 {report['sample0_total_residual']} → "
        f"{report['sample0_total_qsilent']} = **{report['sample0_speedup']:.4f}×**",
        f"- 100-group ident-K TCFM5 L1 {identk100['cycles']} vs residual {RESIDUAL100} = "
        f"**{RESIDUAL100 / identk100['cycles']:.4f}×**（不覆盖 1.6957×）",
        f"- S3 Linear5 {lin_r['cycles']} → {lin_q['cycles']} = "
        f"**{lin_r['cycles'] / lin_q['cycles']:.4f}×**",
        f"- leftover: {leftover.get('decision', 'n/a')}",
        "",
    ]
    (args.result_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS all12 {report['sample0_speedup']:.4f}x identk100={RESIDUAL100 / identk100['cycles']:.4f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
