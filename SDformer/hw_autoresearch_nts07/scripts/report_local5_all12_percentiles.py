#!/usr/bin/env python3
"""DATE-facing mean/p50/p95/p99 over sample0 12-block window heads."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GROUP_RE = re.compile(r"GROUP backend=\d+ latency=\d+ group=\d+ cycles=(?P<c>\d+)")
BLOCKS = [
    "s0b0", "s0b1", "s1b0", "s1b1",
    "s2b0", "s2b1", "s2b2", "s2b3", "s2b4", "s2b5",
    "s3b0", "s3b1",
]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def load_cycles(path: Path) -> list[int]:
    return [int(m.group("c")) for m in GROUP_RE.finditer(path.read_text())]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else float("nan"),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    residual: list[float] = []
    qsilent: list[float] = []
    per_block = []
    for tag in BLOCKS:
        r = load_cycles(args.result_dir / f"{tag}_residual.log")
        q = load_cycles(args.result_dir / f"{tag}_qsilent.log")
        residual.extend(r)
        qsilent.extend(q)
        per_block.append(
            {
                "tag": tag,
                "residual": stats([float(x) for x in r]),
                "qsilent": stats([float(x) for x in q]),
                "speedup_sum": sum(r) / sum(q) if q else float("nan"),
            }
        )
    ratios = [a / b for a, b in zip(residual, qsilent)]
    report = {
        "schema": "local5_all12_head_percentiles_v1",
        "heads": len(residual),
        "residual": stats(residual),
        "qsilent": stats(qsilent),
        "per_head_speedup": stats(ratios),
        "sum_speedup": sum(residual) / sum(qsilent),
        "blocks": per_block,
        "claim_boundary": [
            "138 heads = sample0 one window per 12 blocks, not 21600.",
            "p95 is over heads, not over samples.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Local5 sample0 12-block head percentiles",
        "",
        f"- heads {len(residual)}",
        f"- residual mean/p50/p95/p99 {report['residual']['mean']:.0f}/"
        f"{report['residual']['p50']:.0f}/{report['residual']['p95']:.0f}/"
        f"{report['residual']['p99']:.0f}",
        f"- Q-silent mean/p50/p95/p99 {report['qsilent']['mean']:.0f}/"
        f"{report['qsilent']['p50']:.0f}/{report['qsilent']['p95']:.0f}/"
        f"{report['qsilent']['p99']:.0f}",
        f"- per-head speedup mean/p50/p95 {report['per_head_speedup']['mean']:.3f}/"
        f"{report['per_head_speedup']['p50']:.3f}/{report['per_head_speedup']['p95']:.3f}",
        f"- sum speedup **{report['sum_speedup']:.4f}×**",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md))
    print(f"PASS percentiles heads={len(residual)} sum={report['sum_speedup']:.4f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
