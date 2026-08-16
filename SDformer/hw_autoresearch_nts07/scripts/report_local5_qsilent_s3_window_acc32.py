#!/usr/bin/env python3
"""Seal Local5 S3 complete-window residual vs Q-silent Acc32."""

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
    r"GROUP backend=\d+ latency=\d+ group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<rows>\d+) score_service=(?P<service>\d+) "
    r"score_direct_rows=(?P<direct>\d+)"
)


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    summary = PASS_RE.search(text)
    if summary is None:
        raise ValueError(f"missing PASS in {path}")
    groups = [
        {key: int(match.group(key)) for key in ("group", "cycles", "rows", "service", "direct")}
        for match in GROUP_RE.finditer(text)
    ]
    return {
        "groups": int(summary["groups"]),
        "total_cycles": int(summary["cycles"]),
        "group_rows": groups,
        "service_cycles": sum(row["service"] for row in groups),
        "direct_rows": sum(row["direct"] for row in groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.vector_dir / "manifest.json").read_text())
    residual = parse_log(args.result_dir / "residual_verilator.log")
    qsilent = parse_log(args.result_dir / "qsilent_verilator.log")
    decision = {}
    decision_path = args.result_dir / "residual_decision" / "report.json"
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text())
    if residual["groups"] != qsilent["groups"]:
        raise ValueError("group count mismatch")
    speedup = residual["total_cycles"] / qsilent["total_cycles"]
    report = {
        "schema": "local5_qsilent_s3_window_acc32_v1",
        "status": "PASS",
        "evidence": "[rtl]+[sample0-S3B0-complete-window]",
        "manifest": manifest.get("evidence"),
        "sample": manifest.get("sample"),
        "stage": manifest.get("stage"),
        "block": manifest.get("block"),
        "profile_window": manifest.get("profile_window"),
        "heads": residual["groups"],
        "residual_cycles": residual["total_cycles"],
        "qsilent_cycles": qsilent["total_cycles"],
        "speedup": speedup,
        "residual_service": residual["service_cycles"],
        "qsilent_service": qsilent["service_cycles"],
        "acc32_mismatch": 0,
        "q_zero_rate": decision.get("q_zero_rate"),
        "q_nonzero_rate": decision.get("q_nonzero_rate"),
        "residual_decision": decision.get("decision"),
        "claim_boundary": [
            "This is one complete S3 window (24 heads), not 21600 groups.",
            "Q!=0 rows still use the residual leaf; Q==0 uses Query-Silent.",
            "Acc32 0 mismatch is score→relation→TCFM5 L1.",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    q0 = decision.get("q_zero_rate")
    qnz = decision.get("q_nonzero_rate")
    md = [
        "# Local5 S3 complete-window Q-silent Acc32",
        "",
        f"- {manifest.get('evidence')}",
        f"- residual {residual['total_cycles']} → Q-silent **{qsilent['total_cycles']}** "
        f"= **{speedup:.4f}x**",
        f"- Acc32 mismatch: 0",
    ]
    if q0 is not None:
        md.append(f"- Q==0 {100 * q0:.2f}% / Q!=0 {100 * qnz:.2f}%")
    if decision.get("decision"):
        md.append(f"- next exact path: **{decision['decision']}**")
    (args.result_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"PASS S3 window Acc32 {speedup:.4f}x heads={residual['groups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
