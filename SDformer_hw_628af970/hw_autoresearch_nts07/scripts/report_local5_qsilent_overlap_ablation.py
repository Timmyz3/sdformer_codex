#!/usr/bin/env python3
"""Report the fail-closed Local5 Q-silent/ident-K/overlap RTL ablation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<rows>\d+) score_service=(?P<service>\d+) "
    r"score_direct_rows=(?P<direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+) memory_wait=(?P<memory_wait>\d+) "
    r"terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(
    r"PASS Local5 score-to-projection backend=(?P<backend>\d+) "
    r"latency=(?P<latency>\d+) groups=(?P<groups>\d+) "
    r"total_cycles=(?P<cycles>\d+)"
)

CONFIGS = {
    "residual": {"qsilent": 0, "identk": 0, "overlap": 0},
    "q0_serial": {"qsilent": 1, "identk": 0, "overlap": 0},
    "q0_overlap": {"qsilent": 1, "identk": 0, "overlap": 1},
    "q0_ident_serial": {"qsilent": 1, "identk": 1, "overlap": 0},
    "q0_ident_overlap": {"qsilent": 1, "identk": 1, "overlap": 1},
}
SEALED_ANCHORS = {
    "residual": 324605,
    "q0_serial": 191424,
    "q0_ident_serial": 184632,
}


def parse_log(path: Path, expected_groups: int = 100) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    rows = [{key: int(match.group(key)) for key in match.groupdict()}
            for match in GROUP_RE.finditer(text)]
    summary_matches = list(PASS_RE.finditer(text))
    if len(summary_matches) != 1:
        raise ValueError(f"{path}: expected exactly one PASS summary")
    summary = {key: int(summary_matches[0].group(key))
               for key in summary_matches[0].groupdict()}
    if summary["backend"] != 0 or summary["latency"] != 1:
        raise ValueError(f"{path}: not TCFM5 latency-1")
    if summary["groups"] != expected_groups or len(rows) != expected_groups:
        raise ValueError(
            f"{path}: incomplete groups summary={summary['groups']} rows={len(rows)}"
        )
    ids = [row["group"] for row in rows]
    if ids != list(range(expected_groups)):
        raise ValueError(f"{path}: group IDs are not complete and ordered")
    if any(row["rows"] != 450 for row in rows):
        raise ValueError(f"{path}: score row count is not 450 in every group")
    summed_cycles = sum(row["cycles"] for row in rows)
    if summed_cycles != summary["cycles"]:
        raise ValueError(
            f"{path}: cycle sum {summed_cycles} != PASS {summary['cycles']}"
        )
    return {
        "total_cycles": summary["cycles"],
        "groups": expected_groups,
        "score_rows": sum(row["rows"] for row in rows),
        "score_service_cycles": sum(row["service"] for row in rows),
        "qsilent_rows": sum(row["qsilent"] for row in rows),
        "identk_rows": sum(row["identk"] for row in rows),
        "overlap_accepts": sum(row["overlap"] for row in rows),
        "terms": sum(row["terms"] for row in rows),
        "updates": sum(row["updates"] for row in rows),
    }


def build_report(result_dir: Path, vector_dir: Path) -> dict[str, object]:
    manifest = json.loads((vector_dir / "manifest.json").read_text(encoding="utf-8"))
    shape = manifest.get("shape")
    if not isinstance(shape, dict) or shape.get("out_dim") != 2:
        raise ValueError("vector manifest must explicitly bind OUT_DIM=2")

    records: dict[str, dict[str, object]] = {}
    for name, switches in CONFIGS.items():
        record = parse_log(result_dir / f"{name}.log")
        record["switches"] = switches
        records[name] = record

    for name, expected in SEALED_ANCHORS.items():
        actual = int(records[name]["total_cycles"])
        if actual != expected:
            raise ValueError(f"sealed anchor drift {name}: {actual} != {expected}")

    baseline_cycles = int(records["residual"]["total_cycles"])
    for record in records.values():
        cycles = int(record["total_cycles"])
        record["speedup_vs_residual"] = baseline_cycles / cycles
        record["cycle_reduction_vs_residual"] = 1.0 - cycles / baseline_cycles

    return {
        "schema": "local5_qsilent_overlap_ablation_v1",
        "status": "PASS",
        "evidence": "[rtl]",
        "scope": {
            "datapath": "Local5 score-to-TCFM5 projection",
            "groups": 100,
            "tokens_per_group": 450,
            "out_dim": 2,
            "backend": "TCFM5 latency-1",
            "weights": "real checkpoint theta-folded dyadic INT8 head slices",
        },
        "configurations": records,
        "sealed_anchor_checks": SEALED_ANCHORS,
        "claim_boundary": [
            "OUT_DIM=2 projection tile only; not a full encoder result.",
            "No 21600 head-window extrapolation is used.",
            "Overlap is an engineering ablation, not a standalone DATE contribution.",
            "The sealed 184632-cycle ident-K cascade is the overlap-disabled configuration.",
            "The 53084-cycle S3.B0 side-path result is not part of this table.",
        ],
    }


def write_report(report: dict[str, object], result_dir: Path) -> None:
    (result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    records = report["configurations"]
    lines = [
        "# Local5 Q-silent overlap ablation",
        "",
        "Evidence: `[rtl]`. Scope: 100 real-trace groups, TCFM5 latency-1, "
        "`OUT_DIM=2` projection tile.",
        "",
        "| config | Q-silent | ident-K | overlap | cycles | speedup vs residual | Q-silent rows | ident-K rows | overlap accepts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGS:
        row = records[name]
        switches = row["switches"]
        lines.append(
            f"| {name} | {switches['qsilent']} | {switches['identk']} | "
            f"{switches['overlap']} | {row['total_cycles']} | "
            f"{row['speedup_vs_residual']:.4f}x | {row['qsilent_rows']} | "
            f"{row['identk_rows']} | {row['overlap_accepts']} |"
        )
    lines.extend([
        "",
        "Boundaries: this is not a full encoder result, not a 21600-population "
        "extrapolation, and overlap is not claimed as a standalone contribution.",
    ])
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.result_dir, args.vector_dir)
    write_report(report, args.result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
