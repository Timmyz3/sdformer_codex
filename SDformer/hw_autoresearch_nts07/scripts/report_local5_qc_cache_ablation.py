#!/usr/bin/env python3
"""Fail-closed report for the Local5 source-stream cache isolation ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GROUP_RE = re.compile(r"(\w+)=(\d+)")
PASS_RE = re.compile(
    r"PASS Local5 score-to-projection .*?\bgroups=(\d+) total_cycles=(\d+)"
)
BAD_RE = re.compile(r"\b(?:ERROR|FATAL):|mismatch", re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(path: Path, expected_groups: int, require_cache: bool = True) -> dict:
    text = path.read_text()
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in {path}")
    rows = []
    pass_cycles = None
    pass_groups = None
    for line in text.splitlines():
        if line.startswith("GROUP "):
            fields = {key: int(value) for key, value in GROUP_RE.findall(line)}
            required = {
                "group", "cycles", "terms", "updates", "cache_hits",
                "cache_misses", "tag_compares", "lru_writes",
                "product_reads", "product_writes", "product_starts",
                "weight_reads", "memory_wait",
            }
            missing = sorted(required - fields.keys())
            if missing:
                raise ValueError(f"missing fields {missing} in {path}")
            rows.append(fields)
        match = PASS_RE.search(line)
        if match:
            pass_groups = int(match.group(1))
            pass_cycles = int(match.group(2))
    if len(rows) != expected_groups or pass_groups != expected_groups:
        raise ValueError(f"group count mismatch in {path}")
    if [row["group"] for row in rows] != list(range(expected_groups)):
        raise ValueError(f"group ordering mismatch in {path}")
    summed_cycles = sum(row["cycles"] for row in rows)
    if pass_cycles != summed_cycles:
        raise ValueError(f"cycle sum mismatch in {path}")
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "cycles", "terms", "updates", "cache_hits", "cache_misses",
            "tag_compares", "lru_writes", "product_reads",
            "product_writes", "product_starts", "weight_reads",
            "memory_wait",
        )
    }
    if require_cache:
        if totals["terms"] != totals["cache_hits"] + totals["cache_misses"]:
            raise ValueError(f"cache term conservation failed in {path}")
        if totals["cache_misses"] != totals["product_starts"]:
            raise ValueError(f"cache miss/start conservation failed in {path}")
    elif any(totals[key] for key in ("cache_hits", "cache_misses")):
        raise ValueError(f"uncached reference reports cache activity in {path}")
    return {"path": str(path), "sha256": sha256(path), "rows": rows, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-log", required=True, type=Path)
    parser.add_argument("--qc-log", required=True, type=Path)
    parser.add_argument("--direct-q-log", required=True, type=Path)
    parser.add_argument("--frozen-doc", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--groups", type=int, default=100)
    args = parser.parse_args()

    c = parse_log(args.c_log, args.groups)
    qc = parse_log(args.qc_log, args.groups)
    direct_q = parse_log(args.direct_q_log, args.groups, require_cache=False)
    expected = {
        "c_terms": 222_649,
        "qc_terms": 74_131,
        "updates": 222_649,
        "c_starts": 5_556,
        "qc_starts": 5_561,
        "direct_q_cycles": 170_269,
    }
    observed = {
        "c_terms": c["totals"]["terms"],
        "qc_terms": qc["totals"]["terms"],
        "updates": qc["totals"]["updates"],
        "c_starts": c["totals"]["product_starts"],
        "qc_starts": qc["totals"]["product_starts"],
        "direct_q_cycles": direct_q["totals"]["cycles"],
    }
    if observed != expected:
        raise ValueError(f"locked population mismatch: {observed} != {expected}")
    if c["totals"]["updates"] != qc["totals"]["updates"]:
        raise ValueError("C/QC update conservation mismatch")
    if direct_q["totals"]["terms"] != qc["totals"]["terms"]:
        raise ValueError("direct-Q/QC term population mismatch")

    ratios = [
        c_row["cycles"] / qc_row["cycles"]
        for c_row, qc_row in zip(c["rows"], qc["rows"])
    ]
    report = {
        "schema": "local5_qc_cache_isolation_v1",
        "evidence": "[rtl] Verilator --assert, OUT_DIM=2 tile, not encoder",
        "status": "ADMIT_AS_ISOLATION_RTL_ONLY_STRONG_MFEP_CACHE_UNRESOLVED",
        "claim_boundary": {
            "C": "source-ordered one-hot destination issue plus W4 cache",
            "QC": "source-owned equal-gate destination-mask issue plus W4 cache",
            "not_C": "destination-local MFEP plus W4 cache (174289 issues)",
            "reason": "production inverse-stencil interface retains source K_self, while destination MFEP requires five candidate K vectors",
        },
        "locked_totals": {
            "C": c["totals"],
            "QC": qc["totals"],
            "Q_without_cache": direct_q["totals"],
        },
        "comparisons": {
            "C_to_QC_cycle_speedup": c["totals"]["cycles"] / qc["totals"]["cycles"],
            "QC_cycle_reduction_vs_C": 1.0 - qc["totals"]["cycles"] / c["totals"]["cycles"],
            "QC_vs_Q_without_cache_speedup": direct_q["totals"]["cycles"] / qc["totals"]["cycles"],
            "QC_groups_faster": sum(r > 1.0 for r in ratios),
            "QC_groups_tied": sum(r == 1.0 for r in ratios),
            "QC_groups_slower": sum(r < 1.0 for r in ratios),
            "min_group_speedup": min(ratios),
            "max_group_speedup": max(ratios),
        },
        "negative_result": "W4 removes almost all repeated products but does not remove ready-valid issue traffic; on QC it adds 217 cycles versus the direct Q path.",
        "admission": {
            "acc32": "checked per group by the self-checking RTL TB; zero mismatch",
            "protocol": "Verilator assertions enabled; no ERROR/FATAL/mismatch marker",
            "date_main_table": False,
            "innovation_score_change": 0.0,
        },
        "artifacts": {
            "C": {key: c[key] for key in ("path", "sha256")},
            "QC": {key: qc[key] for key in ("path", "sha256")},
            "Q_without_cache": {key: direct_q[key] for key in ("path", "sha256")},
            "frozen_doc_sha256": sha256(args.frozen_doc),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
