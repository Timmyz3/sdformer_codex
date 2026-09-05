#!/usr/bin/env python3
"""Separate group-major locality from consumer-union transaction coalescing."""
import argparse
from collections import Counter
import json
from pathlib import Path

from m2252_masked_c2_cycle_model import chunks, run_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vcs-third-axis", type=Path)
    args = ap.parse_args()
    totals = {name: Counter() for name in ("ordinary_demand", "group_demand", "group_union")}
    tails = Counter()
    worst = None
    count = 0
    anchors = {}
    checks = 0
    if args.vcs_third_axis:
        vcs = json.loads(args.vcs_third_axis.read_text())
        if vcs["status"] != "PASS" or vcs["consumer_union_enabled"]:
            raise ValueError("Need completed group-demand VCS axis")
        anchors = {point["slot"]: point for point in vcs["rows"]}
    sensitivity = {latency: {axis: Counter() for axis in totals} for latency in (1, 2, 4, 8)}
    for prefix, row, words in chunks():
        count += 1
        axes = {"ordinary_demand": run_chunk(words, 0),
                "group_demand": run_chunk(words, 1, prefetch_union=False),
                "group_union": run_chunk(words, 1)}
        demand, union = axes["group_demand"], axes["group_union"]
        anchor = anchors.get(row["slot"]) if prefix.startswith("m2051") else None
        if anchor:
            if any(demand[key] != anchor[key] for key in ("cycles", "bank_reads")):
                raise ValueError("Group-demand RTL/model mismatch")
            checks += 1
        if demand["bank_reads"] != union["bank_reads"]:
            raise ValueError("Unexpected read-count change; inspect prefetch waste")
        ratio = demand["cycles"] / union["cycles"]
        tails["faster" if ratio > 1 else "tie" if ratio == 1 else "slower"] += 1
        if worst is None or ratio < worst["ratio"]:
            worst = dict(fixture=prefix, slot=row["slot"], ratio=ratio,
                         demand_cycles=demand["cycles"], union_cycles=union["cycles"])
        for axis, point in axes.items():
            totals[axis].update(point)
        for latency, points in sensitivity.items():
            for axis in points:
                point = run_chunk(words, int(axis != "ordinary_demand"),
                    prefetch_union=axis == "group_union", memory_latency=latency)
                points[axis].update(point)
    demand, union = totals["group_demand"], totals["group_union"]
    if anchors and checks != 3:
        raise ValueError("Missing causal RTL pilot")
    result = dict(scope="CPU causal ablation, 4320 cold G48 chunks; same 4-row partial-valid cache and ports",
        chunks=count, totals=totals, union_vs_group_demand_cycle_ratio=demand["cycles"]/union["cycles"],
        union_extra_bank_read_reduction=0,
        union_refill_transaction_reduction=1-union["refill_beats"]/demand["refill_beats"],
        per_chunk_cycles=tails, worst=worst,
        explanation="Group-major order saves repeated bank reads; union co-fills consumer needs to reduce refill transactions/waits",
        rtl_status="Six ordinary/union VCS anchors; group-demand has three additional VCS anchors" if checks else
            "Ordinary-demand and group-union have six VCS pilot anchors; group-demand is model-only",
        group_demand_vcs_cycle_and_read_matches=checks,
        uniform_memory_latency_sensitivity=[dict(latency_cycles=latency,
            union_vs_group_demand=points["group_demand"]["cycles"] / points["group_union"]["cycles"],
            union_vs_ordinary=points["ordinary_demand"]["cycles"] / points["group_union"]["cycles"],
            cycles={axis: point["cycles"] for axis, point in points.items()})
            for latency, points in sensitivity.items()],
        sensitivity_scope="Uniform response latency replaces bank-specific 9-b; same independent readiness and backpressure. Not separately RTL-calibrated.",
        power_or_area_result=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
