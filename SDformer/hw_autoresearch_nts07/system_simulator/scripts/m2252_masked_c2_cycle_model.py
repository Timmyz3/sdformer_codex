#!/usr/bin/env python3
"""M2249's serial bank-fill FSM, calibrated to actual mask-aware VCS pilots.

Scope is independently reset G48 chunks, counted once; not complete FC latency.
No fitted latency or scale factor: readiness is the M2160 memory equation,
with its reset now deasserted on the same negedge as the TB clock counter.
"""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from m2241_c2_weight_lifetime_dse import chunks, issue, commit
from m2244_consumer_union_bank_reads import masked_reads


@lru_cache(None)
def response_delay(phase, mask, uniform_latency=None, always_ready=False):
    # Request is accepted in cycle 'phase', not phase+1. Accepted bank b
    # loads delay=8-b; the response is consumed 9-b cycles later. A blocked
    # ready phase adds one cycle. M803 assembles the latest required response.
    return max(int(not always_ready and (phase+b*2) % 7 == 0) + (9-b if uniform_latency is None else uniform_latency)
               for b in range(8) if mask >> b & 1)


def run_chunk(words, mode, start=384, prefetch_union=True, memory_latency=None, always_ready=False):
    cycle, age, reads, refills, partial = start, 0, 0, 0, 0
    cache = {}
    order = ((c,g) for c in range(4) for g in range(48)) if mode == 0 else (
        (c,g) for g in range(48) for c in range(4))
    for c,g in order:
        active = words[c][g] & 65535
        if not active:
            continue
        cycle += 1  # ST_FIND.
        age += 1
        needed = active if mode == 0 or not prefetch_union else (
            words[0][g] | words[1][g] | words[2][g] | words[3][g]) & 65535
        valid = cache.get(g, (0,0))[0]
        missing = needed & ~valid
        partial += int(bool(valid and missing))
        if g not in cache and len(cache) == 4:
            del cache[min(cache, key=lambda key: cache[key][1])]
        cache[g] = (valid | needed, age)
        for half in (missing & 255, missing >> 8):
            if half:
                refills += 6
                reads += 6 * half.bit_count()
                for _ in range(6):
                    cycle += response_delay(cycle % 7, half, memory_latency, always_ready) + 1
        cycle = issue(cycle, 6*(bool(active & 255)+bool(active >> 8)))
    return dict(cycles=commit(cycle)-start, bank_reads=reads,
                refill_beats=refills, partial_refills=partial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcs-result", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    vcs = json.loads(args.vcs_result.read_text())
    if vcs["status"] != "PASS":
        raise ValueError("Completed VCS pilots required")
    anchors = {(r["axis"],r["slot"]):r for r in vcs["rows"]}
    totals = {a:Counter() for a in ("ordinary","tsbg")}
    layers, checks, count = {}, 0, 0
    for prefix, row, words in chunks():
        count += 1
        layer = layers.setdefault(row["layer_id"], {a:Counter() for a in totals})
        for mode,axis in enumerate(totals):
            point = run_chunk(words,mode)
            traffic = masked_reads(words,axis,4)[0]["bank_reads"]
            if point["bank_reads"] != traffic:
                raise ValueError("Independent read model mismatch")
            anchor = anchors.get((axis,row["slot"])) if prefix.startswith("m2051") else None
            if anchor:
                for metric in ("cycles","bank_reads"):
                    if point[metric] != anchor[metric]:
                        raise ValueError(f"VCS mismatch {axis}/{row['slot']}/{metric}")
                checks += 1
            totals[axis].update(point)
            layer[axis].update(point)
    if checks != 6:
        raise ValueError("Pilot calibration missing")
    result = dict(scope="4320 independently reset G48 chunks, counted once; CPU model, not full-layer or system latency",
        chunks=count, exact_vcs_pilot_axis_matches=checks, totals=totals,
        modeled_cycle_ratio=totals["ordinary"]["cycles"]/totals["tsbg"]["cycles"],
        modeled_bank_read_reduction=1-totals["tsbg"]["bank_reads"]/totals["ordinary"]["bank_reads"],
        per_layer=layers, area_or_power_measured=False,
        limitations=["Six pilots calibrate the FSM but do not validate all modeled chunks in RTL",
            "Fixed directed SRAM readiness/latency and bridge/commit backpressure",
            "No output-tile multiplicity, continuation overhead or DRAM model"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
