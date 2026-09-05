#!/usr/bin/env python3
"""Count actual needed bank reads, including a mask-aware ordinary baseline.

This is a traffic experiment, not cycle/PPA estimation. Four context masks
already exist in the frozen B4 descriptor bundle. Six output slices share the
same source-bank mask. A cache entry's missing banks are filled only on demand;
partial-valid metadata is modeled for the fair ordinary/cache baseline.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from m2241_c2_weight_lifetime_dse import chunks


def masked_reads(words, mode, capacity, initial=None):
    cache = {} if initial is None else dict(initial)
    age = max((v[1] for v in cache.values()), default=0)
    reads, lookup, missing_halves = 0, 0, 0
    if mode == "ordinary":
        rows = [(g, words[c][g] & 65535) for c in range(4) for g in range(48)]
    else:
        rows = [(g, (words[0][g] | words[1][g] | words[2][g] | words[3][g]) & 65535)
                for g in range(48)]
    for group, needed in rows:
        if not needed:
            continue
        lookup += 1
        age += 1
        valid = cache.get(group, (0, 0))[0]
        missing = needed & ~valid
        reads += 6 * missing.bit_count()
        missing_halves += 6 * (bool(missing & 255) + bool(missing >> 8))
        if capacity:
            if group not in cache and len(cache) == capacity:
                del cache[min(cache, key=lambda g: cache[g][1])]
            cache[group] = (valid | needed, age)
    return dict(bank_reads=reads, nonempty_refill_beats=missing_halves,
                row_lookups=lookup), cache


def run():
    totals = {axis: Counter() for axis in (
        "ordinary_mask_lru4", "ordinary_mask_lru1", "ordinary_mask_nocache",
        "tsbg_fullbank_nocache", "tsbg_union_lru4", "tsbg_union_nocache")}
    layers = {}
    count = 0
    for _, row, words in chunks():
        count += 1
        axes = {}
        for capacity in (4, 1, 0):
            key = "ordinary_mask_" + (f"lru{capacity}" if capacity else "nocache")
            axes[key] = masked_reads(words, "ordinary", capacity)[0]
        axes["tsbg_union_nocache"] = masked_reads(words, "tsbg", 0)[0]
        axes["tsbg_union_lru4"] = masked_reads(words, "tsbg", 4)[0]
        assert axes["tsbg_union_nocache"] == axes["tsbg_union_lru4"]
        nonempty = sum(any(words[c][g] & 65535 for c in range(4)) for g in range(48))
        axes["tsbg_fullbank_nocache"] = dict(bank_reads=96 * nonempty,
            nonempty_refill_beats=12 * nonempty, row_lookups=nonempty)
        # Every source contributing to every private context remains covered.
        for g in range(48):
            union = (words[0][g] | words[1][g] | words[2][g] | words[3][g]) & 65535
            for c in range(4):
                assert ((words[c][g] & 65535) & ~union) == 0
        key = str(row.get("layer_id", row.get("target", "unknown")))
        layer = layers.setdefault(key, Counter())
        for axis, item in axes.items():
            totals[axis].update(item)
            layer[axis] += item["bank_reads"]
        layer["chunks"] += 1
    ordinary = totals["ordinary_mask_lru4"]["bank_reads"]
    old = totals["tsbg_fullbank_nocache"]["bank_reads"]
    new = totals["tsbg_union_nocache"]["bank_reads"]
    # Repeated identity can make a persistent cache preferable. Do not silently
    # assume all live operation is cold just because fixture chunks are cold.
    warm = []
    for groups in (1, 4, 8):
        words = [[(1 << ((g+c) % 16)) if g < groups else 0 for g in range(48)] for c in range(4)]
        cached, state = masked_reads(words, "tsbg", 4)
        hot, _ = masked_reads(words, "tsbg", 4, state)
        uncached, _ = masked_reads(words, "tsbg", 0)
        warm.append(dict(groups=groups, masked_lru4_two_bundles=cached["bank_reads"]+hot["bank_reads"],
                         borrowed_union_two_bundles=2*uncached["bank_reads"]))
    return dict(scope="4320 cold G48 chunks, all counted once; exact traffic only",
        chunks=count, axes={k: dict(v) for k,v in totals.items()},
        per_layer={k: dict(v) for k,v in layers.items()},
        reduction_vs_mask_aware_ordinary_lru4=1-new/ordinary,
        incremental_reduction_vs_fullbank_tsbg=1-new/old,
        zero_copy_incremental_read_reduction_vs_same_union_cache=0.0,
        warm_counterexamples=warm,
        decision="MEASURE_TIMING_AND_ENERGY_NEXT; no RTL/cycle claim from read counts",
        limitations=["Metadata OR/selection delay and partial-cache-valid physical cost not mapped",
            "No intra-transaction weight-value or product reuse across signed contexts",
            "Does not remove existing M803 response slot storage",
            "No full-token/output-tile scaling; no DRAM or joules conversion"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k,v in result.items() if k != "per_layer"}, indent=2))
