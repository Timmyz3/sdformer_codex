#!/usr/bin/env python3
"""Small cycle/port experiment; no new sealing or approval machinery.

Baselines reproduce the M2018 FSM and frozen directed memory/ready schedule.
Candidates borrow M803 response slots, keeping each beat until all consumers
accept. Candidate timing is a proposed microarchitecture model, not RTL PPA.
Continuation chunks are counted once, without output-tile or wrapper scaling.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
FIX = HW / "tb_m2018/fixtures"
START = 383


@lru_cache(None)
def memory_delay(phase, extra_latency=0):
    # One request/response per bank; readiness and response skew match M2051.
    return max(int((phase + 1 + bank * 2) % 7 == 0) + 9 - bank
               for bank in range(8)) + extra_latency


def issue(cycle, count):
    for _ in range(count):
        cycle += int(cycle % 11 == 3) + 1
    return cycle


def commit(cycle):
    cycle += 1  # End-of-work selection, as in M2018.
    for _ in range(24):
        cycle += int(cycle % 13 == 5) + 1
    return cycle


def cache_run(words, mode, capacity, frozen_age=False, cache=None,
              start=START, extra_latency=0):
    cache = {} if cache is None else dict(cache)
    clock = max(cache.values(), default=0) + 1
    cycle, reads, updates = start, 0, 0
    order = ((c, g) for c in range(4) for g in range(48)) if mode == 0 else (
        (c, g) for g in range(48) for c in range(4))
    # Physical slots reproduce the original tie-to-low-index age policy.
    slots = [None] * capacity
    if frozen_age:
        assert cache == {}
    ages = [0] * capacity
    for c, g in order:
        mask = words[c][g] & 65535
        if not mask:
            continue
        cycle += 1
        if frozen_age:
            hit = g in slots
            if hit:
                ages[slots.index(g)] = clock
            else:
                victim = slots.index(None) if None in slots else min(
                    range(capacity), key=lambda i: (ages[i], i))
                slots[victim], ages[victim] = g, clock + 1
        else:
            hit = g in cache
            if not hit and len(cache) == capacity:
                del cache[min(cache, key=cache.get)]
            cache[g] = clock
        clock += 1
        if not hit:
            reads += 96
            for _ in range(12):
                cycle += memory_delay(cycle % 7, extra_latency) + 1
        n = 6 * (bool(mask & 255) + bool(mask >> 8))
        updates += n
        cycle = issue(cycle, n)
    return dict(cycles=commit(cycle) - start, reads=reads, updates=updates), cache


def slot_run(words, mode, depth, start=START, extra_latency=0,
             response_stage=0):
    return stream_run(words, mode, depth, start=start,
                      extra_latency=extra_latency,
                      response_stage=response_stage)[0]


def stream_run(words, mode, depth, start=START, extra_latency=0,
               response_stage=0, capacity=0, cache=None):
    """Two resources: serial 8-bank refills and one Acc24 update per cycle.

    depth limits live adapter response slots, including the currently consumed
    beat. No overlap of full-mask refills is assumed. A slot cannot be reused
    until the clock after its final consumer. Refill may overlap consumption
    only when depth=2. All 12 beats are fetched, including empty half beats,
    so bank/half skipping is not bundled into the lifetime result.

    With capacity > 0, a row-cache receives each returned beat while the same
    bypass serves the consumers. Thus a miss pays exactly the zero-copy
    schedule, including group selection and response stages. Hits use one
    128-byte cache read per beat and require no SRAM refill. Cache write/read
    energy and mapping cost are deliberately not inferred from byte counts.
    """
    cache = {} if cache is None else dict(cache)
    clock = max(cache.values(), default=0) + 1
    cycle, reads, updates, copies, hits = start, 0, 0, 0, 0
    groups = [(g, [c for c in range(4) if words[c][g] & 65535])
              for g in range(48)] if mode == 1 else [
                  (g, [c]) for c in range(4) for g in range(48)
                  if words[c][g] & 65535]
    for g, consumers in groups:
        if not consumers:
            continue
        cycle += 1  # One group/row selection, then pending-context retirement.
        hit = g in cache
        if capacity:
            if not hit and len(cache) == capacity:
                del cache[min(cache, key=cache.get)]
            cache[g] = clock
            clock += 1
        if hit:
            hits += 1
            for beat in range(12):
                n = sum(bool((words[c][g] >> (8 * (beat // 6))) & 255)
                        for c in consumers)
                updates += n
                cycle = issue(cycle + 1, n)
            continue
        completed, retired = [], []
        base = cycle
        for beat in range(12):
            req = base if beat == 0 else completed[-1] + 1
            if beat >= depth:
                req = max(req, retired[beat - depth])
            done = req + memory_delay(req % 7, extra_latency)
            completed.append(done)
            # Registered response ownership: first consumer in the next cycle.
            ready = max(done + 1 + response_stage,
                        retired[-1] if retired else base)
            n = sum(bool((words[c][g] >> (8 * (beat // 6))) & 255)
                    for c in consumers)
            updates += n
            retired.append(issue(ready, n))
            reads += 8
            copies += 128 if capacity else 0
        cycle = retired[-1]
    return dict(cycles=commit(cycle) - start, reads=reads, updates=updates,
                row_cache_payload_write_bytes=copies, row_cache_hits=hits), cache


def chunks():
    for prefix, extent in (("m2051_ep34_tsbg_full40_s1920", 48),
                           ("m2067_ep34_fc2_exact_continuation_s960", 192)):
        meta = json.loads((FIX / (prefix + ".json")).read_text())
        packed = [int(x, 16) for x in (FIX / (prefix + ".memh")).read_text().split()]
        assert len(packed) == len(meta["rows"]) * 4 * extent
        for row in meta["rows"]:
            full = [packed[(row["slot"] * 4 + c) * extent:
                           (row["slot"] * 4 + c + 1) * extent] for c in range(4)]
            for part in row.get("chunk_rows", [{"global_group_base": 0}]):
                begin = part["global_group_base"]
                yield prefix, row, [v[begin:begin + 48] for v in full]


def run():
    vcs = json.loads((HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/result.json").read_text())
    anchors = {r["workload_slot"]: r for r in vcs["rows"]}
    totals = defaultdict(lambda: defaultdict(int))
    ratios, count, calibrated = [], 0, 0
    for prefix, row, words in chunks():
        count += 1
        axes = {}
        for mode, name in ((0, "ordinary"), (1, "tsbg")):
            for capacity in (1, 4):
                axes[f"{name}_lru{capacity}"] = cache_run(words, mode, capacity)[0]
            frozen = cache_run(words, mode, 4, frozen_age=True)[0]
            axes[name + "_frozen4"] = frozen
            if prefix.startswith("m2051"):
                anchor = anchors[row["slot"]]
                assert frozen["cycles"] == anchor["base_cycles" if mode == 0 else "tsbg_cycles"]
                assert frozen["reads"] == anchor["scalar_base" if mode == 0 else "scalar_tsbg"]
                calibrated += 1
            for depth in (1, 2):
                axes[f"{name}_slots{depth}"] = slot_run(words, mode, depth)
            for capacity in (1, 4):
                axes[f"{name}_stream_lru{capacity}"] = stream_run(
                    words, mode, 2, capacity=capacity)[0]
        # The cold group-major stream has no repeated row; giving a row-cache
        # the same streaming path must expose, rather than hide, that equality.
        for capacity in (1, 4):
            for metric in ("cycles", "reads", "updates"):
                assert axes[f"tsbg_stream_lru{capacity}"][metric] == axes["tsbg_slots2"][metric]
        axes["tsbg_slots2_extra_stage"] = slot_run(words, 1, 2, response_stage=1)
        assert len({v["updates"] for v in axes.values()}) == 1
        for axis, result in axes.items():
            for key, value in result.items():
                totals[axis][key] += value
        ratios.append(axes["tsbg_lru1"]["cycles"] / axes["tsbg_slots2"]["cycles"])
    # Explicit same-weight repeated-window stress, not measured frame locality.
    warm = []
    for active_groups in (1, 2, 4, 8):
        words = [[65535 if g < active_groups else 0 for g in range(48)] for _ in range(4)]
        item = {"groups": active_groups, "scope": "synthetic same-identity repeated bundle"}
        for capacity in (1, 4):
            cold, cache = cache_run(words, 1, capacity)
            hot, _ = cache_run(words, 1, capacity, cache=cache,
                               start=START + cold["cycles"] + 384)
            item[f"lru{capacity}_two_bundle_cycles"] = cold["cycles"] + hot["cycles"]
            item[f"lru{capacity}_two_bundle_reads"] = cold["reads"] + hot["reads"]
            cold, cache = stream_run(words, 1, 2, capacity=capacity)
            hot, _ = stream_run(words, 1, 2, capacity=capacity, cache=cache,
                                start=START + cold["cycles"] + 384)
            item[f"stream_lru{capacity}_two_bundle_cycles"] = cold["cycles"] + hot["cycles"]
            item[f"stream_lru{capacity}_two_bundle_reads"] = cold["reads"] + hot["reads"]
        first = slot_run(words, 1, 2)
        second = slot_run(words, 1, 2, start=START + first["cycles"] + 384)
        item["slots2_two_bundle_cycles"] = first["cycles"] + second["cycles"]
        item["slots2_two_bundle_reads"] = first["reads"] + second["reads"]
        warm.append(item)
    ratios.sort()
    return dict(scope="4320 cold G48 chunks from 2880 fixed-region workloads; no wrapper/output-tile scaling",
        calibrated_vcs_axis_rows=calibrated, calibration_mismatches=0, chunks=count,
        axes=dict(totals), slots2_vs_tsbg_lru1=dict(
            ratio_of_sums=totals["tsbg_lru1"]["cycles"] / totals["tsbg_slots2"]["cycles"],
            min=min(ratios), median=ratios[len(ratios)//2], max=max(ratios),
            regressions_over_5pct=sum(r < 1/1.05 for r in ratios)),
        warm_stress=warm, payload_bytes=dict(row_cache4=6144, row_cache1=1536,
            added_zero_copy_payload=0, existing_m803_slots=1024, shared_acc24=1152),
        attribution=dict(
            zero_copy_vs_equal_streaming_cache_cycle_ratio=(
                totals["tsbg_stream_lru4"]["cycles"] / totals["tsbg_slots2"]["cycles"]),
            streaming_vs_old_fsm_includes="refill/consume overlap and once-per-group selection",
            zero_copy_claim="removes row-cache payload copies/state; area and energy unmeasured"),
        limitations=["Candidate cycle model, not VCS or PPA",
            "No lossless sparsity change; same update count and full 8-bank refill",
            "One/two logical slots use existing adapter storage; its 1024 B remains",
            "Cold fixtures do not establish warm deployed-cache locality",
            "No energy/area savings inferred from payload bytes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
