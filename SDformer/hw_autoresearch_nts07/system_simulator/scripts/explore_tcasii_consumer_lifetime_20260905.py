#!/opt/anaconda3/bin/python3.12
"""CPU structural screen for a consumer-complete weight-beat buffer.

No cycle simulator, RTL, synthesis, energy estimate, or checkpoint arithmetic
is claimed. Cold-start fixtures match the existing verification population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
FIX = HW / "tb_m2018/fixtures"
SOURCES = {
    "m2051_ep34_tsbg_full40_s1920.json": "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    "m2051_ep34_tsbg_full40_s1920.memh": "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    "m2067_ep34_fc2_exact_continuation_s960.json": "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    "m2067_ep34_fc2_exact_continuation_s960.memh": "c617c6311ce44f15fb820f5dba5460ebd127235a13acd56724b56ccbb10cd594",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_misses(groups, capacity):
    cache = []
    misses = 0
    for group in groups:
        if group in cache:
            cache.remove(group)
        else:
            misses += 1
            if len(cache) == capacity:
                cache.pop(0)
        cache.append(group)
    return misses


def frozen_age_misses(groups):
    # M2018 hit uses pre-increment age; fill uses post-increment age.
    # Equal-age ties select the lowest physical entry. True LRU is also
    # evaluated separately, so the candidate cannot benefit from this quirk.
    tags, ages, clock, misses = [None] * 4, [0] * 4, 1, 0
    for group in groups:
        if group in tags:
            ages[tags.index(group)] = clock
        else:
            victim = tags.index(None) if None in tags else min(range(4), key=lambda i: (ages[i], i))
            tags[victim], ages[victim] = group, clock + 1
            misses += 1
        clock += 1
    return misses


def events(words, beat_major):
    """Emit accepted accumulator updates, without assuming their latency."""
    groups = len(words[0])
    for group in range(groups):
        if beat_major:
            order = ((ctx, half, sl) for half in range(2)
                     for sl in range(6) for ctx in range(4))
        else:
            order = ((ctx, half, sl) for ctx in range(4)
                     for half in range(2) for sl in range(6))
        for ctx, half, sl in order:
            active = (words[ctx][group] >> (8 * half)) & 255
            sign = (words[ctx][group] >> (16 + 8 * half)) & active
            if active:
                yield ctx, sl, group, half, active, sign


def directed_exactness():
    rng = random.Random(20260905)
    accepted, stalled, comparisons = 0, 0, 0
    for case in range(128):
        groups = rng.randrange(1, 12)
        words = [[(lambda a: a | ((rng.getrandbits(16) & a) << 16))(
            rng.getrandbits(16)) for _ in range(groups)] for _ in range(4)]
        old = list(events(words, False))
        new = list(events(words, True))
        # Order is conserved separately for each private accumulator, not just
        # as a commutative sum. This avoids an unjustified overflow assumption.
        for ctx in range(4):
            for sl in range(6):
                a = [e for e in old if e[:2] == (ctx, sl)]
                b = [e for e in new if e[:2] == (ctx, sl)]
                if a != b:
                    raise ValueError("per-accumulator ordering changed")
                for lane in range(16):
                    accumulators = []
                    for sequence in (a, b):
                        acc = 0
                        for _, _, g, h, active, sign in sequence:
                            delta = sum(((-1 if sign & (1 << k) else 1) *
                                (((g * 137 + h * 31 + sl * 17 + lane * 7 + k * 53 + case) & 255) - 128))
                                for k in range(8) if active & (1 << k))
                            acc += delta
                            if not -(1 << 23) <= acc < (1 << 23):
                                raise ValueError("Acc24 overflow")
                        accumulators.append(acc)
                    if accumulators[0] != accumulators[1]:
                        raise ValueError("integer mismatch")
                    comparisons += 1
        # Abstract ready/valid rule: pending consumers retire only on accept.
        index = 0
        while index < len(new):
            pending = new[index]
            ready = rng.randrange(4) != 0
            if ready:
                accepted += 1
                index += 1
            else:
                stalled += 1
                if pending != new[index]:
                    raise ValueError("payload changed under backpressure")
    return dict(cases=128, acc24_comparisons=comparisons,
                accepted_updates=accepted, stalled_cycles=stalled,
                mismatches=0, scope="Python model; not RTL verification")


def analyze():
    for name, expected in SOURCES.items():
        path = FIX / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("fixture identity drift: " + name)
    totals = dict(workloads=0, chunks=0, active_rows=0, live_groups=0,
                  tsbg_lru4_misses=0, tsbg_lru1_misses=0,
                  ordinary_lru4_misses=0, ordinary_lru1_misses=0,
                  ordinary_frozen_age4_misses=0,
                  original_issue_events=0, beat_major_issue_events=0,
                  descriptor_chunks_with_mismatch=0)
    for prefix, extent in (("m2051_ep34_tsbg_full40_s1920", 48),
                           ("m2067_ep34_fc2_exact_continuation_s960", 192)):
        meta = json.loads((FIX / (prefix + ".json")).read_text())
        packed = [int(x, 16) for x in (FIX / (prefix + ".memh")).read_text().splitlines()]
        if len(packed) != len(meta["rows"]) * 4 * extent:
            raise ValueError("fixture extent")
        for row in meta["rows"]:
            slot = row["slot"]
            totals["workloads"] += 1
            full = [packed[(slot * 4 + ctx) * extent:(slot * 4 + ctx + 1) * extent]
                    for ctx in range(4)]
            chunks = row.get("chunk_rows", [dict(global_group_base=0,
                          ordinary_misses=row.get("base_misses"),
                          tsbg_misses=row.get("tsbg_misses"), issues=row.get("issues"))])
            for chunk in chunks:
                begin = chunk["global_group_base"]
                words = [w[begin:begin + 48] for w in full]
                ordinary = [g for ctx in range(4) for g in range(48)
                            if words[ctx][g] & 65535]
                tsbg = [g for g in range(48) for ctx in range(4)
                        if words[ctx][g] & 65535]
                mo4, mo1 = cache_misses(ordinary, 4), cache_misses(ordinary, 1)
                mf4 = frozen_age_misses(ordinary)
                mt4, mt1 = cache_misses(tsbg, 4), cache_misses(tsbg, 1)
                old, new = list(events(words, False)), list(events(words, True))
                if (mf4 != chunk["ordinary_misses"] or mt4 != chunk["tsbg_misses"]
                        or len(old) != chunk["issues"] or len(old) != len(new)
                        or mt4 != mt1 or mt1 != len(set(tsbg))):
                    raise ValueError(f"frozen cache/issue recurrence mismatch {prefix} slot={slot} begin={begin} actual={(mo4,mt4,len(old),len(new),mt1)} expected={(chunk['ordinary_misses'],chunk['tsbg_misses'],chunk['issues'])}")
                # Explicitly preserve the update sequence of each Acc24.
                a, b = {}, {}
                for e in old: a.setdefault(e[:2], []).append(e[2:])
                for e in new: b.setdefault(e[:2], []).append(e[2:])
                if a != b:
                    raise ValueError("real descriptor arithmetic order mismatch")
                for key, value in dict(chunks=1, active_rows=len(tsbg),
                        live_groups=len(set(tsbg)), tsbg_lru4_misses=mt4,
                        tsbg_lru1_misses=mt1, ordinary_lru4_misses=mo4,
                        ordinary_frozen_age4_misses=mf4,
                        ordinary_lru1_misses=mo1, original_issue_events=len(old),
                        beat_major_issue_events=len(new)).items():
                    totals[key] += value
    return {
        "status": "STRUCTURAL_GO_FOR_DESIGN_REVIEW_ONLY",
        "source_sha256": sha(Path(__file__)), "input_sha256": SOURCES,
        "population": "2880 frozen fixed-region workloads; continuation counted once per 48-group chunk, not multiplied by output tiles",
        "cold_start_trace": totals,
        "payload_state_bytes_only": {"frozen_lru4": 6144, "stronger_lru1": 1536,
            "one_8bank_16lane_int8_beat": 128, "two_beats": 256,
            "one_pending_consumer_bitmap_bits": 4,
            "excludes": "M803 slots, Acc24, tags, masks, control, SRAM capacity, timing registers"},
        "warm_cache_counterexample": {
            "group_stream": [0, 1, 2, 3] * 2,
            "lru4_misses": cache_misses([0, 1, 2, 3] * 2, 4),
            "lru1_misses": cache_misses([0, 1, 2, 3] * 2, 1),
            "consequence": "Group-major order alone does not justify shrinking across B4 boundaries. Current M2018 preserves cache in ST_DONE; retain a fallback or charge extra refetches."},
        "directed_model_exactness": directed_exactness(),
        "claim_boundary": {"rtl": False, "cycles": False, "energy": False,
            "area": False, "paper_admitted": False, "new_algorithm": False,
            "proposed_effect": "reduce physical payload lifetime inside C2; no new product sparsity"},
        "next_gate": ["same-cold and warm LRU1/LRU4 controls",
            "fixed M803 port service and context Acc24 bank conflicts charged",
            "VCS backpressure/reorder/abort and exact outputs",
            "matched mapped area/hold and energy; <=5% cycle regression; >=15% total component energy reduction"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))
