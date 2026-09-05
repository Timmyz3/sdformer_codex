# TCAS-II consumer-lifetime screen: independent review

Date: 2026-09-05. Reviewer: `/root/m2224_lm_discovery_review`. Scope: read-only inspection of the new CPU screen, frozen descriptor fixtures, M2018/M803 RTL, primary literature, and independent CPU recomputation. No source edits, RTL creation, EDA, license query, or Git mutation.

**Decision: CONDITIONAL GO for one bounded cycle/port model; current evidence does not authorize a production RTL/physical campaign.** The idea is useful as a circuit specialization inside C2. Its strongest form is to retire an existing response slot only after its final private consumer, thereby eliminating the second weight copy. It does not establish a new sparsity algorithm or a third paper contribution.

## Recomputed evidence

The independent `analyze()` result exactly equals the frozen `result.json`. Screen source SHA: `238477a7b50b5da1cfd8f3fbc2e439c1ac7e293fced366a778083a05a6d93ab7`. All four fixture hashes match their pinned identities.

| Measurement | Independent result |
|---|---:|
| Fixed-region workloads / 48-group chunks | 2,880 / 4,320 |
| Active context-group rows / distinct live groups | 195,470 / 78,333 |
| Accepted issue events, old and beat-major | 1,662,312 each |
| TSBG cold true-LRU4 / true-LRU1 misses | 78,333 / 78,333 |
| Ordinary cold true-LRU4 / true-LRU1 misses | 194,250 / 195,345 |
| Ordinary frozen-age4 misses | 194,240 |
| Ordinary LRU4 to LRU1 extra misses | 0.5637% |
| Same-LRU1 TSBG miss reduction | 59.9002% |
| Directed model accumulator comparisons | 49,152; zero mismatch |

These are cold fixed-region descriptor results. They are not the full-token population, measured RTL cycles, silicon area, or energy. The old 4-row cache is weak as the sole new capacity comparator: one row preserves all cold TSBG hits and costs ordinary only 0.56% more misses on this population. Presenting 6,144 B to 128 B as a uniquely TSBG-derived 97.9% hardware reduction would hide the strong one-row baseline.

## What is actually proved

Within one B4 chunk, the existing group-major sequence visits each source group contiguously. Once all live contexts of a group retire, that group is not revisited. A one-row buffer therefore has the same cold misses as four rows under this schedule. This follows from the schedule, rather than relying on an average measured reuse rate.

The original local issue order is `(group, context, half, slice)`; the candidate is `(group, half, slice, context)`. For every private `(context, slice, lane)` Acc24, both induce the same ordered `(group, half, active, sign)` updates. The script checks this on all real descriptor chunks. M2018 RTL also advances slice before half and holds separate `acc_q[context][slice][lane]`, so the mapping matches the inspected implementation.

Preserving each accumulator's sequence is stronger than comparing only a commutative sum: with identical initial state and exact per-bank sign extension, every intermediate value is identical for arbitrary legal INT8 weights. The M2018 production bound is `48 * 16 * 128 = 98,304`, below signed Acc24. The directed model's synthetic weights and fewer than 12 groups are a sanity check; they must not be described as a real-checkpoint numerical validation or a new full-extent bound. Continuation/commit handling remains a separate implementation gate.

The random-ready loop is an abstract retirement illustration. It compares `new[index]` with itself while the index is unchanged, so its 12,354 stall iterations do not establish ready/valid correctness, stable RTL outputs, independent-bank reorder, or fault containment.

## The warm-cache boundary is real

The provided stream `[0,1,2,3]` repeated across two B4 bundles yields four LRU4 misses and eight LRU1 misses. M2018 preserves `cache_valid_q`, tags, and payload through `ST_DONE`, so this is a valid counterexample when both bundles use the same weight namespace. Cold equality cannot be promoted to unconditional equivalence across bundle boundaries.

Use one of two explicit contracts: retain a warm-residency fallback and charge its actual storage/control, or remove the cache and charge all cross-bundle refetches. Model only sequential windows belonging to a common layer/output-tile/weight identity. Reusing a numeric group tag across different weight namespaces is not a legitimate warm hit. A physical fallback that retains the old payload cannot claim that payload's area was removed.

Keep the frozen M2018 age update and true LRU separate. Their ten-miss difference is small here, but the alternative must not obtain its advantage from the old pre-/post-increment tie behavior.

## The more useful circuit variant

The cleanest candidate uses the **existing M803 response slot as the only owner of the current weight beat**. A four-bit consumer mask controls private typed-sign/Acc24 updates; the slot is released after the last accepted consumer. M803 already holds response ownership under backpressure and stores `8 slots * 8 banks * 16 lanes * 1 B = 1,024 B`. This suggests removing the second cache copy, rather than introducing another 128 B buffer beside it.

That is a proposal, not a demonstrated implementation. In particular, the consumer controller must not release the slot on the first update, repeatedly update a stalled context, change response identity, or create a combinational ready/valid loop. Epoch/slot/generation checks and malformed-response behavior remain necessary. A group row cannot be marked fully retired until its required halves and slices have completed. Output commits must retain the existing context/tag/slice/terminal contract.

The 128 B figure is one completed eight-bank payload. It excludes M803 slots, context Acc24 (1,152 B), input descriptors, tags, control, and timing registers. Neither a one-beat design nor an adapter-slot design may claim 128 B total storage. Compare a serialized one-beat and a two-beat prefetch variant: filling the next beat while the current beat fans out may be required to hide memory latency, and its ports/state must be charged.

## Fair next experiment and kill gate

Use the same memory latency, eight bank ports, M803 service, 96 output lanes, Acc24 contexts, clock target, and acceptance rules in all axes:

| Axis | Purpose |
|---|---|
| Ordinary frozen-age4 | Link back to admitted predecessor |
| Ordinary true-LRU4 and true-LRU1 | Stronger cache controls |
| Existing group-major TSBG true-LRU4 and true-LRU1 | Isolate surplus row capacity |
| Ordinary one-beat/no-row-cache | Apply the same buffer removal to baseline |
| TSBG one-beat or two-beat consumer retirement | Isolate scheduling plus physical lifetime |

Run cold and contiguous warm same-identity windows, including zero groups, single consumer, all four consumers, mixed signs, asymmetric halves, partial continuation, and long memory/bridge/commit stalls. Count accepted bank activations, refill beats, bridge accepts, cycles, maximum occupied slots, and live payload bits. A new bit of sparsity is not required; the contribution is worthwhile only if total circuit cost falls without consuming the throughput benefit already admitted for TSBG.

Do the bounded CPU cycle/port model first. Proceed to a small RTL variant only if it shows no more than 5% cycle regression against the strongest existing TSBG control and identifies an implementable buffer/port saving. Then require VCS exact outputs and protocol attacks, followed by matched mapped area/setup/hold and total component energy. The predefined target of at least 15% total component energy reduction is a final measurement gate, not a conclusion that follows from payload bytes. Do not add a new EDA queue before the current matched-power campaign closes.

## Novelty and paper position

Ordinary broadcast/multicast, reuse-driven buffering, and returning storage after its last consumer are established techniques. Eyeriss v2 explicitly supports broadcast and grouped/interleaved multicast with reuse-aware dataflow; ELSA combines bundled events with mini-batch spiking Gustavson products to reduce communication and memory traffic. These primary sources constrain any generic novelty claim: [Eyeriss v2, JETCAS 2019](https://eems.mit.edu/wp-content/uploads/2019/04/2019_jetcas_eyerissv2.pdf), [ELSA, 2026](https://arxiv.org/abs/2605.20802).

The defensible object/constraint difference is narrower: schedule-derived finite lifetime for an exact typed-signed beat serving four private Acc24 contexts over a tagged, out-of-order eight-bank response interface. The paper can show why a general-purpose row cache is unnecessary inside the bounded group-major interval, where warm reuse invalidates that claim, and what the mapped circuit gains from the replacement. That is an inference about a useful specialization, not proof of a previously unpublished mechanism.

Broad algorithmic novelty is modest (approximately 2.5/5); fit as a measured TCAS-II circuit refinement is good (approximately 4/5). At present the added paper-score value is zero because area/energy and the strong one-beat ordinary comparison are absent. If those pass, integrate it into C2's storage/energy paragraph and ablation, with the old C2 fabric and C1 unchanged. Its value would come from a cleaner, smaller implementation and a precise limit, not from adding another contribution bullet.
