# M472 independent hammer of M468R3/M469 R6

**Score: 88/100. Status: CPU-DSE integrity PASS; performance and RTL NO-GO.**

The reviewer independently recomputed all 324 cycle/traffic/capacity points from the typed 4,250,880-row NPZ, all 270 comparison rows, both row3000/bank8/BW32 anchors, same-budget minima, K resource ledgers, r1-r5 marker-only failure chain, and the producer manifest plus outer seal.

- Stored-PWP anchor: 517,041,352 cycles.
- Strong-zero anchor: 742,148,386 cycles.
- Best 4-bank/128 B-cycle lazy point: row128/K8, 725,989,364 cycles; 1.036627x versus best zero and 1.201743x versus best stored.
- Best 8-bank/128 B-cycle lazy point: row64/K8, 736,160,660 cycles; 1.032859x versus best zero; no stored-PWP point passes both 240 KiB capacity gates.
- No point clears both the 1.15x zero and 1.10x stored gates.
- K8 is explicitly not same-resource: 768 B/cycle, 8 source banks/ports, 672 preadder proxies and 768 product slots.

See the JSON receipt for complete findings and claim boundaries.

## Reviewer execution note

A first fail-closed dry run exposed a reviewer-only schema mistake: the optional non-lazy `overlap_upper_bound_cycles_without_commit` field is stored as zero, while the total overlap field equals normal cycles. The launcher corrects only that comparison interpretation; normal cycles, traffic, capacity, anchors, producer files and gates are untouched.
