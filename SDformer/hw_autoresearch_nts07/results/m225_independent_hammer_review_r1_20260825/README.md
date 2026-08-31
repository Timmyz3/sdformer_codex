# M225 independent hammer review

**Score: 88/100. P0: 1. P1: 5. P2: 2.**

The 100 raw FC1 bitpacks were independently decoded and recomputed without importing or invoking either the M224 or M225 production analyzer. All payload hashes pass. Raw, spatial and temporal residuals, K1/K2/K4/K8 row-bounded masks, F1/F2/F4/F8 service ceilings, weight reads, held replays, every serialized overhead and all ten sample distributions match M225 with zero mismatch.

## Verdict

M225 is admitted as an exact M51-s10 opportunity screen, but its automatic F2/F4 advance reference must be corrected before the numbers can be used as a hardware-mechanism gate.

The existing headline-like trace ratios are arithmetically correct:

| Point | Serial cycles | Ratio vs raw K1/F1 |
|---|---:|---:|
| raw K1/F1 | 1,087,104,872 | 1.000000x |
| spatial K8/F2 | 603,154,056 | 1.802367x |
| spatial K8/F4 | 483,481,572 | 2.248493x |

However, raw K1/F1 has only 1,824 bits of Acc19 context state. The selected K8 points have 14,592 bits and K8 grouping/descriptor machinery. The correct three-way factorization is:

| Effect | F2 path | F4 path |
|---|---:|---:|
| K1 to K8 grouping/descriptor geometry | 1.028613x | 1.028613x |
| raw K8 to spatial K8 parent reduction | 1.189066x | 1.189066x |
| spatial K8/F1 to same-parent F2/F4 multicast | 1.473619x | 1.838372x |
| Product versus raw K1/F1 | 1.802367x | 2.248493x |

Against equal-context-capacity **raw K8/F1**, the combined spatial-parent-plus-multicast points remain positive at `1.752230x` and `2.185947x`. Against **spatial K8/F1**, which isolates held-weight multicast at the same parent, K and state capacity, F2 is only `1.473619x` and F4 only `1.838372x`. Both are below M225's frozen mechanism thresholds of `1.5x` and `2.0x`.

Therefore:

- Keep raw K1/F1 as a conventional end-to-end reference, not the sole primary mechanism reference.
- Add raw K8/F1 as the equal-capacity combined-engine reference.
- Add spatial K8/F1 as the multicast-isolation reference.
- Correct both isolated multicast threshold decisions to false.
- A parameterized K8/F1/F2/F4 RTL/DC experiment remains worthwhile for pricing the **combined** parent-plus-multicast engine, because the capacity-matched combined opportunity still passes `1.5x/2.0x`.

## What the trace proves

- Raw events: 112,213,979 positive, zero negative.
- Spatial residual: 66,165,761 positive and 21,043,777 negative events.
- Temporal residual: 103,461,611 positive and 4,236,286 negative events.
- Positive plus negative equals residual presence exactly in every mode.
- Spatial K8/F2 sample ratios versus raw K1 range from `1.768939x` to `1.842518x`; F4 ranges from `2.212879x` to `2.305815x`.
- M225 does **not** directly turn saved weight reads into cycle savings. Raw K8/F1 cuts weight reads from 1,010,523,752 to 391,666,724 while service remains exactly 1,010,523,752 cycles. Cycle reduction comes from simultaneous F-context updates, plus the separately identified grouping and parent effects.

## Required M226 closure

The matched RTL/DC contract must price more than `K*96*19` accumulator bits. At C384/K8 it also needs at least a 768-bit held-weight register, 3,072 presence-mask bits, 3,072 sign-mask bits, source enumeration, replay state and valid/epoch/tag control. It must show that the 256-bit activation scanner can build masks and feed the unique-source walker without hidden bubbles.

F2 uses 192 signed add lanes and F4 uses 384, versus 96 in F1. Their cycle speedup divided by lane multiplier is only `0.9012` and `0.5621` versus raw K1; this diagnostic is not area efficiency. Final selection requires matched 3 ns DC plus SAIF/PTPX and the same SRAM interface.

VCS must cover signed Acc19 replay, negative events, empty masks, full-eight masks, mixed signs, stalls/faults and row tails. Two nonbinary stage-3 FC1 modules remain conventional, so no complete-FC1, FFN, system or headline claim is admitted.

Recompute command:

```bash
/opt/anaconda3/bin/python3 results/m225_independent_hammer_review_r1_20260825/m225_independent_raw_recompute.py
```

It must report `PASS_EXACT_RECOMPUTE`, zero mismatches, raw K1/F1 `1,087,104,872`, spatial K8/F2 `1.8023668434055926x`, and spatial K8/F4 `2.2484928794762835x`.
