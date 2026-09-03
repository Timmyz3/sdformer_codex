# M1866 TSBG ep34 same-I/O quick-kill independent review

Verdict: **PASS evidence, 99/100, P0/P1/P2 = 0/0/0. Select B4 only for the next RTL source-design milestone; no RTL execution, VCS, EDA, power, paper, or system-speedup claim is admitted.**

## What was independently checked

- The full sealed M1707 tree was rehashed and replayed: 40 samples, four DSEC sequences, 32 captured layers, 960 FC1/FC2 pairs, 11,040 frames, and 44,640,000 tokens. Every zlib extent, CRC, support/nnz relation, and nonzero code stream passed.
- The quick-kill result tree, contract/source/checkpoint identities, rows CSV, and frozen docs/359 SHA were checked.
- All 15 all/sequence rows were recomputed directly from `fc_frames.bin` without importing the producer. LRU hits/misses, per-bank weight bytes, compute/commit/schedule work, bundle setup, roofline, serialized cycles, and state bytes match exactly.
- The 93 sealed rows partition exactly into one all, four sequence, two family, and 24 layer rows for each B point. No omitted or duplicated layer was found.
- Mutation checks reject an uncached-K1 baseline, a false same-area promotion, and an unauthorized RTL promotion.

## Fair baseline ruling

This is not the old uncached-K1 comparison. Within each B point, both arms have an **ordinary persistent B-row LRU weight buffer**, the same eight banks at 16 B/cycle each, the same ports, the same trace, the same eight-source compute work, and the same Acc24 commit work. TSBG changes traversal so one fetched row serves multiple private token contexts; it does not share signed products, prune work, or approximate values.

The candidate's extra token contexts are explicitly priced, so `same_area=false` remains mandatory. The three B points are fair within each comparison but are not mutually equal-area configurations.

| Bundle | Serialized CPU premodel | Weight-byte reduction | Worst sequence | Maximum incremental state | Ruling |
|---|---:|---:|---:|---:|---|
| B2 | 1.594x | 40.04% | 1.585x | 3,388 B | low-state ablation |
| **B4** | **2.534x** | **65.20%** | **2.517x** | **10,164 B** | **next source-design point** |
| B8 | 3.893x | 80.20% | 3.880x | 23,716 B | upper DSE only |

B4 is the risk-balanced point: it retains a large margin above the 1.15x gate and reduces weight bytes by 65.20%, while its maximum incremental state is 10,164 B rather than B8's 23,716 B. B2 is useful as a low-state ablation. B8 must not be implemented first or presented as a hardware headline before its context SRAM, fanout, arbitration, and area are proven.

## ISCAS review

As a CPU-premodel-only mechanism this is approximately **3.3/5, borderline weak accept evidence, not yet a hardware result**:

- novelty 3.4/5: the defensible claim is typed-signed H67 token-context weight-row broadcast embedded in C2, not invention of generic weight broadcast;
- soundness 4.5/5: full sealed replay, strengthened baseline, four sequences, and exact independent arithmetic are strong;
- significance 4.0/5: B4's 2.534x/65.20% opportunity is large enough to justify hardware;
- implementation 1.5/5: there is no B4 RTL, VCS, DC, macro timing, or power;
- evaluation 4.0/5 for the premodel stage, with the above physical boundaries.

If B4 closes exact RTL protocol, VCS, matched DC including extra context storage, SRAM timing, and mapped energy while retaining a material same-resource gain, this mechanism can lift the combined C1+C2+C3 paper toward roughly **3.8--4.0/5**. It should be written inside C2 as the memory-specialization, not as a fourth independent novelty bullet.

## Non-negotiable claim boundary

The current ratios are a conservative serialized **CPU premodel**. They are not cycle-accurate RTL latency, frame time, energy, system speedup, or a paper-admitted result. The INT8 row service is a hardware design point; the capture does not grant model-bit-exact or trained weight-quantization authority.

The only next authorization recommended here is: author a new fail-closed **B4 source-design contract**. That contract must precede any RTL edit, and a separate independent source review/release must precede VCS or EDA.
