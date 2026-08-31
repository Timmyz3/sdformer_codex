# M255 independent hammer of M254

Score: **89/100**. Severity: **P0=0, P1=6, P2=4**.

Verdict: **GO for a paired, trace-level, activity-driven isolated four-Conv
direction.** This is not an accuracy-performance Pareto, integrated RTL result,
system speedup or headline.

The independent implementation decoded and rehashed 40 packed support plus 40
float payloads for each arm. Every positive, negative and temporal-change plane
matches its decompressed float tensor exactly. It replayed 51,840,000 partition
vectors per arm without importing M251 or M254 and reproduced all work and
cycle values. Clean producer replay is byte-identical; wrong SHA injection is
rejected before output creation.

- PAFT bit-sparse work reduction: `13.85907284561026%`.
- PAFT candidate work reduction: `13.155263677442946%`.
- WIDE144 PAFT/control throughput: `1.1514612792502774x`.
- SHARED96 PAFT/control throughput: `1.152956085401344x`.
- PAFT is faster on `10/10` samples and lower-work on `4/4` Conv operators.

The mechanism needs careful wording. PAFT pattern efficiency is actually
`0.8104223675148693%` worse than control under the same catalog, so the gain
comes from reduced activity. Global single-seed valid825 running-BN AEE improves
`0.5730215096601543%`, but the complete profiled `zurich_city_09_a` sequence
regresses `1.0189020311889285%`. The exact ten hardware frames average only a
`0.237301327675421%` AEE gain with a 5/5 win-loss split.

The remaining gates are broader paired sequence traces, a multi-seed/common-
evaluator accuracy result, INT8/PWP numerical identities, and an executable
SHARED96 matcher/packer/service boundary.
