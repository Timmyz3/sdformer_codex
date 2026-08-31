# M411 independent hammer: M410R2 full ordered runtime VCS

Decision: **94/100, P0/P1/P2 = 0/0/3, GO standalone DC/Formality/PT**.

The exact-SHA Synopsys VCS rerun independently passed 17,280 configurations and all 51,840,000 ordered H67 rows/results. The frozen counters reproduced exactly: pass1 16,037,540, debug pass0-early 3,751,608, and PWP 16,971,357, with zero metadata, arithmetic, configuration, task-flag, protocol, or assertion mismatch.

The old M410 r1 run remains failed and non-citable. Its first failure at row 291 is a reference-contract bug: the source population is one and `use_pwp=0`, so hardware deliberately skips pass1 and returns the exact q16 pass0 fallback `(center=2,distance=3)`. R1 incorrectly expected unused global-q32 metadata `(25,1)`.

An independent 51,840,000-row r1/r2 comparison found byte-identical configurations and preserved row order. Exactly 846,081 rows changed, all population-one rows and all in center/distance together. Original 16-bit rows, `use_pwp`, `pass1`, `early`, reserved bits, populations, and aggregate task/PWP ledgers have zero drift. No population-at-least-two row changed.

The SVA `cp_early=5,350,591` counts every population-at-least-two final result with distance zero. The debug early counter `3,751,608` counts only pass0 zero-distance early stops. The remaining 1,598,983 rows execute pass1 and then finish at exact distance zero, so the two counters reconcile exactly.

Cycle namespace remains narrow: raw TB cycles 67,981,225 equal 67,877,540 matcher tasks plus six harness config/release cycles per phase and five reset cycles. The admitted M401 matcher term is 67,912,100 = 67,877,540 + two cycles per phase. Raw TB cycles are not paper speed, and M411 adds no system, energy, headline, or new speedup claim.

P2 gaps are the absent integrated full-selected-slice real-trace run, the always-ready/attack-free nature of this population TB, and pending standalone DC/Formality/PT plus memory/SAIF energy evidence. Directed R3 evidence remains the source for stall/fault atomicity.

Protected `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
