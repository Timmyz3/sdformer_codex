# 05 — Profile evidence (directional, not paper RTL)

Source: `synopsys_date_dual/h67_ep35_real_tile_trace_s1_p64_2d_r2_20260822/.../nts11_hardware_p0_profile.md`  
Checkpoint in that file is **ep35/ep40 path**, **one sample**. Frozen paper identity is **ep34**. Use as motivation for the ep34 census in `07_next_stats_rtl_gates.md`. Do **not** paste these percentages into a paper table as sealed ep34.

## Attention / T=2

| Metric | Value | Mechanism it supports |
|---|---:|---|
| Q toggle density | 0.813120% | dirty-lane skip |
| K toggle density | 2.912012% | dirty-lane skip |
| Q-or-K update density | 3.698706% | Rank 4 |
| t1 ideal lane skip | 96.301294% | Rank 4 upper bound (lane, not row) |
| full T=2 ideal TX compare reduction | 48.150647% | Rank 4 |
| zero-update token/head | 67.185317% | Rank 4 |
| mean changed-token run length | 4.1353 | Rank 7 dirty-run |
| empty 4-token update bundle | 51.907112% | Rank 7 |
| empty 8-token update bundle | 44.164101% | Rank 7 |

## True Token-Time Bundle (T=2 × spatial tokens × 32 lanes)

| spatial tokens/bundle | Q-or-K density | empty | K-zero | no K-motion |
|---:|---:|---:|---:|---:|
| 1 | 2.062% | 66.97% | 75.98% | 76.07% |
| 2 | 2.062% | 59.28% | 71.05% | 71.10% |
| 4 | 2.062% | 51.80% | 66.06% | 66.10% |
| 8 | 2.062% | 44.10% | 60.65% | 60.68% |

## TTX / H67 pair stats (1,512,000 temporal pairs)

| Metric | Value |
|---|---:|
| all-four-vector empty | 66.973347% |
| K motion zero | 76.069114% |
| Q/K temporal update zero | 67.185317% |
| TTX paired scores equal | **97.787368%** |
| H67 paired scores equal | **97.538161%** |
| both K slices zero | 75.976455% |
| exactly one K slice zero | 16.030886% |
| both K slices active | 7.992659% |
| per-token K zero | 83.991898% |

## Stage-wise (H60)

| stage | calls | q_active | k_active | K-zero token | TTB2 empty |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0.00281 | 0.01059 | 0.8585 | 0.3761 |
| 1 | 2 | 0.00136 | 0.00346 | 0.9595 | 0.4271 |
| 2 | 6 | 0.00750 | 0.02396 | 0.7855 | 0.3093 |
| 3 | 2 | 0.00792 | 0.05219 | 0.6243 | 0.1938 |

## ATLIF snapshot in that file

| group | modules | activity | pos_rate | neg_rate |
|---|---:|---:|---:|---:|
| ternary | 0 | 0 | 0 | 0 |
| binary | 93 | 0.063273 | 0.063273 | 0 |

Do not revive ternary Q/K from this snapshot.

## Explicit lossless vs illegal skip (quoted from that profile)

- Q/K empty still produces silent/silent scores and participates in Shiftmax.
- Only **Delta score reuse** and **K-zero value gating** have standalone equivalence proofs for lossless skip.
- Old `TTB2 empty` is a historical proxy, not proof that full attention can be skipped.

## Envelope caveat (from the second workflow, historical)

Attention historically ~0.59% of a cycle envelope ⇒ infinite attention speedup ≈1.006× **as system cycles**. That figure is **not** a restated ep34 census. Remeasure before selling Rank 1 as a system speedup. Rank 1 can still be a **leaf-object** paper (cycles/energy of the score island).
