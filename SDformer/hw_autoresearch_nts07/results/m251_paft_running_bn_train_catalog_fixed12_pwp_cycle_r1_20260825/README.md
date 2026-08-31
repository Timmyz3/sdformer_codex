# M251 PAFT running-BN fixed12 PWP cycle replay

M251 replays the disjoint DSEC-train-only M77 k16/q16 catalog on all 40 exact
four-bottleneck records captured from the M87 PAFT-ep4 checkpoint under
deployable running BN.  All packed source payloads are rehashed and expanded
into the exact Conv3x3 row/partition population.

The one-PWP-plus-signed-Hamming candidate reduces exact vector work by
35.092% versus a bit-sparse zero baseline (`1.540642x` natural vector-work
ratio).  Under a common isolated-module cycle model, a 144-byte PWP port plus
96-byte weight port gives `18.833088x` versus dense and `1.540557x` versus
bit-sparse.  Reusing one 96-byte port gives `15.072003x` versus dense and
`1.232898x` versus bit-sparse.  All 17,280 phases are compute-bound after
double-buffered next-partition DMA.

Every 16-source signed-INT8 PWP fits signed12 by construction
(`[-2032,2032]` within `[-2048,2047]`), so this cycle point does not depend on
an optimistic checkpoint-specific PWP width.  It does not yet prove the PAFT
gain over a running-BN no-PAFT source trace, the PAFT checkpoint Acc19 bound,
RTL-integrated cycle equality, power/energy, system speedup or paper PPA.
