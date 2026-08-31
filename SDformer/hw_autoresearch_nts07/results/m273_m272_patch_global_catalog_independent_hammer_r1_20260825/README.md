# M273 independent hammer review of M272 Patch global-K16

Verdict: **the trace/work/cycle numbers reproduce exactly, but the wide result
is not a strict same-operand-resource comparison and the dedicated-wide rejection
is a priority heuristic rather than a measured hardware conclusion.** Evidence
quality scores `95/100`; hardware admission scores `50/100`. Findings are
P0/P1/P2 = `0/5/3`.

## Full independent replay

The audit imports no M272 analyzer. It rehashed and decoded all 60 selected M51
payloads (`645,120,000` packed bytes), covering ten samples, six modules and
2,970 tap/channel-partition phases.

M222 receptive-field contributions were independently derived both by explicit
3×3 tap sampling and by input-pixel spatial multiplicity. Both methods and all
60 M222 rows match with zero error. The aggregate work is:

| Work | Exact count |
|---|---:|
| Valid source contributions / bit-sparse ops | 1,774,268,587 |
| PWP ops | 126,603,294 |
| Correction ops | 1,513,888,168 |
| Candidate ops | 1,640,491,462 |
| Natural work speedup | 1.081546980× |

PWP is beneficial on only `6.426824%` of the 1,969,920,000 partition rows.

## Catalog and leakage boundary

Independent count aggregation over 27,648 local M77 catalog entries finds 277
unique masks and reproduces the exact 16 centers and counts. Those 16 account
for `47.384635%` of 12,582,775 local-centroid calibration assignments. Prim's
26 flips match independent Kruskal optimal cost.

The independent selector consumes only M77 and freezes centers before any M51
payload is read. Thus the M272 selection path itself has no Patch-evaluation
input. M77 also reports train-only calibration, no validation use and zero
train-valid825 overlap.

However, the handoff contains neither raw M77 training sample keys nor the
training trace manifest named by its SHA. The zero key overlap cannot be
independently reconstructed. Correct wording is “selector-path leakage-free
under frozen M77 train-only attestation,” not “independently raw-key-proven.”

## Exact cycles and fairness attack

The contract-level totals reproduce exactly:

| Point | Cycles | Speedup |
|---|---:|---:|
| Binary-aware one-source 96-lane reference | 1,883,717,407 | 1.000000× |
| Wide one-cycle PWP | 1,749,942,022 | 1.076445610× |
| Shared96 two-cycle PWP | 1,876,545,316 | 1.003821965× |

Input scan, output commit, tails, initial load, matcher, packer and materializer
recurrences are consistently charged. Wide and shared paths are compute-bound
on all 2,970 phases; minimum compute-over-noncompute margins are 104,913 and
120,121 cycles respectively.

The fairness distinction is crucial. The binary reference and shared96 point
carry 768 operand bits/cycle. The wide point carries 96×12 = 1,152 bits/cycle,
1.5× the reference operand bandwidth. Therefore `1.076446×` is a provisioned
wide sensitivity point, not a strict same-resource result. The matched-width
headline is only `1.003822×`.

That matched gain is fragile: it saves only 7,172,091 cycles. An average extra
`0.056650` cycle per PWP—2.83% of the modeled two-cycle service—removes the gain.
There is no port/arbitration or RTL evidence establishing that precision.

## Dedicated-wide decision

Deprioritizing a dedicated wide PWP path is reasonable: this global catalog
gives only 7.64% isolated-module gain even after adding operand bandwidth, while
the matched path is nearly flat. It is not, however, a measured conclusion.
There is no area, frequency, energy, complete-Patch Amdahl or system evidence
that proves `dedicated_wide_datapath=false`.

The catalog is also a top-frequency merge of local M77 centroids, not a global
optimization over raw Patch-training masks. Its weak result is not an upper
bound on a disjoint patch-specific catalog.

Patch INT8/PWP numeric equivalence remains absent; the M267 seal is a
materialization precedent, not a Patch-weight bit-exact bridge. All evaluations
also come from one sequence and exclude the dense first Patch head, PED shortcut
and complete Patch Embed.

Clean and relocated full-data replays are byte-identical. Mutated M51 manifest
and payload SHAs both exit nonzero and emit no result. The producer directory
and `docs/359` were not modified, and no RTL flow was run.
