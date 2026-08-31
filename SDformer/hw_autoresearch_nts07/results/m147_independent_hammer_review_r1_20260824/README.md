# M147 independent hammer review

## Verdict

M147 is reproducible as a frozen H67/Motion ep35 heldout, same-clock service-island opportunity, but it is not yet admissible as physical, system, PPA, or headline speedup.

The production analyzer was first rerun into a temporary directory and reproduced the sealed JSON byte-for-byte. This review then independently reconstructed the heldout rows, materialized and decoded all 188,148,490 `(row, destination, source)` presence tuples, replayed block-K4 and mosaic-K4 counts, derived PWP1024 beats from the frozen width catalog, and reimplemented the four-bank recurrence without importing M147 or the M141 recurrence. All four production cycle dictionaries match exactly.

Overall score: **5.8/10**. Ledger/reproducibility is strong; the critical destination-combine engine, signed metadata proof, 1024-bit macro, finite endpoint behavior, and matched physical evidence are missing.

## Exact results

| Configuration | Independent cycles | Incremental interpretation |
|---|---:|---:|
| block-K4 + PWP512 (M143 B4 replay) | 135,461,009 | parent |
| block-K4 + PWP1024 | 126,581,635 | 1.070147x vs parent |
| mosaic-K4 + PWP512 | 122,267,417 | 1.107908x vs parent |
| mosaic-K4 + PWP1024 | 75,029,590 | 1.805434x vs M143 B4 |
| mosaic-K4 + PWP1024, no same-destination combine | 137,150,654 | 0.987680x vs M143 B4 |

The published arithmetic is exact:

- 4.6845432315x versus M132 compact256.
- 3.2718546110x versus M132 dualrow512.
- 1.8054344826x versus the cleaner incremental M143 B4 parent.

The first two ratios use the same frozen heldout workload and fixed8 tail, but bundle mosaic packing, four-bank overlap, four conflict-resolved updates, and a 1024-bit PWP source against older serial architectures. They are comparison-table opportunities, not the primary architecture-only claim. The 1.805434x parent comparison is the defensible top-line cycle-model number before RTL and macro closure.

## Destination conflict is the critical dependency

Tuple conservation is exact for row, destination, and source presence. The mosaic stream digest is recorded in the machine result. However, a four-write-port label alone does not implement the needed semantics:

- 35,725,177 of 47,037,211 descriptors (75.95%) contain at least two tuples for the same destination.
- 9,918,824 descriptors (21.09%) put all four tuples on one destination.
- Exact one-write-per-destination service without a combine tree needs 111,038,175 update cycles, 2.360645x the ideal one-cycle-per-descriptor count.
- The corresponding recurrence reaches 137,150,654 cycles, slightly slower than M143 B4.
- A one-cycle implementation must absorb 81,369,024 within-descriptor combine additions while preserving fixed-width signed arithmetic behavior.

Therefore, the 1.805434x opportunity stands or falls on a real conflict-resolved update engine. Independent destination banks solve distinct-destination traffic; they do not solve multiple writes to one destination in the same cycle. A sufficiently wide, non-saturating per-destination combine tree may be semantically equivalent, but its width, timing, switching, and overflow behavior have not been proved.

## PWP1024 saves cycles, not transferred bits

The frozen PWP width counts replay exactly: 11,164,284 width-8, 32,360,036 width-9, 13,936,011 width-10, and 1,509,043 width-11 vectors. Widths 8/9/10 fit in one 1024-bit beat; width 11 requires two.

- PWP512: 119,447,791 beats and 61,157,268,992 transferred bits.
- PWP1024: 60,478,417 beats and 61,929,899,008 transferred bits.
- PWP1024 transfers 1.26335% more raw bits despite reducing beat count 49.37%.
- Aggregate payload utilization falls from 84.22% to 83.17%; a width-11 vector uses only 51.56% of its two 1024-bit beats.
- At 3 ns, the ideal source must deliver 42.67 GB/s, twice the 512-bit port's 21.33 GB/s.

Thus PWP1024 is a latency/bandwidth trade, not an energy reduction. The production contract correctly keeps `pwp1024_sram_macro`, macro energy, matched frequency, physical speedup, and headline admission false.

## Descriptor cost

Mosaic removes block-local K4 tail padding: 99,847,888 block descriptors become 47,037,211 mosaic descriptors. Only 354 of 188,148,844 mosaic slots are padding.

Destination identity is not free. Three bits per valid tuple require 564,445,470 semantic tag bits (about 67.3 MiB), or 564,446,532 allocated slot bits. Even so, a narrow lower bound using only source indices for block-K4 versus source plus destination for mosaic-K4 falls from 1,597,566,208 to 1,317,041,908 bits, a 17.56% reduction because block-local padding is large.

That estimate excludes sign/negate, valid/count, row/window metadata, alignment, ECC, buffering, update-state banking, and macro granularity. It supports continued RTL exploration but not an SRAM-area or energy claim.

## Findings by severity

No P0 ledger corruption was found inside the pinned heldout model. The existing M147 contract is appropriately fail-closed.

| Severity | Finding | Impact |
|---|---|---|
| P1 | Destination conflict engine is assumed, not implemented | Without same-destination combine, the candidate loses its M143 advantage. |
| P1 | Signed tuple metadata is not replayed | Presence tuples are exact, but per-event sign/negate conservation and fixed-width arithmetic equivalence remain unproved. |
| P1 | PWP1024 macro is unpriced | Twice the peak bandwidth and slightly more transferred bits can erase cycle gains through timing or energy. |
| P1 | Baseline bundles architecture generations | 4.6845x/3.2719x are valid opportunity ratios but weaker as primary innovation attribution than 1.8054x versus M143 B4. |
| P1 | Ready, contention, and endpoint latency remain ideal | No finite SRAM/memory contention or matched physical timing is included. |
| P2 | Descriptor/tag/alignment envelope is incomplete | Minimum slot bits improve, but total storage and switching are not closed. |
| P2 | Local5 generalization is absent | All M147 evidence is H67/Motion ep35 heldout. |

## Admission path

1. Implement the mosaic packer and correction engine together. VCS/SVA must cover zero/tail rows, stable ordering, backpressure, repeated destinations, sign/negate, and exact fixed-width accumulation.
2. Synthesize the combine tree at 3 ns. Report area, setup/hold, fanout, and switching; do not treat four nominal write ports as proof.
3. Build a realizable 1024-bit PWP bank/macro envelope and compare energy per vector and total read energy against 512-bit, including width-11 waste.
4. Replay finite ready/latency and measured conflict behavior. Keep M143 B4 as the primary incremental baseline; retain compact256 and dualrow512 as broader genealogy comparisons.
5. Run the same tuple/conflict/width audit on Local5 before claiming dual-line generality.

Until those gates close, the paper-safe wording is: **75,029,590-cycle frozen-heldout same-clock opportunity, 1.805434x versus M143 B4; 4.684543x and 3.271855x are bundled comparisons to older serial parents. No physical, system, energy, PPA, or headline speedup is admitted.**

## Reproduction

From the `SDformer` repository root:

```bash
python3 hw_autoresearch_nts07/results/m147_independent_hammer_review_r1_20260824/independent_recompute_m147.py \
  --output /tmp/m147_independent_recompute.json
```

The script refuses output overwrite and hard-pins M147, M143, M141, M132, M109, the heldout manifest, width inputs, and `docs/359` identities. The durable machine output is `independent_recompute_and_attack.json`; `immutable_manifest.sha256` pins the review artifacts.
