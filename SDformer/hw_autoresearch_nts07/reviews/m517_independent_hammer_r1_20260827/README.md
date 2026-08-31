# M517 independent hammer review r1

## Verdict

**GO, 95/100, P0=0, P1=4.** The M517 `KILL_NO_RTL` decision is supported under its declared same-resource contract: one shared set of eight logical banks, with at most one source issued per bank per cycle. This review authorizes the negative result only; it does not admit FC2/FFN/system speedup or energy.

## Independent reconstruction

The checker in `recompute_m517_independent.py` does not import or execute the production analyzer. It independently selected 120 FC2 records from the sealed M51 manifest, streamed the 350 MiB `tar.zst`, verified all 120 member sizes and SHA256 identities, decoded the little-endian channel bit packs, and reconstructed the service ledgers.

All requested anchors match exactly:

| Measure | Independent value | Production match |
|---|---:|---:|
| records / modules / samples | 120 / 12 / 10 | yes |
| tokens / 96-bit tiles | 5,580,000 / 36,480,000 | yes |
| active events | 143,894,510 | yes |
| full-vector K8 service floor | 70,657,362 cycles | yes |
| tile-partition service floor | 118,651,292 cycles | yes |
| all-dense sequential work | 1,105,920,000 cycles | yes |
| tiles at least 50% dense | 1,922 | yes |
| maximum nnz per tile | 55 | yes |
| complete nnz/max-bank histograms | exact | yes |

Ten-sample stability is also reproduced: events CV is 0.781%, tile-floor CV is 0.459%, and the empty fraction stays within 47.628–49.074%. The high50 count varies from 130 to 247 (21.2% CV), but remains negligible in every sample.

## Fairness and KILL gate

The decisive result is mathematical and stronger than the empirical tail argument. For every 96-bit tile,

`sparse floor = max(bank occupancy) <= 12 = dense sequential cycles`.

Therefore dense has zero strict wins at zero tax; only six tiles tie. Across tiles, `max(sum bank counts) <= sum(max tile bank counts)`, so retaining cross-tile aggregation only improves the sparse reference. The sealed M216 strong baseline is 90,196,785 cycles; forced tile segmentation is 1.31547x slower, while the best no-op router remains 1.0x and fails the predeclared 1.10x gate.

The calibrated uniform-overhead sensitivity also reproduces exactly: 1.035509759 cycles per nonzero tile routes only six tiles, saves 6.213 cycles before router/format/queue tax, and yields 1.000000069x. This is not exact latency attribution and must remain a sensitivity, but the KILL does not depend on it.

## Required P1 hardening

1. Qualify “no meaningful density mix” to the shared-eight-bank contract. At the lower 25% threshold, 2.6809% of tiles carry 19.655% of events; an independently banked, area-matched parallel path was not evaluated.
2. Treat 118.651M as the explicitly segmented candidate, not an inherent lower bound for every hybrid. Lead with the per-tile dominance proof and no-op result.
3. Keep uniform overhead out of admitted speedup/energy tables.
4. Seal the production result with the M517 contract SHA before archival use.

No energy saving is admitted: matched SAIF/PTPX and SRAM energy are absent, and the production result correctly marks router, format-conversion and ordered-merge taxes as omitted.
