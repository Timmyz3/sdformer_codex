# M1065 independent hammer: M1064 frozen exact-1RW source

**Verdict: STOP.** M1064 correctly freezes the common-service ledger, capacity
arithmetic, strict schemas and the four-group 1RW arbiter, but it does not bind
the cycle-driving preprocess/work values to the frozen row payload. M1066 full
execution release is therefore not authorized.

## Evidence that survives

- Independent service-only traversal reproduced **812,160 tasks**, **10 sample
  commits**, counts `psum=12,994,560`, `weight=70,853,184`,
  `source=51,840,000`, `dma=1,476,108`, `commit=960,000`, and digest
  `a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea`.
- Empty/partial/duplicate/out-of-order populations fail. Three-design task ID,
  row, row count, preprocess and common-receipt mismatches fail.
- Boolean/extra/duplicate receipt attacks and unsealed/fake/extra/bool/duplicate
  contract attacks fail.
- Physical storage is independently `122,880 + 49,152 + 42,880 = 214,912 B`,
  with `30,848 B` margin under 240 KiB. Caller capacity `0`/override is absent.
- The frozen M1056 kernel still models four groups, one 1RW port per group,
  different-address serialization, same-address RAW and cascade delay.

## P0 counterexample

`FrozenTaskRecord` contains no row masks or row-provenance digest.
`validate_frozen_record` checks only that preprocess agrees across the record and
that each work value is nonnegative; it does not rederive either quantity from
the frozen row bytes.

The hammer retained task 0's exact task ID, coordinate, row, row count and
canonical M1016 common receipt, then changed shared preprocess to 0 and work to
`candidate=0`, `strongest_zero=999999`, `same_coordinate_bit=999999`.
Validation accepted the record.

Independently, a record built from the real first 64 frozen masks and one built
from caller-supplied all-zero masks both validate. Their coverage service state
and digest are identical, while preprocess changes **210 -> 146** and work
changes **1664/4392/4392 -> 0/0/0**. Because `replay_frozen_sample(records)`
accepts caller records, a wrapper could preserve the complete service proof but
manufacture cycle totals.

## Required repair

The sanctioned replay entry must internally read the exact sealed row file and
derive masks, preprocess and all three work values at the point of consumption;
it must not accept caller-built cycle records. Coverage must additionally bind
row payload/provenance, preprocess and per-design work. These two attacks must
become regression tests before a new different-author hammer.

No full replay, EDA, GPU or remote job was run. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
