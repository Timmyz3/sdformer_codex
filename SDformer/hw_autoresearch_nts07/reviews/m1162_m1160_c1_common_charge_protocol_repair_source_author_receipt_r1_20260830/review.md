# M1162 — C1 common-charge protocol repair author receipt

Status: **source repair complete; fresh different-author hammer required; VCS
and EDA remain prohibited.**

M1162 is an additive replacement for only the broken M1116C wrapper protocol.
Frozen M935 and the exact M1116C storage ledger are unchanged.  Weight and
first-beat psum requests now assert independently of both ready inputs, latch
one request tuple, and track acceptance independently without duplicate issue.
Responses may be skewed after their own request acceptance, must hold until
ready, and are joined atomically before frozen M935 consumes them.

## Frozen protocol coordinate

- Outstanding depth: 1.
- Request valid depends on ready: false.
- Minimum external response latency: one cycle after its own request accepts.
- Zero-stall, one-cycle-response completed issue-data II: 2.
- Reset cancels the transaction; post-reset responses without a new request
  are spurious.
- Added state: 36 request tuple bits plus four accept/fault bits = 40 bits.
- Response payload FIFO: 0 bits.

The II=2 fact is not a speed result.  In particular, the M1114 raw CPU
`1.7591725402x` coordinate is not inherited.  A matched three-axis,
service-aware replay must charge request acceptance, response latency, skew,
backpressure and the same external common-charge technology before any cycle
claim.

## Source validation performed

- Python bounded unit tests: 8/8 pass.
- Static source/identity check: pass.
- Directed future TB and SVA plan cover both partial-accept orders, both
  response orders, long request stalls, long response backpressure, reset
  while pending, spurious response, sticky errors and no duplicate request.
- The future TB source was authored but not compiled or simulated.

No VCS, DC, PT, Formality, PTPX, GPU, remote job or performance replay ran.
`docs/359` remains SHA-256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Accounting boundary retained exactly

| class | bytes | placement | physical count |
|---|---:|---|---:|
| parent_scratch | 18,432 | foundry_macro_internal | 9 |
| psum_store | 122,880 | identical_external_common_charge | 0 |
| weight_store | 49,152 | identical_external_common_charge | 0 |
| metadata_reserve | 24,448 | identical_external_common_charge | 0 |

Total represented capacity is 214,912 B with 30,848 B margin under 240 KiB.
The external 196,480 B has no numeric area/energy in this source receipt, and
the complete capacity is not physically integrated.

