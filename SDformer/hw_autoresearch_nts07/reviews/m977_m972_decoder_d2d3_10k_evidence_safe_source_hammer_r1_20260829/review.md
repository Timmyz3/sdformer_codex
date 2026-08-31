# M977 independent M972 source hammer

Verdict: **STOP**, score 72/100, P0/P1/P2 = 2/0/0.

The M972 computational repair is directionally correct. Frozen M946, M896,
and the M961 forensic identities verify. The static checker and all five
synthetic tests pass. Independently recomputed source-fetch geometry is
`ceil(231600/192)=1207` for D2 and `ceil(465600/192)=2425` for D3. Bytes and
requests are no longer conflated; prefixes may cross transaction classes and
include commit requests. The runner orders D2 before D3, the normal synthetic
exception path persists a traceback and double-seals its row, and an inert
invocation consumes neither result nor attempt. M977 ran no real 10K/100K,
EDA, GPU, or remote work.

M972 is nevertheless not release-eligible.

First, its source contract, Python driver, and shell runner all hard-bind the
old `M973 -> M974 -> M975` authority chain. M973 is already the C1 result-hammer
request and M974 is already the C2 PT/SAIF/PTPX review. Renaming this review to
M977 does not change the frozen canonical strings and cannot silently satisfy
the exact M973 gate.

Second, a deterministic synthetic attack interrupted recursive sealing after
`SHA256SUMS` was published but before `SHA256SUMS.seal.sha256`. The cleanup
checks only whether `SHA256SUMS` exists. It therefore skips resealing and can
move an unsealed directory into the failure quarantine. This violates the
contract's all-failures-double-sealed claim and is a release-blocking
durability error.

The legal successor is additive and newly numbered: M981 repairs the canonical
chain and partial-seal recovery; M982 independently hammers that source; M983
authors one release; M984 hammers it; and only M985 may execute one D2-then-D3
10K pair. No 100K escalation is authorized.

`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
