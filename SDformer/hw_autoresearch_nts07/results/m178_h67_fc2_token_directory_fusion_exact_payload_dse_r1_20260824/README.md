# M178 H67 FC2 token-directory fusion exact-payload DSE

Status: **PASS exact payload; NO latency gain; retain only as a memory/energy option.**

M178 tested whether the explicit zero-bitmap EOT descriptor charged by M176
could be removed from the serialized 96-bit descriptor lane.  The conditional
architecture builds a separate per-token start/count directory at the ATLIF
write boundary.  At read time, the directory is assumed available beside the
first nonzero descriptor; a count-zero entry preserves a two-cycle empty-token
completion.  This is a native/preindexed architecture, not a posthoc scanner.

Across all 120 frozen H67 FC2 payloads:

- 5,580,000 EOT descriptors are removed from the serialized descriptor lane.
- Descriptor count changes from 24,449,376 to 18,869,376, a 22.822668% cut.
- K1 wall remains 424,060,394 cycles.
- K4 wall remains 144,146,504 cycles.
- Every stage has exactly 1.000000x explicit-EOT/directory-fused K4 latency.

The result is intentionally negative for performance.  Nonzero-token EOT
acceptance was already hidden under source-group replay, while an all-zero token
still needs its directory entry and completion handshake.  Therefore M178 must
not be presented as a cycle speedup.  It can be revisited only after concrete
descriptor/directory SRAM widths and energy are modeled.

The next performance milestone is finite composition of M177 with weight-bank
response, M169 arithmetic and accumulator-context turnover.  That path can
remove exposed replay/commit bubbles; further EOT compression cannot.

Not included: producer or directory RTL, SRAM ports, memory bits/energy,
weight response, arithmetic, accumulator context, complete FC2, physical or
system speedup, paper-ready PPA, or headline admission.
