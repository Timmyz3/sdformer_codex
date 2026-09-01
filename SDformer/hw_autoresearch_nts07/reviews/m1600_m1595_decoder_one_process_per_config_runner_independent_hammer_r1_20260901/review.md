# M1600 — M1595 decoder one-process-per-config runner independent hammer

Verdict: **PASS different-author source hammer. Authorize exactly one global
actual pilot attempt for D0/call0 across the three frozen non-product
configurations, one fresh fixed Python child per configuration. Do not execute
from this review. This is not the 120-call population.**

## Identity and scope

M1595 pins the exact M1583 engine (`f92c91f0...`), the fully sealed M1592
engineering review, the fixed CPython 3.10.18 executable, docs/359, and resource
manifest `64661d82...`. Both M1592 and the M1595 author receipt pass complete
member-manifest and outer-seal verification. The M1595 contract passes its
inner and outer seals.

The only admitted order is:

1. `DENSE_TYPED_K8`
2. `BIT_EQUAL_SERVICE_K1X8`
3. `BIT_TYPED_K8`

`PRODUCT_CAPTURE_TYPED_K8` is rejected before the M1583 entry. Each child
command starts the pinned Python binary on the pinned M1595 source with a clean
environment, a target-bound private ticket, and exactly one
`M1583.one_shot_worker_entry(config)` call. The parent requires three distinct
child PIDs and tickets. This is structural source admission; the hammer itself
started no real child and opened no payload.

## Attempt and result conservation

The global attempt marker is created with exclusive-create before the first
child. Success and failure publish through `renameat2(RENAME_NOREPLACE)`. A
synthetic second-child failure left the attempt consumed, published a sealed
failure tree, and a second invocation was rejected before reaching the
launcher. A successful synthetic run likewise rejected reuse.

Every child envelope binds parent PID, child PID, configuration, target ticket,
M1583 source, and a canonical result digest. M1583's unchanged result gate
checks exact configuration/resource identity, positive cycles/requests,
request-count equals the sum of kind counts, nonnegative byte counts,
address/commit/payload digests, exact D0/call0/T10 scope, nonmaterialized
streaming, positive RSS gate calls, strict RSS `< 8,388,608 KiB`, and monotonic
RSS maxima. The parent additionally requires one common resource manifest and
one common commit sequence across the three configurations.

The independent hammer rejected 62/62 mutations under CPython 3.6.8 and
62/62 under CPython 3.10.18; the JSON reports are byte-identical. The existing
author suite was independently replayed at 6/6 on both runtimes. All success
and failure exercises used an injected synthetic launcher. Actual worker call,
payload open, GPU, and EDA counts are all zero.

## Narrow release

The sealed authorization is one invocation of M1595 `--run`, consuming one
global attempt and exactly these three fresh child processes for the same
`decoder_stage=D0`, `module_ordinal=0`, `call_ordinal=0`, `timesteps=10` pilot.
It does not authorize product capture, retry after any failure, production,
the full 120-call population, GPU, RTL, or EDA work.

The future result is still diagnostic-only and must receive a separate
independent result hammer. Until then there is no citable cycle, traffic,
speedup, energy, or paper result. M1600 preserves docs/359 at SHA-256
`dedde7ce...` and changes no author file.
