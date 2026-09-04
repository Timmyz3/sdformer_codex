# M2212 independent directed-VCS result hammer

Verdict: `PASS_M2212_M2211_RESULT_HAMMER__SELECTIVE_BANK_DIRECTED_RTL_ADMITTED__PHYSICAL_GATE_REQUIRED`, 96/100, P0/P1/P2 = 0/0/2.

M2211 admits the selective-bank-fill RTL function and legal-protocol behavior under this directed workload. The unique fresh result has one compile, one top-level simv launch, and one immutable-parser invocation. Both ordinary and TSBG modes completed 72 exact Acc24 commits, with 72/72 context, tag, slice, and terminal identity checks and no assertion or fatal failure. Partial hits, evictions, response reorder, and request/bridge/commit backpressure are all nonzero.

The ledger is internally conserved: ordinary refill reads equal scalar requests at 588; TSBG refill reads equal scalar requests at 156; both modes execute 3264 products and 72 commits. Thus the directed test observes 73.47% fewer refill reads for TSBG while preserving product and commit work. This is a directed access-count result only. Per-mode cycle counters were not emitted, and the two modes run concurrently, so the 6,766,500 ps simulation finish time is not a per-mode latency and no RTL speedup is admitted.

The test stresses legal protocol behavior and stall stability but does not inject an explicit illegal or stale response. That and the missing cycle counters are the two P2 evidence gaps. They do not invalidate the admitted positive-function claim.

The canonical result contains exactly seven sealed evidence files. `simv`, `vc_hdrs.h`, `csrc`, `simv.daidir`, `simv.vdb`, and symbolic links are absent. The result is exhaustively double sealed.

The next gate is a fresh, separately reviewed physical-measurement chain: expose per-mode cycle/read/weight-beat/product counters; synthesize matched ordinary and TSBG modes with identical ports, banks, cache capacity, constraints, and clock; close setup and hold; then use matched SAIF/PTPX plus SRAM-read energy. Until that closes, no CPU multiplier, same-area, timing, hold, power, energy, system-speedup, or paper-PPA claim is allowed. This review authorizes no new execution.
