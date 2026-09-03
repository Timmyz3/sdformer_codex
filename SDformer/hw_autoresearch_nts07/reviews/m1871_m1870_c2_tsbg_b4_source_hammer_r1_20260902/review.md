# M1871 independent source hammer of M1870 TSBG-B4

Verdict: **FAIL-CLOSED, 86/100, P0/P1/P2 = 0/1/0. M1872 is not authorized; no VCS, simulator, EDA, license, attempt, result, or release action was taken.**

## What passed

- M1866 is intact and double-sealed. Its 99/100 ruling selects only B4 for a source-design milestone and explicitly withholds execution and paper admission.
- The M1870 contract and author directory seals verify. Frozen docs/359 remains `dedde7ce...`.
- M1870 is an additive B4/LRU4 specialization. Normalizing its B4/LRU4 names and parameters back to B8/LRU8 is byte-exact with immutable M1794, so the predecessor compute/commit path was not edited.
- The architecture is coherent at source level: four independent Acc24 contexts; the real M803 eight-bank, 16 B/bank/cycle protocol; typed `{-1,0,+1}` sources; exact 9-bit `-(-128)=+128`; and no signed-product reuse or approximation.
- The static ledgers independently recompute to 48 rows each, 576 issues each, 9,216 signed products each, and 24 commits each. The ordinary token-major LRU4 point is hit/miss/evict `0/48/44` with 576 aggregate bundle beats and 4,608 scalar bank beats. B4 TSBG is `36/12/8`, 144, and 1,152 respectively.
- The official checker and all 21 tests pass under CPython 3.6 and 3.12. These are source checks only, not VCS evidence.

## Blocking finding

The producer's semantic mutation coverage is not strong enough to authorize the sole VCS campaign. An independent probe preserved the producer's searched vocabulary while weakening 15 stated obligations:

- replayed slot, generation, tag, and 16-lane payload no longer come from the accepted response;
- replay sticky-fault, reset-clear, full post-reset service, post-reset cache/bridge, and local `>=1.15x` gates are neutralized;
- load/bridge/commit B4 context assertions become tautologies;
- the bank-request stability antecedent is disabled;
- the recovery cover is made impossible.

`validate_tb_text` / `validate_sva_text` accepted all **15/15** mutations under both interpreters. In contrast, two positive controls changing `BUNDLE=4` and the candidate hit ledger `36` were rejected. Therefore this is a semantic blind spot, not a probe that bypassed the checker.

Exact source SHA pinning remains valuable identity protection, but it is not a substitute for proving that the checker rejects a meaning-preserving-vocabulary regression. This is the same distinction already enforced in the earlier TSBG governance chain.

## Required correction

Leave M1870 and this review immutable. Create an additive successor that structurally pins the replay identity/payload provenance, zero-accept/sticky gates, both reset recoveries, full post-reset ledgers, the exact local-cycle expression, non-tautological context assertions, effective stability antecedents, and a satisfiable recovery cover. Add all 15 attacks to its mutation suite, then obtain a fresh different-author P0=0/P1=0 review.

Until that gate passes, **do not create M1872 and do not run VCS**. The CPU quick-kill remains opportunity evidence only; no same-area, RTL-cycle, energy, component-speedup, system-speedup, or paper result is admitted here.
