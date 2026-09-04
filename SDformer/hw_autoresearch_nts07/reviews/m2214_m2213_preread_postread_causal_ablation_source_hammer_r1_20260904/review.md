# M2214 independent source-only hammer

Status: `PASS_M2214_M2213_SOURCE_HAMMER__M2215_ONE_SHOT_VCS_AUTHORIZED`

Score: 98/100. Severity: P0/P1/P2 = 0/0/0.

## Decisive finding

M2213 is a valid causal control for TSBG read suppression. The frozen ordinary axis is M2018 mode 0 (token-major); the matched post-read and pre-read axes are group-major B4 with LRU4. On a post-read hit, M2213 retains the resident cache index, enters `ST_FETCH_REQ`, issues all 12 bundles with all eight banks enabled, accepts and identity-checks the corresponding responses, discards the returned payload under a miss-only cache-write guard, and enters the typed bridge only after the final accepted response. The pre-read M2018 axis admits the same cache hit before fetch.

The testbench broadcasts one descriptor workload to the three axes, uses the same per-bank memory protocol and the same bridge/commit backpressure, and independently checks every Acc24 lane plus context, tag, slice, terminal, unique commit, product count, request count, and per-bank response count. It also requires `postread_reads - preread_reads == observed_postread_hit_bank_accepts`.

## Mechanical result

- Recomputed 14 source/tool identities and all 10 contract inventory entries.
- Frozen M2018 is `96fb3557...`; M803 is `cd264021...`; `docs/359` is `dedde7ce...`.
- The production parser imports only Python standard-library modules; it has no custom transitive helper to pin.
- VCS, lmutil, and production Python executable identities and modes are pinned in the runner.
- The author test and independent hammer pass; 20/20 independent mutations are rejected.
- M2215 result, attempt, lock, and work identities are virgin.
- The runner requires this exhaustive double seal, score at least 95, severity 0/0/0, caller-pinned runner/review identities, no same-UID EDA collision, and memory headroom.

## Authorization and boundary

Exactly one M2215 license query, one VCS compile, one simv run, and one parser run are authorized. Automatic retry and reuse are forbidden; all other EDA is forbidden.

This review is source-only. It establishes no VCS result, performance, PPA, power, energy, matched-area, paper, or headline claim. The post-read-only debug counters explicitly prevent a matched-area comparison until observability is equalized or a separately reviewed counter-stripped configuration exists. M2215 raw output must still pass independent M2216 review.
