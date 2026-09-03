# M2053/M2051 full-cohort parser-successor source hammer

**PASS, 99/100; P0/P1/P2 = 0/0/0.** The exact M2053 runner is
authorized for one bounded production attempt only.

## Successor closure

The M2053 parser differs from the failed M2052 parser only in:

1. namespace/docstring/result schema changing from M2052 to M2053;
2. the empty-workload marker changing from obsolete `M2048_EMPTY_...` to the
   exact `M2051_EMPTY_...` string emitted by the unchanged TB; and
3. geometric mean changing from product-then-root to
   `exp(sum(log(speedup))/len(rows))`.

The runner changes only its parser identity, M2053 namespaces/receipts, and
the exact parser SHA pin. Generator reproduction of both files is byte-exact.
The failed M2052 sources, M2051 fixture/TB, M2018/M803 RTL, SVA, filelist, and
40-sample selection remain unchanged.

## Independent semantic attacks

Eleven of eleven static parser tests pass:

- a valid empty workload is accepted;
- the obsolete M2048 marker and unequal empty cycles are rejected;
- a valid 0.998x nonempty workload is accepted, preserving honest tail
  reporting rather than demanding every workload improve;
- duplicate PASS, fatal text, missing SVA covers, and compile errors are
  rejected; and
- 1,920 identical 3x ratios produce a finite geometric mean of
  3.0000000000000004 instead of overflowing.

The unchanged fixture has 1,920 workloads, including 286 empties. It covers
40 samples, four sequences with ten samples each, all 12 FC1 layers, four
G48-supported FC2 layers, and first/middle/last B4 token quartets. Empty rows
remain in aggregate statistics and must have equal baseline/candidate cycles.

## One-shot M2053 authorization

The exact runner SHA
`27da3bf90047918085b0af04795184c4816967f59ab09dc5348d42e2c9628fa2`
is authorized for:

- one fresh M2053 attempt;
- one license query and one VCS compile;
- exactly 1,920 `simv` executions at parallelism four;
- no automatic retry and no other EDA run; and
- atomic publication only after the fail-closed parser succeeds.

Any source/hash/namespace change voids this authorization. Failure must remain
quarantined and cannot be retried under M2053. A successful raw result still
requires an independent result hammer before citation.

The future result remains a component distribution over all 40 captured
samples, all 12 FC1 and four supported FC2 layers, and three fixed token
regions. It cannot establish all-FC2, full-token-population, real-weight,
same-area, macro, hold, power, energy, or system claims.
