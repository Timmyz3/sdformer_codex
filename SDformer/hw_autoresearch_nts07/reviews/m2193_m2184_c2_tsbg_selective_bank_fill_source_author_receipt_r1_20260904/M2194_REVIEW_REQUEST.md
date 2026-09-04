# M2194 independent M2193 source-hammer request

Independently review the exact sealed M2193 source candidate. Do not run
`lmutil`, VCS, `simv`, any EDA tool, or a GPU job. M2194 is a source-only
hammer and must not create M2195 attempt, result, work, or lock state.

Recompute every source, parent, contract, and receipt hash. Confirm M2018,
M803, and docs/359 are unchanged. Review the RTL and directed environment for
the following exact semantics:

1. the low/high needed-bank masks are the exact union of all four loaded B4
   contexts for the selected source group;
2. a cache hit requires the resident per-bank valid masks to cover that union;
3. a partial hit requests only `needed & ~valid`, uses
   `source_count=popcount(request_mask)`, and merges returned banks only after
   all six slices are complete;
4. the unchanged M803 supports the nonzero selective mask, independent bank
   backpressure, and out-of-order responses;
5. ordinary and TSBG instances use the same public ports, B4 data, cache
   capacity, and selective-bank ability; only static scheduling order differs;
6. sign, product, destination, tag, terminal, and Acc24 state remain private;
7. the directed test and SVA cover cold refill, partial refill, eviction,
   response reorder, all three backpressure sites, positive and negative
   sources, exact Acc24 including INT8 -128 negation, commit identity, and a
   zero-needed group with no memory request.

Re-run only the static source suite with Python 3.12 and independently inspect
all twelve semantic mutations and five parser mutations. Reject any syntax
uncertainty, stale identity, weakened mask rule, ordinary/TSBG resource
asymmetry, or future-runner budget escape.

Authorize M2195 only at score >=95 and P0/P1/P2=0/0/0, with exact status
`PASS_M2194_M2193_SOURCE_HAMMER__M2195_ONE_SHOT_VCS_AUTHORIZED`. After that
status, M2195 may make exactly one license query, one VCS compile, and one
`simv` run with no automatic retry and no other EDA run. M2196 must
independently hammer the raw result before any functional-verification claim.

