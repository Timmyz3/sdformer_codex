# M505-PVRF frozen offline audit independent preflight hammer (r1)

Date: 2026-08-27  
Reviewer role: independent, receipt-blind preflight  
Verdict: **GO_FOR_ONE_FROZEN_FULL_CPU_AUDIT_ONLY**  
Score: **96/100**

## Scope and boundary

This review authorizes exactly one frozen CPU opportunity audit.  It does not
authorize RTL, VCS, DC, Formality, PT/PTPX, GPU work, integrated macro PPA,
full-network speedup, or a DATE headline.  No production file was modified by
the review.

The production identity stabilized during review after the main thread
tightened single-use elision to require the sole consumer to be the immediately
next active row.  The final reviewed identity is:

- analyzer: `9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced`
- contract: `3c1e769fbb9f99e3b3bf50ee7d4658d62ae70aedcc736d5b5d59708f9b0bd5a5`
- runner: `80c11b9886290f8e64731c5654673b593f383cbb1d00cc10d7d43a9d23790c7e`

The runner pins the final analyzer and contract hashes and refuses an existing
output directory.  All frozen input hashes, the M504 result-hammer outer seal,
and the frozen `docs/359` hash were independently checked.

## Contract audit

The three modes are semantically distinct and must remain separate:

1. `m504_baseline`: every active result is stored; existing ordered forwarding
   does not suppress the store.
2. `dead_write_only`: only exact-refcount-zero stores are suppressed.
3. `combined_pvrf`: dead stores are suppressed and a refcount-one store is
   suppressed only when its unique ordered edge is forwarded at producer
   completion to the immediately next active row.

The analyzer preserves the exact M504 task order
`sample, operator, row-chunk, partition`, exact arithmetic issue count, active
row count, parent-edge count, architectural commit, one-cycle synchronous read
response, two-entry ordered queue-plus-pending capacity, no same-cycle consume
credit, and producer validity.  A macro read and macro write cannot occur in
the same cycle.  Multi-use parents always write; a refcount-one result that
cannot make the legal immediate forward also writes.

The full pipeline is reconstructed independently for M473 ideal, M504,
dead-only, and combined work vectors, with the frozen preprocess, eight-bank
replication, tail, and commit terms.  Both the M473 and M504 full-pipeline
anchors are fail-closed.

All five conjunctive materiality gates are implemented before RTL nomination:

- M505 cycle overhead versus M473 at most 5%;
- retained speed versus same-budget M468 zero at least 1.50x;
- generated 1RW macro area reduction versus exact DP fallback at least 80%;
- generated 1RW macro area reduction versus overdepth DP proxy at least 70%;
- scratch-access reduction versus M504 at least 10%.

Any failed gate forces `NO_GO_M505_RTL`; a separate result hammer is required
even if all five pass.

## Independent tests

- Python compilation: PASS.
- Runner shell syntax and exact-SHA/no-overwrite checks: PASS.
- Built-in deterministic self-test: 1,028 cases, zero regression versus M504;
  146 strict cycle-improvement cases, maximum two cycles.
- Exhaustive masks: all 37,448 sequences of length 1--5 over masks 0--7.
  Every case preserved active rows, parent edges, arithmetic issues, FIFO edge
  accounting, and write-liveness accounting.  Accesses and cycles were
  monotone `combined <= dead-only <= M504`; 37,443 cases reduced accesses and
  3,964 reduced cycles.  Maximum exact parent refcount observed was four.
- Independent explicit-ID BFS: all 1,364 sequences of length 1--5 over masks
  0--3.  Combined and dead-only never beat the legal 1RW/synchronous-response
  oracle; combined matched the oracle in every enumerated case.
- Frozen phase 0: 3,000 rows, 47 row tiles, 2,137 active rows, 1,442 parent
  edges, refcount histogram 1,359/412/366 for 0/1/2+, maximum refcount 13.
  Issue-window cycles were 4,664 / 4,494 / 4,494 and macro accesses were
  3,463 / 2,104 / 2,088 for M504 / dead-only / combined.  All per-tile
  invariants passed.

## Non-blocking obligations for the result hammer

These are not P0 blockers for the one full run, but must be closed before any
RTL discussion:

1. The M505 result and CSV carry the essential three-mode cycles/accesses, but
   do not duplicate every M504 baseline stall/hold/forward counter.  The result
   hammer must join the frozen M504 receipt and publish one explicit three-mode
   ablation table rather than cite combined alone.
2. The result package manifest covers result JSON and CSV; the result hammer
   must additionally record and seal the runner SHA shown above.
3. Recompute the two explicit contract anchors `maximum_m505_cycles` and
   `old_macro_accesses` in the result hammer, even though the equivalent cycle
   gate and frozen M504-access equality already fail closed in the analyzer.
4. Refcount metadata is a 128-bit-per-task hardware obligation, not a free
   oracle.  It must be included in matched RTL/DC if and only if the full audit
   passes all five gates.
5. Generated 1RW mapping is not integrated macro PPA.  Preferred DP macro PPA
   remains open and no macro-inclusive energy/area claim is admitted here.

## Final decision

No P0 functional, ordering, port, liveness, identity, or gate-composition flaw
was found in the final frozen identity.  One full CPU audit is authorized under
the runner's three-worker/no-overwrite contract.  **RTL remains forbidden until
all five gates pass and an independent result hammer closes the obligations
above.**

