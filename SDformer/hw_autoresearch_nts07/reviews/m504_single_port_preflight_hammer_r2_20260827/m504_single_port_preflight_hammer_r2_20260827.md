# M504 r2 single-port parent scratch independent preflight hammer

Date: 2026-08-27  
Verdict: **GO for the frozen full CPU audit only**  
Score: **94/100**  
RTL authorization: **not yet; it remains conditional on the full four-gate result and a post-result hammer**

## Decision

Both r1 P0 findings are fixed.

1. The analyzer now uses the exact M473 task layout `[sample, operator, row-chunk, partition]` and assigns each phase into `[..., :, partition]`. The ideal 1R1W path is rebuilt with the same frontend, 160-cycle eight-bank weight-DMA floor, eight output banks, two-cycle tail and 960,000-cycle commit term. The full run must equal the frozen 389,974,420-cycle M473 anchor or abort.
2. The old work-conserving path is explicitly a diagnostic upper bound. Admission uses a deterministic, executable deadline-lookahead policy and sets `deadline_lookahead_policy_is_global_optimum=false`. A compressed BFS is used only as a small-task oracle.

The full 17,280-phase CPU audit may run after the active Synopsys resource discipline permits it. A passing audit may nominate M504 RTL; it does not itself admit RTL, VCS, PPA, system speedup or a DATE headline.

## Deadline-lookahead legality

The policy holds a ready current final beat only when the next unmet parent edge belongs to the immediately following active row, the requested parent was already written, the request cannot use current-row forwarding, and `queue + pending < 2`. The hold cycle launches exactly one macro read. Since that acceptance advances the unmet-edge pointer and creates a pending response, the same final beat cannot be repeatedly held for the same edge.

For masks `[1,3,5]`, mapping is residual `[1,2,4]`, parent `[-1,0,0]`:

| Cycle | Current action | Single-port action | Edge state |
|---:|---|---|---|
| 0 | row0 final | write row0 and forward its first child edge | `Q=[0], P=-` |
| 1 | hold row1 final | read already-written row0 for row2 | `Q=[0], P=0` |
| 2 | row1 final consumes old head | write row1; prior response enters after pop | `Q=[0], P=-` |
| 3 | row2 final consumes head | write row2 | queue empty |

This obeys one read XOR one write per cycle, producer validity, synchronous response timing, M498's pop-before-response edge order, ordered head consumption, and the no-consume-credit reservation rule. Production results are work-conserving 5, deadline-lookahead 4, BFS oracle 4.

## Small independent tests

No full workload, Synopsys, VCS or GPU job was run.

### Production self-test

The built-in 260 deterministic cases pass:

- deadline-lookahead faster than oracle: zero;
- deadline-lookahead slower than work-conserving: zero;
- deadline-lookahead nonoptimal cases: zero in this selected set;
- work-conserving nonoptimal cases: 24, maximum gap one cycle;
- mandatory `[1,3,5]`: `oracle=4`, `work=5`, `deadline=4`.

### Exhaustive independent mask test

All 19,607 sequences of one through five nonzero 3-bit masks were tested against the production compressed BFS:

| Rows | Cases | Deadline gap 0 | Work gap 0 | Work gap 1 | Work gap 2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 7 | 7 | 0 | 0 |
| 2 | 49 | 49 | 49 | 0 | 0 |
| 3 | 343 | 343 | 312 | 31 | 0 |
| 4 | 2,401 | 2,401 | 1,897 | 504 | 0 |
| 5 | 16,807 | 16,807 | 10,608 | 5,989 | 210 |

Deadline-lookahead matched the oracle on every exhaustive case. This is test evidence, not a global-optimality claim for 64-row H67 tasks.

### Compressed BFS audit

The production oracle state is `(row_cursor, beat, accepted_edge_prefix, queue_occupancy, pending)`. It does not store row values, FIFO IDs or a 64-bit producer-valid vector:

- completed/written producers are exactly the active-order prefix before `row_cursor`;
- accepted edges form a prefix of the fixed consumer-order requirement list;
- consumed edge count follows completed parent-valid consumers;
- queue and pending IDs are therefore slices of that requirement prefix.

An independent explicit-ID BFS retained the full FIFO ID tuple and pending ID. It matched the compressed oracle on 2,400 random tasks, 300 at every length from one through eight. The five reachable reservation shapes remain `(0,0),(0,1),(1,0),(1,1),(2,0)`. The redundant accepted-prefix coordinate does not make reachable state exponential because the in-order invariant fixes it from consumed edges plus reserved entries.

## M473 identity and order

The r2 storage and flatten order now match M473:

- M473: shape `(S,O,chunks,P)`, assignment `[sample,operator,:,partition]`.
- M504 r2: the same shape and assignment.

Eight deterministic phases spanning operator/sample/partition boundaries (`0,1,53,431,432,970,1727,17279`) were compared directly against M473 row_tile=64 task metrics. `row_count`, `active_rows`, `search_rows`, `parent_edges` and `ideal product_issue_per_block` all matched exactly.

The full anchor cannot be independently demonstrated without running the prohibited full population, but the implementation now reconstructs the same per-task data and order and has a hard equality against the frozen M473 result. This is sufficient to authorize the full audit.

## Macro gates

Both area sensitivities are now present and separately gated:

| Mapping comparison | Area reduction | Gate | Result |
|---|---:|---:|---|
| 9x generated SP 128x128 versus 32x QRT DP 64x36 exact-capacity fallback | 83.336% | >=80% | pass analytically |
| 9x generated SP 128x128 versus 16x QRT DP 128x72 over-depth proxy | 72.376% | >=70% | pass analytically |

The preferred 16x DP 64x72 organization remains compiler-legal but PPA-open. The contract explicitly forbids substituting either proxy for that missing preferred-macro PPA and keeps integrated macro PPA and macro-inclusive power false. This resolves r1's one-sided 83% presentation.

## RTL obligation after a passing full audit

The deadline policy is hardware-realizable, but the future matched comparison must include more than the existing M498 queue core:

1. one registered next-active descriptor carrying `parent_valid` and `parent_id`;
2. next-consumer adjacency qualification;
3. producer-written validity lookup;
4. reserved-capacity and non-forwardable-parent compare;
5. the one-cycle final-beat hold/injected-prefetch control;
6. preservation of current issue payload while held;
7. the nine parallel generated 1RW macro interfaces and response timing.

These controls must be inside the synthesized M504 boundary. Reusing M498's logic-only area while omitting the lookahead scheduler would invalidate the cycle-to-PPA pairing.

Required VCS/SVA crosses include current-with-parent and current-without-parent holds, queue occupancy zero/one, pending-response plus current consume/write, same-row forwarding exclusion, producer-not-written block, full reservation block, sink-backpressured final stability, and reset during a held lookahead transaction.

## Minor non-blocking items

- The r2 output directory and schema are v2, but result and CSV basenames still end in `_r1`. Rename before the paper artifact freezes to avoid receipt ambiguity.
- Add an explicit runtime assertion that `selected.product_cycles == contract.required_anchor`, even though both the M473 result and contract are exact-SHA frozen and the reconstructed equality already fail-closes.
- The comment at analyzer lines 146-149 describes earliest-edge behavior; keep it scoped to accepted-edge ordering so it is not misread as a global policy-optimality statement.
- The cycle audit assumes an always-ready psum sink, inherited from the frozen M473 coordinate. State this in the eventual result receipt; M498 backpressure correctness does not mean random sink stalls were included in the cycle number.

## Integrity

- Analyzer SHA256: `3017dbc290db06924d4f05be7346ef2c4955169afa94fb9d24287bafd353f8df`.
- Contract r2 SHA256: `a6bddb1c94c5e2e5379e8886abfc65349bbb6a0cceb45376efb16672df9e64a1`.
- Runner r2 SHA256: `668f8be7531e010a86a19be83e554bcef73d1773886fedb41a7083a2b45342ad`.
- All six frozen inputs and the r1 review recovery hash matched.
- Python compile, JSON parse and runner shell syntax checks passed.
- The r2 result directory remained absent; the full audit was not started.
- `docs/359_DATE终局冻结_20260813.md` was not modified; SHA256 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
