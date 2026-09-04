# M504 single-port parent scratch independent preflight hammer

Date: 2026-08-27  
Verdict: **REVISE**  
Score: **73/100**  
Execution authorization: **do not start the full 17,280-phase audit yet**

## Outcome

The physical direction is worth preserving: replacing the open 64x1152b 1R1W parent scratch with nine existing 128x128b 1RW generated macros is a credible way to remove a large hidden memory tax without changing M473 arithmetic. The local state machine correctly models most of the sealed M498 protocol: one pending synchronous read, two response slots, no same-cycle consume credit, ordered head-only consumption, response-before-forward enqueue order, producer-before-read, one read XOR one write, and same-row write forwarding.

The current analyzer is not ready for the full audit. It has two P0 modeling defects. One changes the frozen M473 task order; the other incorrectly labels a work-conserving greedy issue policy as optimal. The latter is disproved by a four-cycle legal schedule on a three-row task for which the analyzer reports five cycles.

## P0-1: M504 does not preserve the frozen M473 task order

M473 stores task arrays as `[sample, operator, chunk, partition]` and flattens that layout. M504 allocates and stores `[sample, operator, partition, chunk]`, then reconstructs the frontend in `operator -> partition -> chunk` order. Pipeline cycles contain adjacent-task overlap terms `max(work[i], preprocess[i+1])`, so this is not a harmless permutation.

- M473 allocation/assignment: `analyze_m473_h67_online_subset_live_pwp.py:980-1005`.
- M504 assignment: `analyze_m504_h67_single_port_parent_scratch.py:321-334`.
- M504 frontend order: `analyze_m504_h67_single_port_parent_scratch.py:351-364`.

The final 389,974,420-cycle equality check is good fail-closed protection, but it occurs only after the expensive full population has been processed. It does not make the preceding implementation identical to M473.

Required fix:

1. Store M504 arrays as `[S, O, chunk, partition]`, assigning `[..., :, partition]`, exactly as M473; or explicitly transpose before every flatten.
2. Reuse one frozen helper for task order and pipeline accounting instead of duplicating the loop.
3. Add a cheap pre-run deterministic order fingerprint and a small prefix reconstruction check, while retaining the full 389,974,420 anchor.

## P0-2: the work-conserving earliest policy is not cycle-optimal

The contract says delaying an eligible earliest read cannot make a consumer ready sooner. The analyzer therefore always issues a ready row (`issue = parent_ready`) and forbids a read on that row's final/write cycle. This misses a profitable legal action: deliberately hold a one-beat final issue for one cycle, use that cycle to read the next already-written parent, then overlap the returning response with the held row's consume/write.

### Exact counterexample

Input masks are `[1, 3, 5]`. Clean-room subset mapping gives residuals `[1, 2, 4]`, parents `[-1, 0, 0]`, issue order `[row0,row1,row2]`, and parent requirements `[0,0]`. Every row has one issue beat.

| Cycle | Issue action | Scratch port | Queue/pending edge result | Legality |
|---:|---|---|---|---|
| 0 | issue/final row0 | write row0; forward first parent-0 edge | `Q=[0], P=-` | same-address forwarding; no macro read |
| 1 | deliberately hold ready row1 | read row0 for the second parent-0 edge | `Q=[0], P=0` | producer row0 was already written; reserved occupancy was one |
| 2 | issue/final row1 and consume `Q[0]` | write row1 | pop old head, then prior read response enters: `Q=[0], P=-` | one port operation this cycle; response is from the prior cycle |
| 3 | issue/final row2 and consume `Q[0]` | write row2 | queue drains | complete |

This schedule obeys the M498 rules: no read/write overlap, no consume credit in `prefetch_ready`, response at the following edge, response cannot satisfy the request cycle, head-only FIFO consumption, and no read before producer completion. It finishes in **4 cycles**. The current analyzer finishes in **5 cycles** because it writes row1 in cycle 1, reads row0 in cycle 2, waits for the registered response in cycle 3, and issues row2 in cycle 4.

An independent exhaustive state search over nonzero 3-bit masks found:

| Rows | Cases | Greedy optimal | Greedy +1 | Greedy +2 |
|---:|---:|---:|---:|---:|
| 2 | 49 | 49 | 0 | 0 |
| 3 | 343 | 312 | 31 | 0 |
| 4 | 2,401 | 1,897 | 504 | 0 |
| 5 | 16,807 | 10,608 | 5,989 | 210 |

Thus the current result would be a valid deterministic **work-conserving upper bound**, but not the proposed best single-port schedule and not a sound basis for killing the RTL direction.

### Exact shortest-path state for row_tile <= 64

An exact per-task shortest path is small; it does not need an exponential `written[64]` state.

With the M473 row order frozen, all rows before `row_cursor` have completed and all later rows have not. Producer validity is therefore derived from the producer's position in that order. Because parent edges are accepted and consumed strictly in consumer order, the FIFO IDs are also derived from the requirement list and the number of completed parent consumers.

Use the state:

```text
(row_cursor, beat, queue_occupancy, read_pending)
```

Derived values are:

```text
consumed_edges = count(parent_valid rows before row_cursor)
accepted_edges = consumed_edges + queue_occupancy + read_pending
queue IDs       = requirements[consumed_edges : consumed_edges + queue_occupancy]
pending ID      = requirements[consumed_edges + queue_occupancy] when read_pending=1
producer_ready  = issue_position[parent_id] < row_cursor
```

Each unit-cost transition chooses issue/hold and no-prefetch/read/forward, subject to the M498 reservation and single-port rules. Remove the idle transition that advances neither issue progress nor accepted edges. BFS is then exact. There are only five reachable `(queue_occupancy, read_pending)` reservation shapes: `(0,0),(0,1),(1,0),(1,1),(2,0)`. The state count is `O(5 * (sum(issue_beats) + active_rows))`; for 64 rows and 16-bit masks this is small. A deterministic predecessor tie-break can produce a reproducible schedule trace.

Required fix:

1. Report the current work-conserving policy as an upper bound, not an optimum.
2. Add the exact shortest-path result as the candidate point, or implement a documented port-aware policy and separately bound its gap to the exact small-state oracle.
3. Differential-test the production policy against the exact oracle for exhaustive small tasks and randomized 64-row tasks.
4. Preserve the four-cycle `[1,3,5]` trace as a mandatory unit test.

## Protocol portions that passed review

- Reservation uses `queue occupancy + one pending read` and never borrows current consume credit.
- A macro read is excluded on any final issue/write cycle; forwarding suppresses the macro read.
- A request accepted in one model cycle is pending first and only enters the FIFO at the next state edge; it cannot satisfy the request cycle or the response-enqueue edge's issue.
- Edge update order matches M498: consume head, enqueue older macro response, enqueue same-cycle forwarded word.
- Every parent edge is represented independently and FIFO order is preserved, including duplicate parent IDs.
- Producer-before-read is enforced; a current final row can only serve a future edge through forwarding.
- Arithmetic issue count and parent-edge/read/write accounting are fail-closed.

The exact cycle contract should additionally state that the frozen M473 comparison assumes the psum sink is always ready. M498's backpressure-safe wrapper remains required for RTL, but stochastic sink stalls are not modeled here.

## Macro evidence and gate review

The nine-macro single-port mapping is arithmetically correct: `9 x 128b = 1152b`, with the lower 64 of 128 rows used. Its generated-view area is 78,825.2454 um2 and slow macro cycle/access are 0.6160/0.4679 ns, both correctly labeled as macro-only evidence.

The reported 83.336% reduction is only versus the narrow exact-capacity QRT fallback `32 x DP 64x36`. The same audit also contains a wider over-depth QRT proxy `16 x DP 128x72`, 285,350.64 um2; against that proxy the reduction is 72.376%, below the contract's 75% gate. The preferred `16 x DP 64x72` configuration is compiler-legal but has no generated PPA.

Therefore:

- Retain both sensitivity rows and their evidence labels.
- Do not treat the 75% gate as physical admission until the preferred DP organization is generated or the paper explicitly defines the narrow fallback as the baseline.
- Keep logic, parent scratch, response slots, and psum macros as separate area lines; no integrated macro PPA claim is currently legal.

The 5% cycle-overhead gate is conservative but not unsafe. For decision quality, use two tiers: `<=5%` preferred physical replacement; `5-15%` remains a viable area/performance Pareto point if retained speedup versus M468 remains >=1.50x. This avoids killing a strong point solely through an arbitrary conjunctive threshold.

## Re-run gate

Full M504 execution is authorized only after all of the following:

1. M473 task order is made byte-for-byte/order-for-order identical and checked before full execution.
2. The optimizer claim is removed or replaced by the exact compact-state shortest path.
3. `[1,3,5]` returns four cycles in the optimized column and five in the work-conserving upper-bound column.
4. Small exhaustive differential checks pass.
5. Macro comparison reports both DP sensitivities and keeps integrated PPA false.
6. Frozen input hashes and docs/359 remain unchanged.

No M504 RTL, VCS, Synopsys PPA, system speedup, or DATE headline is admitted by this preflight.

## Integrity checks

- Analyzer SHA256: `3120cb600210548a19fc9756add0e45a5ab900fe776b5ec131dd53c9b9854e1e`.
- Contract SHA256: `162e3bfdc1ae45f03d9d8da0aad64d819bbb1d6842fe925836547bf7eb7c35d6`.
- Runner SHA256: `b60641b8307ddb07255d284a71dee0643998c51a6a0a903c1744b97066a9c42f`.
- All six contract-frozen file inputs matched.
- Analyzer compiled under `/opt/anaconda3/envs/pytorch310/bin/python` 3.10.18.
- The full CPU audit, VCS, DC, PT/PTPX, Formality, GPU and docs/359 were not touched.
- docs/359 SHA256 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
