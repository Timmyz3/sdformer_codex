# M734 fresh hammer: M725 two-complete-bank 1RW parent scratch fast-kill

## Verdict

**ADMIT_KILL_NO_FULL_REPLAY_NO_RTL, 97/100, P0/P1/P2 = 0/1/2.**

The sealed stratified result is internally consistent and the negative decision is robust.  A second complete nine-macro 1RW bank fits the common 240 KiB budget, but the best of twelve XOR mappings improves only the local issue window by **1.010470004301x** and removes only **9.009374071%** of single-bank stalls.  Both performance gates fail, so no full-population replay, RTL, VCS, synthesis, PPA, energy, system, or headline claim is admitted.

No new author model, CPU workload replay, GPU, RTL, VCS, EDA, remote, or network job was run.  This hammer reverified sealed identities, inspected the schedule statically, and independently recomputed aggregate integer identities from the existing sealed result.

## Recomputed result

| Quantity | Single complete 1RW bank | Best two-bank mapping | Consequence |
|---|---:|---:|---:|
| XOR mask | — | 2 | minimum cycles, correctly selected |
| Arithmetic issues | 1,139,388 | 1,139,388 | conserved |
| Cycles | 1,287,456 | 1,274,116 | local speedup 1.010470004301x |
| Stalls | 148,068 | 134,728 | reduction 0.090093740714 |
| Parent edges | — | 461,710 | reads 418,303 + forwards 43,407 |
| Live writes | — | 254,635 | unchanged across all mappings |
| Cross-bank read+writes | — | 29,099 | legal only across complete banks |

The capacity sum also closes exactly: `213,376 + 18,432 = 231,808 B`, leaving `13,952 B` under `245,760 B`.

## Population, grain, selftest, and conservation

The contract selects 10 samples × 4 bottleneck Conv operators × 11 deterministic partitions = **440 phases** out of 17,280.  Every selected phase reads all 3,000 rows in 47 chunks of at most 64 rows.  This is a stratified local schedule probe, not a full-population or full-pipeline cycle result.

All twelve mapping rows preserve the same 1,139,388 issues, 461,710 parent edges, 691,482 active rows, 254,635 writes, and single-bank denominator.  For every row, `cycles = issues + stalls`, `parent_edges = reads + forwards`, and the stored speedup/reduction fields reproduce exactly from the integer counters.  The best row is sorted correctly by `(cycles, xor_mask)`.

The sealed selftest reports 512 random cases × 12 mappings, zero regressions, and zero strict improvements.  Static inspection confirms the selftest checks issue conservation, parent-edge conservation, and non-regression against M505.  It is a safety test rather than evidence of benefit; the measured gain comes only from the selected H67 strata.

## Port and capacity fairness

The candidate adds a **complete** second logical 1,152-bit row bank: each bank contains all nine 128-bit slices.  The simulator permits one read and one write in a cycle only when their bank selections differ, and asserts against a same-bank read/write collision.  It does not treat the nine bit slices of one logical row as independent ports.  Thus there is no free-port or pseudo-dual-port credit in the reported 1.01047x.

The 231,808 B figure is macro-rounded capacity only.  It is not post-macro area, frequency, or energy evidence; those would still require an admitted design, which this result does not authorize.

## P1 conservative scheduler defect and robustness bound

The two-bank `deadline_hold` predictor omits the response-queue `capacity` guard present in M505.  When the two-entry response queue is already reserved, this can conservatively hold a final write even though no concurrent read could be launched.  It can overcount candidate stalls, so the result should not be described as an exact optimum for the two-bank scheduler.

This defect cannot overturn the kill.  The best row reports only 22,975 deadline holds.  Giving the candidate the impossible benefit of deleting **every one** yields the conservative upper bound:

- cycles: `1,274,116 - 22,975 = 1,251,141`;
- local speedup: `1,287,456 / 1,251,141 = 1.029025505519x`, still below 1.05x;
- stalls: `134,728 - 22,975 = 111,753`;
- stall reduction: `1 - 111,753 / 148,068 = 0.245258935084`, still below 0.30.

Therefore a repair-and-rerun cannot pass either gate through this defect alone, and spending a full replay or RTL run is not justified.

## Paper boundary

The paper may use this only as a model-labeled negative ablation: within 440 deterministic H67 ep35 bottleneck-Conv strata, adding a complete second parent bank costs 18,432 B but improves the local issue window by only 1.01047x; even the defect-favoring analytical upper bound is below 1.03x.

It must not be presented as Conv or system speedup, same-ledger or full-pipeline performance, full-population evidence, RTL/VCS/Synopsys/PPA/energy evidence, or a paper headline.  The admitted action is to stop: no full replay and no RTL.
