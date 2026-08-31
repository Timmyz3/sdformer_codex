# M145 independent hammer review

Verdict: **80/100; P0=0, P1=2, P2=5**. Conditional accept as a one-cycle, always-ready synthetic control bound. Do not describe it as a cycle-exact M144/engine result, physical speedup, system speedup, or headline result.

## Exact recomputation

All arithmetic and frozen identities reproduce:

- units: `20 records * 8 windows * 432 partitions = 69,120`;
- four per-unit handoffs: `4 * 69,120 = 276,480`;
- zero-work floors: `1,332 + 300 = 1,632`;
- barrier edges: `160 * 2 = 320`;
- total serial charge: `276,480 + 1,632 + 320 = 278,432` cycles;
- charged B2/B3/B4: `194,250,942 / 147,915,807 / 135,739,441` cycles;
- B4 ratios: `2.589367949x` versus compact256 and `1.808508332x` versus dualrow512.

The charge is only 0.205544% of the M143r2 B4 base. A production rerun is byte-identical at SHA256 `fe3c74fdf2dc92eec09875abebc0a86cbaecd1c97f931e7ea19949f03d4c3ba2`.

## Bound-semantics attack

For a recurrence that explicitly defines every new handoff and zero-work response as exactly one cycle and assumes every endpoint is always ready, adding all charges serially is conservative: overlap can only make a realizable schedule shorter.

That assumption is not frozen today. M144 has unbounded legal waits on `descriptor_ready`, `pwp_ready`, `correction_ready`, and external commit acknowledgement. More importantly, the 1,332/300 zero-work charges are named **minimum** one-cycle floors. A minimum cannot prove an upper bound. The sealed five-job TB itself uses PWP and correction countdowns greater than one while remaining legal.

Therefore `135,739,441` is not a finite upper bound on arbitrary M144 ready/valid execution. It is an exact value for a narrow, synthetic one-cycle control recurrence. The two speedup numbers are lower bounds only inside that same conditional recurrence and against the unchanged frozen baselines.

Several new charges are also potentially duplicate rather than incremental: M143r2 already includes one fill boundary and one dispatch edge per unit, 160 flush edges, and 480,000 commit cycles. Double counting does not break the conditional conservative direction, but it prevents treating 278,432 as measured M144 overhead.

## M144 resource audit

The sealed M142 and M144 DC manifests verify 40/40 members. The same-flow values reconcile exactly:

- M142: 2,562.462012 um2, 3,313 cells, 561 sequential cells;
- M144 integrated: 3,902.472014 um2, 4,998 cells, 847 sequential cells;
- delta: 1,340.010002 um2, 1,685 cells, 286 sequential cells.

The 286 sequential-cell delta exactly equals the independently enumerated 286 wrapper state bits. Because both designs were flattened and optimized separately, 1,340.010002 um2 is an integrated same-flow delta, not an isolated wrapper block area.

Setup is technically MET at +0.0019 ns, only 0.0633% of the 3 ns period; hold is reported +0.0000 ns. The critical endpoint is `correction_accept`. This is ideal-clock, ZeroWireload, zero-macro logic-only evidence, not physical timing margin.

## Required closure

1. Freeze either exact one-cycle always-ready endpoint behavior or finite maximum response/backpressure bounds.
2. Replace the scalar ledger with an edge-labeled recurrence showing which M143 edges are retained, replaced, or newly inserted.
3. Replay all 69,120 units and 160 barriers through M144 or a cycle-identical RTL driver.
4. Keep B2/B3 explicitly model-only; M144 production implements B4 only.
5. Add timing margin and macro-inclusive PT before any physical/frequency statement.

See `independent_recompute_and_bound_attack.json` for machine-readable arithmetic, lineage, resource reconciliation, double-charge evidence, and unbounded-interface checks. `review_score_and_findings.json` contains the scored disposition.
