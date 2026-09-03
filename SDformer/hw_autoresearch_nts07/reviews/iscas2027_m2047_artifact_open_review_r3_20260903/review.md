# ISCAS 2027 M2047 artifact-open independent review r3

- Date: 2026-09-03 (Asia/Shanghai)
- Reviewer identity: `/root/iscas_recent_paper_reviewer`
- Scope: `paper/iscas2027/main.tex`, the four-page `build/main.pdf`, the claim-boundary linter and paper manifest; the double-sealed M2047 VCS result and independent hammer; and the double-sealed M2030 matched-DC independent hammer.
- Mode: read-only review. No EDA or GPU job was launched. The paper, predecessor artifacts, and `docs/359` were not modified by this review.

## Verdict

**Overall: 4.1/5, Accept (estimated 75--85% acceptance tendency). The current paper is a solid ISCAS component paper, but it has not reached Strong Accept.**

M2047 materially improves the implementation and evaluation case: the paper now contains an exact post-load VCS component-cycle result on four fixed, performance-independent ep34 G48 groups, and the exact parametric RTL source is separately tied to M2030's matched logic-only DC ablation. The paper correctly keeps this result separate from full-FC and whole-network performance and does not infer same-area, macro-inclusive, hold-closed, power, or energy claims.

## Scores

| Dimension | Score / 5 | Review |
|---|---:|---|
| Novelty | 3.8 | C1 has a clear finite-capacity/single-port object difference. TSBG is an explicit typed-signed, private-Acc24-context specialization of established weight/bundle reuse rather than a new reuse paradigm. |
| Soundness | 4.5 | Model, VCS, and physical evidence are kept separate; the main limitations and unsupported compositions are disclosed. |
| Implementation | 4.3 | M2047 supplies exact VCS component cycles and M2030 supplies matched DC on the same M2018/M803 source identity. |
| Evaluation | 4.0 | Four sealed DSEC sequences provide a valid representativeness starting point, but the exact RTL interval is still four G48 groups from one FC1 layer and four tokens, with no full-FC, system-energy, or macro-inclusive closure. |
| Presentation | 4.2 | The four-page paper is readable, claim-bounded, and visually clean. It is necessarily dense but the main table exposes evidence boundaries. |
| **Overall** | **4.1** | **Accept; not Strong Accept.** |

## Independent M2047 recomputation

The aggregate arithmetic is correct:

- Ordinary LRU4 cycles: `86,713`.
- TSBG-B4 cycles: `30,775`.
- Speedup: `86,713 / 30,775 = 2.8176441917x`.
- Time reduction: `1 - 30,775 / 86,713 = 0.6450935846`, or **64.509358%**, correctly rounded in the paper to **64.51%**.
- Bundle requests: `7,620 -> 2,292`.
- Scalar requests: `60,960 -> 18,336`.
- Both request reductions are `0.6992125984`, or **69.921260%**, correctly rounded to **69.92%**.
- Per-group speedups span `2.680935x` to `2.978686x`; the weighted aggregate and the reported `2.8176x` are consistent.

The selection boundary is also correct:

- The four groups are the first captured sample of each of four sealed ep34 DSEC sequences.
- The service point is layer 28, the first 48-source-group FC1 service, with token contexts 0--3.
- Selection is fixed independently of performance.
- The measured interval uses real ep34 activity masks; every nonzero source code in those selected groups is `+1`.
- Signed products, reset recovery, and the INT8 `-128` corner are covered only in a subsequent synthetic recovery phase. The paper states this distinction.
- Deterministic signed INT8 weights exercise arithmetic but do not affect scheduling. They are not described as captured model weights.
- The identical 383-cycle preload on both axes is explicitly excluded, so the paper calls the metric post-load execute cycles rather than end-to-end latency.

## Same-source and physical-boundary audit

M2047 compares the same parametric M2018 B4/G48/LRU4 RTL and the same M803 adapter, public ports, cache geometry, memory-port shape, directed bank timing, and backpressure functions. Only static `SCHEDULE_MODE=0/1` changes token-major versus source-group-major ordering and the matching row-clear index. This is a **same-parametric-RTL scheduling comparison**, not an absolute same-area claim.

M2030 separately reports matched TSMC-28 logic-only DC on the exact same source identity:

- Ordinary LRU4: `249,710.451846 um2`, setup WNS `+0.0264 ns`.
- TSBG-B4: `249,739.809848 um2`, setup WNS `+0.0688 ns`.
- Logic-area overhead: `+0.0117568175%`, correctly rounded to `+0.0118%`.
- Both hold diagnostics are `-0.0164 ns`.
- The comparison is pre-macro, ideal-clock, ZeroWireload, and implements state arrays in standard cells.
- Matched macro cost, hold closure, power, energy, and paper-ready PPA remain open.

The VCS and DC facts may be presented side by side with both scopes attached. Their ratios must not be multiplied into an area-efficiency, energy-efficiency, full-FC, or system claim.

## Paper and artifact integrity

- The final PDF is four US-letter pages with embedded fonts and no overfull box, undefined-control-sequence, LaTeX error, or fatal-error finding.
- `check_claim_boundaries.py` returns `PASS_ISCAS2027_CLAIM_BOUNDARY_LINTER`.
- The paper `SHA256SUMS` verifies `main.tex`, `references.bib`, the linter, `build/main.pdf`, and `README.md`.
- The M2047 result, M2047 independent hammer, and M2030 independent hammer each pass their available inner-manifest and outer-seal checks.
- The abstract and Table II both report `86,713 -> 30,775`, `2.8176x`, time `-64.51%`, and requests `-69.92%` with a no-full-FC/system boundary.

## Severity findings

### P0: 0

No claim-integrity, arithmetic, seal, or presentation blocker was found.

### P1: 3

1. M2047 covers four G48 groups, one FC1 layer, and four token contexts. It does not establish full-FC or cross-layer generality.
2. The C2/TSBG physical anchor is logic-only. Hold, SRAM macros, and power are not closed.
3. C1's `1.6945x` remains a cycle-model ratio with a separate mapped physical anchor rather than one workload that closes model, RTL throughput, and physical implementation together.

These findings keep the paper below Strong Accept, but they do not block submission as an ISCAS component paper.

### P2: 3

1. The measured real-activity interval has no natural negative source code. The signed path is separately checked by synthetic recovery and is disclosed correctly.
2. Directed INT8 weights are not captured real weights. This is acceptable for a schedule whose decisions are weight-value independent, but the distinction must remain explicit.
3. The paper worktree should be committed and submission debris excluded before release; this is operational hygiene rather than a scientific claim issue.

## Strong-Accept gate

The paper does **not** currently reach Strong Accept because the `2.8176x` result remains a narrow component microbenchmark. The highest-return improvement would be to apply the same predeclared M2047 procedure to a broader fixed population spanning more FC1/FC2 layers and token groups and report the distribution, while preserving the ordinary-LRU4 same-port baseline. A matched macro-aware, hold-closed power result would be the next most valuable physical closure. Neither is required for an honest ISCAS submission at the present Accept level.

## Permitted concise wording

> On four performance-independent ep34 FC1 G48 service groups, one per DSEC sequence, exact post-load VCS execution on the same parametric B4/LRU4 RTL reduces cycles from 86,713 to 30,775 (2.8176x; 64.51% less execution time) and weight requests by 69.92%. A separate matched 3-ns TSMC-28 logic-only DC ablation on the same source adds 0.0118% area and meets setup; hold, macros, power, energy, full-FC, and system performance remain open.

## Forbidden promotions

- Full-FC, full-capture, whole-network, FPS, or system speedup.
- Real-weight or empirically bipolar ep34 cycle workload.
- Absolute same-area, macro-inclusive, hold-closed, power, energy, or paper-ready PPA.
- Multiplication of the M2047 cycle ratio by the M2030 area result or by another component ratio.
- Relabeling Prosperity, Phi, FireFly-T, ELSA, or SpikeX mechanisms or published results as this work's measured result.
