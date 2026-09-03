# ISCAS 2027 a29be040 artifact-open independent review r6

- Date: 2026-09-03 (Asia/Shanghai)
- Reviewed commit: `a29be040339d36c41e8387f865aa05545f1ab969`
- Remote branch head: identical to the reviewed commit at review time
- Mode: read-only paper/evidence review; no EDA, VCS/simv, GPU, license query,
  network experiment, paper mutation, evidence mutation, or `docs/359` mutation

## Verdict

**Overall 4.4/5: Accept. Estimated Accept probability 88--93%; Strong Accept
probability 30--40%.**

The revised four-page paper is now a credible and unusually well-bounded ISCAS
component paper. Commit `a29be040` closes both P1 defects from r5: C1 now states
that its 51.84-million-row model population is ten samples from the single
`zurich_city_09_a` sequence, and C2 now uses the single-campaign M903 metric
(`4.5411x`, `131,086` versus `585,479 um2`, `77.61%`) instead of a hybrid
cross-artifact product. It also clarifies mapped-to-mapped C1 Formality,
separates the integer bridge from the task-level mixed-precision candidate,
discloses the non-integration of the full 240-KiB ledger, and describes the
M2053/M2057 lineage correctly.

The most persuasive result is C2/TSBG: 1,920 fixed real-activity RTL workloads
over 40 samples and four DSEC sequences give `12,522,876 -> 5,124,365` post-load
cycles (`2.443790792x`, `59.07997%` less time) and `64.25234%` fewer scalar
weight requests. The paper keeps the common 383-cycle preload, deterministic
verification weights, natural-unipolar descriptor population, eight excluded
FC2 layers, fixed token quartets, seven slower cases, logic-only physical
anchor, and no-system-speedup boundary visible. This is legitimate
Prosperity/Phi-style component evaluation rather than a disguised full-network
claim.

Strong Accept is not dependable because the paper still lacks a matched
power/hold/memory-closed TSBG point and because TSBG does not yet cover the eight
FC2 layers above G48 or the full token population. C1's headline is a
one-sequence cycle model calibrated by one RTL tile rather than a measured RTL
cycle ratio. These limitations are honestly disclosed, so they cap strength
rather than create a correctness defect.

## Scores

| Dimension | Score / 5 | Assessment |
|---|---:|---|
| Novelty | 3.9 | C1's finite-1RW parent-lifetime object difference is clear. TSBG is correctly claimed as a typed-signed/private-context specialization of established bundle and weight reuse rather than a new reuse principle. |
| Soundness | 4.7 | Claim classes, fair denominators, one-sequence scope, cross-attempt lineage, accuracy caveats, and open physical fields are explicit. No ratio multiplication or system promotion was found. |
| Implementation | 4.7 | C1/C2/C3 have commercial-flow anchors and TSBG has a large directed VCS population. TSBG power, hold closure, and macro state remain open. |
| Evaluation | 4.6 | Forty samples, four sequences, retained empty/slower workloads, same-port/cache comparison, and an 825-frame accuracy gate are strong for four pages; FC2-above-G48 and full-token coverage remain absent. |
| Presentation | 4.4 | Four clean pages, no overfull boxes, a useful claim-boundary table, and no misleading cross-platform ranking. The paper is dense and spends visible space on provenance. |
| **Overall** | **4.4** | **Accept; not yet dependable Strong Accept.** |

## Paper and PDF audit

- The reviewed `main.tex` and PDF match the paper manifest; the paper manifest
  verifies in full.
- PDF: four US-letter pages, all fonts embedded, no clipping or unreadable cell
  found in a four-page visual inspection.
- Build log: no overfull box, undefined reference/citation, LaTeX error, or
  fatal error. The remaining underfull warnings do not obscure content.
- Abstract: 245 words by a conservative TeX-stripped tokenizer, below 250.
- Claim linter: `PASS_ISCAS2027_CLAIM_BOUNDARY_LINTER`.
- `docs/359` remains at SHA256
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The linter is a wording guard, not independent numerical evidence. This review
therefore reopened the sealed C1 model, C1 PT/Formality, C1 energy, C2 M903,
TSBG M2030, C3, task-accuracy, and M2057 result families. Their inner and outer
seals verified.

## Claim-to-evidence reconciliation

### C1: finite-1RW exact product capture

- M1597 independently admits a single-sequence same-ledger cycle model:
  `648,741,051 / 382,848,700 = 1.694510262x`, or `40.985899%` less modeled
  time. Ten samples and `zurich_city_09_a` are now stated in the abstract,
  evaluation text, and Table II boundary.
- The concurrent-access `1.902x` point remains a ceiling, not a physicalized
  result.
- The mapped C1 identity contains nine 128x128 1RW SRAM macros. The canonical
  M1740 receipt admits 3-ns independent PT setup/hold WNS
  `+0.027871/+0.001827 ns` and 16,549 passing mapped-to-mapped Formality points,
  with zero failure classes. The paper now states the proof direction.
- The `166,514 um2` area comes from the exact sealed M1701 mapped identity
  audited in M1714 (`166,514.312080 um2`). Internally, M1714 labels it a
  salvage candidate and M1740 admits PT/Formality rather than DC. The number
  and mapped identity are consistent, but a zero-EDA area-authority receipt
  would make a future artifact handoff cleaner.
- M1789 admits `29.0763 mW` and `22.0689 nJ` only for a 253-cycle directed
  prelayout mixed-corner component window including nine SRAM Liberty leaves.
  The paper does not call it energy/frame or signoff power.

### C2: typed K8 service

- M903 is now used without hybridization. Five frozen directed workloads sum
  to K8 `1,913` versus equal-bandwidth K1x8 `1,945` cycles, or `1.016728x`.
- The same campaign reports K8/K1x8 logic-only area
  `131,086.241193/585,479.153645 um2`, `77.6104%` less logic, and
  `4.541078x` directed-throughput/logic-area.
- The paper explicitly avoids the unfair K8-versus-one-K1 throughput headline
  and leaves matched hold/power open.

### C2/TSBG: context-safe weight broadcast

- The independently re-opened M2057 population has 1,920 unique workload
  slots, 1,634 nonempty and 286 empty. It contains 1,343 improved, 570 tied,
  and seven slower workloads; the worst nonempty speedup is `0.993527508x`.
- Recomputed aggregate: ordinary LRU4 `12,522,876` cycles, TSBG-B4
  `5,124,365`, weighted speedup `2.443790792x`, and time reduction
  `59.0799669%`.
- Recomputed scalar requests: `8,774,304 -> 3,136,608`, a `64.2523441%`
  reduction.
- Per-sequence speedups are `2.5458x`, `2.4706x`, `2.3814x`, and `2.3907x`;
  the aggregate is not carried by one sequence.
- The 1,917 inherited logs and three successor logs are byte-identical to their
  sealed sources and use one compiled `simv`. The paper correctly says M2053
  is not promoted as a successful result.
- M2030 uses the same M2018/M803 RTL source identity and reports
  `249,710.451846` versus `249,739.809848 um2`, or `+0.0117568%` logic-area
  overhead. Both meet setup; both retain diagnostic hold slack `-0.0164 ns`.
- The RTL result remains post-load component execution, not full FC, decoder,
  network, FPS, or system speedup. The paper does not multiply it by the C2
  throughput/area number.

### C3 and numerical/task binding

- C3 remains coverage only: exact 17-cycle/tile service, `63,756.125879 um2`,
  setup/hold met, and 11,180 mapped-to-mapped compare points. No speedup or
  energy is attributed to it.
- The operator-local INT8 bridge and the 825-frame mixed-precision deployment
  candidate are now grammatically separate. The task table reports baseline
  AEE `1.199514` and candidate `1.197367` (`-0.002147`) but also exposes three
  worsening auxiliary metrics, 10/18 per-sequence AEE regressions, and the
  backend mismatch. This is an accuracy compatibility gate, not a causal
  accuracy-improvement claim.

## Corrected severity findings

### P0: 0

No fabricated or numerically inconsistent headline, system-speedup promotion,
failed M2053 promotion, hidden M2058 power result, broken seal, or modified
frozen ledger was found.

### P1: 0

The previous P1 one-sequence omission and C2 hybrid throughput/area authority
are both closed in `a29be040` without introducing a new P1.

### P2: 5

1. C1's exact area is supported by a sealed positive mapped candidate, while
   the later paper-citable M1740 receipt formally admits PT/Formality and keeps
   `dc=false`. A zero-EDA exact-identity area-authority receipt would simplify
   artifact review; there is no observed numerical or identity mismatch.
2. Natural M2057 descriptors are unipolar (`+1` for every nonzero); signed and
   `-128` behavior is synthetic recovery coverage. Keep “typed-signed protocol”
   distinct from “naturally bipolar workload.”
3. TSBG evaluates all 12 FC1 layers but only four of 12 FC2 layers and three B4
   quartets, not all FC2 layers or the full token population.
4. The TSBG physical ablation is logic-only standard-cell state with
   `-16.4 ps` diagnostic hold slack; matched power, memory macros, and hold
   closure are open. M2061 may address power but not automatically the other
   fields.
5. C1 performance is a one-sequence model result calibrated by a single RTL
   tile, and no monolithic/full-network result exists. This is fully disclosed
   but limits Strong-Accept significance and external comparability.

## The one highest-ROI increment outside M2061

**Select exact G48 continuation tiling for the eight currently excluded FC2
layers; do not spend the increment on a new sparsity mechanism or on a standalone
hold-only spin.**

Why this is the best marginal use of time:

1. It closes the most visible workload hole in the strongest `2.4438x` RTL
   result, instead of adding a fourth idea.
2. It reuses the same C2/TSBG G48 engine and ep34 capture, changes neither the
   checkpoint nor task accuracy, and can be written as evaluation depth inside
   C2 rather than a new contribution.
3. The current population has 1,440 FC1 and 480 FC2 workloads. Adding the eight
   excluded FC2 layers over 40 samples and three fixed quartets adds exactly
   `8*40*3 = 960` workloads, expanding the fixed set from 1,920 to 2,880 and
   making all 12 FC2 layers represented.
4. A hold-only repair is lower ROI while M2061 is evaluating the current mapped
   identity: changing the netlist afterward would split power and timing
   identities and could require repeating power. The disclosed `-16.4 ps` is
   small and does not invalidate the current logic-only result.

Executable staged gate:

- First perform an exact CPU/source-level dry run only. Partition each G>48
  FC2 source group into chunks of at most 48, retain each token's Acc24 context
  across chunks, and assert terminal/commit only on the final chunk.
- Charge both ordinary and TSBG modes for identical chunk boundaries, state
  continuation, row-cache behavior, weight requests, and public-port service.
  Descriptor preload remains either included on both sides or excluded on both
  sides; it cannot disappear only from TSBG.
- Require zero integer-oracle mismatches, zero Acc24 overflow, all 960 fixed
  workload slots present, ratio-of-sums speedup at least `1.20x` on the newly
  covered FC2 population, and no aggregate regression before launching VCS.
- If the dry run passes, add one continuation bit/state path to the existing
  wrapper and run the same first/middle/last B4 population. Report the new
  all-FC2 result as component VCS, not full-FC or system speedup. If it fails
  the gate, keep the current paper and disclose the G48 boundary.

If M2061 yields a citable matched power point and the all-FC2 continuation
study passes, the likely paper score rises to roughly 4.55--4.65/5, with
Accept probability about 93--96% and Strong Accept about 45--55%. These are
review tendencies, not guarantees.

## Final recommendation

The current draft is ready for an ISCAS submission after ordinary copyediting;
new experimental work is an upgrade, not a rescue. Preserve the present
two-contribution structure: C1 finite-capacity product capture and C2 typed K8
with TSBG. Keep C3 as exact-service completeness. Do not add S2, RQTB, decoder
matchers, or a whole-network estimate to this four-page version, and do not
convert component time reduction into system FPS.

