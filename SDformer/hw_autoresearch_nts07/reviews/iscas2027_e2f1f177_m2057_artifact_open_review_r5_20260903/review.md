# ISCAS 2027 e2f1f177 / M2057 artifact-open independent review r5

- Date: 2026-09-03 (Asia/Shanghai)
- Reviewed commit: `e2f1f1772c4b1849dca249116742e129a3f463e7`
- Mode: read-only artifact-open review; no EDA, simulator, GPU, license query,
  network task, or paper/evidence mutation

## Verdict

**Overall 4.2/5: Accept, estimated 80--88% acceptance tendency. Strong Accept
probability is 20--30%; dependable Strong Accept has not yet been reached.**

M2057 materially improves the evaluation over the earlier M2050 draft: the
exact TSBG distribution now includes all 40 frozen samples from four DSEC
sequences rather than one sample per sequence. The 2.4438x post-load component
result is numerically correct, keeps empty and slower workloads, and is not
promoted to full-FC or system speedup. The paper also cites the relevant reuse
priors and describes TSBG as a typed-signed/private-context specialization,
not a newly invented reuse paradigm.

The draft is already a credible ISCAS component paper, but two evidence-wording
issues prevent a Strong-Accept assessment: C1 omits its one-sequence cycle-model
scope, and C2's 4.5507x throughput/area combines the old five-workload VCS
cycles with a newer DC area pair without exposing that cross-artifact formula
or an exact source-compatibility proof. Neither changes the central M2057
2.4438x result, and both can be fixed without new EDA.

## Scores

| Dimension | Score / 5 | Assessment |
|---|---:|---|
| Novelty | 3.7 | C1 has a clear finite-1RW object difference. TSBG is useful but is explicitly a specialization of established bundle/weight reuse. |
| Soundness | 4.3 | M2057 lineage and scope are unusually transparent; two composite/scope wording gaps remain. |
| Implementation | 4.6 | Large directed RTL population plus C1/C2/C3 commercial-flow anchors; C2/TSBG power attempt remains failed and contributes no result. |
| Evaluation | 4.5 | Forty samples and four sequences are strong for a component paper, but eight FC2 layers, full token coverage, matched power, and whole-network metrics remain open. |
| Presentation | 4.2 | Clean four-page PDF and disciplined boundaries, though very dense and occasionally provenance-oriented rather than reader-oriented. |
| **Overall** | **4.2** | **Accept; not dependable Strong Accept.** |

## Paper/PDF/linter audit

- `main.tex`, PDF, README, references, linter, figure, and paper manifest all
  match commit `e2f1f177` and the manifest verifies.
- PDF: four US-letter pages; all fonts embedded; no clipping or unreadable
  table cell found on visual inspection.
- Build log: no overfull box, undefined control sequence, LaTeX error, fatal
  error, unresolved citation, or unresolved reference. Underfull warnings do
  not obscure content.
- Abstract: 227 words under the paper linter's tokenizer.
- Claim linter: `PASS_ISCAS2027_CLAIM_BOUNDARY_LINTER`.
- `docs/359` remains at SHA256
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The linter is a useful textual guardrail, not an evidence verifier. In
particular, it requires the 4.5507 string but does not test whether the VCS and
DC operands forming that value share one admitted identity.

## Independent M2057 and cross-attempt audit

All 1,920 sealed rows independently sum to:

- ordinary LRU4 cycles: `12,522,876`;
- TSBG-B4 cycles: `5,124,365`;
- weighted speedup: `2.4437907916x`;
- time reduction: `59.0799669%`;
- scalar weight requests: `8,774,304 -> 3,136,608`;
- request reduction: `64.2523441%`;
- 286 empty workloads retained;
- 1,343 improved, 570 tied, and seven slower workloads;
- worst nonempty ratio: `0.9935275081x`.

The result and result-hammer inner/outer seals pass. Exactly 1,917 canonical
logs are byte-identical to the individually valid M2053 logs, and slots 86,
893, and 1755 are byte-identical to the M2057 successor raw logs. M2053's
failure receipt remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE`, exit 123, with no
canonical M2053 result. The successor uses the same compiled `simv`, no new
compile, `-no_save`, and serial execution.

The paper gives the required lineage twice: the Evaluation section says that
1,917 inherited and three successor logs share the same compiled image, and
the Limitations section calls it a double-sealed 1,917+3 same-image
cross-attempt result. This is sufficient disclosure for a scoped component
result. It never says M2053 passed or that the population came from one
successful attempt.

One wording should still be tightened: “the failed parent attempt is not
cited” is literally awkward because 1,917 logs are inherited from it. The
intended and evidence-consistent statement is that M2053 is not promoted as a
successful result, while its individually valid logs are inherited with an
explicit lineage.

## TSBG novelty and scope

The novelty claim is appropriately narrow. ELSA and SpikeX are cited for
bundle/Gustavson and cross-window weight reuse; TSBG claims only a mapping to a
finite row cache with four private destination/Acc24 contexts and no signed
product reuse. The equal-port/cache, same-parametric-RTL baseline and
performance-independent workload selection are strong experimental choices.

The paper correctly discloses that:

- the 1,920 workload population covers all 12 FC1 layers but only four FC2
  layers with at most G48;
- it samples fixed first/middle/last aligned B4 quartets rather than the full
  token population;
- descriptor preload is 383 cycles per workload and is excluded from both
  post-load axes;
- activity/sign descriptors are real ep34 data, while weights are
  deterministic directed INT8 verification values;
- all naturally nonzero measured codes are `+1`; signed and `-128` behavior is
  exercised by a later synthetic recovery phase; and
- the result is a component RTL microbenchmark, not full-FC, decoder, network,
  FPS, or system speedup.

This scope is enough for ISCAS Accept. It limits novelty strength because the
measured natural population does not itself exercise bipolar signs and because
eight FC2 layers require tiling beyond G48.

## C1/C2/C3 evidence audit

### C1

The arithmetic is correct: `648,741,051 / 382,848,700 = 1.694510x`, or
40.9859% less modeled time. The paper labels it `[model]`, separates the
166,514-um2 nine-SRAM mapped anchor, identifies the concurrent-access point as
a ceiling, and says the real-mask VCS tile calibrates events/arithmetic rather
than the 1.6945x cycle ratio.

However, the admitted M1597 wording requires “ten zurich_city_09_a samples,
one sequence.” The paper says only “ten ep34 samples.” The 51.84-M-row result
therefore looks more representative than its actual one-sequence scope. This
is P1-1. Also, C1's 16,549-point Formality proof is mapped-to-mapped
(M1665 mapped reference to M1701 mapped implementation), not a direct
RTL-to-gate proof; unlike C3, Table II does not disclose that direction.

The finite 240-KiB-class model and the mapped nine-parent-SRAM island are
separate axes. The full 214,912-byte ledger is not physically integrated; the
paper avoids saying it is, but a one-line distinction would reduce reviewer
confusion.

### C2 and TSBG

The fair K8 denominator is equal-bandwidth K1x8. The five frozen workloads
support `1,945 / 1,913 = 1.016728x`, not a large cycle claim. The newer M1830
DC pair supports 130,822.775 versus 585,534.972 um2 and 77.6576% logic-area
reduction, with setup met and hold/power open.

The paper's 4.5507x is the mathematically correct hybrid product
`(1945/1913) * (585534.972/130822.775) = 4.550657x`. But M1830 itself admits
only setup/area and explicitly leaves performance-cycle speedup false, while
the already sealed same-campaign M903 metric is 4.5411x using its own
131,086/585,479-um2 area pair. The current 4.5507x therefore needs either an
explicit cross-artifact formula plus exact source/function compatibility, or
replacement by the already admitted 4.5411x metric. This is P1-2.

M2057 and M2030 do share exact M2018/M803 source hashes, so the 2.4438x VCS
distribution and `+0.0118%` matched schedule-mode logic-area ablation can be
reported side by side, as the paper does. They must not be called a
macro/hold/power-closed same-area silicon speedup.

The M2058 matched-power attempt stopped at an ordinary-LRU4 mapped-simulation
fatal/XZ check with zero SAIF and zero PTPX runs. The manuscript correctly
keeps matched hold and power open and uses no M2058 number.

### C3 and numerical binding

C3's 63,756-um2, 3-ns setup/hold result and 11,180 compare points match the
sealed evidence. The paper correctly states that Formality is between pre- and
post-hold-repair mapped netlists and assigns C3 no speedup or energy claim.

The INT8 operator replay and Acc24 bounds are correctly scoped. The abstract,
however, grammatically makes the “operator-local INT8 bridge” the subject that
“passes” the 825-frame AEE gate. The gate actually belongs to a separate
mixed-precision deployment candidate with hardware-order attention, eight QDQ
weights, and other operators at checkpoint precision. The body fixes this
distinction; the abstract should do so too.

## Severity findings

### P0: 0

No fabricated number, seal failure, system-speedup promotion, M2053 promotion,
or hidden M2058 power result was found.

### P1: 2

1. **C1 representativeness omission.** The 1.6945x cycle model is one sequence
   (`zurich_city_09_a`), but the paper discloses only ten samples. Add the
   one-sequence identity wherever the 51.84-M-row scope is introduced.
2. **C2 hybrid throughput/area authority.** The 4.5507x value combines an older
   VCS cycle pair with the newer M1830 area pair, while no single cited hammer
   admits that composite. Use the already admitted M903 4.5411x or explicitly
   prove and state the cross-artifact compatibility and formula.

### P2: 6

1. Replace “failed parent attempt is not cited” with “M2053 is not promoted as
   a successful result; 1,917 valid logs are inherited.”
2. State that C1's 16,549-point Formality comparison is mapped-to-mapped, not
   direct RTL-to-gate.
3. Separate the operator-local INT8 bridge from the mixed-precision AEE-gated
   deployment candidate in the abstract grammar.
4. Clarify that the 240-KiB-class/common-charge ledger is not fully integrated
   into the nine-parent-SRAM mapped C1 island.
5. TSBG's natural measured interval is unipolar and excludes eight FC2 layers
   and full tokens; keep synthetic signed recovery and natural distribution
   distinct in every shortened version.
6. The quantitative prior-work table is a mechanism-boundary table rather than
   a normalized performance comparison. This is defensible for non-identical
   networks but keeps Strong-Accept evaluation strength below papers with a
   matched public workload or silicon system row.

## Minimum path to a stronger submission

1. Fix the two P1s and the abstract AEE grammar; rerun the linter/build and
   reseal. No experiment is required.
2. Keep M2058 failed; do not cite or salvage a power number. If a new power
   campaign is attempted, it must be a separately reviewed successor.
3. For a material Strong-Accept lift, close one matched C2/TSBG hold+power point
   under an explicit memory implementation, or tile the eight FC2 layers above
   G48 and extend beyond three token quartets. Either is more valuable than a
   new lossy mechanism.
4. Do not add C1/C2/C3 ratios, multiply M2057 by area ratios, or convert any
   component reduction into whole-network FPS/energy.

After item 1, the draft remains a strong ISCAS Accept candidate. Items 2--3
raise confidence and potential score but are not required to submit a credible
four-page component paper.
