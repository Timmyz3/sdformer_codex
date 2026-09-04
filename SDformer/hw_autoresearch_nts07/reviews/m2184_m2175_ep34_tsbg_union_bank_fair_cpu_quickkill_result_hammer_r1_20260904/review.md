# M2184 independent M2175 CPU quick-kill hammer

## Verdict

**PASS, 98/100, P0/P1/P2 = 0/0/0.** M2175 is a valid sealed CPU
quick-kill for considering a later RTL implementation. It is not an RTL,
same-area, energy, component-speedup, system-speedup, paper, or headline
result. The only authorization is `GO_RTL_CONSIDERATION` under a new source
and independent RTL gate.

## Reproduced evidence

- Exhaustive double seals pass for the eight-member ep34 capture, frozen M2158
  dense result, two-member M2175 result, and one-member author receipt. The
  source, contract, result, receipt, and protected docs/359 identities match.
- An independent full binary header scan sees exactly 11,040 frames in
  canonical sample/layer/token order. The 40 samples cover four sequences
  equally; the 24 FC layers are 12 FC1 plus 12 FC2. Layer token geometry
  independently sums to 11,160,000 aligned B4 quartets.
- The three frozen dense anchors reproduce exactly:
  `313603627826 / 150234338522 / 67992387648` for ordinary cycles, TSBG cycles,
  and TSBG scalar bank reads.
- The fair masked totals are `244386356403 / 120075325155` ordinary/TSBG
  cycles and `17316452106` TSBG scalar reads. Independent arithmetic gives
  2.035275x TSBG over ordinary, 74.5318% fewer TSBG scalar reads than dense
  fill, and 20.0746% fewer modeled TSBG cycles than dense fill. All three
  contract gates pass.
- The source builds one B4-union mask and passes that same mask to mode 0 and
  mode 1. Both use four rows, one request-port recurrence, the same accept and
  response latency, issue/commit recurrence, and the same axis-specific
  continuation residual. On 54 actual-capture quartets from five layers and
  both modes, an independent scalar implementation matches all 864 fields,
  with zero mismatch.

## Breakdown and tails

FC1 and FC2 ratios-of-sums are 2.103354x and 1.866327x. Across the four
sequences the range is narrow, 2.027524x--2.041654x. All 24 layer rows and
their sums were independently checked.

The aggregate must not hide the tail. Layer 11 is only 1.418096x. There are
618,919 quartets (5.5459%) where TSBG is slower than mask-aware ordinary; the
worst ratio is 0.983542x. Layer 19 has the largest slow-case share, 53.1967%.
Nevertheless, no quartet is slower than the frozen dense-fill TSBG model.
These tails must remain visible in any future RTL DSE.

## Claim boundary

The continuation residual is inherited from the frozen dense model and is
not successor RTL calibration. Therefore 2.035x, 74.53%, and 20.07% remain
CPU-model opportunity/quick-kill numbers. M2184 does not authorize a paper
claim or automatic RTL run. A successor must separately specify RTL source,
VCS equivalence/coverage, same-resource memory timing, synthesis/hold/power,
and an independent result review.
