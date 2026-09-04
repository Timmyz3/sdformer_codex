# M2159 independent M2158 full-token result hammer

## Verdict

**PASS, 97/100; P0/P1/P2 = 0/0/3.** M2158 is admitted for narrowly
labelled TCAS-II component-model use.  It is not admitted as an RTL, same-area,
energy, full-network, system-speedup, FPS, or abstract-headline result.

The independent checker did not import or invoke the M2145 production
analyzer.  It reimplemented the event/cache recurrence, reconstructed both
frozen VCS populations, decoded all 557 MiB of the sealed ep34 FC capture, and
recomputed every aggregate and every FC1/FC2, sequence, and layer breakdown.
All reported fields match.

## What is now citable

The legal TCAS-II wording is:

> In a VCS-calibrated CPU cycle model over all 11.16M aligned B4 quartets in
> the frozen 40-sample, four-sequence ep34 FC capture (12 FC1 and 12 FC2
> layers), context-safe TSBG reduces modelled execution from 313.604G to
> 150.234G cycles (2.0874x ratio of sums, or 52.0942%) and scheduled scalar
> weight-read requests from 192.483G to 67.992G (64.6762%).

That sentence must remain paired with these boundaries: the inputs are real
ep34 activity and sign descriptors, but weights are directed timing values;
the result is an FC component CPU cycle model, not full-network or RTL
execution; and G96/G192 unseen descriptors use a median residual calibrated
from VCS.  The number may enter a component evaluation table or results
paragraph with a `[VCS-calibrated model]` tag.  It may not enter the abstract
as an unqualified hardware speedup.

Useful robustness diagnostics are also admitted: FC1 and FC2 give 2.1766x and
1.9005x; the four sequence ratios span only 2.0807x--2.0968x; p10/p50/p90 are
1.0000x/1.5414x/2.4388x.  There is no dynamic fallback: 3.1013% of quartets
are marginally slower, with a worst ratio of 0.99755x.  This distribution must
not be hidden if the aggregate ratio is emphasized.

## Independent evidence

- The result contains exactly four regular, non-symlink nodes.  Both JSON
  members and the outer manifest seal verify; there are no unlisted files,
  directories, or links.
- The capture contains exactly 40 samples, four balanced DSEC sequences, 24
  FC layers, 960 sample-layer pairs, 11,040 canonical frames, and 11,160,000
  aligned B4 quartets.  All zlib, CRC, extent, padding, support/sign, and
  `{-1,0,+1}` descriptor checks pass.
- A literal independent state walk matches all 3,840 ordinary/TSBG cycle
  fields in the 1,920 G<=48 VCS rows with zero mismatch.
- The 960 G96/G192 calibration rows reconstruct 1,920 cycle fields under the
  declared descriptor-keyed residual policy.  All 960 selected first/middle/
  last descriptors are found at the exact capture locations and SHA-match.
- A separate Numba implementation recomputes the complete capture.  Every
  integer total, ratio, reduction, percentile, worst/slower/equal/empty count,
  cache statistic, and residual-sensitivity value agrees for the aggregate,
  24 layers, four sequences, and both FC targets.
- Cache accounting is internally closed: scalar weight-read requests equal
  misses times 12 bundles times eight banks on both schedules.

## High-group extrapolation boundary

Of 780,000 G96/G192 quartets, 960 are exact keyed calibration hits and 779,040
(99.877%) use the same-geometry median residual.  Those extrapolated rows are
6.981% of the complete 11.16M population.  Replacing every extrapolated
ordinary residual with the observed minimum and every TSBG residual with the
observed maximum changes the aggregate from 2.087430x to 2.086979x, only
0.0216% relatively.

This is a strong sensitivity result, not a formal error bound: unseen
descriptors are not proven to remain inside the observed calibration extrema.
Also, the 960 continuation rows are in-sample residual anchors, not held-out
validation.  The paper may say “anchored to 2,880 VCS workloads”; only the
1,920 G<=48 rows support “exact recurrence with zero fitted residual.”

## P2 wording and governance notes

1. Call the G96/G192 min/max calculation an **observed residual-envelope
   sensitivity**, never a pessimistic or formal bound.
2. The 40 samples are uniformly selected across four sequences without using
   performance, but they are not the full DSEC dataset.  “Full-token” means
   every aligned B4 token in this frozen capture, not every dataset token.
3. The analyzer has no global attempt lock.  This review confirms one sealed
   M2158 output path and no sibling M2158 result, but cannot infer an execution
   history from a result directory that intentionally contains no run log.

## Claim release

Allowed after this review is sealed: `2.0874297508x`, `52.0941962%`, and
`64.6762431%` in the TCAS-II component-results section with the exact model,
population, and extrapolation labels above.  Denied are “2.09x RTL speedup,”
“52.1% measured execution-time reduction,” “64.7% DRAM traffic/energy
reduction,” same-area claims, full-network/system/FPS claims, and multiplication
with C1 or C2 ratios.
