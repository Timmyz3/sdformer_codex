# M181 independent hammer review

Verdict: `PASS_WITH_P1_GATES`, score **90/100**.

This review independently read and hashed all 120 FC2 payloads (437,760,000
bytes) from the pinned M51 tar.zst.  It does not import M181, M179, or M172.
All 34 aggregate and 136 per-stage numeric checks match M181 exactly, and a
second scalar recurrence matches the vector recurrence in 792 tests.

## Verified decisions

- Reject fixed-pair K4.  XOR4 is 148,298,103 cycles versus 127,581,198 for
  global-top4 and 144,146,504 for D1.  A review-only enumeration of all 105
  perfect bank pairings found a best point of 148,154,743 cycles; even a
  posthoc per-stage best-of-105 sum is 147,646,255.  Both still lose to D1.
- Advance K8 only as an analytic scale point.  The same-depth K8 result is
  97,607,807 cycles, giving exact analytic ratios 1.307079853x over K4-top4
  and 4.344533568x versus the independently recomputed D1 K1 schedule.
- M181 correctly states that K8 depth is not independently optimized.  A
  review-only, selection-biased sweep finds D={2,4,16,32} and 95,410,406
  cycles rather than M181's D={2,4,8,8}.  This is diagnostic evidence, not a
  claim or a replacement headline.

## Hard gates

The 4.344533568x number is not physical, complete-FC2, or system speedup.
Eight weight-bank responses, eight accumulator lanes, producer/directory
bandwidth, window storage, and cross-descriptor source selection are absent.
K8 needs VCS/SVA, matched Synopsys area/timing, and arithmetic/memory
composition before physical admission.  Fixed bank ownership removes global
bank sorting, but the complete frontend has not yet been shown to be
"wiring-only" or low-cost.

Two wording/reproduction corrections are also needed.  XOR4 is 2.8801% slower
than D1; 2.7995% is D1's reduction relative to XOR4.  The upstream README also
uses an absent `/tmp/m176_payload.QWZFzA` extraction path; reproduction should
pin and extract the M51 archive with SHA256
`aa261ebe64015bbd295f65f4b734efcb6b26c11c3dd0828e9e7a659433f6c3b4`.

## Reproduction

From `hw_autoresearch_nts07`:

```bash
python3 results/m181_independent_hammer_review_r1_20260824/independent_recompute_m181.py \
  --repo-root . \
  --output /ABSOLUTE/NEW/independent_recompute.json

python3 results/m181_independent_hammer_review_r1_20260824/independent_constructive_checks.py

PYTHONPATH=results/m181_independent_hammer_review_r1_20260824 \
python3 results/m181_independent_hammer_review_r1_20260824/independent_all_105_pairings_screen.py \
  --repo-root . \
  --output /ABSOLUTE/NEW/independent_all_105_pairings_screen.json
```

The recomputation reads the tar.zst directly through system `libarchive` and
therefore does not depend on a temporary extraction directory.
