# M182 independent hammer review

Verdict: `PASS_WITH_P1_GATES`, score **89/100**.

I independently decoded and SHA-checked all 120 frozen H67 FC2 payloads
(437,760,000 bytes) without importing or executing M182 or any upstream
analyzer.  D={1,2,4,8,16,32} aggregate/per-stage window, group, replay and wall
results match exactly.  A separately written scalar scheduler matches the
vector two-window recurrence in 1,560 tests.

## Verified result

- The in-sample oracle is D={2,4,16,32}, 95,410,406 cycles and
  4.444592700x relative to independently recomputed optimized K1.
- The correct first RTL point is D={2,4,8,8}, 97,607,807 cycles and
  4.344533568x versus K1.  It is only 2.303104% slower than the oracle while
  reducing maximum two-buffer bitmap payload from 6,144 to 1,536 bits.
- M181's same-depth value is reproduced exactly.  Stage2 is an exact D16/D32
  tie; lower-storage D16 is the correct deterministic oracle tie break.

These are analytic frontend schedule ratios.  The comparable K4-to-K8 gain is
1.337183263x for the oracle and 1.307079853x for bounded D8, despite doubling
nominal event lanes.  Eight weight-bank responses, eight accumulators,
producer/directory traffic, physical SRAM, frequency and power are absent.

## Required corrections and gates

M182's wall-cycle and selection results are correct, but its per-stage
`optimized_k1_over_*` and `m179_k4_over_*` fields are semantically invalid:
`enrich()` reuses global K1/K4 numerators inside every stage.  Remove them or
use stage-specific baselines before automated table export.

Keep 4.444592700x diagnostic-only: depths were selected on the reported
population.  Precommit bounded D={2,4,8,8} and replay held-out PAFT/sequences.
The next RTL should be two physical D8 ping-pong windows with 2/4/8/8 active
depth, eight fixed per-bank selectors, held output-block replay, exact partial
close, same-cycle release/refill and fail-closed malformed-directory handling.
VCS/SVA must prove conservation and backpressure; matched Synopsys DC must
compare fmax, area, sequential bits and throughput/area against M180 K4.

The stated 1,536/6,144-bit values are bitmap payload only, not total storage.
Also, 2.303104% is bounded/oracle minus one; oracle cycle reduction relative
to bounded is 2.251255%.  The upstream reproduction command uses an ephemeral
`/tmp` extraction path and should be replaced with deterministic archive
extraction or direct archive streaming.

## Evidence

- `independent_recompute_m182.py`: independent full-payload decoder and two
  independently written schedulers.
- `independent_recompute.json`: complete aggregate/per-stage D/K1/K8 ledgers,
  selections and exact M181/M179/M182 crosschecks.
- `m182_independent_hammer_review.json`: score, P0/P1/P2 findings and RTL gate.

