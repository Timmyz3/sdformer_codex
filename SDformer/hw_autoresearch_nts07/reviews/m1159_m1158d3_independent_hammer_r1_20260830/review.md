# M1159 independent hammer of M1158D3

## Overall assessment: ready to share for the fast-kill decision only

The author result is reproducible, and the preregistered decision remains `NO_GO_RTL__ALL_FOUR_1P20_GATE_FAILED`. This hammer did not import or execute the author analyzer. It unpacked the frozen D3 bitpack, rebuilt the K3/S2/P1/OP1 destination topology, counted modulo-8 bank occupancy at every destination and timestep, and recomputed the same-width A1 ledgers from sealed M712 rows.

The exact replay returns 96,760,057 contributors and 17,288,869 actual bank-conflict groups. The latter is 1.052797x the M712 conflict-free optimistic count. All ten timestep rows and every author cycle field match.

| Width | D3 local ratio | Fixed all-four ratio | Gate |
|---|---:|---:|---|
| 128 bit | 1.354761x | 1.153212x | all-four FAIL |
| 96 bit | 1.351047x | 1.151846x | all-four FAIL |

The fixed policy is D0-D2 A1-OSG plus D3 static-weight-fit, selected by the compile-time 13-of-16 capacity predicate. There is no sample, sequence, density, miss-rate or runtime oracle. D3 may remain a local CPU support point, but no RTL, VCS, DC, decoder headline or system speedup is authorized.

## Calculation and identity checks

- M699, M712, M718, M1157, M1158 and the author receipt pass inner and outer seals; the M1158 contract triple passes.
- Six fail-closed attacks pass: payload bit flip, dropped timestep, policy oracle, M712 baseline mutation, ratio mutation and 12-entry capacity mutation.
- The 13-entry and 16-entry logical ledgers occupy 196,258 and 237,730 bytes. Physical macro capacity, ports and timing remain unverified.
- The frozen 96-bit fixed-phase scan charges 422,400 probes/timestep. A valid-edge minimum would be 420,722; using it changes the all-four ratio only to 1.151874x, so the decision is insensitive to this conservative boundary charge.

## Required claim boundary

This is one H67_ep35 sequence/sample, includes diagnostic D1, and is not decoder-population-complete or final-checkpoint-bound. The 1.35x value is D3-local support only. The citable decision is that the four-call fixed mixture fails 1.20x and therefore does not proceed to RTL or EDA.
