# M1239 — M1232/R11 checker/tests-only hardening

Status: **CHECKER-SOURCE GO for a fresh independent hammer only.**  Release
authoring, VCS/simv, EDA, GPU, remote work, TB/RTL/SVA mutation, and M1221
retry remain forbidden.

M1235 found no P0 defect in the exact R11 TB.  Its sole P1 was mechanical: the
checker accepted three destructive mutations to the legal-random path.  M1239
therefore keeps the candidate TB byte-identical at
`850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`
and changes only checker/tests.

The checker now requires:

- `random_request_window_active = 1'b1` before random service, preserving the
  per-window exact-one handshake counters;
- a positive `hold_cycles = 1 + ...` assignment and the exact
  `repeat (hold_cycles)` response-backpressure body;
- the post-retirement sampled-edge conjunction to retain the exact response
  count alongside weight/psum request counts.

Three new negative tests attack those exact constructs.  All prior tests are
preserved.  The resulting suite passes 18/18: one canonical positive and 17
negative mutations.  The canonical checker passes with no errors.

Exact identities:

- frozen R11 TB: `850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`
- hardened checker: `ccec195091bd79d8d24008ac9b1d4b2e6259a7c38b51cb695a17bff2678d5a94`
- hardened tests: `56c279d71e7fcf5350166f8e31dca010d2635de1aaf414df6c0d36c68e0b9f36`

This is source/checker evidence only.  It neither publishes a release nor
proves functional VCS, timing, cycles, performance, PPA, energy, system
speedup, or paper admission.
