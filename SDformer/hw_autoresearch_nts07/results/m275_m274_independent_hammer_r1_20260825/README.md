# M275 independent hammer review of M274

Verdict: **96/100, freeze the M274 online-MRU candidate before RTL**.  There
are no P0 or P1 findings.  The producer seal verifies, and an independent
implementation that does not import the producer analyzer rehashes all 60 raw
bitpacks (645,120,000 bytes) and replays 1,969,920,000 ordered mask rows across
2,970 tap/channel partitions.

Every M222 receptive-field source contribution and every producer per-record
work/cycle field matches exactly.  Synthetic attacks also confirm the intended
state semantics: zero and singleton rows do not replace the memo, equal
expensive masks separated only by those bypass rows hit, different eligible
masks replace in MRU order, and no hit crosses a partition boundary.

The independent totals are:

- `1,774,268,587` bit-sparse vector operations;
- `1,741,710,204` online-MRU vector operations;
- `22,501,871 / 463,457,618 = 4.855217%` eligible hits;
- `1.018693341x` natural vector-work opportunity;
- `1,883,717,407 / 1,851,159,024 = 1.017588107x` analytical module-cycle
  opportunity.

All records, samples and modules are slightly faster in that model, but this
does not rescue the candidate.  It saves only 32,558,383 cycles and realizes
5.185% of the savings needed for the frozen 1.5x gate.  Even before charging
the extra 96-lane signed12 builder, it is 595,347,420 cycles above the 1.5x
target.  Nonnegative builder, dependency, area, timing, or energy cost can only
make the result worse, so stopping before RTL is robust.

Two wording corrections remain.  `1.017588x` is an optimistic ordered-trace
cycle upper bound, not an executable/VCS-calibrated schedule, because the
builder-ready dependency is not modeled.  Also, M272 and M274 freeze the tested
global-K16, online-one-entry-MRU, and current Patch cache-only workstream; they
do not prove every possible cache policy ineffective.  Any reopening should
first demonstrate a materially different policy whose analytical upper bound
crosses 1.5x.

No RTL, DC, energy, complete-Patch, system, PPA, or headline admission is made.
No open-source RTL flow was run, and docs/359 was not modified.
