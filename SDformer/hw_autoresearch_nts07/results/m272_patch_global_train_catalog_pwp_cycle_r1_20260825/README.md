# M272 train-only global catalog replay on Patch/early Conv

M272 replays the admitted M77 DSEC-train-only global K16 pattern catalog on
the six exact-binary Patch/early Conv3x3 modules in the M51 trace.  Catalog
selection never reads these ten evaluation samples.  Every payload SHA,
active-bit count, and M222 receptive-field source contribution is checked.

All 60 records, ten samples, and six modules are individually faster than the
strong one-source/96-output-lane bit-sparse reference when each PWP has a
separate one-cycle 96-lane path.  Aggregate module cycles fall from
1,883,717,407 to 1,749,942,022, or 1.076446x.  If the PWP shares the existing
96-lane path and occupies two cycles, the result is only 1.003822x.  Natural
vector work is reduced 1.081547x; all 2,970 measured phases remain compute
bound.

The result is deliberately a weak-opportunity boundary.  A global catalog
transfers correctly but is not selective enough to justify a dedicated wide
PWP datapath.  The M267 Hamming-tree materializer can eliminate stored PWP
payload, but it does not change these arithmetic limits.  Further work should
use a disjoint train-only patch-specific catalog or change the activation
distribution in training, then replay this same frozen evaluator.

This is an exact trace-driven isolated-module cycle model.  Patch INT8 numeric
equivalence, RTL/VCS/DC, complete Patch Embed, energy, system speedup, paper
PPA, and headline admission remain false.
