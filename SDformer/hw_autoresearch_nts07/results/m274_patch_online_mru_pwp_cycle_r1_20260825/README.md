# M274 online one-entry PWP memo screen for Patch/early Conv

M274 tests a leakage-free temporal-reuse alternative after the M272 global
catalog transferred only weakly.  Each tap/channel partition starts with an
invalid one-entry memo.  Zero and singleton masks bypass it; an expensive-mask
miss follows the strong one-source/96-lane add-only path while a separate
96-lane signed12 builder captures the PWP, and the next matching expensive mask
uses one PWP update.  No training or evaluation profile selects the tag.

The exact ordered replay covers all 60 M51 records, ten samples and six binary
Patch/early Conv3x3 modules.  It reproduces all M222 receptive-field source
contributions.  Only 22,501,871 of 463,457,618 eligible rows hit, a 4.855217%
hit rate.  Natural vector work improves 1.018693x and the matched-boundary
module-cycle model improves only 1.017588x.  All records and modules are
slightly faster, but the frozen 1.5x RTL-promotion gate fails.

The proposal is therefore stopped before RTL.  Its extra 96-lane builder and
1,169 bits of logical memo state are not worth pricing for this gain.  Together
with M272's 1.076446x wide-static and 1.003822x shared-path results, this closes
cache/catalog-only Patch optimization on the frozen activation distribution.
Reopening Patch requires a disjoint train-only patch-specific distribution or
a different dataflow, not a larger unpriced cache.

This is an exact trace-driven isolated-module negative screen.  PWP numeric
RTL, cache macro, VCS, DC, energy, complete Patch Embed, system speedup, paper
PPA, and headline admission remain false.
