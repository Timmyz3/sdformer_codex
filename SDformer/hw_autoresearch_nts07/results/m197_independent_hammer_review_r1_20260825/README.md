# M197 independent hammer review

Score: **86/100**. Verdict: the sealed M197 arithmetic is an exact and
well-bounded screen of a **precompacted nonzero-descriptor stream**. It is not
an admitted scanner, RTL, physical, complete-FC2, FFN or system speedup.

The independent checker imports none of the production analyzers. It decodes
all 120 frozen payloads, checks their SHA/extent/popcount, runs 48,000 random
scalar-versus-vector recurrence attacks, and exactly reproduces every F/B
aggregate and per-stage wall-cycle count. The F2/B2 decomposition closes at
1.056447x packing-only and 1.037961x pair-fusion increment, for a combined
1.096550x versus the wider-baseline-mismatched legacy point.

The blocking issue is the nonzero oracle. Only 18,869,376 of 36,480,000 raw
96-bit beats are nonzero (51.7253%). Supplying two compacted descriptors per
cycle therefore requires 3.8666 raw beats per cycle on average before burst,
tail and stall losses. A real approximately 384-bit scanner/stable compactor,
finite queue, tagged SRAM response path and matched Synopsys VCS/DC evidence
are P0 before any performance admission.
