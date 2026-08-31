# M267 independent hammer review of M266

Verdict: **88/100, sensitivity evidence GO with claim corrections**.  There are
no P0 findings.  The six first-ten paired AEE/DSEC-Fl/spike tables, ranking,
target counts and hook calls independently recompute.  Every tested subset
fails the `+0.25%` AEE gate; no accuracy-safe approximate subset is admitted.

The actual non-additivity evidence is the residual against frozen M263 all24.
All24 regresses AEE by `+1.108332%`; the four pairwise-disjoint stage runs sum
to `+2.932801%` (`-1.824469` percentage-point residual), while disjoint BN1 and
BN2 runs sum to `+4.084177%` (`-2.975845` point residual).  Stage2 having more
modules but a smaller delta is only non-monotonic sensitivity, not by itself a
proof of additivity or causality.

All 27 remote payload rows, 34 local rows and 14 M263 rows verify.  A relocated
clean replay is byte-identical to producer SHA `e33451cd...`; a mutated
`stage0/per_frame.csv` is rejected before output creation.  The evaluator and
config rehash locally, but the raw ep35 checkpoint is absent, so its `4f33...`
identity remains remotely receipted rather than locally reconstructed.

The main P1 is wording: the experiment rejects the tested Q20N2 coefficient
approximation but does not admit a shared exact moment/reciprocal hardware
engine.  That is a design recommendation until service demand, buffering,
utilization, integrated RTL and Synopsys evidence exist.  At the tested subset
granularity, all 24 FFN BN coefficient paths should stay exact/high precision;
finer module selection or QAT/calibration recovery remains unknown.

No DC, valid825 or docs/359 modification was performed.
