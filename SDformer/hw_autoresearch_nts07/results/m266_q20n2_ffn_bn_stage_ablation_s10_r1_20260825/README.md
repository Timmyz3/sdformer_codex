# M266 Q20N2 dynamic-BN stage/position ablation

This milestone compares six isolated balanced-Q20/two-Newton dynamic-BN subsets
against the same frozen M263 FP reference on the first ten frames of
`zurich_city_09_a`.  It is a sensitivity experiment, not a trained
mixed-precision network.

All six subsets pass the local output-error bound (`max < 1.13e-4`) but fail the
paired AEE regression gate (`<= 0.25%`): stage3 `+0.4981%`, stage2 `+0.5800%`,
stage1 `+0.7819%`, stage0 `+1.0728%`, BN2 `+1.7849%`, and BN1 `+2.2993%`.
Stage and BN-position subsets overlap and their effects are visibly non-additive;
these deltas must not be summed or treated as causal attribution.

The hardware decision is to stop lowering the dynamic-BN coefficient precision.
The next admissible direction is a shared exact/high-precision moment and
reciprocal service, or algorithm-side training/calibration using the deployed
recurrence.  There is no RTL, VCS, DC, valid825, speedup, energy, system, PPA, or
headline admission in this milestone.

`m266_q20n2_ffn_bn_stage_ablation_analysis_r1.json` contains the fail-closed
source identities, exact module populations, paired metrics, and boundaries.
