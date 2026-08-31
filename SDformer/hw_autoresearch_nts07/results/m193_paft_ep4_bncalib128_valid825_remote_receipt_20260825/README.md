# M193 PAFT BN recalibration and valid825

The remote A800 run recalibrated all 78 BatchNorm modules from 128 evenly
spaced DSEC training samples.  Exactly 234 running buffers changed and every
non-BN tensor remained bit-identical.  The calibrated checkpoint then passed
the same 825-frame, 18-sequence running-BN validation identity as M162 with a
strict 0/0 checkpoint load and 105 ATLIF plus 12 attention modules.

The result rejects the candidate: AEE is 1.491473173, worse than the original
PAFT running-BN AEE of 1.469150671 and far worse than the non-foldable,
sample-statistic AEE of 1.309925157.  Spikes fall only 0.323620% relative to the
original running policy.  Therefore neither the calibrated checkpoint nor the
sample-statistic accuracy is promoted to hardware.

If PAFT is reopened, it needs training-time frozen-running-stat or explicit
BN-fold consistency and a matched non-PAFT control.  Current hardware work
continues without claiming a PAFT accuracy gain.
