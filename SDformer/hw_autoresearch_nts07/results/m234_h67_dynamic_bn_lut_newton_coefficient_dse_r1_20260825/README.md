# M234 H67 dynamic-BN LUT+Newton coefficient DSE

M233 provides 220,800 checkpoint-bound FFN BN coefficient pairs. The selected integer path uses a hardware-addressable segmented 64-entry UQ1.18 rsqrt LUT, one RNE Newton step, and 20-bit Q16 invstd/alpha/offset outputs. It has zero rails on the captured population.

The maximum coefficient-only affine deviation over every captured per-channel input interval is below 0.0018. This uses an exact endpoint bound for the approximate-minus-reference affine function, but it does not prove ATLIF threshold/event equivalence or BN2 residual accuracy.

The candidate schedules five multiplies onto one scalar multiplier and targets first-result latency and output interval of 16 cycles, matching the corrected M232 service target. Those timing values require RTL/VCS/DC. The sum/sumsq-to-mean/variance divider remains outside this module.
