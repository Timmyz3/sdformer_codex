# M163 r1 cross-hidden-lane moment correction

M163 r1 is revoked for dynamic-BN semantics even though its sealed VCS run
passed.  The RTL and TB both reduced `2 x 16` input values into one scalar
moment stream while the contract called those 16 lanes hidden channels.
`no_running` BN1 requires a different mean and variance for every hidden
channel, so a self-consistent scalar TB did not test the production grain.

The correct interface keeps 16 independent `sum[47:0]` and `sumsq[55:0]`
states.  Each accepted beat issues 32 squares but updates each hidden lane with
only its two temporal samples.  One `T=10` tile therefore increments the shared
per-lane count by 10, not 160.  Multiple spatial tiles with the same tag
continue to accumulate into the corresponding lane.

The r1 VCS pass and r1 DC area/timing are not admissible for the corrected
module.  M163r2 must rerun both commercial flows and independently compare all
16 moment lanes.
