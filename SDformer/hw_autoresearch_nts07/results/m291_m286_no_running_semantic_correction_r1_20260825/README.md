# M291: M286 no-running semantic correction

M286's G7/G8 negative screening remains valid, but its `927` FFN zero-path
count came from M160's rejected constructor-default running-stat path.  The
frozen evaluator uses current-batch `no_running` BatchNorm.  Under that policy,
zero FC1 input produces zero sn2 activations, while the complete dynamic-BN2
branch remains nonzero at all 44,160 audited values.  An exact FFN bypass must
therefore model dynamic BN2 and the complete branch, not synthesize the old
running-stat constant.

The sealed M286 artifact is retained unchanged.  This overlay revokes only the
misclassified `927` field and binds the M160 and M290 independent evidence.
G7 and G8 remain main-axis NO-GO.  No accuracy, cycles, RTL, PPA, power, energy,
system-speedup or headline claim is admitted here.
