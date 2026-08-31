# M247 PAFT versus no-PAFT paired valid825

M247 compares the five-epoch PAFT arm and its matched no-PAFT continuation
control on the same ordered 825-frame, 18-sequence validation cohort.  The
training configurations are identical after removing labels and the declared
PAFT block; PAFT targets only the four bottleneck Conv operators.

The hardware-foldable running-BN policy is primary.  PAFT changes AEE from
`1.477617736` to `1.469150669` (`0.573022%` lower) and total spikes by
`-0.115337%`.  AEE improves on 430/825 frames and 12/18 sequence means.  This is
a small single-seed direction, not an accuracy headline or statistical claim.

The no-running policy is diagnostic only because it uses sample-dependent BN
statistics.  The next gate is an INT8 export of the PAFT bottleneck weights and
exact replay of the M248 PAFT source trace through the Conv cycle model.  This
receipt admits no Conv/system cycle speedup, energy or paper PPA.
