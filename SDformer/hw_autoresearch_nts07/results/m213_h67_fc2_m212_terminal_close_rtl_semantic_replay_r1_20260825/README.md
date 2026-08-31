# M213 frozen-H67 FC2 replay on M212 terminal close

M213 rehashes and replays all 120 frozen H67 FC2 records through the M212
recurrence calibrated against Synopsys VCS.  It covers 5,580,000 tokens,
36,480,000 raw beats, 18,869,376 nonzero descriptors, 6,523,707 compact
windows, and 143,894,510 events.

The isolated sparse-frontend result is 90,388,767 cycles.  Compared with the
matched M210 control recurrence, M212 saves 795,772 cycles (1.008803882x):
572,258 / 103,844 / 108,749 / 10,921 in stages 0--3.  Stage 0 reaches the old
analytic lower bound exactly.  Stages 1--3 retain a combined 281,490-cycle
gap, now localized to paired-window done-fence/load boundaries.

The terminal hint closes 1,149,920 partial tails early; 795,772 of them become
an actual one-cycle saving under the frozen schedules.  The older serial
analytic baseline divided by M212 is 1.268360149x, but it is mixed fidelity and
must not be presented as a matched RTL, complete-FC2, FFN, physical, system, or
headline speedup.  `docs/359` was not modified.
