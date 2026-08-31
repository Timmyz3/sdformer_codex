# M199 independent hammer review

Score: **90/100**, conditional pass for the abstract DSE only.

An independent decoder and causal FIFO/max-plus scheduler reproduced every
sealed aggregate, per-stage value, and all nine S/F points across all 120 FC2
payloads.  The S4/F2 backlog-four bound was also reached and verified with
65,536 exhaustive 16-beat patterns; 216 named and 108,000 randomized point
checks covered full, partial, zero, tail, and mid-scan window boundaries.

S4/F2 is the right standalone RTL screen: its 92,464,838 abstract cycles are
1.239882x versus S1/F1/W1, and it retains 97.5749% of S4/F4 abstract throughput
while halving descriptor write width.  These remain replay-model values, not
physical or complete-FC2 speedups.

The P0 is implementation proof.  A four-entry post-emit reservoir needs a
stable six-candidate merge/bypass at the worst burst, explicit behavior under
descriptor backpressure, and residual-lane handling when a window closes in
the middle of a raw4 group.  Run Synopsys VCS/SVA attacks and matched 3 ns DC
for S4/F2 and S4/F4 before admitting any area/timing advantage.
