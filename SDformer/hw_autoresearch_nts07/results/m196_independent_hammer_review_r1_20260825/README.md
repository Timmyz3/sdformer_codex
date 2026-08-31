# M196 independent hammer review

Score: **88/100**. Verdict: pass the negative F=1 finite-buffer DSE;
packed ingress is an investigation, not an admitted speedup.

The review imports none of the production M172/M192/M195/M196 analyzers.  An
explicit little-endian decoder replays all 120 frozen H67 FC2 payloads and an
independently authored vector recurrence exactly reproduces the sealed M196
aggregate and every stage.  A scalar/vector randomized attack covers 60,000
additional recurrence cases.

The buffer-only rejection is robust.  Under the sealed recurrence B2 is
0.996919x, B3 is 0.999792x and B4 is only 1.002237x versus W1/B2.  Under a
stricter one-edge registered fill/service and reuse sensitivity model the
same points are 0.996311x, 0.999151x and 1.002155x.  Stage 0 is the major
regression and only stage 3 can use more than two buffers at this geometry.

M196 therefore correctly stops buffer-only promotion and points to descriptor
arrival bandwidth.  It does not establish that a packed ingress is cheap or
fast: the producer currently assumes a preindexed nonzero stream, and there
is no integrated scanner/compactor, dual-write queue, SRAM response, Acc24,
stale-response quarantine, or stalled edge-accurate VCS replay.  Those are
P0 gates before any packed-ingress, complete-FC2, physical, FFN, system, or
headline speedup claim.
