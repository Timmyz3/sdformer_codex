# M193 independent hammer review

Score: **91/100**

Verdict: `PASS_NEGATIVE_RESULT__FREEZE_CURRENT_PAFT__REJECT_HARDWARE_ACCURACY_PROMOTION`

The M193 execution and negative decision are trustworthy.  All 10 M193
manifest entries and all 9 nested M162 entries pass SHA-256 verification.  The
128 calibration filenames are unique, cover all 18 DSEC sequences, and have
zero exact filename overlap with the frozen 825-frame validation population.
Both the source and calibrated checkpoint load receipts are strict 0/0, and
the fail-closed script permits changes only to the 78 running means, 78 running
variances and 78 batch counters.  It reports 234 changed BN buffers and zero
changed non-BN tensors.

Independent CSV/profile recomputation confirms 825 unique files in 18
sequences, the exact frozen validation order, AEE `1.4914731733726732`,
`82,460,606,306` spikes, and `460/825` frames lower than the original PAFT
running-BN result.  The frame count does not rescue the candidate: mean AEE is
`0.0223225024` (1.5194%) worse than original running BN and `0.1815480160`
(13.8594%) worse than sample-statistic BN.  Spikes fall only 0.3236% versus
original running BN.

The present PAFT accuracy line should therefore stop.  The recalibrated
checkpoint must not be selected, and the non-foldable no-running AEE must not
be promoted as hardware accuracy.  AEE is an accuracy metric, not an
acceleration factor; neither the AEE nor spike-count delta is cycle, throughput,
energy, or physical evidence.

The calibration probe is a useful rejection test but not proof that every BN
recalibration can never work.  It uses one batch-1 cumulative pass over 128 of
7345 train files, weighted by concatenated file order rather than balanced by
sequence.  Also, all BN modules run in training mode together, so later-layer
moments are collected under upstream batch-stat normalization instead of the
final all-running-stat deployment path.  Any reopening needs training-time
deployment/fold consistency, a matched non-PAFT running-BN valid825 control,
and exported folded/quantized equivalence.

The imported package omits both large checkpoints.  The code and receipts bind
their hashes and fail closed, but an independent local tensor-by-tensor diff is
not reproducible from this package alone; importing checkpoints or a tensor
digest ledger remains P1.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
