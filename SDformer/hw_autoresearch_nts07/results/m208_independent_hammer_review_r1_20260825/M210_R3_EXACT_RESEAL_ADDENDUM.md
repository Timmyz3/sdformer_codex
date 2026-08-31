# M210 r3 exact-input reseal addendum

Read-only verdict: **PASS_EXACT_INPUT_RESEAL; P0 repair admitted within the
tested isolated-FC2 scope.**

The r3 seal closes the one reproducibility condition left open by this review:

- all 11 result-directory SHA entries pass when checked from the sealed
  directory;
- all nine exact-input SHA entries pass against the current workspace;
- the current bank48 TB is exactly `ff7e8284...`, matching the receipt;
- Synopsys VCS accepts two legal 48-event packets, emits 192 groups, and
  completes one token in 195 header-to-done cycles;
- the separately recorded software recurrence also returns 195 cycles;
- the bank48 SVA cover matches twice, with no assertion failure;
- the inherited 256-case calibration remains 0-mismatch.

Therefore the r2 `EXACT_INPUT_RESEAL_REQUIRED` restriction is removed.  M210's
six-bit bank-sum repair is admitted for the tested bank48 regression and its
inherited isolated-token calibration.

This is a scoped admission, not a restoration of M207/M208/M209 evidence and
not a complete hardware claim.  The following remain false until separate
evidence exists: replay of this review's independent 152-vector suite on M210,
M210-hashed frozen-H67 replay, complete FC2, physical/energy results, system
speedup, and headline speedup.

`docs/359` remained unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
