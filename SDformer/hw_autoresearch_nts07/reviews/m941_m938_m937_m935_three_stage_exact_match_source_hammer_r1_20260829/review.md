# M941 | M938/M937/M935 exact-match verification-repair source hammer

## Verdict

`PASS_M941_M938_SOURCE_HAMMER`, score 96/100.  Verdict: `GO` for preparation
of a separate exact-SHA VCS release; M941 itself does not issue that release.

All four M937 P1 verification gaps are closed in the frozen M938 source.  The
M935 RTL remains byte-identical at its pinned SHA, the M938 contract binds the
real M937 artifacts correctly, and the M938 static checker independently
passes.  No VCS, DC, EDA, GPU, remote, network, or license command was run.

## M937 identity correction and M938 binding

The on-disk M937 identities are:

- `review.json`: `34d1f64ba97da8209f1c9fd0976c082e880ea336b276375c2d5d10b2f0c5be78`;
- `review.md`: `a4c4ae5cd5b24820ff419caced795b0ae08d0cea1194698b444110ed0cba52e8`;
- `SHA256SUMS`: `182ece3a8b389ab5d6d495c86f1890a53d6c9aa45057b81058a5236543b10b39`;
- outer-seal file `SHA256SUMS.seal.sha256`:
  `752b28ab04502efeb7436443960c2aa840e28c8a0a76c709986b932548691097`;
- the outer-seal file declares the `182ece...` manifest hash.

Thus `182ece...` is the manifest hash, not the outer-seal file hash.  The M938
contract distinguishes `manifest_sha256`, `outer_seal_file_sha256`, and
`outer_seal_declared_manifest_sha256` correctly.  Its M937 status
`PASS_M937_SOURCE_HAMMER`, verdict `REPAIR_BEFORE_VCS_RELEASE`, and counts
P0=0/P1=4/P2=3 match the real `review.json`.

M938 candidate hashes also match the contract: supplemental SVA
`eb20ffb5...`, TB `6b5d58bd...`, checker `3265da87...`, inherited execution
SVA `ad89adc7...`, and frozen M935 RTL `e834b524...`.  `docs/359` remains
`dedde7ce...`.

## Four repaired P1s

1. **Opposite-bank overlap:** SVA asserts
   `match_g_valid && exec_active |-> match_g_bank != exec_bank` and covers the
   same bank-distinct expression.  The bind supplies `exec_bank_q`.  The TB
   treats same-bank overlap as an error, increments its counter only for a
   distinct bank, requires that counter nonzero in the normal coverage fatal
   gate, and prints it in the final PASS token.

2. **Row-63 same-bank READY:** the unqualified `|| exec_active` escape is gone.
   The next SVA sample requires stable `match_g_bank` and the dynamically bound
   state of that bank to equal `BANK_READY`.  NBA inspection remains consistent:
   R63 writes the directory and READY on edge 66; launch can observe it only on
   edge 67.

3. **Dynamic F/G/R reset:** `reset_during_match_stage` targets F row0 before G,
   G row0 before its R commit, and G row63 before terminal R/READY.  Each case
   checks the seed before reset, asynchronous control/bank clearing, no stale
   directory write or READY/execute/done leak while reset and after release,
   and a fresh recovery task.  All three calls occur at the start of `initial`;
   their counters must each equal one before normal tests and appear in the
   final PASS token.

4. **External bank/epoch oracle and distinct overlap payload:** ownership is
   derived from public accepted-prep and task-done handshakes.  DUT
   `bank_epoch_q` is compared as an observation but never selects the reference
   slot.  F and G tags are checked against the external owner.  Epoch 1 uses
   directed masks, while overlapping epoch 2 uses seed `32'h9380_0002`, so a
   swapped bank tag cannot pass through identical directories.

## Static and syntax audit

The checker reran with:

```text
PASS_M938_THREE_STAGE_EXACT_MATCH_REPAIR_SOURCE_STATIC algorithm_rows=262208 metadata_bits=283 ports_unchanged=true execution_tail_byte_exact=true inherited_m919_sva_exact=true m937_bound=true bank_distinct_overlap=true row63_ready_bank_qualified=true external_bank_epoch_oracle=true distinct_overlap_masks=true reset_F_G_R63_present=true source_only=true vcs=false dc=false timing=false speedup=false ppa=false energy=false system=false headline=false
```

Source inspection found no new NBA ownership conflict or reused SV reserved
identifier.  The asynchronous reset sampling uses inactive-edge entry plus a
delay before inspection; reset cases cannot backfill normal matcher coverage.
Dynamic hierarchical unpacked-array accesses and the dynamic bank-state bind
are plausible SystemVerilog but remain VCS compile facts.  The future release
must pin the M938 checker and contract themselves, compile both SVA files, and
require exactly one final PASS token with all new counts nonzero.

## Issues and claim boundary

- **P0=0:** no source semantic, ownership, reset, drain, or frozen-identity
  failure found.
- **P1=0:** all four M937 repair blockers are closed at source level.
- **P2=3:** commercial compile must adjudicate bind/hierarchical syntax; the
  launcher must pin its own checker/contract; committed directory and
  parent-live dumps remain a desirable post-VCS independent closure.

`GO` means only that a separately reviewed exact-SHA VCS attempt contract may
now be prepared.  Functional VCS, timing, cycles, speedup, area, PPA, power,
energy, full-system performance, and paper/headline admission all remain false.
