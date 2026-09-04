# M922 — M919 C1 metadata-pipeline VCS failure forensic

## Verdict

`PASS_FORENSIC_REVIEW`, while M919 remains
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`.  The quarantine is recursively sealed and
the failure is reproducible from the frozen source semantics without another
EDA run.

The primary failure is a **phase-invalid inherited TB attack**, not evidence of
an arithmetic/queue RTL failure, an SVA failure, or a foundry `UNIT_DELAY`
observation-window problem.  The attack was copied from the non-pipelined M528
corpus.  It waits until row 1 is already visible at the issue interface, then
changes row 1 in the backing directory.  M912 has already snapshotted that row
into `active_ctx_*`, and explicitly excludes active/next rows from the backing
directory tournament.  Consequently the forced backing bits are no longer an
authoritative input to the current issue or to `fault_condition_w`.

M923 is authorized only as a fresh additive source identity with an adapted,
phase-correct negative test, followed by a new independent source hammer and a
new one-shot release.  M919 itself must not be edited or rerun.

## Exact failure path

At simulation time 7,729,500 ps, `expect_fault` exhausted its 20-cycle
watchdog on the fourth attack and issued:

`protocol attack not detected: wrong parent and illegal dead-parent relation`

The frozen attack does the following:

1. Loads the directed task for epoch 400.  Row 1 has mask `16'h0003` and its
   legal parent is row 0 (`16'h0001`).
2. Waits for `issue_request_valid && issue_request_row_id == 1`.
3. Forces only `directory_q[0][1][21:16]` to parent 63 and
   `parent_live_q[0][63]` to zero.
4. Expects sticky `protocol_error` within 20 clocks.

That stimulus triggered the old M528 implementation because its
`current_directory_w` was a live combinational read of
`directory_q[exec_bank][current_row]`; its fault predicate therefore saw parent
63 and the false live bit immediately.

M912 is different by construction:

- `issue_request_parent_id` comes from `active_ctx_parent_q`, not the backing
  directory.
- `active_ctx_relation_ok_q` was computed and registered when the row was
  selected.  The backing mutation does not change it.
- The row tournament excludes both `active_ctx_row_q` and `next_ctx_row_q`, so
  the now-forced directory word is not re-read as a candidate while row 1 is
  active.
- The relevant RTL fault predicates are
  `active_ctx_valid_q && !active_ctx_relation_ok_q` and
  `next_ctx_valid_q && !next_ctx_relation_ok_q`.  Both cached relation bits
  remain legal in this attack.

Thus the attack never creates the condition its label claims.  It mutates stale
backing state after the authoritative context boundary.

## What did and did not pass

- VCS compilation completed successfully with the foundry model and
  `+define+UNIT_DELAY`; no compile fatal/error marker was found.
- The normal directed/random phase reached all emitted gates before the attack:
  14 normal covers, two-cycle-fill count 7, 20 consecutive-distinct-read
  witnesses, 189 response-identity checks, and all held-final recovery minima.
- The first three attack tasks must have returned successfully, because the TB
  invokes the failing wrong-parent attack fourth and `expect_fault` is fatal on
  any earlier miss.
- No SVA assertion failure is present.  The SVA block observes and proves
  invariants; it is not the generator of the sticky protocol fault.
- The final PASS token is absent, so functional VCS verification is false.
  `simv` returned success after VCS translated `$fatal` to `$finish`; the runner
  then correctly exited 22 at its missing-PASS gate and quarantined the run.
- The failure does not involve a parent SRAM response and is independent of
  `UNIT_DELAY` sampling latency.

## Minimal additive repair for M923

Keep the M912 RTL unchanged for the first repair.  In a new TB identity, inject
the malformed row-1 directory relation **after `execute_busy` rises but before
either active or next metadata context is populated**.  A deterministic form is:

1. `wait (execute_busy); @(negedge clk_core);`
2. Assert `!dut.active_ctx_valid_q && !dut.next_ctx_valid_q`; fail closed if the
   phase contract is not true.
3. Force row 1's backing parent ID to 63 and its parent-live bit to zero.
4. Prove row 1 is captured with a false relation predicate, then require sticky
   `protocol_error` within the bounded watchdog.
5. Preserve the other five attacks and every normal/P2/held-final minimum.

Do not “repair” the test by forcing `active_ctx_relation_ok_q = 0`; that would
only force the detector input and would not prove malformed metadata is
recognized.  Also do not silently add a full backing-directory reread to the
functional ready/valid cone.

If post-snapshot backing-storage corruption is intended to be a separately
claimed fault model, it requires a new RTL identity with parity/ECC or explicit
active-context consistency checking and its cost must be measured.  That is a
different resilience claim, not required to preserve the inherited protocol
attack class.

## Integrity and claim boundary

The quarantine's 111 regular-file manifest entries and its outer manifest seal
both verify.  Two tool-generated internal symlinks exist; the runner's
`find -P -type f` seal intentionally covers every regular file and neither
symlink is followed.  The manifest has no omitted or extra regular file.

`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
No EDA or license query was run by M922, and no M912/M919 source was modified.
No M919 timing, cycle, speedup, PPA, energy, or paper claim is admitted.
