# M216 matched-DC recovery audit

This directory does not conceal or replace the fail-closed r1 parent result and
does not run Design Compiler again.  The immutable r1 parent remains
`FAILED_OR_INCOMPLETE_DO_NOT_CITE` with exit code 40 because its contract
incorrectly required the two post-mapping sequential-cell counts to be exactly
equal.

Both exact-SHA Synopsys DC subruns completed independently.  The r2 audit
verifies every entry in each subrun's evidence manifest, preserves the parent
failure receipt, and compares the mapped sequential instance-name sets.  K1
and K8 share 2,770 sequential instances; the only K8-only instances are the
three upper bits of `group_source_count_q`.  They are constants for K1 and were
legally folded by DC.  No queue, dual-D8-window, tag, done, stall, or external
group-lane storage is absent from K1.

At the matched 3 ns, ideal-clock, ZeroWireload, zero-macro point, K1 is
20,436.696076 um2 and K8 is 20,587.392080 um2.  K8 therefore costs 0.737379%
more logic area.  Both setup and hold checks pass, but this remains a
logic-only pre-macro ablation, not complete-FC2, physical, system, paper-PPA,
or headline evidence.
