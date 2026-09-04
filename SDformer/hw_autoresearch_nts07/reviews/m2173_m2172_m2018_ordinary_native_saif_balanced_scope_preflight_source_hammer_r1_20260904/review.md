# M2173 independent M2172 source hammer

## Verdict

**FAIL at 90/100; P0/P1/P2 = 1/0/0.  M2174 is not authorized.**
No license query, VCS, `simv`, SAIF acquisition, DC, PT/PTPX, ICC2, or GPU
job was run.

M2172 correctly repairs the balanced-SAIF ownership defect.  The independent
hammer parsed a sealed synthetic measurement with exactly 93,971 activity
records under one balanced `dut_ordinary` subtree, zero records outside it,
TX=0, exact conservation, and all eight critical cones toggling.  Eleven
independent wrong/duplicate/empty/unbalanced/out-of-scope/TX/conservation/
critical/seal mutations were rejected.  The author 42-test suite and static
parser/runner also reproduce, and the topology remains one direct
`SCHEDULE_MODE=0` frontend with no second axis.

The reset repair is still fail-open for minimal, ordinary diagnostic wording.
`has_reset_context` requires a second word such as `power`, `request`,
`activity`, or `counter` in addition to `reset` or `clear`.  Consequently 11 of
14 contract-derived probes escape, including:

- `Warning: reset failed.`
- `Error: reset denied.`
- `Warning: reset ignored.`
- `Warning: reset unsupported.`
- `Error: reset cannot complete.`
- `Warning: clear failed.`

Appending one of these lines to an otherwise valid runtime allows
`final_result` to assert `power_reset_acceptance.accepted=true` even though the
log explicitly says reset failed.  That violates the contract's normalized
failed/denied/ignored/unsupported/cannot/unable reset-or-clear gate and is P0.

The minimal successor should preserve the balanced hierarchy parser unchanged,
but treat `reset` or `clear` itself as the operation context before applying the
negative-semantic predicate.  It must add the minimal bypasses and accepted
success controls under a fresh source identity and pass another independent
source hammer.  M2172 must not be edited or executed, and M2174 remains absent.

This is a source-rejection result only.  It proves no production SAIF, mapped
activity, power, energy, speedup, or paper-citable PPA.
