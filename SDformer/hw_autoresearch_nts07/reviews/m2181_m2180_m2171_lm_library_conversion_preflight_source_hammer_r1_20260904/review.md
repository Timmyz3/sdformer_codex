# M2181 independent M2180 source hammer

## Verdict

**FAIL, 92/100, P0/P1/P2 = 0/1/0. M2182 is not authorized.**

The proposed repair correctly moves the failed M2170 operation from ICC2 to
regular Library Manager, pins the real `lm_shell_exec` and StarRC `Milkyway`
executables, sets and reads back `lib.setting.milkyway_exec` before the one
non-overwriting `generate_frame_from_mw`, and contains no design import or P&R
command. The frozen M2171 failure fingerprint and all source identities match.

One evidence-boundary defect blocks execution. The process monitor labels only
known EDA names as unexpected, and the checker requires the expected processes
but does not reject every additional descendant. An independent mutation added
a fourth connected `/usr/bin/sleep` child below the exact `lm_shell_exec`; the
checker accepted it and returned four identities. This violates the requested
"wrapper plus exactly one actual LM executable plus exactly one Milkyway child"
census and is therefore P1.

## Checks that passed

- M2171 and the M2180 author receipt are exhaustive and double-sealed.
- The source contract sidecar and outer seal match.
- All five source SHA-256 identities match the contract.
- `lm_shell`, `lm_shell_exec`, and `Milkyway` are executable regular,
  non-symlink files at the exact paths and hashes in the contract.
- The option set, readback, Gate 2, and conversion occur in strict order.
- There is exactly one frame conversion and no overwrite, `create_lib`, RTL
  import, placement, routing, timing, area, or power command.
- Seven isolated subdirectories and fresh result/attempt/work/lock checks are
  present. The current M2182 filesystem and running-process census is empty.
- The existing source suite passes its 1 native mutation and 12 process-tree
  mutations. It does not include the extra-child mutation above.
- This review ran zero LM, EDA, license, GPU, or P&R actions.

## Required repair

Create a new source identity; do not edit or execute frozen M2180. The monitor
and checker must classify every observed descendant and reject any identity
other than the root bootstrap/wrapper, exactly one `lm_shell_exec`, and exactly
one `Milkyway`. Add the connected fourth-child mutation to the source suite,
then repeat an independent source hammer. No M2182 preflight, license query, or
retry is authorized by this review.

