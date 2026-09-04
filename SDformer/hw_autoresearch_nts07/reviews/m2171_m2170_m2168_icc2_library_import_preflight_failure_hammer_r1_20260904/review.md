# M2171 independent M2170 failure hammer

## Verdict

**PASS failure diagnosis, 100/100; P0/P1/P2 = 0/0/0.**  M2170 is one
uniquely consumed, exhaustively sealed, non-retriable failure.  It proves no
library compatibility and cannot be cited as a physical/P&R result.

The failure is exact: the approved layout, output-absence and execution-
contract gates completed; one license query and one `icc2_shell -no_init`
session ran; Gate 1 completed; ICC2 then returned
`Error: unknown command 'generate_frame_from_mw' (CMD-005)`.  The shell exited
42, the outer fail-closed runner exited 1 and quarantined the run.  Gates 2--6,
frame generation, design-library creation and all P&R stages never occurred.

## Execution census

- Attempt and quarantine seals are exhaustive: 1 and 15 members respectively.
- The license log contains one ICCompilerII response; the exact approved runner
  has one LMUTIL site.
- The process census observed one real ICC2 `dgcom_exec`, no conversion child,
  and a completed process-monitor receipt.
- Budget is exactly license=1, top-level ICC2=1, P&R=0, retry=false.
- No canonical M2170 result exists; the single attempt and single failure
  quarantine are now permanent.

## Minimum legal successor

Do not retry `icc2_shell`.  The installed documentation places
`generate_frame_from_mw` and every other required library command in the LM
command set.  The minimum successor is a fresh source identity using the
regular executable `/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell` (the
`icc2_lm_shell` path is only a symlink alias) and pinning its real
`lm_shell_exec` child.

Before frame conversion, the successor Tcl must set and read back
`lib.setting.milkyway_exec` to the installed executable
`/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway`.  The local LM
manual explicitly requires this option before commands operating on Milkyway
FRAMs.  These identities make the successor technically justified; they do
not pre-prove that conversion will succeed.

Recommended fresh chain: M2176 source, M2177 independent source hammer, one
M2178 LM-library-only attempt, then M2179 independent result hammer.  Preserve
the same isolation, stale-output, execution-contract, process-tree and no-P&R
gates, updated to pin `lm_shell`, `lm_shell_exec`, and `Milkyway`.
