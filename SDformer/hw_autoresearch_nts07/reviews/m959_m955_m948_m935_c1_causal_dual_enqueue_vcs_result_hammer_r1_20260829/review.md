# M959 | M955 C1 functional VCS result hammer

Verdict: `GO`, score 97/100, P0=0/P1=0/P2=2. Mandatory qualifier:
**foundry UNIT_DELAY functional negative-attack PASS with exactly one expected
SVA failure; not a zero-assertion-failure or clean-SVA regression.**

The M955 result recursively validates against manifest `654aef80...` and outer
seal `ad526c07...`. Its receipt, runner, contract, RTL/SVA/TB/checker identities
match. The attempt marker is consumed, no active work/quarantine remains, and
no same-UID VCS or simv process is active.

Compile and simulation followed the runner's immediate `PIPESTATUS==0/0`
fail-closed path: only after those gates, token checks and receipt generation
could `RUN_COMPLETE`, sealing and the canonical result move occur. Compile and
sim logs contain no fatal/error execution line. PASS, causal, reset, exact-match,
metadata and P2 tokens each occur exactly once; all required attack and coverage
counts pass.

The complete sim log has exactly one assertion failure:
`ap_candidate_after_active` at 10,168,500 ps. The next line identifies the
illegal `row_candidate_relation_ok` predicate and the following line is the
unique `M923_WRONG_PARENT_PHASE_CORRECT row=1 parent=63 relation_ok=0
capture_watchdog=4` token. TB lines 1764–1815 deliberately force that dead
wrong-parent relation before cached-context capture, verify it is illegal, then
require `protocol_error`. PASS reports `wrong_parent=1` and `attacks=6`. No
other assertion failure exists, so this failure is the intended negative attack,
not an unexplained design failure.

P2: numeric compile/sim return codes are not serialized in the receipt and are
proven through fail-closed publication control flow; the expected negative-test
assertion exception is not encoded in the original receipt and therefore must
remain attached via this hammer. Two normal VCS-generated symlinks resolve
inside the sealed result; no external symlink exists.

This admits functional UNIT_DELAY behavior only. Timing, workload cycles,
speedup, PPA, power, energy, system, headline and paper claims remain false.
