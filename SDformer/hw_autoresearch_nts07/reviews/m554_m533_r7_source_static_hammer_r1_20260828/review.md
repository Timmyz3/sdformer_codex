# M554 / M533 r7 fresh independent source-static hammer

Verdict: **FAIL, 90/100, P0/P1/P2 = 0/1/0.** This was a fresh read-only source/control-flow review. It ran no runner, VCS, simv, HDL/EDA tool, CPU/GPU experiment, or remote job. It did not create the new result identity, touch or seal the old partial, author a candidate hammer or launch release, or modify `docs/359`.

## What passed

- `bash -n` accepts runner r7. Strict duplicate-key/non-finite JSON parsing passes for the request, handoff, source contract, launch candidate, M544 review, M551-labelled review, and macro binding plan. All supplied inner/outer seals verify.
- TB r4 differs from TB r3 on exactly four lines: the illegal local identifier `packed` and its three references become `packed_row`. Core r2, SVA r2, macro adapter, macro binding plan, and all functional/coverage/attack tokens remain frozen at the requested hashes.
- The old consumed r3 partial still contains exactly the eight M544-frozen plain regular names and all eight exact SHA256 values, with no extra member. Runner r7 calls the same closed-inventory verifier once during provenance admission and again immediately before the new attempt gate.
- The M551 PASS-tail defect is otherwise repaired: preflight cleanup precedes success sealing; INT/TERM/HUP are ignored across the success seal; after the verified final seal only shell-state assignment, EXIT-trap removal, and `exit 0` remain. Cleanup preserves the captured runner return code except the explicitly modeled unsealed-zero translation and terminal-seal failure.
- The artifact inventory binds every path. A PASS symlink binds its raw `readlink` target, canonical in-result target, target byte count, and target SHA; external, broken, directory-target, or special objects cannot produce a PASS. The live path/type/content set is reverified before terminal member sealing.
- Both terminal receipt kinds bind the live runner, VCS binary, foundry manifest/Verilog/DB, source contract/static review, launch candidate/candidate hammer, final release/final hammer, TB/core/SVA/macro/binding plan, and M544/M551/M547 provenance.
- Before this review was authored, the new result, this review, candidate-hammer review, `launch_now=true` release, and final-release-hammer review were absent. The candidate remains exactly `launch_now=false` with a closed prospective one-VCS/one-simv authorization and zero authorization for all other run classes.

## Blocking finding

### M558-P1-01 — catchable signal can consume the attempt between `mkdir` and ownership-state publication

Runner r7 line 647 executes external `mkdir -- "${RESULT_DIR}"`, and only line 648 sets `RESULT_CREATED=1`. INT/TERM/HUP remain caught by `signal_exit` throughout this two-command boundary. Bash may dispatch a pending catchable signal after `mkdir` returns successfully and before the following shell assignment. In that state the result directory exists and the attempt is consumed, but cleanup line 398 sees `RESULT_CREATED=0`, skips failure-receipt creation, and exits with an unsealed empty/partial result identity.

This is a post-`mkdir` ordinary catchable-signal path, so it directly contradicts the source contract and request claim that every ordinary post-attempt exit is double sealed. It also reintroduces a terminal atomicity gap adjacent to the M551-P1-01 repair, even though the later PASS critical section is now correct.

Minimum repair: make attempt creation and `RESULT_CREATED=1` one catchable-signal critical section. For example, ignore INT/TERM/HUP before the final absence check and `mkdir`; on a failed `mkdir`, restore the signal traps and fail pre-attempt; on success, set `RESULT_CREATED=1` before restoring the traps. A new runner/contract/candidate identity and fresh source-static hammer are required. The cleanup condition may additionally use a safely established attempt-owner token, but must not claim or seal a result created by another process.

## Decision

The required 100/100 threshold is not met. Candidate hammer is denied, `launch_now=false` remains effective, and no VCS attempt is authorized. The four original M551 findings are materially repaired, but the new attempt-publication race must be closed under a new exact identity before this chain can continue.

Claim boundary: no functional VCS, RTL verification, recurrence, speedup, PPA, energy, full-network, system, or paper-headline result is established.
