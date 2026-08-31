# M835 receipt-blind fresh source hammer: M832 NO-GO

## Verdict

`NO_GO_M832_SOURCE_CANDIDATE__P1_1__PUBLICATION_BOUNDARY_REPAIR_REQUIRED`, score 92, P0/P1/P2 = 0/1/0.

The requested PASS100 is withheld. M832 may not author a true release and its formal runner may not be invoked.

## Blocking finding

M832 correctly pins the parent/results directory FDs, rebinds the current pathname before and after private-stage creation, writes the three receipt members with exclusive `openat`, and uses `renameat2(RENAME_NOREPLACE)` on the same results FD. The last publication boundary is still open in two ways.

First, the last current-path dev/inode assertion occurs before the final prefix scan and `renameat2`; there is no rebind after publication. An independent hook replaced the current `results` pathname inside the rename wrapper and then called the exact original rename. The helper returned `PASS_M832_DIRECTORY_FD_BOUND_ATTEMPT_CONSUMED`, but the current canonical pathname had no attempt. The attempt existed only inside the detached old directory, while a protected artifact in the replacement directory remained unchanged.

Second, the stage members and seals are checked before the prepublication hook. Immediately before publication, M832 verifies only the stage directory inode. An independent hook overwrote `attempt.json`; the helper again returned PASS and published the tampered attempt, which then failed `verify_sealed`.

These are one P1 root cause: the consumed-attempt publication boundary is not jointly bound to the current results pathname and the exact sealed stage contents. A formal runner can therefore return from consumption without a valid canonical sealed attempt, leaving exactly-once provenance unproven.

## Positive evidence

- Exact request, candidate, contract, source package, M833 handoff, M831 authority, parent closure, all member sidecars, and applicable double seals pass.
- Python 3.10 and Python 3.6 pass compile, self-test, M832 12/12, M828 12/12, M809 9/9, M815 10/10, candidate validation, and the exact M832→M828→M819→M809 zero-row traversal.
- Persistent protected-prefix regular files, directories, symlinks, dangling symlinks, FIFOs, sockets, and concurrent injections reject without clobber; post-stage rejection cleans only the self-created stage. Stage/attempt collisions reject without clobber. Close wrong prefixes remain nonmatches. A transient object already absent at the boundary is accepted, matching the explicit no-history claim.
- Release preflight and the resource gate precede the sole atomic helper; the started latch and production call follow it. No shell path-based attempt `mkdir`/`mv` exists.
- Runtime remains 40 M686 plus 120 M699, T10, A1/K1x8/K8, 96 lanes, 245760 B, Acc24, 3 ns, and 192 B/cycle. D1 stays charged/nonheadline; only K8 versus equal-service K1x8 may headline.
- No true release, formal attempt, result, failure, log, or production row was created. No production, VCS, EDA, license, GPU, or remote task ran. `docs/359` remains exact.

## Required repair

Create a new additive identity that keeps the stage FD open through publication, revalidates its exact three-member population and seals immediately before rename, and rebinds the current results pathname after publication. If that postpublication rebind fails, remove only the exact self-created published attempt through the pinned FD using its recorded inode and exact member allowlist, then fail without touching the replacement directory. Repeat both attacks in a fresh source hammer. Do not edit or invoke M832.
