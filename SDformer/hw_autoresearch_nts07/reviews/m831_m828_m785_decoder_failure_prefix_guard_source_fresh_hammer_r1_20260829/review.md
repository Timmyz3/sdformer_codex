# M831 receipt-blind fresh source hammer: M828 NO-GO

## Verdict

`NO_GO_M828_SOURCE_CANDIDATE__P1_1__DIRECTORY_BINDING_TOCTOU_REPAIR_REQUIRED`, score 92, P0/P1/P2 = 0/1/0.

The requested PASS100 is withheld. M828 may not author a true release and its formal runner may not be invoked.

## Blocking finding

M828 opens the results directory with `O_DIRECTORY|O_NOFOLLOW` and performs two samples on one FD, but the dev/inode token is compared only to later `fstat` calls on that same FD. It is never rebound to the current `results` pathname before return. The shell then closes that FD and creates the attempt stage through the pathname.

The independent attack renamed the clean results directory aside between the samples, created a replacement directory at the same pathname, and placed a canonical failure-prefix artifact in the replacement. Both samples kept seeing the old clean FD, so the guard returned `PASS_M828_STABLE_FAILURE_PREFIX_ABSENCE` while the current results pathname contained the protected artifact. A matching create-then-unlink wholly inside the yield interval could also return PASS.

This is a P1 because it violates the explicit TOCTOU fail-closed contract and leaves the M825 pre-consumption provenance invariant unproven.

## Positive evidence

- Exact request, source package, handoff, M825 authority, all source sidecars and applicable double seals pass.
- Python 3.10 and 3.6 pass compile, self-test, M828 12/12, M809 9/9, M815 10/10, candidate validation, and the exact M828 to M819 to M809 zero-row traversal.
- Persistent regular, directory, symlink, dangling symlink, FIFO, socket, and between-sample matching injections reject without clobber. Close wrong prefixes remain accepted as nonmatches.
- Frozen runtime remains 40 M686 plus 120 M699, T10, A1/K1x8/K8, 96 lanes, 245760 B, Acc24, 3 ns, and 192 B/cycle. D1 remains charged/nonheadline; only K8 versus equal-service K1x8 may headline.
- No true release, formal attempt, result or failure artifact exists. No production, VCS, EDA, license, GPU, or remote task ran. `docs/359` remains exact.

## Required repair

Create a new additive identity that verifies the current directory pathname and performs attempt-stage creation relative to the same verified directory FD, or provides an equivalent atomic boundary. It must reject directory replacement and concurrent matching artifacts without clobbering them. Then repeat a receipt-blind source hammer. Do not edit or invoke M828.
