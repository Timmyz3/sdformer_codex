# M560 / M533 r8 fresh independent source-static hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0.** This was a fresh read-only source, contract, and control-flow review. It ran no runner, VCS, simv, HDL/EDA tool, CPU/GPU experiment, or remote job. It did not create the new result identity, touch or seal the old partial, author a candidate hammer or launch release, or modify `docs/359`.

## M558-P1-01 closure

The r8 runner closes the only r7 blocker. At lines 661–668, `INT`, `TERM`, and `HUP` are ignored before the final exact-result absence recheck and remain ignored through `mkdir`. On success, the shell publishes `RESULT_CREATED=1` before restoring the original failure traps. A catchable signal can therefore no longer enter cleanup in the consumed-but-unowned state identified by M558.

On `mkdir` failure, r8 first captures the return code, restores all three traps, leaves `RESULT_CREATED=0`, and checks that no result exists before reporting a pre-attempt failure. If a competing process created the path, cleanup still neither removes, claims, nor seals that unowned identity. The critical section contains no workload or functional-source change.

## Regression closure

- The exact r7→r8 diff contains the expected identity/schema/release-path updates, adds the M558 failure review to admission and terminal provenance, and makes one non-identity control-flow repair: the attempt-publication critical section above. No other runtime mechanism, resource threshold, workload, functional gate, or terminal behavior changed.
- `bash -n` passes. Strict duplicate-key/non-finite JSON parsing and all supplied inner/outer seals pass. Request, handoff, contract, candidate, runner, prior failures, functional sources, foundry assets, VCS binary, and `docs/359` cross-bind to their declared SHA256 values.
- TB r4 remains the four-line mechanical `packed`→`packed_row` repair. Core r2, SVA r2, macro adapter, and macro binding plan are byte frozen.
- The old consumed r3 partial still contains exactly the eight M544 plain regular files and all eight exact SHA256 values. The runner rechecks this closed inventory both during provenance admission and immediately before attempt creation.
- r7 terminal closure remains intact: PASS symlinks bind path, raw target, internal resolved target, bytes, and content SHA through the sealed artifact inventory; the full live path/type/content set is reverified before sealing. Both terminal kinds bind the live source, tool, foundry, release, and failure-provenance chain. Preflight cleanup precedes PASS sealing; after the verified success seal only shell-state assignment, EXIT-trap removal, and `exit 0` remain.
- Before this review was authored, the r8 result, this source-static review, candidate-hammer review, `launch_now=true` release, and final-release-hammer review were absent. The prospective candidate remains `launch_now=false`.

## Decision

Source-static admission passes. The candidate hammer may now be authored and independently reviewed, but **no VCS attempt is authorized yet**. A passing candidate hammer, separate `launch_now=true` release, and 100/100 final-release hammer are still mandatory before the unique VCS attempt.

Claim boundary: source-static control-flow closure only; no functional VCS, RTL verification, recurrence, speedup, PPA, energy, full-network, system, or paper-headline result is established.
