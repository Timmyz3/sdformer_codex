# M547 / M533 r6 fresh independent source-static hammer

Verdict: **FAIL, 68/100, P0/P1/P2 = 0/3/1.** This was a fresh read-only source review. It ran no runner, VCS, simv, open-source HDL simulator, DC, Formality, PT/PTPX, CPU/GPU experiment, or remote job. The new result/attempt identity remains absent.

## What passed

- TB r4 differs from TB r3 on exactly four lines in `oracle_pack_row12`: the illegal local identifier `packed` and its three references become `packed_row`. Module name, oracle behavior, stimuli, functional/coverage/P2 tokens, and attack tokens are otherwise byte-identical. A conservative declaration-context keyword scan found no remaining reserved identifier.
- Core r2, SVA r2, macro adapter, macro binding plan, TB r4, source contract, candidate, M544 review, and `docs/359` recompute to the requested hashes. All supplied JSON/review/request/handoff seals verify, strict JSON parsing succeeds, and `bash -n` accepts runner r6.
- The old consumed r3 partial still contains exactly the eight M544-frozen regular files with all eight exact hashes unchanged; it still has no `simv`, `sim.log`, member manifest, or outer seal. The new result path, source-static review before this report, candidate-hammer review, final release, and final-release-hammer review were absent.
- The four-stage launch gate is fail-closed before `mkdir`: this report does not authorize VCS, and the other three future release members remain absent.
- Ordinary post-`mkdir` compile/sim/functional/resource/collision failures do route through the EXIT trap to a failure receipt and member/outer seals. The receipt carries phase, runner/child status, monitor state, resource/collision presence, regular-file inventory/hashes, and the listed functional-source hashes.

## Blocking findings

### M551-P1-01 — a sealed PASS can still be followed by a nonzero runner exit

Runner lines 903–907 seal `RUN_COMPLETE.json` and set `TERMINAL_SEALED=1`, then execute an unguarded final `echo`. If that write fails, or INT/TERM/HUP arrives in this post-seal window, cleanup observes `TERMINAL_SEALED=1` and preserves the PASS receipt while the wrapper exits nonzero. Independently, cleanup line 202 runs `rm -rf` under `set -e`; a cleanup failure can replace the captured runner status after either terminal receipt is already sealed. Therefore the claimed one-to-one mapping between process exit and sealed terminal state is not proven for **every** post-`mkdir` exit.

Minimum repair: remove every fallible operation after success sealing, perform temporary-directory cleanup before the success receipt, explicitly make the success seal the last fallible operation, and close/ignore catchable signals across the final seal-to-`exit 0` critical section. Cleanup must preserve the captured return code even if temporary cleanup fails, or write a separately modeled cleanup failure before sealing.

### M551-P1-02 — the recursive seal silently excludes VCS-generated symlinks

The receipt inventory labels symlinks as `symlink_forbidden`, but never rejects them and records neither link target nor a link-object hash. The shell `find -P ... -type f` member manifest also omits them. This is not theoretical for this tool: the existing results tree contains 238 `csrc` symlinks (562 symlinks total), including VCS archive links into `simv.daidir`. A successful VCS build can therefore be sealed PASS while mutable, unhashed filesystem members remain inside the result identity.

Minimum repair: either fail on every symlink before success/failure sealing, or freeze a canonical symlink manifest containing path and `readlink` target and bind that manifest in `SHA256SUMS`. If tool build directories are intentionally out of scope, move them outside the sealed result or delete them before the last seal and state that policy explicitly.

### M551-P1-03 — the terminal receipt is not self-binding to the exact executable launch identity

`immutable_source_hashes` contains the source contract, TB, core, SVA, macro adapter, and binding plan, but omits the runner itself, VCS executable, foundry Verilog/model manifest, source-static review, launch candidate, candidate hammer, final release, and final-release hammer. Those identities are checked before launch, yet none is copied or hashed into the sealed result receipt. A later receipt-only audit cannot reconstruct the exact-SHA tool/release identity from the terminal evidence.

Minimum repair: add path+SHA entries for the runner, VCS binary, foundry `.v` and asset manifest, source-static review, candidate, candidate hammer, final release, and final-release hammer to both success and failure receipts; bind the live hashes used by the launch gate, not merely paths.

### M551-P2-01 — future launch does not re-freeze all eight old-partial members

The static state is currently clean: all eight M544 hashes match. At future launch, however, runner r6 rechecks only `compile.log` plus selected absences. It verifies the sealed M544 review but does not compare the other seven live partial members or exact file count against `partial_result_sha256`. Thus its claim that the old partial “remains untouched” is weaker than the source package promises.

Minimum repair: compare the exact eight-name/eight-SHA closed inventory to M544 immediately before the new attempt, and reject any extra, missing, symlinked, or changed member.

## Decision

Source-only PASS is denied. `launch_now=false` remains effective, no result was created, and no candidate hammer, final release, final-release hammer, or VCS attempt is authorized. Author a new runner identity and repeat the fresh source-static hammer; the next review must reach 100/100 with P0/P1/P2 = 0 before proceeding.

Claim boundary: no functional VCS, RTL verification, recurrence, speedup, PPA, energy, full-network, system, or paper-headline claim is established by this review.
