# M496 foreign-idle simv resource-policy static hammer r2

Verdict: **STATIC NO-GO**. Score: **93/100**. P0: **1**. P1: **2**.

This was a source-only review. DC, VCS, simulation, Formality, PrimeTime, and
DSE were not launched. No production file or docs/359 was modified.

## Closure of the three r1 P0 findings

1. **Preflight consumption is closed.** K1's three-sample preflight runs in a
   unique temporary directory while both canonical run and attempt remain
   absent. A failure in this phase removes only the temporary logs. If failure
   occurs after canonical creation but before attempt publication, the EXIT trap
   moves the canonical run to a same-filesystem PID-unique prelaunch quarantine,
   restoring an absent canonical path and leaving attempt absent.
2. **Start/end identity is closed.** `m485_verify_all_inputs` includes the
   absolute runner, tool, both libraries, all RTL, filelist, SDC, TCL, contract,
   upstream VCS/cycle evidence, review seals, and docs/359. It runs initially,
   before and after every DC point, before receipt generation, and again after
   receipt generation. The initial `input_sha256.txt`, which also includes the
   runner, is rechecked at both final boundaries.
3. **Runtime OOM admission is closed.** The background monitor latches any
   nonzero failcnt, `under_oom`, or `oom_kill`; its return code is captured after
   DC wait. Each point requires DC rc=0, monitor rc=0, a fresh clear cgroup, and
   exact input identity before its local PASS. Final receipt preparation also
   requires a clear cgroup, and another clear check follows the Python step.

## State-machine checks that pass

- K1→K8→K1x8 remain strictly serial: the script waits for both DC and its monitor
  before returning from each point.
- EXIT trap captures the incoming status. Pre-attempt failures quarantine only a
  created canonical run; post-attempt failures retain the consumed canonical run
  and add `RUN_FAILED_OR_INCOMPLETE.txt`; complete success does neither.
- `.attempt_staging` and the canonical attempt directory are on the same
  filesystem, so the absent-target `mv` is a rename boundary immediately before
  first DC launch.
- Current canonical run and attempt paths are absent. The sole observed foreign
  `simv` still qualifies at the snapshot (`fangyl`, `Sl`, 0.0%, 110032 KiB) and
  is not modified or signaled.
- `bash -n` passes.

## Remaining P0

### P0: formal PASS artifacts are published before the last hard checks

The embedded Python writes the final receipt, canonical `RUN_COMPLETE.txt`,
`evidence_manifest.sha256`, and a valid outer seal. Only after those formal PASS
artifacts exist does the runner perform its last input rehash and cgroup check.
If either postcheck fails—or if Python fails after writing `RUN_COMPLETE.txt`
but before fully completing—the EXIT trap merely adds
`RUN_FAILED_OR_INCOMPLETE.txt`. A standard-named PASS receipt and, in the
postcheck case, a valid sealed PASS set remain on disk beside the failure marker.
This is not fail-closed for downstream consumers that locate `RUN_COMPLETE.txt`
or the sealed receipt without separately interpreting the trap marker.

Required repair: generate the root receipt, completion marker, and evidence seal
under non-admitted staging names/directory; run the final input/cgroup checks;
then atomically publish the complete PASS set and set `m485_complete=1` only
after rehash. Alternatively, the EXIT trap must atomically quarantine/remove all
standard PASS names before adding the failure marker. No post-publication hard
check may leave an admitted PASS set on failure.

## P1 findings

1. The attempt marker contains `ATTEMPT_CONSUMED.txt` and `identity.sha256` but
   has no inner/outer seal and is not rehashed at final completion. Its existence
   still enforces one-shot behavior, but its forensic contents are weaker than
   the run receipt. Seal it before the rename and rehash it at final completion.
2. Plain `mv source target` treats an unexpectedly existing target directory as
   a container and can place `.attempt_staging` beneath it. Concurrent canonical
   runners are already excluded by canonical `mkdir`, so this is a narrow
   external-race risk. Use `mv -T` (same-filesystem) or an equivalent
   no-replace/target-must-be-absent primitive and verify the two expected members
   at the canonical attempt root.

The sampled foreign-idle `simv` policy retains the previously disclosed
TOCTOU/lifetime-CPU limitation; it is not an exclusive reservation.

## Re-review gate

Repair only final PASS publication and, preferably, both P1 attempt-seal issues.
Do not change RTL, point order, compile effort, libraries, filelist, SDC, TCL,
or metric gates. Relock the runner SHA and obtain another static review before
DC launch.

Frozen docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
No PPA, power, system-speedup, or DATE-headline claim is admitted.
