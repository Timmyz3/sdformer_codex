# M518 r3 VCS pre-compile failure hammer r1

Date: 2026-08-27  
Verdict: `DIAGNOSTIC_CONFIRMED__R3_RESULT_NOT_ADMITTED__R4_STATIC_REVIEW_REQUIRED`

This review is production-read-only and failed-result-read-only. It did not run
the M518 runner, VCS, DC, Formality, PT/PTPX, or an open-source RTL tool. It did
not modify the candidate, the failed directory, or `docs/359`.

## Finding

The r3 one-shot stopped before compilation because the preliminary tool identity
query omitted `-full64`:

```text
M518 identity query:  ${task_vcs}/bin/vcs -ID
M518 compile command: ${task_vcs}/bin/vcs -full64 ...
```

The identity query returned exit 19. Its log says the launcher selected
`/opt/synopsys/vcs/V-2023.12-SP1/linux/bin/vcs1`, which does not exist. The
installed compiler is
`/opt/synopsys/vcs/V-2023.12-SP1/linux64/bin/vcs1`. Static inspection of the VCS
launcher confirms that `-full64` selects 64-bit mode.

The same-machine M519 differential supports this diagnosis: its runner invokes
`vcs -full64`; `compile.rc=0`, `sim.rc=0`, its link line uses `linux64` libraries,
and its simulator identifies itself as `V-2023.12-SP1_Full64`. This M519 evidence
is used only to establish the tool-architecture path. Its result directory has a
later failure marker, so this review does not elevate it to a complete M519
package.

## What the r3 directory proves

- Exact runner expected/observed SHA matched `09a24967...`.
- The automatic wrong-RTL negative returned the required exit 10; its member
  manifest and outer seal pass.
- Positive input SHA and sealed-spec preflights completed.
- The runner then wrote `FAILED_OR_INCOMPLETE_DO_NOT_CITE`, exit 19.
- There is no `compile.log`, `compile.rc`, `simv`, `sim.log`, `sim.rc`, assertion
  report, positive receipt, `RUN_COMPLETE`, or positive manifest.

Therefore the directory is diagnostic only. It contains no SystemVerilog compile
evidence, no VCS behavioral evidence, and no basis for V01--V20, numerical,
cycle, DC, performance, or paper claims. The failure also does not show an RTL,
SVA, or TB defect because those sources were never compiled.

The r3 one-shot authorization is consumed. Its result path now exists, and the
fail-closed runner would reject reuse of that path. Neither the old runner nor
the failed directory is authorized for another admission attempt.

## Minimum r4 repair

1. Preserve the existing RTL, SVA, TB, and filelist identities exactly.
2. Create a superseding r4 contract, a new result path, and a new runner
   identity/path. Do not overwrite, rename, or repurpose the r3 failure directory.
3. Use the literal identity query `${task_vcs}/bin/vcs -full64 -ID`; retain
   `-full64` on compilation and retain the automatic wrong-RTL exit-10 negative.
4. Bind all r4 paths and SHA identities fail-closed.
5. Obtain a new independent static review that authorizes exactly one literal r4
   runner SHA before execution.

This failure review does not authorize r4 execution, DC, Formality, PT/PTPX,
performance, energy, PPA, system speedup, or a headline.

At review close, `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
