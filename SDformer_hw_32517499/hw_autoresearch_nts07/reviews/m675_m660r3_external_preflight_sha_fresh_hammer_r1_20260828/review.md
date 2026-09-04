# M675 — M660-r3 external-preflight-SHA fresh hammer

## Verdict

**NO-GO, 94/100, P0=0 / P1=1 / P2=0. Do not run or consume the GPU one-shot.**

M660-r3 substantially repairs M672 P1. Both reviewed preflight SHA roots are
now mandatory, malformed/missing roots fail, nested seals are checked first,
and wrong receipt or wrong outer SHA independently exits 41 while both attempt
and output remain absent.

One narrow admission race remains: the two identities are checked only once.

## P1-1 — no second identity check at `mkdir(attempt)`

The runner performs the correct nested-seal and two-SHA check at lines
193–202. It then launches a separate semantic-check Python process at lines
203–223 and creates the attempt at line 225. It does not repeat the receipt and
outer-seal digests after that process and immediately before the attempt.

This leaves a TOCTOU window in which the preflight can be replaced after the
reviewed identities pass. A replacement preserving status, contract SHA,
parameter name, and zero load counts passes the semantic subset, yet the
one-shot is consumed with an identity different from the reviewed object.

M672 explicitly required both a post-seal comparison and a second comparison
immediately before `mkdir(attempt)`. M660-r3 implements only the first.

Minimum repair: retain the current check and repeat the identical two digest
comparisons after the semantic Python process returns and directly before the
attempt directory is created. Either mismatch must exit 41 with attempt/output
absent. No algorithm, producer, checkpoint, GPU execution, or EDA change is
needed.

## Passed evidence

- All M674 target SHAs and nested seals independently match.
- Runner is a non-symlink executable regular file (`0775`).
- M660-r2 + M665 + M673 tests: **44/44 passed**.
- Wrong receipt with correct outer: exit 41, attempt/output absent.
- Correct receipt with wrong outer: exit 41, attempt/output absent.
- Fresh real H67 CPU exact-load: missing/unexpected `0/0`, exact four
  ConvTranspose modules, wrapper `Spiking_neuron`, leaf `ATLIFTernaryPSN`,
  theta bytes `b3ff7f3f`, exact match with the frozen preflight receipt, and no
  forward.
- Canonical output and attempt remain absent.
- docs/359 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

No command is published because M674 requires P1=0. No payload, performance,
RTL, EDA, energy, PPA, system-speedup, or DATE-headline claim is admitted.
