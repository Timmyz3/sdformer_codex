# M685｜M682-r5 native-TF32 capture fresh static hammer

## Verdict

**NO_GO_P1 — do not create the CPU preflight and do not consume the GPU one-shot.**

Score: **84/100**. Severity: **P0=0, P1=2, P2=0**.

The r5 repair correctly binds M680/M683, restores native cuDNN TF32, keeps the remaining recorded controls fixed, checks the frozen S00/D0 bitpack sentinel, and preserves fresh output/attempt state. It nevertheless fails static admission for two independent reasons.

## P1-1: incomplete adjacent seal recheck

The runner fully verifies the preflight double seal at lines 194–196, then parses `preflight.json`. Its adjacent second gate at lines 229–232 recomputes only:

- the SHA of `preflight.json`; and
- the SHA of the outer-seal file itself.

It does not rehash `SHA256SUMS` or the sealed members. An independent temporary-directory attack changed `RUN_COMPLETE.txt` after the first full check. Both r5 second comparisons still passed, while full double-seal verification failed. The runner would therefore create the attempt directory; only the later producer-side verification would reject the preflight. Since the contract says a failed attempt cannot restore authorization, this is a material one-shot-consumption bug.

Required repair: repeat both `sha256sum -c` checks immediately before the two external-root comparisons and `mkdir attempt`, with no semantic parser or unrelated command in between. Preserve the mutation as a negative regression.

## P1-2: resolver output is mislabeled as effective CuPy backend

The producer records `resolve_snn_backend(config) == cupy` and CuPy package version. That proves request resolution and installation only. SpikingJelly's setter changes only matching modules exposing a backend attribute. The frozen model has zero surviving PSN targets after the overlay and 105 `ATLIFTernaryPSN` modules; the latter execute `torch.addmm`. Consequently, r5 does not close M680's actual-backend receipt requirement.

Required repair: report the effective module inventory and implementation honestly—zero PSN backend targets, 105 `ATLIFTernaryPSN`, `torch.addmm`, source SHA and live matmul controls. CuPy may be recorded as installed/requested, not as the executed backend.

## Evidence that passed

- Four target SHA identities match the assigned roots.
- Python compilation, contract JSON parsing and runner shell syntax pass.
- Author static suite: **23/23 PASS**.
- Required inputs: **39/39** present, regular, non-symlink and exact-SHA matching.
- M680 and M683 review populations and both seal layers independently verify.
- `cudnn_allow_tf32=true`; deterministic algorithms, cuDNN deterministic, benchmark, CUDA matmul TF32 and CUBLAS controls are checked after config/model, before and after every sample, and at finalization.
- S00/D0 gate is fail-closed at 839586 ones, 3768414 zeros and packed SHA `ad2251f...`.
- Frozen M660-r4 attempt/failure evidence remains untouched and independently verifies.
- M682-r5 preflight, canonical output and attempt were absent throughout review.
- D1 candidate scrub tests cover early, middle and late failure phases.

No preflight, GPU, payload, performance simulation, RTL or EDA action was executed by M685.
