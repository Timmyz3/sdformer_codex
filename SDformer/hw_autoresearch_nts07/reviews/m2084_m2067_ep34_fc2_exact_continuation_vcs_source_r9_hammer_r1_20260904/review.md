# M2084 — M2067 R9 owner-safe source hammer

Verdict: **PASS, 99/100; authorize exactly one fresh, no-retry, 960-slot Synopsys VCS campaign.**

## What changed and why it is sound

R8 was externally interrupted after 210 parser-valid slots, and its duplicate-launch path exposed an ownership race in failure publication. R9 fixes that infrastructure defect without changing the R8 parser, filelist, zero-aware testbench, continuation wrapper, C2 scheduler, adapter, fixtures, or oracle.

The R9 ordering is fail-closed:

1. verify the pinned authority chain read-only;
2. acquire the nonblocking R9 owner lock;
3. acquire the shared same-UID EDA queue lock;
4. reject every canonical or private R9 namespace residue;
5. create an owner-bound attempt containing PID, random nonce, and runner SHA;
6. perform one license preflight, one VCS compile, and serial slots `0..959`;
7. atomically publish a double-sealed success or owner-only failure without replacement.

Failure publication independently reopens `owner.json` and requires all three owner fields to match the current process. Static negative tests confirmed that a lock loser and a nonce-mismatched process publish nothing, while the true owner can atomically publish one exhaustive double-sealed quarantine. Residual private stages, unsealed extra files, and replacement of an existing destination are rejected.

## Frozen workload and budgets

The unchanged R8 parser revalidated 30 frozen sources and fixtures covering 960 workloads, 2,400 row/chunk records, and 1,843,200 integer checks per axis. R9 inherits zero R8 logs. Its hard budgets are one `lmstat`, one VCS compile, 960 serial `simv` invocations, and no automatic retry.

## Admission boundary

This review ran no VCS, license query, GPU job, or other EDA tool. It admits source and one execution only. It admits no RTL cycles, speedup, full-FC wall time, system speedup, same-area result, power, energy, real-checkpoint-weight result, or paper claim. Even a successful sealed R9 campaign remains `PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE` until a separate result hammer verifies all 960 logs and the aggregate.

`docs/359_DATE终局冻结_20260813.md` remains SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
