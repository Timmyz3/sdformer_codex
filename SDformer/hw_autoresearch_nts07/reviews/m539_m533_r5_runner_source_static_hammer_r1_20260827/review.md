# M539 / M533-r5 fail-closed runner source-static hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0.** This was a fresh independent read-only source-static hammer. It did not execute the candidate runner, VCS, simv, Icarus, Verilator, DC, Formality, PT, PTPX, a CPU/GPU experiment, or remote work; it created no result/attempt directory. Static PASS alone does not authorize VCS.

## Closure of the sealed M537 findings

1. Every embedded Python validator is invoked as `python3 -I`. A nonempty `PYTHONOPTIMIZE` is rejected before semantic validation, and every schema, status, `launch_now`, exact closed ten-key authorization, SHA, score, and P0/P1/P2 decision uses an explicit `require(...)->RuntimeError` gate. No security decision uses Python `assert`.
2. The runner double-seal-checks the frozen source-static review and then hard-checks its exact member SHA `0e0b38901c2c1f380e4500a4253b9d2174424d2e6881295b1f66a226bf1caf4c`. It likewise binds the sealed M536 and M537 FAIL reviews before consuming their required next-gate semantics.
3. Runtime monitoring is fail-closed. The monitor writes each heartbeat through a temporary file plus atomic `mv`; the parent requires a live, at-most-three-second-old heartbeat before compile, after compile, and immediately before the final request. After the compile or simv child ends, the parent creates the final-request marker. The monitor takes one additional synchronous sample of numeric session and `user.slice` failcnt/OOM/usage fields, checks failcnt against the prelaunch baselines and all OOM fields against zero, writes an exact atomic acknowledgment, and exits. The parent consumes the actual `wait` status, requires exit code zero, rejects a violation marker, and requires exactly one final acknowledgment and one final-sample row. No success path discards the monitor wait status.

## Preserved release, collision, resource, and attempt semantics

- The non-circular release chain remains `launch_now=false` candidate -> independent 100/100 candidate hammer -> `launch_now=true` final release. Both candidate and final release must equal the exact ten-key authorization dictionary: one VCS run and zero Icarus, Verilator, DC, Formality, PT, PTPX, CPU, GPU, and network/remote runs.
- The same-UID `/proc` classifier and two collision scans remain before result creation. The cgroup-v1 prelaunch gate still requires three samples, two seconds apart, with 128 GiB MemAvailable, 32 GiB SwapFree, 32 GiB commit headroom, stable session and `user.slice` failcnt, and zero OOM state.
- At the pre-output snapshot, the r5 static review, M540 candidate, M540 candidate hammer, M540 final release, and r3 result identity were all absent. After this review is written, the remaining M540 release artifacts and r3 result path are still absent, so the runner still fails before preflight/result creation. Compile or simv failures occur only after the sole atomic result-directory creation point and therefore consume the attempt.

## Frozen identity and decision

The exact r5 runner is `24c833dc41e922f9568dc502fc7a5dc3335eccb7be0d45dd4bf2e9e26ccc941f`; the repair contract is `968cdad1219e6b8074f9190b479ad490df58b6ef09293ec233f45e61eebb7ed3`. Frozen top r2, macro adapter, binding plan, SVA r2, TB r3, source contract, and `docs/359` recompute to their required hashes; `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The bounded r5 runner source therefore passes this static gate. A separately authored and independently hammered M540 release candidate plus a final launch release are still required. This review establishes no functional correctness, recurrence, speedup, PPA, energy, full-network result, or paper headline.
