# M518 matched Fixed/rank3 r2 DC launch-admission hammer

Date: 2026-08-27  
Review scope: admission-only, source-only; no EDA/VCS invocation  
Score: **100/100**; P0/P1/P2 = **0/0/0**  
Verdict: **admission valid for one future runner-gated r2 DC attempt; not runnable now**

## Outcome

The double-sealed admission JSON is internally consistent with every jq field
and live identity consumed by the frozen r2 runner. It authorizes only DC, with
`max_attempts=1`; VCS, Formality, PT and PTPX remain false. Root may retain this
admission as the sole authority for one future r2 attempt.

This is not a DC result and is not unconditional permission to launch. At the
review snapshot, one `simv` process was present. The runner's collision gate
would therefore reject a launch before resource sampling, work creation or
attempt consumption. The single resource snapshot happened to exceed the
numeric thresholds, but neither one snapshot nor this review substitutes for
the runner's three live samples and process-collision check.

## Double seal and immutable identity

- Admission JSON SHA256:
  `08df8b2b477ebbfe10999bce6e41580429b503980cdb47c3634d37d9238e7ee5`
- Member sidecar SHA256:
  `1c77c058832ad09baa766ba73234c70b289a5a7ae1a83624cd2222c4891163af`
- Outer sidecar file SHA256:
  `94334de319820ea4d13c44a5052bb011ec0139f2a993e3ecf3f7e0c438a37bf6`

Both sidecar checks pass. The runner does not parse these sidecars; instead the
caller must pin the independently reviewed admission JSON SHA. That exact hash
immutably covers the independent r2 review block as well as every authorization,
identity, runtime, fairness and claim-boundary field.

## Runner jq and live-hash audit

The admission has the exact status
`AUTHORIZED_ONE_M518_MATCHED_FIXED_RANK3_R2_DC_ATTEMPT` and the exact
authorization tuple required at runner lines 71–78. All 13 runner-read identity
keys are present, 64-hex and equal to the live files:

1. r2 author contract, runner and Tcl;
2. common filelist and SDC;
3. slow and fast TSMC28 DBs;
4. Fixed VCS result/review outer seals;
5. rank3 VCS result/review outer seals; and
6. r1 static-review verdict and outer seal.

The hard-coded DC launcher and both RTL identities also match live files.
`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
All prerequisite member manifests and outer seals verify recursively.

The independent r2 static review is additionally bound by path, verdict SHA,
member-manifest SHA and outer-seal SHA, and its P0/P1 counts are zero.

## r1 permanent exclusion and one-shot state

The sealed r1 verdict still says `NEEDS_REVISION__R1_LAUNCH_NOT_AUTHORIZED`,
with root-r1-admission false and DC false. The r2 runner rechecks that exact
verdict and seal before resource sampling. No r1 launch admission exists.

The r2 canonical result and r2 attempt sentinel were absent during this audit.
The r2 runner forbids canonical override, refuses canonical/attempt/work/
quarantine collision, performs all identity and seal checks before its three
resource samples, and consumes the one-shot sentinel only after all preflight
checks pass.

## Resource and current-state separation

One read-only diagnostic snapshot at `2026-08-27T18:25:51+08:00` reported:

- commit headroom: 69,525,896 KiB (threshold 67,108,864 KiB);
- MemAvailable: 413,923,008 KiB (threshold 134,217,728 KiB);
- SwapFree: 57,266,172 KiB (threshold 33,554,432 KiB);
- cgroup failcnt/under_oom/oom_kill: 0/0/0; and
- live `simv`: 1, so the runner collision gate was not clear.

These values are ephemeral diagnostic evidence only. A future invocation must
independently pass the runner's collision gate and three fresh samples. If a
preflight fails, no attempt is consumed; if the first DC launch is reached,
the sealed attempt sentinel makes that the sole allowed attempt.

## Exact future caller environment

Only after all external activity is clear and the operator intentionally
chooses to spend the one shot, the sole reviewed invocation is:

```bash
M518_MATCHED_EXPECTED_DC_RUNNER_SHA256=05ada3ea4e2b653262f2693602eab83c3cc75ea7af35fc4e501f9da2a481147e \
M518_MATCHED_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=08df8b2b477ebbfe10999bce6e41580429b503980cdb47c3634d37d9238e7ee5 \
/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m518_matched_fixed_rank3_logic_only_r2_exact_sha.sh
```

Do not set `M518_MATCHED_DC_RUN_DIR`; the runner explicitly forbids it. Do not
reuse the r1 runner or admission path.

## Claim boundary

Admission validity does not admit DC completion, STA, area, timing,
throughput/mm2, macro-inclusive PPA, power, energy, trained-rank3 accuracy,
system speedup, paper-ready PPA or a headline. A future raw result remains
uncitable until a separate independent receipt hammer passes.

