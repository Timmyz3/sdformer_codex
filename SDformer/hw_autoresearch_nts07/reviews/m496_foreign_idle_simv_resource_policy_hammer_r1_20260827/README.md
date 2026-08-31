# M496 foreign-idle simv resource-policy static hammer r1

Verdict: **STATIC NO-GO**. Score: **81/100**. P0: **3**. P1: **3**.

This was a source-only review. DC, VCS, simulation, Formality, PrimeTime, and
DSE were not launched. No production file was modified.

## What is correct

- The caller must provide an expected runner SHA before any run directory is
  created. This review authorizes only the literal
  `36737228f1355a5f45cb8248dbeae0ed21ad638f704e1f5d819132c675c60f46`;
  dynamically recomputing the value at launch is not authorized.
- DC/FM/PT and project CPU-DSE checks are repeated in all three 10-second-spaced
  samples and immediately before each point. VCS executable matching does not
  self-match the `pgrep` process.
- Same-user `simv` is forbidden regardless of state. A foreign `simv` is
  allowed only when `STAT` begins with `S` or `I`, lifetime CPU is at most 0.5%,
  and RSS is at most 262144 KiB. Every sample and final gate records the
  classification.
- The currently observed sole `simv` is owned by `fangyl`, state `Sl`, CPU
  `0.0%`, RSS `110032 KiB`, elapsed over 2.5 days. It qualifies at this instant;
  the runner neither signals nor modifies that process.
- The resource thresholds are exact 64/128/32 GiB commit-headroom,
  MemAvailable, and SwapFree gates. `failcnt`, `under_oom`, and `oom_kill` must
  all be zero before every point.
- K1, K8, and K1x8 call the same top/filelist/SDC/TCL serially in that order;
  each DC process and its runtime monitor are waited before the next point.

## P0 findings

### P0-1: a failed preflight bricks the canonical replay

The canonical run directory and EXIT trap are created before the first resource
gate. Any low-resource sample, process collision, or idle-`simv` classification
failure leaves `RUN_FAILED_OR_INCOMPLETE.txt` in that directory. A later launch
then refuses overwrite even though the attempt marker was never created and no
DC was launched. This contradicts the contract statement that a preflight
failure before first DC does not consume the one exact replay. It is concrete
on the current host whenever the 64-GiB commit-headroom gate is not met.

Required repair: run K1 preflight in a unique staging/preflight directory, then
atomically create the canonical run and attempt marker immediately before the
first DC launch; or explicitly make an attempt-absent preflight directory
recoverable without manual deletion. The canonical run must remain absent after
preflight-only failure.

### P0-2: three sequential DC points lack start-to-end input identity

All RTL, filelist, SDC, TCL, contract, libraries, and tool binary are checked
only once before K1. The runner records their initial SHAs but never rechecks
them before K8/K1x8 or before sealing the root receipt. A shared-workspace edit
during the multi-hour sequence can make the three points use different inputs
while the final manifest still reports the initial identity.

Required repair: factor the exact SHA checks into one function, call it before
each point launch and after K1x8, include the absolute runner itself, and verify
the initial identity manifest at the end. Any drift must fail the complete run.

### P0-3: runtime OOM can still produce a final PASS

The runtime monitor deliberately executes `m485_resource_snapshot ... || true`.
Therefore `failcnt`, `under_oom`, or `oom_kill` violations during K1x8 are only
logged. There is no post-K1x8 resource/cgroup gate before receipt generation, so
the run can be sealed PASS despite the contract's explicit rule that any OOM
fails the complete r3 run. Earlier-point violations are usually caught by the
next preflight; the final point has no such protection.

Required repair: latch runtime cgroup failure separately from ordinary memory
threshold excursions, wait for the monitor, and require zero failcnt/under_oom/
oom_kill violation after every point and again before final sealing. Runtime
MemAvailable/headroom threshold drops may remain observational if the contract
says so, but OOM state cannot.

## P1 findings

1. Foreign-idle allowance has an unavoidable time-of-check/time-of-use window:
   an `Sl` process may wake immediately after the final sample or during a long
   DC. The policy is a sampled coexistence rule, not an exclusive reservation.
   Receipts must not claim concurrency was impossible; a cooperative lock with
   the other owner is the only strong closure.
2. CPU is `ps` lifetime-average CPU, not interval CPU. Three spaced samples plus
   `STAT` reduce risk, but a low-lifetime-average process may still have a short
   active burst. Record this definition verbatim in the contract/receipt.
3. `pgrep -f '(^|/)(vcs)( |$)'` correctly avoids self-match and covers a normal
   `vcs` executable, but does not prove absence of every VCS-family helper such
   as `vcs1`/`vlogan`. If “VCS forever forbidden” means the complete tool
   family, add explicit exact-command matchers.

## Re-review gate

Patch only the runner guardrails: preserve all RTL, libraries, filelist, SDC,
TCL, point order, compile effort, and output schema. Close the three P0s, update
the literal runner SHA, and obtain another static hammer before DC launch.

Frozen docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
No PPA, power, system-speedup, or DATE-headline claim is admitted.
