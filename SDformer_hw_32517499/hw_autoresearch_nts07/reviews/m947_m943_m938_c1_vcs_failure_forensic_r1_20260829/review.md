# M947 | M943/M938 C1 VCS failure forensic

## Verdict

`PASS_M947_M943_FAILURE_FORENSIC`, score 98/100.  Design verdict:
`FAIL_M943_TB_COVERAGE_ONLY__DO_NOT_RERUN__ONE_TB_ONLY_SUCCESSOR`.
P0=0, P1=1, P2=1.

The unique canonical quarantine is
`results/m943_m938_m935_c1_three_stage_exact_match_unit_delay_vcs_r1_20260829.failed_or_incomplete.2807848.quarantine`.
Its recursive manifest and outer seal pass.  M947 did not run VCS or modify the
quarantine, source, attempt marker, or `docs/359`.

## What failed

Compilation completed and produced the sealed `simv`/compile database.  The
simulation ran to the normal coverage gate and exited through the TB fatal at
line 2027.  The quarantine records exit code 22 only because the runner later
found the required PASS token absent.  The simulation's causal failure is:

```text
normal coverage minima missed 22 7 19 0 195 1 1 8 8 2 63 1 1 fill=7
```

The fourth value is `cov_pending_plus_forward=0`; every other normal minimum is
nonzero.  Reset F/G/R coverage passed 1/1/1.  SVA reports show the new full-II1,
bank-distinct overlap, same-pop tie and inherited execution covers firing.
There is no RTL assertion, supplemental SVA, exact parent miter, external bank
owner, reset, arithmetic, queue, or protocol-attack failure before the fatal.
The main PASS token is absent, as required by fail-closed operation.

## Root cause

The M938 six-row `make_dual_enqueue_masks` corpus is byte-equivalent to the
M923/M926 witness.  M935's execution RTL and one-cycle debug observer tail are
also byte-identical to M912: `debug_dual_enqueue_event` observes the prior
cycle's functional `read_pending_q && forward_accept_w`; it does not create the
event.  The cleanroom cover correctly credits the functional same-cycle pair
`expected_read_response && expected_forward`, while separately checking the
one-cycle debug pin.

M926 hit this cover once.  M938 changed the preceding overlapped epoch2 from
the same directed corpus to `make_random_masks(32'h9380_0002)` to prove bank-tag
independence.  That task has a different execution length.  Because the sink
ready LFSR runs continuously throughout the normal suite, epoch4 consequently
starts at a different ready phase.  In the failed log, the intended P1 row is
stalled long enough that the parent response is queued before the later
forward; no cycle has both `read_pending=1` and `forward=1`.  This is a fragile
TB scheduling assumption exposed by the correct distinct-mask repair, not an
RTL/SVA functional regression.

## Minimum TB-only repair

Keep the six masks, the cover, and the `>=1` minimum unchanged.  In the
dedicated epoch4 task only:

1. Force the two public sink-ready inputs high across the intended P1 two-beat
   window, independent of the inherited LFSR phase; do not force DUT internal
   queue, prefetch, read, or forward state.
2. Add a bounded phase monitor that first requires an accepted macro read for
   the intended C0 consumer on P1's first residual beat.
3. On the immediately following cycle require the real DUT state/event pair
   `read_pending_q && forward_accept_w`, with exact pending parent/consumer and
   forward consumer identities; then require the delayed
   `debug_dual_enqueue_event` on the next sample.
4. Let the cleanroom oracle independently produce
   `expected_read_response && expected_forward` and increment the existing
   `cov_pending_plus_forward`; fatal on any phase/identity/watchdog mismatch.
5. Release sink forces only at an inactive edge after the witnessed handshake,
   then complete the unchanged end-to-end oracle and all existing minima.

This makes the witness causal and reproducible without weakening coverage or
changing RTL/SVA semantics.

## Evidence that must remain

The successor must retain the M938 reset F/G/R63 attacks and 1/1/1 gate,
opposite-bank assertion/cover/counter with distinct epoch1/epoch2 masks,
external accepted-prep/task-done ownership oracle, all 64-row parent-directory
miters, same-pop lowest-ID witness, inherited M919 assertions, foundry response
checks, held-final recovery, and all six protocol attacks.

## Unique successor rule

M943 is consumed and may never be rerun.  Exactly one successor lineage is
allowed: a new TB-only revision implementing the causal dual-enqueue phase,
with M935 RTL and both SVA files unchanged; then a fresh static checker,
independent source hammer, new runner/contract/release, and a new attempt/result
identity.  Any RTL/SVA edit, reduced minimum, deleted cover, reused M943 marker,
or second parallel successor is NO-GO.

Functional VCS remains false.  Timing, cycles, speedup, PPA, power, energy,
system speedup, and paper/headline claims remain false.
