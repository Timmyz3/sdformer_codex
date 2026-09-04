# M963 | M962/M960 C1 macro-aware DC source hammer

Verdict: `GO_SOURCE_ONLY`, score 98/100, P0=0/P1=0/P2=2. This review
legally repairs the M960 process blocker by a **narrow superseding admission**;
it does not authorize or execute DC.

## Exact exception

M934's blanket zero-assertion admission is superseded for exactly one sealed
negative-test event and nothing else: the M955/M959
`ap_candidate_after_active` failure started and failed at 10,168,500 ps. The
complete log contains one such failure and zero unexpected assertion failures.
The immediately adjacent predicate is the illegal candidate relation, followed
by the unique token
`M923_WRONG_PARENT_PHASE_CORRECT row=1 parent=63 relation_ok=0
capture_watchdog=4`.

The exact TB identity deliberately forces directory row 1 to parent 63 while
`parent_live_q[0][63]` is false, observes `relation_ok=0`, emits the token, and
calls `expect_fault("wrong parent and illegal dead-parent relation")`. The PASS
token reports `wrong_parent=1` and `attacks=6`. The sealed result, log, TB, SVA,
RTL, time, row, parent, token and `expect_fault` argument therefore form a
one-to-one negative-test identity. This is sufficient to accept that single
assertion as fault-activation evidence; it is not permission to ignore an
assertion.

Every other assertion failure, fatal/error, directory mismatch, premature
ready, dropped/duplicated row, stale bank tag, changed parent tie, or extra
scratch event remains P0 and stops launch. M934's no-false-path rule and all
timing/area/cycle gates remain unchanged. M959 must continue to be described as
a foundry UNIT_DELAY functional negative-attack PASS with one expected
assertion failure, never as a zero-assertion or clean-SVA regression.

## M962 source package

The exact M962 contract, runner, Tcl, SDC and two-entry filelist validate against
their nested SHA sidecars. The contract is still
`STOP_M960_FUNCTIONAL_GATE_UNSATISFIED__SOURCE_ONLY__NO_DC_AUTHORIZED`, with
`dc_runs_now=0`. The runner is syntactically valid, admits only the pinned M935
RTL plus nine-macro wrapper, contains no behavioral macro model in its filelist,
and rejects false paths, multicycle paths, disabled timing arcs, case analysis,
and path-specific max/min delay constraints. No M962 attempt, result, work or
lock artifact existed during review.

This review sets the exact decision fields required by the runner and admits the
M962 source hammer. It does **not** grant a DC run. A separately double-sealed
M964 release must bind this review's SHA, the exact source contract and runner,
select `SUPERSEDING_ADMISSION`, and authorize at most one attempt. A complete
negative 3 ns setup result must be sealed rather than discarded; only a setup
pass may become positive timing evidence.

## P2 and claim boundary

- M959 remains a qualified, deliberately non-clean negative-test identity; its
  expected-assertion qualifier is permanent.
- M962 remains source-only until M964. Timing, hold, cycles, speedup, PPA,
  power, energy, system and paper claims are all false.

M963 ran no VCS, DC, PT, PTPX, Formality, GPU, remote or network workload and
did not modify RTL, the M962 sources, or `docs/359`.
