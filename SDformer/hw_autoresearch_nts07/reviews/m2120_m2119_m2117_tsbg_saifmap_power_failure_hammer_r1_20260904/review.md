# M2120 independent failure hammer: M2119 matched TSBG power campaign

## Verdict

**The failure audit passes 100/100; M2119 itself is a consumed failure with two
P0 findings and no citable power or energy.**  The review invoked no license
query, VCS, `simv`, Design Compiler, PrimeTime, or GPU process.  M2119 may not
be retried.

The attempt token has one exhaustively sealed member and says
`M2119_ATTEMPT_CONSUMED`, `automatic_retry=false`.  The failure quarantine has
seven exhaustively sealed members and zero symlinks.  Executed counts are one
license query, one VCS compile, one ordinary `simv`, zero DC, zero PT, and zero
admitted SAIF files.  One raw SAIF was written but rejected.  The canonical
result and launch lock are absent; the residual VCS work tree is unsealed and
noncitable.  No `power.rpt`, result JSON, DC netlist, or PTPX report exists.

## P0-1: the TB/UCLI window mixes clock phases

The ordinary functional ledger reaches both wrapper stops without a functional
or assertion fatal, but the raw SAIF duration is 60,877.5 ns rather than the
contracted `20292 * 3 ns = 60,876 ns`.

This is not a parser arithmetic bug.  VCS exits the second stop at 62,041,510
ps.  Subtracting the reported 60,877,500-ps SAIF duration gives a first stop at
1,164,010 ps.  Modulo the 3,000-ps clock, the first stop is 10 ps after a
negedge while the second is 10 ps after a posedge.  Therefore the measured
interval is exactly `(20292 + 0.5) * 3 ns`.

The static source explains the phases.  The M2051 descriptor loader returns at
a negedge and immediately writes `full_execute_start_cycle`.  M2117 observes
that variable and stops after only `#0.01`, so activity starts on that negedge.
At completion, `base_done_cycle` is assigned by a posedge scoreboard; M2117
again stops after only `#0.01`, so activity ends on a posedge.  UCLI correctly
enables and disables power at those two stops; the stops themselves are wrong.

The production parser is correct to require the integer-cycle denominator.
A mechanical mutation proves that a zero-TX 60,877.5-ns SAIF is rejected and
the otherwise identical 60,876-ns SAIF is accepted.

## P0-2: duration is not the only problem

The rejected SAIF contains 93,971 activity records.  Of these, 58,277 have
nonzero `TX`, and their `TX` sum is 40,619,426 time units.  This is consistent
with inactive/internal four-state state becoming known gradually or remaining
unknown during the measurement window.  Replacing only the duration header
with 60,876 ns was mechanically rejected by the existing parser for nonzero
`TX`.  Accordingly, neither loosening the duration tolerance nor changing the
expected denominator to 20,292.5 cycles is a valid repair.

## The only allowed successor

This review permits **source authoring only** for M2125, followed by independent
M2126 source review and, only if that passes, one M2127 VCS-only diagnostic with
an independent M2128 result hammer.  It does not authorize VCS now and does not
authorize any DC/PT rerun.

The minimal successor keeps the exact M2117 RTL, fixture, slot-42 workload,
ordinary/TSBG schedule modes, ports, and ledgers.  It changes only the activity
measurement discipline:

1. After `full_execute_start_cycle` is observed, wait one explicit settled
   negedge before the begin stop.  After selected completion is observed, wait
   one explicit settled negedge before the end stop.  This changes the current
   negedge-to-posedge interval into an exact negedge-to-negedge interval without
   changing the cycle ledger.
2. Compile with exactly one `+vcs+initreg+random` instrumentation option and run
   each axis with exactly one `+vcs+initreg+0`.  This must be disclosed as
   deterministic zero-delay four-state simulation handling, not a reset,
   silicon power-on state, timing proof, or hardware feature.
3. Retain the 383-cycle descriptor preload and settled-edge public warmup/check.
   All unconditional public controls/counters and valid-qualified payloads must
   be known.
4. M2127 contains no DC or PT stage.  It passes only if ordinary and TSBG retain
   their exact functional ledgers, durations are exactly 60,876 and 22,707 ns,
   respectively, and every SAIF record has `TX=0`.

Only after an exhaustive M2128 PASS may a separate DC/saif_map/PTPX production
source be authored and independently reviewed.  The diagnostic does not revive
M2119 or allow reuse of its rejected SAIF.

## Claim boundary

M2119 establishes only that the ordinary source-RTL workload reached the two
functional boundary markers before its SAIF was rejected.  It establishes no
admitted SAIF, mapped activity, logic power, SRAM energy, hold closure, system
speedup, energy per frame, or paper-ready PPA.

`docs/359_DATE终局冻结_20260813.md` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
