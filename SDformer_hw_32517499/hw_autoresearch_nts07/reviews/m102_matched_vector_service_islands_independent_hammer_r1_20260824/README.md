# M102 matched vector-service islands independent hammer

Date: 2026-08-24

Status: `CONDITIONAL_GO_LEDGER_AND_BASELINE_NO_GO_CANDIDATE_FAIL_CLOSED_AND_DC`

This review is read-only with respect to M102 production RTL, VCS inputs and
sealed outputs, contracts, cycle-ledger results, M88 evidence and `docs/359`.
It writes only this independent review directory and uses no open-source tool
result.

## Verdict first

The M102 arithmetic and frozen-evidence chain are strong.  The baseline
fixed-INT8 service island is a valid directed implementation of one 256-bit
slot per cycle and exact three-beat 96-lane service.  The combined candidate's
legal PWP/correction/fallback paths also demonstrate one accepted 256-bit beat
per cycle with no directed mode-switch bubble.  The analytical service ledger
reconstructs exactly:

```text
baseline = 371,461,096 vector-op * 3 = 1,114,383,288 cycles
candidate PWP = 226,222,255 cycles
candidate correction/fallback = 188,148,490 * 3 = 564,445,470 cycles
candidate total = 790,667,725 cycles
ratio = 1,114,383,288 / 790,667,725 = 1.409420484439276x
```

However, **M102 is NO-GO for a fully admitted combined-candidate VCS contract
and NO-GO for the next matched DC sweep in its current SHA**.  There is a
concrete fail-closed hole: after a top-level request fault, the candidate still
passes through M82 `output_valid` and `output_accept`.  A stalled old output
can therefore remain valid and later be accepted while `protocol_error=1`.
The baseline explicitly suppresses this case; the candidate SVA and TB do not
test it.

Also, 1.409420x is a **frozen service-slot work ratio**, not the number of
cycles executed by the current single-context candidate RTL.  The candidate
parser blocks service for 128 edges per phase, while the ledger hides those
edges using M88's future inactive-slot overlap assumption.  Without that
unimplemented overlap, 8,640 phases add at least 1,105,920 exposed parser
cycles and the comparable ratio is at most approximately 1.407452x before
phase-load edge accounting.  This difference is only 0.140%, but the semantic
boundary matters before frequency normalization.

Score: **84/100 for the scoped M102 milestone**.

Severity: `P0=1 / P1=5 / P2=5`.

## Exact-SHA and sealed-evidence audit

All available manifests verify with their documented working-directory
semantics:

| Evidence set | Result |
|---|---:|
| M102 VCS input manifest | 13/13 OK |
| M102 VCS output manifest | 7/7 OK |
| M102 cycle-ledger manifest | 6/6 OK |
| M102 preflight manifest | 6/6 OK |
| independent M88 review manifest | 3/3 OK |

The input manifest pins M82, both M102 RTL tops, both SVA files, both
testbenches, both VCS filelists, the contract, M88 result and preflight
receipts.  Current hashes match the contract and launch-time checks.  The
recorded runner SHA also matches the current VCS runner.  All four compile/sim
return codes are zero, both PASS lines match exactly, output logs/reports match
their manifest, and no assertion-failure signature is present.

The runner is recorded only in a separate `runner_sha256.txt`; it is not a
member of the input or output manifest and the cycle-ledger analyzer does not
pin it.  Current provenance is recoverable and consistent, but the next sealed
run should include the runner itself in a root-relative manifest.

The M102 result manifest is relative to `hw_autoresearch_nts07`; the preflight
and M88-review manifests are relative to their own review directories.  A
failed check from a different directory is a path-semantics issue, not digest
drift, but the mixture should be normalized in the next run.

## RTL, TB and SVA semantic audit

### Baseline

The baseline is coherent for its stated port cut:

- requests are already precompacted `(source, output_block)` vector-ops;
- `(source*8+block)*24 + beat*8` selects exactly 24 32-bit words;
- all frozen vectors are bank-aligned and use three 256-bit beats;
- M82 assembles 96 signed INT8 lanes and sign-extends them to signed12;
- continuation beat/source/block/tag identity is locked;
- output backpressure is stable and a legal next first beat is backpressured,
  not faulted;
- a request fault suppresses both input readiness and buffered output until
  reset.

The directed run reports 90 completed vectors, 274 accepted beats, 94 starts
including four attack prefixes, 23 explicit II=3 start checks, 28 stalled
output cycles, six attack classes and seven reset recoveries.  The SVA covers
the sequence, bank mapping, sign extension, fault stickiness and stalled
output.  This is good directed evidence, not actual-record replay.

### Candidate legal service paths

For legal traffic, the combined top implements the intended service shapes:

- PWP uses metadata-selected signed widths 8/9/10/11 and 3/4/4/5 beats;
- correction and block-local fallback use three fixed8 beats;
- correction `service_negate` applies a lane-wise signed12 two's-complement;
- fallback is legal only on code4 and emits real weight-vector data rather
  than M82's old zero escape token;
- one `service_valid/service_ready` stream feeds one M82, so at most one
  aggregate 256-bit beat is accepted per cycle;
- the directed sequence exercises eight vectors, 28 beats, all four PWP
  widths, positive and negative correction, fallback, three stalled-output
  cycles and seven adjacent start-interval checks.

This proves aggregate request-slot serialization for the directed cases.  It
does not prove a physical shared SRAM port: `bank_select_pwp` leaves the
PWP-versus-weight memory selection and 256-bit response mux outside the top,
and TB directly manufactures already-selected `bank_words`.

### P0 fail-closed counterexample

The candidate defines:

```text
protocol_error = request_fault_q || phase_poison_q || m82_protocol_error
output_valid   = m82_output_valid
output_accept  = m82_output_accept
```

Unlike the baseline, neither output signal is gated by the top-level fault.
Consider this legal trace:

1. A valid vector finishes and its M82 output is held with
   `output_valid=1, output_ready=0`.
2. While that old output is stalled, present an invalid new first request.
3. On the next edge `request_fault_q` becomes one.  M82 retains the old output
   because it was not accepted.
4. The following cycle has `protocol_error=1` and still
   `output_valid=1`; raising `output_ready` can assert `output_accept` under
   fault.

The candidate SVA has no equivalent of the baseline's
`protocol_error |-> !output_valid && !output_accept`, and every candidate
attack is issued after reset with no buffered output.  Thus the sealed directed
tests do not refute the counterexample.  This blocks the contract's broad
`fail-closed protocol attacks` admission and should be repaired before DC.

The minimum repair is to gate the candidate output path and M82
`output_ready` with the full top-level fault in the same fail-closed style as
the baseline, then add a stalled-old-output plus invalid-new-request test and
SVA.  Exact output ownership during the fault edge must be frozen explicitly;
the fix requires a new source SHA and resealed VCS evidence.

### Remaining verification gaps

The combined parser is copied into the new top rather than instantiating the
sealed M99 top, but all seven candidate phase loads use legal metadata.  No
reserved metadata code, wrong pattern base, fetch/cursor overflow or zero
terminal case drives `metadata_error`.  `ap_bad_metadata_blocks_service` is
therefore never meaningfully exercised.  The contract's `metadata legality`
wording is broader than this run.

The only continuation mutation in the candidate TB changes source.  It does
not independently attack pattern, block, kind, tag or negate, and the directed
transition order does not test the likely PWP-to-correction next-vector seam.
Static identity checks exist, but admission should cover these interfaces
after the P0 repair.

Assertion cover matches are occupancy matches, not transaction counters.  For
example `cp_pwp=7` comes from four PWP vectors and `cp_protocol_fault=11` from
six attacks.  The PASS-line counters, not cover-match totals, are the correct
directed event counts.  Likewise, `cp_positive_correction` matches both the
positive and negated correction because it only checks `output_kind==1`.

## Single-256-bit-slot fairness

The cycle-level bandwidth comparison is fair in one narrow sense:

- both sides expose one 256-bit data response and accept at most one beat per
  cycle;
- baseline fixed8 vectors take exactly three beats;
- candidate PWP vectors pay physical alignment as 3/4/4/5 full 256-bit slots;
- correction and fallback pay three full slots;
- candidate service kinds are serialized rather than executed on hidden
  parallel PWP and weight ports.

The PWP cycle sum independently reconstructs:

| Width | Uses | Slots/use | Slots |
|---:|---:|---:|---:|
| 8 | 11,164,284 | 3 | 33,492,852 |
| 9 | 32,360,036 | 4 | 129,440,144 |
| 10 | 13,936,011 | 4 | 55,744,044 |
| 11 | 1,509,043 | 5 | 7,545,215 |
| total | 58,969,374 | — | 226,222,255 |

The comparison is not equal-area or complete physical-port fairness:

- candidate listed storage is 116,525 B versus the baseline's 24,576 B;
- memories, decoders, ECC, data mux and response timing are excluded;
- matcher/enumerator, DMA/controller, accumulator and destination update are
  excluded;
- baseline begins from a precompacted active source, so mask scanning and
  enumeration are free at this boundary.

Therefore `one 256-bit aggregate slot/cycle` is admitted as a command/data
ledger constraint, not a physical SRAM implementation.

## Cycle-ledger reconstruction and inflation audit

The analyzer's arithmetic is exact and uses the correct scopes.  No double
count or missing PWP/correction population was found:

- baseline ops: 371,461,096;
- candidate correction/fallback ops: 188,148,490;
- candidate PWP ops: 58,969,374;
- combined candidate vector-ops: 247,117,864;
- PWP plus correction/fallback slot total: 790,667,725;
- service-only ratio: 1.409420484439276x.

The frozen M88 bounded values are different and also reconstruct:

```text
1,114,402,488 / 790,706,475 = 1.409375695323603x
```

They include startup/tail/fill and a hypothetical finite double-buffer
schedule.  Service-only and bounded numbers must never be mixed across the two
sides.

The 1.409420x number is nevertheless not an actual VCS cycle ratio.  VCS runs
only 90 baseline and eight candidate directed vectors; `actual_record_replay`
is correctly false.  The large populations come from SHA-pinned M78/M88
valid825-internal ledgers.

The present candidate top has one `metadata_q` and blocks service while
`parse_active`.  It cannot perform M88 inactive-slot concurrent preparation.
At 8,640 phases, its 128-edge parser represents at least:

```text
8,640 * 128 = 1,105,920 exposed cycles
1,114,383,288 / (790,667,725 + 1,105,920)
    = 1.407451858290x
```

Adding one non-overlapped load-accept edge per phase would yield approximately
1.407436500047x.  These are not new admitted headlines; they bound the current
single-context execution semantics and show that the analytical 1.409420x
ratio is inflated by about 0.140% if mislabeled as this RTL's runtime.  M88's
larger 1.4093757x remains a model with an unimplemented inactive-slot loader,
not evidence that this top overlaps parsing.

## Readiness for matched Synopsys DC

Current status is **NO-GO until the P0 fix and VCS reseal**.  After that, the
logic-only A/B can proceed, but several preflight conditions should be frozen:

1. Create production-only DC filelists.  Both current M102 filelists include
   SVA and TB and are VCS-only.
2. Use identical DBs, corners, period grid, uncertainties, I/O delay/transition
   and output loads, compile passes, hold repair, ideal clock and ZeroWireload.
3. Keep the result named pre-macro, port-cut service-island timing.  The
   external bank selection/mux and address-to-SRAM-to-data timing remain open.
4. Use a per-period recompile and first-pass/last-fail bracket for each actual
   M102 top.  Do not use M85/M99 frequency.
5. Audit precompile operators.  The baseline weight address uses explicitly
   sized arithmetic, while the candidate uses unsized constants in
   `((service_source*8)+service_block)*24` and in several dynamic-index
   expressions.  Before freezing DC, share or type the weight mapper so an
   accidental 32-bit arithmetic family does not make the A/B a coding-style
   comparison.  Any RTL change requires VCS resealing.
6. Report the physical-frequency formula only as a logic-island estimate:
   `(1114383288/f_baseline)/(790667725/f_candidate)`.  It remains NO-GO as
   physical throughput until the excluded memory path can sustain the required
   clock and the two-memory response mux is included.
7. If the service-only ledger is retained, state the inactive-slot overlap as
   an external modeled precondition.  If timing the actual single-context top,
   charge its exposed parser/load cycles instead.

## Findings

### P0

1. `M102-H-P0-01-CANDIDATE-FAULT-LEAKS-BUFFERED-OUTPUT`: candidate top-level
   request fault does not suppress an already buffered M82 output.  This
   contradicts fail-closed admission and blocks current-SHA DC promotion.

### P1

1. `M102-H-P1-01-PARSER-OVERLAP-NOT-IMPLEMENTED`: 1.409420x omits 1,105,920
   parser edges that the current single-context RTL cannot overlap.
2. `M102-H-P1-02-PHYSICAL-SHARED-PORT-IS-A-CUT`: `bank_words` is already
   selected; PWP/weight SRAM response mux, timing and area are not present.
3. `M102-H-P1-03-METADATA-AND-SEAM-COVERAGE`: no poisoned metadata and only
   one continuation-identity mutation are exercised in the copied parser/top.
4. `M102-H-P1-04-DC-INPUTS-NOT-FROZEN`: current filelists contain TB/SVA and
   candidate weight mapping uses differently sized arithmetic from baseline.
5. `M102-H-P1-05-SAME-BANDWIDTH-NOT-EQUAL-AREA`: candidate storage and control
   resources exceed baseline and macro/decoder/ECC costs remain excluded.

### P2

1. `M102-H-P2-01-SERVICE-VS-BOUNDED`: 1.409420484x and 1.409375695x have
   distinct scopes and cannot be mixed.
2. `M102-H-P2-02-DIRECTED-NOT-ACTUAL`: the 247,117,864-op candidate population
   is ledger evidence, not replay by the eight-vector VCS test.
3. `M102-H-P2-03-COVER-MATCHES-ARE-OCCUPANCY`: cover matches exceed event
   counts; `cp_positive_correction` also includes the negated correction.
4. `M102-H-P2-04-PROVENANCE-PATHS`: manifest working directories differ and
   runner SHA is separate rather than sealed into the main manifests.
5. `M102-H-P2-05-VALID825-AND-ESCAPE-THINNESS`: inputs remain
   valid825-internal and only 362 block-local escape rows underlie the fallback
   workload.

## GO / NO-GO matrix

| Decision surface | Verdict |
|---|---|
| exact-SHA frozen evidence | GO |
| baseline directed RTL/VCS/SVA | GO |
| candidate legal PWP/correction/fallback directed paths | GO |
| aggregate one-slot command serialization | GO, cycle-level only |
| 1.409420484x service-slot ledger arithmetic | CONDITIONAL GO |
| 1.409420484x as actual current-RTL VCS/runtime ratio | NO-GO |
| combined candidate fail-closed contract | NO-GO |
| current-SHA matched DC sweep | NO-GO pending repair/reseal |
| repaired production-only logic-island DC sweep | GO after gates |
| physical Fmax/throughput or equal-area result | NO-GO |
| full-network/system/DATE headline | NO-GO |

