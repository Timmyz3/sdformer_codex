# M518 repaired production candidate independent static hammer r2

Date: 2026-08-27  
Verdict: `STATIC_NO_GO__DO_NOT_RUN_EXACT_RUNNER__THREE_SEALED_SEMANTICS_GAPS_REMAIN`  
Score: **87/100**  
Findings: **P0=3, P1=4**

This is a receipt-blind, source-only review of the repaired M518 candidate. The
reviewer did not modify any production RTL, SVA, TB, filelist, runner, contract,
or `docs/359`, and did not run VCS, DC, Formality, PTPX, Verilator, iverilog, or
any other RTL/EDA tool. `bash -n`, JSON parsing, SHA checks, text inspection, and
an independent Python enumeration of the frozen schedule were the only executed
checks.

## 1. Literal decision

The repaired candidate is substantially stronger than r1. The RTL body remains
structurally consistent with the sealed Fixed T10 specification, and the repair
closes the r1 blockers for real-edge oldest-bank ownership, held release states,
close-stall atomic state, FIFO/tile conservation, mandatory covers, runner
self-identity, automatic isolated wrong-RTL negative control, observed VCS ID,
and relative sealed manifests.

However, the campaign still cannot truthfully publish the literal sealed
`V01--V20` PASS string:

1. V03 hits all six wide-sum points, but the lower overflow case is compared
   against `-8388607` rather than the Q24 minimum `-8388608`; saturation before
   comparison is therefore not observably distinguished from an unsaturated
   compare.
2. V15's sealed eight-cycle held-release attack is implemented for four sampled
   edges and the repaired contract weakens "eight cycles" to "multiple cycles".
3. V18's r1 repair gate explicitly required dense reset boundaries
   `c0/c11/c12/c15/c16`; the repaired campaign covers only `c5/c12/c16` and the
   contract narrows the original phrase "every dense phase" to three categories.

These are narrow TB/contract repairs, not objections to the 17-cycle RTL
architecture. They are nevertheless P0 because the runner is designed to emit
one indivisible `sealed_V01_V20` receipt. No exact runner invocation is
authorized from this review.

## 2. Independently rechecked positive facts

### 2.1 Identity and no-author-EDA boundary

- The sealed M518 baseline specification member manifest and outer seal pass.
- Current exact identities are recorded in the JSON companion.
- The M518 and M273r2 module headers independently parse to 50 public ports each,
  with exact direction/name/width-expression equality.
- `bash -n` passes; the repaired contract parses as JSON; the filelist contains
  exactly RTL, SVA, and TB in that order.
- No `results/m518*` directory, M518 `compile.log`, M518 `simv`, M518 assertion
  report, or live M518 VCS/DC process was found. Unrelated users' pre-existing
  simulator/GUI processes were not treated as M518 activity. This establishes
  only that there is no repository/process evidence of an author M518 EDA run;
  it is not a historical proof about activity outside the workspace.

### 2.2 Frozen schedule and datapath

An independent enumeration, separate from the runner preflight, produced active
slot populations:

```text
[96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,64]
```

The enumeration contains exactly 1,600 tuples, exactly 1,600 unique tuples, and
equals `{row 0..9} x {lane 0..15} x {time 0..9}`. The RTL still has five config
beats, five raw beats, two raw banks, no intermediate/product register, direct
FIFO pushes on cycles12--16, 25-bit accumulators, 26-bit update expressions, and
final signed-Q24 saturation.

The repaired TB now includes a separate frame-decoding/full-precision integer
oracle and four fixed-seed random contexts. Its rail constructor really produces
the six requested wide sums:

```text
8388606, 8388607, 8388608, -8388609, -8388608, -8388607
```

The remaining V03 blocker is observability/threshold selection, not failure to
construct those six arithmetic inputs.

### 2.3 Repaired protocol and SVA structure

- V06 deposits a controlled dual-ready completed-bank snapshot, then crosses a
  real positive issue edge. It checks bank1 selection, bank1 tag, ready-to-owned
  transition, continued ownership, and subsequently checks first-result order
  through the independent scoreboard.
- V16 now has five distinct attacks: partial raw, dense cycle0, cycle12, cycle16,
  and FIFO-only drain. Held release remains asserted into the common finish path,
  and any early retirement would break the result/conservation checks.
- Targeted full-FIFO tests separately hit no-credit phase12 and phase16; full
  simultaneous pop/push is asserted to advance both pointers and retain count16.
- `ap_close_stall_holds` no longer has a next-cycle-credit escape. Its consequent
  directly holds dense cycle/bank/tag, the packed 4,000-bit accumulator image,
  FIFO pointers, and debug issue/push/departure counters, while retaining selected
  raw ownership.
- Ownership, oldest choice, exact FIFO conservation, 17-issue/five-push ledgers,
  tile-done predecessor/tag, busy, context-cycle/retire consistency, X/Z gates,
  and the requested key covers are present. The runner requires every named cover
  to have a nonzero match.

### 2.4 Repaired V20 launcher structure

The runner checks an externally supplied exact runner SHA before creating a run
directory or querying any tool. It then creates a nested, name-disjoint negative
directory, injects an all-zero expected RTL SHA, requires exit10, checks that no
compile log/simv/positive receipt/RUN_COMPLETE exists, and seals a relative-path
negative manifest. Only after that does it run the positive exact-SHA gate.

The positive path records and checks actual `vcs -ID` for V-2023.12-SP1, embeds
negative-manifest hashes in the author receipt, and creates relative-path run
manifests with outer seals. Static inspection found no path by which next-cycle
FIFO credit can bypass the close-stall assertion and no path by which the nested
negative control creates a positive receipt.

## 3. P0 findings

### P0-1 | V03 does not dynamically distinguish saturate-then-compare

`build_rail_case` constructs the correct six unsaturated values, but the two
contexts use thresholds `8388607` and `-8388607`. For an upper overflow, both an
incorrect unsaturated value and the saturated maximum compare true against any
representable signed-Q24 threshold. For the lower overflow `-8388609`, both the
incorrect unsaturated value and saturated minimum compare false against
`-8388607`. Thus the output bit cannot distinguish the required ordering.

Minimum fix: change/add the lower-rail context to threshold `-8388608`. Then
`-8388609` must saturate to equality and fire; a compare-before-saturate bug would
not fire. Keep all six distinct points and the independent oracle. If the claim
is intended to cover the internal Q24 value itself at both rails, add a bind
observation/assertion for the saturated value or narrow the claim to the
observable event result.

### P0-2 | V15 shrinks eight held cycles to four

The sealed specification says the zero-tile release is held eight cycles. The TB
holds it on the fault edge and through three `check_quarantine` probe edges, then
deasserts it: four sampled positive edges total. The repaired contract changes
the frozen phrase to "held for multiple cycles", and the runner's semantic
preflight does not check V15 wording.

Minimum fix: hold and count at least eight consecutive sampled edges, require
zero release/retire on every edge, exactly one registered fault transition, and
sticky quarantine thereafter. Restore the literal eight-cycle text and add a
runner static term/counter gate so it cannot regress.

### P0-3 | V18 does not close the r1 reset boundary matrix

R1 required at least dense `c0/c11/c12/c15/c16`, FIFO-full close stall, and
quarantine, with a clean exact-N1 context after every reset. The repaired TB has
seven reset attacks total but only dense `c5/c12/c16`; it misses the start/end
of the prologue and the end of the four-beat close group (`c0/c11/c15`). The
contract correspondingly replaces the sealed "every dense phase" language with
generic prologue/close/tail categories.

Minimum fix: use dense resets at `c0/c11/c12/c15/c16`, retain partial config,
partial raw, FIFO-full close stall, and quarantine, and run the existing exact
N1 clean probe after every one. This makes nine reset attacks. Restore the
contract wording and require the per-cycle attack counters rather than only a
total count.

## 4. P1 findings

1. **Canonical port identity is only partially frozen.** The runner now compares
   the 50-port signatures mechanically, but it reads the current M273 RTL without
   SHA-gating that source or checking a frozen canonical signature hash. Add the
   sealed M273r2 SHA or a literal canonical signature digest to preflight.
2. **V13 has only an aggregate monitor count.** The two sample points are useful,
   but `legal_halfcycle_checks>0` does not prove every config/raw beat phase was
   observed. Add per-config-beat/per-raw-beat phase counters, or make the required
   count exact.
3. **V19 does not construct a literal opposite-data pair.** Four diverse contexts
   run without reset, which is a good stale-state test, but none is mechanically
   checked to be the bitwise/sign opposite of another. Add an explicit opposite
   config/raw pair and a counter.
4. **V10/V12 noncommit is mostly source-inspected, not dynamically conserved.**
   V14 checks the attacked raw word, but the generic config/raw framing attacks
   mainly check fault/quarantine. Snapshot the target frame/bank word and relevant
   accepted-beat counter around every offending edge.

## 5. Closure status against r1

| r1 item | r2 status |
|---|---|
| Random oracle and six exact rail inputs | Partially closed; V03 observable ordering remains P0 |
| Real-edge oldest bank1 ownership/result | Closed |
| Five held-release live states | Closed |
| Reset matrix and clean next N1 | Partially closed; all seven current resets recover, but required phase boundaries are absent |
| Ownership/oldest/FIFO/tile/atomic-hold SVA and key covers | Closed statically |
| Runner self-SHA, isolated negative, actual VCS ID, relative sealed manifests | Closed statically |
| Canonical 50-port preflight | Partially closed; live M273 source is not frozen |
| Strong status/tile-done assertions | Closed statically |

## 6. Admission boundary

Admitted by this review: repaired source structure, exact 50-port match at the
reviewed identities, independently complete 17-cycle/1,600-product schedule,
and a substantially repaired fail-closed launcher design.

Not admitted: SystemVerilog compile, Synopsys VCS behavior, literal V01--V20
completion, 29/80 RTL cycles, numerical equivalence, RTL speedup, DC, Formality,
STA, power, energy, PPA, system speedup, or headline.

The next review may be limited to the three P0 repairs and identity drift. Only
if it returns P0=0 may the exact runner execute its automatic negative control
and one positive Synopsys VCS campaign.

At review close, `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
