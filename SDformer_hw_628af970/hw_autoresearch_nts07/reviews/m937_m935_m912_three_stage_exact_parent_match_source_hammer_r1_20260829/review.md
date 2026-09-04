# M937 | M935/M912 three-stage exact parent-match source hammer

## Verdict

Review status: `PASS_M937_SOURCE_HAMMER` (88/100).  Release verdict:
`REPAIR_BEFORE_VCS_RELEASE`.

The additive M935 RTL has no source-level P0 in the exact matcher, bank
controller, or inherited M912 execution tail.  The 8x8 local reductions plus
8-to-1 root implement maximum-popcount/lowest-row-ID selection exactly;
directory popcount reuse is consistent; the added state is 283 metadata bits;
and the last-prep/F/G/R/READY NBA schedule is internally consistent.

The current verification source does not, however, close three checks it says
or implies it closes: the overlap cover is not bank-distinct, the row-63 SVA
can be satisfied by execution of the *other* bank, and no directed reset is
applied while F, G, or R contains valid metadata.  A fourth weakness makes the
bank-tag oracle less independent than advertised: it chooses the reference
slot from the DUT's own `bank_epoch_q`, while the only concurrent pair carries
identical masks.  These are repairable verification P1s, not evidence of an
RTL semantic failure.  No VCS release is issued by M937.

M937 ran no VCS, DC, EDA, GPU, remote, network, or license command and did not
modify M935 candidate sources or `docs/359`.

## Identity and frozen-boundary checks

- M935 RTL: `e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8`.
- Inherited execution SVA: `ad89adc7e9aefd350a225e58e85540ec579bbbe1ce9730891633f311de4eb4f5`.
- Supplemental match SVA: `16babee198fa7db89ccf4feee72c5e0bc10bbc5091f22ebc1d2bb94cda23f110`.
- Unit-delay TB candidate: `fec518825c51565280fc117896d75b2f25b4accb08c417ef9093fb835edacf00`.
- Static checker: `22b34b06e108ddc95d21729ab4ca116966df52d81696f7e0b500fd27b57c65c2`.
- DRAFT contract: `3657757e029102523b7752ba05bf0f94d1de7b9b1e1a3826202af4f5cc2a1b38`.
- M912 RTL: `eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53`.
- `docs/359`: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The normalized top-level port block is unchanged from M912.  The execution
combinational cone from `// Directory stage:` through `integer reset_lane;`
is byte-identical, and the sequential execution tail beginning at
`if (!exec_active_q && ready_bank_valid_w)` is byte-identical through EOF.
The inherited M919 assertion file is byte-identical after only module and bind
target renaming.

## Exactness and first-principles control audit

For every legal candidate, `parent_winner` compares the cached 5-bit
population first and the 6-bit row ID second.  Its reduction is associative
with respect to the total order `(-population,row_id)`.  The eight balanced
8-way winners and the balanced 8-to-1 root therefore equal the frozen
ascending strict-greater scan.  The invalid tuple is neutral, equal-mask
current/later rows are excluded, and a legal equal-population tie retains the
lowest ID.  M937 independently checked 20,000 64-candidate corpora, or
1,280,000 tuples, against a separately expressed sorted-order oracle.

The directory seed `{4'b0,popcount16(mask),23'b0}` occupies the existing
`directory_q[27:23]` population field.  All 64 unique rows are accepted before
matching starts, prep is blocked for the duration of `match_active_q`, and R
is the only directory writer during `BANK_MATCH`.  Thus the candidate reads a
complete seed table and no prep/R same-bank write collision exists.

Independent old-state/NBA modeling gives this edge schedule relative to the
accepted last prep row:

| Edge | Event/state after edge |
|---:|---|
| 0 | bank enters MATCH; F/G invalid |
| 1 | F row 0 |
| 64 | F row 63; issue is closed |
| 65 | G row 63 |
| 66 | R writes row 63 and the bank becomes READY |
| 67 | execution launch is eligible |

At the R63 edge, `ready_bank_valid_w` still observes the old MATCH state, so
execution cannot consume the bank before the directory write commits.  The F
and G valids drain without a ghost row.  Matching may coexist with execution
because they own distinct banks in the intended controller path; this intended
fact still needs an explicit bank-distinct verification check as described
below.

## Verification P1s and minimum repairs

1. **Overlap is not proved to be opposite-bank.**  In
   `tb_m935_three_stage_match_pipeline_unit_delay_r1.sv:502-503`, the coverage
   counter increments on `dut.exec_active_q` alone.  In
   `m935_three_stage_exact_match_assertions_r1.sv:101`, the cover is likewise
   only `match_g_valid && exec_active`.  Both pass even if matching and
   execution incorrectly name the same bank.  Minimum repair: expose/bind
   `exec_bank_q`, assert that simultaneous match/execute implies
   `match_g_bank_q != exec_bank_q`, and count/cover only that expression.

2. **The row-63 readiness assertion is bank-unqualified.**  At
   `m935_three_stage_exact_match_assertions_r1.sv:90-93`, the consequent allows
   `(match_bank_state == BANK_READY || exec_active)`.  `exec_active` can belong
   to the other bank and can therefore mask a bad state for the drained match
   bank.  In this RTL, the next assertion sample must see the target match bank
   as READY; it cannot already be EXEC because launch sees READY only on the
   following edge.  Minimum repair: require `match_bank_state == BANK_READY`
   at that next sample.  If a future controller permits zero-gap ownership
   transfer, bind `exec_bank_q` and allow EXEC only when it equals the delayed
   match bank.

3. **Reset/drain is inspected statically but not attacked dynamically.**  The
   normal TB resets only between complete scenarios; it never asserts reset
   with a valid row in F, a valid winner set in G, or row 63 waiting in R.
   Minimum repair: add three directed reset cases, one per occupied stage, and
   prove F/G valid, issue-done, match-active and both bank states return to the
   reset contract with no later directory write, READY pulse, or execution
   launch from the aborted task.

4. **The match oracle's bank tag is partly DUT-derived.**  At
   `tb_m935_three_stage_match_pipeline_unit_delay_r1.sv:469`, the oracle slot is
   selected with `dut.bank_epoch_q[dut.match_g_bank_q][2:0]`.  The concurrent
   epochs 1 and 2 use the same directed masks, so a swapped tag can retain the
   same expected directory.  Minimum repair: maintain an external accepted
   prep-to-bank ownership model, compare both F/G bank tags and epoch against
   that model, and use different masks for the two overlapped tasks.  The DUT
   epoch may remain an observation, but not the selector of its own oracle.

After these four source repairs, run the static checker again, freeze fresh
SHAs in a new non-DRAFT exact-attempt contract, and obtain a separate hammer
before any VCS invocation.

## Static checker and syntax-risk audit

The provided checker reran successfully:

```text
PASS_M935_THREE_STAGE_EXACT_MATCH_SOURCE_STATIC algorithm_rows=262208 metadata_bits=283 ports_unchanged=true execution_tail_byte_exact=true inherited_m919_sva_exact=true source_only=true vcs=false dc=false timing=false speedup=false ppa=false energy=false system=false headline=false
```

M937 also found no reused SystemVerilog reserved identifier such as the prior
`packed` failure.  `packed_row` is legal.  The arrayed combinational trees,
function-local tuples, integer part-selects, SVA sequence delays and bind
target names are source-consistent.  Dynamic hierarchical selects such as
`bank_state_q[match_g_bank_q]` in a bind connection and TB accesses to unpacked
arrays remain compiler/tool facts, not source proofs; only the future exact-SHA
VCS compile may admit them.

The checker itself pins frozen ancestors but does not pin the candidate files
or DRAFT contract it is checking.  That is acceptable for a DRAFT source
screen only.  The future executable release must fail closed on all candidate,
TB, SVA, checker and contract SHAs rather than inheriting this script as an
exact-attempt launcher.

## P0 / P1 / P2

- **P0=0:** no source-level exactness, NBA-order, port, inherited-execution,
  storage-count, or same-cycle bank/row hazard was found.
- **P1=4:** make overlap bank-distinct; qualify row-63 readiness to the drained
  bank; add F/G/R reset attacks; make the bank-tag oracle externally owned and
  use distinct overlap payloads.
- **P2=3:** exact-SHA VCS must adjudicate SV/bind syntax; the future launcher
  must pin all candidate sources; committed directory/parent-live closure
  should be dumped and compared directly in addition to the pre-write R miter.

## Claim boundary

M935 remains a source candidate only.  M937 admits neither functional VCS,
timing, cycles, speedup, area, PPA, power, energy, trace recurrence, full-system
performance, nor a paper/headline claim.  The verdict is REPAIR rather than
NO-GO because the RTL exactness and control schedule survive the hammer and
all blockers are localized verification-source repairs.
