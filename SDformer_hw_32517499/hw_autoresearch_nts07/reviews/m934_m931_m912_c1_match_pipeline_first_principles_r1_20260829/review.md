# M934 | M931/M912 C1 parent-match timing cone first-principles review

## Verdict

Review status: `PASS_M934_FIRST_PRINCIPLES_REVIEW` (98/100).  Design
verdict: `FAIL_M931_SETUP__DO_NOT_CITE__ONE_EXACT_II1_SUCCESSOR_ALLOWED`.

M931 is a valid raw macro-aware DC diagnostic, but it is not a timing-closed
or paper-PPA point.  At 3.000 ns its setup WNS is `-4.9058 ns`, critical path
length is `7.69 ns`, TNS is `-15026.33 ns`, and 3,128 setup paths violate.
All 100 reported worst paths start at `match_bank_q_reg` and end at a
`directory_q_reg[bank][row][residual-bit]` register.  The old M892
`exec_bank_q -> psum_write_valid` cone is absent from the top 100.  M912
therefore repaired the execution/control boundary it targeted, and exposed a
second, independent functional bottleneck in the frozen M528 preprocessing
algorithm.  That functional match cone must not be false-pathed.

M934 ran no EDA, VCS, GPU, remote, network, or license command.  It did not
modify RTL, M931, M929, or `docs/359`.

## Evidence boundary

- Canonical M931 area is `158975.102204 um^2`, consisting of
  `78825.243164 um^2` macro area and `80149.859040 um^2` standard-cell area.
- M931 has 85,396 cells: 73,030 combinational, 12,356 sequential, and nine
  bound `TS1N28HPCPHVTB128X128M4S` SRAM macros.
- Precompile gates are clean: `TIM-209=0`, `OPT-150=0`; macro count is 9 before
  and after compile.  No behavioral macro model was read by DC.
- The setup path starts at the bank-select flop, traverses 511 logic levels,
  arrives at `7.6852 ns`, and misses the `2.7794 ns` required time.  The first
  endpoint is `directory_q_reg[1][60][13]`; all 100 endpoint families are
  directory residual/write bits.
- Hold WNS `-0.0894 ns` is diagnostic only.  It is not a setup repair and is
  not hold signoff.
- M929 remains the authority only for the frozen M912 foundry-`UNIT_DELAY`
  functional workload.  It does not upgrade M931 or any CPU cycle estimate.

## Why the critical endpoint moved

The frozen match statement performs the following in one clock:

1. `match_bank_q` and `match_row_q` select one 16-bit current mask from two
   banks of 64 masks.
2. The current popcount is computed.  For each of 64 candidates, RTL computes
   subset/equality, nonzero popcount, the equal-pattern row-ID exception, and
   candidate popcount.
3. A procedural low-to-high scan carries `match_best_valid/pop/id/mask`
   through every candidate.  Strict-greater replacement implements maximum
   popcount while preserving the lowest row ID on a tie, but synthesis must
   retain the data dependence and produced a long priority/mux chain.
4. The selected parent mask is XORed with the current mask, packed into
   `match_directory_w`, decoded by bank and row, and written to
   `directory_q[*][*]`.

M912 balanced the execution row and prefetch selectors and registered their
outputs, but deliberately left this M528 preprocess scan unchanged.  Once the
old execution cone was cut, the single-clock parent scan became the global
maximum.  Repeated residual-bit endpoints and the exact 100/100
`match_bank_q -> directory_q` population rule out a debug-only, single-bit,
or SRAM-port explanation.

## Exact winner representation

For current row `r` and candidate `c`, define candidate validity exactly as in
M528:

```text
current_pop >= 2
and candidate_pop >= 1
and (candidate_mask & current_mask) == candidate_mask
and not(candidate_mask == current_mask and c >= r)
```

Encode a valid candidate as `{1'b0, ~candidate_pop[4:0], c[5:0]}` and an
invalid candidate as `12'hfff`.  Pairwise unsigned `min` then selects the
largest candidate popcount and, on equal popcount, the lowest row ID.  This is
exactly the frozen low-to-high/strict-greater choice; it is not a heuristic,
approximation, or new sparsity definition.  The selected mask remains the
payload used for the exact residual XOR.

Candidate popcounts need not add a 640-bit array.  On `prep_store_w`, the
already existing `directory_q[row][27:23]` can be preseeded with
`popcount(prep_mask)` while all other seed bits are zero.  Final match writes
the same original-popcount field.  Reading a row whose final directory was
already written is therefore equivalent to reading the seed.  Prep and match
are mutually exclusive in the frozen controller, so this creates no second
simultaneous writer.

## Candidate structures

### A. Aggressive G/R, II=1 (`+1 cycle/task`)

Compute eight independent 8-candidate local winners directly from the dynamic
current-mask read and register them; reduce 8-to-1 and write the directory in
the next stage.  It adds about 253 metadata bits and predicts
`437729819` cycles, `1.737031x` versus strong-zero and `1.731540x` versus bit.
The first stage still contains the dynamic current-row mux, subset logic and
three winner levels.  Given M931's 7.69 ns path, this is unnecessarily risky
for the next one-shot DC and is not recommended.

### B. Recommended F/G/R, II=1 (`+2 cycles/task`)

- **F (fetch):** register valid, bank, row, current mask and current popcount.
- **G (group):** form 64 exact candidate keys, reduce eight groups of eight,
  and register each group's key and winning 16-bit mask together with current
  metadata.
- **R (root/write):** reduce the eight registered winners, compute exact
  residual XOR, write the directory, and OR the winning parent ID into
  `parent_live_q`.

F/G/R launches one row every cycle.  Sixty-four rows therefore take 64 launch
cycles plus two fill/drain cycles, not two or three cycles per row.  Estimated
new match metadata is about 282 bits; together with M912's frozen 130-bit
boundary the total remains about 412 bits, below the 512-bit gate.  No
1152/1824-bit payload register, third parent slot, or scratch port is added.

The fail-closed paper/model prediction is `438541979` cycles:
`1.733814x` versus M468 strong-zero and `1.728333x` versus same-coordinate bit.
These are arithmetic projections from 812,160 frozen tasks, not RTL-measured
cycles or paper-admitted speedups.  Double-bank overlap may hide some match
fill/drain time, but the model must charge all two cycles until a unified
replay proves otherwise.

### C. Conservative key-only F/Q4/Q16/root/write, II=1 (`+4 cycles/task`)

Carry only keys through 16 four-candidate winners, four four-winner reducers,
a registered root, and a final mask-read/XOR write stage.  Estimated new
metadata is about 368 bits, or about 498 bits including M912.  It predicts
`440166299` cycles and `1.721956x` versus bit.  It is the timing-safe fallback,
but it leaves almost no 512-bit margin and adds a final dynamic mask read.
It must not be implemented or sent to DC in parallel with B.  A fresh review
is required before using it after a failed B attempt.

Any serial scan with `II>1`, including eight candidates per cycle for every
row, is rejected.  The frozen ledger already shows that merely adding one
cycle per active row falls to about `1.638x` versus bit.

## Preserved semantics

The F/G/R successor must carry the match-bank tag through all stages and may
assert `BANK_READY` only when row 63's result is actually committed.  Match
results remain one write per cycle in row-ID launch order.  Parent-live updates
remain an OR reduction over exact parent IDs.  Matching one bank may still
overlap executing the other bank.  Stable execution order, active/next
contexts, prefetch, two parent-response slots, one-port nine-macro scratch,
signed reconstruction, atomic psum/row completion and every external port are
unchanged.

## VCS admission gate

Before any new DC attempt, a fresh additive RTL/TB/SVA identity must pass all
of the following under foundry `UNIT_DELAY`:

1. Compare all 64 final directory entries and `parent_live_q` against the
   frozen procedural oracle for every task, not only final psums.
2. Directed parent cases: all-equal masks; equal-popcount different subsets;
   an eligible strict subset at a later row ID; equal-mask earlier/later-ID
   boundary; zero and popcount-one current rows; zero candidate; parent IDs 0
   and 63.
3. Exactly 64 launches and 64 directory writes; full F/G/R occupancy; row 63
   write before `BANK_READY`; no bank-tag crossing; reset at each pipeline
   stage; execution of the opposite bank during matching.
4. Preserve M926/M929's six attack classes and all arithmetic, queue,
   single-port, backpressure and completion checks.  The wrong-parent attack
   must occur after the pipelined directory is complete but before context
   capture, avoiding the already quarantined stale attack phase.
5. Prove no inter-row execution bubble beyond the already charged M912
   control behavior, and record match fill/drain separately.  A same-ledger
   replay must charge every exposed cycle.

Any directory mismatch, premature ready, dropped/duplicated row, stale bank
tag, changed parent tie, extra scratch event, or assertion failure is P0 and
stops the DC launch.

## DC admission gate

Only candidate B is recommended for the next one-shot macro-aware DC.  Its
source hammer and separate release must bind the passing VCS result and exact
cycle receipt.  DC must keep the M931 conditions: TSMC 28 nm, 3.000 ns ideal
clock, ZeroWireload, nine bound 128x128 1RW macros before/after compile,
slow/fast macro pair, no behavioral macro Verilog, no debug false path, one
`compile_ultra`, no incremental compile, and clean `TIM-209/OPT-150`.

Pass requires all of:

- setup WNS `>= 0.0000 ns` and no setup violator;
- exactly nine macros;
- total cell area `<= 172034.361455 um^2` and standard-cell area
  `<= 85326.593975 um^2`;
- mechanically counted added metadata `<= 512 bit`, with no new wide payload
  queue or third parent slot;
- same-ledger total `<= 445851049` cycles, equivalent to at least `1.70x`
  versus the frozen bit denominator;
- independent result hammer confirms the worst paths are not a negative
  `match_bank/current -> directory` or hidden functional commit cone.

Hold, power, energy, PPA, system speedup and headline remain false after a
setup-only pass.  Formality/PT/PTPX follow only after setup and area admission.

## P0 / P1 / P2

- **P0=1:** M931 fails setup by 4.9058 ns.  It and the CPU/model 1.74x family
  must not be cited as a timing-closed RTL speedup or PPA result.
- **P1=3:** implement only exact F/G/R; prove the expanded VCS gate and replay
  cycles; run one fresh macro-aware DC and enforce timing/area/cycle gates.
- **P2=2:** if setup passes, close Formality and real hold/PT/PTPX; then place
  C1 in the decoder-complete memory-inclusive system table.  Neither is
  authorized by this review.

## Stop conditions

Stop C1 DATE timing iteration if F/G/R changes exact parent/order semantics,
requires `II>1`, exceeds 512 metadata bits, adds a wide payload/port/slot,
falls below `1.70x` versus bit, exceeds either area gate, or fails the single
fresh 3 ns DC attempt.  Do not fall back to a weaker baseline, false-path the
directory, relabel model cycles as RTL, or multiply this component projection
with another component speedup.

