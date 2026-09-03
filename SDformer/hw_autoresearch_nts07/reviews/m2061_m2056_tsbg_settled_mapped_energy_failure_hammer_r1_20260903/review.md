# M2061 / M2056 TSBG settled mapped-energy independent failure hammer

## Verdict

**M2061 failed and is consumed: no retry.** Failure-hammer quality is 98/100;
P0/P1/P2 = 0/2/2. The run produced no mapped completion, SAIF, PTPX, power or
energy result and must never be cited as a success.

The attempt token, M2062 source admission and M2061 failure quarantine all pass
their inner and outer seals. The sealed work-tree fingerprint exactly matches
the still-present raw work tree: 111 members, comprising 109 regular files,
two tool-generated symlinks and 236,958,615 regular-file bytes. The sealed and
raw compile/runtime logs are byte-identical.

## What actually ran

Only ordinary-LRU4 ran: one license preflight, one VCS compile and one mapped
simulation. There was no TSBG compile/simulation and no PTPX invocation.

The exact runtime sequence was:

1. reset and all four contexts x 48 groups completed loading;
2. the last descriptor completed at cycle 383 and 1,164,000 ps;
3. `full_execute_begin cycle=383` was printed;
4. at the next negedge plus 10 ps, time 1,167,010 ps, the first M2061 check
   failed on `ordinary.cycle_count`;
5. no `M2061_SAIF_WINDOW_BEGIN`, wrapper `$stop`, UCLI `power -enable`, SAIF,
   final M2051 PASS or useful request/issue/commit occurred.

M2061 did improve diagnosis over M2058: the first failing logical signal is now
known. It did not print the unknown bit indices of the 32-bit vector, and the
fatal prevented all later counter and payload checks. Therefore this review
cannot claim that only `cycle_count` was unknown.

## Reset-race finding

The inherited source TB contains a real scheduling defect:

```systemverilog
repeat (5) @(posedge clk_core);
rst_core = 0;
```

The blocking assignment releases a synchronous reset in the same active time
slot in which source `always_ff` blocks and mapped cell primitives sample the
clock. The same pattern occurs at both later recovery resets. This is an
invalid testbench protocol and must be changed to a negedge release everywhere.

It is **not**, however, sufficient to call that race the fatal's sole cause.
The mapped counter has a stronger four-state convergence problem.

## Static bit-0 proof

Source RTL assigns `cycle_count_q <= 0` under synchronous active-high reset.
The mapped netlist implements bit 0 with a `DFQD2BWP35P140` having no reset pin;
reset is folded into its D cone. The exact relevant equations are:

```text
n233864 = rst_core
n234281 = rst_core
n234488 = ~rst_core
n287961 = ~(n227719 & n234488)
n287906 = n233864 | n227719
n227749 = ~debug_cycle_count[0]
n109359 = ~((debug_cycle_count[0] & n287961)
            | (n287906 & n227749))
```

With `rst_core=1`, `n287961=1` and `n287906=1`. For a physical binary Q,
`D=~(Q|~Q)=0`, so the synthesized logic is Boolean-correct. For an initially
unknown four-state Q, both Q and `~Q` are X and the same expression is X. The
TSMC `tsmc_dff` UDP has no initialized Q and the mapped flop has no asynchronous
clear. Thus asserted synchronous reset need not make this simulated counter
known, regardless of which edge later releases reset.

This is strong evidence of mapped four-state X-pessimism plus a testbench reset
race. It is not evidence of broken silicon arithmetic or an RTL hardware bug.
No SDF is annotated, the cell model does not consume `UNIT_DELAY`, and the log
contains no timing-violation message.

## M2063 decision

A fresh **M2063 source direction** is conditionally allowed, but this review
does not authorize execution. It may proceed only as:

- a new additive TB identity and new runner/parser/contract/namespaces;
- reset asserted before a sampling edge and released on negedge at all three
  reset sites;
- an explicitly disclosed deterministic two-state mapped initialization such
  as an exact-pinned `+vcs+initreg+0`, or an independently equivalent
  multi-initial-state proof;
- the complete reset, 383-cycle preload, attack/recovery, exact ledger and
  final M2051 PASS flow retained;
- all qualifiers, faults, busy signals and counters still checked
  unconditionally; no unknown counter may be hidden;
- the same M2029 netlists/SDCs, scopes, 20,292/7,569 denominators, UCLI and PTPX
  annotation gates retained;
- independent source hammer before one no-retry P1-serial attempt and an
  independent result hammer before any success claim.

A successor that changes only reset deassertion to negedge is explicitly not
allowed to execute: the bit-0 proof shows why it may fail again. Deterministic
initialization must be disclosed as a zero-delay simulation boundary, never as
silicon power-on behavior or a gate-delay fix.

No EDA, license query or GPU job was launched. No M2061/M2058 evidence, source,
attempt, raw work tree or `docs/359` was modified.
