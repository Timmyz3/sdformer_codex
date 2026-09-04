# M1247 — independent hammer of M1246 R11 checker/tests hardening

Verdict: **PASS; fresh disjoint release authoring is authorized.** Score:
**100/100**, P0=0, P1=0, P2=0. No further checker expansion is warranted
without a relevant new failure.

## Independent result

The R11 TB remains byte-identical at
`850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`.
M528, M935, M1162, the R3 SVA, R10 TB, and `docs/359` retain their frozen
hashes. Both layers of the M1246 author seal and source-contract seal verify.
The canonical checker exits zero, and the exact declared inventory of 24 tests
passes 24/24.

An independently implemented bounded SystemVerilog lexical blanker agrees
byte-for-byte with M1246 on the complete candidate. It also agrees on a
purpose-built probe combining a multiline block comment, line comment, and a
string with an escaped quote. Input length and every newline offset are
preserved, so statement offsets remain comparable.

The independently stripped `random_legal_transaction` has exactly two
executable request-window writes: one enable and one retirement disable. The
positive enable precedes request forcing and the exact-fire loop; the sole
disable is in the frozen ready-retirement sequence before response
backpressure. `hold_cycles` has exactly one executable write, the positive
`1 + prng_q[9:7]` assignment, and it dominates the sole
`repeat (hold_cycles)` body with no intervening overwrite.

## Nearby counterexamples

Eight fresh in-memory attacks were all rejected:

1. request-window enable only in a block comment;
2. request-window enable only in an escaped-quote string;
3. request-window enable and immediate disable on one source line;
4. request-window enable moved into a different task;
5. positive hold assignment only in a block comment;
6. positive hold assignment only in an escaped-quote string;
7. positive hold assignment and zero overwrite on one source line;
8. positive hold assignment moved into a different task.

These cover the requested comment/string stripping, task boundary, statement
order, unique window lifetime, and positive-hold dominance properties. No
mutation was accepted with an empty error list.

## Boundary

This is checker/tests-only evidence. It authorizes only a new, disjoint release
authoring step. No candidate TB, DUT RTL, SVA, workload, or frozen document was
changed. No VCS, simv, EDA, GPU, or remote command was invoked. Functional VCS,
timing, cycles, speedup, PPA, energy, system speedup, headline, and paper
admission remain false.
