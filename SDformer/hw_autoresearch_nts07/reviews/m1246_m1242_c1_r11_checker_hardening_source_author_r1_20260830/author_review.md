# M1246 — R11 checker/tests-only control-flow hardening

Status: **checker source GO; fresh independent hammer required before any
release authoring.** The R11 TB remains byte-identical at
`850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`.
No candidate TB, DUT, M935, M1162, SVA, workload, or frozen document changed.

## Repair

The executable audit now blanks SystemVerilog line comments, block comments,
and string bodies while preserving offsets and newlines. Task extraction and
ordered control checks therefore cannot be satisfied by diagnostic prose,
comments, or `$display` strings.

Within `random_legal_transaction`, the checker now requires exactly one
executable request-window enable and exactly one executable disable. The
enable must precede `force_request` and the bounded exact-fire loop; the sole
disable must be the frozen request-ready retirement assignment and must precede
response backpressure. An immediate zero overwrite therefore creates an
illegal third write and fails.

The response hold budget similarly requires one and only one executable write,
the positive `1 + prng_q[9:7]` assignment. It must textually dominate the sole
`repeat (hold_cycles)` body, with no intervening or later overwrite accepted.
The response-retirement checker also retains the original no-extra-sampled-edge
gate after comment stripping.

## Tests

All 18 M1239 tests remain and pass. The four M1242 nearby counterexamples now
fail closed: immediate window disable, immediate hold-budget zeroing,
comment-only window enable, and comment-only hold loop. Two additional string
decoys for the same executable anchors also fail closed. Total result:
**24/24 PASS** (one canonical positive plus 23 negative mutations).

## Boundary

This is checker/tests-only evidence. It does not authorize release publication
or prove functional VCS. No VCS, simv, EDA, GPU, or remote work was performed.
Timing, cycles, speedup, PPA, energy, system speedup, headline, and paper
admission remain false.
