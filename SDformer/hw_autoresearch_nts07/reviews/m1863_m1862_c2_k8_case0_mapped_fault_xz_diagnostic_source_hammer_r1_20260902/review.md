# M1863 independent hammer of M1862 diagnostic source

## Verdict

**FAIL CLOSED — P0=0, P1=1, P2=0. Do not create M1864.**

The exact M1862 source is narrowly scoped and internally consistent. Its sealed identity, author receipt, failed M1857 predecessor, K8/case0 workload, one-compile/one-simulation future budget, no-UCLI/no-SAIF/no-PTPX boundary, and frozen docs/359 digest all verify. The official checker and 31 tests pass on both CPython 3.6 and 3.12. All twelve synchronized-inventory mutations that escaped M1856 are now rejected on both runtimes.

The remaining blocker is direct control-flow binding. With each mutated source digest synchronized into the contract inventory, four new mutations escape on both runtimes:

- guard `verify_authority()` with Python `if False`;
- guard `ATTEMPT.mkdir()` with Python `if False`;
- guard the only compile `run()` with Python `if False`;
- guard the first diagnostic `$finish` with SystemVerilog `if (1'b0)`.

The checker currently counts AST nodes, source lines, and tokens without proving these critical actions are direct executable statements. Therefore it cannot yet authorize a launch even though the checked-in source itself is well formed.

## Required successor

Keep sealed M1862 unchanged. An additive successor must bind the authority/freshness/lock/attempt/tool/parser/publication sequence to direct statements in the expected block, forbid constant/dead control ancestors, and require the matching `$display` plus unconditional `$finish` as direct terminal statements in every stop task. Add all four synchronized mutations to both-runtime tests, then obtain a new different-author review and a new release label. M1864 is not authorized by this review.

No EDA, simulator, license, attempt, result, release, source/predecessor write, docs/359 write, or `ucli.key` access occurred.
