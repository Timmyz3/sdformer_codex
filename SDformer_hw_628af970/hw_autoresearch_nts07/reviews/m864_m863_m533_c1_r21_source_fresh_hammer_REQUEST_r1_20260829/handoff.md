# M864 / M863 C1 R21 source handoff

M863/R21 source package is ready for a fresh independent source hammer. The only functional delta from frozen TB r9 is inside `test_held_final_stale_parent_then_legal`: observe `valid&&ready` before the accepting edge, hold all forces through exactly one posedge, release at the following negedge, never resample ready after acceptance, then require exactly one psum and one row completion and emit the dedicated recovery token/cover.

RTL/SVA/macro/binding/foundry identities and all 13 normal covers, P2 gates, six attacks and the final token are frozen. The exact-diff test, synthetic event-order model with four rejected negative mutations, runner closure mutations, fake timeout and pre-mkdir zero-side-effect dry run all pass. No EDA or license action was performed.

R20 remains permanently `FAILED_DO_NOT_CITE`. This package is a new R21 identity, not a retry. A different reviewer must author the requested PASS100 source hammer before any candidate/release chain or live attempt.
