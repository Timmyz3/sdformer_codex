# M852 / M849 C1 R20 source hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

The R19 failure is a testbench epoch-order self-attack, not admitted RTL functionality. R20 changes exactly the complete P2 epoch consumer triplet from epoch 3 to epoch 14: `build_reference`, `load_task`, and `wait_done`. The normal frontier remains epochs 1/2/4/10/11/12/13, no reset is inserted, and P2 remains after the full 13-cover normal gate and before the held-final and six-attack suites.

Independent checks passed:

- TB r8 to r9 is exactly three literal changes and nothing else.
- RTL r2, SVA r2, macro adapter, binding plan and foundry UNIT_DELAY model are byte-frozen.
- The normalized compile/run/functional-gate tail is byte-identical to R19.
- All 102 `require_regular_sha` literals are lowercase 64-hex and the exact pre-mkdir dry run crossed their live-match gates.
- Function closure passed (35 definitions, 281 calls, 21 pinned external commands); all three negative mutations failed closed.
- Fake simv exercised fast, TERM, TERM-to-KILL and tee-failure paths with no orphan; the real command remains `timeout ... 300s ./simv -no_save`.
- Stub execution reached the live VCS/license boundary with zero live probes, zero compile/simv, and zero result/attempt creation.
- `docs/359` remains `dedde7ce...`.

This source hammer authorizes the fresh candidate hammer next. A release may be authored only after that candidate hammer passes. It does not authorize VCS, simv, license queries, EDA, or direct launch.
