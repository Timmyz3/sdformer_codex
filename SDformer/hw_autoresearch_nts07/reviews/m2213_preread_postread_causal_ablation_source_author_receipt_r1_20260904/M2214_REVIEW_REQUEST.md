# M2214 independent source-hammer request

Review M2213 read-only. Do not run VCS, simv, a license query, any EDA, GPU work, or Git; do not modify M2018, M803, M2213 sources, or `docs/359`.

Required checks:

1. Recompute every source identity and prove frozen M2018 remains `96fb3557...` and `docs/359` remains `dedde7ce...`.
2. Reject escaped `\`backtick` tokens and inspect the complete RTL/TB/SVA interface for compile/elaboration hazards.
3. Prove ordinary is frozen token-major M2018, pre-read is frozen group-major M2018, and post-read is an additive group-major LRU4 control with the same external ports and private Acc24 behavior.
4. Prove a post-read hit enters `ST_FETCH_REQ`, accepts 12 bundle and 96 bank requests/responses, validates returned identity, never writes the returned payload into a valid hit row, and only then bridges the resident row.
5. Prove the directed scoreboard independently checks tag/context/slice/terminal/Acc24 on all three axes, equal issue/product/commit counts, physical per-bank request/response counts, and `postread - preread == causally suppressed reads`.
6. Re-run the static test and independently mutate the hit bypass, response identity, request conservation, axis presence, parser ledger, and retry gates.
7. Prove M2215 result/attempt/lock are virgin and the one-shot runner cannot execute without an exhaustive double-sealed M2214 scoring at least 95 with P0/P1/P2 = 0/0/0.
8. Preserve the physical fairness boundary: the post-read-only debug counters prohibit any current matched-area claim. Future area work must equalize counters across all axes or use a separately reviewed counter-stripped configuration.

Only an exhaustive double-sealed PASS may authorize exactly one M2215 license query, VCS compile, simv run, and parser run. Expected status:

`PASS_M2214_M2213_SOURCE_HAMMER__M2215_ONE_SHOT_VCS_AUTHORIZED`

M2213/M2214 are source-only and establish no performance, PPA, power, energy, or paper claim.
