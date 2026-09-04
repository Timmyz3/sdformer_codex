# M864 / M863 C1 R21 source hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

R20 remains permanently `FAILED_DO_NOT_CITE`. Its normal and P2 gates passed, but the held-final testbench sampled `ready` after the legal transfer had already retired the row. R21 is a new TB-only identity: it observes `valid && ready` before the accepting edge, holds the authoritative slot and sinks through exactly one positive edge, releases on the following inactive edge, then requires exactly one psum commit and one row completion before emitting a dedicated recovery cover/token.

Independent checks passed under Python 3.6.8 and 3.10.16; Python 3.12.13 was also run as an additional compatibility check:

- TB r9 to r10 reconstructs byte-for-byte after removing only the held-final event-order repair, its one counter declaration/initialization, and its exact-one cover gate.
- The canonical synthetic order passes. Four mutations—post-accept `ready` sampling, release before accept, double accept, and accept without pre-edge observation—fail closed.
- RTL r2, SVA r2, parent-scratch macro, binding plan and foundry model remain byte-frozen.
- The 13-cover normal gate, epoch-14 P2 triplet and minima, held-final phase, all six exact attacks, and final token remain ordered and unweakened.
- All 110 logical `require_regular_sha` path edges use lowercase 64-hex literals and reached live match in the exact dry-run. The sole repeated digest is the intentionally identical R19/R20 `FAILED_DO_NOT_CITE` marker at two distinct paths.
- Function closure passes with 36 definitions, 297 custom call sites and 21 pinned external commands. Three closure mutations fail closed.
- The exact embedded M770 predicate passes; missing-key and wrong-value mutations both fail under Python 3.6 and 3.10.
- Fake simv covers fast return, TERM, TERM-to-KILL and tee failure without orphan. The pre-mkdir stub reaches its live VCS/license boundary with zero VCS probe, license query, compile, simv, result or attempt creation.
- `docs/359` remains `dedde7ce...`.

This review authorizes only a fresh independent hammer of the closed launch candidate. A release may be authored only after that candidate hammer passes. It does not authorize VCS, simv, a license query, or any EDA action, and it establishes no functional, timing, cycle, speedup, PPA, energy, system or paper claim.
