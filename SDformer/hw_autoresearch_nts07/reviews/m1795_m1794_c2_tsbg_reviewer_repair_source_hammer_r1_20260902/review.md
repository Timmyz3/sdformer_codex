# M1795 independent source hammer of M1794 TSBG

Verdict: **FAIL-CLOSED, 84/100, P0=0, P1=2, P2=0. No VCS or EDA is authorized.**

The substantive M1788 repairs are source-visible and internally consistent. The fixed production point is `SOURCE_GROUPS=48`, its conservative absolute Acc24 bound is `48*16*128=98,304`, the directed G12 bound is 24,576, and both directed DUT tuples are legal. The TB captures an actually accepted bank-3 epoch/slot/generation/tag and all 16 signed-INT8 payload lanes, waits for bundle retirement, replays that exact tuple on bank 3, requires zero acceptance and sticky protocol/stale state, then performs a three-clock reset and a complete legal B8 service through requests, reordered responses, the typed bridge, Acc24 commits, and eight terminals. The SVA accepts a 1–8-cycle reset and covers a later clean terminal. CPython 3.6 and 3.10 both pass the author checker and all 14 author tests. No EDA or license action was run.

Two governance defects still block launch:

1. M1794 has no one-shot runner or release object. Consequently there is no exact runner/review/release SHA pin, attempt latch, one-compile/one-sim budget, collision gate, no-retry rule, or atomic publication boundary. The contract correctly says a separate exact release is required, but an absent execution path cannot itself be authorized as the sole directed VCS campaign.
2. The author suite has only three semantic source mutations, all on the parameter predicate. The replay and recovery tests are token-presence/unit checks; they do not mutate and reject slot, generation, tag, payload, zero-accept, full post-reset service, or SVA reset/terminal logic. Contract SHA rejection is identity protection, not proof that the checker detects a semantically weakened successor.

Required correction: add an immutable one-shot runner plus future release schema that binds the exact runner, M1794 source contract seals, this review successor, docs/359, an all-false prelaunch boundary, exact one-compile/one-sim budget, attempt uniqueness, no retry, and no-replace publication. Add semantic mutations for all replay identity/payload fields, the zero-accept/sticky checks, reset duration, full legal recovery ledgers, and the SVA terminal-after-reset obligation. Then obtain a fresh different-author review. Do not modify this M1795 receipt or overwrite M1794.

This verdict does not kill TSBG. It says the repaired RTL source is plausible but has not yet earned the uniquely governed VCS launch needed for admission.
