# M1208 independent M1207 C1/R7 acyclic release hammer

**Verdict: GO, 100/100, P0=0, P1=0.** The exact M1198/M1201 source
corpus, the M1207 runner/checker/contracts, and all recursive authorities are
sealed and unchanged. Independent checks reject environment, self-reference,
gate-order, UNIT_DELAY, count, oracle, timeout and claim-boundary mutations.

The acyclic protocol is sound: the review contains no self manifest/outer
identity, while review, manifest and outer-seal-file hashes must arrive as
three independent environment values and are verified before the persistent
attempt token. Exactly one foundry UNIT_DELAY compile and one bounded simv run
are authorized. No VCS, simv, license, EDA, GPU, or network action occurred in
this hammer.

This is source/release authorization only. Functional VCS, timing, cycles,
speedup, PPA, power, energy, system speedup and paper citation remain false
until a future sealed result passes a fresh different-author result hammer.
