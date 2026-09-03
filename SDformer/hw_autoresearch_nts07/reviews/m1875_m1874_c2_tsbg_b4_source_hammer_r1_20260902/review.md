# M1875 independent fail-closed review of M1874 B4 TSBG source

## Verdict

**FAIL_CLOSED — P0/P1/P2 = 0/1/0; score 87/100.** M1874 fixes all 15 exact M1871 mutations, but its semantic checker still accepts nine newly designed bypasses. M1876 is not authorized. No VCS, simv, EDA, license query, attempt, result, or release was run or created.

## What passed

- CPython 3.6 and 3.12 both pass the official checker and all 36 author tests.
- The original 21 obligations remain 21/21 on both interpreters.
- The exact M1871 attack inventory is now rejected 15/15 on both interpreters.
- M1874 normalizes byte-exactly to M1870 by namespace only, and to M1794 by the declared B4/LRU4 specialization.
- M1866 authorizes only a B4 source, not execution or paper admission.
- Independent ledgers reproduce LRU4 baseline `0/48/44`, candidate `36/12/8`, equal work `576 issue / 9216 product / 24 commit`, signed `-1/0/+1`, the `-(-128)=+128` corner, directed accumulator `[-255,510]`, production bound `98304`, and the 8076-byte source resource model.
- Contract, author, M1871, M1866, and docs/359 identities and seals are intact.

## P1 finding

The checker accepted all nine new attacks under both interpreters:

1. neutralize the baseline Acc24 arithmetic scoreboard;
2. neutralize the TSBG Acc24 arithmetic scoreboard;
3. make the default SVA disable permanently true;
4. make bank-response stability vacuous;
5. make bridge-header stability vacuous;
6. make bridge-payload stability vacuous;
7. make commit-header stability vacuous;
8. make commit-payload stability vacuous; and
9. insert a time-zero forged PASS followed by `$finish` before the workload and protocol proof.

Two positive controls, `BUNDLE=4 -> 8` and candidate hits `36 -> 35`, were rejected. The result is therefore a semantic-coverage failure, not an import or probe-dispatch failure.

## Required next gate

Create an additive immutable successor that structurally proves both arithmetic scoreboards, reset-only default disable, non-vacuous response/bridge/commit stability, and unique causal PASS placement after the final protocol ledger. Add the nine attacks to CPython 3.6/3.12 regression and obtain a new different-author zero-P0/P1 source hammer.

Even if that successor passes, it authorizes only a later, separately reviewed campaign source. A bare M1876 release must not directly execute M1874.
