# M694 fresh hammer: M519-R9 three-axis DC release

Verdict: **GO for exactly one DC-only attempt after an immediate live recheck**, score **98/100**, P0/P1/P2 = **0/0/1**.

The additive repair closes M580's only P1: `candidate_hammer_status` now equals the sealed M576 `review.status` verbatim, and the R9 runner checks the M576 review, manifest, outer seal and exact status before preflight. R8 and M580 remain unchanged. Runner, contract and admission SHA identities close, while the canonical result and attempt sentinel are new and absent.

The frozen three-axis workload is K1/K8/K1x8 at 3.000 ns. Existing VCS evidence remains sealed and admitted; 17 exact files, the 12-file RTL list, SDC, DC entry/wrapper/actual ELF, both TSMC28 DBs and docs/359 all rehash correctly. No EDA was run by this review.

P2-1 discloses a foreign UID 1909 `simv` process. It was not signaled or altered. In three 10-second samples, current UID 1913 had zero EDA collision and commit, available-memory and swap gates all passed. The snapshot never replaces the runner's own preflight; root must recheck immediately before the single command recorded in `review.json`.
