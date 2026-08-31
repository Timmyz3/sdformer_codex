# M1207 C1/R7 acyclic release author receipt

## Verdict

**PASS source-only.** M1207 supersedes the stopped M1204 release without changing any M1198/M1201 RTL, SVA, testbench, filelist, workload, assertion, cover, attack, oracle, or tool identity.

The future M1208 hammer review is recursively sealed but does not embed the hash of its own manifest or outer seal. The launcher instead requires four exact runtime values: release JSON, hammer review, hammer manifest, and hammer outer-seal-file SHA-256. It verifies the recursive seal and all three independently supplied hammer identities before creating the persistent attempt token.

The static gate passed 110 checks. It rejected all missing, truncated, and uppercase environment identities, four self-reference field mutations, and direct embedding of either future self-digest. The exact M1198/M1201 regression remains 16 assertions, 6 covers, 7 protocol attacks, 2 service-assumption attacks, 24 deterministic legal transactions, 29 legal-mask clears, II=2, one normal row, and one normal task.

No VCS, simv, license checkout, EDA, GPU, or network action occurred. A fresh different-author M1208 release hammer is mandatory before the unique M1207 UNIT_DELAY attempt.

Functional VCS, timing, cycles, speedup, PPA, power, energy, system, paper-citable, and headline claims remain false.
