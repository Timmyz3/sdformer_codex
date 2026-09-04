# M1220 independent M1219/R9 observability source hammer

Verdict: **PASS for release authoring only. VCS, EDA, release execution, and automatic retry remain unauthorized.**

The exact M1219 identities match: TB `9666e086…`, checker `2639ecfe…`, tests `b365f3b8…`, contract `fd4a23ea…`, and author review/manifest/outer `0aa4b451…` / `3924a2b4…` / `5a7007ce…`. Both contract and author double seals verify with exact membership.

Independent source analysis found no `wait(...)` statement and eight `while` loops, all watchdog-bounded. Four previously unbounded sites are fail-closed: random weight request, optional random psum request, random response acceptance, and every normal preload `prep_ready` row. The clean-reset `prep_ready` check is a separate bounded gate ordered after reset/legal-mask clearing and before normal issue/load.

Seven phase families have unique ENTER/COMPLETE tokens and flushes; the random phase also emits indexed ENTER/COMPLETE pairs for all 24 transactions. Timeout evidence includes wrapper handshakes/counters, boundary/core faults, and frozen-M935 fault, prep, match, and bank state before `$fatal`.

Frozen SHA identities prove R8, M528, M935, M1162, R3 SVA, and docs/359 are unchanged. Workload and claim boundaries remain 24 random transactions, seven protocol attacks, two service-assumption attacks, II=2, one normal row/task, request-ready quiescence, and no timing/cycle/PPA/energy/system claim.

The canonical checker passed; all seven supplied tests passed. Eight independent mutations were rejected: watchdog removal, timeout-dump removal, clean-reset gate removal, phase completion removal, M935 dump-field removal, random-count change, ready-quiesce break, and timing-claim inflation.

No VCS, simulator, synthesis, STA, PTPX, GPU, or remote action was performed by M1220.
