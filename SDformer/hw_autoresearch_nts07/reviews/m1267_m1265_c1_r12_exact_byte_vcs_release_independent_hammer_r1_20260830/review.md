# M1267 M1265 exact-byte release independent hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

The source-only hammer passed 113 checks and bound the exact runner, filelist,
checker, tests, contracts, source-author recursive seal, and M1265a exact-TB
reachability PASS. Requested adversarial classes all fail closed: any corpus
byte drift, missing/malformed external pin, duplicate compile/simulation,
timeout/quarantine/attempt-gate deletion, claim inflation, old TB/filelist
seepage, and existing result/work/failure namespace.

The frozen runner hard-codes an M1266 review path/schema although global M1266
was later used by an unrelated read-only audit. A separately sealed compatible
alias directory contains the exact required schema/status; both directories
carry the same `alias_binding.json` and identify M1267 as the true fresh author.

No source was changed and no VCS, simv, EDA, GPU, or remote action was run. The
only authorization is one future exact-pin M1265 compile and one simulation,
without retry. A PASS remains boundary-only functional evidence, not timing,
cycle, PPA, power, energy, system-speedup, or paper evidence.
