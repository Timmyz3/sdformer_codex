# M1868 independent source review: M1867 C2 K8 mapped X/Z diagnostic

Verdict: **FAIL CLOSED (P0/P1/P2 = 0/1/0). Do not create M1869.**

The sealed M1867 source does close both known predecessor defects. On CPython 3.6 and 3.12, the official checker and all 35 tests pass; the independent synchronized-inventory hammer rejects all twelve M1857 attacks and all four M1863 constant-false control-ancestor attacks. The exact four SystemVerilog stop tasks each bind the sampled public/endpoint value to a direct display followed by an unconditional `$finish`. The package remains diagnostic-only: one future compile, one future simulation, no UCLI/SAIF/PTPX, no power/performance/paper claim.

One new P1 remains. The Python checker excludes only **direct** terminal statements before publication. It accepts all four independent mutations below while their source hashes are synchronized into the contract:

1. `if True: return 0` before `verify_authority()`;
2. `if True: raise ...` before `ATTEMPT.mkdir()`;
3. `if True: return 0` before the compile `run()`;
4. `if True: return` inside `run()` before `subprocess.run()`.

The counted critical actions remain direct and ordered, but are unreachable. M1869 therefore cannot be authorized. An additive successor should reject terminal-containing compound predecessors in both `main()` and `run()` (or enforce a strict allowed statement-shape prefix), add these four attacks to the dual-runtime suite, and undergo a new different-author review and fresh release label.

No runner, EDA, VCS, simv, license query, attempt, result, release, UCLI, SAIF, PTPX, `ucli.key`, commit, push, or docs/359 mutation occurred in this review.
