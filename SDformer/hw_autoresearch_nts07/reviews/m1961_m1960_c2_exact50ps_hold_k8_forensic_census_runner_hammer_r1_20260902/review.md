# M1961｜M1960 C2 exact-50-ps K8 forensic-census runner hammer

## Verdict

**PASS, 99/100, P0/P1/P2 = 0/0/0.** This is a source-only authorization for constructing the M1962 release. It does not authorize a license query, attempt marker, DC, Formality, or PrimeTime run.

The runner is frozen at SHA256 `b6a8d63d716f94cd3e21814f83e8efab61315757733a55eadb41194ff0ed926a` and binds the M1953 failure review at `2c618cd7915d4364077eb426da3c3e2392b81098ab694ed338a2450a2f8e8fe0`.

## Failure-path census

`census()` writes a same-directory temporary file and atomically renames it to `WORK/tool_census.txt`. `finish()` refreshes this file before writing and sealing the terminal failure receipt.

| Injected/inferred path | license observed | DC observed | terminal evidence |
|---|---:|---:|---|
| Failure after WORK activation, before lmstat | 0 | 0 | failure receipt and census agree |
| lmstat failure | 1 | 0 | counters persisted before launch |
| Failure after lmstat, before DC | 1 | 0 | census remains current |
| DC failure | 1 | 1 | counters persisted before launch |
| Raw success | 1 | 1 | raw receipt and census agree |

“Observed” means the sole launch call-site was reached; the counter is persisted immediately before `exec`, so a tool invocation that returns failure is still counted. The immutable attempt marker contains authorized budgets only (plus status, K8 axis, and `retry=false`), not a stale observed snapshot.

## Area and timing gates

- Exactly one anchored numeric `Total cell area:` row is required.
- The independent Python predicate requires a finite, strictly positive area no larger than `137363.9139348 um2`.
- The frozen baseline is `130822.775176 um2`; the ceiling is exactly `1.05x` or `+5.0%`.
- Baseline, post area, ratio and growth are included in the machine summary and raw-success receipt.
- Setup and hold must both be `MET` with zero violating paths; all five design-rule sections must report no violated constraints.

## One-shot and publication checks

The runner remains K8-only with one lmstat call-site, one DC call-site, no retry, same-UID EDA exclusion, memory/commit preflight, owned atomic lock, signal-specific terminal receipts, immutable attempt publication, double-sealed result/failure publication, and exact future M1962/M1963 parsers. The result remains non-citable until a raw run passes an independent result hammer plus transitive Formality and PrimeTime.
