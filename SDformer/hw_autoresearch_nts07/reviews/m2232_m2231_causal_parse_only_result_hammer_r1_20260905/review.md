# M2232 independent result admission

**PASS, 98/100; P0/P1/P2 = 0/0/0. Directed function and pre-read request causality are admitted.**

The independent recomputation uses the unchanged raw compiler and simulator logs directly and compares all receipt fields. Seven directories pass exhaustive double-seal checks, including the new result and consumed attempt. All 16 pinned inputs and original M2213 source hashes are unchanged. The original M2215 failure marker remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE`. M2231 records one CPU parse and no additional EDA or license query; its stdout contains one PASS and stderr is empty.

| Directed axis | Accepted 128-bit bank reads | Raw cycles | Exact commits | Signed products |
|---|---:|---:|---:|---:|
| Ordinary | 2,304 | 3,386 | 24 | 4,608 |
| Post-read gating | 2,304 | 3,386 | 24 | 4,608 |
| Pre-read TSBG | 576 | 1,119 | 24 | 4,608 |

All outputs match the golden values. The pre-read reduction is exactly 75%. The 1,728 avoided reads equal the actual post-read-hit bank requests and responses; 216 post-read bundles also have 216 checked identity responses. All three SVA cover counts reproduce, with 3,443 attempts each and no observed assertion failure.

Minimum citable conclusion: **In a directed B4 six-group experiment with common SRAM ports and backpressure, ordinary and post-read schedules each issue 2,304 bank reads, while pre-read TSBG issues 576, a 75% reduction, with identical products and exact commits. The ablation attributes the saved requests to suppression before SRAM access.**

The workload is one deterministic directed scenario. The cycle counts may be shown with that scope, but they do not establish ep34 population acceleration, mapped timing, equal-area speedup, power, energy or a paper headline. This review recovers a new result identity; it does not relabel the failed M2215 pipeline as successful. The immutable M2231 receipt remains pending-labelled, and this separate review is its bounded admission authority.

No production parser was rerun, no new tool process was launched, and no source, raw log, result or frozen document was edited. The review score expresses confidence in this result's stated scope, not an estimated TCAS-II acceptance score.
