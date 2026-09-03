# M1844 independent source hammer

## Verdict

**FAIL CLOSED — P0=0, P1=1, P2=0, score 88/100.** M1844 authorizes zero M1843 attempts, zero Formality runs, and zero PrimeTime runs.

The exact M1843 runner does contain the intended M1834 repairs: all 13 live RTL sources and the M1811 filelist are exact-bound, authority is checked again immediately before attempt consumption, the future release must bind the M1844 review/manifest/outer triplet, and PT emits plus parses the required reports and raw semantic counts.

The blocker is the checker/test authority. The official 50 tests mutate a source while leaving its contract `source_files` digest stale, so the inventory layer rejects first. With the inventory digest updated consistently, the independent hammer rejects only 57/71 mutations on both CPython 3.6 and 3.12. Fourteen escape; eight materially weaken the second authority check, unique attempt ledger, timing coverage/constraint semantics, or verbatim hold-slack publication.

## Required repair

Create an additive successor; do not edit M1843/M1844. Its checker must directly bind the second just-before-attempt authority call, the attempt call before EDA, the Formality/PT result verification calls, check-timing uniqueness, exact coverage rows and conservation, constraint-count/raw-visibility checks, and verbatim setup/hold WNS writes. Both author tests and the next independent hammer must synchronize `source_files` for source mutations and reject all material attacks on both runtimes.

No EDA, license query, attempt, result, release, commit, push, or frozen-document modification occurred in this review.
