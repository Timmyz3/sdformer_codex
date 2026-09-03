# M1957 independent M1956 TSBG SVA-fail-closed runner hammer

Verdict: **PASS source-only; authorize release construction only**. This review ran no license query, attempt, VCS, simv, DC, or PT.

M1956 closes the admission-critical M1948 hole. It requires exactly one directed-TB PASS token and separately rejects all of the following in `simv.log`:

- the installed VCS concurrent-property signature `: started at ... failed at`;
- broader `Assertion ... failed` text;
- `Error-[SVA` diagnostics;
- `$error`, `$fatal`, and `Fatal:` diagnostics.

The exact ERE rejects both repository historical false-pass logs identified by M1948, including logs where a TB PASS followed an SVA failure. It does not reject a clean cover-reporting reference log containing ordinary cover-property `attempts`/`match` lines.

The installed VCS V-2023.12-SP1 documentation recognizes `-assert global_finish_maxfail=N`, says it needs neither `enable_diag` nor `enable_hier`, and defines it as global termination at the Nth SVA failure. M1956 supplies `N=1`, requests SVA compilation, and does not apply `no_fatal_action`. The post-simulation signature check is independent, so publication remains fail-closed even if runtime termination behavior is not itself observed during this source-only review.

All prior governance remains intact: exact source/review/failure SHA pins, inner and outer seals, frozen docs/359 identity, clean environment, scoped `VCS_HOME`, fresh namespace, same-UID EDA exclusion, memory headroom, attempt-before-license ordering, one license query, one VCS compile, one simv run, no retry, signal-safe failure quarantine, no-replace publication, and double seals. The future M1958 release and M1959 audit parsers bind exact schemas, statuses, identities, budgets, and gates.

No execution, speedup, PPA, energy, system, or paper claim is admitted. The next legal action is to create M1958 and obtain an independent M1959 PASS; this review alone authorizes no EDA attempt.
