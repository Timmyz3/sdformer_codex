# M868 Python-3.10-only full-first-row source-author handoff

M868 is the minimum additive successor to M865-P1-1. It changes no M861, M785, M768, payload, resource, scheduling, or cycle-class semantics. The only authority change is explicit runtime binding: every Python action must name `/opt/anaconda3/envs/pytorch310/bin/python3.10`, version 3.10.18, SHA256 `9f78cd42...`; ambient `python3`, a Python shebang, and PATH fallback are rejected.

The source-only identity contains a no-work dry-run and a future exactly-once runner for only `M854_FIRST_D0_A1_T0` (`M686_ZURICH_CITY_09_A_S10`, first normalized record, module 0, sample 0, A1_OSG, timestep 0). The expected 9,582,057 compressed transactions and 38,672,612 expanded requests are cardinality gates. The runner cannot launch until a new fixed-path M869 review is double sealed, scores 100 with P0/P1/P2 all zero, grants exactly one nonproduction diagnostic, and its review and outer-seal SHAs are supplied by the caller.

The runner checks canonical and prefix collisions, then requires 2 GiB free disk and 96 GiB MemAvailable/commit headroom before consuming a one-way attempt. Attempt and result publication use sealed private stages and `renameat2(RENAME_NOREPLACE)`. Any post-attempt failure moves partial state to a unique quarantine and emits a double-sealed fail-closed receipt; retry and attempt restoration are forbidden.

Author validation passed 5/5 source-only tests and the exact no-work dry-run. The dry-run created no files, consumed no attempt, enumerated no request, and did not require or synthesize the future hammer. No full row, population, production, cycle/speedup result, VCS, DC, PT, FM, EDA/license, GPU, remote, or training action ran. M865 remains a 92/100 fixed-path failure and is not treated as admission evidence.

The next and only gate is a fresh independent M869 source hammer. Even a future successful full-row diagnostic remains nonproduction and noncitable, requires a fresh result hammer, and does not complete the decoder population or Table A.
