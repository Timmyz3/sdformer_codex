# M1711 independent source hammer

Verdict: **PASS, 97/100; P0=0, P1=0, P2=1.** M1710 closes both release-blocking M1699 findings: the launch-capable runner exact-SHA binds and active-force scans all six frozen direct execution sources at both runtime gates, and `lexists` permanently rejects the M1686/M1700 payload plus both sidecars. The shared lock, post-lock rescan, attempt-consumption order, per-tool rescan and no-retry geometry are intact.

The hammer rejected 15/15 explicit mutations: twelve regular-file or dangling-symlink attacks across both forbidden release triples, plus brace-inline, semicolon-inline and nested Tcl `force` commands. CPython 3.6 and 3.12 independently reproduce all 12 author tests.

One non-blocking P2 remains. Tcl command substitution inside double quotes is executable, while the scanner treats the whole quoted span as inert. Exact-SHA binding at both runtime gates makes that limitation non-exploitable for this frozen M1710 attempt, but the scanner must not be reused as a standalone semantic proof without repair.

This review authorizes only the separate authoring of M1712 for one future attempt. It did not query a license, create a release/attempt/result, or run VCS, simv, SAIF, PTPX or any EDA tool.
