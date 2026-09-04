# M1226 — C1 R10 additive TB service-boundary repair

Status: **SOURCE GO for a fresh independent source hammer only.**  VCS, simv,
release publication, EDA, GPU, remote work, and M1221 retry remain forbidden.

R10 is a new TB identity over immutable R9.  M528, M935, M1162, R3 SVA,
`docs/359`, workloads, attacks, II=2, and the row-0 `16'h0003` normal workload
retain their frozen SHA/semantics.

The normal service task now proves one request fire before withdrawing both
request-ready inputs at a negedge.  Responses are driven only afterward, held
stable until the exact response counter increment, and withdrawn at the
immediately following negedge.  There is no extra response posedge.  Beat two
cannot enter until response count advanced once, both response valids are low,
and `request_active_q` is retired.  Normal request, response, and task watchdogs
all call the inherited full state dump before fatal.

The canonical static audit passes.  Eleven tests pass: one canonical positive
and ten fail-closed mutations covering ready retirement, one-fire overshoot,
restored extra response posedge, unstable response, removed beat-retirement
gate, missing request/response timeout dump, missing zero-SVA gate, workload
mutation, and claim inflation.

Exact new identities:

- R10 TB: `f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b`
- checker: `708703b01babf9bcfc9915e72874d2167f2ef7f45cac3d4276ab8d541bfaf0e2`
- tests: `bb351955023c0bfcd273a8c48c2833090b2ca521ef1aeaaf9e39ae5a0279c535`

Any later launch must require zero unmasked SVA failures and the normal M935
completion token.  This author package does not publish or authorize a release.
