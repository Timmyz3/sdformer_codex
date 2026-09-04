# M2206 independent source review request

Review the fresh additive M2205 source chain. Do not execute `lm_shell`,
`lmutil`, any EDA tool, or a GPU workload.

Required independent checks:

1. Bind the failed M2190 review and confirm M2182/M2191 remain permanently
   unauthorized with zero run artifacts.
2. Verify every source and tool identity, the 1,051-member Milkyway manifest,
   `docs/359`, contract sidecar, and exhaustive author-receipt seals.
3. Confirm the Tcl gate is the first executable phase and cannot create or
   modify a frame before monitor release.
4. Confirm the monitor uses Python-native polling, waits for one stable exact
   `lm_shell_exec` plus its exact Tcl PID marker, and checks frame/Milkyway
   absence before release.
5. Confirm the evidence is explicitly limited to sampled live processes. It
   must not claim exhaustive capture of micro-short wrapper helpers.
6. Confirm every sampled post-gate descendant below actual LM is either the
   pinned actual identity or the single pinned Milkyway identity; attack it
   with an additional sampled helper, reparenting, duplicates, environment
   drift, and claim widening.
7. Confirm before/after same-UID censuses, exact command/log/result/output
   manifest closure, one native control/mutation, 25 process mutations, and
   five full-receipt mutations.
8. Confirm one regular `lm_shell` command, one license preflight query, one
   conversion command, no retry, no design import, and no P&R.

Only an exhaustive double-sealed M2206 review with score at least 95 and
P0/P1/P2 = 0/0/0 may authorize the single M2207 run. M2208 must independently
review any raw M2207 result before it supports a compatibility claim.
