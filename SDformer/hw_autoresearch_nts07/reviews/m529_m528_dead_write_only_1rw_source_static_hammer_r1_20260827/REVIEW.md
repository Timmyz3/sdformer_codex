# M529 / M528 complete-source independent static hammer

Verdict: **FAIL, 70/100, P0/P1/P2 = 1/4/2.** No VCS launch admission is authorized for this source identity. This was a read-only static review; no HDL, EDA, CPU, or GPU job was run.

The datapath structure is not a placeholder. The 64-row M504 matcher, ping-pong ownership, stable population/row ordering, single earliest-parent lookahead, no-consume-credit two-entry queue, dead-write-only scratch policy, signed arithmetic, atomic architectural outputs, and nine foundry-macro slices are all explicitly present. Combined PVRF, concurrent 1R1W, second lookahead, decoder/full-network scheduling, and register-array scratch fallback are absent.

The blocking defect is fail-closed atomicity. A parent-only synthetic beat with nonzero payload raises `fault_condition_w`, but that predicate is absent from `base_issue_ready_w`; the beat can still commit psum/row completion and a live write in the same edge. In addition, overflow can be computed and latched from stale `slot0_data_q` before the matching parent response is ready.

Verification is also insufficient to authorize VCS. The cleanroom scoreboard does not independently derive parent refcounts/live rows or expected macro event counters, so a common-mode liveness error can pass. Required SVA covers are not turned into pass/fail gates. The runner only requires its PASS token. Finally, the small functional VCS test must be kept separate from frozen-trace recurrence: the already sealed M528-r4 CPU DSE remains a precondition, a repaired unit VCS attempt proves protocol/function, and a later independently admitted trace-driven RTL/cycle receipt owns the 435,293,339-cycle recurrence and the two 1.50× comparisons.

Because the prior author admission was consumed by this package, the minimum recovery is a new bounded repair-only author admission, edits limited to the top/SVA/TB/runner plus a revised source contract and handoff, then a fresh independent source-static hammer. Matcher/order/queue/dead-write policy/arithmetic/macro/claim identities must remain unchanged. Full findings and line evidence are in `review.json`.
