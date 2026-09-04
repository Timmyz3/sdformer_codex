# M1192 independent hammer of M1191/R5 source

Status: `FAIL_SOURCE_CONTRACT__DO_NOT_AUTHOR_LAUNCHER__NO_VCS_NO_EDA`.

The R4 forensic explanation is plausible: presenting both responses while
forcing the wrapper-side core-ready copy can create issue data outside frozen
M935's genuine request context, and the frozen core fault is ORed into wrapper
`protocol_error`. R5 also contains the intended weight-only and psum-only skew,
plus explicit own-fault-one / peer-fault-zero / boundary-core-composed-zero
oracles.

However, R5 does not satisfy its central no-force contract. Both service tests
call `force_request()`, and that helper executes
`force dut.core_issue_data_ready = 1'b1`. Thus the actual call path is:

`service_assumption_attacks -> force_request -> hierarchical core-ready force`.

The absent peer does naturally keep response ready low, so this force may be
functionally irrelevant to this particular stimulus. It is still present and
directly contradicts the sealed source contract and author receipt. The author
checker missed it because it searches only the textual body of
`service_assumption_attacks`, not its helper-call closure.

No launcher or release may be authored from this identity. The minimum repair
is a service-specific request helper that never forces core ready, or an
equivalently explicit parameterized path with a call-graph-aware static check.
The repaired source needs a new identity and another different-author hammer.
No VCS, `simv`, license checkout, or other EDA command was invoked here.
