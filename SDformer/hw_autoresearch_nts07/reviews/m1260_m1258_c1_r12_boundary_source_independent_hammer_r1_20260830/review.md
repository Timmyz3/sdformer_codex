# M1260 — independent source hammer of M1258/R12

Verdict: **FAIL CLOSED; release authoring and VCS remain forbidden.** Score
82/100, P0=0, P1=3, P2=0. The canonical R12 TB itself is correctly scoped,
but its checker is not strong enough to guard a release identity.

The author double seal verifies. The canonical checker exits zero and the
declared suite passes 12/12. Independent inspection confirms that every real
synthetic force/release in the canonical TB is on the child
`dut.u_frozen_m935` output seam; no parent `dut.issue_request_*` or
`dut.core_issue_data_ready` force remains. The three normal M935 task bodies
are byte-identical to R11, their real call is present, and the canonical PASS
token honestly says boundary-only, non-integrated random, and no integrated
M935 claim for synthetic traffic.

Fresh nearby mutations exposed four fail-open cases:

1. `issue_request_valid_shadow` passes the child-seam inventory because the
   checker uses `startswith` and an unbounded substring rather than an exact
   signal allowlist.
2. An actual `boundary_only=false` passes if a comment supplies the decoy
   `boundary_only=true` marker.
3. An actual `integrated_normal_m935_evidence=false` passes under the same
   comment-decoy pattern.
4. Commenting out the only `normal_m935_completion()` invocation passes because
   the commented call satisfies the raw-text marker.

The same independent suite confirms useful negatives: executable parent force,
child-comment/parent-actual substitution, integrated-random inflation,
integrated-M935 inflation, and semantic drift in either normal load or service
tasks are rejected. A parent-force comment remains correctly inert.

Only an additive M1261 checker/tests repair is permitted: exact child signal
allowlisting, executable PASS-field parsing, and exactly one executable normal
completion call. The R12 TB, DUT/M528/M935/M1162/SVA, and docs/359 must stay
frozen. No VCS, simv, EDA, GPU, remote, performance, or paper claim follows.
