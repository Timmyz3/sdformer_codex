# M1193 R6 service helper call-closure source author receipt

Status: `PASS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA`.

M1192 correctly rejected R5: both service tests still reached the generic
`force_request()` helper, which forced `dut.core_issue_data_ready=1` even though
the service task body itself contained no such statement. R6 is a new,
non-overwriting source identity. It adds `force_request_no_core_ready()` and
routes exactly the two service-negative tests through that helper. The helper
forces only the nine request valid/tuple fields and never forces core ready.

The author checker now constructs the task-call closure rooted at
`service_assumption_attacks`. The exact reachable set is the root plus
`force_request_no_core_ready`, `reset_dut`, `release_request`, and
`clear_public_drivers`; generic `force_request` is unreachable. Every reachable
`force` target must belong to the nine-field request allowlist, and aliases are
forbidden. New mutations cover a call changed back to the generic helper, a
direct core-ready force inserted inside the service helper, and an aliased
core-ready force. All are rejected.

R5's useful repair remains intact: weight-only and psum-only skew naturally hold
the attacked response without a joined core transaction. The oracle requires
own service fault one, peer service fault zero, and boundary/core/composed
protocol faults zero. Frozen RTL and R3 SVA are unchanged. The complete 16
assertions, six covers, seven protocol attacks, two service attacks, 24 legal
transactions, 29 legal mask-clear checks, three reset states, explicit II=2 and
normal frozen-M935 row/task remain present.

Static author testing passed 934 checks and rejected 11 mutations. No VCS,
`simv`, license checkout, or other EDA tool was invoked. A fresh different-author
hammer is mandatory before any launcher or release may be authored.
