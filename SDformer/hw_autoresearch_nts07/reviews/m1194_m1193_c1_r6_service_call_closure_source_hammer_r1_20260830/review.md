# M1194 independent hammer of M1193/R6 source

Status: `FAIL_SOURCE_CONTRACT__DO_NOT_AUTHOR_RELEASE__NO_VCS_NO_EDA`.

The exact sealed R6 testbench does close M1192's concrete P0. Starting from
`service_assumption_attacks`, the actual reachable set is the root plus
`force_request_no_core_ready`, `reset_dut`, `release_request`, and
`clear_public_drivers`. Generic `force_request` is unreachable. No reachable
task forces or aliases core ready, and the service-specific helper forces
exactly the nine request valid/tuple fields. Weight-only and psum-only skew are
present, with own service fault required one and peer service, boundary, core,
and composed protocol faults required zero. The 16 assertions, six covers,
seven protocol attacks, two service attacks, 24 legal transactions, 29 legal
mask checks, three reset states, explicit II=2, and normal M935 row/task are
also preserved.

The source milestone nevertheless fails one P1 gate: the mandatory sealed
author checker accepts six adversarial relaxations that this independent hammer
rejects. Its call parser only recognizes a helper call at the beginning of a
line with parentheses, so both a legal bare `helper;` task invocation and a
same-line `statement; helper();` invocation can make a core-ready-forcing helper
reachable without entering the computed closure. It also accepts gating either
peer-service-fault term or the composed `protocol_error` oracle term permanently
false, and accepts deleting one of the nine request forces because it checks a
subset rather than exact equality.

This does not allege a defect in the frozen R6 bytes; it means the claimed
fail-closed source gate is not strong enough to authorize a release. R6 must not
be launched or reused. The minimum repair is a new additive identity whose
checker recognizes both task-call spellings at every statement position,
requires the exact nine-field force multiset, and matches the complete two
service oracles exactly, with rejecting mutations for all six bypasses.

Old R4/R5 reuse and a docs/359 mutation were independently rejected. docs/359
remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
No VCS, `simv`, license checkout, or other EDA command was invoked.
