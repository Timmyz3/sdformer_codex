# M1057 independent M1056 exact-1RW source hammer

## Verdict

`STOP_M1057_M1056_C1_EXACT_1RW_SOURCE_HAMMER`

`STOP_FULL_REPLAY_RELEASE__ADDITIVE_SOURCE_REPAIR_REQUIRED`

The core 1RW scheduling primitive is sound on the directed cases, but the
future three-design/full-replay boundary is not fail-closed. M1056 must not be
released for the 51.84M-row replay in its current form.

## What passed from first principles

The independent oracle reproduced four packed groups with one 1RW port per
group. Two and three same-cycle accesses serialize deterministically by
arrival cycle then program order. Different physical addresses still collide
on the shared group port, and a same-address read cannot pass the preceding
write.

The address organization is internally coherent: each pair of logical banks
maps to one 128-deep group with
`address=(bank mod 2)*64+row`. Fifteen 128-bit macros provide the 1,824-bit
logical wide row for each group, so four groups imply sixty macros and 122,880
physical psum bytes. This is only the psum component of the old 214,912-byte
capacity hypothesis.

Backpressure also works in the directed cascade. Two span-one tasks change
from nominal 20 cycles to 22 cycles. Task starts are `[0,11]`, nominal ends
are `[8,19]`, and effective ends are `[9,20]`. The result is therefore not
computed as old cycles plus a flat conflict count.

## Why the hammer stops release

The common-service proof compares only caller-provided receipt streams against
one another. Three empty streams return PASS, although they cannot equal the
frozen M1016 stream or its authority digest
`a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea`.
M1056 also accepts mismatched task populations across the three designs: one
candidate task, two strongest-zero tasks with different rows, and one bit task
with task ID 99 all pass when paired with one common caller receipt.

Capacity is similarly caller-authoritative. Passing `capacity_bytes=0`
returns `capacity_bytes_pass=true`; the 122,880-byte psum, weight and
parent/other stores are not rederived inside the executable path. Finally,
`validate_source_contract` accepts an unsealed temporary JSON carrying only
the expected status/launch fields, and boolean service counts pass the Python
integer check.

These are release-boundary failures, not defects in the one-port arbitration
kernel. They nevertheless permit a future wrapper to report a matched and
capacity-feasible result for the wrong workload/resource coordinate.

## Required additive repair

The successor must pin the frozen M1016 row/source identity, exact task
population, canonical service totals and service digest. All three designs
must share task IDs/order, psum rows/address multiset and sample/commit
boundaries, with only named preprocess/work fields allowed to differ. The
214,912-byte capacity must be rederived internally, sealed contract identities
must be checked by the called production entry, and booleans must be rejected
as counts. The five reproduced attacks belong in the next directed suite.

No full replay, EDA, GPU or remote job was run. M1056 and `docs/359` were not
modified. No capacity, matched-cycle, speedup, RTL, PPA or paper claim is
admitted.
