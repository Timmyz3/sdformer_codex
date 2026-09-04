# M1550 independent rehammer of M1549

Verdict: **NO-GO** for the single-call non-product pilot.

The M1549 repair correctly rejects the original module-2/call-7 witness, a
plain custom call-0/D0 plane, foreign same-shape files, hardlinks, symlinks,
SHA/shape mutations, product configuration, pilot/production CLI and all 120-
call production routes.  The full 122-member M1521 identity, three non-product
configurations, scheduler calendar/outstanding/counter/digest state, nine-slot
weight cache and RSS gate were independently rechecked on Python 3.10 and
CPython 3.6.

Two stronger P0 witnesses remain.  First, an exact `MmapLittleBitPlane` can
open corrupt foreign bytes and then mutate its public `path` and
`expected_sha256`; `stream_tensor` hashes the canonical pathname, not the mmap
that `bit()` consumes.  Second, `isinstance` accepts a subclass overriding
`bit()`.  Both reached scheduler construction.  A sentinel stopped execution
before request zero, so no pilot, production or product work ran.

The executable entrypoint must construct an immutable canonical plane
internally, or exact-bind the opened descriptor/mmap rather than mutable object
attributes.  M1550 authorizes only successor interface-repair authoring.
