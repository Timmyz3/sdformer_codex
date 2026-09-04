# M1235 — independent M1232/R11 C1 TB source hammer

Verdict: **the exact R11 candidate closes the M1229 random-service P0, but
release authoring remains NO-GO until its checker rejects three independent
random-path mutations.**

Score: **86/100**; P0=0, P1=1, P2=0.

## What passed

The M1232 author package and source contract both verify through their inner
manifests and outer seals.  The contract binds the exact M1229 no-go verdict;
M528, M935, M1162, R3 SVA, R10, and `docs/359` retain their frozen hashes.
The author checker passes, and all 15 declared tests pass.

The canonical R11 source is a materially correct TB-only repair.  Every shared
R10 task other than `random_legal_transaction` is byte-identical, including
`serve_normal_beat`; the only new helper releases the nine forced issue-tuple
fields and deliberately leaves core ready forced.  For all 24 random
transactions, the source proves exact weight and conditional psum request
fires, rejects overshoot, retires request ready at a negedge, and validates one
handshake per enabled service.  It drives core backpressure before either
response, preserves odd/even response skew, holds valid and zero payload,
raises ready only at a negedge, and waits for exactly one common response
accept.

After that accepting edge, the source releases the forced tuple before the next
sampled edge, verifies wrapper/request inactivity, retires response valid at
the immediately following negedge, and samples one final posedge to reject any
duplicate request, duplicate response, boundary fault, or core fault.  The R10
normal path's legal next-beat prefetch rule remains byte-for-byte unchanged.
All frozen phase tokens, directed attacks, service attacks, 24 random
transactions, II=2, row-0 mask `0x0003`, and two normal M935 issues remain
enabled.  No legal-random task changes an SVA attack mask, and a later release
must require zero unmasked SVA failures.

This is structural evidence only.  It does not prove that a VCS execution will
pass.

## P1 — the checker still has three random-path blind spots

An independent mutation campaign changed only the R11 random legal path.  The
M1232 checker accepted all three mutants with an empty error list:

1. disable `random_request_window_active`, which makes the claimed exact-one
   per-window counters inert;
2. remove the post-retirement response-count oracle, which weakens duplicate
   response closure;
3. replace the random response backpressure loop with `repeat (0)`, which
   removes the explicitly claimed exercised hold body.

The canonical source retains all three correct constructs, so no new TB design
is requested.  The narrow repair is checker-and-tests only: bind the request
window enable, the post-retirement response-count term, and a positive
`hold_cycles` loop; add one negative test per construct; then re-seal the
metadata and request another independent hammer against the unchanged R11 TB
SHA.

## Authorization and claim boundary

- Checker/tests-only repair: **authorized**.
- Candidate TB mutation, release authoring, VCS/simv, EDA, GPU, remote work,
  RTL/SVA mutation, and M1221 retry: **not authorized by this hammer**.
- Functional VCS, timing, cycles, speedup, PPA, energy, system speedup, and
  paper admission remain false.

No VCS, simv, EDA, GPU, or remote action was performed.  The candidate source,
RTL, SVA, workloads, and `docs/359` were not modified.
