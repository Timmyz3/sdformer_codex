# M1229 — independent M1226/R10 C1 TB source hammer

Verdict: **NO-GO for release authoring or VCS.  Author a new additive TB
source that also repairs the inherited random-service boundary, then hammer it
again.**

Score: **58/100**; P0=1, P1=1, P2=0.

## What passed

The M1226 author package and contract double seals verify.  The source hashes
bind the exact M1225 forensic, frozen M528/M935/M1162/R3-SVA RTL, R9 TB, and
`docs/359`.  The author checker passes and its 11 declared tests pass.

The new normal `serve_normal_beat` is materially better than R9: it proves one
request fire, retires both request-ready inputs before presenting responses,
holds zero response payloads through the exact response counter increment,
withdraws valids at the immediately following negedge, and dumps state on all
normal watchdog paths.  Its beat-two admission is prefetch-safe: it permits a
new unaccepted M935 tuple to occupy M1162, but rejects the just-served source or
any accepted service request.  The final normal completion gate still rejects
`protocol_error`, so tuple mutation remains fail-closed.

## P0 — the frozen random workload still has known unmasked SVA failures

M1226 changes only normal service choreography.  Its complete
`random_legal_transaction` task is byte-identical to R9.  In the sealed M1221
R9 log, between `PHASE_M1219R9_RANDOM_ENTER` and
`PHASE_M1219R9_RANDOM_COMPLETE`, that exact task produced **22 unmasked SVA
failures**:

- 11 `ap_weight_request_hold`;
- 11 `ap_weight_response_hold`;
- 0 other failure classes in that phase.

These failures occur before the new normal task can influence execution.  Thus
R10 cannot presently justify its mandatory all-workload
`zero_sva_failures_required=true` gate.  A fresh release would spend another
one-shot VCS attempt on a source already contradicted by sealed deterministic
evidence.

Required repair: add a new TB identity and make every random transaction use
the same exact request/response retirement discipline as the repaired normal
path.  Do not mask the assertions or mutate M1162/SVA.  In particular, preserve
the forced request tuple until the one-fire proof, retire ready at a race-free
edge, hold each response through the exact accept, withdraw at the next
negedge, and do not reset/release the request until all sampled hold obligations
have retired.  The next static checker must cover both normal and random
service paths.

## P1 — the source checker is normal-only

The checker accepts two independent destructive mutations to the inherited
random task with zero errors:

1. keeping both random request-ready inputs high after the intended fires;
2. restoring an extra response-valid posedge after exact accept.

Its existing 10 negative mutations therefore protect the normal repair but do
not protect the claimed whole-regression SVA-zero boundary.  This is a checker
coverage gap, not authorization to weaken the release gate.

## Authorization and claim boundary

- Additive TB-only repair authoring: **authorized**.
- Release authoring, VCS/simv, EDA, GPU, remote work, RTL/SVA mutation: **not
  authorized by this hammer**.
- Functional VCS, timing, cycles, speedup, PPA, energy, system speedup, and
  paper admission all remain false.
- M1221 is consumed and must not be retried.  M1226 remains a useful source
  artifact but is not releasable.

No VCS, EDA, GPU, remote action, RTL mutation, or source mutation was performed
by this review.
