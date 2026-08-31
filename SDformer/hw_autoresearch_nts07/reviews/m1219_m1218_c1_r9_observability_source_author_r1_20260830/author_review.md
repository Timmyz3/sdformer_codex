# M1219 — C1 R9 source-only observability repair author receipt

## Outcome

An additive R9 TB was created without overwriting R8.  It addresses M1218's
failure-localization gap and does not alter M528, M935, M1162, R3 SVA, the
24-random workload, attacks, II=2, normal M935 row/task, or R8 request-ready
quiescence semantics.

This is source evidence only.  No VCS/EDA tool was invoked and no functional,
timing, cycle, PPA, energy, system-speedup, headline, or paper claim is admitted.

## Exact change boundary

R9 replaces the three random unbounded waits with bounded edge-count loops:

- weight request count reaches exactly `w0+1`;
- optional first-beat psum request count reaches exactly `p0+1`;
- response-accept count reaches exactly `response0+1`.

It also bounds each `prep_ready` wait in the 64-row normal preload and adds a
separate bounded `prep_ready` gate immediately after the clean normal reset.
Counter overshoot is fail-closed rather than allowed to wait forever.

Every main stage has a flushed unique ENTER/COMPLETE token.  Each of the 24
random transactions emits an indexed ENTER/COMPLETE pair.  Any new liveness
timeout first prints wrapper transaction state, service valid/ready state,
global handshake counters, and M935 fault/prep/match/bank state, then calls
`$fatal`.

## Frozen identities

- R8 TB: `060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b`
- M528: `8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783`
- M935: `e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8`
- M1162: `639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595`
- R3 SVA: `c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472`
- `docs/359`: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

New R9 TB SHA-256 is
`9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75`.

## Static evidence

The canonical checker verifies frozen hashes, rejects every `wait(...)`,
requires all eight `while` loops to have watchdog bounds, checks phase pairs,
timeout sites, frozen coverage minima, ready quiescence, and claim boundaries.
Seven unit tests pass: one canonical positive and six fail-closed mutations
(unbounded loop, missing timeout site, missing phase completion, ready
quiescence mutation, random-count mutation, and claim mutation).

The double-sealed source contract is
`contracts/m1219_m1218_m1213_c1_r9_observability_source_contract_r1_20260830.json`.
It authorizes only a fresh different-author source hammer.  A hammer must not
convert this package into a launch release; VCS remains separately gated.

