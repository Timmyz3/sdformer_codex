# M1270 — C1 R13 real-M935 integrated protocol source

Date: 2026-08-30  
Mode: source-only author review; no VCS/EDA/GPU/remote  
Decision: **SOURCE READY; fresh independent hammer required**  
Score: **98/100**  
P0/P1/P2: **0/0/1**

## Outcome

R13 replaces the synthetic boundary phases with one authoritative integrated
path:

`public prep -> frozen M935 -> frozen M1162 -> external weight/psum services`.

The workload is the established 64-row normal task with row-0 mask `0x0003`.
It naturally creates exactly two M935 issue tuples: first/source-0 and
non-first/source-1. There is no procedural override of any parent or child
`issue_request_*` object. M528, M935, M1162 and R3 SVA retain their frozen
SHA256 identities.

The first beat accepts one weight and one psum request, then presents weight
response alone for two sampled cycles. Both response-ready signals and the
response-accept count must remain low until the psum response appears. The
second beat accepts one weight and zero psum requests and completes from weight
response alone. The harness checks a response-accept gap of at least two
cycles, two issue accepts, one psum commit, one row completion, one task
completion, epoch `0x9001`, and zero boundary/core/M935/service faults.

## Diagnostic repair

Every dynamic check routes through the sole `oracle` task. Before its only
`$fatal`, it prints:

- check site/pass state and beat identity;
- expected first flag;
- weight, psum and response count deltas;
- cycle, real M935 valid/first/last/source tuple;
- request and response valid/ready operands;
- wrapper active/accepted state;
- boundary, core and M935 faults;
- row/task completion counts.

Consequently a future failure cannot repeat M1265's compound-predicate
ambiguity.

## Static validation

- static checker: PASS;
- source tests: 16/16, including 15 rejected mutations;
- one initial block and one real integrated task;
- issue-request parent override count: 0;
- issue-request child override count: 0;
- `$fatal` sites outside the operand printer: 0;
- exact phase enter/complete/PASS token order: PASS;
- claim escalation attacks: rejected;
- `docs/359` observed SHA256 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The test corpus specifically rejects parent/child override keywords,
hierarchical assignment, second-beat-first mutation, one-source mask, psum on
every beat, weakened II, deleted join hold, weakened completion/fault checks,
missing SVA instance, extra fatal, PASS claim escalation, near-neighbour phase
token, and a second initial block.

## Claim boundary and next gate

This is source-only evidence. It establishes neither compilation nor functional
simulation, zero runtime SVA failures, timing, measured cycles, speedup, PPA,
energy, system speedup, nor a paper headline. It does not authorize a launch.

The next and only gate is a fresh different-author independent source hammer.
Only after that hammer may a separately authored exact-byte release be
considered. M1265 remains consumed and must never be retried.

P2: the harness is minimal in protocol semantics, but its line count remains
dominated by the frozen top-level port and SVA wiring. This is not a functional
risk and should not be reduced by hiding ports before source hammering.
