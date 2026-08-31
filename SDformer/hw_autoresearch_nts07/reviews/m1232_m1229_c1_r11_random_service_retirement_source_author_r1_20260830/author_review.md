# M1232 — C1 R11 additive random-service retirement source

Status: **SOURCE GO for a fresh independent source hammer only.**  VCS, simv,
release publication, EDA, GPU, remote work, M1221 retry, and DUT/SVA mutation
remain forbidden.

M1229 correctly rejected R10 release authoring because its inherited random
task was byte-identical to the R9 workload that had 22 sealed random-phase SVA
failures.  R11 preserves R10 normal service byte-for-byte and repairs the legal
random service boundary under a new namespace.

For every one of the 24 random transactions, R11 now proves exact weight and
conditional psum request fires before retiring both ready inputs at a negedge.
Core ready is forced low before response drive and can rise only at a negedge.
Odd/even response skew is retained, with valid and zero payload checked through
the exact response counter increment.  After acceptance, only the forced issue
tuple retires before the next sampled edge; response valid retires at the
immediately following negedge, and a final posedge proves no duplicate request,
response, active wrapper state, or boundary/core fault survived.

The static audit passes.  Fifteen tests pass: one canonical positive and
fourteen fail-closed mutations, including the two M1229-required mutations
(random ready not retired and extra random response posedge), core-ready
posedge race, tuple retirement removal, response stability weakening,
post-retirement edge removal, random SVA-mask injection, early core-ready
release, state-dump removal, R10 normal mutation, workload/row mutation,
zero-SVA removal, and claim inflation.

Exact identities:

- R11 TB: `850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`
- checker: `729184404ee23a0152848d5525deb36329756023da31c0e58c81936f3bab63d7`
- tests: `5e926a9e99dfa180e6c8232a387ecc3dc06d5bbd425841f17db2feb4f8397da4`
- source contract: `8a75c7592f9e6f8cf98e35fcfd092d83a0a6f7dd6c56c01a8c3ae6cbea6dbdf6`

This is a source-only result.  It does not prove that the prior SVA failures
are gone; only a fresh, differently authored release followed by its one-shot
VCS execution may establish that.  Any later release must require all 24
random completion tokens, normal M935 completion, and zero unmasked SVA
failures.
