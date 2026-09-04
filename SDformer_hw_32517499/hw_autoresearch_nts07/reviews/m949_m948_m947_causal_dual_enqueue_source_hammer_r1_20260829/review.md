# M949 | M948 causal dual-enqueue TB-only source hammer

## Verdict

`PASS_M949_M948_CAUSAL_DUAL_SOURCE_HAMMER`, 98/100, verdict `GO`.
P0=0, P1=0, P2=1.  This is source-only GO for a separately sealed successor
runner/release; M949 ran no VCS/DC/GPU/remote and issues no launch release.

M947's real manifest (`bb33245c...`) and outer-seal file (`dffdc680...`) both
verify, and the M948 DRAFT binds them correctly.  The M948 contract's JSON
sidecar and sidecar-seal both validate from the repository root.  M948 TB,
checker, and contract hashes are respectively `ab4b4d41...`, `6e829689...`,
and `9efa7c54...`; the static checker independently passes.

The diff against M938 is TB-only.  M935 RTL, inherited M919 execution SVA,
M938 match SVA, `make_dual_enqueue_masks`, all other corpora, and the inherited
normal-minimum block remain frozen.  Only three counters, one causal task, its
epoch4 call/gate, and the extended PASS token were added.

The causal task forces only public `psum_write_ready` and
`row_complete_ready`, at the inactive edge before row2/source2 acceptance.  It
then requires:

1. phase N: real `issue_accept_w` plus macro read for consumer3/parent0;
2. phase N+1: real pending response for consumer3/parent0 plus simultaneous
   forward for consumer4/parent2 on row2/source3 final;
3. phase N+2: delayed dual/read/forward debug observers and independently
   incremented cleanroom `cov_pending_plus_forward`.

The force is released only at the inactive edge after phase 3.  No internal
DUT force exists.  The monitor and load/wait branches join, so successful
completion cannot leak the force into later tasks.  Each phase has exact
identity checks or a watchdog/fatal.  The new three counters must each equal
one and the original cleanroom cover must remain nonzero before the suite can
continue; the final PASS token exposes all four values.

The existing F/G/R reset tests, opposite-bank proof, external ownership oracle,
parent miter, six attacks, and every original minimum remain mandatory.

- **P0=0:** no semantic, force-leak, frozen-source, or seal failure found.
- **P1=0:** M947's unique TB-only repair is closed at source level.
- **P2=1:** commercial compile/simulation must still prove scheduler ordering
  and the expected three-cycle witness under foundry UNIT_DELAY.

M943 remains consumed and cannot be rerun.  Only one fresh successor attempt
with exact M948 identity is allowed.  Functional/timing/cycle/speedup/PPA/
energy/system/paper claims all remain false.
