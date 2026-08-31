# M518 r9 independent source-only static hammer

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_R9_VCS_AUTHORIZED`  
Score: **98/100**; P0/P1/P2 = **0/1/1**

## Decision

The r9 source identity is statically admitted for exactly one invocation of
`dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r9_exact.sh` at SHA256
`f43a5d48bdf38d0d98663243a522f7bd26e44edeb51df0b03a25629d4d2d5933`.
The only authorized tool activity is the runner's isolated wrong-TB preflight,
one Full64 VCS identity query, one Full64 VCS compile, and one fixed-seed
simulation. DC, Formality, PT/PTPX, open-source EDA, performance, PPA, system,
and headline claims remain unauthorized.

## Independent findings

1. The author request's member manifest and outer seal pass. The runner's 28
   expected input paths are unique and all 28 independently recomputed hashes
   match.
2. The r9 TB contains exactly one
   `#0.2;release_valid=1'b1;raw_valid=1'b1;` fragment. Replacing only it with
   the r8 fragment recovers frozen r8 TB SHA256
   `d03fd23a19046d7b96819f2f8b7753a03cb2cf3454564579b03647026a480de2`.
   Therefore every other TB byte—including V06, phase flow, expected cycles,
   the numeric/conservation/protocol oracles, PASS signature, and cover
   campaign—is preserved.
3. The V08 line-765 fragment
   `@(negedge clk_core);result_ready=1'b0;#0.2;` remains unique. RTL and SVA
   remain byte exact to r8 at their frozen hashes. The source still has 51
   assertion labels and the exact 25-cover set enforced by the runner.
4. The causal repair matches the sealed r6 diagnosis: `send_config` already
   returns at a negative edge, so the r8 V16 task inserted one redundant full
   cycle. The r9 `#0.2` changes only stimulus cadence and does not relax the
   fixed 29-cycle expectation or any oracle.
5. No `$deposit`, DUT `force/release`, hierarchical DUT-state LHS, writing
   bind, or `always_ff` downgrade was found.
6. `M518_RUN_DIR`, runner SHA, and a strict-finite double-sealed admission are
   checked before canonical result creation. The wrong-TB control must fail
   with exit 10 before VCS and can write only its disjoint, double-sealed
   negative directory. The r8 static chain, r6 failure review, and eight r8
   failure artifacts are exact-SHA bound before tool launch.

## Residual findings

- **P1 — publication atomicity:** `RUN_COMPLETE.txt` is emitted before the
  final member and outer seals. A late sealing failure can leave contradictory
  markers. The mandatory independent post-run receipt hammer must reject such
  a topology; a future runner should publish a fully sealed staging directory
  by atomic rename.
- **P2 — static/runtime boundary:** this review executed no EDA and therefore
  proves no compilation, runtime behavior, numeric closure, cycle count, or
  production equivalence. Those remain contingent on the one-shot VCS run and
  a different receipt-blind reviewer.

## Required launch environment

```text
M518_EXPECTED_RUNNER_SHA256=f43a5d48bdf38d0d98663243a522f7bd26e44edeb51df0b03a25629d4d2d5933
M518_EXPECTED_STATIC_ADMISSION_SHA256=311a05a2495ed170093b40ef2a7fda09e012c931038dca1098be5673562a2eeb
M518_RUN_DIR must be unset
```

Required result path:
`results/m518_matched_fixed_t10_atlif_vcs_r9_exact_20260827`.

This admission is not runtime, cycle, PPA, energy, speedup, or headline
evidence. A different independent reviewer must hammer the completed receipt.
