# M518 matched Fixed T10 ATLIF independent static hammer r6

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_R6_VCS_AUTHORIZED`  
Score: **96/100**  
Findings: **P0=0, P1=3, P2=1**

This is a receipt-blind, source-only independent review. The reviewer did not
execute the candidate runner, VCS, DC, Formality, PT/PTPX, or any open-source
EDA tool. No author source, failed result, or `docs/359` was modified.

## Decision

Exactly one invocation of the following runner is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r6_exact.sh
SHA256 050db5ce70013ba0b61093ce2abbb544b645542af55e48061a1d9bc3e60c2a4d
```

The operator must set `M518_EXPECTED_RUNNER_SHA256` to that exact digest,
leave `M518_RUN_DIR` unset, and use the currently absent default result path
`results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827`. This authorizes
only the runner's isolated wrong-SVA negative control, exact preflight, one
Full64 VCS identity query, one Full64 compile, and one fixed-seed simulation.
A separate independent post-run receipt hammer is mandatory.

No DC, Formality, PT/PTPX, performance, energy, PPA, system-speedup, or
headline claim is authorized.

## Exact repair and preserved identity

- The r6 SVA has exactly one repaired suffix and zero r5 suffixes. Replacing
  that suffix once in memory deletes exactly one byte and reconstructs frozen
  r5 SVA SHA256
  `977f95652bb788047549d58ff94e416f00542c9d3e63fa6f83e09fe582c910f4`.
- The repaired `ap_dense_start_ownership` statement has 8 opening and 8
  closing parentheses. The r5 diagnostic had 8/7.
- RTL, TB, filelist, M273 reference, and `docs/359` retain their frozen r5
  identities. M518 and M273 independently parse to the same ordered 50-port
  public interface.
- The filelist still contains only RTL, SVA, and TB in that order. The SVA has
  51 assertion labels and the exact 25-cover set; the TB contains the exact
  PASS signature once. The r4 source of truth still enumerates V01--V20.

## Frozen provenance

All **35/35** exact runner SHA-map entries exist, are unique, and match. Four
member manifests and four outer seals verify: the baseline specification, r4
compile-failure audit, r5 static review, and r5 SVA-failure audit. The r4 and
r5 results remain diagnostic only. In particular, the sealed r5 audit permits
only r6 source authoring and explicitly requires this fresh static readmission.

## Runner audit

The out-of-band runner SHA gate precedes result creation and every negative or
tool side effect. The automatic wrong-SVA control is constructed to exit 10,
has a disjoint subdirectory, forbids compile/simv/receipt/RUN_COMPLETE, and is
member-sealed plus outer-sealed. The positive 35-input preflight, seal checks,
one-character reverse-SHA proof, 50-port comparison, assertion/campaign checks,
and strict finite contract parse all precede tools.

There are exactly two direct VCS command lines: one `vcs -full64 -ID` and one
`vcs -full64 -sverilog -assert svaext` compile. Exactly one later `simv`
invocation is possible. Compilation, the exact PASS line, all 25 nonzero
covers, strict finite receipt round-trip, member manifest, and outer seal must
all pass before `task_complete=1` disables the failure trap.

At review time the canonical r6 path was absent and `M518_RUN_DIR` was unset.
No M518 tool process was present. Unrelated users' simulators and the separate
M523 campaign were not treated as M518 activity.

## Findings

1. **P1-CONCURRENT-ATTEMPT-LOCK:** `[[ ! -e path ]]` followed by `mkdir -p`
   provides a serial one-shot guard but not an atomic lock against two
   policy-violating concurrent invocations. This authorization is for one
   operator invocation only; do not launch it concurrently.
2. **P1-PUBLICATION-ATOMICITY:** `RUN_COMPLETE.txt` precedes final manifest
   publication and no atomic directory rename is used. A late seal failure can
   leave both positive and failure markers. The mandatory post-run reviewer
   must reject any contradictory or incomplete topology.
3. **P1-RUN-DIR-OVERRIDE:** the script accepts an output override. This
   authorization is valid only while `M518_RUN_DIR` remains unset.
4. **P2-STATIC-RESIDUAL:** exact static repair closure cannot promise that VCS
   will not reveal a later parser, assertion, or runtime defect. That is the
   purpose of the authorized one-shot and independent receipt hammer.

The P1 items are operational hardening limitations already contained by the
single-invocation authorization and mandatory post-run rejection rules. None
permits an unreviewed identity or converts this static review into runtime
evidence.

## Claim boundary

Admitted now: exact r6 source repair identity and one-shot VCS authorization.
Not admitted: SystemVerilog compilation, VCS behavior, V01--V20 runtime,
numeric equivalence, RTL cycles, DC, Formality, STA, power, energy, speedup,
PPA, system speedup, or headline.

