# M518 matched Fixed T10 ATLIF independent static hammer r4

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_VCS_AUTHORIZED`  
Score: **96/100**  
Findings: **P0=0, P1=6**

This is a receipt-blind static review. The reviewer did not run the candidate
runner, VCS, DC, Formality, PT/PTPX, Verilator, or another RTL/EDA tool. No
production RTL, SVA, TB, filelist, contract, runner, old r3 diagnostic result,
or `docs/359` file was modified. Only these review artifacts were added.

## Decision

The r4 repair statically closes the r3 pre-compile architecture-selection
failure. Exactly one invocation of this identity is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r4_exact.sh
SHA256 d656d11dc32e11e018c7035112567a5b0b2de52dc5e2ad6073778295883ef55b
```

The operator must supply that SHA through
`M518_EXPECTED_RUNNER_SHA256`, leave `M518_RUN_DIR` unset, and use the absent
default result path
`results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827`. The invocation
must first seal the wrong-RTL exit-10 control and then perform exactly two VCS
launcher calls before simulation: one `vcs -full64 -ID` query and one
`vcs -full64 -sverilog ...` compile.

This static decision does not admit compilation, runtime behavior, V01--V20,
29/80 cycles, numerical equivalence, DC/Formality/STA/PTPX, power, speedup,
PPA, system speedup, or a paper headline. A separate independent receipt
hammer is mandatory before any DC admission.

## Frozen identity and failure-chain checks

- RTL `09b1d976...`, SVA `977f9565...`, TB `e7973a91...`, and filelist
  `09e43560...` exactly equal the frozen r3 identities. The filelist still
  contains only RTL, SVA, and TB in that order.
- The r3 runner remains `09a24967...`. Its recorded failure marker, VCS-ID log,
  runner identity, negative manifest, and negative outer seal remain byte
  identical to the independent failure review.
- The r3 failure review member manifest and outer seal both verify. Its JSON
  still classifies the stop as
  `PRE_COMPILE_TOOL_ID_ARCH_SELECTION_DEFECT`, records that compile never ran,
  and forbids r3 runner reuse.
- The new canonical r4 result path was absent during review. `docs/359` remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- The current frozen M273 reference remains `11d5c6...`; M273 and M518 expose
  exactly the same 50 public ports. Independent schedule enumeration remains
  96 active slots for cycles 0--15, 64 for cycle 16, and a complete bijection
  over all 1,600 row/lane/time products.

## Fail-closed runner inspection

The externally supplied runner SHA is checked before creation of either result
or negative-control state. The all-zero wrong-RTL control must exit 10, cannot
call VCS, cannot create compile/simulation/positive artifacts, and is sealed in
a disjoint directory. All 17 positive SHA-map entries match before either VCS
call. The runner then validates the sealed baseline specification and both
seals of the r3 failure review.

There is exactly one executable identity call with the required literal
`"${task_vcs}/bin/vcs" -full64 -ID`, no executable legacy `vcs -ID` call, and
exactly one full64 compile call. Contract and failure-review parsing reject
non-finite constants; receipt serialization uses `allow_nan=False`, then
strictly reparses, recursively checks finiteness, and checks exact round-trip
equality. The receipt precedes `RUN_COMPLETE`; `RUN_COMPLETE` precedes the
member manifest and outer seal; the failure trap remains armed until those
seals exist.

## Retained P1 limitations

1. `M518_RUN_DIR` can redirect output, so this authorization is only for the
   unset override and reviewed default path.
2. Publication files are enumerated and double-sealed but are not compared to
   a hard-coded exact whitelist. The post-run hammer must reject missing,
   unexpected, or contradictory members.
3. The static-review seal is not a runtime SHA-map member; authorization relies
   on this sealed decision plus the out-of-band exact runner SHA.
4. M273 is compared by the runner but is not itself a direct SHA-map member;
   this review independently froze its identity and 50-port equality.
5. The inherited r2 contract field named as an outer-seal digest still stores
   the r2 member-manifest digest. Actual seal checks pass.
6. The key r3 failure artifacts are unchanged and the independent review is
   double-sealed, but the entire old diagnostic directory lacks a single
   whole-directory publication seal.

These are provenance/operational hardening items. None opens a static path for
the known r3 tool-ID defect or for an unreviewed RTL/SVA/TB/filelist identity.
