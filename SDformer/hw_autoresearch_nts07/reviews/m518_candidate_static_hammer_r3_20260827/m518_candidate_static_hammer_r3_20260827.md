# M518 repaired production candidate independent static hammer r3

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_VCS_AUTHORIZED`  
Score: **95/100**  
Findings: **P0=0, P1=5**

This is a receipt-blind, source-only review. The reviewer did not run the M518
runner, VCS, DC, Formality, PT, PTPX, Verilator, or any other RTL/EDA tool and
did not modify production RTL/SVA/TB/filelist/contract/runner or `docs/359`.
Only SHA/seal checks, JSON parsing, `bash -n`, text inspection, and independent
Python enumeration were used.

## Decision

The three r2 blockers are statically closed at the fixed identities supplied
for r3. Exactly one invocation of the following runner identity is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif.sh
SHA256 09a2496745692078bb6eb9ab108f9fb95bf66efe4e027c25b7c50f2e432f728d
```

That one invocation must execute the runner's automatic wrong-RTL-SHA negative
preflight first and then at most one positive Synopsys VCS V-2023.12-SP1
campaign. Any identity drift or incomplete/failing output voids this
authorization. DC, Formality, PT/PTPX, performance, energy, PPA, system, and
headline claims remain unauthorized. The VCS receipt requires a new independent
review before any downstream DC admission.

## r2 P0 closure

1. **V03 closed.** The six independently reconstructed wide sums remain
   `8388606, 8388607, 8388608, -8388609, -8388608, -8388607`. The lower context
   now uses threshold `-8388608`. For the lower overflow input `-8388609`, the
   event is 1 after Q24 saturation and 0 under compare-before-saturation, so the
   ordering is externally observable through the result scoreboard.
2. **V15 closed.** `release_valid` remains asserted over exactly eight
   consecutive sampled positive edges. Every edge rejects release/retire/work,
   the transition counter must equal one, and three further probes check sticky
   reset-only quarantine after deassertion.
3. **V18 closed.** The TB invokes nine classes: partial config, partial raw,
   dense c0/c11/c12/c15/c16, FIFO-full close stall, and quarantine. Nine
   individual exact-one gates plus `total_reset_attacks==9` and
   `total_clean_after_reset_probes==9` close the class ledger. `reset_dut`
   checks deterministic empty state, and every class is immediately followed by
   a clean exact N1=29 context.

## Identity and launcher checks

- Fixed candidate SHA identities match the task: RTL `09b1d976...`, SVA
  `977f9565...`, TB `e7973a91...`, filelist `09e43560...`, contract
  `483ad933...`, and runner `09a24967...`.
- The sealed M518 baseline-spec member manifest and outer seal pass. The r2
  review member manifest and outer seal also pass. `docs/359` remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- M518 and the frozen M273r2 identity each expose 50 exactly matching default
  public ports. Independent schedule enumeration gives
  `[96 x 16 cycles, 64]`, 1,600 tuples, 1,600 unique tuples, and complete
  `{row0..9} x {lane0..15} x {time0..9}` coverage.
- The runner verifies its externally supplied SHA before creating the result
  directory. It then requires an isolated all-zero wrong-RTL SHA check to exit
  10 before any tool call, forbids positive artifacts in that negative
  directory, seals the negative evidence, rechecks all positive input SHAs, and
  only then queries/launches VCS. Positive and negative artifact names are
  disjoint.

No M518 result directory or M518 EDA process was present at review time.

## Retained P1 items

1. The r3 contract field named `r2_static_hammer_outer_seal_sha256` stores the
   r2 manifest digest `c89e18...`; the actual outer-seal-file digest is
   `b14d2e...`. Both seal checks pass, so this is a provenance label defect, not
   a candidate/VCS identity blocker.
2. The runner compares M273/M518 ports, but M273 is not itself a direct member
   of the runner SHA map. This review separately confirms its frozen
   `11d5c6...` identity.
3. V13 closes on an aggregate half-cycle observation count rather than an exact
   per-config/per-raw-phase ledger.
4. V19 exercises diverse/extreme back-to-back contexts but lacks a dedicated
   counter labeling a literal opposite-data pair.
5. Generic V10/V12 malformed-frame cases rely on RTL commit guards plus
   quarantine; only the strongest combined fault-edge case snapshots storage
   and commit counters individually.

These P1 items constrain claim wording and future hardening but do not create a
false-positive path through the repaired V03/V15/V18 campaign.

## Admission boundary

Admitted now: exact reviewed source identity, static closure of the three r2 P0
repairs, exact public-interface match, complete frozen schedule enumeration, and
one-shot exact-runner authorization.

Not admitted: SystemVerilog compilation, VCS behavior, runtime V01--V20 PASS,
29/80 RTL cycles, numerical equivalence, speedup, DC/Formality/STA/PTPX, power,
energy, PPA, system speedup, or a paper headline.
