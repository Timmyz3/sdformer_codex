# M924 independent source-hammer request for M923/M912

## Scope

Perform a fresh, read-only, no-EDA hammer of the additive M923 TB-only repair.
Do not invoke the runner, VCS, `simv`, any Synopsys shell, or a license query.
Do not create the M923 attempt marker or result namespace.  The source author
must not author this review or the subsequent release.

The hammer output has one fixed directory:

`reviews/m924_m923_m912_c1_metadata_pipeline_vcs_source_hammer_r1_20260829`

It must be recursively manifested and outer-sealed.  A passing `review.json`
must use status `PASS_M924_M923_M912_SOURCE_HAMMER`, score 100, P0=0 and P1=0.

## Exact source identity presented to the hammer

- runner: `dc_handoff/scripts/run_vcs_m923_m912_c1_metadata_pipeline_unit_delay_r3_exact_sha.sh`
  (`adb1cb87ed4e7dfa7ff3b9e787ab3087dfb81e91d8e219ac3783573aebf0c63e`)
- M912 RTL: `eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53`
- frozen M919 SVA: `7dfb91f6d11aa2be8f8c9472ba3784145f290215b67a826fd9f53e32c22b7837`
- new M923 TB: `040da58093c8338e21970462f3c82f4b46bb391146e7b2c714c32be3f93ec47d`
- M923 static checker: `73f38ec26d35b9cbe59b7059a39c496ca5d3fd426861a066fdcd9a89e8c7340a`
- frozen M919 TB: `de19e962c1ffb16d74f6505e425843f3fbe399ef47d746bf3329770d48daa78d`
- docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

The hammer must compute and bind the live M923 contract SHA rather than trust
this request.  It must verify the contract's double seal and the runner's
`bash -n` syntax before scoring.

## Mandatory checks

1. Prove byte-for-byte that the M923 TB differs from the frozen M919 TB only
   inside `attack_wrong_parent_and_dead_live`.
2. Prove the repaired attack waits for `execute_busy`, advances to a negedge,
   fails closed unless active and next contexts are both invalid, then forces
   backing row 1 parent 63 and dead parent-live state.
3. Prove the test explicitly observes row 1 in active or next context with
   the corresponding `relation_ok_q===0`, uses a bounded capture watchdog,
   then calls the existing bounded sticky-fault watchdog.
4. Reject any direct force of `active_ctx_relation_ok_q` or
   `next_ctx_relation_ok_q`.
5. Prove all six attack tasks/calls, all normal/P2/held-final coverage gates,
   the foundry `UNIT_DELAY` compile define, and the fail-closed claim labels
   remain present.
6. Verify the recursively sealed M919 failed quarantine and M922 forensic
   exact hashes.  M919 remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE`.
7. Prove the M923 result, attempt and work identities are absent.
8. Prove the runner requires a separate, double-sealed fixed-path M925 release
   binding this hammer's review/manifest/outer hashes before attempt creation.

## Claim boundary

This request and hammer are source-integrity evidence only.  They establish no
functional VCS result, timing, cycle count, speedup, PPA, energy, system result,
headline, or paper-citable claim.
