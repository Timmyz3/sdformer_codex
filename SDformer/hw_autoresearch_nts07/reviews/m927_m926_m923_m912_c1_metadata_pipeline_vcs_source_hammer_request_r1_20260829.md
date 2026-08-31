# M927 independent source-hammer request for M926/M923/M912

## Scope

Perform a fresh, read-only, no-EDA hammer of the M926 runner-only additive
repair.  Do not invoke either M923 or M926 runner, VCS, `simv`, any Synopsys
shell, or a license query.  Do not create an attempt/result/release.  Do not
modify any source, prior contract, quarantine, review, or docs/359.

Write only the fixed directory:

`reviews/m927_m926_m923_m912_c1_metadata_pipeline_vcs_source_hammer_r1_20260829`

A PASS must be recursively manifested and outer-sealed with status
`PASS_M927_M926_M923_M912_SOURCE_HAMMER`, score 100, P0=0 and P1=0.

## Exact identity presented

- M926 runner: `dc_handoff/scripts/run_vcs_m926_m923_m912_c1_metadata_pipeline_unit_delay_r4_exact_sha.sh`
  (`08c7133945be9985710702d3c883faf36229aa7acc34daf6dfda5be7cde85685`)
- M923 TB: `040da58093c8338e21970462f3c82f4b46bb391146e7b2c714c32be3f93ec47d`
- M923 static checker: `73f38ec26d35b9cbe59b7059a39c496ca5d3fd426861a066fdcd9a89e8c7340a`
- M912 RTL: `eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53`
- frozen M919 SVA: `7dfb91f6d11aa2be8f8c9472ba3784145f290215b67a826fd9f53e32c22b7837`
- docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

Compute and bind the live M926 contract SHA after verifying its double seal.

## Mandatory checks

1. Diff M926 runner against frozen M923 runner.  Apart from additive identity,
   predecessor gates, result/attempt/release namespaces and receipt schema,
   the only semantic repair must be recursive-seal completeness enumeration.
2. Prove M926 uses `os.walk(..., followlinks=False)`, prunes symlink
   directories, classifies file entries via `os.lstat`, and includes only
   `stat.S_ISREG` entries.  It must not use `Path.is_file()` or `rglob()` for
   seal completeness.
3. Reproduce without EDA that the frozen M919 quarantine has manifest=111,
   corrected actual regular files=111, and two excluded symlinks.  Verify its
   inner and outer seals.
4. Verify M924 is a recursively sealed non-PASS predecessor with exact hashes:
   review `bf1dc1ff996b4f92ecdfe86ca1e9e5e4e67d5a4e05cdfa2d6915d32959ba6b2b`,
   manifest `444f8194cc8c3dc3bc801d0d4e136223daca9943da31ea728da87445d93bc9ff`,
   outer file `2cfcb42df30b64c08c3b63668d099a98c573d9e3f95b761061edb3ead5e8de70`.
5. Verify M923 TB/static checker/RTL/SVA are byte-identical to their pinned
   hashes and the M923 contract remains double sealed.  M923 is never rerun.
6. Verify M926 result/attempt/work namespaces are absent.
7. Prove M926 requires a separately authored, double-sealed fixed-path M928
   release which binds this M927 review/manifest/outer trio before attempt
   creation.
8. Preserve all M923 VCS source gates: foundry `UNIT_DELAY`, six attacks,
   phase token, coverage/P2/held-final tokens, exact source hashes, collision
   and memory checks, timeout, quarantine-on-failure and fail-closed claims.

## Claim boundary

This is source-integrity evidence only.  It establishes no functional VCS
result, timing, cycles, speedup, PPA, energy, system result, headline, or
paper-citable claim.
