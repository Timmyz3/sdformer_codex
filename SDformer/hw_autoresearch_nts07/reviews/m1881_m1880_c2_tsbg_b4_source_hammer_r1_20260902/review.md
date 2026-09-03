# M1881 independent fail-closed review of M1880 B4 TSBG source

## Verdict

**PASS — P0/P1/P2 = 0/0/0; score 99/100.** M1880 closes the exact 15 M1871 attacks, the exact nine M1875 attacks, and 18 new M1881 structural-semantic attacks. This is a source-only ruling: it permits only additive M1882 campaign-source authoring. It does not permit a naked release, VCS, simv, EDA, a license query, an attempt, a result, or any paper/performance claim.

## Evidence that passed

- The official checker and all 45 tests pass independently on CPython 3.6 and 3.12; the original M1874 suite remains 36/36 on both interpreters.
- Exact semantic regressions reject M1871 15/15 and M1875 9/9 on both interpreters.
- The independent hammer rejects 18/18 additional attacks on both interpreters, with byte-identical probe output. These attacks target unique causal PASS/finish, both Acc24 scoreboards and duplicate-commit guards, reset-only SVA disable, full request/response/bridge/commit stability, complete post-reset response/terminal ledgers, a real second reset, and the final attack/reset/recovery ledger.
- Two controls changing `BUNDLE=4` and the candidate hit ledger are also rejected, confirming that the validators were actually exercised.
- M1880 RTL/SVA/TB normalize byte-exactly to immutable M1874 by namespace only. RTL also normalizes byte-exactly to M1870 and to the declared M1794 B4/LRU4 specialization.
- Independent ledgers reproduce ordinary LRU4 `0/48/44`, TSBG LRU4 `36/12/8`, equal work `576 issue / 9216 signed product / 24 commit`, signed `-1/0/+1`, `-(-128)=+128`, directed Acc24 `[-255,510]`, production bound `98304`, and the 8076-byte source resource model.
- M1880 author, M1871, and M1875 directories are sealed; docs/359 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Claim boundary

This review establishes static source integrity and semantic-checker resistance only. `same_area=false`, `rtl_executed=false`, `vcs=false`, `dc=false`, `ptpx=false`, `energy=false`, `paper_admitted=false`, and both component/system speedup admission remain false.

## Required next gate

M1882 may now be authored as a distinct campaign source. It must then pass a new different-author fail-closed source review and receive a separate sealed launch release before a single VCS attempt. M1881 must never be treated as a bare execution release.
