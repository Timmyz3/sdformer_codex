# M1335 — C1 M1334/R14 runtime-witness source blind review

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

The package binds the correct M1333 readiness audit, frozen M528/M935/M1162,
R3 SVA, R13 TB, Python 3.10 binary and seven active filelist members.  The
214,912-byte ledger is presently represented honestly as 18,432 integrated
bytes plus 196,480 external bytes.  Author tests are 13/13 PASS, and ordinary
missing/duplicate response, core accept, psum-write, row, task, mask and design
fault attacks are rejected.

The runtime witness nevertheless has six fail-open paths:

1. `core_after` includes the current edge.  Consequently, the second weight
   request may fire on the same edge as the first core accept and still pass.
2. The second accept, psum commit, row completion and task completion may
   collapse onto one edge through the `*_after` values and reach DONE.
3. Source index, psum address, row id and task epoch use ordinary four-state
   equality.  With X data, `if (!(condition))` does not set the fault, yet the
   corresponding event counter advances.
4. Static seam checks are token searches.  Real child outputs and
   attack/design-fault inputs can be tied to constants while the required
   strings remain in comments or declarations.
5. The unique PASS display can move before the fatal test and still satisfy
   the checker, allowing a failing run to emit a PASS token first.
6. Contract ledger values are not checked.  A disposable mutation changing
   214,912/18,432/196,480 to 1/1/0 still obtains source PASS.

The repair should remain verification-only: use prior registered stages for
strict milestone order, reject all X controls/identities, structurally bind
the real seams, control-dominate PASS with `pass===1`, and verify the complete
ledger dictionary.  Frozen design RTL does not need to change.

No VCS, simv, release, DC, PT, PTPX or other EDA job ran.  No RTL was modified,
and `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
