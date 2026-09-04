# M518 r10 launch pre-tool failure hammer r8

Date: 2026-08-27  
Verdict: `FAIL_CLOSED_PRETOOL__R10_NOT_CITABLE__R11_RUNNER_ONLY_REPAIR_REQUIRED`  
Score: **98/100**; P0/P1/P2 = **0/1/0**

## Finding

The reported r10 exit code `6` has exactly one source location: runner line 17,
which checks the value captured from the exact environment-variable name
`M518_EXPECTED_STATIC_ADMISSION_SHA256` against 64 lowercase hexadecimal
characters. The live admission SHA is
`c82772d628b2c3279326101e3b1bf23fbdffb19009f5b8f6c62b5ea8400848e1`;
it is 64 characters, passes the same Bash regex, equals the current admission
file SHA, and both admission seals validate. An isolated `env -u M518_RUN_DIR`
assignment using both exact variable names also passes the regex.

Therefore this is not evidence that the admission file or regex is wrong. If
the supplied exit code is correct, the value visible to the runner under the
exact `M518_EXPECTED_STATIC_ADMISSION_SHA256` name was absent, malformed, or
contained extra text/whitespace. Because the effective argv/environment was not
sealed and the bare gate prints nothing, the surviving evidence cannot identify
which launch-plumbing case occurred. In particular, this review does **not**
pretend to prove a specific typo.

## Nothing reached a tool

Line 17 precedes canonical `mkdir` at line 62. The first VCS identity command is
at line 292, compilation at line 299, and `simv` at line 313. The canonical r10
result directory is absent; no r10 compile log, simulation log, assertion
report, `simv`, or runtime receipt exists. Hence the r10 attempt did not run the
wrong-TB negative control, VCS identity, VCS compilation, or simulation. It
also authorizes no DC, Formality, PT or PTPX statement.

## Unique minimal repair

Do not retry r10. Its double-sealed admission says one **runner invocation**, so
the reported pre-tool invocation consumed that authorization even though it did
not invoke EDA.

The only authorized next action is an r11 runner-only launch repair:

1. Keep RTL, SVA, TB and filelist byte exact to r10; use a new r11 canonical
   result identity and new one-shot admission.
2. Before `mkdir`, reject unset, wrong-length, non-lowercase-hex, live-SHA,
   seal, and semantic mismatches separately. Diagnostics may report only
   presence and length, never the supplied hash value.
3. Strictly require JSON integer `1`, rejecting Boolean `true`.
4. Replace hand-entered environment assignments with one immutable r11 launch
   wrapper or command file that sets both exact SHA variable names and unsets
   `M518_RUN_DIR`. Freeze and independently review that launch artifact.
5. After a different source-only review, root may create a new exact-SHA,
   double-sealed r11 admission. Only that admission may authorize one r11 VCS
   campaign, followed by a different receipt hammer.

This review authorizes r11 authoring only. It does not authorize r11 VCS, any
physical tool, or any numeric/performance claim.

