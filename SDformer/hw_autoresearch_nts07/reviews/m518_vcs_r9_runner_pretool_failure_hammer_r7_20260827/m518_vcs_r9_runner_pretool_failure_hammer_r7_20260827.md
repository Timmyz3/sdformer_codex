# M518 r9 runner pre-tool failure hammer r7

Date: 2026-08-27  
Verdict: `DIAGNOSTIC_CONFIRMED__R9_RUNNER_FIELD_SCHEMA_MISMATCH_BEFORE_VCS__R10_RUNNER_ONLY_READMISSION_REQUIRED`

This was an independent, read-only failure review. It did not run the runner,
VCS, DC, Formality, PT/PTPX, or any open-source EDA tool. It did not modify the
r9 sources, runner, admission, failed result, sealed history, or `docs/359`.

## Bottom line

The r9 invocation never called VCS. It failed inside the runner's semantic
Python preflight because the runner queried a JSON field that does not exist:

```text
runner query: decision.r9_vcs_authorized
sealed r8 key: decision.r9_vcs_authorized_by_this_review
sealed value: false
Python value for missing runner key: None
runner check: None is not False -> true -> SystemExit
```

This is not a boolean-polarity error. The old r8 diagnostic review must say
`false`: it diagnosed the failure and required a new independent r9 static
readmission, but could not itself authorize r9. The separately sealed r9 static
review and launch-admission object correctly carry the one-shot authorization.

## Why no VCS tool was invoked

The r9 directory contains the wrong-TB negative control, positive input-SHA
preflight, historical seal checks, and copied contract/admission. It does not
contain `PREFLIGHT_COMPLETE.txt` or `vcs_id.txt`. In runner order, the semantic
Python block ends before `PREFLIGHT_COMPLETE.txt`, and the first VCS command is
later, with shell redirection to `vcs_id.txt`. That redirection would create the
file even if the VCS executable itself failed. The absence of both files,
together with exit code 1 and the exact missing-key predicate, closes the
boundary: the runner exited in Python before any VCS process launch.

There is consequently no compile log/return code, `simv`, simulation log/return
code, assertion report, author receipt, `RUN_COMPLETE`, or positive manifest.
No SystemVerilog compilation, V01--V20 behavior, numerical equivalence, cycles,
DC, PPA, performance, or headline claim is admitted.

## Findings and score

- P0: 0.
- P1: 2. The runner-to-review JSON key mismatch consumed the attempt, and the
  independent static review failed to mechanically validate the exact external
  JSON path used by the runner.
- P2: 1. The result-local failure marker records only exit 1; semantic-preflight
  stderr was not frozen. The source/topology evidence is nevertheless
  deterministic.
- Score: 96/100 for the failure diagnosis and repair contract, not for r9 VCS.

## Admission boundary

The r9 runner-invocation authorization is consumed because the canonical path
exists and the authorized runner was invoked. The VCS invocation itself was not
consumed because it never occurred. The existing r9 directory is permanently
diagnostic-only and must not be deleted, overwritten, renamed, completed, or
cited as VCS evidence. The r9 runner must not be reused.

## Unique r10 repair

There is exactly one functional repair:

```python
decision = failure.get("decision", {})
if "r9_vcs_authorized_by_this_review" not in decision or \
        decision["r9_vcs_authorized_by_this_review"] is not False:
    raise SystemExit("M518 r8 failure-review authority semantics drift")
```

Do not change `false` to `true`. Do not grant launch authority to the r8 failure
review. New r10 path/version/SHA bookkeeping is mandatory one-shot hygiene, not
a second functional repair.

The r10 authoring package must keep these production identities byte-exact:

- RTL `8a7ec118...f142d6`
- SVA `89d4d711...17c1f5`
- TB `88775120...962e56`
- filelist `09e43560...22fea`

It must also preserve the r9 release/raw `#0.2` repair, the r8 V08 line-765
settle repair, V01--V20 phase flow, expected cycles, numeric oracles, 51
assertions, 25 covers, exact PASS signature, wrong-TB negative control, and
Full64 VCS commands. It needs a new r10 contract, runner identity, double-sealed
launch admission, and canonical path
`results/m518_matched_fixed_t10_atlif_vcs_r10_exact_20260827`. A new independent
static reviewer must check exact external JSON-key existence/value and authorize
one literal r10 runner SHA. This review does not authorize that execution.

At review close, `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
