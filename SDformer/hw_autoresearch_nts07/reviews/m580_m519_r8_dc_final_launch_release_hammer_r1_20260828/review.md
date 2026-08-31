# M580 / M519 R8 DC final-launch release receipt-blind hammer r1

Date: 2026-08-28  
Verdict: `FAIL_FINAL_RELEASE_HAMMER__M576_STATUS_PROVENANCE_MISMATCH__NO_DC_AUTHORIZED`  
Score: **96/100**  
Severity: **P0=0, P1=1, P2=0**

## 1. Outcome

The M519 R8 final release does **not** pass this fresh independent,
receipt-blind, read-only final-release hammer. Its executable identity and
fail-closed runner chain are otherwise closed, but one final-release
provenance assertion is false: the release records an M576 candidate-hammer
status string that is not the status stored in the exact sealed M576
`review.json` it binds.

This is a bounded metadata repair, not evidence that the M576 directory is a
fabricated legacy M553 review. The exact review SHA, manifest SHA, outer-seal
file SHA, milestone, directory, score, and severity counts all establish that
the sealed package is the fresh M576 successor. Nevertheless, a strict final
release may not replace an upstream receipt's literal status with an invented
normalization while naming the field `candidate_hammer_status`.

No runner, DC, VCS, simulator, Formality, PT, PTPX, CPU/GPU experiment, remote
command, or network operation was invoked. The canonical result and attempt
sentinel remain absent. `docs/359` remains at the frozen SHA.

## 2. Blocking finding

### P1-1: final release does not reproduce the sealed M576 status

The final release says:

```text
fresh_successor_provenance.candidate_hammer_status =
PASS_M576_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER__NO_DC_AUTHORIZED
```

The exact sealed review named by the same release says:

```text
review.json.status =
PASS_M553_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER
```

The review itself separately and correctly identifies
`review_identity.milestone=M576` and
`review_identity.directory=reviews/m576_m519_r8_dc_launch_admission_candidate_hammer_r1_20260828`.
It also explicitly states that the legacy literal M553 review path is absent
and that a future release must bind this M576 successor. Thus the correct
repair is to preserve the M576 path/SHA/seals and record the upstream status
verbatim; it is not to create or impersonate the absent M553 directory.

Impact: the frozen release JSON contains a false provenance value. The runner
does not machine-check this auxiliary provenance object, so the error would not
fail inside the runner. That makes the independent final-release hammer the
only fail-closed gate that can prevent the inaccurate release from being used.
Execution is therefore not authorized.

Required repair:

1. author a new final-release revision whose
   `candidate_hammer_status` exactly equals the sealed M576 `review.json`
   status;
2. preserve the exact candidate SHA, M576 review SHA, M576 manifest and outer
   seal hashes, the 36-key identity, authorization, and all claim boundaries;
3. double-seal the new release and obtain a fresh independent final-release
   hammer;
4. do not edit, rename, or fabricate the existing M576 receipt or an old M553
   review path.

## 3. Final release and candidate identity checks that passed

- Final release:
  `contracts/m519_r8_setup_area_three_axis_dc_launch_admission_r1_20260827.json`
- Final-release SHA256:
  `426acd92672037dcab072c98fa3183bbb953cc35924adc26499cf82b1ba439ba`
- Member-sidecar-file SHA256:
  `74388d22345903581df7263e5369ac502d305f33718ea71a5ff9b2735cbbc4bd`
- Outer-seal-file SHA256:
  `c32e65b4fe811a732728f1f1315e8f262e9adfc9b4ad3a6d9280fc011c5b2b1b`
- Strict JSON parsing, regular-file checks, member sidecar, and outer seal:
  PASS.
- Status/launch: exact authorized status and `launch_now=true`: PASS.
- Authorization is the closed seven-key object with `max_attempts=1`, only
  `run_dc=true`, and all other run flags false: PASS.
- The final release and the frozen M553 candidate have exactly 36 identity
  keys and their complete identity objects are byte-value equal: PASS 36/36.
- Candidate SHA frozen by the release is the current exact candidate SHA:
  `43e601df0d20754d5e7f65033b0958c42f2dd0b99b4abe3336d051bf22f7ad59`.

The final release correctly binds the actual M576 review member SHA
`2a1203f45acd2594d123c724a722e33874a13ded6f404cda59034de72a4aa7b0`,
manifest-file SHA
`432256f837fc97120f00938c4c33385e4e9482ece02fd10e8f3f6dda58571899`,
and outer-seal-file SHA
`9b7a8c28f100a1f7b5f701bd147f5097792db71f96867dd08a0a05af3194ef21`.
Both M576 seals verify. Its score and severity are exactly 100 and 0/0/0.
These facts prove freshness and make the incorrect status literal repairable,
but do not waive P1-1.

## 4. Closed files, tools, libraries, and source provenance

All 17 recovery-contract `exact_files` paths are present as regular files and
their current SHA256 values match. This includes the exact runner, Tcl,
filelist, SDC, twelve RTL files, and `docs/359`.

The launch-time tool identities also match:

| Object | Current SHA256 | Result |
|---|---|---|
| `dc_shell` entry | `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2` | PASS |
| `snps_shell` wrapper | `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2` | PASS |
| `common_shell_exec` | `bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391` | PASS |
| slow DB | `79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af` | PASS |
| fast DB | `a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a` | PASS |

`dc_shell` resolves to the frozen `snps_shell` wrapper. The M546 author handoff
and M550 independent static hammer verify under both seals. M550 remains
97/100 with P0/P1/P2=0/0/2 and does not itself authorize execution.

## 5. Failed-predecessor isolation

All five R5 bases verify under their inner and outer seals. R5 remains
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`; its partial QoR cannot be promoted to R8.
The sealed R6 review remains
`FAIL_STATIC_HAMMER__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION`, 84/100,
P0/P1/P2=0/3/1. The sealed R7 review remains
`DISQUALIFIED_REVIEWER_TOOL_INVOCATION__R7_SOURCE_BLOCKED__NO_LAUNCH_ADMISSION`,
P0/P1/P2=2/2/2, reviewer-ineligible, with one recorded accidental DC-executable
invocation. R8 uses a distinct result and attempt identity. No failed or
disqualified predecessor is reopened, rewritten, or cited as R8.

## 6. Unique-attempt and current shared-host state

At review time both were absent:

- `dc_handoff/runs/m519_r8_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260827`
- `dc_handoff/runs/.m519_r8_channel_local_fault_dc_attempt_consumed`

Release authoring and this hammer therefore consumed no attempt. A process
snapshot still showed foreign UID 1909 PID 580855 running `simv`. This is not a
fresh full launch preflight, but it is sufficient to keep the project-wide
shared-host execution state **BLOCKED**. Even a repaired PASS release must not
be used until a new full shared-host collision check is clear and the frozen
runner independently passes all resource/collision preflights.

## 7. M550 P2 descendant-identity boundary

The final release preserves M550 P2-1 accurately:

- the external-collision TSV records the full eleven-field tuple;
- `descendant_identity_faults.log` records only timestamp, sample, PID, and
  status and must not be described as a complete tuple ledger;
- ancestry/candidate identity faults set the fail-closed identity flag, force
  a nonpassing runtime/final gate, and cannot silently produce a passing
  result;
- a future receipt with such a fault is noncitable unless the complete tuple
  is reconstructed from independently sealed evidence.

This inherited disclosed boundary is not a new P2 finding in this release
hammer. It also does not mitigate P1-1.

## 8. Final decision

Decision fields:

- `final_release_pass=false`
- `dc_launch_authorized=false`
- `score_out_of_100=96`
- `P0/P1/P2=0/1/0`
- `result_absent=true`
- `attempt_sentinel_absent=true`
- `current_shared_host_execution_blocked=true`

The next legal step is a bounded final-release revision and a fresh independent
final-release hammer. No one-shot runner invocation is permitted from the r1
release, including after the foreign `simv` clears. A future PASS release would
still authorize only one attempt after a new live full shared-host
collision/resource preflight passes; it would not be a result receipt or a
paper-PPA claim.

