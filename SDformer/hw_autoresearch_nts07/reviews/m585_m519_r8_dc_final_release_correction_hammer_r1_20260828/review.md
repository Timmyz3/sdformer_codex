# M585 / M519 R8 DC final-release correction hammer r1

Date: 2026-08-28  
Verdict: `PASS_M519_R8_FINAL_RELEASE_STATUS_CORRECTION_HAMMER__ACTUAL_DC_BLOCKED_BY_FOREIGN_SIMV`  
Score: **100/100**  
Severity: **P0=0, P1=0, P2=0**

## 1. Outcome

The M583 provenance-correction overlay passes this fresh independent,
receipt-blind, read-only hammer. It exactly binds the immutable M519 R8 release,
the sealed M580 failed final-release hammer, and the literal status in the
sealed M576 successor review. It corrects only M580 P1-1's reviewer-facing
interpretation of one redundant status field that the runner does not consume.

The overlay is not a replacement launch contract and grants no execution:
`max_attempts=0`, `run_dc=false`, and every other run flag is false. It does not
alter the original release bytes, 36-key execution identity, recovery contract,
17 exact files, runner, Tcl, SDC, filelist, tool, wrapper, actual executable,
timing libraries, authorization budget, canonical result, attempt sentinel,
M550 P2 boundary, or any claim boundary.

No runner, DC, VCS, simulator, Formality, PT, PTPX, CPU/GPU experiment, remote
command, or network operation was invoked. The result and attempt sentinel are
absent. `docs/359` remains frozen.

## 2. Correction identity and double seal

- Overlay:
  `contracts/m583_m519_r8_dc_release_status_correction_overlay_r1_20260828.json`
- Overlay SHA256:
  `20353e40e4d2420eeb7a3b94aaf10a8a506661d27e12729afeb5b081489dbd65`
- Member-sidecar-file SHA256:
  `2b9ef1f0f4d878259a1f8fe71c003e3537d3b32f0118ee5846ae52d060f3793b`
- Outer-seal-file SHA256:
  `e4b38e2ef0f3cb4592be87cc4a2c726131b80144ad0de3340ea0bbaa5cbd9f13`
- Strict JSON parsing, regular-file check, member sidecar, and outer seal:
  PASS.
- Overlay authorization is a closed no-execution statement:
  `max_attempts=0` and all six run flags false.

The original release remains byte-identical at:

```text
426acd92672037dcab072c98fa3183bbb953cc35924adc26499cf82b1ba439ba
```

This exactly equals `bound_release.sha256`. Its member and outer seals still
verify. The overlay names the exact original field and value:

```text
fresh_successor_provenance.candidate_hammer_status =
PASS_M576_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER__NO_DC_AUTHORIZED
```

## 3. M580 failure and authoritative M576 status

The overlay binds the exact sealed M580 failure package:

| Evidence | Live SHA256 | Overlay match |
|---|---|---|
| M580 `review.md` | `7ec8e1db63ca91a24780eb82e104983c322be76fc2456e60ea7acfc3c2945f3a` | PASS |
| M580 `review.json` | `c52b3c34d0cf98ab5f8c526e2ca0a2c869ebc700115e5664dc3f9a90f84e021e` | PASS |
| M580 manifest file | `b66ec94f291e8a085eb9c4aa1ad2d243fc1700276bcae1258818464bcb7953f3` | PASS |
| M580 outer-seal file | `5cdfe1b34523c1be026a6c5a5ea48b0fb73153f35eed3f0416eae3ac6bd47a10` | PASS |

M580 remains FAIL 96/100, P0/P1/P2=0/1/0, with exactly the status-literal
mismatch as its only finding.

The authoritative source is the exact sealed M576 `review.json` at SHA256:

```text
2a1203f45acd2594d123c724a722e33874a13ded6f404cda59034de72a4aa7b0
```

Its literal `/status` value exactly equals the overlay correction:

```text
PASS_M553_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER
```

The review independently identifies `milestone=M576`, its M576 directory,
100/100, and P0/P1/P2=0/0/0. The absent legacy literal M553 review path is not
created or impersonated. Thus the correction preserves the distinction between
the candidate-named status token and the fresh M576 receipt identity.

## 4. Execution identity and live-byte closure

The immutable release and M553 candidate still contain exactly 36 identity
keys, and their complete identity objects are value-equal. The recovery
contract remains double-sealed and contains exactly 17 `exact_files`; all 17
current SHA256 values match.

The live tool chain remains unchanged:

| Object | SHA256 | Result |
|---|---|---|
| `dc_shell` entry | `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2` | PASS |
| `snps_shell` wrapper | `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2` | PASS |
| `common_shell_exec` | `bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391` | PASS |
| slow DB | `79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af` | PASS |
| fast DB | `a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a` | PASS |

The original release remains the runner-consumed one-attempt authorization:
`launch_now=true`, `max_attempts=1`, only `run_dc=true`. The overlay does not
change or add runner input. This is appropriate because the corrected status
is audit provenance, not a runner-consumed admission key.

## 5. Unique attempt, P2 boundary, and claim limits

At hammer time both remain absent:

- `dc_handoff/runs/m519_r8_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260827`
- `dc_handoff/runs/.m519_r8_channel_local_fault_dc_attempt_consumed`

Therefore no attempt was consumed by release authoring, correction authoring,
or either hammer.

M550 P2-1 remains unchanged: the descendant identity-fault side log is not a
complete tuple ledger; any such fault remains fail-closed and forces a
nonpassing result; post-run review must reconstruct a complete tuple from
sealed evidence or keep the result noncitable. The overlay does not weaken,
rename, or upgrade this boundary.

No DC completion, area, timing, hold closure, power, energy,
throughput-per-area, complete FC2, paper-PPA, system-speedup, or headline claim
follows from this PASS.

## 6. Current execution decision

A live process snapshot still shows foreign UID 1909 PID 580855 running
`simv`. This keeps actual M519 DC execution **BLOCKED** under the project-wide
shared-host policy. The snapshot is not a launch preflight and cannot be used
to reserve resources or waive any gate.

Only after that collision has cleared may root make a new go/no-go decision.
Before exactly one invocation, root must freshly establish full shared-host
collision clearance and stable resource headroom, pin the immutable runner and
original release SHAs, and then let the runner independently pass all per-axis
collision/resource gates. A preflight rejection consumes no attempt; a failure
after consumption must be quarantined and double-sealed. No bypass is allowed.

Final decision:

- `correction_overlay_pass=true`
- `M580_P1_1_closed=true`
- `score_out_of_100=100`
- `P0/P1/P2=0/0/0`
- `actual_dc_currently_blocked=true`
- `one_attempt_may_be_considered_only_after_fresh_live_gates=true`

