# M576 / M519 R8 DC launch-admission candidate receipt-blind static hammer r1

Date: 2026-08-28  
Verdict: `PASS_M553_M519_R8_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER`  
Score: **100/100**  
Severity: **P0=0, P1=0, P2=0**

## 1. Outcome

The exact `launch_now=false` candidate passes this fresh independent,
receipt-blind, read-only static hammer. It correctly freezes the future M519 R8
three-axis setup/area-only DC identity while authorizing **no execution now**.

This review invoked no runner, Synopsys executable, VCS executable, simulator,
CPU/GPU experiment, remote command, or network job. It created neither the R8
canonical result nor the attempt sentinel and did not modify `docs/359`.

The pass authorizes only the next release-chain step: a separately authored and
double-sealed `launch_now=true` final release may be prepared for independent
final-release hammering. It does **not** authorize DC.

## 2. Candidate identity and zero-authority boundary

- Candidate:
  `contracts/m553_m519_r8_setup_area_three_axis_dc_launch_admission_candidate_r1_20260827.json`
- Candidate SHA256:
  `43e601df0d20754d5e7f65033b0958c42f2dd0b99b4abe3336d051bf22f7ad59`
- Candidate member-sidecar file SHA256:
  `c65dccef71c34b4070114780a92eba763fcb8bc9aa9a1052b4ec7c7960af1418`
- Candidate outer-seal-file SHA256:
  `81b2176b8cf121241a82ff9e421cf4565ad463d0f6a5580af5db67fd70ccba9f`
- Strict JSON parsing with duplicate-key and non-standard-number rejection:
  PASS. The candidate is a regular file, not a symlink.
- Schema and status exactly match the candidate contract; `launch_now=false`.
- The authorization object is a closed seven-key object:
  `max_attempts=0`, and `run_dc`, `run_vcs`, `run_formality`, `run_pt`,
  `run_ptpx`, and `run_remote` are all `false`.
- The separately described future authorization is not current authority. Its
  `max_attempts=1`/`run_dc=true` values are usable only after the remaining
  release-chain gates.
- Candidate and recovery-contract member sidecars and outer seals verify.

No area, setup timing, hold closure, power, energy, throughput/area, complete
FC2, system speedup, paper-PPA, or headline claim follows from this review.

## 3. Closed R8 source, tool, and library identity

The candidate identity has exactly the same **36 unique keys** as the frozen R8
runner's literal closed-key set. No key is missing, added, or renamed. Every
candidate path/SHA field cross-equals the corresponding recovery-contract
field.

The recovery contract contains the exact closed **17-file** set. All 17 current
bytes match, including runner, Tcl, filelist, SDC, twelve RTL files, and
`docs/359`.

Key recomputed identities are:

| Object | SHA256 |
|---|---|
| R8 exact-SHA runner | `bd830577a7f31413189c78355c3e9467a567e0b90c1e0edcd6d1707d1b7e73c2` |
| R8 setup/area Tcl | `c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe` |
| R8 recovery contract | `33273e1411cff09f793906a61d4c68964c299aad8dceae91921a5229bdf5acf4` |
| DC entry / `snps_shell` wrapper | `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2` |
| live `common_shell_exec` | `bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391` |
| slow DB | `79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af` |
| fast DB | `a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a` |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

Static wrapper inspection confirms that the `dc_shell` symlink resolves to the
frozen `snps_shell` wrapper and that the long-lived executable is the separately
frozen `common_shell_exec`. No tool executable was invoked to establish this.

## 4. M546/M550 and failed-predecessor provenance

The M546 author handoff and M550 independent source hammer both pass their
member manifests and outer seals. Candidate-bound identities match:

- M546 manifest:
  `305d672755e78e8a849f0ce28edbb17e23cea5606185d0250045d4f7ec8f8797`
- M546 outer-seal file:
  `b081cb4b5cd8bff1db62976efc8a2db36298dcc518eafe25356b6f8d266e1dd4`
- M550 review JSON:
  `c29560558fbe519ef7969e17c9545b56d9c87525bd4e5b533dd76e6e45ef73f8`
- M550 manifest:
  `3b326d0582d68a75d77e39d3c55db1e623f2d64bcf2fd1af1f5cc474f65a06e7`
- M550 outer-seal file:
  `a6fff5def6c655cdf6f32b2f33a8430eb485fc3eed18db0b35c4cb14fc35d585`
- M550 remains `97/100`, `P0/P1/P2=0/0/2`, and explicitly does not
  authorize a launch.

All five R5 sealed bases pass both seals and their actual outer-seal-file hashes
match the candidate. The R6 failed review and R7 disqualified review likewise
pass their seals and remain correctly classified:

- R5 DC attempt: `FAILED_OR_INCOMPLETE_DO_NOT_CITE`; runner rc 42, monitor rc
  1, and runtime resource latch 1. Its intermediate K1 QoR is not citable.
- R6: `FAIL_STATIC_HAMMER__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION`, 84/100,
  `P0/P1/P2=0/3/1`.
- R7: `DISQUALIFIED_REVIEWER_TOOL_INVOCATION__R7_SOURCE_BLOCKED__NO_LAUNCH_ADMISSION`,
  42/100, `P0/P1/P2=2/2/2`; reviewer eligibility is false and the recorded
  accidental DC-executable invocation count is one.

R8 uses a distinct canonical result and attempt sentinel, freezes new runner/Tcl
bytes, and checks the permanent predecessor states before preflight. Therefore
the R5 intermediate reports cannot be silently admitted as R8 evidence.

## 5. Unique-attempt and release-chain state

At review time all of the following are absent:

- `dc_handoff/runs/m519_r8_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260827`
- `dc_handoff/runs/.m519_r8_channel_local_fault_dc_attempt_consumed`
- `contracts/m519_r8_setup_area_three_axis_dc_launch_admission_r1_20260827.json`
- the candidate-hammer path named by the candidate release chain
- the final-release hammer path named by the candidate release chain

Thus candidate creation consumed no attempt, no old result can be overwritten,
and no final authority currently exists.

## 6. Fail-closed resource and collision gates

The runner preserves the following mandatory gates for every axis:

- three samples separated by 10 seconds;
- CommitHeadroom at least 64 GiB, MemAvailable at least 128 GiB, and SwapFree
  at least 32 GiB;
- zero cgroup fail/under-OOM/OOM-kill counters;
- no same-UID DC/FM/PT/VCS collision;
- a runtime latch for three consecutive samples strictly below 32 GiB commit
  headroom and immediate latches for memory, swap, cgroup, collision, or identity
  faults;
- synchronous `runtime_final` gate plus both monitor rc=0 and
  `PASS_FINAL_GATE_ACK` before a point can pass;
- new preflight for K8 and K1x8 plus a final recovery preflight.

First-preflight rejection does not consume the attempt. Any failure after
attempt consumption is labeled `FAILED_OR_INCOMPLETE_DO_NOT_CITE`, double
sealed, and moved to a unique quarantine.

A single read-only host snapshot during this review showed approximately 78.56
GiB CommitHeadroom, 395.96 GiB MemAvailable, 54.56 GiB SwapFree, and zero
cgroup OOM counters. This is **not** the required multi-sample launch evidence.
Moreover, foreign UID 1909 PID 580855 was still running `simv`. The R8 runner's
machine check is same-UID, but the project-wide conservative shared-host policy
requires visible EDA/simulator collision clearance before an actual launch.
Therefore M519 remains blocked from execution now regardless of this PASS.

## 7. Explicit reproduction of M550 P2-1

M550 P2-1 is accurately preserved and is not silently upgraded:

- `resource_*_external_collisions.tsv` has the complete eleven-field tuple:
  timestamp, label, kind, PID, PPID, UID, starttime, state, comm hex,
  executable hex, and NUL-preserving command-line hex.
- `descendant_identity_faults.log` has only timestamp, sample, PID, and status.
  It is **not** a complete tuple ledger.
- An ancestry mismatch or candidate identity change sets
  `M519_R8_DESCENDANT_IDENTITY_FAULT=1`; the runtime monitor maps this to an
  identity fault, latches a nonpassing result, and requires both a zero monitor
  rc and final ACK. Thus such a fault cannot silently pass.
- If a future run encounters this fault, the post-run receipt review must
  reconstruct the full tuple from independently sealed evidence or keep the
  result noncitable.

The underlying P2 remains a disclosed receipt-quality boundary, not a new
candidate defect; hence this candidate hammer has P2=0.

## 8. Decision and only legal next steps

Decision fields:

- `candidate_pass=true`
- `launch_authorized_now=false`
- `separate_launch_now_true_final_release_required=true`
- `fresh_final_release_hammer_required=true`
- `wait_for_collision_clearance_and_stable_resources=true`

The true release, if authored, must:

1. preserve the exact 36-key identity and 17-file closure reviewed here;
2. use a closed seven-key authorization with **only one** attempt:
   `max_attempts=1`, `run_dc=true`, and every other run flag false;
3. bind this fresh review and the exact candidate SHA in separately sealed
   provenance, without weakening M550 P2-1 or reopening R5--R7;
4. receive a fresh independent final-release hammer;
5. freshly establish shared-host collision clearance and stable resources;
6. still pass the frozen runner's own per-axis collision/resource preflights;
7. pin both runner SHA and final-release SHA at invocation time.

Only after all seven conditions may root invoke the runner once. A failed or
consumed attempt may not be renamed, reused, overwritten, or rerun. A successful
raw result will still be pending independent receipt review and will not by
itself be paper-ready PPA.

