# M540 / M519 R7 setup-area flow static hammer r1

Date: 2026-08-27  
Verdict: `DISQUALIFIED_REVIEWER_TOOL_INVOCATION__R7_SOURCE_BLOCKED__NO_LAUNCH_ADMISSION`

## Outcome

**42/100; P0=2, P1=2, P2=2.** R7 fixes the three principal R6 source defects: the contract `exact_files` set is closed and verified 17/17 before preflight, the runtime loop and `runtime_final` use the same gate and consecutive-commit counter, and the campaign/ancestor checks carry PID starttime identities into liveness, collision exclusion, and TERM decisions. The synthesis Tcl is also a clean setup/area flow with one `compile_ultra`, zero incremental compile, and zero pre-CTS hold optimization.

R7 nevertheless must not receive a launch admission. The source has one critical executable-identity defect that can run a full DC child without the runtime monitor. In addition, this reviewer accidentally invoked the Synopsys wrapper while investigating that defect. The invocation used an unsupported `-print_exec` argument, printed the tool usage text, and exited without reading a design or creating a synthesis result, but it still violates this request's strict zero-DC-reviewer rule. This review is therefore evidence and a return-to-author report only; it is not an eligible fresh zero-EDA admission review.

## P0 findings

### P0-1 | Frozen campaign executable is the wrapper script, not the live DC executable

Runner line 8 sets:

```text
m519_r7_dc_exe=$(realpath .../bin/dc_shell)
```

On this installation, `dc_shell` resolves to `.../bin/snps_shell`. That file is a shell wrapper. Its static source sets `dcexec_name="common_shell_exec"` and ends with `exec .../linux64/syn/bin/common_shell_exec ... -shell dc_shell ...`. Consequently the launched PID's `/proc/<pid>/exe` transitions from the shell interpreter to `common_shell_exec`; it does not equal `.../bin/snps_shell`.

`m519_r7_capture_dc_identity` requires exact equality with the wrapper path. After about two seconds it returns failure. Runner lines 786-789 then execute `wait <child>` before returning 47, and the runtime monitor is never launched. A real synthesis can therefore run to completion without the 10-second resource gate, emergency latch, collision monitor, or exact TERM path. The result will eventually quarantine, but the resource-safety contract has already been bypassed.

Required repair: freeze and hash the actual live executable (`.../linux64/syn/bin/common_shell_exec`, including the invocation/shell selector identity), or launch through a small supervised wrapper whose stable PID/executable contract is explicitly modeled. Identity-capture failure must terminate the exact launch tree fail-closed; it must never wait for an unmonitored DC job to finish.

### P0-2 | This reviewer is not a valid zero-EDA fresh reviewer

During read-only wrapper investigation, this reviewer mistakenly issued:

```text
/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -print_exec
```

The unsupported argument caused `common_shell_exec` to print its usage and exit. No filelist, Tcl, RTL, library, design, synthesis command, result directory, canonical identity, or attempt sentinel was used or created. Nevertheless, a DC executable process was started once, contrary to the request's `eda_runs_authorized=0` rule. A different fresh reviewer must independently review the repaired identity before any launch admission is created.

## P1 findings

### P1-1 | Current tool/library and sealed-basis bytes are not launch-time pinned

The runner correctly cross-compares admission fields with the contract. It also verifies every one of the 17 `exact_files`. But the launch path never calls `m519_r7_expect` on the current `dc_shell`, slow library, or fast library. Those files are not in `exact_files`; admission/contract string equality alone does not verify their current bytes.

Likewise, the five R5 basis directories are checked only for internally self-consistent `SHA256SUMS` and outer seals. The actual SHA256 of each `SHA256SUMS.seal.sha256` file is not compared with the outer-seal hash frozen in the contract/admission. A modified-and-resealed basis can therefore pass the runner even though the declared provenance no longer names its actual bytes.

Current review-time bytes happen to match the contract: DC wrapper `23a410...e6d2`, slow DB `79fb5f...51af`, fast DB `a707b6...c91a`, and all five R5 outer-seal file hashes match. That snapshot does not replace launch-time checks.

Required repair: before preflight, compare the current DC wrapper, actual live executable, both libraries, and all five R5 outer-seal files against the contract/admission hashes.

### P1-2 | R6 failed-review provenance is asserted but not cryptographically carried into launch

The contract names the R6 review path, verdict, score, and P-counts, but it contains no R6 outer-seal file SHA. The runner neither verifies the R6 review's nested seals nor pins its outer-seal file. This breaks the requested R5/R6 provenance chain for a bounded R6-to-R7 repair.

The present R6 review outer-seal file SHA is `ae0b56971ffe4da537527364ea39b88e8c433e32988c4c0206e67a208e903d70`. Required repair: freeze this identity in the new contract and future admission, verify the actual outer-seal file plus inner/outer seals before preflight, and include it in canonical/quarantine root evidence.

## P2 findings

### P2-1 | Runtime collision provenance is not complete

Preflight PID-tree records and campaign-descendant runtime/high-water records preserve PID, PPID/UID where required, starttime, comm hex, executable hex, and NUL-preserving cmdline hex. However, an external runtime EDA collision is written only as `pid:comm` in `resource_runtime.log`. It has no collision starttime, UID, executable, or complete cmdline identity in that sample. This is sufficient to latch a recognized collision but insufficient for independent reconstruction and PID-reuse audit.

Required repair: emit a separately sealed, full identity record for every runtime collision/mismatch using the same hex encoding as the descendant evidence.

### P2-2 | Normal exit can be confused with identity mismatch in the zombie window

`m519_r7_proc_identity` maps an unreadable `/proc/<pid>/exe` to the literal `UNREADABLE` while still returning success. `m519_r7_root_state` then reports state 2. A normally exited child may briefly remain a zombie before the parent reaps it, in which state `/proc/<pid>/exe` can be unreadable. The monitor can therefore label an ordinary exit as campaign PID identity mismatch. This is fail-safe, not an unsafe pass or wrong TERM, but it creates nondeterministic false quarantine.

Required repair: parse and distinguish the `/proc/<pid>/stat` zombie state, and synchronize exit/reap notification so a normal exact-child exit maps to absent/completed rather than PID reuse.

## Passed static checks

- Request nested seals and request identity pass.
- Contract JSON and nested/outer seals pass; author SHA identities match the request.
- `exact_files` has exactly the expected 17 paths and all 17 current files match their frozen SHA256.
- Future admission, R7 canonical result, and R7 attempt sentinel are absent.
- Future admission authorization and identity key sets are closed; its path/SHA fields are cross-compared with the contract before preflight.
- Runtime loop and `runtime_final` call the same `m519_r7_gate_current_snapshot`; the final sample updates the same consecutive commit counter, emits a final gate ACK, and the parent checks both ACK and monitor return code.
- Campaign identity carries PID, starttime, UID, and executable; ancestry walks capture and reread each `(PID,starttime)` pair. Identity mismatch latches and is not signalled.
- Preflight is three samples with two 10-second intervals for every axis; K8/K1x8 receive fresh preflight and post-K1x8 receives recovery. Commit is a strict `<32 GiB` three-consecutive latch; MemAvailable, SwapFree, cgroup/OOM, and recognized collision are immediate gates.
- Per-preflight evidence is double sealed and moved under the run root; attempt-consumed failures quarantine and the enclosing root seal covers nested evidence.
- Tcl static command count is one `compile_ultra`; no incremental compile, `only_hold_time`, or hold-fix command exists. All three architecture points use the same Tcl, filelist, SDC, libraries, process corner, 3 ns constraint, and setup/area gate. Hold remains diagnostic only.
- `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Claim boundary and next action

`launch_admission=false`, `dc_completed=false`, `area=false`, `timing=false`, `hold_closed=false`, `power=false`, `energy=false`, `throughput_per_area=false`, `complete_fc2=false`, `system_speedup=false`, `headline=false`.

The author must produce a new bounded source identity repairing P0-1 and P1 provenance closure. Because of P0-2, a different fresh reviewer must then obtain `P0=0 && P1=0`. This review neither authorizes nor recommends any EDA launch.
