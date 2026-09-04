# M550 / M519 R8 setup-area flow fresh independent static hammer r1

Date: 2026-08-27  
Verdict: `PASS_STATIC_HAMMER__P0_P1_CLOSED__SEPARATE_LAUNCH_ADMISSION_PERMITTED`

## Outcome

**97/100; P0=0, P1=0, P2=2.** The R8 source closes the launch-safety defects identified in the R6 failed review and the R7 disqualified review. The three architecture points are eligible for a separately authored, separately double-sealed, one-attempt launch admission. This review is not itself a launch admission and does not authorize DC, VCS, PT, PTPX, Formality, remote work, or a CPU-heavy experiment.

This reviewer performed a fresh read-only static audit. No Synopsys executable, VCS executable, runner, CPU DSE, or remote command was invoked. The review read the complete request, author handoff, R8 runner/Tcl/contract and all 17 `exact_files`, the static `snps_shell` wrapper text, the R5 final failed receipt, the R6 failed review, and the R7 disqualified review.

## Identity and provenance closure

- R8 runner SHA256: `bd830577a7f31413189c78355c3e9467a567e0b90c1e0edcd6d1707d1b7e73c2`.
- R8 Tcl SHA256: `c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe`.
- R8 recovery contract SHA256: `33273e1411cff09f793906a61d4c68964c299aad8dceae91921a5229bdf5acf4`; its inner and outer seals pass.
- The request and author-handoff inner/outer seals pass. The future launch admission, R8 canonical result, and R8 attempt sentinel are absent.
- Contract `exact_files` is a closed 17-path set. All 17 current files match their frozen SHA256, and the runner verifies the closed set and every current byte before the first resource preflight.
- `dc_shell` resolves to the frozen `snps_shell` wrapper. Static wrapper inspection confirms that this entry execs `common_shell_exec` with selector/root/Tcl argv. Current hashes match the contract for the entry/wrapper (`23a410...e6d2`), actual executable (`bf91e6...391`), slow DB (`79fb5f...51af`), and fast DB (`a707b6...c91a`). The runner performs each current-byte check before preflight; admission paths and hashes are cross-equal to the contract.
- All five R5 bases independently pass inner/outer seal verification and their actual outer-seal file hashes equal the contract. The R6 failed review and R7 disqualified review also pass their seals and status checks; R7 is explicitly ineligible to authorize launch.
- `docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Process identity and fail-closed launch review

- Stable campaign capture requires one fork birth with exact parent PID, proc starttime, UID, actual `common_shell_exec` path, and the seven-element NUL-safe argv `common_shell_exec -shell dc_shell -r <install-root> -f <exact-R8-Tcl>`. The process identity and argv are reread after parsing to close the wrapper-to-exec race.
- Capture failure sends TERM only after the fork birth `(PID,starttime,UID,parent)` still matches, then applies a bounded grace and KILL only to that same birth, waits for it, returns 47, and enters the EXIT-trap quarantine path. The helper restores `set -e` before `run_point` returns 47, so the top-level cannot continue to K8/K1x8 after capture failure.
- Exact-child liveness uses PID/starttime/UID/parent/executable/full-cmdline. A zombie with the exact birth is treated as completed; any non-zombie mismatch returns the fail-closed identity-mismatch state and is never signalled.
- Descendant exclusion walks and then rereads every ancestor `(PID,starttime)` pair. External DC/FM/PT/VCS candidates are recorded in a separately sealed TSV with timestamp, label, kind, PID, PPID, UID, starttime, state, comm hex, executable hex, and full NUL-preserving cmdline hex. A candidate whose identity changes before recording retains its previous complete tuple.

## Resource, failure, and synthesis-flow review

- Every axis has three preflight samples spaced by 10 seconds. All samples must satisfy commit headroom at least 64 GiB, MemAvailable at least 128 GiB, SwapFree at least 32 GiB, zero cgroup/OOM counters, and no same-UID external EDA collision. K8 and K1x8 receive fresh preflights; a final recovery preflight follows K1x8.
- Runtime loop and `runtime_final` call the same gate. The final sample updates the shared consecutive-commit counter, immediate Mem/Swap/cgroup/collision gates, and synchronous final ACK. The parent requires both monitor rc=0 and `PASS_FINAL_GATE_ACK`.
- The runtime commit latch is strict `<32 GiB` for three consecutive samples and resets after a recovered sample. Mem/Swap/cgroup/collision/identity faults latch immediately. Global commit accounting is not adjusted by per-process VmSize.
- First-preflight rejection is double sealed without consuming the attempt. Any failure after attempt consumption writes `RUN_FAILED_OR_INCOMPLETE`, double seals the work root, and moves it to a unique quarantine. Success also double seals before canonical publication.
- The Tcl contains exactly one executable `compile_ultra`, zero incremental compile, zero `set_fix_hold`, and zero hold-only optimization. K1/K8/K1x8 use the same Tcl, filelist, SDC, library pair, 3 ns constraint, and logic-only 0-macro boundary. Setup and design-rule checks are pass gates; hold is diagnostic and explicitly not closed at DC.

## P2 caveats

### P2-1 | Descendant identity-fault side log is not a complete tuple ledger

The external-collision TSV is complete, but runner lines 754--772 write ancestry/candidate identity faults to `descendant_identity_faults.log` using only timestamp, sample, PID, and status. The same event is fail-closed and sets `M519_R8_DESCENDANT_IDENTITY_FAULT`, so it cannot create an unsafe pass. For receipt quality, a future runner should preserve the last complete candidate/ancestor tuple in this side log as well.

### P2-2 | The future admission schema does not machine-check this fresh review identity

The admission identity key set is closed over the tools, libraries, RTL/SDC/Tcl, R5 bases, R6 failed review, and R7 disqualified review, but it has no key for this M550 fresh static hammer. This does not change the reviewed runner bytes or the P0/P1 launch behavior, and the main agent can still double-seal a post-review admission. The admission should carry this review's outer-seal file SHA in an auditable top-level provenance field and receive a separate static release check before execution; a future runner revision can make that relationship a machine-enforced gate.

## Authorization and claim boundary

The fresh static pass condition `P0=0 && P1=0` is met. The main agent may now author one separate, double-sealed launch admission that pins this exact R8 runner and cites this review. Until that admission exists and is independently checked, `run_dc=false`.

No PPA or performance claim follows from this source review: `dc_completed=false`, `area=false`, `setup_timing=false`, `hold_closed=false`, `power=false`, `energy=false`, `throughput_per_area=false`, `complete_fc2=false`, `system_speedup=false`, `headline=false`, and `paper_ppa_ready=false`.
