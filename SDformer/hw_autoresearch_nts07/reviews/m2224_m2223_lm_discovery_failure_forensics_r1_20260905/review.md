# M2224 independent review: M2223 failure forensics

Date: 2026-09-05. Reviewer: independent `/root/m2224_lm_discovery_review` agent. Scope: read-only source, sealed raw logs, receipts, and CPU parsing; no license query or EDA execution.

Verdict: **PASS, 97/100** for failure forensics and a new-identity parse-only recovery. This does not promote the quarantined M2223 package into a production PASS. The consumed M2223 attempt and its failed marker remain intact.

## What the evidence establishes

The authorized LM discovery exited with code 0. Its pipeline exited with code 2 at the Python checker. The raw directory, consumed-attempt directory, and M2222 source review have exhaustive valid double seals. The six contract-pinned source files remain unchanged. The two repository snapshots are identical; the before/after tool censuses are empty; the frame directory is empty and no NDM/NLIB file exists.

The log contains exactly one `M2221_FATAL_FAIL_CLOSED:` literal: the echoed Tcl source line `puts stderr ...`. LM's own command log records `source -echo -verbose`. No anchored runtime fatal marker appears. The original frozen checker reproduces `Tcl fatal diagnostic` using the sealed logs alone, before producing a receipt. The runtime startup, command, option, set/readback, no-side-effect, and raw-PASS markers are present exactly once in order.

Measured runtime observations:

| Observation | Result |
|---|---|
| Four queried LM commands | All available |
| `lib.configuration.local_output_dir` | Query rc=1; unregistered; `Invalid option name` |
| `lib.setting.milkyway_exec` | Query rc=0; registered |
| Session-local Milkyway option set/readback | rc=0/0; exact expected executable path |
| Conversion / create-lib / P&R calls | Zero in the inspected source and recorded execution |

These observations explain the M2207 documentation/runtime mismatch. Presence of `generate_frame_from_mw` and a successful option readback do not establish that conversion will succeed.

## Recovery requirements

1. Preserve the old checker, raw directory, seal, failure marker, and consumed attempt. A fresh parser identity may read these pinned artifacts and emit a fresh result elsewhere.
2. Distinguish exact runtime markers from echoed Tcl text. Injected anchored fatal markers must fail, as must duplicate/missing runtime markers, nonzero return codes, and changed raw hashes. Removing every line that contains the token is unacceptable.
3. Handle the authenticated relocation explicitly. Runtime startup and `execution_contract.json` bind `.m2223_m2221_lm_command_option_discovery_work.3569314/isolated_cwd`; cleanup renamed the package to the PID-3569314 quarantine. Replacing the fatal check alone would expose a second failure because the old checker compares logged paths with the current quarantine path. Accept only this exact sealed mapping; do not weaken arbitrary path checks.
4. Retain every other option-state, set/readback, output-manifest, isolation, snapshot, and seal check. Review the new parser source independently, then independently review its parse-only result.

Authorization is limited to that new parser and CPU-only re-analysis. **No LM retry, license query, Milkyway invocation, library conversion, or P&R is authorized by M2224.** A later conversion requires a separately reviewed source and execution identity.

One P2 caveat remains: before/after censuses do not continuously observe transient processes. The inspected Tcl provides no executable conversion path, so this limitation does not block the narrow discovery interpretation; it prevents upgrading the result into exhaustive process-level or library-compatibility evidence.

`docs/359` remains at its frozen SHA. No Git mutation or legacy-file edit was performed by this review.
