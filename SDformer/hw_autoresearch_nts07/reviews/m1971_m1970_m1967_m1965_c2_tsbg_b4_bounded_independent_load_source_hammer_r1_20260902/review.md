# M1971 independent source hammer: M1970 bounded independent-load TB

## Verdict

**FAIL, 89/100; P0=0, P1=2. No runner and no EDA are authorized.**

M1970 fixes the functional deadlock path from M1956/M1965 and the three functional/liveness omissions in M1967: each side has an independent valid and latched accept, the shared descriptor is driven on negedge and held until both accepts, a 100000-cycle whole-test watchdog starts before any task, and both named `join_any` forks are explicitly disabled.

The gate still fails because M1965 required runner-grade observability before a fresh attempt. M1970 has only four load-phase tokens, not BEGIN/END tokens for reset, full execute, replay, stale attack, recovery execute, and final checks. Its bounded timeout also lacks a distinct `M1970_LOAD_TIMEOUT` token plus phase, valid, accept, and explicit pending fields.

## Findings

- P1: only `full_load` and `recovery_load` BEGIN/complete tokens exist; the complete M1965 phase-token contract is absent.
- P1: the timeout fatal is bounded and partially diagnostic, but it is not a parseable full state dump meeting M1965.

## What passed

- Independent base/TSBG valid wiring and accept latching.
- Shared payload stability until both sides accept.
- Race-free negedge descriptor presentation.
- 10000-cycle per-descriptor and 100000-cycle whole-test bounds.
- Two named completion forks with two matching `disable` statements.
- Exact filelist entries for the frozen adapter, RTL, SVA, and M1970 TB.
- Preservation of 31 prior fatal sites, the unique PASS token, arithmetic, ledgers, attacks, local cycle gate, SVA/cover source, and docs/359.

The 100000-cycle global bound is safely above a conservative static envelope for this directed workload: the full phase has 576 baseline bundle beats and the recovery phase has 12, with bounded one-cycle request bubbles and at most eight response-delay clocks. This is a source argument, not a VCS measurement.

## Minimum repair

Create one additive successor that preserves all M1970 handshake/watchdog/fork logic and adds only:

1. BEGIN/END tokens for reset, full execute, replay attack, both reset recoveries, stale attack, recovery execute, and final checks.
2. A parseable `M1970_LOAD_TIMEOUT` line before fatal including phase, context/group/last, both valid/ready/accept/pending values, busy/fault state, and cycle.

After a different-author source PASS, a fresh runner may be authored. M1971 performed no license query, VCS, simv, DC, PT, or other EDA run.
