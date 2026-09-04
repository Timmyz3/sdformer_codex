# M2161 independent M2160 source hammer

## Verdict

**M2160 fails the independent source gate at 82/100; P0/P1/P2 = 2/0/0.**
M2162 is not authorized.  This review ran no license query, VCS, `simv`, SAIF
acquisition, DC, PT/PTPX, ICC2, or GPU job.

The report-before-reset control flow itself is now structurally correct, and
all 20 author-declared mutations were independently rejected.  Two new
adversarial tests nevertheless show that the final parser can still turn an
invalid acquisition into an admitted result: reset-rejection synonyms escape
the warning gate, and activity records outside an empty or wrong `INSTANCE`
escape the DUT-ownership gate.

## What is correct

The exact committed source at `c74321e3...` contains one direct M2018 frontend
at `SCHEDULE_MODE=0`, four filelist sources, and no parent dual-axis testbench,
public-name adapter, second executable axis, or schedule-mode-1 instance.  The
testbench uses the frozen slot-42 fixture and checks the 383-cycle preload,
228-element internal census, 20,292 cycles, 149 rows, 1,278 issues, 29,472
signed products, 24 commits, 1,788 bundles, 14,304 scalar reads/responses, and
all 24 context/slice accumulators.

The UCLI order also repairs M2152's control-flow defect.  Observation is enabled
before the first `run`; census and begin occur inside that run; after its
`$stop`, UCLI disables and reports `rtl_prehistory.saif` before requesting the
power reset.  It then re-enables observation and enters the second `run`; end
and functional pass occur before the second return marker, followed by disable
and `rtl_measurement.saif`.  Both raw files are given two-level file seals
before any parser call.  The diagnostic path is never passed to a power tool,
and the testbench claims only `power_reset_requested=1`.

The author receipt is exhaustive and double-sealed, the M2152 lineage and tool
identities match the contract, protected docs/359 remains `dedde7ce...`, and no
M2162 attempt, lock, or result exists.

## P0-1: reset-warning synonyms pass

`parse_runtime` rejects the exact M2151 messages but not their semantics.  Each
of the following explicit rejection messages was appended to an otherwise
valid runtime and unexpectedly passed:

- `Warning: This reset request was ignored.`
- `Warning: Power information reset request ignored.`
- `Warning-[POWER_RESET_IGNORED] Switching counters were not cleared.`
- `Warning: request to reset switching activity has been ignored.`

Consequently `final_result` can publish
`power_reset_acceptance.accepted=true` even when the log says the switching
counters were not cleared.  This directly violates the contract's
ignored/equivalent-warning rule; exact measurement duration alone does not
repair a false reset-acceptance fact.

## P0-2: SAIF records are not scoped to the DUT instance

`parse_saif` checks only that some `(INSTANCE` token exists, then counts all
activity records globally with regular expressions.  Two sealed forgeries pass
all 93,971-record, TX, conservation, toggle, and critical-cone gates:

1. an empty `dut_ordinary` instance followed by all records at `SAIFILE` scope;
2. an empty `impostor` instance followed by all records at `SAIFILE` scope.

Thus the claimed nonempty DUT instance and DUT-only critical activity are not
proved.  A header/foreign-record artifact can be promoted to a measurement
SAIF despite the fail-closed contract.

## Required additive repair

Do not run M2162 and do not edit or retry M2151.  Use a fresh source identity:

1. Normalize warning/error lines and reject reset/clear plus
   power/switching/SAIF rejection semantics, including ignored, rejected,
   not-cleared, not-reset, and equivalent forms.  Add the four bypasses above
   as mutations while preserving the duration gate.
2. Parse balanced SAIF hierarchy.  Require the intended reported root instance,
   count all 93,971 records and critical cones inside that subtree, and reject
   activity records outside it.  Add empty-target, wrong-target, and
   out-of-instance forgeries as mutations.
3. Preserve every existing duration, TX=0, conservation, critical-toggle,
   ledger, arithmetic-scoreboard, knownness, topology, double-seal, and
   one-shot gate.  The prehistory SAIF remains diagnostic-only.

Only a fresh independent source hammer with P0/P1/P2 = 0/0/0 and score at least
95 may authorize one fresh query, compile, simv, and two-SAIF acquisition.

## Evidence boundary

This is a source-rejection result.  It provides no VCS/SAIF admission, mapped
activity, logic or SRAM power, energy, speedup, system result, or paper-citable
PPA.  M2160's better UCLI sequence is not enough to cross the acquisition gate.
