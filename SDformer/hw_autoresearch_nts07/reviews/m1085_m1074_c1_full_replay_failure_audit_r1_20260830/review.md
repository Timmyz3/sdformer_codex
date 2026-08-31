# M1085 independent failure audit of M1074

Verdict: **M1074 consumed its sole attempt and failed closed; do not retry it. A new-namespace additive zero-work repair is allowed.**

## First concrete failure

The minimum canonical prefix stops at task 207, coordinate `[0,0,0,207]`, design `candidate`. The exact 64-row record is schema-valid and rederives `work_cycles=0`; its row-byte SHA is `e8636aaf...`.

M1056 nevertheless emits all sixteen bank read/write pairs. With `span=1`, bank 1 has `read_cycle=start+1`, while `write_cycle=min(work_end, read_cycle)` becomes `start`. Group 0 event `t207:b1:W` therefore depends on nonempty event `t207:b1:R` with exact-integer `delay_cycles=-1`. The first failing field is `Dependency.delay_cycles`; it fails the frozen `delay_cycles >= 0` predicate.

This is an **M1056 event-geometry source bug exposed by a valid M1072 zero-work record**. It is not an illegal canonical trace, M1072 provenance corruption, empty dependency id, boolean/type loophole, or numeric candidate mismatch. Tasks 0–206 and all dependencies before this event pass the same bounded derivation.

## Frozen runtime state

- Attempt receipt and quarantine independently reverify their M1074 atomic manifests and outer seals.
- The attempt status is `CONSUMED_BEFORE_CANONICAL_ROWS_OPEN`; `maximum_attempts=1`, `automatic_retry=false`.
- Published M1074 result and original work directory are absent; the unique sealed quarantine records return code 1 and `FAILED_OR_INTERRUPTED__NO_RETRY`.
- Exactly two M1074 runtime namespaces exist: the consumed attempt and the quarantine. M1074 may not be retried.

## Minimum additive repair gate

Keep M1056/M1072/M1074 and their evidence frozen. A successor must use a fresh source/result/attempt/lock namespace and define zero-work service explicitly: zero psum port events and grants, no `last_write` mutation, zero port-delay/excess counters, and deterministic stream timing. It must not merely clamp a negative delay while retaining causally impossible reads/writes.

Before any new one-shot attempt, a different-author source hammer must:

1. replay the exact task-207 row and prove candidate zero-work produces no psum events while both baselines remain unchanged;
2. prove the following task's start/effective-end and previous-address state are deterministic;
3. exhaustively preflight all 812160 canonical task/design work values for zero/short-work geometry without running the cycle replay, and fail source admission if any unsupported `1..14` case remains;
4. retain the frozen positive-work 1RW RAW/cascade anchors and rejection of empty ids, booleans and negative delays;
5. rebind full row provenance, services, parent conservation, capacity and atomic one-attempt policy;
6. authorize one new CPU attempt only after the hammer passes.

No cycles, speedup, full-trace feasibility, RTL, PPA, energy, Table-A, or paper claim is admitted by this audit. `docs/359` remains `dedde7ce...`.
