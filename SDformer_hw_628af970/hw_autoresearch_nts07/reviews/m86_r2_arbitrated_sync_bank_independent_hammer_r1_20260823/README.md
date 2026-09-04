# M86-R2 independent hammer review

Verdict: **78/100; P0=0, P1=4, P2=3. Scoped GO only for the exact R1 payload-versus-descriptor repair.** It is not yet a complete phase arbiter and does not readmit any performance result.

Exact source, SVA, TB, filelist, contract, runner, and inherited R1 hashes match. The sealed `simv` independently reran with rc=0. A separate source-recompiled VCS bench exercised 1,433 onehot cycles, 25 issue/response pairs, eight completed outputs, busy transitions, bounded starvation, OOB/duplicate/metadata errors, and both original R1 trigger directions.

## What is fixed

- Before phase commit, simultaneous payload+descriptor grants payload.
- After phase commit, simultaneous payload+descriptor grants descriptor.
- Independent VCS reproduced both priorities and observed no dual selection or dual acceptance.
- A losing payload survived eight accepted descriptors and moved correctly once descriptor traffic stopped and the pipeline drained.

Therefore the exact M86-R1 silent deadlock trigger is closed.

## What still blocks admission

The third request channel is not arbitrated. With only row 0 loaded, the independent bench held row 1 payload, `phase_load_valid`, and an early descriptor for four cycles. Payload was selected by R2, but R1 suppressed its ready because phase-valid was forwarded; phase-ready stayed low because the image was incomplete. Ready, accept, busy, and protocol-error all remained zero. This is another silent deadlock at a natural load/commit boundary.

Fixed priority also has no fairness bound. A committed-phase payload can be denied forever by a continuous descriptor stream; the contract correctly excludes starvation freedom. The minimal robust design is an explicit `LOAD -> COMMIT -> EXECUTE -> FAULT` FSM, with phase-load included in arbitration and exactly 128 descriptor accepts before returning to load.

Error behavior is fail-closed but underspecified. A loaded OOB payload is masked while descriptor wins; after the winner stops, it may be forwarded while that descriptor is still busy, assert a sticky fault after the first issue, and leave busy high until reset. A FAULT state needs defined drain/kill/reset semantics.

The production SVA should also exempt `protocol_error/metadata_error` from progress assertions. As written, correct OOB or poisoned-metadata fail-closed behavior can intentionally violate the unconditional contention-ready properties.

Finally, R2 sealed evidence is one zero signed8 vector—three reads—not the 1,728-phase/221,184-descriptor actual replay already achieved at R1. That replay must be rerun through the wrapper.

## Admission boundary

- GO: exact two-channel R1 trigger repair and exclusive payload/descriptor selection.
- NO-GO: complete three-channel arbitration, starvation freedom, actual-record integration, M78/M88 speedup, system performance, compiled SRAM, PPA/energy, accuracy, DATE, or best-paper claims.
