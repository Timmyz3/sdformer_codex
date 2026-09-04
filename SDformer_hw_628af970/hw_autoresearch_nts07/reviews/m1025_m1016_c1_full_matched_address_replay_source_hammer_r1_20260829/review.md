# M1025 — independent hammer of M1016 full matched-address replay source

**Verdict: GO to author one exact execution release/runner chain only. Do not run the 51.84M-row replay yet.** Score 98/100; P0/P1/P2 = 0/1/0.

M1010 authority and the M1016 receipt verify against their manifests and outer seals. The receipt transitively pins the engine, generic one-shot runner, checker, ten tests, contract, frozen 466,560,000-byte M410 ledger, and `docs/359`. The frozen 10/10 tests, engine self-test, source checker, JSON parsing, and runner `bash -n` all pass.

Sixteen independent negative cases fail closed. Empty, tiny, truncated, duplicate-tile, missing-phase, changed-ledger/reordered-phase, missing-block, missing-design, service-count mismatch, service-digest/phase-order mismatch, parent-conservation mismatch, and unfinished-service-merge states cannot derive `raw_full_replay_complete`. A coverage CLI is rejected; an environment variable is ignored; an unknown JSON coverage field has no path into `DerivedCoverage`, while duplicate JSON keys are rejected. A truncated M410 replacement is rejected at size preflight before output creation.

The production implementation is memory-bounded at source level: it uses `os.pread` with at most 64 × 9 = 576 bytes per raw tile, keeps only bounded coverage arrays and per-tile parent state, and exposes parent events as a generator. A 64-row all-active stress tile produced 110 address events with an observed Python peak below 0.1 MB. This is evidence about implementation memory growth, not a physical SRAM allocation.

The common-service plan remains an M528 aggregate-anchored same-coordinate cycle-model construction. It is not a measured SRAM/DRAM trace and carries no physical-memory energy. The 214,912-byte value is a capacity-only hypothesis. No matched cycles or speedup are admitted before a complete result and independent result hammer.

P1 remains on launch authority: the generic source runner accepts caller-selected release/hammer paths and caller-supplied expected hashes. The execution successor must use a new additive runner with one hardcoded release and hammer namespace, then receive an independent chain hammer. The generic runner must not be executed directly.

This review authorizes only writing that exact execution release/runner chain. It authorizes no full replay, EDA, capacity admission, cycle admission, speedup, or paper-ready claim.
