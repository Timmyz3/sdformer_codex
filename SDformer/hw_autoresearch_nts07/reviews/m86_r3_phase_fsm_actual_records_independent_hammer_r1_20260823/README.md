# M86-R3 phase-FSM actual-record independent hammer

Verdict: **86/100, P0=0, P1=2, P2=5**. M86-R3 is a scoped GO for the explicit five-state phase protocol and for wrapper transparency relative to the unchanged R1 datapath. It is not a speedup, PPA, system, or DATE-headline milestone.

## What was independently rerun

- Recompiled both production VCS benches from the exact-SHA source set with the production runner. Directed and all-1728-phase actual-record runs pass.
- Actual replay accepts 221,184 descriptors, returns 221,184 outputs, issues 835,383 bank beats, covers 5,215 backpressure and 4,900 FIFO-full cycles, and reports zero R1 lockstep mismatches.
- Added a separate boundary/fault/reset VCS bench. It attacks all-three-valid contention in LOAD/COMMIT/EXECUTE, row 459/460/461, descriptor 127/128/129, early/late commit attempts, six-cycle DRAIN backpressure, a held next loader, duplicate/OOB/metadata faults, and reset from five lifecycle classes.
- The independent bench also proves an important interface gap: 128 repetitions of the same `(pattern=0, block=0)` descriptor are accepted and retire the phase.

The machine-readable evidence and exact SHA pins are in `m86_r3_machine_audit.json`. The primary assessment is in `m86_r3_phase_fsm_actual_records_independent_hammer_review.json`.

## Key interpretation

R3 closes the R2 three-channel silent-deadlock problem for legal ordered phase traffic. It enforces 460 unique payload rows, one commit, 128 accepted descriptors, finite DRAIN handoff after downstream release, sticky FAULT, and destructive reset without ghost output.

The 1728-phase comparison is a strong wrapper-transparency regression, not an independent datapath oracle: both DUT and reference instantiate the same R1 RTL. This review pins the earlier independent R1 binary/address/signed reconstruction oracle, but does not relabel the R3 lockstep test as independent.

The two P1 gaps are:

1. Descriptor identities/order are not protected; only the count of 128 is enforced.
2. LOAD, COMMIT, EXECUTE, and DRAIN are serialized. There is no double buffering or loader/execute overlap, so this milestone cannot re-admit module or system speedup.

## Reproduction

Run the production exact-source replay into a new directory:

```bash
bash hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m86_r3_phase_fsm_actual_records_sva.sh \
  --records /tmp/m85_inputs/m83_cap11_phase_records.bin \
  --offsets /tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin \
  --run-dir <new-run-directory>
```

Run the independent boundary attack:

```bash
bash hw_autoresearch_nts07/reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/run_independent_boundary_vcs.sh \
  <new-run-directory>
```

Rebuild the machine audit:

```bash
python3 hw_autoresearch_nts07/reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/validate_m86_r3_machine_audit.py \
  --hw-root hw_autoresearch_nts07 \
  --exact-rerun hw_autoresearch_nts07/reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/exact_runner_recompile_rerun \
  --boundary-run hw_autoresearch_nts07/reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/independent_boundary_vcs \
  --records /tmp/m85_inputs/m83_cap11_phase_records.bin \
  --offsets /tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin \
  --output hw_autoresearch_nts07/reviews/m86_r3_phase_fsm_actual_records_independent_hammer_r1_20260823/m86_r3_machine_audit.json
```

The directories named `attempt*_failed` preserve early testbench-debug attempts and are not evidence. Only `exact_runner_recompile_rerun` and `independent_boundary_vcs` are cited as passing runs.
