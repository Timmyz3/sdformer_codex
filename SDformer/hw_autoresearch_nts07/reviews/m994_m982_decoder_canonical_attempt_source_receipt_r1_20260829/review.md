# M994 — canonical-attempt consumption source repair

## Verdict

`PASS_M994_CANONICAL_ATTEMPT_SOURCE__NO_REAL_10K`, 100/100, P0=0, P1=0.

M994 is an additive successor to frozen M981 and answers the sole sealed M982
P0. The atomic `mkdir` of the canonical M998 attempt directory is now the
irreversible consumption point. Only after that visible directory exists and
its parent is fsynced may the attempt receipt and atomic two-file seal be
written. There is no random attempt stage and no stage-to-canonical rename.

## Directed interruption evidence

Seven tests passed. Faults injected after canonical mkdir, after receipt, and
after seal all leave the canonical namespace present; every second consumption
is rejected. Empty or unsealed canonical attempts are preserved rather than
repaired, deleted, renamed, or quarantined. The future runner creates work only
after successful canonical attempt sealing, and cleanup never touches the
canonical attempt.

The future chain is M994 → M995 source hammer → M996 one-attempt release →
M997 release hammer → sole M998 D2-then-D3 10K execution. M996 is additionally
required to freeze exact D2/D3 order and reject retry, 100K, full-row,
production, EDA/GPU, and remote expansion.

No 10K prefix, EDA, GPU, or remote job ran. `docs/359` remains at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
The frozen M979 future M993 runner is separately prohibited because that
milestone collides with C1 M993; C2 must be rekeyed at M1001 or later.
