# M995 independent hammer of M994 canonical-attempt source

## Verdict

`PASS_M995_M994_CANONICAL_ATTEMPT_SOURCE_HAMMER`

`GO_AUTHOR_M996_RELEASE_ONLY`

M994 repairs the single M982 P0. The atomic `mkdir` of the final canonical
M998 ATTEMPT directory is now the consumption point. It occurs before receipt
writing, sealing, work-directory creation, or either decoder row. The parent
directory is immediately fsynced. No random attempt stage exists, and cleanup
never deletes, repairs, renames, or quarantines the canonical attempt.

This GO authorizes only the root author's M996 release contract. It does not
authorize M998 execution. M997 must independently hammer the exact M996 release
before the single M998 run can be considered.

## Independent interruption attacks

Three temp-directory attacks passed:

| Fault point | Canonical ATTEMPT | Receipt | Seal | Second attempt | Result |
|---|---:|---:|---:|---:|---:|
| after canonical mkdir | present | absent | absent | blocked | absent |
| after receipt | present | present | absent | blocked | absent |
| after seal | present | present | present/valid | blocked | absent |

Every case also had zero random `.stage.*` attempt directory. The production
M998 ATTEMPT, result, work and failure namespaces were absent before and after
the attacks.

## Frozen execution and ordering

The milestone chain is exactly M994 source → M995 source hammer → M996 release
→ M997 release hammer → sole M998 execution. The only rows are:

1. D2, sample 0, A1_OSG, timestep 0, prefix 10,000;
2. D3, sample 0, A1_OSG, timestep 0, prefix 10,000.

The runner is sequential under `set -e`. Frozen M981 `run_row` returns only
after writing the completed row and atomically sealing its row directory.
Consequently D2 is sealed before the D3 subprocess starts; a D2 failure stops
the loop and quarantines work without running D3.

M946 and M896 match their frozen SHA identities. The M994 driver also pins the
frozen M981 driver, so M994 changes the attempt boundary rather than the row,
payload, miter, seal, or quarantine semantics.

## Prohibited expansion

The future M996 validator requires all authorization bits for retry, 100K,
full-row, production, and EDA/GPU/remote execution to remain false. The future
runner is inert without exact M995, M996 and M997 SHA identities. The frozen
M979/M993 execution remains prohibited because that namespace collides with C1;
it must be rekeyed to M1001 or later.

No D2/D3 10K row, 100K row, full row, EDA, GPU or remote command ran during
M995. This evidence is not paper-citable and creates no Table-A row or speedup.

## Checks

- author unit tests: 7/7 PASS;
- author static checker: PASS;
- author source self-test: PASS;
- independent fault attacks: 3/3 PASS;
- source receipt and M982 STOP seals: PASS;
- bash syntax: PASS;
- `docs/359`: unchanged at `dedde7ce...`.

One P2 boundary remains: the tests model cooperating process interruption, not
malicious deletion of ATTEMPT or physical-media loss before durability. Under
the stated workspace-integrity threat model this does not reopen retry.
