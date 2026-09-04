# M1074 full exact-1RW one-shot source receipt

## Verdict

`PASS_M1074_SOURCE_ONLY__M1075_REQUIRED_NO_LAUNCH`

M1074 packages the independently hammered M1072 iterator behind a one-shot,
CPU-only execution boundary. It does not execute the 51.84M-row replay. The
source, runner, checker, tests and contract are frozen for a different-author
M1075 hammer before any launch.

## Exact execution path

The future cycle path contains exactly one call to
`M1072.iter_canonical_full_replay_results()` and supplies zero arguments. No
caller can select rows, records, samples, work, preprocess, capacity or
coverage. The wrapper explicitly rejects the old `raw + 403922` shortcut: all
parent delays must have propagated through M1072's exact four-group 1RW
arbitration and sample cascade.

If eventually authorized, the raw payload contains ten exact sample task
boundaries and, for all three designs, per-sample plus aggregate cycles,
delayed accesses and nominal excess accesses. It also carries the three frozen
service ledgers/digests, candidate parent conservation, the internally derived
214,912-byte capacity ledger, and the execution-provenance digest binding all
51.84M row bytes to preprocess/work/parent fields.

## One-shot ordering

Before the canonical attempt can exist, the runner closes the source,
contract, M1072, M1073, interpreter and tool identities; requires the exact
future M1075 seal; rejects EDA process collisions; acquires the global C1
nonblocking lock; verifies 16 GiB commit headroom and available-memory floors;
and checks all canonical namespaces are absent. None of these operations opens
or hashes the M410 row file.

The atomic `mkdir` of the exact canonical attempt directory is the irreversible
consumption point. Only after its sealed receipt exists may M1072 advance and
open rows. Automatic retry is prohibited.

## Durable publication and failure

Attempt, result and failure evidence use recursive manifests inside a complete
`.m1074_atomic_seal` directory. Seal bundles and final result/quarantine roots
are published with Linux `renameat2(RENAME_NOREPLACE)`, so a late collision
cannot be overwritten. The canonical result appears only after the complete
work root verifies.

After attempt consumption, an error or signal closes an interrupted attempt,
moves any partial result and interrupted seal stages into a unique quarantine,
recursively seals it, and publishes it without replacement. If quarantine
sealing itself fails, the stage remains in place instead of being discarded.

## Evidence and claim boundary

Fifteen directed tests pass. They cover sample population/order, port-stall
schema, service/parent/provenance/capacity forgery, attempt collision,
no-replace publication, symlinks and failure quarantine. Source self-test and
checker do not advance the M1072 generator, open/hash canonical rows, create an
attempt, or create a result.

Consequently no full cycles, speedup, capacity admission, RTL cycles or PPA are
claimed. A different author must complete M1075 and bind all five source files
before the one-shot runner may be launched. Any eventual raw result still
requires a separate independent result hammer.

`docs/359` was not modified.
