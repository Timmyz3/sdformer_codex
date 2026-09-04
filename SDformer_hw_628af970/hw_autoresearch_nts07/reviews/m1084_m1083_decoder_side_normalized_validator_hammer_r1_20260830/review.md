# M1084 receipt-blind hammer of M1083

Verdict: **STOP. Do not launch M1085.**

## What passed

- The synthetic M1082 first-failure shape is accepted: D0 candidate/baseline are both 623 cycles while address and terminal hashes remain side-specific.
- Every one of the 13 documented semantic projection fields was independently mutated and rejected. This is not a cycles-only comparator.
- Candidate and baseline `total_cycles` are separately bound to their reported cycle values; bool/int aliases reject.
- Stale side-hash, stale binding, reset-semantic and paired-hash mutations reject.
- New decoder result/attempt/work/quarantine path shapes are isolated from old M1078 and from the sealed M1085 C1 review directory. The runner consumes the attempt before payload verification, has one consume call, and contains the failure-quarantine path.

## P0 counterexample

The side-specific provenance receipt is self-signed. `window_bindings()` derives `binding_sha256` from whatever caller-visible hashes are currently in the window; the frozen exact/reset validators check only schema and 64-hex shape for those fields. Therefore mutation followed by recomputation of `window_bindings()` is accepted.

Four independent re-signed attacks survive:

1. homogenize both transaction-address hashes;
2. arbitrarily replace candidate terminal-readiness hash;
3. homogenize all six candidate/baseline reset-side hashes;
4. arbitrarily replace `paired_reset_semantics_sha256`.

The author self-test only mutates values while leaving the old binding stale, so its rejection does not cover this attack.

## Minimum acceptable repair

M1083, its contract/release/runner, and this STOP must remain frozen. A successor needs a new source, contract, release, result, attempt and lock namespace.

Each address, terminal and reset provenance root must be derived from a trusted witness independent of the summary under validation. The minimum acceptable root is either:

- recomputation from the frozen lower-level paired-replay transaction/readiness/reset objects bound to the canonical window body and identity; or
- an independently sealed producer receipt whose exact per-side roots, canonical body/window identity and producer-source SHA are pinned before validation.

The validator must recompute `paired_reset_semantics_sha256` from the canonical reset semantic projection rather than accept the supplied hash as its own root. Candidate and baseline roots remain distinct and individually checked before any private shadow normalization.

A different-author successor hammer must re-run all four attacks after recomputing every caller-visible binding; each must reject. It must also preserve the 623/623 positive case, all 13 semantic mutations, per-side total binding, recursive bool/int rejection, namespace isolation, attempt-before-payload order and failure quarantine.

No M1085 attempt, real payload, pilot, GPU, EDA or remote work was used. No decoder result or performance claim is admitted.
