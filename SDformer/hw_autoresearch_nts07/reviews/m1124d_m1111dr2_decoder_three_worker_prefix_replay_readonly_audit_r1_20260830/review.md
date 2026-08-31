# M1124D — M1111Dr2 decoder three-worker prefix-replay read-only audit

## Verdict

**Directly rebasing the existing compressed per-call summaries is unsafe and is
STOPPED.** A deterministic parallel successor is safe only as a two-pass
design: pass A discovers each call's transaction count and duration; a strict
single-coordinator prefix scan assigns global transaction/cycle seeds; pass B
reruns every call from those seeds and regenerates all three SHA-256 digests.
No runner or production was launched by this audit.

The currently live M1111Dr2 process (PID 4122290 when observed) and its attempt,
lock, work and result namespaces were only read. They were not signalled,
attached, waited, modified, renamed, deleted or reused.

## Why call execution can be isolated

`execute_call` begins at `scheduler.end_cycle`. At the preceding call boundary,
that value is one cycle past the greatest return. For every frozen port,
`initiation_interval=1`; consequently every `next_port` value is no later than
its transaction return, and every outstanding return is strictly earlier than
the next call start. Prior-call scheduler state therefore cannot delay the next
call. Addresses and dependency-token names already include the frozen global
call ordinal. A fresh scheduler seeded with the exact transaction and cycle
prefix is thus equivalent to the serial scheduler for that call.

## Why summary-only rebase is impossible

The current `CallAudit` hashes fields that the rebase changes:

- address digest: global transaction ordinal, kind, bank, address and width;
- dependency digest: global transaction ordinal, dependencies and producer;
- schedule digest: global transaction ordinal, absolute issue/return/commit
  cycles, stall reason and identity.

Only the terminal SHA-256 hex digests and aggregate endpoint/count summaries are
retained. SHA-256 is not composable or invertible from those values. Editing the
ordinal/cycle fields in a summary would therefore create a row whose digests do
not describe that row. A full transaction journal could be rebased and rehashed,
but the observed first four calls already contain 158,549,790 transactions; the
storage/I/O cost is not an acceptable successor design.

## Admitted successor shape

At most three fixed workers may run. Calls are assigned by
`global_call_ordinal % worker_count`, never by worker completion order. Each
worker has a distinct unpredictable temporary directory and cannot write any
canonical namespace.

Pass A runs every call with an empty zero-seeded scheduler and emits only
discovery facts. The coordinator requires exactly calls 0–119, scans counts and
durations in that order, and calculates each call's transaction and cycle
prefix. Pass B reruns each call with a fresh scheduler seeded by those two
prefixes. It must not reuse or rewrite pass-A digests. The coordinator then
checks zero normalized semantic mismatches, orders the 120 canonical rows,
recomputes the stream digest, runs the original strict publication validator,
and performs one atomic no-replace publish. Any missing/duplicate/extra row,
worker failure, malformed JSON, mismatch or digest failure forbids publication
and quarantines the whole coordinator attempt. There is no automatic retry.

## Read-only serial oracle

The four completed live rows were read as a bounded oracle, not promoted to a
publication authority. Their row SHA-256 values are:

1. `67e55e0558ea896abc07b90267e23dbda18403c4e8950c37f7713af5a92698ca`
2. `f13848cd070a7a2c171dc92ef75b96fc5fa977ad52613d165c3d7c640ee381e4`
3. `8936c6f90ec693bd90a4b3a1dcf7a0fe8188ff2fa969312e5376ee3f9ca61e34`
4. `a80f214dfcd39ff7e96c24d1f7474be3b7bb35c7d6716fad725e43f4e15642ce`

Their concatenated four-row prefix is
`38e7d6e47196c242b04da93c87947fa7a4379360a12347b38b3f4868344fb5eb`.
Transaction intervals and cycle intervals are strictly contiguous with zero
mismatch. A future implementation must reproduce these four rows byte for byte
after pass-B prefix seeding. If the serial M1111Dr2 result publishes, the full
120-row stream digest must also be identical; otherwise no same-digest claim is
admitted before independent hammering.

## Boundary

This is a source-design audit and contract only. It authorizes neither a runner
nor workers nor production, reports no new transactions/cycles/traffic, and
admits no performance, system-speedup, energy or PPA claim. `docs/359` remains
at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
