# M1127C independent static hammer — M1126C C1 three-axis exporter

Status: `PASS_M1127C_M1126C_GAP_LOCALIZATION_AND_SYNTHETIC_SOURCE__CANONICAL_STOP`

## Verdict

**GO only for gap localization and the bounded synthetic source; STOP for canonical export.** The only next authorization is an additive weight-service addressed-ledger source. No canonical row open, 51.84M replay, runner, RTL, EDA, GPU, remote job, traffic, cycle, energy or performance claim is authorized.

## Canonical path

Independent AST/control-flow inspection and the bounded source self-test agree:

- `CanonicalRowReader` is never constructed by M1126C;
- `iter_canonical_transactions()` calls the exportability audit and fails its `canonical_export_ready is True` gate before its unreachable yield;
- canonical rows read = `0`;
- transaction rows emitted = `0`.

The available common receipt fields include `counts` and global `weight_beat_first`, but they do not state native READ/WRITE, local 24-slice address, logical bytes/byte-enable, native macro activation multiplicity, or an exact-once mapping from service beat to the on-chip store. Converting a count or capacity interval into those fields would fabricate transactions.

## Exact reconstructability boundary

- Candidate parent address events are reconstructable.
- Baseline parent evidence is only a sealed zero-parent aggregate; it is not an addressed stream.
- Psum logical bank/address/op/base-ready events and their arbitrated 1RW grant cycle/group/address/op are reconstructable.
- Source task and row-provenance identities are reconstructable.
- Weight native addressed transactions are not reconstructable.
- The residual `24,448 B` is an identical conservative capacity denominator only; residual accesses are prohibited.

Parent/psum partial rows therefore may support repair development, but cannot be labeled a complete three-axis transaction trace while weight remains absent.

## Synthetic oracle and attacks

The sealed oracle, bounded runtime output, and an independent scheduler all agree on `5` unique transactions, `2` explicit stalls, `0` final 1RW conflicts and `0` weight half-slot overlaps. Eleven attacks were rejected, including duplicate exact-once identity, residual access, same-cycle half-slot overlap, count/`weight_beat_first` fabrication, canonical-ready escalation, nonzero real rows/transactions, and an injected row reader.

The synthetic result validates schema and arbitration mechanics only. It is not an H67 traffic or cycle result.

## Required additive repair

For each weight service beat, the next source must freeze: source task/local ordinal, global beat ordinal, native READ/WRITE, half-slot plus local address, bytes plus byte-enable, native 128-bit macro activation count, exact-once on-chip-store relation, and source-row provenance SHA256. A later different-author hammer must seal that ledger before any successor exporter can open the full population.

The M1126C source remained SHA256 `d54640b0bb85e7ba2e4222655a4325b23310aab8eb75b88c13ed00ad5ef12e27`. `docs/359` remained `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
