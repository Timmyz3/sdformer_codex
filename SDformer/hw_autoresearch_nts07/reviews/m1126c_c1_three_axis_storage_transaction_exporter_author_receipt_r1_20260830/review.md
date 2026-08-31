# M1126C — C1 three-axis storage transaction exporter author receipt

## Outcome

The source, contract and bounded synthetic oracle are complete. The canonical
exporter is intentionally **fail-closed STOP** before the frozen 51.84M-row file
is opened. No full transaction row was emitted.

The source binds M1102, M1000, M1123C and M1125C by exact identity. It defines
the required transaction schema, deterministic per-bank 1RW serialization,
explicit stall accounting, exact-once source identities, weight-half-slot
mutual exclusion and a hard prohibition on residual accesses.

## Precise frozen-source gap

The frozen chain contains enough information to reconstruct:

- candidate parent address events and baseline zero-parent aggregates;
- psum logical bank/address/op requests and their exact arbitrated 1RW grant
  cycles;
- task and source-row SHA provenance.

It does not contain enough information to turn the weight service into exact
on-chip SRAM transactions. The common receipt supplies only a service count and
global `weight_beat_first`; the packing audit supplies an interval and
half-slot. Neither identifies READ versus WRITE, local 24-slice address,
logical byte/byte-enable mapping, native activation multiplicity, or an
exact-one mapping from DRAM/service beat to the on-chip weight store.

Inferring those fields from the 24-macro capacity geometry would synthesize
evidence, so the canonical iterator rejects the export before constructing a
`CanonicalRowReader`.

## Bounded oracle

Five synthetic transactions exercise parent, psum and both weight half slots.
Two nominal same-cycle conflicts become explicit one-cycle stalls. Final 1RW
conflicts and half-slot overlaps are zero. Duplicate source identity and a
fabricated residual access are rejected. The canonical iterator independently
raises the frozen weight-provenance STOP.

This oracle validates only source mechanics. It is not H67 transaction,
traffic, cycle or energy evidence.

## Required repair

The minimum additive repair is a frozen weight-service ledger produced from the
same M1102 task iterator. Each service beat must state its on-chip operation,
half-slot/local address, bytes and byte enable, native 128-bit slice activation
count, source-row provenance and exact-one relation to the global beat ordinal.
Only after a different-author hammer seals that ledger may a successor enable
the full exporter.

Only a different-author static hammer is authorized next. No runner, canonical
row open, 51.84M-row replay, EDA, RTL, GPU, remote job, dynamic energy, cycle or
performance claim is authorized. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
