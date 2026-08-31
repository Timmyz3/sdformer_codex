# M271 independent hammer review — Conv M267 Hamming-tree PWP

This review covers
`results/m267_hamming_tree_pwp_materialization_r1_20260825`; it is deliberately
separate from the `m267_m266` review.

Verdict: **the exact reconstruction, MST, signed12 and modeled-byte claims are
sound; physically hidden generation and unchanged cycles remain conditional.**
Evidence quality scores `93/100`, hardware admission scores `60/100`, with no
P0, four P1 and three P2 findings.

## Independently reproduced

The independent audit imports no M267 analyzer. Starting from the frozen M77
patterns and all four M256 signed-INT8 weight payloads, it rebuilt every tree
and PWP:

| Population or result | Independent value |
|---|---:|
| Operators / partitions | 4 / 1,728 |
| Tree edges and 768-lane PWP vectors | 27,648 / 27,648 |
| PWP scalar values | 21,233,664 |
| Scalar mismatches | 0 |
| Direct/tree INT16LE digest | `99da66ef...063749d` |
| Endpoint and actual generator transient range | `[-1026, 960]` |
| Minimum / mean / maximum MST flips | 21 / 36.273148 / 50 |

Deterministic Prim cost matched an independently implemented Kruskal MST on all
1,728 partitions; repeated Prim transcripts had zero mismatches. All declared
flip histograms and operator totals match. Determinism is for the frozen pattern
node order and the explicit `(distance,parent,child)` tie-break.

Signed12 is valid for endpoints and every ascending-bit generator intermediate.
The exact transient range was also `[-1026,960]`; independently, any subset of
16 `[-127,127]` weights is universally bounded by `[-2032,2032]`.

## Descriptor and byte result

All 27,648 four-byte descriptors round-trip with zero mismatch. Five parent
bits, five child bits and 16 XOR-mask bits fit in 26 bits; the remaining six
bits are zero. Add/subtract direction is recovered from the already-counted
16-entry pattern table, and every descriptor parent is constructed before its
child.

The exact accounting is:

| Quantity | Bytes per catalog pass |
|---|---:|
| Frozen weights | 21,233,664 |
| Fixed12 PWP payload removed | 31,850,496 |
| Pattern table | 55,296 |
| Four-byte descriptors | 110,592 |
| Fixed-PWP total | 53,139,456 |
| Tree-materialized total | 21,399,552 |

Thus PWP payload elimination is 100%, and modeled total catalog-pass
traffic/storage reduction is exactly `59.729448491%`. This is not a measured
DRAM-transaction or energy number.

Four bytes are sufficient but conservative. Because XOR is derivable from the
patterns, parent+child fit in two bytes; that variant would model `59.833507%`
reduction and save one worst-case preparation cycle.

## The important cycle correction

The numerical envelope is correct:

- weight + patterns + descriptors load: 387 cycles;
- worst tree generation: `50 × 8 = 400` cycles;
- serial preparation: `787 < 960`, leaving 173 cycles inside the old DMA
  envelope;
- both M251 ports report all 17,280 phases compute-bound.

DMA and tree generation do **not** need to overlap each other—the 787 bound is
already serial. But the whole 787-cycle preparation must overlap the current
partition's compute. M251's abstract compute binding does not prove that a
96-lane generator, weight reads and PWP read/write traffic are independent of
the current compute resources.

At minimum the lifecycle needs 36,864 bytes of current+next PWP storage and
24,576 bytes of current+next weight storage, with legal simultaneous current
reads and next-bank generation accesses. A shared arithmetic lane, SRAM bank or
port can expose generation latency. Therefore `cycles unchanged` is admitted
only conditionally, pending a partition-level port-feasible simulation or RTL.

## Energy and remaining work

Generation adds 501,440 96-lane add/sub cycles per catalog pass, equal to
48,138,240 scalar lane updates; over the M251 ten-sample replay it would be
5,014,400 vector cycles if regenerated each sample. The net energy sign requires
DRAM, SRAM, generator and descriptor-decode energy together.

The next milestone should specify a dedicated or explicitly arbitrated
generator, two-bank weight/PWP SRAM ports, initial fill, operator/sample
boundaries and final drain. Exact reconstruction and modeled-byte reduction can
remain; physical hidden-cycle and energy claims must wait.

Clean replay and deep relocation are byte-identical. Mutated catalog and weight
payload SHAs both exit nonzero and emit no result. M271 ran no DC flow and left
`docs/359` at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
