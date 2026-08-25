# M80 r1 addressable phase-record PWP independent hammer

## Verdict

**GO for the arithmetic/storage/traffic envelope.**

**NO-GO for calling r1 a uniquely implementable byte-addressable format, and
NO-GO for the proposed four dedicated width-class staging banks.**

The reported `-72.6128%` double-buffer reduction is real under the stated bank
architecture.  It is not an arithmetic bug: it is the cost of sizing four
non-shareable banks to four mutually different worst phases.  A unified
32-bit-word-interleaved buffer changes the same logical payload from a 72.6%
capacity regression to an 18.75% saving versus fixed12.

## Independent reconstruction

The oracle did not import M80 or M78.  Starting from the pinned M72 centers and
four M41 INT8 weight files, it rebuilt all `221,184` output-block entries and
obtained:

| width | entries | dedicated-bank peak | peak phase `(op, partition)` |
|---:|---:|---:|---:|
| 8 | 52,248 | 92 | `(1, 374)` |
| 9 | 128,893 | 120 | `(0, 233)` |
| 10 | 37,144 | 58 | `(3, 53)` |
| 11 | 2,898 | 22 | `(2, 342)` |
| 12 escape | 1 | — | — |

The four independent peaks come from four different phases and sum to `292`
entries even though one phase has only `128`.  Their dedicated-bank capacities
are `8,832 + 12,960 + 6,960 + 2,904 = 31,656 B` per buffer.  Adding the 160 B
local descriptor to each ping/pong side yields `63,632 B`, versus `36,864 B`
for fixed12:

`1 - 63,632 / 36,864 = -72.612847%`.

All other submitted arithmetic reproduced exactly:

- elastic payload: `23,776,068 B`;
- headers: `82,944 B`;
- phase padding: `24,988 B`;
- 1,729-entry offset table: `6,916 B`;
- addressable catalog: `23,890,916 B`, `24.9904%` below fixed12;
- five-sample weight+PWP traffic saving: `14.9943%`;
- phase-row SHA: exact match;
- minimum record DMA envelope: `401` cycles versus `128` parser cycles;
- production replay: byte-identical to result SHA
  `dec76e2a...72a7da`.

## P0

### P0-1: r1 is not a canonical byte serialization

The JSON defines field widths, entry order and record sizes, but not:

- bit order for the 128 packed 3-bit header codes;
- byte/bit order and signed two's-complement serialization for each 9/10/11-bit
  lane;
- byte endianness of the 32-bit phase offsets;
- exact bit packing of the 10-bit local descriptors.

LSB-first and MSB-first header encoders produced different bytes for all 1,728
phases while satisfying every r1 size equation.  Likewise, little- and
big-endian offset tables have distinct SHA256 values.  No catalog binary,
offset binary, golden parser vector, or pack/unpack round trip is emitted.

Therefore `byte_addressable_phase_record_format=true` is stronger than the
evidence.  r1 is an address-capacity DSE, not yet a uniquely implementable
format.

## P1

### P1-1: reject four dedicated width banks

The 72.6% regression is intrinsic to this particular peak-per-class capacity
rule.  It overwhelms the catalog-storage benefit and gives an unnecessarily
poor SRAM starting point.  Preserve it as a rejected DSE, not as the selected
microarchitecture.

### P1-2: parser hiding is necessary arithmetic, not a finite-cycle proof

`128 <= 401` proves only that header decoding is nominally faster than the
smallest phase DMA.  It does not prove the 48-byte-header/32-byte-beat split,
payload beat steering, partial first/last beats, FIFO depth, backpressure,
ping/pong handoff, or bank conflicts.  The original result appropriately marks
finite-queue RTL false; this must remain outside speedup claims.

### P1-3: evidence scope remains valid825-internal and pre-RTL

The result admits neither SRAM macro feasibility/PPA nor full-network speedup
or accuracy.  The address format must ultimately be rebuilt from a train-only
catalog before PAFT/training or paper use.

## Alternative comparison

| design | double buffer | vs fixed12 | catalog storage saving | five-sample traffic saving |
|---|---:|---:|---:|---:|
| r1 four class banks | 63,632 B | **-72.61%** | 24.99% | 14.99% |
| unified 32-bit-word interleaved | **29,952 B** | **+18.75%** | 24.99% | 14.99% |
| per-entry 32 B aligned records | 33,984 B | +7.8125% | 15.70% | 9.42% |

The unified alternative uses a `14,720 B` 32-byte-aligned payload buffer plus
128 x 16-bit descriptors (`256 B`) per ping/pong side.  A concrete descriptor
that fits every observed phase is:

```text
[15] escape | [14:13] width_class | [12] reserved | [11:0] payload_word_offset
```

The maximum observed 32-bit-word offset is `3,648`, below the 12-bit limit
`4,095`.  Contiguous entries require `3/4/4/5` 32-byte service beats for widths
8/9/10/11, matching M80's service envelope.  This alternative cuts capacity
52.93% versus the r1 four-bank design while retaining the packed catalog and
traffic results.

The per-entry-32B alternative rounds entries to `96/128/128/160 B`, adds
`2,956,156 B` of global payload padding, and reduces the five-sample traffic
saving to 9.42%.  It is a simpler-control fallback, not the preferred point.

## Next minimum repair

Create M80 r2 around the unified word-interleaved buffer:

1. Freeze exact header, descriptor, signed-lane and offset byte/bit endian
   rules; reserve invalid codes and require zero padding.
2. Emit `catalog.bin` and `phase_offsets.bin` with SHA manifests.
3. Add an independent parser that round-trips all phases and all four widths,
   including the sole escape, and checks every offset boundary/padding byte.
4. Replace the selected staging result with the 29,952 B unified-buffer
   envelope and retain four-bank/entry32 as rejected alternatives.
5. Then implement a small VCS parser+8x32-bit-bank model with randomized DMA
   stalls; prove no overwrite, offset overflow, bank conflict or ping/pong
   hazard before claiming finite cycles.

## Scores

| dimension | score / 100 |
|---|---:|
| arithmetic and source identity | 98 |
| storage/traffic accounting | 97 |
| format completeness | 42 |
| selected staging architecture | 30 |
| alternative quality | 86 |
| performance evidence | 48 |
| innovation potential | 72 |
| DATE completeness contribution | 38 |
| overall M80 r1 milestone | **61** |
