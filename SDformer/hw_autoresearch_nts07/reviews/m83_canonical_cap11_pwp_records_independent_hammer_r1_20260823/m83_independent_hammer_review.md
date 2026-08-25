# M83 canonical cap11 PWP records independent hammer R1

## Verdict

**GO: M83 closes the M80 canonical-serialization P0 for the audited binary
artifacts.**

**NO-GO remains for an RTL reader, M81 bank integration, finite-cycle
throughput, SRAM/PPA/energy, accuracy, system speedup, and any DATE headline.**

No P0 was found in M83's admitted serialization scope.  The exporter identity,
local receipt and RUN_COMPLETE are internally consistent, and a read-only
remote audit independently decoded every stored vector from the actual binary
files rather than trusting the receipt.

## Immutable identities

| artifact | bytes | SHA256 |
|---|---:|---|
| exporter | — | `1e279f5b...d3e45` |
| local receipt | 2,568 | `46893b0d...4303` |
| local RUN_COMPLETE | 91 | `f7120407...9347` |
| remote phase records | 23,884,000 | `6de1521b...7190d` |
| remote offsets | 6,916 | `1cddfc80...bf30c` |

The remote files were inspected read-only at the authorized A800 workspace
`/root/private_data/work/sdformer_codex/SDformer`.

## Independent full decode

The independent checker did not import the M83 exporter or M78 analyzer.  It
rebuilt every expected PWP directly from the pinned M72 center masks and M41
INT8 weights, then decoded the actual records using a separately written bit
extractor.

Results:

- `1,728` phases and `1,729` offsets;
- offsets start at `0`, are strictly increasing and all 32-byte aligned;
- terminal offset is exactly `23,884,000`, equal to the records file size;
- phase lengths range from `12,832` to `14,784` bytes;
- `221,183` stored PWP vectors and `21,233,568` signed lanes compared;
- `55,296` header fields crossing a byte boundary checked;
- exact width histogram: 52,248 / 128,893 / 37,144 / 2,898 / 1 for
  signed8/9/10/11/escape;
- payload `23,776,068 B` and all `24,988` phase-padding bytes checked;
- zero header, payload, signed-lane, padding, offset, or identity mismatches.

The signed checks include both polarities for every stored width and exact
boundary values:

| width | observed minimum | observed maximum | negative lanes | positive lanes |
|---:|---:|---:|---:|---:|
| 8 | -128 | 127 | 2,494,217 | 2,474,504 |
| 9 | -256 | 255 | 6,186,576 | 6,095,770 |
| 10 | -512 | 511 | 1,806,839 | 1,745,705 |
| 11 | -964 | 970 | 140,673 | 136,836 |

This directly validates signed two's-complement masking, sign extension, lane
order and little-endian byte order rather than merely checking sizes.

## Unique escape

The sole escape is:

```text
operator=2, partition=378, pattern=5, output_block=5, header_entry=45
range=[-1089, 549]
```

`-1089` does not fit signed11 but fits signed12.  Header entry 45 starts at bit
135, so its three-bit code crosses bytes 16/17; the independent cross-byte
decoder recovered code 4.  It consumes no payload, leaves the payload cursor
unchanged, and the following entries decode exactly.  Thus the escape is
implementable at the serialization level.  Whether an RTL reader correctly
routes it to the bit-sparse weight fallback remains unproved.

## P0

None in the admitted M83 serialization scope.

M80's ambiguity is closed by explicit and verified rules:

- entry 0 at header bit 0, three-bit codes LSB-first;
- pattern-major then output-block-major entry order;
- lane-major signed two's-complement, lane 0 least-significant;
- little-endian payload bytes and uint32 offsets;
- zero padding once per phase to a 32-byte boundary.

## P1

### P1-1: binary evidence is remote-only and lacks one sealed handoff manifest

The local tree contains the receipt and RUN_COMPLETE but not the two binary
artifacts.  Their remote bytes and SHA values are verified, but durability
currently depends on that workspace.  Archive the binaries in a checksummed
handoff pack or content-addressed store and add one manifest binding exporter,
upstream inputs, binaries, receipt and RUN_COMPLETE.

### P1-2: malformed-input behavior is not yet a frozen reader contract

Codes 5--7 never occur, all audited offsets are valid and all padding is zero.
The canonical spec should additionally state that a reader must fail closed on
reserved codes, decreasing/out-of-range offsets, truncated payloads, nonzero
padding and offset/record-size disagreement.  Add negative golden vectors.

### P1-3: format correctness is not hardware-reader correctness

There is no VCS reader connected to M81's 8x32-bit bank, no randomized DMA
stall/backpressure test, no ping/pong overwrite proof, and no proof that escape
dispatch preserves the weight fallback.  M83's receipt correctly marks reader
and bank integration false.

### P1-4: the audited catalog remains valid825-internal

M83 serializes the frozen M78 development catalog.  It cannot be used for PAFT
training, accuracy selection or a paper headline.  The exact export/round-trip
must be repeated for the future train-only catalog identity.

## Next minimum gate

1. Seal and transfer records, offsets, receipt and RUN_COMPLETE in one SHA
   manifest.
2. Add positive golden records covering widths 8/9/10/11 and the cross-byte
   escape, plus malformed negative vectors.
3. Implement a VCS reader against the exact M83 binary SHA and M81 bank format;
   randomize 32-byte DMA stalls and phase boundaries, and compare every emitted
   descriptor/word/lane to the software oracle.
4. Only after reader+bank integration may finite throughput be reconsidered;
   PPA/system claims still require Synopsys and full-network replay.

## Scores

| dimension | score / 100 |
|---|---:|
| software/serialization correctness | 97 |
| independent evidence quality | 96 |
| milestone completeness | 94 |
| innovation | 68 |
| performance advantage evidence | 58 |
| DATE completeness contribution | 46 |
| overall M83 milestone | **81** |
