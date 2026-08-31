# M37-r10 area-recovery architecture milestone

This milestone is a source-structure candidate, not a Synopsys result.  It was
created in `rtl_m37_r10/` so the frozen r8 snapshot, live r9 RTL, and the
in-progress r9 DC run remain untouched.

## Why r9 grows

R8 met 3 ns at 63,671.579642 um2 but its reference read produced five
`FMR_ELAB-147` messages: two dynamic reads from the ten-entry unpacked bias
array and three dynamic reads from the thirty-entry unpacked descriptor arrays.
R9 replaced those accesses with bounded phase and coefficient equality scans.
That is source-static, but it exposes the synthesizer to a replicated
phase-by-thirty selection network at each arithmetic consumer.  The incomplete
r9 run's roughly 180k-um2 transient is diagnostic only and is not an admitted
area result.

## R10 topology

R10 stores each two-row phase as one packed 168-bit bundle:

```
6 coefficients * (4 valid + 4 negative + 12 shift) = 120 bits
2 biases * 24 bits                                  =  48 bits
one phase bundle                                    = 168 bits
five phase bundles                                  = 840 bits
```

One explicit five-arm `case` selects a constant 168-bit packed slice.  The 96
CSD consumers use only generated constant slices from that shared bundle.
There is no selected-coefficient variable, coefficient equality scan, padding,
or stored unpacked descriptor/bias array.

The two input banks are packed 384-bit vectors.  One explicit 2:1 mux selects
the active payload and tag, after which rank/lane consumers also use constant
slices.  The selected 48-bit bias pair is pipelined with the 96 products.  The
result stage therefore has no dynamic row read.

## Static resource boundary

The source-visible state equation is:

```
phase table + threshold                         864
config/protocol flags                            2
two input payloads + tags                      864
bank/compute/phase control                       7
product data + bias pair + tag/beat/valid     1828
FIFO state                                    2365
done state                                      49
total                                         5979 bits
```

The same equation for r8 is 5,931 bits.  R10 adds only the 48-bit bias-pair
pipeline, or 0.8093%.  Its explicitly shared top-level selection boundary is at
most 672 2:1 mux-bit equivalents for the 168-bit 5:1 phase selector plus 432
for the 384-bit payload and 48-bit tag bank selector: 1,104 total.  This count
does not include CSD term shift cases or prove mapped cell area.

The physical admission gate is no more than 1.10x the fresh r8 cell area under
the identical logic-only boundary: 70,038.737606 um2.  Static equations only
make that target plausible; they do not establish it.

## Required closure

The next gate is an independent static hammer.  If admitted, the exact RTL SHA
must then pass the frozen 245-tile VCS/SVA workload with every r9 metric equal.
Only after that may a fresh 3 ns DC run test zero multiplier resources, clean
setup/hold, and the area cap.  Strict Formality must link with zero
`FMR_ELAB-147`, use no mismatch filter, and report `SUCCEEDED` with zero failing
and unmatched points.  A second independent hammer is required after those
Synopsys receipts.
