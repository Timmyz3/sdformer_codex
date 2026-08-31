# M397 H67 fixed-product q/O finite DSE

This result executes the M396-repaired four-point experiment on all 17,280
phases reconstructed from the frozen H67 ep35/no-running M40 S10 payload.
Every point uses the first `q` entries of the same nested train-only catalog,
strict exact fallback, one SHARED96 source port, a SERIAL16 matcher, one
II1/L8/D8 descriptor stream, one 32-byte/cycle DMA server and two 32-KiB
tile slots.  It does not use WIDE144, SYSTOLIC_Q or rejected PAFT runtime
counts.

At the frozen cmd32/L8 decision point:

| q | output tile O | candidate cycles | speedup vs common bit-sparse |
|---:|---:|---:|---:|
| 16 | 8 | 676,931,968 | 1.0963411703x |
| 32 | 4 | 669,012,336 | 1.1093194341x |
| 64 | 2 | 677,482,456 | 1.0954503389x |
| 128 | 1 | 730,375,445 | 1.0161190263x |

The common baseline is 742,148,386 cycles for all four rows.  q32/O4 exactly
reproduces M394.  q128/O1 charges a 160-byte physical stride for each
144-byte useful PWP record, four bitmap-seal cycles, seven additional
SERIAL16 passes per eligible row and eight output-tile replays.

The selected point is therefore the existing q32/O4 anchor.  Its 1.109319x
ratio does not reach the contract's pre-frozen 1.15x selected-RTL gate, so
fixed-product q/O scaling is `NO_GO` as a new performance axis.  This result
is a standalone four-bottleneck-Conv trace-cycle estimate, not an RTL,
full-network, energy, physical-SRAM, PPA or DATE-headline result.

The next exact optimization hypothesis is hierarchical early termination of
the nested matcher when a prefix already finds Hamming distance zero.  That
targets the measured large-q matcher cost without changing reconstruction or
checkpoint identity; it is not admitted by M397.
