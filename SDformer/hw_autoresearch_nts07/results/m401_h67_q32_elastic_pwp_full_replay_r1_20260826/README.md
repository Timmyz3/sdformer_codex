# M401 H67 q32/O4 exact elastic PWP full replay

M401 replays all 17,280 frozen H67 ep35/no-running phases and compares four
q32/O4 variants under the same cmd32, L8/D8 descriptor SRAM, SHARED96 port,
two-slot DMA recurrence and 742,148,386-cycle bit-sparse baseline.

| variant | candidate cycles | speedup vs bit-sparse |
|---|---:|---:|
| M397 fixed-signed12 anchor | 669,012,336 | 1.1093194341x |
| q16-prefix exact early-hit only | 665,260,728 | 1.1155752245x |
| exact elastic PWP only | 645,542,312 | 1.1496510333x |
| combined | 641,790,704 | 1.1563713550x |

The exact codec stores each 96-lane signed12 block as 96 low8 bytes plus a
64-byte aligned high4 sidecar.  A static block-local flag permits one
SHARED96 issue beat only when all 96 values are exact signed8 sign
extensions; otherwise the second sidecar beat is consumed.  The q32/O4
center stride is therefore a fixed 640 bytes.  A 64-byte pattern table plus
32-byte narrow bitmap forms one 96-byte config record.  The worst slot is
26,720 bytes and fits the frozen 32-KiB slot.

All 442,368 static blocks and 42,467,328 lanes reconstruct exactly.  Runtime
replay observes 24,586,812 narrow block descriptors out of 135,770,856
(18.1090%) and 549,754 per-phase used-center occurrences.  The longer stride
is charged by used center per tile, never by descriptor frequency.  The
combined point has zero exposed tile1-DMA cycles in the frozen trace.

The combined point clears the pre-frozen 1.15 gate, but the margin is
sensitive to descriptor blocking: 0.25 added cycle per replayed descriptor
reduces the modeled ratio to 1.132284x.  M401 therefore authorizes only an
independent full-replay hammer and, if accepted, q32-specific standalone
RTL/VCS/Synopsys validation.  It is not an RTL-measured, energy, physical
SRAM, system, paper-PPA or DATE-headline result.
