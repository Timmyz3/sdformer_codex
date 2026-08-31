# M179 H67 FC2 finite dual-window reservoir DSE

Status: **PASS exact-payload finite-buffer cycle DSE; RTL/open memory ports remain open.**

M179 targets the expensive fragmentation left by M177: events from separate
96-bit descriptors cannot currently share a four-bank source group.  It uses
two ping-pong windows.  Each window accepts one native/preindexed nonzero96
descriptor per cycle, pools at most `D` descriptors, then emits at most four
distinct-bank sources per group.  The other window may fill while one drains;
a window cannot be reused before its prior drain finishes.  This finite rule is
charged exactly, including fill wait and buffer reuse stalls.

All 120 frozen H67 FC2 payloads (437,760,000 bytes) were SHA/size/popcount
checked.  K1 and K4 use the same window sizes and schedule.  The selected
per-stage depths are `D={2,4,8,8}`:

| stage | D | matched K1 wall | K4 wall | K1/K4 | D1/K4 gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 56,712,625 | 24,508,539 | 2.313995x | 1.003543x |
| 1 | 4 | 69,552,384 | 21,118,710 | 3.293401x | 1.092922x |
| 2 | 8 | 216,516,509 | 59,002,077 | 3.669642x | 1.163815x |
| 3 | 8 | 88,135,752 | 22,951,872 | 3.840025x | 1.211341x |

Aggregate matched K1/K4 is **430,917,270 / 127,581,198 = 3.377592x**.
Relative to M176/M177's one-descriptor K4 wall of 144,146,504, bounded pooling
adds a further **1.129841x**.  Maximum raw bitmap storage is two eight-entry
windows, or 1,536 payload bits before index, validity and identity metadata.

The key qualification is that this is a finite cycle simulator, not a free
global bank queue.  It still assumes an ATLIF-native/prebuilt index and a token
directory count.  Producer/directory RTL, physical window storage, the wider
cross-entry selector, weight response, M169 arithmetic, accumulator context and
reordered-accumulation numeric equivalence remain open.  The 3.377592x ratio is
therefore a standalone matched frontend schedule, not physical, complete-FC2,
FFN, network, system or headline speedup.

Stage 0 gains only 0.35% from cross-beat pooling, whereas stages 2/3 gain
16.38%/21.13%.  A practical RTL should retain the cheap D2 mode for stage 0 and
activate D4/D8 only where replay provides enough time to hide fill.
