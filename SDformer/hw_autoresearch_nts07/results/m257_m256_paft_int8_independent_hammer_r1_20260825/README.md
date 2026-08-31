# M257 independent hammer review of M256

M257 independently reviewed the sealed M256 PAFT-ep4 running-BN bottleneck
INT8/Acc19 bridge.  The review did not use the M256 result JSON as a numeric
source.  It started from the original checkpoint, config, M248 source payloads
and all sealed M256 output bytes.

The numeric bridge itself is strong.  All 22 sealed entries and 18 generated
payloads verify.  A clean replay on a local RTX3090 with torch 2.7.1 and numpy
2.1.2 reproduces all 18 payload SHA values exactly from the A800/torch
2.2.2/numpy 1.26.4 run.  A wrong checkpoint SHA exits before emitting payloads.

Independent reconstruction requantized all 21,233,664 weights, reproduced all
four `I_KY_KX_O` layouts and all 3,072 channel rows, found no `-128` code and no
preclip saturation.  The largest channel `sum(abs(qweight))` is 215,301, giving
a signed19 positive-side margin of 46,842 for one signed unit contribution per
coefficient.

All forty M248 sources and their three packed planes reconstruct exactly.  A
separate full GPU replay covers 92,160,000 raw Conv outputs and passes all four
predeclared local gates.  Overlay loading independently reaches 210/210 keys,
missing 0, unexpected 0, ATLIF 105 and Shiftmax attention 12.

The review score is **91/100**, with P0=0, P1=3 and P2=5.  The three material
remaining hardware gates are:

1. The local gate is float source versus dequantized-weight Conv, not the
   executable `qweight -> Acc19 -> scale -> running BN -> ATLIF` integer path.
2. A 96-output tile is 663,552 bytes but current residency is 193,728 bytes, so
   the output-contiguous payload still lacks a feasible load/evict schedule.
3. Literal 96-output value masks expand dense weight bytes by 8.07x--8.72x and
   serve only 1.47--1.58 destinations per command; this is a negative screen,
   not a compression or speed result.

Verdict:
`GO_CHECKPOINT_BOUND_INT8_ACC19_NUMERIC_BRIDGE__NOGO_EXECUTABLE_INTEGER_BN_CHAIN_PHYSICAL_SCHEDULE_OR_SPEEDUP`.

M256 and `docs/359` were not modified.
