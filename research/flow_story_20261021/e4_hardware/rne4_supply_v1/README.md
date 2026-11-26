# Ordinary RNE4: independent weight/scale supply, v1

This is an executed finite-resource **operator address-event model**, not a network timing result, tick-by-tick RTL simulation, new RTL implementation, or an energy estimate. It evaluates the ordinary E0 quantizer without assigning it the free shared-weight cost of the old nested bit-plane experiment. The frozen E4 and E2 directories are unchanged.

## Source and numerical contract

`SOURCE_BINDINGS_V1.json` binds the frozen E0 `mixed_integer_core.py`, `mask_contract.py`, R30 evaluation mask, R27 inventory and original E4 supply source. `prepare.py` generates this independent C++ model and the exact evaluation masks; `reproduce.sh` compiles and runs all 20 arms on CPU. `check_and_report.py` independently reconstructs all row counts, accepted arithmetic beats, weight/scale refills, descriptors, variant controls and index costs from the frozen geometry/mask.

E0 RNE uses independent per-output-channel scales: `s8=maxabs/127`, `q8=round_even(W/s8)` clipped to [-127,127]; `s4=maxabs/7`, `q4=round_even(W/s4)` clipped to [-8,7]. Its low carrier is `q4<<4` with scale `s4/16`. Low planes are a separately quantized weight representation. They are **not** Q8's high four bits. Existing integer PE start-plane arithmetic can express this carrier in principle; these E0 endpoint weights/actual operands and the new weight/scale selection circuit have not been RTL validated or DC synthesized here.

There are five arms for each of four complete T=10 domains:

1. Q8 ordinary parallel16, one representation/scale.
2. Q8 bitplane128, one representation/scale.
3. Q4 all rows including true tails, one representation/scale. **Component reference only; not the actual E0 Q4 quality arm.**
4. Actual E0 Q4: RNE4 on complete 9x9 tiles, Q8 on true tails, two representations/scales.
5. Actual frozen M: selected complete tiles and true tails Q8, remaining complete tiles RNE4, two representations/scales. The fixed mask is the original R30 evaluation mask, not a runtime flow predictor.

## Finite supply contract

Resources stay at E4 capacity/ports: one PE context with 16 output lanes; parallel16 or serial128 reduction; 4MiB resident binary input; two 512KiB input and two 512KiB output slots; two 1MiB weight slots; 32KiB weight RF; 256KiB accumulator RF and 256KiB statistics RF. Each pingpong input/output pair shares one 16-bank 32-bit 1RW port group. Weight SRAM/RF has one 256B/tick port; DMA is shared 64B/tick with 32-tick startup. An additional fixed-mask table is at most 910 bytes, reserved within 2KiB of the existing statistics RF, not added capacity. The second weight SRAM slot is not treated as free extra RF or bandwidth.

Each weight representation includes its own 64B scale vector per output16. Each representation is explicitly fetched by DMA (including a separate transaction startup) and each output16 block/scale refilled from SRAM. SRAM holds compact low weights at four bits/weight; physical K padding is paid. Weight RF regions align each weight-plus-scale block to 256B; wasted RF alignment capacity is paid. Scale vector reads take one separate RF tick per row/output16, serialized against weight accesses; conversion/multiply takes two ticks, canonical output write one more. The latter scale-read/output-write charges tighten the old E4 assumption for **all** arms, including Q8; hence old and new absolute ticks are not interchangeable.

The hot conv, stage2 FC1 and K96 cases hold both representations/scales simultaneously in the 32KiB RF and use natural row order. Stage2 FC2 requires 37,376B for both and therefore does not fit. It classifies each batch into two bounded row lists, stores explicit 32-bit indices after the halo+packed input region, reads each index for every output16, refills each representation separately and scatters output to its original canonical position. Classification, index stores/reads, variant control and per-variant scale refills are charged. The model checks that each read index equals the intended original row. No unbounded reordered output buffer is assumed.

The fixed M descriptor is read on each new contiguous complete-tile span (per output16 for natural scheduling, once during partition for grouped scheduling). A byte table occupies the stated capacity; actual reads are aligned 32-bit transactions. One explicit tick is paid for each variant switch, plus the descriptor read tick. This is fixed ordinary selection, not a scored runtime scheduler.

Input gather uses clipped conv halo addresses, exact source-word bank multiplicity and padded packed-row addresses. Input DMA stalls PE input reads; output DMA blocks output retirement. The C++ checks/hash-accumulates every accepted weight/activation beat and emits a bounded trace prefix. Global source objects use relative address spaces; this is not an integrated physical network memory map. Full per-beat traces are not saved. Finite scheduling is event-advanced, not clock-tick RTL.

All affine outputs are stored before three complete mean/variance/normalization BN passes and explicit finalization barriers. This pays full-domain BN traffic and service. Floating-point reduction equivalence, the vector unit, PSN/consumer timing, the complete upstream/downstream schedule and top-level RTL are **not** closed by this model.

## Executed result

Ticks below include affine and the three BN passes. The Q8 denominator is the faster resource-matched ordinary or bitplane implementation for that operator (ordinary wins conv and K96; ties at the other two). These are individual operator results, not a sum normalized into a network speedup.

| Complete operator | Strong Q8 ticks | Actual Q4 + Q8-tail ticks / speedup | Fixed M RNE4+Q8 ticks / speedup |
|---|---:|---:|---:|
| patch conv, K864/Cout96/H240/W320 | 321,222,167 | 207,641,720 / 1.547002x | 242,430,302 / 1.325008x |
| stage2 FC1, K384/Cout1536/H30/W40 | 45,432,288 | 34,540,050 / 1.315351x | 37,507,263 / 1.211293x |
| stage2 FC2, K1536/Cout384/H30/W40 | 32,227,448 | 21,436,598 / 1.503384x | 24,236,005 / 1.329734x |
| stage0 FC1, K96/Cout384/H120/W160 | 97,225,612 | 89,585,130 / 1.085287x | 94,980,141 / 1.023641x |

All 20 arms pass independent geometry/count/refill/control/index and capacity checks in `results/ACCOUNTING_CHECKS_V1.json`. Complete transaction and accepted-beat counters are in `SUPPLY_REPLAY_V1.json`; the concise table is `SUMMARY_V1.json`. All counters are model outputs under declared port/vector costs, not measured silicon cycles. There is no new GPU, quality evaluation, power measurement or DC run in this extension.

The 28-operator logical ledger (`STORAGE_LEDGER_28_V1.json`) contains 17,584,128B Q8 weights, 8,792,064B additional packed Q4 weights, and 89,856B **per** scale representation: dual total 26,555,904B before physical padding. It explicitly flags all four stage3 FC operators as exceeding a single 1MiB weight slot, even for single Q8 and single Q4; two stage3 FC2 operators also exceed 32KiB RF for one Q8 output16 block. Their full service requires weight DMA stripes/reduction chunks that this four-domain model has not implemented. Do not extrapolate four domains into whole-network coverage.

The same frozen-M algorithm quality result can be paired only with the fixed-M dual-representation arm under these model limits. The Q4-tail and all-row component-reference costs must not be paired with fixed-M quality. The findings support ordinary RNE as a stronger baseline with a concrete hardware cost. They do not demonstrate a flow-specific predictor advantage or inherit the paused E2 patch-pair savings.
