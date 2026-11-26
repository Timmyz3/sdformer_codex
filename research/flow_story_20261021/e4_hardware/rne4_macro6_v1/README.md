# RNE4 background: six single-macro Q8 upgrades, finite cost v1

This CPU experiment reuses the **frozen independent-weight/scale** model in `../rne4_supply_v1` and binds the **prepared, not endpoint-bound** E3 probe contract. It does not claim algorithm quality, runtime prediction, RTL implementation, or whole-network latency. The old Q6 E2 experiment is not used as a benefit source.

The actual action is one row-major macro of 151 complete 9x9 tiles promoted from independently scaled RNE4 directly to Q8. All four patch convolutions choose the same macro. There are six disjoint choices covering 906 of 910 complete tiles; the remaining four complete tiles remain RNE4. True tails always use Q8. This is **extra computation to buy accuracy relative to the RNE4 background**, not savings relative to old Q6.

`prepare.py` reads the exact prepared contract, checks the 151 tiles for all four targets and materializes its 910-entry group-ID table. `reproduce.sh` runs one complete hot-conv geometry (T10/H240/W320/K864/Cout96) for Q4 background and all six static/dynamic choices: 13 complete operator arms. `check_and_report.py` independently reconstructs every row, tile-span descriptor, precision transition, extra arithmetic beat and control/DMA delta. The Q4 background is checked against the frozen RNE supply result for unchanged counts/timing.

## Static and dynamic control

Both cases have the same dual Q8/RNE4 weights and scales: 129,792B in one 1MiB SRAM slot, and 22,016B in the existing 32KiB weight RF. All per-row scale reads, conversions, canonical output writes, actual input halo/bank access, DMA stalls and three BN passes remain charged. The two representations are both explicitly loaded/refilled; neither is an alias of the other.

* Static: one 910B precomputed plane-selection table. A new complete-tile span reads one aligned 32-bit descriptor, taking one statistics-RF tick. Precision changes each pay one additional control tick.
* Dynamic: one fixed 910B group-ID table, plus a legal 3bit selected-macro register. Only values 0–5 are accepted; group 6 denotes the four unselected remainder tiles. A new span reads one aligned 32-bit group ID in one tick and compares/decodes the selected ID in another tick. Precision changes pay the same control tick as static. One descriptor read is reused through the contiguous horizontal tile span, including batch splits. Selection is stable for this complete operator, as required by the prepared algorithm action.

The table and selection register use the existing 2KiB statistics-RF reservation; no port/capacity is added. The table DMA is 960B after 64B rounding. Dynamic selection adds a separate 64B transaction (including 32-tick startup), and one latch tick. This deliberately pays a conservative external handoff for the already computed 3bit decision. A future on-chip predictor handoff could change that small cost, but its computation/feature reads must be paid separately. **No prediction computation or future-information access is fabricated here.**

The dynamic controller is a finite model, not newly synthesized logic. Weight/scales are existing E0 representations; actual endpoint operands and this decoder have not been RTL validated. The underlying model is event-advanced with per-beat address/port checks, not a tick-by-tick RTL trace or a complete network physical memory map.

## Result

Background Q4+Q8-tail affine+BN service is 207,641,720 ticks. Each action upgrades 122,310 rows, yielding 153,210 Q8 rows including true tails and 614,790 Q4 rows. It adds exactly **20,548,080 accepted arithmetic beats per convolution**, or **82,192,320 for the four isolated convolution services**. The latter is not network wall-clock latency.

| Selected macro | Static affine+BN ticks | Dynamic affine+BN ticks | Dynamic overhead versus same static |
|---|---:|---:|---:|
| 0 | 228,700,049 | 229,213,723 | 513,674 |
| 1 | 228,701,342 | 229,215,016 | 513,674 |
| 2 | 228,701,342 | 229,215,016 | 513,674 |
| 3 | 228,700,502 | 229,214,176 | 513,674 |
| 4 | 228,701,342 | 229,215,016 | 513,674 |
| 5 | 228,701,342 | 229,215,016 | 513,674 |

There are 519,480 descriptor reads per operator. Dynamic pays that many additional decode ticks plus 34 selection handoff ticks. Its later PE schedule absorbs 5,821 input-DMA and 19 output-DMA stall ticks; therefore the observed net dynamic cost is 513,674 ticks rather than the unoverlapped 519,514 ticks. The independent check verifies this complete difference identity. Macro service spread is only 1,293 ticks, despite distinct spatial placements.

Dynamic action adds 21,572,003–21,573,296 total ticks over the Q4 background. The four **isolated service sums** add 86,288,012–86,293,184 ticks; dynamic-versus-static overhead sums to 2,054,696 ticks. These sums are useful for provisional budgeting, with no claim that BN/PSN, consumers or inter-layer overlap have been integrated.

The declared Q8 reference costs 321,222,167 ticks, leaving 92,007,151–92,008,444 ticks before this one dynamic operator loses its speed advantage over that Q8 reference, **before any future predictor or integration costs**. This is only an upper bound under the declared resource schedule. The reference is the faster of two implemented choices, parallel16 and bitplane128; it is **not** a lower bound on all possible sparse hardware, and does not include actual-activation zero-group cancellation or all published accelerator mechanisms. This scope clarification also applies to the word “strong” in the frozen RNE20-arm report.

There is **no free predictor budget versus the same static macro**: dynamic is already 513,674 ticks slower before scoring. To justify the dynamic algorithm, future measured selection must retain more flow accuracy at a comparable total budget, or pay for its overhead with additional explicit computational savings. The Q8 headroom cannot substitute for demonstrating advantage over an ordinary same-budget static or general-score baseline. The prepared E3 oracle probe supplies no measured quality result yet.

All results and checks are in `results/`; `SOURCE_BINDINGS_V1.json` pins input sources. No GPU job, new RTL/DC experiment, old source edit, or network claim is part of this deliverable.
