# M226 independent hammer review

**Score: 91/100. P0: 0. P1: 3. P2: 2.**

M226 closes the M225 reference-attribution blocker. The M225 production, independent-review and M226 seals all pass. Independent division of the sealed integer ledgers reproduces all six factors and both ten-sample raw ranges with zero mismatch.

## Verdict

`raw K8/F1` is the correct **equal-context/storage** reference for this RTL screen: it has the same K8 grouping geometry, 14,592-bit Acc19 resident state and 768-bit weight interface as raw K8/F2/F4. It is not an equal-total-logic or equal-area reference: F1/F2/F4 have 96/192/384 signed add lanes. Therefore raw F2/F4 are valid exploratory parameter points for matched RTL/VCS/DC pricing, while `1.568695x/2.112902x` remain trace-cycle ratios rather than achieved hardware speedup or area efficiency.

| Factor | Independent recompute |
|---|---:|
| raw K1/F1 to raw K8/F1 grouping/descriptor | 1.028612929x |
| raw K8/F1 to spatial K8/F1 parent | 1.189066375x |
| spatial K8/F1 to F2 multicast only | 1.473618660x |
| spatial K8/F1 to F4 multicast only | 1.838372181x |
| raw K8/F1 to spatial K8/F2 combined | 1.752230399x |
| raw K8/F1 to spatial K8/F4 combined | 2.185946545x |
| raw K8/F1 to raw K8/F2 | 1.568695409x |
| raw K8/F1 to raw K8/F4 | 2.112901791x |

Raw F2 ranges from `1.559855466x` to `1.578960520x` over ten samples; raw F4 ranges from `2.084994467x` to `2.147819440x`. The inherited population is 100 exact-binary records: ten samples by ten stage-0/1/2 FC1 modules. Two nonbinary stage-3 FC1 modules remain conventional.

## Minimum RTL scope ruling

M226's quantified minima are correct for C384/K8 and must be present in every F variant:

- 14,592 bits of Acc19 state;
- 3,072-bit presence and 3,072-bit sign masks;
- one 768-bit held-weight register and the same 768-bit weight-read interface;
- 96/192/384 signed lanes for F1/F2/F4;
- 256-bit scanner, source walker/replay, tag/epoch/valid and stall/fault isolation.

This is a sound high-level minimum, but not yet a complete executable RTL contract. The next contract must explicitly include source/output-block/weight-address/context-select state, F-way accumulator porting and commit drain, request/ready/credit backpressure, and mask clear/generation behavior. All three variants must instantiate the same mask/sign/protocol frontend.

The raw performance traces contain no negative events and do not exercise row or output-block tails. Before DC admission, VCS must include a bit-exact signed miter plus empty-mask, full-eight, mixed-sign, row-tail, partial-output, stall/fault and backpressure cases.

## Claim boundary

- M226 correction and raw parameterized RTL screen: **GO**.
- Raw K8/F1 as equal-context/storage reference: **GO**.
- Raw K8/F1 as equal-area or equal-total-compute reference: **NO-GO**.
- Achieved hardware speedup, throughput/area, complete FC1/FFN/system or headline: **NO-GO** until matched Synopsys evidence and fallback composition exist.

No M226 artifact, paper body or `docs/359` was modified by this review.
