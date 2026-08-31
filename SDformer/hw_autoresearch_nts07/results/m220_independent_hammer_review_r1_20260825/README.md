# M220 independent hammer review

Verdict: **90/100, P0=0, P1=7, GO with a strictly bounded claim.** M220 is admissible as supporting evidence that the exact M218 K8 and native-cropped M219 K1 service RTLs are bit-exact over the stated directed matrix and follow the stated L4/II1/O8 acceptance schedule. It is not a complete FC2, FFN, physical-memory, frozen-H67-speedup, or system result.

## Independent reproduction

- The sealed `SHA256SUMS` checks in full. Its SHA-256 is `a2a70b067ba96e8e3fb3927c06e00f2628aef38dad9fda5de05789c7c63ced02`; `RUN_COMPLETE.txt` is `8c975f48ece56cca33c2fdb95b2069d00e2afcf08d9a900868e8e5f69eb60a1d`.
- All five RTL/TB/filelist/contract SHA pins and the protected `docs/359` SHA match the runner contract.
- A fresh Synopsys VCS V-2023.12-SP1 compile and run with seed `220925` returned RC 0/0, no compile warning/error, and the exact 33-pair PASS with 3912 recurrence checks.
- The seed change does not add coverage because the TB is deterministic. The replay is evidence of reproducibility only.
- All 33 printed rows independently satisfy `M218 cycles = 12B+7`, `M219 cycles = 6B(N+1)+7`, and `active reads = 6BN`. The recurrence-check total independently recomputes to 3912.

## Semantic review

For each M218 group, the stimulus asserts the first `N` banks, sets each active channel's low three bits to its bank ID, and the RTL expands the group into six slices. M219 receives the corresponding single-bank groups in source-major order, with blocks inside each source, and expands every group into the same six slices. Thus the same `(block,slice,bank,channel)` weight multiset reaches both accumulators. Static inspection confirms M218 sign-extends eight INT8 values through a sufficient 11-bit tree before Acc24 addition, while M219 sign-extends each INT8 value directly into Acc24. With no tested overflow, the two orders are arithmetically equivalent. The dynamic `-128/+127` case checks the principal signed endpoint path for banks 0 and 1.

The memory model accepts at most one request per DUT per cycle, records epoch/slot/generation/tag and request semantics, returns the oldest due generation, and checks response acceptance exactly four cycles after request acceptance. Slot reuse on a same edge is explicitly protected. Conservation checks require requests = responses = context writes, six result beats per output block, and equal active-bank reads between M218 and M219.

The K1 source-major distance between consecutive updates to one `(block,slice)` context is `H=6B`, so the recurrence used by the TB is structurally correct:

`issue[i] = max(issue[i-1]+1, issue[i-8]+4, issue[i-(6B)]+4)`.

There is no omitted response-skid hazard in that bound: context ownership is released at response acceptance; the old response commits from the skid one cycle later, and the newly issued request cannot return until four cycles later.

However, under the tested parameters `L=4`, `O=8`, and `H>=6`, both non-II terms are strictly nonbinding. The 3912 comparisons therefore validate the fixed L4 matrix but do not exercise an O8-full or same-context wait. This is the main reason the result is admitted only as a bounded claim.

## P1 findings

1. M218 output is the numeric reference for M219; there is no independent arithmetic oracle. Static arithmetic inspection narrows the risk, but a shared semantic bug can still evade the miter.
2. The TB is deterministic, so different seeds do not vary data, ordering, latency, or stalls.
3. O8 and same-context recurrence branches are nonbinding at L4; only II1 controls the measured issue schedule.
4. Only oldest-generation responses and always-ready request/result/done interfaces are covered. Arbitrary OOO response order and backpressure are outside this milestone.
5. The signed endpoint case covers banks 0/1 only; it is not an Acc24 overflow-boundary sweep.
6. Each DUT has an identically configured but separate behavioral memory model, and the response widths differ (1024b versus 128b). No shared physical macro or memory energy is proven.
7. Small-token cycles include fixed result drain. They must not be interpreted as, or used to recalibrate, the frozen H67 4.952x standalone-service aggregate.

## Admission decision

- Allow `m218_m219_cross_module_equivalence` only as: **33-pair bounded directed equivalence for N=1..8, B=1/2/4/8, L4/II1/O8, oldest-response and always-ready interfaces**.
- Allow `l4_ii1_o8_bounded_recurrence` only for that same model, explicitly noting that O8 and context terms are nonbinding.
- Allow M220 to support M218/M219 performance-area admission only alongside the already sealed exact DC evidence and its pre-macro/logic-only boundaries.
- Do not allow M220 to claim complete FC2/FFN, macro capacity/energy, physical or system speedup, or any headline speedup. Do not substitute its small-token cycle ratios for the frozen-H67 aggregate.
