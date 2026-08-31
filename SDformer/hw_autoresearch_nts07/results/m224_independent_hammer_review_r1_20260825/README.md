# M224 independent hammer review

**Score: 94/100. P0: 0. P1: 4.**

The review independently unpacked and recomputed all 100 selected M51 bitpacks. It did not import or invoke `analyze_m224_h67_fc1_parent_delta_bank_service_screen.py`. The production receipt was used only after recomputation as an equality oracle.

## Verdict

M224 is admitted as an exact, fail-closed trace screen. It is also correctly a **NO-GO** for the current same-vector multi-source K-bank FC1 line: neither fixed-product-lane family reaches the frozen `1.5x` gate.

- Shared-96 best legal point: spatial K1/D96, `1.1902524563411723x` versus raw K1/D96.
- Matched-128 best legal point: spatial K1/D128, `1.1760546313993623x` versus raw K1/D128.
- Raw shared-96 K2/K4/K8 ratios are `0.900952x / 0.780774x / 0.640354x`; multi-source grouping loses to the strong K1 after expanded-destination slicing.
- Cross-family speedup was not used. Every point stays within its fixed 96- or 128-product-lane family and at most eight 128-bit weight-bank equivalents.

The only positive opportunity is spatial parent delta: raw work `112,213,979` drops to `87,209,538` signed events (`1.286716815x` source-work). After serially charging current scan, valid parent scans, choice bits, prior-output seed, service and final commit, it becomes only `1.190252x` in the 96-lane family.

## Independent checks

- 100/100 payload hashes and packed-byte/shape identities pass.
- The selected set is exactly 10 samples x 10 module indices `{6,8,11,13,16,18,20,22,24,26}`.
- All selected runtime inputs have binary01 ratio exactly one and all weights are `[4C,C]`.
- Raw/spatial/temporal positive plus negative events equal source events exactly.
- Every recomputed aggregate integer field matches the production result; all 24 per-sample ratio distributions match as well.
- The two stage-3 FC1 rows have binary01 ratios `0.8996497393` and `0.8643710613`; no stage-3 FC1 bitpack is in the admitted payload, so conventional fallback is identity-bound.
- The frozen M224 `SHA256SUMS` verifies, and `docs/359` remains `dedde7ce...`.

## Remaining P1 boundaries

1. The recurrence is not RTL: SRAM conflicts, finite queues, line alignment and controller bubbles are not measured.
2. Acc19 range and signed-INT8/fixed-point equivalence remain unproved.
3. Parent-output seed cycles are charged, but the retained-output storage/macro contract is absent.
4. Two nonbinary stage-3 FC1 modules remain conventional, so this is not complete FC1/FFN/system evidence.

These limits are consistent with M224's existing admissions. Missing physical costs can only weaken the candidate, so they do not invalidate the K-bank no-go.

Recompute command:

```bash
/opt/anaconda3/bin/python results/m224_independent_hammer_review_r1_20260825/m224_independent_raw_recompute.py
```

The recompute must end with `PASS_INDEPENDENT_RECOMPUTE_NO_GO_1P5X`, zero aggregate mismatches, and zero distribution mismatches.
