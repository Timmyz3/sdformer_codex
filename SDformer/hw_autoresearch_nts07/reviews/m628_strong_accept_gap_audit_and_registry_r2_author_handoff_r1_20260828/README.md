# M628 Strong-Accept gap audit and paper-metric registry r2

Status: **author self-check PASS; pending M629 independent hammer**. This milestone admits no paper headline.

## Outcome

- Table A is intentionally empty: 0 eligible rows and no admitted speedup/energy/PPA headline.
- Table B holds local M528/M573/M623/M481/M480/M518/M519/M523 evidence.
- Table C holds M618/M619 external Prosperity validation and M526/M532 Prosperity/Phi/FireFly-T context.
- The executable gate accepts only `DIRECT_UNIFIED_CYCLE_SIM` Table A rows, requires the complete baseline ladder, fixes Dense96 Fixed-T10 as numerator, and forbids multiplication or B/C promotion.

## Analytical 1.794–1.823 audit

Independent Decimal division gives:

- `1442206883 / 803774000 = 1.7942940217026179000564835389`
- `1442206883 / 790920000 = 1.8234548159105851413543721236`

The arithmetic is sound; admission is not. The range is a partial analytical envelope without an exact decoder trace, memory timing/stalls, common-resource and fixed-numerator completion receipts, full-network completion, population coverage, system energy/PPA closure, or an independent result hammer.

## Minimum evidence set: 3.4/5 to conditional >=3.8/5

1. Decoder-complete exact full-network traces with frozen population identity.
2. One unified CPU cycle+memory replay for Dense96/B1/K1/K1x8/K8/Ours exact, with same resources, fixed numerator, completion receipts, and all five aggregates.
3. Later matched logic+macro area/STA and logic/SRAM/DRAM system-energy closure.
4. Three sequences or preregistered low/mid/high event-density strata, followed by a blind independent hammer.

This is a conditional evidence-quality lift, not a guaranteed review score. A fourth matcher is neither required nor proposed.

## No-EDA work now

The registry additionally locks six executable tasks: full-100 Prosperity replay (Table C only), address-bearing bank-timed M481 replay, M480 address/numerical miter, preregistered density aggregation, a unified Table A adapter, and a component-to-system-refusing energy combiner.

## Verification

`python3 -m unittest -v hw_autoresearch_nts07/system_simulator/tests/test_m628_h67_paper_metric_registry_r2.py` passes 11/11 tests. The builder reports:

```text
M628_REGISTRY_PASS sources=12 table_a_eligible=0 headline_admitted=false analytical_admitted=false
```

No EDA/GPU was run, no paper prose was written, and docs359 remained SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
