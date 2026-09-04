# M623 M617 parent-scratch energy r5 result hammer

## Verdict

**PASS — 99/100; P0=0, P1=0, P2=0.** The canonical M617 result and permanent consumed coordinate are correctly double-sealed and bound to the exact M617 runner/contract/candidate, M620 PASS98, M621 authorization/release, M622 PASS98, M597 source/M599 source hammer, M504/M528 frozen ledgers, generated-macro mapping, and read-only docs359 identity.

This closes the result-hammer gate only for the bounded claim below. It does not rewrite the canonical result or its pending-gate token.

## Independent recompute

Frozen physical accesses over ten sampled inferences are 131,926,088 reads for both designs, 218,444,544 all-write writes, and 79,581,608 dead-write-only writes. The 1,714,628 RAW forwards per output block satisfy reads + forwards = parent edges and are not charged as macro reads. Dead writes plus 17,357,867 elisions equal the active rows. Every byte count equals physical accesses × 144 B.

For nine generated 128x128-bit 1RW slices at the frozen slow corner:

- read: `9 × 11.6754 uA/MHz × 0.9 V = 94.57074 pJ` per full 1152-bit access;
- write: `9 × 11.1923 uA/MHz × 0.9 V = 90.65763 pJ` per full 1152-bit access;
- leakage: `9 × 66.6783 uA × 0.9 V = 0.54009423 mW`.

Using `(reads × read_pJ + writes × write_pJ) / 10` with pJ→mJ conversion and `leakage_mW × cycles × 3 ns / 10` gives:

| Schedule | Dynamic (mJ) | Leakage (mJ) | Component total (mJ) |
|---|---:|---:|---:|
| M504 all-write | 3.228001241293584 | 0.0738875876245375 | 3.3018888289181215 |
| M528 dead-write-only | 1.969102774033416 | 0.070529826225400191 | 2.039632600258816191 |

Difference: **1.262256228659305314 mJ per frozen sampled inference**, or **38.228307918921945%**. The cycle ratio `456016645 / 435293339 = 1.0476076800247109` is verified as a component-schedule diagnostic, not system speedup.

## Result, receipt, and one-shot

The formal result has an exact six-member manifest and valid outer seal. The terminal receipt rehashes every pre-seal output member and binds the exact authorization, runner, adapter, upstream analyzer, and source contract. The permanent consumed coordinate has an exact one-member manifest and valid outer seal; its receipt says consumption occurred before the analyzer and retry is false.

Final coordinate state is correct: result present, permanent consumed present, plain attempt absent, and no result staging, runtime staging, qraw, qstage, or qfinal residue. No caller-visible path component or sealed member is a symlink.

Ten read-only/in-memory boundary cases were rejected: duplicate key, NaN, Infinity, non-object JSON, member tamper, schema drift, one-access drift, eight-macro equation drift, RAW-forward energy charging, and zero sample denominator. A temporary non-canonical symlink case was also detected. No analyzer, runner execution, EDA, GPU, or remote action was run; canonical artifacts and docs359 were not modified.

## Admitted bounded claim

On ten frozen sampled inferences, dead-write suppression reduces the **nine-generated-macro parent-scratch datasheet component model** from 3.3018888289181215 mJ to 2.0396326002588162 mJ per frozen sampled inference: 38.228307918921945% and 1.2622562286593053 mJ saved.

This is component-only and per frozen sampled inference. It is not a camera-frame metric, C1 total, system/full-network energy, system speedup, integrated macro PPA, silicon measurement, or DATE/paper headline.
