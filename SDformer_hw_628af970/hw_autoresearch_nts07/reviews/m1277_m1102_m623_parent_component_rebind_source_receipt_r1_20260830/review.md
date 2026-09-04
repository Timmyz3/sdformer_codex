# M1277｜M1102↔M623 parent-component rebind source receipt

## Outcome

The additive source-only adapter and contract are complete. Python 3.6.8
compilation passed; the positive exact-identity binding passed; all 11
fail-closed attacks were rejected.

The adapter binds these facts without creating a new energy result:

- identical frozen scope: H67 ep35, one sequence, ten sampled inferences,
  812,160 tasks and four bottleneck Conv3x3 operators;
- identical candidate parent vector between M1102 and M623: `131,926,088`
  reads, `79,581,608` writes and `13,717,024` RAW forwards;
- M1102 candidate cycles `434,242,823` versus the M617/M528 energy row's
  `435,293,339`, a difference of `-1,050,516` cycles or
  `-0.241335188453228318%`;
- M1102's `1.7591725401987818x` denominator is strongest-zero or
  same-coordinate-bit, both with zero parent accesses;
- M623's `38.228307918921945%` denominator is M504 all-write for the same
  candidate mechanism.

The machine-readable binding therefore fixes both merge prohibitions to
`false`: it cannot claim candidate-vs-zero/bit energy reduction, and it cannot
combine `1.759x` and `38.2283%` as one energy-efficiency pair.

## Identity

- source SHA256:
  `92b9cc8135b30e1fbba7b15f5e4575cf31cdfb668695de043ad84d5bf51343b1`
- contract SHA256:
  `59c08c0d0f09df349ded2d033bac33b41eab3488df36073fff8a5309a5f9c0d8`
- exact authorities: M1102 result `a229c21b...`, M1114 review
  `8ced2392...`, M617 result `be384c45...`, M623 review `96812391...`
- docs359 SHA256:
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## Fail-closed self-test

Rejected attacks:

1. M1102 read drift;
2. M1102 write drift;
3. M1102 forward drift;
4. nonzero strongest-zero parent access;
5. M617 dead-write count drift;
6. M617 checkpoint/scope drift;
7. M1102 candidate-cycle drift;
8. M1114 energy-admission promotion;
9. M623 C1-total promotion;
10. duplicate JSON key;
11. nonfinite JSON.

The test output is frozen in `selftest.json`. `--print-binding` also completed
successfully. Neither mode writes a canonical result.

## Claim boundary

This is a provenance bridge for two separately labelled component-table rows.
It is not updated leakage/dynamic energy, candidate-vs-baseline energy, total
C1/system energy, RTL/EDA evidence or paper-PPA-ready evidence. No EDA, GPU,
remote job or production analyzer was run, and docs359 was not modified.
