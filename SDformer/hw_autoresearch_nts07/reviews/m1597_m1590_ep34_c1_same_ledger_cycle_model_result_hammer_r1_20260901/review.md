# M1597 — M1590/M1579 ep34 C1 production result hammer

Verdict: **PASS for the narrowly scoped CPU `[cycle model]` result**.  On the
same sealed `51,840,000`-row ledger, the product-capture candidate takes
`382,848,700` cycles versus `648,741,051` for strong zero and `646,619,098`
for same-coordinate bit: `1.694510×` and `1.688968×`, respectively.

## What was independently checked

- The five payload members, manifest and outer seal are exact.  The ledger is
  `466,560,000 B`; every row is exactly `0000<lowercase support16>\n`; its SHA
  is `daa62651...`, and an independent streaming scan found exactly
  `51,840,000` rows and `78,668,732` active support bits.
- The checkpoint, capture manifest, ordered record stream, M1524 mapping,
  M504/M505/M528 recurrences, frozen producer, release and M1589 authority all
  match their pinned identities.  The ordered capture contains 40 retained C1
  records and `9,275,617` active inputs across the four operators.
- All ten sample rows independently sum to the published aggregate cycles.
  Every row ratio and every sample/operator distribution field was recomputed.
  Sample-major speedup versus strong zero ranges `1.661091–1.723155×` with
  geometric mean `1.694574×`; versus bit it ranges `1.656397–1.716181×` with
  geometric mean `1.689030×`.  The 40 operator-isolated rows are explicitly
  non-summable.
- Product, parent and completion conservation all close.  The published
  `27,160,940,160 B` traffic is correctly reconstructed, but it is parent
  scratch traffic only—not total SRAM or DRAM traffic.
- There is one consumed read-only attempt marker, one canonical result, no
  retry artifact, a one-CPU-run release and a frozen source gate rejecting more
  than three workers.  The actual selected worker count is not written into the
  final receipt; this is a P2 observability gap, not a deterministic-cycle
  correctness failure.

## Capacity correction

M1590 inherited M528's old `213,376 B macro-rounded` field.  It is superseded
by M1591/M1596: `214,912 B logical`, `215,040 B mapped`, still within the
`245,760 B` budget with `30,720 B` margin.  This is a P1 presentation/resource
accounting correction and does not change any M1590 cycle or ratio.  The paper
must use the newer capacity and must not cite `213,376 B` as final C1 storage.

## Paper-safe wording

> For Motion C12 ep34 live93 on ten `zurich_city_09_a` samples and four
> bottleneck Conv3x3 operators, exact single-port product capture requires
> `382,848,700` cycles versus `648,741,051` for strong zero and `646,619,098`
> for same-coordinate bit on the same ledger—`1.694510×` and `1.688968×`
> `[cycle model]`.

This is not RTL or mapped-gate cycles, wall-clock latency, a full-network or
system speedup, multi-sequence evidence, energy, power, or PPA.
