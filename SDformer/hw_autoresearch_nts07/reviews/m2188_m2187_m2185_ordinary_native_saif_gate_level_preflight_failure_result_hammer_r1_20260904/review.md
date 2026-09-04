# M2188 independent M2187 failure/result hammer

## Verdict

M2187 remains **failed, permanently non-citable, and non-retryable**. There is no canonical result, so neither raw file may be promoted to power or energy evidence under this identity.

The technical failure is nevertheless narrow and well isolated: it is a **diagnostic-only sub-timescale conservation-parser false negative**, not empty SAIF and not activity pollution.

## What passed

- VCS compiled all six modules; simulation completed with the frozen ordinary-axis PASS token.
- Runtime parsing, the arithmetic scoreboard, and the exact ledger pass: 20,292 cycles, 149 rows, 1,278 issues, 29,472 products, 24 commits, 1,788 bundles, and 14,304 scalar weight reads.
- Both SAIF files are large, nonempty, individually double sealed, and contain exactly 93,971 activity records under exactly one `dut_ordinary`; zero activity records occur outside that DUT.
- The measurement SAIF covers exactly 60,876 ns. All 93,971 records conserve exactly, all eight critical valid/accept interfaces toggle, 76,264 records toggle, and TX is zero in every record.

## Exact failure diagnosis

The prehistory SAIF declares `TIMESCALE 1 ns` and `DURATION 1167.01`. Every T0/T1/TX field is an integer, and every one of the 93,971 records sums to exactly 1167. Thus every record has the same 0.01 ns residual. This uniform residual is the fractional part of the duration and is strictly below one 1 ns SAIF tick.

That shape is incompatible with random missing activity or hierarchy pollution: hierarchy, record count, critical `load_valid` activity, and file seals are correct, while the integral-duration measurement window conserves perfectly. It is high-confidence evidence that VCS retained the fractional duration header but quantized activity fields to the declared 1 ns timescale.

The diagnostic file has 45 TX-bearing records totaling 45 ns. This is expected pre-reset diagnostic history and remains isolated from annotation; the post-reset measurement has zero TX.

## Minimal successor

The attached `m2193_minimal_successor_contract.json` permits only a parser-source successor. For the diagnostic role alone, it may accept the observed case only when all activity fields are integer ticks and every record sums exactly to `floor(DURATION)`, with an identical residual equal to the strictly sub-tick fractional remainder. Measurement conservation remains exact.

The successor tests must reject a 1.01-tick residual, negative/ceil residual, nonuniform residual, fractional activity field, any measurement residual including 0.01, measurement TX, wrong hierarchy, missing records, and missing critical activity. This rule cannot hide a full-cycle error.

M2195 execution remains unauthorized until a separately sealed M2193 source and M2194 independent source review exist. M2187 raw files may not be reused.
