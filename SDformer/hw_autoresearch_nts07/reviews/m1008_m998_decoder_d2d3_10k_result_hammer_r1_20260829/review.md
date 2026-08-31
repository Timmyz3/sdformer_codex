# M1008 independent M998 result hammer

## Verdict

`PASS_M1008_M998_D2D3_10K_RESULT_HAMMER`

`ADMIT_D2_D3_BOUNDED_PREFIX_NUMERIC_AND_CYCLE_DIAGNOSTIC_ONLY`

M998 is internally sound at its frozen grain. The canonical ATTEMPT, D2 row,
D3 row and result root all pass atomic exact-set verification. The result array
is D2 then D3, each embedded row is byte-semantically equal to its sealed
`row.json`, and the release identities match M996 and M997.

## Numeric and cycle decision

Both bounded prefixes pass the frozen M768/M861/M890/M896 exact miter:

| Layer | Expanded requests | Compressed transactions | Diagnostic cycles | Commits |
|---|---:|---:|---:|---:|
| D2 | 10,000 | 2,167 | 7,261 | 0 |
| D3 | 10,000 | 1,903 | 8,976 | 0 |

For both rows, the exact-field set covers total cycles, expanded and compressed
request schedules, transaction addresses, cycle classes, response-slot reuse,
terminal readiness and port calendars. The six cycle classes sum exactly to
the reported diagnostic cycle total. Transaction, cycle and terminal hashes
are valid sealed identities.

Therefore:

- numeric exactness for each observed 10K prefix: **PASS**;
- address-timed cycle exactness for each observed 10K prefix: **PASS**;
- complete-row cycle/latency: **not measured**;
- decoder-complete or system speedup: **not admitted**.

The apparent expanded/compressed ratios (`4.615x` D2, `5.255x` D3) are
transaction-count diagnostics, not speedups.

## Geometry and transaction consistency

D2's input geometry independently recomputes to `231,600 B`, or 1,207 source
fetch requests at 192 B/request. D3 recomputes to `465,600 B`, or 2,425 fetch
requests. Each summary's remaining request count equals `10,000 - source
fetches`, and the summary transaction/commit fields match the exact miter.

## Boundary

Both prefixes have zero commit requests, so neither observes full-row
completion. D1 remains common-charge and no complete decoder denominator is
present. M998 must not be described as decoder complete, a Table-A production
row, full-row latency, system speedup, or a headline performance result.

No model, EDA, GPU or remote job was rerun. Existing result files and
`docs/359` were not modified.
