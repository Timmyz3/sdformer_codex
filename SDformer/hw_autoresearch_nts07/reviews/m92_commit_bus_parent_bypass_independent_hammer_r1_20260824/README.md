# M92 commit-bus parent-bypass independent hammer

Verdict: **NO-GO confirmed; 67/100, P0=0, P1=3, P2=3.** The frozen transaction experiment is reproducible and its accounting is correct, but the bypass does not pass M92's own performance gates and is not worth implementing as a performance feature.

## Independent result

The audit verifies exact SHA identity for the M92 contract, probe, raw result, first failed log, completed log, receipt, M91 raw/receipt/probe, M89 receipt, and M45 analyzer. The first remote run failed closed on the `WIDTH`/`W` interface name; the second log contains all 40 record markers and a complete NO-GO result.

The raw 40-record ledger independently reconstructs ten samples with four operators each:

| Metric | M91 | M92 | M92 − M91 |
|---|---:|---:|---:|
| Source cycles | 69,211,896 | 69,270,080 | +58,184 |
| Integrated cycles | 75,930,816 | 75,851,184 | −79,632 |
| p95 nearest-rank cycles | 7,769,480 | 7,760,888 | −8,592 |
| Parent-wait cycles | 1,968,824 | 1,933,856 | −34,968 |
| Response/context wait | 2,040,128 | 1,929,880 | −110,248 |
| Fusion groups | 10,078,648 | 10,203,608 | +124,960 |

All ten samples improve integrated cycles, but all ten regress source cycles. The integrated reduction is only 0.104874%, while the frozen gate requires at least 0.25%. M92 is 110,196 cycles above its maximum promotable result of 75,740,988.

The accounting is exact: 11,805,832 parent demands equal 10,402,176 SRAM reads plus 1,403,656 forwarding hits. Hits are 11.8895% of demand; 980,856 are left-parent hits and 422,800 are up-parent hits, a 69.88%/30.12% split. Previous-timestep parents are never forwarded and events with `commit_time < now` are explicitly rejected.

## Why 1.4 million hits barely help

The parent read is usually not on the critical path. The 1,403,656 hits remove only 34,968 cycles classified as parent wait: one saved wait cycle per 40.14 hits. Earlier readiness also perturbs greedy grouping. M92 produces 124,960 more fusion groups and adds 58,184 source cycles even though logical updates and unique weight issues both decline.

This is also an optimistic cost model. A parent vector is 96 lanes × 24 bits = 2,304 bits. The simulator forwards only `(commit_time, task)` and charges no payload-valid/capture timing, exact-tag comparator, 2,304-bit route, broadcast fanout, context muxing, wire energy, or clock penalty. “Zero payload storage” is true for the transaction-model data structure, not proven for a physical implementation.

The abstract avoided-read volume is 404,252,928 bytes, so an energy-only hypothesis remains possible. It requires SRAM-versus-wire energy characterization; it is not performance evidence.

## Gate discipline

M92 is 1.0774% better than M89 K6 only after stacking it on M91. That cannot override the result: M91 itself missed its own one-percent gate, and M92 misses both its source and incremental 0.25-percent gates. The producer receipt correctly keeps this comparison non-promoting.

## Next minimum performance direction

Do not build the wide bypass. First screen a metadata-only, bank-cycle-monotonic K6 grouping/admission rule over the existing 16 resident descriptors. A bounded two-seed or critical-bank lookahead should reject choices that increase fused `bank_issue_cycles`. This directly attacks the observed +124,960 groups/+58,184 source-cycle coupling without adding a 2,304-bit data route.

Freeze the next gate at source cycles ≤69,211,896, integrated cycles ≤75,740,988, and no per-sample integrated regression. Only after that screen passes should a small metadata arbiter receive VCS/DC/STA effort.

## Reproduction

```bash
python3 hw_autoresearch_nts07/reviews/m92_commit_bus_parent_bypass_independent_hammer_r1_20260824/validate_m92_commit_bus_parent_bypass.py \
  --hw-root hw_autoresearch_nts07 \
  --output hw_autoresearch_nts07/reviews/m92_commit_bus_parent_bypass_independent_hammer_r1_20260824/m92_independent_audit.json \
  --log hw_autoresearch_nts07/reviews/m92_commit_bus_parent_bypass_independent_hammer_r1_20260824/m92_independent_audit.log
```

The validator reads only sealed raw artifacts and producer source; it does not import or execute the M92 producer.
