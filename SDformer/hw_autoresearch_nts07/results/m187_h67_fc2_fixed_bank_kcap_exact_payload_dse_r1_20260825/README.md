# M187 H67 FC2 fixed-bank K-cap exact-payload DSE

Status: `PASS_EXACT_PAYLOAD_FIXED_BANK_KCAP_DSE_RTL_OPEN`.

M187 scans arithmetic issue caps K=1..8 on all 120 frozen H67 ep35 FC2
payload records at the bounded D={2,4,8,8} dual-window point.  Eight physical
weight banks remain; an issue consumes at most one source from a bank and at
most K sources total.  Serving the K largest remaining bank populations
attains the exact group bound
`max(max_bank_population, ceil(total_events/K))`.

## Key result

| K | Exact wall cycles | Speed versus K4 | Weight response bits/issue |
|---:|---:|---:|---:|
| 4 | 127,581,198 | 1.000000x | 3,072 |
| 5 | 109,553,951 | 1.164551x | 3,840 |
| 6 | 100,331,395 | 1.271598x | 4,608 |
| 7 | 97,694,539 | 1.305919x | 5,376 |
| 8 | 97,607,807 | 1.307080x | 6,144 |

K7 is only 0.088858% slower than K8 and retains 99.911221% of K8's schedule
speedup, while reducing the response payload and arithmetic-source count by
12.5%.  It is therefore the selected next RTL screen.  This is a cycle-DSE
selection, not an RTL, physical, complete-FC2, system or headline result.

The K4 and K8 totals exactly cross-check the pinned M179 and M182 ledgers.
Every payload's SHA, extent and popcount was rechecked; the aggregate contains
143,894,510 events over 5,580,000 tokens.

## Reproduction

From `hw_autoresearch_nts07`, invoke
`system_simulator/scripts/analyze_m187_h67_fc2_fixed_bank_kcap_dse.py` with the
same manifest, payload root, pinned M172/M179 analyzers, M179 result and
`docs/359` paths recorded by the result identity.  Use a new output directory;
the analyzer refuses overwrite.
