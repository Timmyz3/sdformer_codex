# M1775 independent result hammer

Verdict: **PASS sealed diagnostic result; no paper, S2, same-resource, energy, or system-speedup admission.** Score 95/100, with P0/P1/P2 = 0/1/2.

## What was independently checked

- The M1763 result has the exact five sealed payload members, and its decision/manifest/outer triple is `722aa302... / e70b0f83... / 9e08414f...`.
- The M1765 release, M1764 source hammer, M1763 source/contract/test, M1763 author receipt, M1744 capture hammer, capture seals, checkpoint identity, sample order, and frozen docs/359 identity are consistent.
- JSON and both CSVs agree row by row. The sealed cohort is 40 samples across **four** sequences, not five: interlaken_01_a, thun_01_b, zurich_city_09_a, and zurich_city_12_a.
- B4/B8 all-scope integers equal the sum of the four sequence rows. Weight-byte, bank, LRU, roofline, and state-lower-bound equations were recomputed. The full per-sequence table is in `mechanical_checks.json`.
- The epsilon axis, all/sequence/layer integer aggregation, extrema, metadata, debt, and admission fields were independently recomputed. The epsilon-zero SHA-256 was reconstructed from the frozen 12-layer IDs/shapes and matches exactly: `7b2d1dd0...`.

## TSBG result boundary

| Bundle | Baseline bytes | Candidate bytes | Byte reduction | Baseline roofline | Candidate roofline | Ratio | Incremental state lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 | 12,495,069,081,600 | 4,348,564,279,296 | 65.20% | 100,581,477,504 | 34,640,732,544 | 2.904x | 912 B |
| B8 | 12,428,033,470,464 | 2,461,306,054,656 | 80.20% | 99,938,547,840 | 19,512,618,240 | 5.122x | 2,128 B |

These are useful **screening ratios only**. The result explicitly has `same_resource_claim=false`, does not price context tags/broadcast control or full area/energy, and admits neither the cycle path nor the energy-only path. They must not be written as hardware or paper speedup.

## Why S2 drops 100% at epsilon >= 0.01

M1744 independently established that all 872,855,874 nonzero diagnostic FC codes are `-1` or `+1`. A 16-wide group therefore has absolute-code sum at most 16. S2 uses

`threshold = floor(epsilon * 16 * 127)`.

The smallest nonzero point, epsilon 0.01, already gives threshold 20; later points give 40, 101, and 203. Hence every active group satisfies the drop predicate at every nonzero point. The observed maximum is 15, and all 8,594,923,488 FC1 nonzero blocks are dropped. This is a degenerate all-drop endpoint, not a usable Pareto result. Paired AEE and same-resource cycles are absent, so **S2 is NO-ADMIT**.

If S2 is ever revisited, use integer thresholds such as 1/2/4/8/12/15 (epsilon below 0.01) and require the already frozen paired-AEE and same-resource gates before RTL.

## Attempt directory ruling

The empty attempt directory is **not P0**. M1765 requires a fresh one-attempt namespace and no automatic retry; it does not promise a sealed attempt receipt. M1763 creates the directory through atomic `mkdir()` only after authority and freshness checks, then never deletes it. That is a valid consumed semaphore. It is nevertheless weakly self-describing (P2).

Minimal additive fix: create a separate double-sealed receipt binding the attempt path, atomic-mkdir semantics, M1765 SHA, M1763 result triple, `analysis_runs=1`, `result_publications=1`, and `automatic_retry=false`. Do not alter the empty directory or result. The receipt can also explain that `RUN_COMPLETE.txt` retains the inherited M1721 diagnostic label while `decision.json` is the authoritative M1763 identity.

## Remaining hash boundary

The transferred result does not include M1707 `fc_frames.bin`. M1775 independently reproduced the epsilon-zero hash and rehashed the complete producer hash map and all epsilon rows, but cannot recreate the four nonzero-epsilon active-bit payload hashes from aggregates alone. Those four values remain sealed producer provenance, not different-author payload replay. This does not change the all-drop proof or the existing no-paper boundary.
