# M627 fresh independent hammer: M626 multi-sequence density/QK evidence

## Verdict

**PASS 99/100** (`P0=0, P1=0, P2=0`). M626 is safe to use as **three-sequence H67 ep35 attention-Q/K robustness evidence**, provided the paper never promotes it to cycles, speedup, energy, PPA, full-network performance, or a DATE headline.

The one-point reserve reflects the intentionally narrow evidence boundary, not a discovered defect: AEE was read from the frozen profiler table rather than reevaluated from raw DSEC event/GT/mask data, which is not packaged locally.

## Independent method

This fresh hammer did not import or execute the M626 analyzer. It used a separate Python 3.6 + NumPy implementation to:

1. hash every handoff member and every sealed M626 result member;
2. open all 1,200 Q/K NPZ files and recompute byte-popcounts and nonzero gates;
3. rebuild sample, sequence, and predeclared density-bin aggregates from `sample_workload.csv`;
4. independently inventory M51/M511/M460/M515 and `activation_records.csv` boundaries.

No GPU, remote host, VCS, DC, PT, or PTPX process was used.

## High-impact checks

| Attack | Independent result |
|---|---:|
| Handoff member identity | 1,230/1,230, zero mismatch |
| Handoff bytes | 2,196,076,814 B, exact |
| M626 inner/outer seal | 7 members, zero mismatch, outer seal valid |
| Workload population | 100 samples |
| Sequence population | `09_a=64`, `07_a=10`, `02_c=26` |
| Additional sequences beyond `09_a` | 2 |
| Workload-to-trace sample-key join | 100/100 |
| Attention population | 1,200 NPZ = 100 samples x 12 blocks |
| Full NPZ SHA/Q/K/gate replay | 1,200/1,200, zero mismatch |
| Frozen density-bin population | `9 / 8 / 9 / 42 / 32 / 0` |
| Density-bin/sequence numeric comparison | zero mismatch |
| Checkpoint-load missing/unexpected/overlay | `0 / 0 / 0 / 0` |
| `activation_records.csv` | 3,400 = 100 x 34 rows |
| Explicit sample identity | 800 = 100 x 8 rows |
| Position-only summary rows | 2,600 = 100 x 26 rows |

## Independently recomputed sequence table

| Sequence | Samples | Density min/mean/max | Mean AEE | Q active | K active | Mean K-zero token |
|---|---:|---:|---:|---:|---:|---:|
| `zurich_city_02_c` | 26 | 0.235941 / 0.330226 / 0.395510 | 1.228826 | 2.01045% | 3.90641% | 84.03285% |
| `zurich_city_07_a` | 10 | 0.180990 / 0.235331 / 0.279025 | 2.001622 | 2.08634% | 3.79313% | 83.54273% |
| `zurich_city_09_a` | 64 | 0.056628 / 0.318913 / 0.398783 | 1.444068 | 1.55534% | 3.79932% | 83.58881% |

Every displayed value agrees with both the sealed M626 JSON/CSV and the independently recomputed raw numerators/denominators.

## Non-attention coverage attack

The missing-coverage warnings in M626 are correct:

- M51 manifest contains 310 Conv/FC bitpack records, but only 160 physical files are present: `Conv2d=60`, `Linear=100`. Missing: `Conv2d=10`, `Linear=140`. All belong to `zurich_city_09_a`.
- At the M627 validation snapshot, M511 has a capture contract but zero captured decoder payload files in `results`, `system_handoff/incoming`, or `system_handoff/outgoing`.
- M460 is a ten-sample, single-sequence, post-compute FFN opportunity/oracle. It explicitly does not certify executable pre-compute skipping.
- M515 is a ten-sample S10 ATLIF state-boundary/accounting audit over the ordered trace. It is not multi-sequence raw ATLIF I/O/state payload and admits no cycle speedup.
- `activation_records.csv` is summary evidence. Only 8/34 rows per sample carry explicit sample identity; the remaining 26/34 are recoverable only by contiguous position.

Therefore M626 does **not** establish multi-sequence Conv, FC, ATLIF, decoder, or full-network execution coverage.

## Paper-safe boundary

Safe statement: the frozen H67 ep35 attention-Q/K package covers 100 samples from three DSEC sequences, and its activity trends remain sparse across predeclared event-density strata.

Unsafe statements include all of the following:

- turning Q/K activity ratios into cycle or speedup values;
- calling the evidence full-network or system-level;
- implying multi-sequence coverage for Conv/FC/ATLIF/decoder;
- using M626 as energy, PPA, silicon, or DATE headline evidence.

## Reproducibility

- `m627_independent_hammer.py`: fresh implementation; M626 analyzer is never imported or executed.
- `m627_independent_recomputation.json`: machine-readable recomputation and mismatch vectors.
- `m627_attack_matrix.csv`: twelve mandatory attacks and evidence.
- `validation.txt`: compact terminal receipt.

