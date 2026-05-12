# E2 ATLIF Loader Benchmark - 2026-04-25

Benchmark setup: 128 train samples, 8 valid samples, 1 epoch, E2 ATLIF overlay.

| config | status | train samples/sec | epoch time sec | step time sec | max GPU mem GiB | note |
|---|---|---:|---:|---:|---:|---|
| `bench_bs4w8.yml` | pass | 4.7093 | 27.18 | 0.8494 | 23.033 | low memory, slower |
| `bench_bs8w4.yml` | pass | 5.4494 | 23.49 | 1.4680 | 44.967 | fastest bs8 |
| `bench_bs8w8.yml` | pass | 5.2875 | 24.21 | 1.5130 | 44.967 | more workers did not help |
| `bench_bs8w16.yml` | pass | 5.0496 | 25.35 | 1.5843 | 44.962 | extra workers hurt |
| `bench_bs12w8.yml` | pass | 5.6308 | 21.31 | 2.1311 | 66.936 | selected for full training |
| `bench_bs14w8.yml` | pass | 5.4438 | 23.15 | 2.5717 | 77.786 | slower and near OOM |
| `bench_bs16w8.yml` | OOM |  |  |  |  | CUDA OOM |

Selected full config: `neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml`.
