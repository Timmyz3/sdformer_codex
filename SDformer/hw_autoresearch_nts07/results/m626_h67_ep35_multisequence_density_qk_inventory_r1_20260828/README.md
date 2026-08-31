# M626 H67 ep35 multi-sequence evidence inventory

Status: `PASS_MULTI_SEQUENCE_QK_CPU_REPLAY__RAW_NONATTENTION_COVERAGE_REMAINS_SINGLE_SEQUENCE_OR_MISSING`

Exact CPU replay verified all 1,200 packaged Q/K NPZ files with zero SHA or manifest-stat mismatches. The frozen population spans 100 samples across three DSEC sequences; two are additional to `zurich_city_09_a`.

| sequence | samples | density min/mean/max | mean AEE | Q active | K active | mean K-zero |
|---|---:|---:|---:|---:|---:|---:|
| zurich_city_02_c | 26 | 0.235941/0.330226/0.395510 | 1.228826 | 2.010452% | 3.906411% | 84.032851% |
| zurich_city_07_a | 10 | 0.180990/0.235331/0.279025 | 2.001622 | 2.086343% | 3.793131% | 83.542735% |
| zurich_city_09_a | 64 | 0.056628/0.318913/0.398783 | 1.444068 | 1.555337% | 3.799321% | 83.588806% |

This closes multi-sequence availability for attention Q/K density evidence only. Conv/FC, FFN, ATLIF and decoder raw payloads are still single-sequence or missing; see the coverage matrix and JSON gaps. No cycle, speedup, energy, PPA, full-network, or headline claim is admitted.
