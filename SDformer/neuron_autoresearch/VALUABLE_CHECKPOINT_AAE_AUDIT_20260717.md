# Valuable Checkpoint AAE Audit (2026-07-17)

All rows use the same DSEC valid825 center-crop evaluation. `AAE-2D` is retained only for historical comparison; `AE-3D` is the DSEC/Barron angular metric used for benchmark-facing reporting.

## Paper Core

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NB0 baseline | 59 | 1.4872 | 9.9300 | 9.2506 | 0.5107 | 0.1891 | 0.0871 | 44.0488 | 37638.01 | ATLIF 0, Shiftmax 0, 0/0 |
| Frozen TTX/H60 | 2 | 1.5019 | 9.8894 | 9.2123 | 0.5169 | 0.1949 | 0.0918 | 23.2396 | 20521.16 | ATLIF 105, Shiftmax 12, 0/0 |
| H67 Motion-XOR (float) | 19 | 1.4671 | 9.4155 | 8.7949 | 0.5002 | 0.1891 | 0.0890 | 26.3898 | 23393.08 | ATLIF 105, Shiftmax 12, 0/0 |
| H67 Motion-XOR (RTL-exact) | 19 | 1.4627 | 9.4040 | 8.7801 | 0.5007 | 0.1886 | 0.0883 | 26.3544 | 23362.23 | ATLIF 105, Shiftmax 12, 0/0 |
| H68 Castling-trained/H60 deploy (RTL-exact) | 19 | 1.4727 | 9.4714 | 8.8441 | 0.5025 | 0.1895 | 0.0891 | 26.4164 | 23414.83 | ATLIF 105, Shiftmax 12, 0/0 |

## Historical Hardware Control

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NTS11bd mixed ternary/binary | 19 | 1.5650 | 9.9234 | 9.3407 | 0.5310 | 0.2101 | 0.1021 | 29.1679 | 23109.18 | ATLIF 105, Shiftmax 12, 0/0 |

## Historical Attention Control

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| All-binary + TX | 19 | 1.5831 | 9.9381 | 9.3482 | 0.5348 | 0.2136 | 0.1041 | 22.4706 | 19780.93 | ATLIF 105, Shiftmax 12, 0/0 |

## Attention Family Ablation

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H66a a-XNOR matrix | 19 | 1.5060 | 9.6469 | 9.0311 | 0.5102 | 0.1973 | 0.0946 | 26.8712 | 23762.69 | ATLIF 105, Shiftmax 12, 0/0 |
| H66b Hamming linear | 29 | 1.5429 | 9.7685 | 9.1403 | 0.5132 | 0.2057 | 0.1023 | 26.2821 | 22901.70 | ATLIF 105, Shiftmax 12, 0/0 |
| H66c TP-TTX | 19 | 1.4757 | 9.5116 | 8.8846 | 0.5038 | 0.1904 | 0.0894 | 26.5044 | 23473.59 | ATLIF 105, Shiftmax 12, 0/0 |

## Mechanism Ablation

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H69 fixed dyadic temperature | 19 | 1.4819 | 9.5177 | 8.8829 | 0.5056 | 0.1920 | 0.0899 | 26.3414 | 23344.73 | ATLIF 105, Shiftmax 12, 0/0 |
| H70 event-selective dyadic TTX | 19 | 1.4852 | 9.5081 | 8.9013 | 0.5052 | 0.1917 | 0.0904 | 26.3213 | 23317.32 | ATLIF 105, Shiftmax 12, 0/0 |
| H71 window-context TTX | 19 | 1.4872 | 9.3892 | 8.8030 | 0.5045 | 0.1933 | 0.0923 | 26.4488 | 23444.92 | ATLIF 105, Shiftmax 12, 0/0 |

## Key Findings

- H67 RTL-exact is the current checkpoint mainline. Versus NB0, AEE improves by 1.65%, AE-3D improves by 5.09%, spikes fall by 40.17%, and the energy proxy falls by 37.93%.
- RTL-exact Shiftmax does not degrade H67: relative to float, delta AEE is -0.0044 and delta AE-3D is -0.0148.
- H68 remains the zero-deployment-increment fallback, but trails H67 RTL-exact by 0.0100 AEE and 0.0640 AE-3D.
- The local angular gap is not evidence of incomplete convergence. The corrected valid825 AE-3D remains around 9 degrees because valid825 center-crop and official DSEC test are different evaluation splits/protocols.

## Loading Audit

Every row completed with the expected installed ATLIF/Shiftmax count and `missing=0, unexpected=0`. The original configs, checkpoints, and historical valid825 outputs were not modified.

## Interpretation Rule

Do not compare these valid825 AE-3D values directly with the paper's official DSEC test AE. Use this table for same-split model comparisons; use official test submission for the final paper benchmark table.
