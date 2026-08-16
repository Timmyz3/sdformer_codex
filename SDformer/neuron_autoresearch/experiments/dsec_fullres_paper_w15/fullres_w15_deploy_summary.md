# DSEC Fullres Window15 Deploy Evaluation

| baseline | epoch | AEE | AAE legacy | AAE benchmark | spikes(G) |
|---|---:|---:|---:|---:|---:|
| NB0 baseline | 29 | 1.4454 | 6.5128 | 6.1803 | 126.1156 |

| candidate | epoch | float AEE | dyadic AEE | hardware-order AEE | hardware-float delta | hardware-order AAE legacy | AAE benchmark | spikes(G) | true mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H67 Motion-XOR | 29 | 2.0730 | 2.0669 | 2.0880 | +0.0150 | 8.1532 | 7.9466 | 87.9802 | False |
| H66d Local-5 | 29 | 2.0912 | 2.1041 | 2.1091 | +0.0179 | 8.2214 | 8.0203 | 89.8145 | True |

The hardware-order column is the frozen integer/LUT numerical path. Fullres SV sign-off additionally requires window15/T450 controller, address, line-buffer, and ordered-trace regression.

- H67 Motion-XOR: hardware-order numeric exact; existing H67 SV row RTL is verified at window9/T162, while fullres window15/T450 controller parameterization still requires RTL regression.
- H66d Local-5: score/gate hardware-order numeric exact with true masked candidates; fullres window15 line-buffer/address-control SV replay remains a separate hardware sign-off item.
