# M484 row-coherent signed-bundle/state-stationary DSE

Status: **NO-GO versus the strong same-resource K8-resident baseline; no RTL or performance admission**.

All 160 locally present target records reconcile to the frozen dual-line selected-source ledger. The population is ten windows from only `zurich_city_09_a`.

| Category | Decision N | K1->K8 resource scaling | M484 vs K8 cycles | M484 vs K8 traffic | Gate |
|---|---:|---:|---:|---:|---|
| Conv | 8 | 6.0503x | 1.0000x | -0.38% | FAIL |
| Conv->ATLIF | 8 | 5.4920x | 1.0000x | -0.33% | FAIL |
| FC1 | 8 | 4.5541x | 1.0000x | -0.04% | FAIL |

Worst window at each category's best N:

- Conv: zurich_city_09_a sample 5 (`zurich_city_09_a_0051.npy`), 1.0000x, traffic reduction -0.38%.
- Conv->ATLIF: zurich_city_09_a sample 7 (`zurich_city_09_a_0071.npy`), 1.0000x, traffic reduction -0.33%.
- FC1: zurich_city_09_a sample 3 (`zurich_city_09_a_0031.npy`), 1.0000x, traffic reduction -0.04%.

Online original-order boundary:

- Conv: `online_original_NCHW_safe_lower_bound`, 1.0000x, traffic reduction -12.02%, gate NO-GO.
- Conv->ATLIF: `online_original_NCHW_safe_lower_bound`, 1.0000x, traffic reduction -10.44%, gate NO-GO.
- FC1: `online_original_C_order_exact`, 1.0000x, traffic reduction -0.04%, gate NO-GO.

Zero finite-slot stalls are structural to the offline destination-major schedule; they are not evidence for an online reorder frontend. Pack wait is measured in accepted events, not wall-clock cycles.
