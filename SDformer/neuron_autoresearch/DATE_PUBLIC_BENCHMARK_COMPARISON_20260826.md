# DATE Public Benchmark Comparison

Retrieved and audited on 2026-08-26. Lower is better for every metric below.
Published values and local values are intentionally separated.

## DSEC Official Test

| method | type | EPE | AE | 3PE (%) | source |
|---|---|---:|---:|---:|---|
| BAT | ANN | 0.655 | 2.439 | 1.773 | AAAI 2026 Table 1 |
| IDNet | ANN | 0.719 | 2.723 | 2.036 | ICRA 2024 / official DSEC entry |
| EDCFlow | ANN | 0.720 | 2.650 | 2.100 | CVPR 2025 Table 1 |
| TMA | ANN | 0.743 | 2.684 | 2.301 | ICCV 2023 |
| E-RAFT | ANN | 0.788 | 2.851 | 2.684 | 3DV 2021 |
| SDformerFlow-v2 | SNN | 1.602 | 4.871 | 10.051 | parent paper Table I |
| OF_EV_SNN | SNN | 1.707 | 6.338 | 10.308 | parent paper Table I |
| Proposed ATLIF + Complete TTX | SNN | pending | pending | pending | requires official submission |

Interpretation:

- `SDformerFlow-v2` is 6.2% better in EPE than the earlier `OF_EV_SNN`, so it is a
  valid SNN parent baseline.
- It remains 103.3% worse in EPE than E-RAFT and 144.6% worse than BAT. The DATE
  claim must therefore be accuracy-preserving efficiency/co-design, not optical-flow
  accuracy SOTA.
- The proposed row cannot be filled using local `valid825`; an official DSEC test
  submission is required.

## DSEC Local Same-Parent Short Control

These values are local valid825 and are not numerically comparable to the official
table above. Full30 reruns are currently queued.

| route | AEE | AE-3D | Fl (%) | spikes (G) |
|---|---:|---:|---:|---:|
| C00 PSN + original attention | 1.427353 | 6.357416 | 7.6801 | 128.9324 |
| C10 binary ATLIF + original attention | 1.480472 | 6.045349 | 8.4317 | 61.5937 |
| C12 binary ATLIF + Complete TTX | 1.435804 | 5.829910 | 7.9355 | 61.8243 |

The valid local claim is C00 to C12: AEE changes by +0.59%, while spikes decrease by
52.05%. C10 to C12 recovers 3.02% AEE at nearly unchanged spikes. The full30 table
must replace this short10 table once complete.

## MVSEC dt1 Public Results

The parent paper reports event-masked sparse flow over OD1/IF1/IF2/IF3. `M` denotes
training on MVSEC; `MDR` denotes training on MDR. Published values are reproduced
from SDformerFlow Table II.

| method | training | OD1 | IF1 | IF2 | IF3 | macro AEE |
|---|---|---:|---:|---:|---:|---:|
| EV-FlowNet2 | M | 0.32 | 0.58 | 1.02 | 0.87 | 0.69 |
| OF_EV_SNN | M | 0.85 | 0.58 | 0.72 | 0.67 | 0.71 |
| Spatiotemporal SNN | M | 0.45 | 0.76 | 1.13 | 0.95 | 0.82 |
| Spike-FlowNet | M | 0.49 | 0.84 | 1.28 | 1.11 | 0.93 |
| SDformerFlow-v2 | MDR | 0.61 | 0.54 | 0.81 | 0.69 | 0.66 |

## Our MVSEC Results

The first three rows use the current direct day2 train/held-out-validation protocol
and fixed800 test. The final row is the older MDR-trained TTX route.

| route | training | OD1 | IF1 | IF2 | IF3 | macro AEE | spikes/energy evidence |
|---|---|---:|---:|---:|---:|---:|---|
| NB0 | MVSEC day2 | 0.8379 | 1.5977 | 2.7469 | 2.1102 | 1.8231 | available |
| binary ATLIF only | MVSEC day2 | 0.7940 | 1.6525 | 2.6564 | 2.0912 | 1.7985 | available |
| binary ATLIF + Complete TTX | MVSEC day2 | 0.8181 | 1.5850 | 2.6212 | 2.0352 | 1.7649 | available |
| binary ATLIF + TTX ep20 | MDR | 0.9779 | 1.1414 | 1.4266 | 1.2617 | 1.2019 | 39.6% fewer spikes than local MDR baseline |

Interpretation:

- Direct day2 Complete TTX improves its matched NB0 by 3.20% macro AEE, but remains
  89.8% worse than published Spike-FlowNet and 148.6% worse than published
  OF_EV_SNN. This row is useful as an internal ablation, not as a competitive external
  accuracy result.
- Relative to published MDR-trained SDformerFlow-v2, direct day2 Complete TTX is
  34.1% worse on OD1 and approximately 194%--224% worse on the indoor sequences.
  The major issue is cross-domain indoor generalization, not OD1 alone.
- The older MDR-trained TTX route reduces macro AEE from 1.7649 to 1.2019 and is
  therefore the better MVSEC paper route. It is still 82.1% worse than the published
  SDformerFlow-v2 macro AEE of 0.66, so the published parent level has not been
  reproduced locally.
- A dyadic Motion-alpha sweep can provide a small internal improvement, but it cannot
  plausibly close the current public-benchmark gap. For a competitive MVSEC table,
  the next experiment must target the MDR-to-MVSEC training/evaluation protocol or
  identify the remaining preprocessing/supervision mismatch.

## Sources

- DSEC benchmark metrics and server:
  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/
- BAT, AAAI 2026:
  https://ojs.aaai.org/index.php/AAAI/article/download/38100/42062
- EDCFlow, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf
- SDformerFlow paper and official DSEC submission report:
  https://dsec.ifi.uzh.ch/wp-content/uploads/sourcenova/uni-comp/optical-flow-benchmark-v1-0/submissions/269/details.pdf
- IDNet official DSEC entry:
  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/idnet/
